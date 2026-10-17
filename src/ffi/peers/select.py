"""Peer selection: same industry, nearest on size, growth and margins."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from ffi.edgar.client import EdgarClient, EdgarError
from ffi.peers.crosssection import calendar_year_of
from ffi.xbrl.facts import FactStore, Panel, load_companyfacts

log = logging.getLogger(__name__)
PEER_FACTS_MAX_AGE = 7 * 86_400  # peers' facts are baselines: a week-old copy saves 15 downloads of 10-50 MB per run


@dataclass
class Peer:
    cik: int
    name: str
    distance: float
    annual: Panel | None = None


@dataclass
class PeerSet:
    peers: list[Peer]
    sic_digits: int
    universe_size: int
    method: str
    notes: list[str] = field(default_factory=list)

    @property
    def loaded(self) -> list[Peer]:
        return [p for p in self.peers if p.annual is not None and not p.annual.values.empty]

    def aligned_period(self, peer: Peer, fye: pd.Timestamp) -> pd.Timestamp | None:
        """The peer's fiscal year end closest to `fye` (within ~6 months)."""
        if peer.annual is None or peer.annual.values.empty:
            return None
        idx = peer.annual.values.index
        diffs = np.abs((idx - fye).days)
        i = int(np.argmin(diffs))
        return idx[i] if diffs[i] <= 183 else None


def _features(xs: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=xs.index)
    f["log_rev"] = np.log(xs["REVENUE"].clip(lower=1))
    f["log_assets"] = np.log(xs["TOTAL_ASSETS"].clip(lower=1))
    gp = xs["GROSS_PROFIT"].fillna(xs["REVENUE"] - xs["COGS"])
    f["gross_margin"] = (gp / xs["REVENUE"]).clip(-1, 1)
    f["op_margin"] = (xs["OPERATING_INCOME"] / xs["REVENUE"]).clip(-1, 1)
    if "REV_PRIOR" in xs:
        f["growth"] = (xs["REVENUE"] / xs["REV_PRIOR"] - 1).clip(-1, 3)
    return f


def select_peers(
    client: EdgarClient,
    target_cik: int,
    universe: list[int],
    sic_digits: int,
    xsec: pd.DataFrame,
    focus_fye: pd.Timestamp,
    k: int,
    as_of: date | None,
    override: list[int] | None = None,
    names: dict[int, str] | None = None,
    target_row: dict[str, float] | None = None,
    size_comparable: bool = True,
    cache: dict[int, tuple[str, Panel | None]] | None = None,
) -> PeerSet:
    """`target_row` supplies the company's own metrics when it is missing from the frames cross-section (IFRS or
    non-USD filers, or revenue tagged with a concept the frames do not sample). Without `size_comparable`
    (amounts in another currency) peers are matched on margins and growth only. `cache` (CIK -> name and annual
    panel) lets the years of a multi-year review load each peer once: the panel does not depend on the year."""
    names = names or {}
    if override:
        chosen = [(c, 0.0) for c in override if c != target_cik]
        method = "user override (--peers)"
    else:
        year = calendar_year_of(focus_fye)
        cur = xsec[xsec["year"] == year].set_index("cik")
        prev = xsec[xsec["year"] == year - 1].set_index("cik")
        if (cur.empty or target_cik not in cur.index) and not (target_row and not cur.empty):
            year -= 1
            cur, prev = prev, xsec[xsec["year"] == year - 1].set_index("cik")
        cur = cur.copy()
        cur["REV_PRIOR"] = prev["REVENUE"].reindex(cur.index)
        own = ""
        if target_cik not in cur.index and target_row:
            row = pd.DataFrame([target_row], index=pd.Index([target_cik], name=cur.index.name))
            cur = pd.concat([cur, row.reindex(columns=cur.columns)])
            own = "; company metrics from its own filings"
        cur = cur[cur["REVENUE"].notna() & cur["TOTAL_ASSETS"].notna() & (cur["REVENUE"] > 0)]
        feats = _features(cur).replace([np.inf, -np.inf], np.nan)
        if not size_comparable:
            feats = feats.drop(columns=["log_rev", "log_assets"])
            own += "; size not compared (different reporting currency)"
        feats = feats.fillna(feats.median())
        if target_cik not in feats.index or len(feats) < 3:
            return PeerSet([], sic_digits, len(universe), "insufficient frames data", ["No peer metrics available."])
        z = (feats - feats.mean()) / feats.std(ddof=0).replace(0, 1)
        weights = pd.Series({"log_rev": 2.0, "log_assets": 1.5, "gross_margin": 1.0, "op_margin": 1.0, "growth": 0.5})
        w = weights.reindex(z.columns).fillna(1.0)
        d = np.sqrt((((z - z.loc[target_cik]) ** 2) * w).sum(axis=1)).drop(target_cik)
        chosen = list(d.nsmallest(k).items())
        basis = "size, margins and growth" if size_comparable else "margins and growth"
        method = f"nearest {k} of {len(d)} SIC-{sic_digits} filers on {basis} (CY{year}){own}"
    from ffi.concurrency import map_concurrent

    def load(item: tuple[int, float]) -> Peer:
        cik, dist = item
        p = Peer(cik=int(cik), name=names.get(int(cik), str(cik)), distance=float(dist))
        if cache is not None and p.cik in cache:
            p.name, p.annual = cache[p.cik]
            return p
        try:
            facts = load_companyfacts(client, p.cik, max_age_s=PEER_FACTS_MAX_AGE)
            p.name = facts.name or p.name
            p.annual = FactStore(p.cik, facts.raw, as_of).annual("first")
        except (EdgarError, KeyError, ValueError) as exc:
            log.warning("peer %s skipped: %s", cik, exc)
            return p  # not cached: a later year retries it
        if cache is not None:
            cache[p.cik] = (p.name, p.annual)
        return p

    peers = [p for p in map_concurrent(load, chosen, workers=6) if isinstance(p, Peer)]
    return PeerSet(peers, sic_digits, len(universe), method)
