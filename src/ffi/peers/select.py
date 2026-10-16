"""Peer selection: same industry, nearest on size, growth and margins (BLUEPRINT §7)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from ffi.edgar import endpoints
from ffi.edgar.client import EdgarClient, EdgarError
from ffi.peers.crosssection import calendar_year_of
from ffi.xbrl.facts import FactStore, Panel

log = logging.getLogger(__name__)


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
) -> PeerSet:
    names = names or {}
    if override:
        chosen = [(c, 0.0) for c in override if c != target_cik]
        method = "user override (--peers)"
    else:
        year = calendar_year_of(focus_fye)
        cur = xsec[xsec["year"] == year].set_index("cik")
        prev = xsec[xsec["year"] == year - 1].set_index("cik")
        if cur.empty or target_cik not in cur.index:
            year -= 1
            cur, prev = prev, xsec[xsec["year"] == year - 1].set_index("cik")
        cur = cur.copy()
        cur["REV_PRIOR"] = prev["REVENUE"].reindex(cur.index)
        cur = cur[cur["REVENUE"].notna() & cur["TOTAL_ASSETS"].notna() & (cur["REVENUE"] > 0)]
        feats = _features(cur).replace([np.inf, -np.inf], np.nan)
        feats = feats.fillna(feats.median())
        if target_cik not in feats.index or len(feats) < 3:
            return PeerSet([], sic_digits, len(universe), "insufficient frames data", ["No peer metrics available."])
        z = (feats - feats.mean()) / feats.std(ddof=0).replace(0, 1)
        weights = pd.Series({"log_rev": 2.0, "log_assets": 1.5, "gross_margin": 1.0, "op_margin": 1.0, "growth": 0.5})
        w = weights.reindex(z.columns).fillna(1.0)
        d = np.sqrt((((z - z.loc[target_cik]) ** 2) * w).sum(axis=1)).drop(target_cik)
        chosen = list(d.nsmallest(k).items())
        method = f"nearest {k} of {len(d)} SIC-{sic_digits} filers on size, margins and growth (CY{year})"
    from ffi.concurrency import map_concurrent

    def load(item: tuple[int, float]) -> Peer:
        cik, dist = item
        p = Peer(cik=int(cik), name=names.get(int(cik), str(cik)), distance=float(dist))
        try:
            cf = endpoints.companyfacts(client, p.cik)
            p.name = cf.get("entityName", p.name)
            p.annual = FactStore.from_companyfacts(p.cik, cf, as_of).annual("first")
        except (EdgarError, KeyError, ValueError) as exc:
            log.warning("peer %s skipped: %s", cik, exc)
        return p

    peers = [p for p in map_concurrent(load, chosen, workers=6) if isinstance(p, Peer)]
    return PeerSet(peers, sic_digits, len(universe), method)
