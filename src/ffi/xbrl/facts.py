"""Point-in-time fact store built from SEC companyfacts.

Every reported value is kept with its filing date. Panels are built "as of" a date and in two
versions: as-first-reported (what the market saw first) and latest (after any restatement).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from ffi.xbrl.canonical import BY_ID, CONCEPT_TO_ITEMS

ANNUAL_DAYS = (340, 390)
QUARTER_DAYS = (80, 100)
ANNUAL_FORMS = ("10-K", "10-K/A", "10-KT", "10-KT/A", "10-K405", "20-F", "20-F/A", "40-F", "40-F/A")
DATE_TOL = pd.Timedelta(days=8)
# Items whose concepts include both totals and components (a filer may tag "DepreciationDepletionAndAmortization"
# for one part and "DepreciationAndAmortization" for the cash-flow total): within a filing the largest value wins.
LARGEST_WINS = {"DA"}
# Share counts tagged in thousands or millions (a common XBRL scaling error) are detected from NI / EPS.
SHARE_ITEMS = {"SHARES_DILUTED": "EPS_DILUTED", "SHARES_BASIC": "EPS_BASIC"}
CURRENCY_ANCHORS = ("REVENUE", "NET_INCOME", "TOTAL_ASSETS", "EQUITY")


FOREIGN_INTERIM_FORMS = ("6-K", "6-K/A")


def fiscal_year_of(end: pd.Timestamp) -> int:
    """The fiscal year a year-end date closes, named by the calendar year of the end date. A 52/53-week year that ends in the
    first days of January (Johnson & Johnson: 2023-01-01; Kellogg: 2021-01-02) closes the year before: it is the company's
    fiscal 2022, and calling it FY2023 gave two different years the same label."""
    return end.year - 1 if end.month == 1 and end.day <= 7 else end.year
# item attributes as lookup tables: Series.map(dict) is many times faster than a Python lambda per row
KIND = {i: it.kind for i, it in BY_ID.items()}
STATEMENT = {i: it.statement for i, it in BY_ID.items()}


def _anchor_rows(raw: pd.DataFrame) -> pd.Series:
    anchor_concepts = {(t, c) for (t, c), items in CONCEPT_TO_ITEMS.items() if any(i in CURRENCY_ANCHORS for i, _ in items)}
    keys = pd.Series(list(zip(raw["taxonomy"], raw["concept"], strict=True)), index=raw.index)
    return keys.isin(anchor_concepts)


def annual_forms_for(raw: pd.DataFrame) -> tuple[str, ...]:
    """Forms whose full-year facts count as annual statements. Most 40-F filers (and some 20-F filers) attach the
    XBRL for their audited annual statements to a Form 6-K, so 6-K is accepted when no annual form carries any."""
    if raw.empty or (_anchor_rows(raw) & raw["form"].isin(ANNUAL_FORMS)).any():
        return ANNUAL_FORMS
    return ANNUAL_FORMS + FOREIGN_INTERIM_FORMS


def reporting_currency(raw: pd.DataFrame, forms: tuple[str, ...] = ANNUAL_FORMS) -> str:
    """The currency of the company's own annual statements.

    Foreign issuers report in their home currency, and some add USD 'convenience translations' of the latest
    year; the currency used for the anchor line items in the most annual reports wins.
    """
    if raw.empty:
        return "USD"
    d = raw[_anchor_rows(raw) & raw["form"].isin(forms) & raw["unit"].str.fullmatch(r"[A-Z]{3}")]
    if d.empty:
        return "USD"
    return str(d.groupby("unit")["accn"].nunique().sort_values(ascending=False).index[0])


def flatten_companyfacts(cf: dict) -> pd.DataFrame:
    rows: list[tuple] = []
    for taxonomy, concepts in cf.get("facts", {}).items():
        for concept, body in concepts.items():
            if (taxonomy, concept) not in CONCEPT_TO_ITEMS:
                continue
            for unit, facts in body.get("units", {}).items():
                for f in facts:
                    rows.append(
                        (taxonomy, concept, unit, f.get("start"), f["end"], f["val"], f["accn"],
                         f.get("fy"), f.get("fp"), f.get("form", ""), f["filed"], f.get("frame"))
                    )
    df = pd.DataFrame(
        rows,
        columns=["taxonomy", "concept", "unit", "start", "end", "val", "accn", "fy", "fp", "form", "filed", "frame"],
    )
    if df.empty:
        return df
    df["start"] = pd.to_datetime(df["start"], errors="coerce")
    df["end"] = pd.to_datetime(df["end"])
    df["filed"] = pd.to_datetime(df["filed"])
    df["val"] = df["val"].astype(float)
    df["days"] = (df["end"] - df["start"]).dt.days
    return df


FLAT_VERSION = 1  # bump when flatten_companyfacts changes


@dataclass
class CompanyFacts:
    """A company's XBRL facts, flattened to the concepts the canonical map uses."""
    cik: int
    name: str
    raw: pd.DataFrame
    sha256: str  # of the SEC companyfacts file as downloaded


def load_companyfacts(client, cik: int, max_age_s: float | None = None) -> CompanyFacts:
    """A company's flattened companyfacts, cached as parquet; the raw download is not kept.

    The SEC file is 10-50 MB of JSON per company (the target and ~15 peers each run). Only the flattened table is
    stored, with the raw file's hash and the fetch time; it is refreshed once it is older than `max_age_s` (default a
    day), and a stale copy is used when the SEC cannot be reached or the run is offline."""
    import hashlib
    import json
    import time

    from ffi.edgar import endpoints
    from ffi.edgar.client import EdgarError

    concepts = hashlib.sha256(json.dumps(sorted(map(list, CONCEPT_TO_ITEMS))).encode()).hexdigest()[:8]
    folder = client.cache_dir.parent / "facts"
    path = folder / f"{cik}-{concepts}-v{FLAT_VERSION}.parquet"
    meta_path = path.with_suffix(".json")

    def cached() -> CompanyFacts | None:
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            return CompanyFacts(cik, meta["name"], pd.read_parquet(path), meta["sha256"])
        except (OSError, ValueError, KeyError):
            return None

    fresh = path.exists() and (client.offline or time.time() - path.stat().st_mtime < (max_age_s or endpoints.DAY))
    if fresh and (hit := cached()) is not None:
        return hit
    try:
        data = client.fetch(endpoints.COMPANYFACTS_URL.format(cik=cik))
    except (EdgarError, OSError):
        if (hit := cached()) is not None:
            return hit  # stale copy beats nothing
        raise
    cf = json.loads(data)
    facts = CompanyFacts(cik, cf.get("entityName", ""), flatten_companyfacts(cf), hashlib.sha256(data).hexdigest())
    try:
        folder.mkdir(parents=True, exist_ok=True)
        for old in folder.glob(f"{cik}-*"):  # superseded versions of this company's file
            old.unlink(missing_ok=True)
        facts.raw.to_parquet(path, index=False)
        meta_path.write_text(json.dumps({"name": facts.name, "sha256": facts.sha256}), encoding="utf-8")
    except (OSError, ValueError):
        pass
    return facts


def map_canonical(df: pd.DataFrame, currency: str = "USD") -> pd.DataFrame:
    """Attach canonical item ids (a concept can feed several items) with priority rank.

    Monetary items accept only the reporting currency, so a company's amounts are never mixed across currencies.
    """
    if df.empty:
        return df.assign(item=pd.Series(dtype=str), rank=pd.Series(dtype=int))
    keys = list(zip(df["taxonomy"], df["concept"], strict=True))
    item_lists = [CONCEPT_TO_ITEMS.get(k, []) for k in keys]
    df = df.assign(_m=item_lists).explode("_m").dropna(subset=["_m"])
    df["item"] = [t[0] for t in df["_m"]]
    df["rank"] = np.fromiter((t[1] for t in df["_m"]), dtype=int, count=len(df))
    df = df.drop(columns="_m")
    units = df["item"].map({i: it.unit.replace("USD", currency) for i, it in BY_ID.items()})
    kinds = df["item"].map(KIND)
    ok_unit = df["unit"] == units
    ok_kind = np.where(kinds == "D", df["days"].notna(), df["days"].isna())
    return df[ok_unit & ok_kind].reset_index(drop=True)


@dataclass
class FactRecord:
    item: str
    concept: str
    taxonomy: str
    unit: str
    start: pd.Timestamp | None
    end: pd.Timestamp
    val: float
    accn: str
    form: str
    filed: pd.Timestamp

    @property
    def fact_id(self) -> str:
        s = self.start.date().isoformat() if self.start is not None and not pd.isna(self.start) else ""
        return f"{self.taxonomy}:{self.concept}|{self.unit}|{s}|{self.end.date().isoformat()}|{self.accn}"


@dataclass
class Panel:
    """Values by period (rows) x canonical item (columns), plus the fact behind each cell."""

    freq: str  # "A" | "Q"
    values: pd.DataFrame
    labels: dict[pd.Timestamp, str]
    prov: dict[tuple[str, pd.Timestamp], FactRecord] = field(default_factory=dict)
    derived: set[tuple[str, pd.Timestamp]] = field(default_factory=set)
    rescaled: dict[tuple[str, pd.Timestamp], float] = field(default_factory=dict)  # (item, period) -> factor applied

    def get(self, item: str, period: pd.Timestamp) -> float:
        if item not in self.values.columns or period not in self.values.index:
            return float("nan")
        return float(self.values.at[period, item])

    def series(self, item: str) -> pd.Series:
        if item not in self.values.columns:
            return pd.Series(np.nan, index=self.values.index, dtype=float)
        return self.values[item].astype(float)

    @property
    def periods(self) -> list[pd.Timestamp]:
        return list(self.values.index)

    def label(self, period: pd.Timestamp) -> str:
        return self.labels.get(period, str(period.date()))


class FactStore:
    def __init__(self, cik: int, raw: pd.DataFrame, as_of: date | None = None, currency: str | None = None) -> None:
        self.cik = cik
        self.annual_forms = annual_forms_for(raw)
        self.currency = currency or reporting_currency(raw, self.annual_forms)
        mapped = map_canonical(raw, self.currency)
        if as_of is not None and not mapped.empty:
            mapped = mapped[mapped["filed"] <= pd.Timestamp(as_of)]
        self.df = mapped.reset_index(drop=True)
        self.as_of = as_of
        self._fye: list[pd.Timestamp] | None = None
        self._memo: dict[tuple[str, str], object] = {}  # built panels and version tables (callers must not mutate them)
        self._root: FactStore | None = None  # the unrestricted store this one was cut from (see `restricted`)
        counts = self.df["taxonomy"].value_counts() if not self.df.empty else pd.Series(dtype=int)
        self.taxonomy = "ifrs-full" if counts.get("ifrs-full", 0) > counts.get("us-gaap", 0) else "us-gaap"

    def restricted(self, cutoff: date) -> FactStore:
        """This store as it stood on `cutoff`: only facts filed by then. A past fiscal year is analysed with what was
        public by the end of its filing season, so no later filing (a restated comparative, a later year's annual report
        or quarterly report) reaches its panels. The reporting currency and annual forms stay those of the company."""
        new = copy.copy(self)
        new.df = self.df[self.df["filed"] <= pd.Timestamp(cutoff)].reset_index(drop=True) if not self.df.empty else self.df
        new.as_of, new._fye, new._memo = cutoff, None, {}
        new._root = self._root or self  # the audit's index of every reported value is built once, on the full store
        return new

    def __getstate__(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_memo")}  # caches are rebuilt on demand

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update({"_root": None, **state})
        self._memo = {}

    @classmethod
    def from_companyfacts(cls, cik: int, cf: dict, as_of: date | None = None) -> FactStore:
        return cls(cik, flatten_companyfacts(cf), as_of)

    @classmethod
    def load(cls, client, cik: int, as_of: date | None = None) -> FactStore:
        """From the cached flattened companyfacts (see load_companyfacts)."""
        return cls(cik, load_companyfacts(client, cik).raw, as_of)

    # ------------------------------------------------------------------ periods
    def fiscal_year_ends(self) -> list[pd.Timestamp]:
        if self._fye is None:
            d = self.df
            ann = d[(d["form"].isin(self.annual_forms)) & d["days"].between(*ANNUAL_DAYS)]
            ann = ann[ann["item"].isin(["REVENUE", "NET_INCOME", "CFO", "OPERATING_INCOME"])]
            self._fye = _cluster_dates(sorted(ann["end"].unique()))
        return self._fye

    def quarter_ends(self) -> list[pd.Timestamp]:
        d = self.df
        q = d[d["days"].between(*QUARTER_DAYS) & d["item"].isin(["REVENUE", "NET_INCOME"])]
        ends = _cluster_dates(sorted(set(q["end"].unique()) | set(self.fiscal_year_ends())))
        return ends

    def fiscal_label(self, end: pd.Timestamp) -> str:
        return f"FY{fiscal_year_of(end)}"

    def quarter_label(self, end: pd.Timestamp) -> str:
        fyes = list(self.fiscal_year_ends()) or [end]
        while fyes[-1] < end - DATE_TOL:  # project future fiscal year ends for quarters after the last 10-K
            fyes.append(fyes[-1] + pd.Timedelta(days=364))
        fy_end = next(f for f in fyes if f >= end - DATE_TOL)
        months_before = round((fy_end - end).days / 30.44)
        q = 4 - int(np.clip(round(months_before / 3), 0, 3))
        return f"FY{fiscal_year_of(fy_end)}Q{q}"

    # ------------------------------------------------------------------ selection (vectorized)
    def _record(self, row) -> FactRecord:
        start = row.start if not pd.isna(row.start) else None
        return FactRecord(item=row.item, concept=row.concept, taxonomy=row.taxonomy, unit=row.unit, start=start,
                          end=row.end, val=float(row.val), accn=row.accn, form=row.form, filed=row.filed)

    @staticmethod
    def _assign(d: pd.DataFrame, anchors: list[pd.Timestamp]) -> pd.DataFrame:
        """Attach each row to the nearest anchor date within DATE_TOL (column 'period')."""
        if d.empty or not anchors:
            return d.iloc[0:0].assign(period=pd.Series(dtype="datetime64[ns]"))
        a = np.array(sorted(anchors), dtype="datetime64[ns]")
        e = d["end"].to_numpy(dtype="datetime64[ns]")
        i = np.clip(np.searchsorted(a, e), 1, len(a) - 1) if len(a) > 1 else np.zeros(len(e), dtype=int)
        if len(a) > 1:
            left, right = a[i - 1], a[i]
            nearest = np.where(np.abs(e - left) <= np.abs(right - e), left, right)
        else:
            nearest = np.repeat(a[0], len(e))
        ok = np.abs(e - nearest) <= DATE_TOL.to_timedelta64()
        out = d[ok].copy()
        out["period"] = pd.to_datetime(nearest[ok])
        return out

    def _choose(self, d: pd.DataFrame, version: str, extra_sort: list[str] | None = None) -> pd.DataFrame:
        big = d["item"].isin(LARGEST_WINS)
        if big.any():
            d = d.copy()
            d.loc[big, "rank"] = d[big].groupby(["item", "period", "accn"])["val"].rank(ascending=False, method="first") - 1
        keys = ["item", "period", "rank"] + (extra_sort or []) + ["filed"]
        asc = [True, True, True] + [True] * len(extra_sort or []) + [version == "first"]
        return d.sort_values(keys, ascending=asc).drop_duplicates(["item", "period"], keep="first")

    def _panel(self, freq: str, chosen: pd.DataFrame, anchors: list[pd.Timestamp], labeler) -> Panel:
        values = chosen.pivot_table(index="period", columns="item", values="val", aggfunc="first")
        values = values.reindex(sorted(anchors)).astype(float)
        values.index.name = "period"
        values.columns.name = None
        prov = {(r.item, r.period): self._record(r) for r in chosen.itertuples(index=False)}
        labels = {p: labeler(p) for p in values.index}
        return _fix_share_scale(Panel(freq, values, labels, prov))

    def annual(self, version: str = "first") -> Panel:
        if ("annual", version) not in self._memo:  # also needed by quarterly() and the multi-year review's every year
            self._memo[("annual", version)] = self._annual(version)
        return self._memo[("annual", version)]  # type: ignore[return-value]

    def _annual(self, version: str) -> Panel:
        fyes = self.fiscal_year_ends()
        d = self.df
        kinds = d["item"].map(KIND)
        stmt = d["item"].map(STATEMENT)
        is_ann = d["form"].isin(self.annual_forms)
        dur = d[(kinds == "D") & (stmt != "DEI") & d["days"].between(*ANNUAL_DAYS) & (is_ann | d["form"].str.startswith("8-K"))]
        inst = d[(kinds == "I") & (stmt != "DEI")].assign(_pref=(~is_ann).astype(int))
        parts = [self._choose(self._assign(dur, fyes), version),
                 self._choose(self._assign(inst, fyes), version, ["_pref"]).drop(columns="_pref")]
        # cover-page (dei) values: latest-dated value inside each fiscal year window from a 10-K
        dei = d[(stmt == "DEI") & is_ann]
        rows = []
        for fye in fyes:
            w = dei[(dei["end"] > fye - pd.Timedelta(days=370)) & (dei["end"] <= fye + pd.Timedelta(days=120))]
            if w.empty:
                continue
            w = w[w["end"] == w.groupby("item")["end"].transform("max")].assign(period=fye)
            rows.append(self._choose(w, version))
        if rows:
            parts.append(pd.concat(rows))
        chosen = pd.concat([p for p in parts if not p.empty]) if any(not p.empty for p in parts) else d.iloc[0:0].assign(period=pd.NaT)
        return self._panel("A", chosen, fyes, self.fiscal_label)

    def quarterly(self, version: str = "first") -> Panel:
        qends = self.quarter_ends()
        fyes = self.fiscal_year_ends()
        d = self.df
        kinds = d["item"].map(KIND)
        stmt = d["item"].map(STATEMENT)
        dur = d[(kinds == "D") & d["days"].between(*QUARTER_DAYS)]
        inst = d[(kinds == "I") & (stmt != "DEI")]
        chosen = pd.concat([self._choose(self._assign(dur, qends), version), self._choose(self._assign(inst, qends), version)])
        panel = self._panel("Q", chosen, qends, self.quarter_label)
        # derive Q4 = FY - (Q1 + Q2 + Q3) for flow items where the fiscal-year value exists
        ann = self.annual(version)
        vals = panel.values
        qidx = list(vals.index)
        for fye in fyes:
            if fye not in vals.index or fye not in ann.values.index:
                continue
            prior = [q for q in qidx if fye - pd.Timedelta(days=380) < q < fye - DATE_TOL][-3:]
            if len(prior) != 3:
                continue
            for item in ann.values.columns:
                if BY_ID[item].kind != "D" or BY_ID[item].statement == "DEI" or BY_ID[item].unit == "USD/shares":
                    continue
                if item in vals.columns and np.isfinite(vals.at[fye, item]):
                    continue
                fy = ann.values.at[fye, item]
                if not np.isfinite(fy) or item not in vals.columns:
                    continue
                qs = vals.loc[prior, item]
                if qs.isna().any():
                    continue
                vals.at[fye, item] = float(fy - qs.sum())
                panel.prov[(item, fye)] = ann.prov[(item, fye)]
                panel.derived.add((item, fye))
        return panel

    # ------------------------------------------------------------------ versions
    def versions(self, freq: str = "A") -> pd.DataFrame:
        """All distinct reported values per (item, period) with the filing that reported each one."""
        if ("versions", freq) not in self._memo:  # read by the knowledge graph and the restatement test
            self._memo[("versions", freq)] = self._versions(freq)
        return self._memo[("versions", freq)]  # type: ignore[return-value]

    def values_by_filing(self) -> dict[tuple[str, str, str], np.ndarray]:
        """Every reported value by (accession, concept, period end as YYYY-MM-DD), for re-reading cited facts (the
        independent audit): a cited value must be the one the filing reports for the cited period, not merely a value
        the filing reports for that concept in some period."""
        if self._root is not None:
            return self._root.values_by_filing()
        if ("values", "accn_concept_end") not in self._memo:
            d = self.df.assign(_end=self.df["end"].dt.strftime("%Y-%m-%d")) if not self.df.empty else self.df.assign(_end=[])
            self._memo[("values", "accn_concept_end")] = {k: g.to_numpy(float) for k, g in d.groupby(["accn", "concept", "_end"])["val"]}
        return self._memo[("values", "accn_concept_end")]  # type: ignore[return-value]

    def _versions(self, freq: str) -> pd.DataFrame:
        d = self.df
        if d.empty:
            return d
        rng = ANNUAL_DAYS if freq == "A" else QUARTER_DAYS
        is_d = d["item"].map(KIND) == "D"
        sel = d[(~is_d) | d["days"].between(*rng)].copy()
        if freq == "A":
            sel = sel[sel["form"].isin(self.annual_forms)]
        best = sel.groupby(["item", "end"])["rank"].transform("min")
        sel = sel[sel["rank"] == best]
        return sel.sort_values(["item", "end", "filed"]).reset_index(drop=True)


def _fix_share_scale(panel: Panel) -> Panel:
    """Rescale share counts that were tagged in thousands or millions, when NI / EPS shows the true count.
    The original fact stays in the provenance; the factor is recorded in `panel.rescaled`."""
    v = panel.values
    if "NET_INCOME" not in v:
        return panel
    for sh_item, eps_item in SHARE_ITEMS.items():
        if sh_item not in v or eps_item not in v:
            continue
        implied = (v["NET_INCOME"] / v[eps_item]).abs()
        for p in v.index:
            sh, imp = v.at[p, sh_item], implied.get(p, np.nan)
            if not (np.isfinite(sh) and np.isfinite(imp) and sh > 0 and imp > 0):
                continue
            for factor in (1e6, 1e3):
                if abs(imp / (sh * factor) - 1) <= 0.05:
                    v.at[p, sh_item] = sh * factor
                    panel.rescaled[(sh_item, p)] = factor
                    break
    return panel


def _cluster_dates(dates: list) -> list[pd.Timestamp]:
    out: list[pd.Timestamp] = []
    for d in dates:
        d = pd.Timestamp(d)
        if out and d - out[-1] <= DATE_TOL:
            out[-1] = d  # keep the later of near-duplicates (52/53-week calendars)
        else:
            out.append(d)
    return out
