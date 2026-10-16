"""Cross-sectional data for industry-year models, built from XBRL frames (BLUEPRINT §5, §7).

Frames return one concept for every filer in a calendar period. Annual (duration) frames give each
firm's fiscal-year value and its period end; instant values are taken from whichever quarterly
instant frame matches that firm's fiscal year end.
Note: frames carry the latest reported value, so cross-section baselines are approximately
point-in-time (restatements of peers can leak in). The target company always uses its own
as-first-reported values.
"""

from __future__ import annotations

import pandas as pd

from ffi.edgar import endpoints
from ffi.edgar.client import EdgarClient

XSEC_VERSION = 2  # bump when the concept lists change, so cached cross-sections are rebuilt
DURATION_CONCEPTS: dict[str, tuple[str, ...]] = {
    "REVENUE": ("Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet",
                "RevenuesNetOfInterestExpense"),  # last: banks and card issuers report revenue net of interest expense
    "NET_INCOME": ("NetIncomeLoss",),
    "CFO": ("NetCashProvidedByUsedInOperatingActivities",),
    "COGS": ("CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold"),
    "SGA": ("SellingGeneralAndAdministrativeExpense",),
    "OPERATING_INCOME": ("OperatingIncomeLoss",),
    "GROSS_PROFIT": ("GrossProfit",),
}
INSTANT_CONCEPTS: dict[str, tuple[str, ...]] = {
    "TOTAL_ASSETS": ("Assets",),
    "AR": ("AccountsReceivableNetCurrent",),
    "PPE_NET": ("PropertyPlantAndEquipmentNet",),
    "INVENTORY": ("InventoryNet",),
}


def calendar_year_of(end: pd.Timestamp) -> int:
    """Calendar year a fiscal year mostly overlaps (how SEC frames assign CY periods)."""
    return end.year if end.month >= 7 else end.year - 1


def annual_frame(client: EdgarClient, item: str, year: int) -> pd.DataFrame:
    parts = []
    for concept in DURATION_CONCEPTS[item]:
        f = endpoints.frame(client, concept, f"CY{year}")
        if not f.empty:
            parts.append(f[["cik", "end", "val"]].assign(_p=len(parts)))
    if not parts:
        return pd.DataFrame(columns=["cik", "end", item])
    df = pd.concat(parts).sort_values("_p").drop_duplicates("cik")
    return df.rename(columns={"val": item}).drop(columns="_p")


def instant_frame(client: EdgarClient, item: str, year: int) -> pd.DataFrame:
    parts = []
    for concept in INSTANT_CONCEPTS[item]:
        for q in (1, 2, 3, 4):
            f = endpoints.frame(client, concept, f"CY{year}Q{q}I")
            if not f.empty:
                parts.append(f[["cik", "end", "val"]])
    if not parts:
        return pd.DataFrame(columns=["cik", "end", "val"])
    df = pd.concat(parts)
    df["end"] = pd.to_datetime(df["end"])
    return df


def build(client: EdgarClient, ciks: list[int] | None, years: list[int], max_age_days: int = 30) -> pd.DataFrame:
    """Rows: (cik, year, fye, REVENUE, NET_INCOME, ..., TOTAL_ASSETS, AR, ...). All-filer builds are cached as Parquet."""
    if ciks is None:
        import time

        path = client.cache_dir.parent / "xsec" / f"xsec_v{XSEC_VERSION}_{'_'.join(map(str, sorted(years)))}.parquet"
        if path.exists() and (client.offline or time.time() - path.stat().st_mtime < max_age_days * 86400):
            return pd.read_parquet(path)
        df = _build(client, None, years)
        if not df.empty:
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(path, index=False)
        return df
    return _build(client, ciks, years)


def _prefetch(client: EdgarClient, years: list[int]) -> None:
    """Download every frame the build needs concurrently (the build then reads them from the cache)."""
    from ffi.concurrency import map_concurrent

    jobs = [(c, f"CY{y}") for y in years for item in DURATION_CONCEPTS for c in DURATION_CONCEPTS[item]]
    jobs += [(c, f"CY{y}Q{q}I") for y in years for item in INSTANT_CONCEPTS for c in INSTANT_CONCEPTS[item] for q in (1, 2, 3, 4)]
    map_concurrent(lambda j: endpoints.frame(client, j[0], j[1]), jobs, workers=8)


def _build(client: EdgarClient, ciks: list[int] | None, years: list[int]) -> pd.DataFrame:
    if not client.offline:
        _prefetch(client, years)
    cikset = set(ciks) if ciks else None
    out = []
    for year in years:
        base: pd.DataFrame | None = None
        for item in DURATION_CONCEPTS:
            f = annual_frame(client, item, year)
            if cikset is not None:
                f = f[f["cik"].isin(cikset)]
            f = f.rename(columns={"end": f"end_{item}"})
            base = f if base is None else base.merge(f, on="cik", how="outer")
        if base is None or base.empty:
            continue
        end_cols = [c for c in base.columns if c.startswith("end_")]
        base["fye"] = pd.to_datetime(base[end_cols].bfill(axis=1).iloc[:, 0])
        base = base.drop(columns=end_cols)
        for item in INSTANT_CONCEPTS:
            inst = instant_frame(client, item, year)
            if cikset is not None:
                inst = inst[inst["cik"].isin(cikset)]
            merged = base[["cik", "fye"]].merge(inst, on="cik", how="inner")
            merged = merged[(merged["end"] - merged["fye"]).abs() <= pd.Timedelta(days=10)]
            merged = merged.drop_duplicates("cik")[["cik", "val"]].rename(columns={"val": item})
            base = base.merge(merged, on="cik", how="left")
        base["year"] = year
        out.append(base)
    if not out:
        return pd.DataFrame()
    return pd.concat(out, ignore_index=True)
