"""Cross-sectional data for industry-year models, built from XBRL frames.

Frames return one concept for every filer in a calendar period. Annual (duration) frames give each
firm's fiscal-year value and its period end; instant values are taken from whichever quarterly
instant frame matches that firm's fiscal year end.
Note: frames carry the latest reported value, so cross-section baselines are approximately
point-in-time (restatements of peers can leak in). The target company always uses its own
as-first-reported values.
"""

from __future__ import annotations

import contextlib
import re

import pandas as pd

from ffi.edgar import endpoints
from ffi.edgar.client import EdgarClient

XSEC_VERSION = 3  # bump when the concept lists change, so cached cross-sections are rebuilt
# Each list keeps the priority order of the canonical item (xbrl/canonical.py; a unit test checks it): total revenue net
# of interest expense before ASC 606 contract revenue, which at banks and card issuers leaves out net interest income
DURATION_CONCEPTS: dict[str, tuple[str, ...]] = {
    "REVENUE": ("Revenues", "RevenuesNetOfInterestExpense", "RevenueFromContractWithCustomerExcludingAssessedTax",
                "SalesRevenueNet"),
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
    """Rows: (cik, year, fye, REVENUE, NET_INCOME, ..., TOTAL_ASSETS, AR, ...). All-filer builds are cached as Parquet,
    one file per calendar year: runs for neighbouring fiscal years (a multi-year review, --period) share three of
    their four years."""
    if ciks is None:
        import time

        folder = client.cache_dir.parent / "xsec"
        paths = {y: folder / f"xsec_v{XSEC_VERSION}_{y}.parquet" for y in sorted(years)}
        fresh = {y: p for y, p in paths.items() if p.exists() and (client.offline or time.time() - p.stat().st_mtime < max_age_days * 86400)}
        missing = [y for y in paths if y not in fresh]
        for url in [u for y in fresh for u in _frame_urls(y)]:  # raw frames kept by earlier versions (~15 MB a year)
            client.evict(url)
        built = _build(client, None, missing) if missing else pd.DataFrame()
        current = re.compile(rf"xsec_v{XSEC_VERSION}_\d{{4}}\.parquet")
        for old in folder.glob("xsec_*.parquet"):
            if not current.fullmatch(old.name):
                with contextlib.suppress(OSError):  # a file another process holds open is removed by a later run
                    old.unlink(missing_ok=True)  # files of earlier versions (multi-year windows), which this one never reads
        for y in missing:
            part = built[built["year"] == y] if not built.empty else built
            if not part.empty:
                try:
                    folder.mkdir(parents=True, exist_ok=True)
                    part.to_parquet(paths[y], index=False)
                except OSError:
                    continue  # not cached (the file is held open elsewhere); the raw frames stay for the next run
                for url in _frame_urls(y):  # the year's raw frames (~15 MB) are only read to build this file
                    client.evict(url)
        frames = [pd.read_parquet(p) for p in fresh.values()] + ([built] if not built.empty else [])
        frames = [f for f in frames if not f.empty]
        return pd.concat(frames, ignore_index=True).sort_values("year", kind="stable", ignore_index=True) if frames else pd.DataFrame()
    return _build(client, ciks, years)


def _frame_jobs(year: int) -> list[tuple[str, str]]:
    """(concept, frame period) of every frame one calendar year's cross-section is built from."""
    jobs = [(c, f"CY{year}") for item in DURATION_CONCEPTS for c in DURATION_CONCEPTS[item]]
    return jobs + [(c, f"CY{year}Q{q}I") for item in INSTANT_CONCEPTS for c in INSTANT_CONCEPTS[item] for q in (1, 2, 3, 4)]


def _frame_urls(year: int) -> list[str]:
    return [endpoints.FRAMES_URL.format(taxonomy="us-gaap", concept=c, unit="USD", period=p) for c, p in _frame_jobs(year)]


def _prefetch(client: EdgarClient, years: list[int]) -> None:
    """Download every frame the build needs concurrently (the build then reads them from the cache)."""
    from ffi.concurrency import map_concurrent

    map_concurrent(lambda j: endpoints.frame(client, j[0], j[1]), [j for y in years for j in _frame_jobs(y)], workers=8)


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
