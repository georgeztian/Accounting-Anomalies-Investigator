"""Quantify financial significance: statement effects, materiality ratios, SAB 99 flags."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from ffi.context import RunContext
from ffi.detectors.base import ImpactEstimate
from ffi.detectors.stats import safe_div

THRESHOLDS = {"pretax": 0.05, "revenue": 0.005, "assets": 0.005, "equity": 0.01}


@dataclass
class Materiality:
    impact_low: float
    impact_mid: float
    impact_high: float
    pretax_effect: float
    net_income_effect: float
    eps_effect: float | None
    pct_pretax: float | None
    pct_revenue: float | None
    pct_assets: float | None
    pct_equity: float | None
    multiple: float  # largest (ratio / threshold); 1.0 = at the auditor-style threshold
    benchmark: str
    pct_market_value: float | None = None
    sab99: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def assess(ctx: RunContext, impact: ImpactEstimate | None) -> Materiality | None:
    """Treat the impact as a pre-tax income effect (an upper bound for balance-sheet gaps)."""
    if impact is None or not np.isfinite(impact.mid) or impact.mid <= 0:
        return None
    tm = ctx.tuning.get("materiality", {})
    thresholds = {"pretax": tm.get("pretax_pct", THRESHOLDS["pretax"]), "revenue": tm.get("revenue_pct", THRESHOLDS["revenue"]),
                  "assets": tm.get("assets_pct", THRESHOLDS["assets"]), "equity": tm.get("equity_pct", THRESHOLDS["equity"])}
    pt, rev, ta, eq = ctx.a("PRETAX_INCOME"), ctx.a("REVENUE"), ctx.a("TOTAL_ASSETS"), ctx.a("EQUITY")
    tax = ctx.tax_rate()
    ni_eff = impact.mid * (1 - tax)
    sh = ctx.a("SHARES_DILUTED")
    eps = safe_div(ni_eff, sh) if np.isfinite(sh) and sh > 0 else None
    ratios = {
        "pretax": safe_div(impact.mid, abs(pt)) if np.isfinite(pt) and pt != 0 else float("nan"),
        "revenue": safe_div(impact.mid, abs(rev)),
        "assets": safe_div(impact.mid, abs(ta)),
        "equity": safe_div(impact.mid, abs(eq)) if np.isfinite(eq) and eq > 0 else float("nan"),
    }
    multiples = {k: v / thresholds[k] for k, v in ratios.items() if np.isfinite(v)}
    bench, mult = max(multiples.items(), key=lambda kv: kv[1]) if multiples else ("n/a", 0.0)
    # the pre-tax benchmark is the auditor's primary one; don't let tiny-asset ratios dominate
    if "pretax" in multiples:
        bench, mult = ("pretax", multiples["pretax"]) if multiples["pretax"] >= mult * 0.5 else (bench, mult)
    sab = []
    pt0 = ctx.a("PRETAX_INCOME", 1)
    if np.isfinite(pt) and pt > 0 and pt - impact.mid < 0:
        sab.append("would turn a pre-tax profit into a loss")
    if np.isfinite(pt) and np.isfinite(pt0) and pt > pt0 and pt - impact.mid < pt0:
        sab.append("would turn pre-tax income growth into a decline")
    eps_now, eps_prev = ctx.a("EPS_DILUTED"), ctx.a("EPS_DILUTED", 1)
    if eps is not None and np.isfinite(eps_now) and np.isfinite(eps_prev) and eps_now >= eps_prev and eps_now - eps < eps_prev:
        sab.append("would reverse the year-over-year EPS increase")
    mv = ctx.a("PUBLIC_FLOAT")
    ni = ctx.a("NET_INCOME")
    pct_mv = None
    if np.isfinite(mv) and np.isfinite(ni) and ni > 0:
        pe = mv / ni
        pct_mv = safe_div(ni_eff * min(pe, 60), mv)
    return Materiality(
        impact.low, impact.mid, impact.high, impact.mid, ni_eff, eps,
        _f(ratios["pretax"]), _f(ratios["revenue"]), _f(ratios["assets"]), _f(ratios["equity"]),
        float(mult), bench, _f(pct_mv) if pct_mv is not None else None, sab,
    )


def _f(v: float) -> float | None:
    return float(v) if v is not None and np.isfinite(v) else None
