"""Signal construction helpers shared by detector modules."""

from __future__ import annotations

from typing import Any

import numpy as np

from ffi.detectors.base import REGISTRY, Signal, SkipDetector
from ffi.detectors.plausibility import hold_back

# Inputs cited by default when a detector emits a signal without explicit evidence: the facts it was computed
# from, or (text detectors) the filing section it analysed. Every flaggable conclusion is then traceable.
DEFAULT_INPUTS: dict[str, tuple[str, ...]] = {
    "A.10": ("GOODWILL",), "A.14": ("TOTAL_ASSETS",), "A.16": ("TOTAL_ASSETS",),
    "B.06": ("NET_INCOME", "CFO", "AR", "INVENTORY", "OTHER_CURRENT_ASSETS", "PPE_GROSS", "TOTAL_ASSETS"),
    "B.07": ("NET_INCOME", "CFO", "TOTAL_ASSETS"), "B.08": ("TOTAL_ASSETS", "RETAINED_EARNINGS", "OPERATING_INCOME", "PUBLIC_FLOAT"),
    "B.09": ("NET_INCOME", "EPS_DILUTED", "TOTAL_ASSETS"), "B.10": ("OPERATING_INCOME", "CFO", "TOTAL_ASSETS"),
    "C.CF.02": ("CFO", "CHANGE_AP", "CHANGE_AR", "CHANGE_INVENTORY"),
    "C.RES.01": ("ALLOWANCE", "INV_RESERVE", "WARRANTY_ACCRUAL", "PRETAX_INCOME"), "C.RES.06": ("LITIGATION_ACCRUAL",),
    "D.02": ("REVENUE", "NET_INCOME", "CFO", "TOTAL_ASSETS", "AR", "INVENTORY"),
    "D.03": ("REVENUE", "NET_INCOME", "CFO", "TOTAL_ASSETS", "AR", "INVENTORY"),
    "D.05": ("TOTAL_ASSETS",), "D.06": ("REVENUE", "NET_INCOME", "CFO", "TOTAL_ASSETS"), "D.08": ("TOTAL_ASSETS",),
    "E.13": ("AR", "INVENTORY", "GOODWILL", "DEFERRED_REV_CURRENT"),
    "C.NG.01": ("EPS_DILUTED",),
}
DEFAULT_SECTIONS: dict[str, str] = {"E.03": "item1a", "E.05": "item7", "E.06": "item7", "E.07": "item7", "E.08": "item8",
                                    "C.RES.06": "item3", "E.01": "item8", "E.02": "item7", "E.09": "item8"}


def default_evidence(detector_id: str, ctx: Any) -> list:
    out = []
    items = DEFAULT_INPUTS.get(detector_id, ())
    if items and hasattr(ctx, "evs"):
        out = ctx.evs(list(items), ks=(0, 1))
    sec = DEFAULT_SECTIONS.get(detector_id)
    doc = getattr(ctx, "doc", None)
    if sec and doc is not None:
        paras = doc.section_paragraphs(sec)
        if paras:
            p = paras[0]
            out.append(ctx.ev_span(doc, p.start, p.end, f"Section analysed: {p.text[:70]}"))
    return out


def mk(detector_id: str, ctx: Any, *, period: str | None = None, **kw: Any) -> Signal:
    spec = REGISTRY[detector_id]
    sev = _f(kw.pop("severity", 0.0)) or 0.0  # a severity computed from missing inputs (NaN) means nothing unusual
    sev = float(np.clip(sev, 0.0, 1.0))
    for k in ("percentile_peer", "percentile_self", "p_value"):  # NaN when there was no reference group
        if k in kw:
            kw[k] = _f(kw[k])
    if kw.get("unit") == "USD":  # monetary results are in the reporting currency
        kw["unit"] = getattr(ctx, "currency", None) or "USD"
    flag_at = kw.pop("flag_at", (getattr(ctx, "tuning", None) or {}).get("scoring", {}).get("flag_severity", 0.5))
    flagged = kw.pop("flagged", None)
    if flagged is None:
        flagged = sev >= flag_at
    if not kw.get("evidence") and sev > 0:
        kw["evidence"] = default_evidence(detector_id, ctx)
    return hold_back(Signal(
        detector_id=detector_id,
        name=spec.name,
        family=spec.family,
        area=kw.pop("area", spec.area),
        period=period or ctx.focus_label,
        raw_value=_f(kw.pop("raw_value", None)),
        expected=_f(kw.pop("expected", None)),
        severity=sev,
        flagged=bool(flagged),
        rationale=kw.pop("rationale", ""),
        flag_at=float(flag_at),
        **kw,
    ), ctx)


def _f(v: Any) -> float | None:
    if v is None:
        return None
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return None
    return fv if np.isfinite(fv) else None


def need(*values: float, reason: str = "required data not reported") -> None:
    for v in values:
        if v is None or not np.isfinite(v):
            raise SkipDetector(reason)


def finite(*values: float) -> bool:
    return all(v is not None and np.isfinite(v) for v in values)


def year_earlier(index, q):
    """The period in `index` a year before `q`, matched by date (within 10 days, for 52/53-week calendars), or None. A
    quarter missing from the filings never shifts a year-on-year comparison onto another quarter."""
    return next((p for p in index if abs((q - p).days - 364) <= 10), None)
