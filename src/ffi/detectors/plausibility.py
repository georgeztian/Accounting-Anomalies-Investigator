"""Plausibility gate: no test result counts when its value or its inputs are physically impossible.

Applied to every result as it is built (`helpers.mk`), so no test can bypass it. A result that fails keeps its text and
its place in the "all tests" tables, but it is not flagged and adds nothing to the risk index; its reason is shown.

Value rules (what a number of this kind can possibly be):
    days                         0 to 1,500
    years                        0 to 100
    money in the reporting currency   at most 5 x the larger of total assets and revenue (a scale or unit error otherwise)
    relative gap / growth        within +/-100 (10,000%)

Input rules (a part cannot exceed its whole): current assets, net PP&E, cash, receivables, inventory and goodwill cannot
exceed total assets by more than 1%. A test that reads such an input is held back for that year. The integrity tests
(family A) are exempt, because finding relationships that do not hold is their job.
"""

from __future__ import annotations

import contextlib
from typing import Any

import numpy as np

from ffi.detectors.base import Signal

PARTS_OF_ASSETS = ("CURRENT_ASSETS", "PPE_NET", "CASH", "AR", "INVENTORY", "GOODWILL")
MONEY_SCALE = 5.0
MAX_DAYS, MAX_YEARS, MAX_GAP = 1500.0, 100.0, 100.0
EXEMPT_FAMILIES = ("A",)


def _fin(x: Any) -> bool:
    return isinstance(x, int | float | np.floating) and bool(np.isfinite(x))


def value_problem(s: Signal, ctx: Any) -> str | None:
    v, unit = s.raw_value, (s.unit or "").lower()
    if v is None or not _fin(v):
        return None
    if unit == "days" and not 0 <= v <= MAX_DAYS:
        return f"{v:,.0f} days is outside any possible range"
    if unit == "years" and not 0 <= v <= MAX_YEARS:
        return f"{v:,.1f} years is outside any possible range"
    if unit in ("relative gap", "growth", "growth spread") and abs(v) > MAX_GAP:
        return f"a change of {v:,.0f}x is outside any possible range"
    cur = (getattr(ctx, "currency", None) or "USD").lower()
    if unit and unit == cur:
        size = max((abs(x) for x in (ctx.a("TOTAL_ASSETS"), ctx.a("REVENUE")) if _fin(x)), default=0.0)
        if size > 0 and abs(v) > MONEY_SCALE * size:
            return f"an amount {abs(v) / size:,.0f} times the company's size (probably a unit or scale error)"
    return None


def incoherent_inputs(ctx: Any) -> dict[str, str]:
    """Inputs of the focus year that exceed total assets, {item: reason}; computed once per context."""
    cached = getattr(ctx, "_incoherent_inputs", None)
    if cached is not None:
        return cached
    out: dict[str, str] = {}
    ta = ctx.a("TOTAL_ASSETS")
    if _fin(ta) and ta > 0:
        for item in PARTS_OF_ASSETS:
            v = ctx.a(item)
            if _fin(v) and v > 1.01 * ta:
                out[item] = f"{item.replace('_', ' ').lower()} ({v / 1e9:,.2f}B) is larger than total assets ({ta / 1e9:,.2f}B)"
    with contextlib.suppress(AttributeError):  # a frozen or slotted context: recomputed on the next call
        ctx._incoherent_inputs = out
    return out


def input_problem(s: Signal, ctx: Any) -> str | None:
    if s.family in EXEMPT_FAMILIES:
        return None
    from ffi.detectors.helpers import DEFAULT_INPUTS
    from ffi.xbrl.canonical import ITEMS

    bad = incoherent_inputs(ctx)
    if not bad:
        return None
    used = set(DEFAULT_INPUTS.get(s.detector_id, ()))
    labels = {it.label: it.id for it in ITEMS}
    names = {e.label.rsplit(", ", 1)[0] for e in s.evidence if e.kind == "fact"}  # "PP&E, net, FY2024" -> "PP&E, net"
    used |= {labels[n] for n in names if n in labels}
    for item in sorted(used & set(bad)):
        return bad[item]
    return None


def problem(s: Signal, ctx: Any) -> str | None:
    """Why this result cannot be right, or None."""
    return value_problem(s, ctx) or input_problem(s, ctx)


def hold_back(s: Signal, ctx: Any) -> Signal:
    """Return the signal unchanged, or held back (not flagged, no weight) with the reason, when it is implausible."""
    if (s.severity <= 0 and not s.flagged) or not callable(getattr(ctx, "a", None)):
        return s  # nothing to hold back, or no figures to check against
    why = problem(s, ctx)
    if not why:
        return s
    s.tags = [*s.tags, "implausible"]
    s.suppressed_by = f"implausible result held back: {why}"
    if s.flagged or s.severity > 0:
        s.rationale += f" [Held back: {why}.]"
    s.severity, s.flagged = 0.0, False
    return s
