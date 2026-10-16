"""Shared robust statistics for detectors (robust z-scores and percentiles)."""

from __future__ import annotations

import contextvars
from collections.abc import Iterable

import numpy as np
from scipy import stats as sps


def clean(values: Iterable[float]) -> np.ndarray:
    a = np.asarray([v for v in values if v is not None], dtype=float)
    return a[np.isfinite(a)]


def safe_div(a: float, b: float) -> float:
    if b is None or a is None or not np.isfinite(a) or not np.isfinite(b) or b == 0:
        return float("nan")
    return float(a) / float(b)


def robust_z(x: float, reference: Iterable[float], min_n: int = 3) -> float:
    """z = (x - median) / (1.4826 * MAD); falls back to std when MAD is ~0."""
    ref = clean(reference)
    if not np.isfinite(x) or len(ref) < min_n:
        return float("nan")
    med = float(np.median(ref))
    mad = 1.4826 * float(np.median(np.abs(ref - med)))
    scale = mad if mad > 1e-12 else float(np.std(ref, ddof=1)) if len(ref) > 1 else 0.0
    if scale <= 1e-12:
        scale = max(abs(med) * 0.05, 1e-9)
    return (x - med) / scale


def percentile_of(x: float, reference: Iterable[float]) -> float:
    ref = clean(reference)
    if not np.isfinite(x) or len(ref) == 0:
        return float("nan")
    return float(sps.percentileofscore(ref, x, kind="mean"))


def z_to_p(z: float, two_sided: bool = False) -> float:
    if not np.isfinite(z):
        return float("nan")
    p = float(sps.norm.sf(abs(z) if two_sided else z))
    return min(1.0, 2 * p) if two_sided else p


def severity_from_z(z: float, start: float = 1.5, full: float = 4.0) -> float:
    """Map a one-sided z-score to 0..1: 0 below `start`, 1 at `full`."""
    if not np.isfinite(z):
        return 0.0
    return float(np.clip((z - start) / (full - start), 0.0, 1.0))


CURRENCY: contextvars.ContextVar[str] = contextvars.ContextVar("reporting_currency", default="USD")
CURRENCY_SYMBOLS = {"USD": "$", "EUR": "€", "GBP": "£", "JPY": "¥", "CNY": "CN¥", "CAD": "C$", "AUD": "A$", "HKD": "HK$",
                    "INR": "₹", "KRW": "₩", "BRL": "R$", "MXN": "MX$", "TWD": "NT$", "SGD": "S$"}


def money_prefix(currency: str | None = None) -> str:
    cur = currency or CURRENCY.get()
    return CURRENCY_SYMBOLS.get(cur, f"{cur} ")


def fmt_money(v: float | None) -> str:
    """Amount in the run's reporting currency (set by the pipeline; USD by default)."""
    if v is None or not np.isfinite(v):
        return "n/a"
    a = abs(v)
    sign = "-" if v < 0 else ""
    c = money_prefix()
    if a >= 1e9:
        return f"{sign}{c}{a / 1e9:,.2f}B"
    if a >= 1e6:
        return f"{sign}{c}{a / 1e6:,.1f}M"
    if a >= 1e3:
        return f"{sign}{c}{a / 1e3:,.0f}K"
    return f"{sign}{c}{a:,.0f}"


def fmt_pct(v: float | None, digits: int = 1) -> str:
    if v is None or not np.isfinite(v):
        return "n/a"
    return f"{v * 100:.{digits}f}%"


def fmt_num(v: float | None, digits: int = 2) -> str:
    if v is None or not np.isfinite(v):
        return "n/a"
    return f"{v:,.{digits}f}"
