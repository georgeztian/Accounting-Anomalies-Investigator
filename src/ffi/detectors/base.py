"""Detector interface, Signal record and registry."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from ffi.context import RunContext


class Area(StrEnum):
    INTEGRITY = "integrity"
    REVENUE = "revenue"
    RECEIVABLES = "receivables"
    INVENTORY = "inventory"
    MARGINS = "margins"
    CAPITALIZATION = "capitalization"
    DEPRECIATION = "depreciation"
    RESERVES = "reserves"
    ACCRUALS = "accruals"
    CASH_FLOW = "cash_flow"
    DEBT = "debt"
    MA = "m&a"
    RELATED_PARTY = "related_party"
    NON_GAAP = "non_gaap"
    DISCLOSURE = "disclosure"
    GOVERNANCE = "governance"
    EARNINGS_QUALITY = "earnings_quality"


class Scheme(StrEnum):
    PREMATURE_REVENUE = "premature_revenue"
    BOGUS_REVENUE = "bogus_revenue"
    ONE_TIME_GAINS = "one_time_gains"
    SHIFT_EXPENSES_LATER = "shift_expenses_later"
    HIDE_LIABILITIES = "hide_liabilities"
    SHIFT_REVENUE_FORWARD = "shift_revenue_forward"
    COOKIE_JAR = "cookie_jar"
    CFO_INFLATION = "cfo_inflation"
    MISLEADING_METRICS = "misleading_metrics"
    ACQUISITION_GAMES = "acquisition_games"
    REPORTING_INTEGRITY = "reporting_integrity"


Direction = Literal["income_increasing", "income_decreasing", "neutral"]


@dataclass
class EvidenceRef:
    node_id: str
    kind: Literal["fact", "table_cell", "span", "derived", "event"]
    label: str
    accession: str | None = None
    doc_url: str | None = None
    concept: str | None = None
    period: str | None = None
    value: float | None = None
    unit: str | None = None
    section: str | None = None
    char_span: tuple[int, int] | None = None
    quote: str | None = None
    filed: str | None = None
    role: Literal["primary", "supporting", "counter"] = "primary"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ImpactEstimate:
    low: float
    mid: float
    high: float
    method: str
    basis: str = ""  # human-readable formula

    def __post_init__(self) -> None:
        lo, mid, hi = sorted([max(0.0, float(self.low)), max(0.0, float(self.mid)), max(0.0, float(self.high))])
        self.low, self.mid, self.high = lo, mid, hi

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Signal:
    detector_id: str
    name: str
    family: str
    area: Area
    period: str
    raw_value: float | None
    expected: float | None
    severity: float
    flagged: bool
    rationale: str
    scheme: Scheme | None = None
    direction: Direction = "neutral"
    unit: str = ""
    percentile_self: float | None = None
    percentile_peer: float | None = None
    p_value: float | None = None
    impact: ImpactEstimate | None = None
    evidence: list[EvidenceRef] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    suppressed_by: str | None = None
    detector_version: str = "1"
    flag_at: float = 0.5  # the severity at which this result is flagged (kept so a recalibrated severity is re-judged)

    @property
    def key(self) -> str:
        return f"{self.detector_id}@{self.period}"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["area"] = str(self.area)
        d["scheme"] = str(self.scheme) if self.scheme else None
        return d


class SkipDetector(Exception):
    """Raised by a detector that cannot run (missing data, needs LLM, etc.)."""


@dataclass
class DetectorSpec:
    id: str
    name: str
    family: str
    area: Area
    fn: Callable[[RunContext], list[Signal]]
    depends_on: tuple[str, ...] = ()
    needs_llm: bool = False
    needs_text: bool = False
    needs_peers: bool = False
    description: str = ""
    version: str = "1"


REGISTRY: dict[str, DetectorSpec] = {}


def detector(
    id: str,
    name: str,
    family: str,
    area: Area,
    depends_on: tuple[str, ...] = (),
    needs_llm: bool = False,
    needs_text: bool = False,
    needs_peers: bool = False,
    description: str = "",
):
    def wrap(fn: Callable[[RunContext], list[Signal]]):
        if id in REGISTRY:
            raise ValueError(f"duplicate detector id {id}")
        REGISTRY[id] = DetectorSpec(
            id, name, family, area, fn, depends_on, needs_llm, needs_text, needs_peers, description or (fn.__doc__ or "").strip()
        )
        return fn

    return wrap


def load_all() -> dict[str, DetectorSpec]:
    """Import every detector module so their decorators register."""
    from ffi.detectors import (  # noqa: F401
        a_integrity,
        b_scores,
        c_accounts,
        d_stats,
        e_text,
        f_graph,
    )

    return REGISTRY
