"""RunContext: everything detectors may read for one company-period."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from ffi.detectors.base import EvidenceRef, Signal
from ffi.edgar.endpoints import FilingRef
from ffi.kg.store import DuckNXGraphStore
from ffi.peers.select import PeerSet
from ffi.profiles import Profile
from ffi.resolve.resolver import CompanyRef
from ffi.text.document import TextDoc
from ffi.xbrl.canonical import BY_ID
from ffi.xbrl.facts import FactStore, Panel

STATUTORY_TAX = 0.21
PERIMETER_CHANGE = 0.15  # a later report moves a year's revenue by more than this: the company changed shape
OFFICER_ROLE_TAGS = ("CFO change", "CEO change", "CAO change")  # written by pipeline.run._officer_change_kind


@dataclass
class Event:
    id: str
    type: str  # AUDITOR_CHANGE, NON_RELIANCE, OFFICER_CHANGE, LATE_FILING, AMENDED_10K, ACQUISITION, ...
    date: date
    accession: str | None
    description: str
    url: str | None = None


@dataclass
class RunContext:
    company: CompanyRef
    store: FactStore
    A: Panel
    A_latest: Panel
    Q: Panel
    filings: list[FilingRef]
    focus: pd.Timestamp
    as_of: date
    kg: DuckNXGraphStore
    peers: PeerSet | None = None
    xsec: pd.DataFrame = field(default_factory=pd.DataFrame)
    industry_ciks: set[int] = field(default_factory=set)
    industry_digits: int = 2
    docs: list[TextDoc] = field(default_factory=list)  # 10-Ks, oldest first, latest = focus year
    releases: list[TextDoc] = field(default_factory=list)  # 8-K Ex. 99 earnings releases
    events: list[Event] = field(default_factory=list)
    auditor: str | None = None
    llm: Any | None = None
    mode: str = "deterministic"
    tuning: dict[str, Any] = field(default_factory=dict)
    signals: dict[str, list[Signal]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    relationships: list[Any] = field(default_factory=list)  # (Relationship, (start, end)) from E.09
    client: Any | None = None  # EdgarClient, for detectors that fetch extra documents
    profile: Profile | None = None  # company type: which detectors and peer ratios apply
    currency: str = "USD"  # reporting currency of all monetary amounts
    annual_form: str = "10-K"  # 10-K | 20-F | 40-F
    history: int = 4  # fiscal years of figures the investigation tools return (all years reviewed in a multi-year run)
    cutoff: date | None = None  # last day this year's analysis reads (end of its filing season, never after as_of)

    # ------------------------------------------------------------------ periods
    @property
    def periods(self) -> list[pd.Timestamp]:
        return [p for p in self.A.periods if p <= self.focus]

    @property
    def perimeter_break(self) -> pd.Timestamp | None:
        """The latest fiscal year before the focus year whose revenue, as first reported, was later re-reported more than 15%
        different (a spin-off, a divestiture presented as discontinued operations: General Electric's FY2023 revenue was
        $67.95B and became $35.35B). Figures of that year and earlier describe another company than the focus year's, so no
        comparison with them is made (`period_at`, `hist`, `comparable_periods`). None when there is no such year."""
        if "_perimeter_break" not in self.__dict__:
            found = None
            try:
                v = self.store.versions("A")
                v = v[(v["item"] == "REVENUE") & (v["end"] < self.focus - pd.Timedelta(days=300))]
                for end, g in v.groupby("end"):
                    first, last = g.sort_values("filed")["val"].iloc[0], g.sort_values("filed")["val"].iloc[-1]
                    if first > 0 and abs(last - first) / first > PERIMETER_CHANGE and (found is None or end > found):
                        found = pd.Timestamp(end)
            except (AttributeError, KeyError, TypeError):  # a store without versions (tests with stand-in contexts)
                found = None
            self.__dict__["_perimeter_break"] = found
        return self.__dict__["_perimeter_break"]

    @property
    def comparable_periods(self) -> list[pd.Timestamp]:
        """The fiscal years up to the focus year that are on the focus year's perimeter (see `perimeter_break`)."""
        b = self.perimeter_break
        return [p for p in self.periods if b is None or p > b]

    def period_at(self, k: int = 0) -> pd.Timestamp | None:
        ps = self.comparable_periods
        return ps[-1 - k] if len(ps) > k else None

    @property
    def focus_label(self) -> str:
        return self.A.label(self.focus)

    def label(self, period: pd.Timestamp | None) -> str:
        return self.A.label(period) if period is not None else "n/a"

    def a(self, item: str, k: int = 0, latest: bool = False) -> float:
        p = self.period_at(k)
        if p is None:
            return float("nan")
        return (self.A_latest if latest else self.A).get(item, p)

    def avg(self, item: str, k: int = 0) -> float:
        cur, prev = self.a(item, k), self.a(item, k + 1)
        if np.isfinite(cur) and np.isfinite(prev):
            return (cur + prev) / 2
        return cur

    def hist(self, item: str, n: int = 6) -> pd.Series:
        s = self.A.series(item)
        b = self.perimeter_break
        return s[(s.index <= self.focus) & ((s.index > b) if b is not None else True)].tail(n)

    def q_periods(self) -> list[pd.Timestamp]:
        return [p for p in self.Q.periods if p <= self.focus + pd.Timedelta(days=8)]

    def qhist(self, item: str, n: int = 20) -> pd.Series:
        s = self.Q.series(item)
        return s[s.index <= self.focus + pd.Timedelta(days=8)].tail(n)

    def tax_rate(self) -> float:
        pt, tax = self.a("PRETAX_INCOME"), self.a("INCOME_TAX")
        if np.isfinite(pt) and np.isfinite(tax) and pt > 0:
            r = tax / pt
            if 0.05 <= r <= 0.40:
                return float(r)
        return STATUTORY_TAX

    # ------------------------------------------------------------------ evidence
    def filing(self, accession: str) -> FilingRef | None:
        for f in self.filings:
            if f.accession == accession:
                return f
        return None

    def doc_url(self, accession: str | None) -> str | None:
        if not accession:
            return None
        f = self.filing(accession)
        nodash = accession.replace("-", "")
        if f is None:
            return f"https://www.sec.gov/Archives/edgar/data/{self.company.cik}/{nodash}/"
        return f"https://www.sec.gov/Archives/edgar/data/{self.company.cik}/{nodash}/{f.primary_document}"

    def ev(self, item: str, k: int = 0, role: str = "primary", panel: str = "A", period: pd.Timestamp | None = None) -> EvidenceRef | None:
        pnl = {"A": self.A, "Q": self.Q, "L": self.A_latest}[panel]
        p = period if period is not None else self.period_at(k)
        if p is None:
            return None
        rec = pnl.prov.get((item, p))
        if rec is None:
            return None
        label = f"{BY_ID[item].label}, {pnl.label(p)}"
        if (item, p) in pnl.derived:
            label += " (derived: FY less Q1-Q3)"
        return EvidenceRef(
            node_id=f"fact:{rec.fact_id}", kind="fact", label=label, accession=rec.accn,
            doc_url=self.doc_url(rec.accn), concept=f"{rec.taxonomy}:{rec.concept}", period=pnl.label(p),
            value=rec.val, unit=rec.unit, filed=rec.filed.date().isoformat(), role=role,  # type: ignore[arg-type]
        )

    def concept_switched(self, item: str, k0: int = 0, k1: int = 1) -> str | None:
        """Note when two periods of `item` come from different XBRL concepts (values may not be comparable)."""
        p0, p1 = self.period_at(k0), self.period_at(k1)
        r0 = self.A.prov.get((item, p0)) if p0 is not None else None
        r1 = self.A.prov.get((item, p1)) if p1 is not None else None
        if r0 and r1 and r0.concept != r1.concept:
            return f"{BY_ID[item].label.lower()} is reported under a different XBRL element this year than the year before"
        return None

    def evs(self, items: list[str], ks: tuple[int, ...] = (0, 1), role: str = "primary") -> list[EvidenceRef]:
        out = []
        for it in items:
            for k in ks:
                e = self.ev(it, k, role)
                if e is not None:
                    out.append(e)
        return out

    def ev_span(self, doc: TextDoc, start: int, end: int, label: str, role: str = "primary") -> EvidenceRef:
        para = next((p for p in doc.paragraphs if p.start <= start < p.end + 1), None)
        return EvidenceRef(
            node_id=f"span:{doc.accession}#{start}-{end}", kind="span", label=label, accession=doc.accession,
            doc_url=doc.url, section=para.section if para else None, char_span=(start, end),
            quote=doc.text[start:end][:400], period=doc.period_end, role=role,  # type: ignore[arg-type]
        )

    def ev_event(self, e: Event, role: str = "supporting") -> EvidenceRef:
        return EvidenceRef(
            node_id=f"event:{e.id}", kind="event", label=f"{e.type.replace('_', ' ').title()} ({e.date})",
            accession=e.accession, doc_url=e.url, quote=e.description[:400], filed=str(e.date), role=role,  # type: ignore[arg-type]
        )

    # ------------------------------------------------------------------ peers
    def peer_values(self, fn: Callable[[Panel, pd.Timestamp, int], float], k: int = 0) -> list[float]:
        """Apply fn(peer_panel, aligned_period, k) to each loaded peer; k shifts back in years."""
        out: list[float] = []
        if not self.peers:
            return out
        target = self.period_at(k)
        if target is None:
            return out
        for p in self.peers.loaded:
            per = self.peers.aligned_period(p, target)
            if per is None:
                continue
            try:
                v = fn(p.annual, per, k)  # type: ignore[arg-type]
            except (KeyError, IndexError, ZeroDivisionError):
                continue
            if v is not None and np.isfinite(v):
                out.append(float(v))
        return out

    # ------------------------------------------------------------------ text
    @property
    def doc(self) -> TextDoc | None:
        return self.docs[-1] if self.docs else None

    @property
    def prior_doc(self) -> TextDoc | None:
        return self.docs[-2] if len(self.docs) > 1 else None

    def officer_changes(self, within_days: int) -> list[Event]:
        """Item 5.02 filings read and found to change the CEO, CFO or chief accounting officer. Most 5.02 filings are
        about pay plans or directors; those, and filings that could not be read, are not officer changes."""
        return [e for e in self.events_of("OFFICER_CHANGE", within_days=within_days)
                if any(tag in e.description for tag in OFFICER_ROLE_TAGS)]

    def events_of(self, *types: str, within_days: int | None = None) -> list[Event]:
        out = [e for e in self.events if e.type in types]
        if within_days is not None:
            lo = (self.focus - pd.Timedelta(days=within_days)).date()
            hi = (self.focus + pd.Timedelta(days=120)).date()
            out = [e for e in out if lo <= e.date <= hi]
        return out


def announced_quarter(focus: pd.Timestamp, released: pd.Timestamp) -> pd.Timestamp:
    """The quarter an earnings release dated `released` announces: the latest quarter end on the fiscal calendar of the
    year ending `focus` at least a week before the release (releases follow their quarter by 2-10 weeks)."""
    ends = [focus + pd.DateOffset(months=3 * k) for k in range(-8, 9)]
    return max(e for e in ends if e <= released - pd.Timedelta(days=7))


def prev_period(panel: Panel, period: pd.Timestamp) -> pd.Timestamp | None:
    idx = list(panel.values.index)
    try:
        i = idx.index(period)
    except ValueError:
        return None
    return idx[i - 1] if i > 0 else None


def pv(panel: Panel, item: str, period: pd.Timestamp | None, back: int = 0) -> float:
    """Value of `item` `back` fiscal years before `period` in a (peer) panel."""
    p = period
    for _ in range(back):
        if p is None:
            return float("nan")
        p = prev_period(panel, p)
    return panel.get(item, p) if p is not None else float("nan")
