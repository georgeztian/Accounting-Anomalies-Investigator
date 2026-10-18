"""Independent audit of key conclusions, with a bounded correction loop.

The auditor does not trust what detectors say about their evidence. For every finding it re-derives
the evidence from the sources the run downloaded:

  evidence_present   at least one primary fact or filing quote supports the conclusion
  fact_values        every cited XBRL value is re-read from the raw companyfacts rows (same filing, concept and period)
  quotes             every quote is re-located verbatim in the filing text at its character span
  impact             low <= mid <= high and materiality is recomputed from scratch and compared
  corroboration      a pattern's members are still flagged, unsuppressed and drawn from >= 2 families (or >= 3 signals)
  narrative          every number in the narrative traces to a source value (facts, ledger, detector outputs)
  agent_text         the innocent explanations and questions the investigation wrote pass the same gate

Each round: audit -> apply corrections (drop or repair evidence, recompute materiality, downgrade
uncorroborated patterns, replace an ungrounded narrative) -> audit again. At most MAX_ROUNDS rounds;
a finding that still fails is marked unverified and loses weight in the risk index.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from ffi.agent import grounding
from ffi.detectors.base import EvidenceRef
from ffi.materiality.impact import assess as assess_materiality
from ffi.text.verify import locate

MAX_ROUNDS = 3


@dataclass
class AuditCheck:
    name: str
    status: str  # pass | fail | warn
    detail: str


@dataclass
class AuditRound:
    round: int
    checks: list[AuditCheck]
    corrections: list[str] = field(default_factory=list)

    @property
    def failed(self) -> list[AuditCheck]:
        return [c for c in self.checks if c.status == "fail"]


@dataclass
class AuditResult:
    status: str  # verified | verified_after_corrections | unverified
    rounds: list[AuditRound]
    auditor: str = "deterministic"

    @property
    def label(self) -> str:
        n = len(self.rounds)
        if self.status == "verified":
            return "Verified by independent audit (round 1)"
        if self.status == "verified_after_corrections":
            fixes = sum(len(r.corrections) for r in self.rounds)
            return f"Verified after {fixes} correction(s) over {n} rounds"
        return f"Unverified after {n} audit rounds"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["label"] = self.label
        return d


# ------------------------------------------------------------------------------------- helpers
def _facts_index(ctx) -> dict[tuple[str, str, str], np.ndarray]:
    return ctx.store.values_by_filing()  # built once per company and shared by every year of a multi-year review


def _concept_name(ev: EvidenceRef) -> str | None:
    if not ev.concept:
        return None
    return ev.concept.split(":", 1)[-1]


def _fact_end(ev: EvidenceRef) -> str:
    """The period end a cited fact is for, from its id ("fact:taxonomy:concept|unit|start|end|accession")."""
    parts = ev.node_id.split("|")
    return parts[3] if ev.node_id.startswith("fact:") and len(parts) == 5 else ""


def _fact_matches(idx, e: EvidenceRef) -> bool:
    """The cited value is what the cited filing reports under the cited concept for the cited period."""
    vals = idx.get((e.accession, _concept_name(e) or "", _fact_end(e)), np.array([]))
    return bool(np.any(np.abs(vals - e.value) <= max(abs(e.value) * 0.0005, 0.5)))


def _doc_for(ctx, accession: str | None):
    for d in list(ctx.docs) + list(ctx.releases):
        if d.accession == accession:
            return d
    return None


def _allowed_numbers(ctx, finding, ledger: list[dict] | None = None) -> list[float]:
    from ffi.agent.tools import facts_table

    vals = [r["value"] for r in facts_table(ctx, finding)]
    for s in finding.signals:
        vals += [x for x in (s.percentile_peer, s.percentile_self) if x is not None]
        for m in grounding.NUM_RE.finditer(grounding.strip_dates(s.rationale)):  # code-generated detector outputs are a valid source
            vals += [abs(c) for c in grounding._candidates(m)] + grounding._candidates(m)
    for e in ledger or []:
        vals += e.get("numbers", []) + grounding.numbers_in(e.get("text", ""))  # incl. amounts in quoted filing text
    from ffi.detectors.catalog import definition

    for d in finding.detectors:  # thresholds in the tests' own definitions ("rising > 10%")
        vals += grounding.numbers_in(definition(d))
    return [float(v) for v in vals if v is not None and np.isfinite(v)]


# ------------------------------------------------------------------------------------- audit
def audit_finding(ctx, finding, ledger: list[dict] | None = None) -> list[AuditCheck]:
    checks: list[AuditCheck] = []
    primary = [e for e in finding.evidence if e.kind in ("fact", "span") and e.role == "primary"]
    checks.append(AuditCheck("evidence_present", "pass" if primary else "fail",
                             f"{len(primary)} primary fact/quote references" if primary else "no primary fact or filing quote supports the conclusion"))

    idx = _facts_index(ctx)
    bad_facts = []
    n_facts = 0
    for e in finding.evidence:
        if e.kind != "fact" or e.value is None or not e.accession:
            continue
        n_facts += 1
        if not _fact_matches(idx, e):
            bad_facts.append(e.label)
    checks.append(AuditCheck("fact_values", "fail" if bad_facts else "pass",
                             (f"{len(bad_facts)} of {n_facts} cited values not found in the filing's XBRL: " + "; ".join(bad_facts[:3]))
                             if bad_facts else f"{n_facts} cited XBRL values re-read from source filings and matched"))

    bad_quotes, n_quotes = [], 0
    for e in finding.evidence:
        if e.kind != "span" or not e.quote:
            continue
        n_quotes += 1
        doc = _doc_for(ctx, e.accession)
        ok = doc is not None and e.char_span is not None and doc.text[e.char_span[0]: e.char_span[1]].startswith(e.quote[:200])
        if not ok:
            bad_quotes.append(e.label)
    checks.append(AuditCheck("quotes", "fail" if bad_quotes else "pass",
                             (f"{len(bad_quotes)} of {n_quotes} quotes not found verbatim at their span: " + "; ".join(bad_quotes[:3]))
                             if bad_quotes else f"{n_quotes} quotes located verbatim in the filings"))

    imp = finding.impact
    if imp is not None:
        ordered = 0 <= imp.low <= imp.mid <= imp.high and imp.mid > 0
        fresh = assess_materiality(ctx, imp)
        same = (fresh is None and finding.materiality is None) or (
            fresh is not None and finding.materiality is not None
            and abs((fresh.pct_pretax or 0) - (finding.materiality.pct_pretax or 0)) < 1e-9
            and abs(fresh.multiple - finding.materiality.multiple) < 1e-9)
        st = "pass" if ordered and same else "fail"
        checks.append(AuditCheck("impact", st, "impact range ordered and materiality reproduces" if st == "pass"
                                 else ("impact range out of order" if not ordered else "materiality does not reproduce from the impact")))

    if finding.kind == "pattern":
        live = [s for s in finding.signals if s.flagged and not s.suppressed_by]
        fams = {s.family for s in live}
        ok = len(live) >= 2 and (len(fams) >= 2 or len(live) >= 3)
        exempt = finding.id == "reporting_integrity" and any(s.severity >= 0.8 for s in live)
        checks.append(AuditCheck("corroboration", "pass" if ok or exempt else "fail",
                                 f"{len(live)} live signals from {len(fams)} families" + (" (single severe integrity break)" if exempt and not ok else "")))

    text = finding.narrative or ""
    if text:
        own = [finding.title] + [x for s in finding.signals for x in (s.name, s.rationale)]
        rep = grounding.check(text, _allowed_numbers(ctx, finding, ledger), list(ctx.docs) + list(ctx.releases), known_text=own)
        st = "pass" if rep.ok else "fail"
        checks.append(AuditCheck("narrative", st, f"{rep.numbers_checked} numbers traced to sources" if rep.ok else
                                 "untraceable: " + ", ".join(rep.unsupported_numbers[:4] + [f'quote "{q}"' for q in rep.unverified_quotes[:2]])))

    bad_items = _ungrounded_agent_items(ctx, finding, ledger)
    if bad_items is not None:
        checks.append(AuditCheck("agent_text", "fail" if bad_items else "pass",
                                 f"{len(bad_items)} innocent explanation(s) or question(s) written by the investigation not traceable: "
                                 + "; ".join(t[:60] for t in bad_items[:2]) if bad_items else
                                 "the investigation's innocent explanations and questions trace to sources"))

    thin = [s.detector_id for s in finding.signals if s.percentile_peer is not None and s.p_value is None and len(finding.signals) == 1]
    if thin:
        checks.append(AuditCheck("baseline", "warn", "single-signal finding resting on a peer comparison"))
    return checks


def _ungrounded_agent_items(ctx, finding, ledger) -> list[str] | None:
    """The innocent explanations and questions the investigation wrote that fail the grounding gate; None when the
    finding carries none (they are the tests' own, written by code)."""
    written = set(getattr(finding, "agent_text", None) or [])
    items = [t for t in list(finding.alternatives) + list(finding.questions) if t in written]
    if not items:
        return None
    allowed = _allowed_numbers(ctx, finding, ledger)
    own = [finding.title] + [x for s in finding.signals for x in (s.name, s.rationale)]
    docs = list(ctx.docs) + list(ctx.releases)
    return [t for t in items if not grounding.check(t, allowed, docs, known_text=own).ok]


# ------------------------------------------------------------------------------------- corrections
def correct(ctx, finding, checks: list[AuditCheck], template_narrative) -> list[str]:
    fixes: list[str] = []
    failed = {c.name for c in checks if c.status == "fail"}
    if "fact_values" in failed:
        idx = _facts_index(ctx)
        keep = []
        for e in finding.evidence:
            if e.kind == "fact" and e.value is not None and e.accession and not _fact_matches(idx, e):
                fixes.append(f"removed unverifiable fact evidence '{e.label}'")
                continue
            keep.append(e)
        finding.evidence = keep
    if "quotes" in failed:
        keep = []
        for e in finding.evidence:
            if e.kind == "span" and e.quote:
                doc = _doc_for(ctx, e.accession)
                ok = doc is not None and e.char_span is not None and doc.text[e.char_span[0]: e.char_span[1]].startswith(e.quote[:200])
                if not ok:
                    loc = locate(doc, e.quote) if doc is not None else None
                    if loc:
                        e.char_span = loc
                        fixes.append(f"re-anchored quote '{e.label}' to its verbatim location")
                    else:
                        fixes.append(f"removed quote not found in the filing '{e.label}'")
                        continue
            keep.append(e)
        finding.evidence = keep
    if "evidence_present" in failed:
        extra = [e for s in finding.signals for e in s.evidence if e.kind in ("fact", "span")]
        for e in extra[:4]:
            e.role = "primary"
            finding.evidence.append(e)
        if extra:
            fixes.append(f"attached {min(4, len(extra))} primary references from the member signals")
    if "impact" in failed and finding.impact is not None:
        imp = finding.impact
        imp.low, imp.mid, imp.high = sorted([max(0.0, imp.low), max(0.0, imp.mid), max(0.0, imp.high)])
        finding.materiality = assess_materiality(ctx, imp)
        fixes.append("re-ordered the impact range and recomputed materiality")
    if "corroboration" in failed:
        live = [s for s in finding.signals if s.flagged and not s.suppressed_by]
        if len(live) < len(finding.signals):
            finding.signals = live or finding.signals
            fixes.append("removed members that are no longer flagged")
        if len(live) < 2 or (len({s.family for s in live}) < 2 and len(live) < 3):
            finding.kind = "signal"
            finding.score *= 0.6
            finding.title += " (single source)"
            fixes.append("downgraded from a corroborated pattern to a single-source signal")
    if "agent_text" in failed:
        bad = set(_ungrounded_agent_items(ctx, finding, finding.ledger) or [])
        finding.alternatives = [a for a in finding.alternatives if a not in bad] or [
            "No innocent explanation written by the investigation could be traced to the evidence."]
        finding.questions = [q for q in finding.questions if q not in bad] or ["What explains this movement, and is it expected to persist?"]
        fixes.append(f"removed {len(bad)} innocent explanation(s) or question(s) that could not be traced to sources")
    if "narrative" in failed:
        finding.narrative = template_narrative(finding)
        finding.narrative_source = f"template (previous {finding.narrative_source} narrative failed audit)"
        fixes.append("replaced the narrative with one generated from verified detector outputs")
    return fixes


def run_audit(ctx, finding, template_narrative, ledger: list[dict] | None = None, max_rounds: int = MAX_ROUNDS) -> AuditResult:
    rounds: list[AuditRound] = []
    for i in range(1, max_rounds + 1):
        checks = audit_finding(ctx, finding, ledger)
        rnd = AuditRound(i, checks)
        rounds.append(rnd)
        if not rnd.failed:
            return AuditResult("verified" if i == 1 else "verified_after_corrections", rounds)
        if i == max_rounds:
            break
        rnd.corrections = correct(ctx, finding, checks, template_narrative)
        if not rnd.corrections:
            break  # nothing left that can be corrected automatically
    return AuditResult("unverified", rounds)

