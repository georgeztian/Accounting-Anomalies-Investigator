"""Scoring, scheme patterns and false-positive control (BLUEPRINT §10.2-§10.4)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ffi.context import RunContext
from ffi.detectors.base import EvidenceRef, ImpactEstimate, Signal
from ffi.detectors.stats import fmt_money, fmt_pct
from ffi.materiality.impact import Materiality, assess
from ffi.models_store import load_calibration
from ffi.pipeline.detect import DetectorResult

FAMILY_WEIGHT = {"A": 0.55, "B": 0.35, "C": 0.45, "D": 0.30, "E": 0.30, "F": 0.35}
PRESSURE_WEIGHT = 0.15
PATTERN_WEIGHT = 0.8
INDEX_K = 2.5


@dataclass
class SchemeTemplate:
    id: str
    title: str
    required_any: list[list[str]]  # each inner list: at least one of these must match
    optional: list[str]
    explained_by: list[str]
    questions: list[str]


TEMPLATES: list[SchemeTemplate] = [
    SchemeTemplate("premature_revenue", "Revenue may be recognised ahead of cash and delivery",
                   [["C.AR.01", "C.REV.01", "C.CF.01"]], ["C.REV.02", "C.REV.03", "C.REV.04", "C.REV.10", "C.REV.11", "E.01:revenue", "B.01:DSRI", "F.01", "D.04:AR", "C.AR.02"],
                   ["ACQUISITION", "ASC606_ADOPTION"],
                   ["What drove the change in payment terms or collection patterns?",
                    "How much revenue was recognised in the last two weeks of the quarter and year?",
                    "Were there bill-and-hold, consignment or extended-terms arrangements?"]),
    SchemeTemplate("accrual_heavy_earnings", "Earnings are running well ahead of operating cash flow",
                   [["B.03", "B.05", "C.CF.01", "B.02"]], ["B.04", "B.06", "B.01", "D.02", "D.03", "C.CF.06", "B.09:real_em"], ["ACQUISITION"],
                   ["Which balance-sheet accounts absorbed the gap between earnings and operating cash flow?",
                    "When are the accrued amounts expected to convert to cash, and what is the evidence?"]),
    SchemeTemplate("expense_capitalization", "Operating costs may be capitalised or deferred",
                   [["C.EXP.01", "C.EXP.02", "A.08", "B.01:AQI"]], ["C.CF.04", "E.01:capitalization", "C.EXP.05", "B.03"],
                   ["ACQUISITION"],
                   ["Which cost categories were capitalised this year that were expensed before?",
                    "How are capitalisation criteria (e.g. technological feasibility) applied and reviewed?"]),
    SchemeTemplate("asset_life_stretching", "Depreciation may be slowed by longer useful lives",
                   [["C.EXP.03", "C.EXP.04"]], ["B.01:DEPI", "E.02", "E.01:lives"], [],
                   ["Were useful lives or salvage values changed, and what engineering evidence supports it?",
                    "What was the depreciation effect of the change in estimate?"]),
    SchemeTemplate("cookie_jar", "Reserves may be released to support earnings",
                   [["C.RES.01", "C.RES.02", "C.RES.03", "C.AR.02", "C.AR.03", "C.INV.04"]], ["B.10", "B.09:threshold", "C.RES.05", "C.RES.04"],
                   [],
                   ["What changed in the assumptions behind each reserve?",
                    "How do reserve releases compare with actual losses and claims experience?"]),
    SchemeTemplate("cfo_inflation", "Operating cash flow may be flattered by working-capital or classification choices",
                   [["C.CF.02", "C.CF.03", "C.CF.04", "C.CF.05", "C.AR.04"]], ["C.BS.01", "C.CF.06", "C.CF.01"], ["ACQUISITION"],
                   ["Were payables extended or supplier-finance programmes used at period end?",
                    "Were receivables sold or factored, and how are collections classified?"]),
    SchemeTemplate("inventory_overstatement", "Inventory may be overstated or under-reserved",
                   [["C.INV.01", "C.INV.02", "C.INV.03"]], ["C.INV.04", "D.04:INVENTORY", "B.09:real_em"], ["ACQUISITION"],
                   ["What is the ageing of finished goods, and how is excess and obsolete inventory assessed?",
                    "Was production increased to absorb fixed overhead?"]),
    SchemeTemplate("hidden_liabilities", "Obligations may sit outside the balance sheet",
                   [["C.BS.01", "C.BS.02", "C.CF.07", "C.BS.04"]], ["C.CF.03", "E.09", "F.07", "C.RES.06"], [],
                   ["What guarantees, purchase commitments and VIE exposures exist beyond reported debt?",
                    "How does average debt during the quarter compare with period-end debt?"]),
    SchemeTemplate("round_tripping", "Revenue may involve related or circular transactions",
                   [["F.02", "C.REV.08", "F.01"]], ["E.09", "F.07"], [],
                   ["Who are the related counterparties, and do they also supply or finance the company?"]),
    SchemeTemplate("acquisition_masking", "Acquisitions may obscure organic performance",
                   [["C.MA.01", "C.MA.03", "C.CF.05", "C.MA.05"]], ["C.NG.01", "C.MA.02", "C.MA.04"], [],
                   ["What is organic growth excluding acquisitions, and how is it computed?",
                    "How were purchase prices allocated between goodwill and amortising intangibles?"]),
    SchemeTemplate("reporting_integrity", "Reported figures are internally inconsistent or were revised",
                   [["A.01", "A.02", "A.03", "A.14", "A.15", "E.12", "C.GOV.02"]], ["A.04", "A.05", "A.09", "A.10", "A.12", "C.GOV.05"], [],
                   ["Why were prior-period figures revised, and was a restatement evaluation performed?",
                    "Which controls failed to detect the inconsistency?"]),
    SchemeTemplate("misleading_metrics", "Headline metrics may present a better picture than GAAP results",
                   [["C.NG.01", "C.NG.02", "C.NG.03", "E.04"]], ["C.EXP.07", "C.EXP.09", "C.REV.07"], [],
                   ["Why are recurring charges excluded from adjusted metrics?",
                    "How did the definitions of key metrics change?"]),
]

# Events or facts that legitimately explain a signal: (condition key, detectors discounted, factor)
EXPLAIN_RULES = [
    ("large_acquisition", {"C.AR.01", "C.INV.01", "C.EXP.02", "C.REV.05", "D.04", "B.01", "A.08", "C.REV.01", "C.CF.02", "D.02", "D.03"}, 0.5),
    ("standard_adoption", {"C.REV.01", "C.REV.02", "C.REV.03", "E.01", "A.14"}, 0.6),
    ("stock_split", {"A.12"}, 0.2),
]


def _explain_conditions(ctx: RunContext) -> dict[str, str]:
    out = {}
    acq, ta = ctx.a("ACQUISITIONS"), ctx.a("TOTAL_ASSETS", 1)
    if np.isfinite(acq) and np.isfinite(ta) and ta > 0 and acq / ta > 0.10:
        out["large_acquisition"] = f"acquisitions of {acq / ta:.0%} of prior-year assets in the focus year"
    for s in ctx.signals.get("E.01", []):
        if "adoption" in s.tags:
            out["standard_adoption"] = "adoption of a new accounting standard"
    sh1, sh0 = ctx.a("SHARES_DILUTED"), ctx.a("SHARES_DILUTED", 1)
    if np.isfinite(sh1) and np.isfinite(sh0) and sh0 > 0 and sh1 / sh0 > 1.8:
        out["stock_split"] = "share count consistent with a stock split"
    return out


def benjamini_hochberg(pvals: list[float]) -> list[float]:
    n = len(pvals)
    if n == 0:
        return []
    order = np.argsort(pvals)
    q = np.empty(n)
    prev = 1.0
    for rank, idx in reversed(list(enumerate(order, start=1))):
        prev = min(prev, pvals[idx] * n / rank)
        q[idx] = prev
    return q.tolist()


@dataclass
class Finding:
    id: str
    title: str
    kind: str  # pattern | signal
    score: float  # 0..1
    signals: list[Signal]
    impact: ImpactEstimate | None
    materiality: Materiality | None
    evidence: list[EvidenceRef]
    questions: list[str]
    alternatives: list[str]
    narrative: str = ""
    narrative_source: str = "template"  # template | agent | agent-degraded
    priority: float = 0.0
    ledger: list[dict] = field(default_factory=list)  # agent evidence ledger (tool results)
    audit: Any = None  # AuditResult from the independent auditor

    @property
    def detectors(self) -> list[str]:
        return sorted({s.detector_id for s in self.signals})


@dataclass
class ScoreResult:
    risk_index: float
    band: str
    probability: float | None
    findings: list[Finding]
    area_scores: dict[str, float]
    family_scores: dict[str, float]
    suppressed: list[Signal]
    pressure: list[Signal]
    explain_conditions: dict[str, str]
    contributions: list[tuple[str, float]] = field(default_factory=list)


def template_narrative(f) -> str:
    """Deterministic narrative assembled from detector outputs (used without the LLM and as the audit fallback)."""
    lead = f.signals[0]
    others = [s for s in f.signals[1:4]]
    parts = [lead.rationale]
    if others:
        parts.append("Corroborating signals: " + " ".join(f"{s.detector_id} {s.name}: {s.rationale}" for s in others))
    if f.materiality:
        m = f.materiality
        parts.append(f"Estimated pre-tax effect {fmt_money(m.impact_low)}-{fmt_money(m.impact_high)} (mid {fmt_money(m.impact_mid)}), "
                     f"{fmt_pct(m.pct_pretax) if m.pct_pretax is not None else 'n/a'} of pre-tax income.")
    return " ".join(parts)


UNVERIFIED_FACTOR = 0.5


def apply_audit(sc: ScoreResult) -> None:
    """After the independent audit: unverified findings lose half their weight; the index is recomputed."""
    contrib = dict(sc.contributions)
    for f in sc.findings:
        if f.audit is None:
            continue
        if f.audit.status == "unverified":
            f.priority *= UNVERIFIED_FACTOR
            key = f"pattern:{f.id}"
            if key in contrib:
                contrib[key] *= UNVERIFIED_FACTOR
        elif f.kind == "signal" and f"pattern:{f.id}" in contrib:  # downgraded by the auditor
            contrib[f"pattern:{f.id}"] *= 0.6
    total = sum(contrib.values())
    sc.risk_index = round(100 * (1 - math.exp(-total / INDEX_K)), 1)
    sc.band = band_of(sc.risk_index)
    sc.contributions = sorted(contrib.items(), key=lambda kv: -kv[1])
    sc.findings.sort(key=lambda x: -x.priority)


def band_of(index: float) -> str:
    return "Low" if index < 25 else "Elevated" if index < 50 else "High" if index < 75 else "Severe"


def _matches(s: Signal, token: str) -> bool:
    if ":" in token:
        did, tag = token.split(":", 1)
        return s.detector_id == did and any(t == tag or t.endswith(":" + tag) or t == f"{did}:{tag}" or t == f"driver:{tag}" or t == f"target:{tag}" or t == f"estimate:{tag}" for t in s.tags)
    return s.detector_id == token


def score(ctx: RunContext, results: list[DetectorResult]) -> ScoreResult:
    calib = load_calibration()
    signals = [s for r in results for s in r.signals]
    # 1. calibration (when a trained map exists) -------------------------------------------
    for s in signals:
        m = calib.get(s.detector_id)
        if m:
            xs, ys = zip(*m, strict=True)
            s.severity = float(np.interp(s.severity, xs, ys))
    # 2. explained-away discounts -------------------------------------------------------------
    conds = _explain_conditions(ctx)
    for key, ids, factor in EXPLAIN_RULES:
        if key not in conds:
            continue
        for s in signals:
            if s.detector_id in ids and s.flagged:
                s.severity *= factor
                s.rationale += f" [Discounted: {conds[key]}.]"
                if s.severity < 0.5:
                    s.flagged = False
                    s.suppressed_by = conds[key]
    # 3. Benjamini-Hochberg across signals that carry p-values ----------------------------------
    with_p = [s for s in signals if s.p_value is not None and np.isfinite(s.p_value)]
    qs = benjamini_hochberg([s.p_value for s in with_p])  # type: ignore[misc]
    fdr_q = ctx.tuning.get("scoring", {}).get("fdr_q", 0.10)
    for s, q in zip(with_p, qs, strict=True):
        if s.flagged and q > fdr_q:
            s.flagged = False
            s.suppressed_by = f"false-discovery control (q = {q:.2f} > {fdr_q})"
    flagged = [s for s in signals if s.flagged]
    suppressed = [s for s in signals if s.suppressed_by]
    pressure = [s for s in flagged if "pressure" in s.tags]
    core = [s for s in flagged if "pressure" not in s.tags]

    # 4. scheme patterns ------------------------------------------------------------------------
    findings: list[Finding] = []
    used: set[int] = set()
    for t in TEMPLATES:
        req = []
        for group in t.required_any:
            req += [s for s in core if any(_matches(s, tok) for tok in group)]
        if not req:
            continue
        opt = [s for s in core if any(_matches(s, tok) for tok in t.optional)]
        members = list({id(s): s for s in req + opt}.values())
        families = {s.family for s in members}
        thin = len(members) < 2 or (len(families) < 2 and len(members) < 3)
        if thin and not (t.id == "reporting_integrity" and any(s.severity >= 0.8 for s in req)):
            continue
        sc = 1 - math.prod(1 - s.severity * 0.85 for s in members)
        imp = max((s.impact for s in members if s.impact), key=lambda i: i.mid, default=None)
        alts = [f"Checked: {v}" for v in conds.values()] + [f"{s.detector_id} discounted: {s.suppressed_by}" for s in suppressed
                                                             if any(_matches(s, tok) for g in t.required_any for tok in g)]
        if not alts:
            alts = ["No acquisition, accounting-standard adoption or stock split in the focus year explains these signals."]
        ev, seen_ev = [], set()
        for s in sorted(members, key=lambda x: -x.severity):
            for e in s.evidence[:3]:
                if e.node_id not in seen_ev:
                    seen_ev.add(e.node_id)
                    ev.append(e)
        findings.append(Finding(t.id, t.title, "pattern", sc, sorted(members, key=lambda x: -x.severity), imp,
                                assess(ctx, imp), ev[:10], t.questions, alts))
        used.update(id(s) for s in members)
    # 5. strong standalone signals (leftover family-B scores fold into one finding) -----------------
    loose_b = [s for s in core if id(s) not in used and s.family == "B" and s.severity >= 0.6]
    if len(loose_b) >= 2:
        sc_b = 1 - math.prod(1 - s.severity * 0.6 for s in loose_b)
        findings.append(Finding("forensic_scores", "Forensic scores are elevated", "pattern", sc_b,
                                sorted(loose_b, key=lambda x: -x.severity), None, None,
                                [e for s in loose_b for e in s.evidence[:2]][:8],
                                ["Which inputs drive the scores, and are they explained by business changes?"],
                                [f"Checked: {v}" for v in conds.values()] or ["No explaining event detected."]))
        used.update(id(s) for s in loose_b)
    for s in core:
        if id(s) in used or s.severity < 0.7:
            continue
        findings.append(Finding(f"{s.detector_id}@{s.period}", s.name, "signal", s.severity * 0.85, [s], s.impact,
                                assess(ctx, s.impact), s.evidence[:6],
                                ["What explains this movement, and is it expected to persist?"],
                                [f"Checked: {v}" for v in conds.values()] or ["No explaining event detected."]))
    for f in findings:
        mult = f.materiality.multiple if f.materiality else 0.0
        f.priority = f.score * (0.5 + 0.5 * min(mult, 3.0) / 3.0) if f.materiality else f.score * 0.5
    findings.sort(key=lambda f: -f.priority)

    # 6. risk index --------------------------------------------------------------------------------
    contrib: dict[str, float] = {}
    in_pattern = {id(s) for f in findings if f.kind == "pattern" for s in f.signals}
    for s in core:
        if id(s) in in_pattern or "derived" in s.tags:
            continue  # counted once, through its pattern (or derived from other signals)
        w = FAMILY_WEIGHT.get(s.family, 0.3) * s.severity
        contrib[s.detector_id] = max(contrib.get(s.detector_id, 0.0), w)
    for s in pressure:
        contrib[s.detector_id] = max(contrib.get(s.detector_id, 0.0), PRESSURE_WEIGHT * s.severity)
    for f in findings:
        if f.kind == "pattern":
            contrib[f"pattern:{f.id}"] = PATTERN_WEIGHT * f.score
    total = sum(contrib.values())
    index = 100 * (1 - math.exp(-total / INDEX_K))
    area: dict[str, float] = {}
    fam: dict[str, float] = {}
    for s in signals:
        area[str(s.area)] = max(area.get(str(s.area), 0.0), s.severity if s.flagged else 0.0)
        fam[s.family] = max(fam.get(s.family, 0.0), s.severity if s.flagged else 0.0)
    prob = None  # only a trained meta-model yields a calibrated probability (see README)
    return ScoreResult(round(index, 1), band_of(index), prob, findings, area, fam, suppressed, pressure, conds,
                       sorted(contrib.items(), key=lambda kv: -kv[1]))
