"""Agent state graph with a scripted fake LLM: checkpoints, interrupts, budgets, retries, resume."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from ffi.agent import budget as agent_budget
from ffi.agent.runner import investigate
from ffi.agent.schemas import (
    AlternativeAssessment,
    AlternativeAssessments,
    EvidencePlan,
    EvidenceTask,
    FindingNarrative,
    Hypotheses,
    Hypothesis,
)
from ffi.agent.ui import TerminalUI
from ffi.audit.llm_auditor import AuditVerdicts, ClaimVerdict
from ffi.llm import BudgetExceeded, LLMError, Spend
from ffi.pipeline.detect import run_detectors
from ffi.scoring.engine import score
from tests.conftest import make_ctx, series


@pytest.fixture(autouse=True)
def fast_retries(monkeypatch):
    from langgraph.types import RetryPolicy

    fast = RetryPolicy(max_attempts=3, initial_interval=0.01, max_interval=0.02, jitter=False, retry_on=agent_budget.transient)
    monkeypatch.setattr("ffi.agent.graph.LLM_RETRY", fast)
    monkeypatch.setattr("ffi.agent.graph.ALT_RETRY", fast)


@dataclass
class FakeLLM:
    narrative: str = "Receivables grew far faster than revenue [L0.1], which warrants inquiry into collection terms."
    supported_alternative: bool = False
    auditor_rejects: bool = False
    fail: dict[str, Exception] = field(default_factory=dict)  # purpose -> exception to raise
    calls: list[str] = field(default_factory=list)
    reasoning_model: str = "fake-reasoning"
    extraction_model: str = "fake-extraction"
    budget_usd: float = 6.0
    spend: Spend = field(default_factory=Spend)

    def remaining(self) -> float:
        return self.budget_usd - self.spend.usd

    def structured(self, schema, system, prompt, *, purpose, model=None, effort="medium", max_tokens=1000, max_usd=None, timeout_s=None):
        self.calls.append(purpose)
        if purpose in self.fail:
            raise self.fail[purpose]
        self.spend.usd += 0.01
        if schema is Hypotheses:
            return Hypotheses(hypotheses=[Hypothesis(statement="Revenue recognised before collection", confidence=0.6,
                                                     innocent_alternatives=["New customer payment terms"])])
        if schema is EvidencePlan:
            return EvidencePlan(tasks=[EvidenceTask(hypothesis_index=0, tool="get_fact", purpose="AR trend", item="AR"),
                                       EvidenceTask(hypothesis_index=0, tool="ratio", purpose="AR to revenue", numerator="AR", denominator="REVENUE"),
                                       EvidenceTask(hypothesis_index=0, tool="get_events", purpose="events", event_types=["ACQUISITION"])])
        if schema is AlternativeAssessments:
            return AlternativeAssessments(assessments=[AlternativeAssessment(
                alternative="New customer payment terms", supported=self.supported_alternative,
                strength="strong" if self.supported_alternative else "weak", ledger_ids=["L0.1"], reasoning="test")])
        if schema is FindingNarrative:
            return FindingNarrative(summary=self.narrative, explanation="The ledger shows the trend [L0.2].",
                                    alternatives=[], questions=["What changed in payment terms?"], cited_ledger_ids=["L0.1", "L0.2"])
        if schema is AuditVerdicts:
            assert "Revenue recognised before collection" not in prompt  # independence: no drafter hypotheses
            v = ClaimVerdict(claim="Receivables grew far faster than revenue", verdict="overstated" if self.auditor_rejects else "supported",
                             evidence_ids=["L0.1"], correction="Say receivables grew faster than revenue" if self.auditor_rejects else "")
            return AuditVerdicts(verdicts=[v], accusatory_language=[])
        raise AssertionError(f"unexpected schema {schema}")


@dataclass
class Opts:
    max_cases: int = 1


def _setup(tmp_path: Path):
    ctx = make_ctx(series(overrides={"AR": {2024: 450.0}, "CFO": {2022: 40.0, 2023: 30.0, 2024: 10.0}}))
    sc = score(ctx, run_detectors(ctx, only={"C.AR.01", "C.CF.01", "B.01", "B.03", "B.05"}))
    assert sc.findings
    return ctx, sc


def test_happy_path_publishes_with_checkpoints(tmp_path):
    ctx, sc = _setup(tmp_path)
    ctx.llm = FakeLLM()
    cases = investigate(ctx, sc, tmp_path, "RUN1", Opts(), ui=TerminalUI(mode="auto"))
    c = cases[0]
    assert c["status"] == "published"
    assert (tmp_path / "checkpoints.sqlite").exists()
    kinds = [d["interrupt"] for d in c["decisions"]]
    assert "review_publish" in kinds and "review_plan" in kinds  # high-severity finding gets a plan review
    assert all(d.get("auto") for d in c["decisions"])
    f = next(f for f in sc.findings if f.id == c["finding_id"])
    assert f.narrative_source == "agent" and "[L0.1]" in f.narrative
    assert c["ledger"] == 3


def test_invented_numbers_fail_grounding_then_degrade(tmp_path):
    ctx, sc = _setup(tmp_path)
    ctx.llm = FakeLLM(narrative="Receivables are overstated by $987.6B [L0.1].")
    c = investigate(ctx, sc, tmp_path, "RUN2", Opts(), ui=TerminalUI(mode="auto"))[0]
    assert c["status"] == "degraded"
    assert ctx.llm.calls.count("agent.draft") == 3  # first draft + two revisions
    assert c["verification"]["unsupported_numbers"]


def test_budget_exhaustion_routes_to_review_budget_then_degrade(tmp_path):
    ctx, sc = _setup(tmp_path)
    ctx.llm = FakeLLM(fail={"agent.draft": BudgetExceeded("draft over budget")})
    c = investigate(ctx, sc, tmp_path, "RUN3", Opts(), ui=TerminalUI(mode="auto"))[0]
    assert c["status"] == "degraded"
    assert any(d["interrupt"] == "review_budget" and d["choice"] == "degrade" for d in c["decisions"])


def test_retry_exhaustion_degrades_after_three_attempts(tmp_path):
    ctx, sc = _setup(tmp_path)
    ctx.llm = FakeLLM(fail={"agent.hypothesize": LLMError("agent.hypothesize: API error 529")})
    c = investigate(ctx, sc, tmp_path, "RUN4", Opts(), ui=TerminalUI(mode="auto"))[0]
    assert ctx.llm.calls.count("agent.hypothesize") == 3
    assert c["status"] == "degraded"
    assert any(d["interrupt"] == "retry_exhausted" for d in c["decisions"])


def test_supported_alternative_closes_finding(tmp_path):
    ctx, sc = _setup(tmp_path)
    ctx.llm = FakeLLM(supported_alternative=True)
    c = investigate(ctx, sc, tmp_path, "RUN5", Opts(), ui=TerminalUI(mode="auto"))[0]
    assert c["status"] == "explained"


def test_pause_then_resume_from_checkpoint(tmp_path):
    ctx, sc = _setup(tmp_path)
    llm = FakeLLM()
    ctx.llm = llm
    paused = investigate(ctx, sc, tmp_path, "RUN6", Opts(), ui=TerminalUI(mode="auto", pause_on=("review_publish",)))[0]
    assert paused["status"] == "paused"
    calls_before = list(llm.calls)
    ctx2, sc2 = _setup(tmp_path)
    ctx2.llm = llm
    resumed = investigate(ctx2, sc2, tmp_path, "RUN6", Opts(), ui=TerminalUI(mode="auto"))[0]
    assert resumed["status"] == "published" and resumed.get("resumed")
    assert llm.calls == calls_before  # nothing re-ran: the graph continued at the pending interrupt


def test_independent_auditor_rejection_loops_three_rounds_then_degrades(tmp_path):
    ctx, sc = _setup(tmp_path)
    ctx.llm = FakeLLM(auditor_rejects=True)
    c = investigate(ctx, sc, tmp_path, "RUN7", Opts(), ui=TerminalUI(mode="auto"))[0]
    assert ctx.llm.calls.count("agent.audit") == 3 and ctx.llm.calls.count("agent.draft") == 3
    assert c["status"] == "degraded"
    assert [a["round"] for a in c["audits"]] == [1, 2, 3]


def test_independent_auditor_pass_is_recorded(tmp_path):
    ctx, sc = _setup(tmp_path)
    ctx.llm = FakeLLM()
    c = investigate(ctx, sc, tmp_path, "RUN8", Opts(), ui=TerminalUI(mode="auto"))[0]
    assert c["status"] == "published" and c["audits"][0]["failing"] == 0
    assert c["verification"]["audit"] == "passed"


def test_unattended_cases_run_in_parallel(tmp_path):
    ctx, sc = _setup(tmp_path)
    ctx.llm = FakeLLM()
    patterns = [f for f in sc.findings if f.kind == "pattern"]
    cases = investigate(ctx, sc, tmp_path, "RUN9", Opts(max_cases=len(patterns)), ui=TerminalUI(mode="auto"))
    assert len(cases) == len(patterns) >= 2
    assert {c["status"] for c in cases} == {"published"}
