"""Run the investigation graph per finding with SQLite checkpoints, interrupts, resume and pause."""

from __future__ import annotations

import contextvars
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from ffi.agent.graph import build_graph
from ffi.agent.ui import PauseRun, TerminalUI, stdin_interactive
from ffi.detectors.base import EvidenceRef
from ffi.scoring.engine import EXPLAINED, EXPLAINED_FACTOR, apply_explained, apply_set_aside

log = logging.getLogger(__name__)


def _drive(app, cfg: dict, inp: Any, ui: TerminalUI) -> dict:
    result = app.invoke(inp, cfg)
    while result.get("__interrupt__"):
        payload = result["__interrupt__"][0].value
        decision = ui.decide(payload)
        result = app.invoke(Command(resume=decision), cfg)
    return result


def _apply(f, result: dict, case: dict) -> None:
    status = result.get("status", "unknown")
    draft = result.get("draft")
    case.update(status=status, decisions=result.get("decisions", []), audits=result.get("audits", []),
                usd=round(sum(s.get("usd", 0) for s in result.get("spend", [])), 4),
                ledger=len(result.get("ledger", [])), verification=result.get("verification"))
    f.ledger = result.get("ledger", [])
    verification = result.get("verification") or {}
    if (status == "published" and draft and draft.get("test_assessment") == "likely_test_error"
            and verification.get("test_error_check") == "supported" and draft.get("test_assessment_reason")):
        # the investigation concluded the test, not the company, produced this result, and the auditor confirmed it from the evidence
        f.set_aside = draft["test_assessment_reason"]
        f.narrative = draft["summary"] + "\n\n" + draft["explanation"]
        f.narrative_source = "agent: set aside as probable test error"
        case["set_aside"] = True
        return
    if status in ("published", "degraded") and draft:
        f.narrative = draft["summary"] + "\n\n" + draft["explanation"]
        f.narrative_source = "agent" if status == "published" else "agent-degraded"
        if draft.get("alternatives"):
            f.alternatives = [f"{a['alternative']}: {a['assessment']}" for a in draft["alternatives"]]
        if draft.get("questions"):
            f.questions = draft["questions"]
        if status == "published":  # a degraded draft re-uses the code-written alternatives and questions
            f.agent_text = list(f.alternatives) + list(f.questions)
        cited = set(draft.get("cited_ledger_ids", []))
        for entry in result.get("ledger", []):
            if entry["id"] in cited:
                for ev in entry.get("evidence", [])[:2]:
                    f.evidence.append(EvidenceRef(**{**ev, "char_span": tuple(ev["char_span"]) if ev.get("char_span") else None}))
    elif status == "explained":
        written = [f"{a['alternative']}: {a['reasoning']}" for a in result.get("alternatives", []) if a.get("supported")]
        f.alternatives = written + f.alternatives
        f.agent_text = written
        f.narrative_source = EXPLAINED
        f.score *= EXPLAINED_FACTOR  # counts for half: in its strength and the ranking here, in the risk index (apply_explained)
        f.priority *= EXPLAINED_FACTOR
    elif status == "dropped":
        f.narrative_source = "agent narrative dropped by reviewer"


def _run_case(ctx, f, saver, run_id: str, ui: TerminalUI) -> dict[str, Any]:
    thread = f"{run_id}:{f.id}"
    cfg = {"configurable": {"thread_id": thread}, "recursion_limit": ctx.tuning.get("agent", {}).get("recursion_limit", 40)}
    app = build_graph(ctx, f, ctx.llm).compile(checkpointer=saver)
    snap = app.get_state(cfg)
    case: dict[str, Any] = {"finding_id": f.id, "thread_id": thread, "status": "running", "decisions": []}
    try:
        if not snap.values:
            result = _drive(app, cfg, {"case_id": thread}, ui)
        elif snap.next:
            pending = [i for t in snap.tasks for i in (t.interrupts or [])]
            result = _drive(app, cfg, Command(resume=ui.decide(pending[0].value)) if pending else None, ui)
            case["resumed"] = True
        else:
            result = snap.values
            case["resumed"] = True
    except PauseRun as p:
        case.update(status="paused", paused_at=str(p))
        ctx.notes.append(f"Agent paused at {p} for '{f.title}'. Resume with: anomaly-report \"{ctx.company.ticker}\" --resume {run_id}")
        return case
    except Exception as e:  # noqa: BLE001 - one case failing must not lose the report
        log.warning("agent case %s failed: %s", f.id, e)
        case.update(status="error", error=f"{type(e).__name__}: {e}")
        return case
    _apply(f, result, case)
    return case


def investigate(ctx, score, run_dir: Path, run_id: str, opts, ui: TerminalUI | None = None) -> list[dict[str, Any]]:
    """Investigate the top findings. Unattended runs investigate cases in parallel; interactive runs one at a time
    (so prompts never interleave) and stop at the first pause."""
    candidates = [f for f in score.findings if f.kind == "pattern"] or list(score.findings)
    cases = investigate_with_backfill([(ctx, f) for f in candidates], opts.max_cases, run_dir, run_id, ui)
    apply_set_aside(score)
    apply_explained(score)
    score.findings.sort(key=lambda x: -x.priority)
    return cases


def investigate_with_backfill(candidates: list[tuple[Any, Any]], max_cases: int, run_dir: Path, run_id: str,
                              ui: TerminalUI | None = None) -> list[dict[str, Any]]:
    """Investigate the first `max_cases` candidates; each one the investigation sets aside as a test error is replaced by the
    next candidate, so the report still gets `max_cases` investigated anomalies when enough exist."""
    cases: list[dict[str, Any]] = []
    todo, rest = candidates[:max_cases], candidates[max_cases:]
    while todo:
        cases += investigate_cases(todo, run_dir, run_id, ui)
        if any(c.get("status") == "paused" for c in cases):
            break
        replaced = sum(1 for _, f in todo if f.set_aside)
        todo, rest = rest[:replaced], rest[replaced:]
    return cases


def investigate_cases(pairs: list[tuple[Any, Any]], run_dir: Path, run_id: str, ui: TerminalUI | None = None) -> list[dict[str, Any]]:
    """Investigate (context, finding) pairs; in a multi-year review each anomaly is read in the context of its own year."""
    ui = ui or TerminalUI(mode="auto")
    cases: list[dict[str, Any]] = []
    with SqliteSaver.from_conn_string(str(run_dir / "checkpoints.sqlite")) as saver:
        interactive = ui.mode == "interactive" and stdin_interactive()
        if interactive or len(pairs) <= 1:
            for ctx, f in pairs:
                case = _run_case(ctx, f, saver, run_id, ui)
                cases.append(case)
                if case["status"] == "paused":
                    break
        else:
            jobs = [(contextvars.copy_context(), ctx, f) for ctx, f in pairs]  # keep per-run settings (currency) in worker threads
            with ThreadPoolExecutor(max_workers=min(3, len(pairs))) as pool:
                cases = list(pool.map(lambda job: job[0].run(_run_case, job[1], job[2], saver, run_id, ui), jobs))
    return cases
