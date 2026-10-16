"""Run the investigation graph per finding with SQLite checkpoints, interrupts, resume and pause (BLUEPRINT §13.3-13.4)."""

from __future__ import annotations

import contextvars
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from ffi.agent.graph import build_graph
from ffi.agent.ui import PauseRun, TerminalUI
from ffi.detectors.base import EvidenceRef

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
    if status in ("published", "degraded") and draft:
        f.narrative = draft["summary"] + "\n\n" + draft["explanation"]
        f.narrative_source = "agent" if status == "published" else "agent-degraded"
        if draft.get("alternatives"):
            f.alternatives = [f"{a['alternative']}: {a['assessment']}" for a in draft["alternatives"]]
        if draft.get("questions"):
            f.questions = draft["questions"]
        cited = set(draft.get("cited_ledger_ids", []))
        for entry in result.get("ledger", []):
            if entry["id"] in cited:
                for ev in entry.get("evidence", [])[:2]:
                    f.evidence.append(EvidenceRef(**{**ev, "char_span": tuple(ev["char_span"]) if ev.get("char_span") else None}))
    elif status == "explained":
        f.alternatives = [f"{a['alternative']}: {a['reasoning']}" for a in result.get("alternatives", []) if a.get("supported")] + f.alternatives
        f.narrative_source = "agent: closed as explained"
        f.score *= 0.5
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
    ui = ui or TerminalUI(mode="auto")
    targets = [f for f in score.findings if f.kind == "pattern"][: opts.max_cases] or score.findings[: opts.max_cases]
    cases: list[dict[str, Any]] = []
    with SqliteSaver.from_conn_string(str(run_dir / "checkpoints.sqlite")) as saver:
        interactive = ui.mode == "interactive" and sys.stdin.isatty()
        if interactive or len(targets) <= 1:
            for f in targets:
                case = _run_case(ctx, f, saver, run_id, ui)
                cases.append(case)
                if case["status"] == "paused":
                    break
        else:
            jobs = [(contextvars.copy_context(), f) for f in targets]  # keep per-run settings (currency) in worker threads
            with ThreadPoolExecutor(max_workers=min(3, len(targets))) as pool:
                cases = list(pool.map(lambda job: job[0].run(_run_case, ctx, job[1], saver, run_id, ui), jobs))
    score.findings.sort(key=lambda x: -x.priority)
    return cases
