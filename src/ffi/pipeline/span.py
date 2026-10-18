"""Multi-year review: analyse each of the last N fiscal years, merge the findings, investigate the top anomalies once.

    anomaly-report AAPL --span 10

The company's filings and facts load once; each year then gets its own peer set, filing text, knowledge graph and
detector run (the figures as first reported, as in a single-year run). Findings are merged across years
(scoring/span.py); only the top merged anomalies are investigated by the AI, each in the context of the most recent
year it was flagged in, with every year's results and the earlier years' annual reports in its evidence.
"""

from __future__ import annotations

import contextlib
import dataclasses
import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ffi.context import RunContext
from ffi.pipeline.run import (
    RunOptions,
    Step,
    YearAnalysis,
    _remove_superseded_reports,
    analyse_year,
    pick_focus,
    prepare,
)
from ffi.report import snapshot
from ffi.scoring import span as span_scoring
from ffi.scoring.engine import REPORTED

log = logging.getLogger(__name__)


@dataclass
class SpanOutput:
    years: list[YearAnalysis]
    span: span_scoring.SpanResult
    contexts: dict[str, RunContext]  # merged anomaly id -> the context it was investigated and audited in
    run_dir: Path
    manifest: dict[str, Any]
    timings: dict[str, float] = field(default_factory=dict)
    agent_cases: list[Any] = field(default_factory=list)
    files: dict[str, Path] = field(default_factory=dict)
    kind: str = "span"

    @property
    def ctx(self) -> RunContext:
        """The latest year's context (company, profile, currency)."""
        return self.years[-1].ctx


def case_context(years: list[YearAnalysis], rep: int) -> RunContext:
    """The representative year's context, with the annual reports of every earlier reviewed year added (so quotes from
    those years can be checked) and tools that return figures for every year reviewed."""
    ctx = years[rep].ctx
    docs = {d.accession: d for y in years[: rep + 1] for d in y.ctx.docs}
    releases = {d.accession: d for y in years[: rep + 1] for d in y.ctx.releases}
    return dataclasses.replace(ctx, docs=sorted(docs.values(), key=lambda d: d.period_end),
                               releases=sorted(releases.values(), key=lambda d: d.period_end), history=max(ctx.history, rep + 1))


def run_span(opts: RunOptions, span: int, choose=None, progress: Callable[[str], None] | None = None,
             agent_ui: Any = None) -> SpanOutput:
    import pandas as pd

    from ffi.audit.deterministic import run_audit
    from ffi.report.explain import _write_one, fallback
    from ffi.scoring.engine import audit_reported, template_narrative

    timings: dict[str, float] = {}
    step = Step(timings, progress)
    prep = prepare(opts, choose, step)
    company, store = prep.company, prep.store
    last = pick_focus(store, opts.period)
    focuses = [p for p in store.fiscal_year_ends() if p <= last][-span:]
    if len(focuses) < 2:
        raise ValueError(f"a multi-year review needs at least two fiscal years of annual data; found {len(focuses)}")
    first_label, last_label = store.fiscal_label(focuses[0]), store.fiscal_label(focuses[-1])
    run_id = opts.run_id or f"{company.ticker or company.cik}_{first_label}-{last_label}_{uuid.uuid4().hex[:6]}"
    run_dir = Path(opts.out_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    years: list[YearAnalysis] = []
    kg_counts: dict[str, Any] = {}
    for i, p in enumerate(focuses):
        label = store.fiscal_label(p)
        ystep = Step(timings, progress, prefix=f"{label} ({i + 1}/{len(focuses)}): ")
        y = analyse_year(prep, opts, p, run_dir, ystep, kg_file=f"kg_{label}.duckdb")
        kg_counts[label] = y.kg_counts
        _close_kg(y.ctx, run_dir / f"kg_{label}.duckdb", keep=opts.keep_artifacts)
        y.ctx.xsec = pd.DataFrame()  # the industry cross-section is only read by that year's detectors (tens of MB per year)
        years.append(y)

    with step("Merging findings across years"):
        result = span_scoring.merge(years)
    by_year: dict[int, RunContext] = {}  # one investigation context per representative year, shared by its anomalies

    def context_of(f) -> RunContext:
        rep = result.rep_year[f.id]
        if rep not in by_year:
            by_year[rep] = case_context(years, rep)
        return by_year[rep]

    reported = result.findings[:REPORTED]
    agent_cases: list[Any] = []
    if prep.llm is not None and reported:
        with step("Investigating top anomalies (agent)"):
            from ffi.agent.runner import investigate_with_backfill

            # anomalies shown in the report are investigated in order; each one set aside as a test error is replaced by the next
            agent_cases = investigate_with_backfill([(context_of(f), f) for f in result.findings[: REPORTED + opts.max_cases]],
                                                    opts.max_cases, run_dir, run_id, agent_ui)
        span_scoring.apply_set_aside(result, years)

    with step("Independent audit of findings"):
        def audit_one(f) -> None:
            if not f.narrative:
                f.narrative = template_narrative(f)
            f.audit = run_audit(context_of(f), f, template_narrative, ledger=f.ledger)

        audit_reported(result.findings, audit_one)
        span_scoring.apply_audit(result)
    with step("Explaining findings in plain language"):
        from ffi.concurrency import map_concurrent

        shown = result.findings[:REPORTED]
        if prep.llm is None:
            for f in shown:
                f.client = fallback(f)
        else:
            written = map_concurrent(lambda f: _write_one(context_of(f), f), shown, workers=4)
            for f, r in zip(shown, written, strict=True):
                f.client = r if isinstance(r, dict) else fallback(f)

    manifest = build_span_manifest(years, result, run_id, opts, span, timings, kg_counts, prep)
    out = SpanOutput(years, result, {f.id: context_of(f) for f in shown}, run_dir, manifest, timings, agent_cases)
    with step("Rendering report"):
        from ffi.report.render import render, to_json

        out.files = render(out, opts.formats)
    (run_dir / "manifest.json").write_text(to_json(manifest), encoding="utf-8")
    try:
        snapshot.save(out)
    except Exception as e:  # noqa: BLE001 - the report is already written; only re-rendering is lost
        log.warning("could not save the run for re-rendering: %s", e)
    if not opts.keep_artifacts:
        import gc

        gc.collect()  # release graph-database handles still held by finished years before deleting their files
        for path in run_dir.glob("kg_*.duckdb*"):
            with contextlib.suppress(OSError):
                path.unlink()
        if not any(isinstance(c, dict) and c.get("status") in ("paused", "error", "running") for c in agent_cases):
            with contextlib.suppress(OSError):
                (run_dir / "checkpoints.sqlite").unlink(missing_ok=True)
    _remove_superseded_reports(run_dir, company.cik)
    return out


def _close_kg(ctx: RunContext, path: Path, keep: bool) -> None:
    """Each year's graph database is only needed by that year's detectors."""
    with contextlib.suppress(Exception):
        ctx.kg.con.close()
    if not keep:
        for f in (path, path.with_name(path.name + ".wal")):
            try:
                f.unlink(missing_ok=True)
            except OSError as e:
                log.debug("could not remove %s: %s", f, e)


def build_span_manifest(years: list[YearAnalysis], result, run_id: str, opts: RunOptions, span: int, timings, kg_counts, prep) -> dict:
    from ffi import __version__

    ctx = years[-1].ctx
    llm = prep.llm
    return {
        "run_id": run_id, "kind": "span", "package_version": __version__,
        "company": {"cik": ctx.company.cik, "ticker": ctx.company.ticker, "name": ctx.company.name, "sic": ctx.company.sic,
                    "resolution": ctx.company.match_note, "warnings": ctx.company.warnings,
                    "annual_form": ctx.annual_form, "currency": ctx.currency},
        "profile": {"key": ctx.profile.key, "label": ctx.profile.label} if ctx.profile else None,
        "periods": result.periods, "span_requested": span, "as_of": str(ctx.as_of),
        "options": {k: (str(v) if isinstance(v, Path) or hasattr(v, "isoformat") else v) for k, v in opts.__dict__.items()},
        "mode": ctx.mode,
        "yearly": result.yearly,
        "detectors": {y.ctx.focus_label: {r.id: {"status": r.status, "flagged": r.flagged} for r in y.results} for y in years},
        "peers": {y.ctx.focus_label: ([p.cik for p in y.ctx.peers.loaded] if y.ctx.peers else []) for y in years},
        "kg": kg_counts,
        "sources": {"companyfacts_sha256": prep.cf.sha256,
                    "documents": sorted({d.url for y in years for d in y.ctx.docs + y.ctx.releases})},
        "llm": ({"backend": llm.backend, "reasoning_model": llm.reasoning_model, "extraction_model": llm.extraction_model,
                 "calls": llm.spend.calls, "usd": round(llm.spend.usd, 4), "by_purpose": llm.spend.by_purpose} if llm else None),
        "risk_index": result.risk_index, "band": result.band,
        "audit": {f.id: f.audit.label for f in result.findings if f.audit is not None},
        "timings_s": timings,
    }
