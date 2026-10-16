"""Run all registered detectors in dependency order.

With the LLM enabled, independent detectors run concurrently (LLM calls are I/O-bound); a detector starts
as soon as everything it depends on has finished. Deterministic runs stay sequential (CPU-bound, and
already ~2 s).
"""

from __future__ import annotations

import contextvars
import logging
import time
import traceback
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from graphlib import TopologicalSorter

from ffi.context import RunContext
from ffi.detectors.base import REGISTRY, Signal, SkipDetector, load_all

log = logging.getLogger(__name__)


@dataclass
class DetectorResult:
    id: str
    name: str
    family: str
    status: str  # ok | skipped | n/a (not applicable to the company profile) | error
    reason: str = ""
    seconds: float = 0.0
    signals: list[Signal] = field(default_factory=list)

    @property
    def flagged(self) -> bool:
        return any(s.flagged for s in self.signals)


def _run_one(ctx: RunContext, did: str, results: dict[str, DetectorResult]) -> DetectorResult:
    spec = REGISTRY[did]
    r = DetectorResult(did, spec.name, spec.family, "ok")
    t0 = time.perf_counter()
    na = ctx.profile.skip_reason(did) if ctx.profile is not None else None
    if na:
        r.status, r.reason = "n/a", na
        return r
    try:
        if spec.needs_llm and ctx.llm is None:
            raise SkipDetector("needs the LLM (deterministic mode, or no Claude Code login or API key)")
        if spec.needs_text and ctx.doc is None:
            raise SkipDetector("10-K text not available")
        bad_dep = [d for d in spec.depends_on if d in results and results[d].status == "error"]
        if bad_dep:
            raise SkipDetector(f"dependency failed: {', '.join(bad_dep)}")
        r.signals = spec.fn(ctx) or []
    except SkipDetector as e:
        r.status, r.reason = "skipped", str(e)
    except Exception as e:  # noqa: BLE001 - a detector bug must not abort the run; it is reported
        r.status, r.reason = "error", f"{type(e).__name__}: {e}"
        log.debug("detector %s failed\n%s", did, traceback.format_exc())
    r.seconds = time.perf_counter() - t0
    return r


def _preload_heavy_modules() -> None:
    """Import scikit-learn once on this thread: two detector threads importing it at the same time can see a
    partially initialised module ('cannot import name ... from partially initialized module')."""
    try:
        import sklearn.covariance  # noqa: F401
        import sklearn.ensemble  # noqa: F401
        import sklearn.feature_extraction.text  # noqa: F401
        import sklearn.metrics.pairwise  # noqa: F401
    except ImportError:
        pass  # the detectors that need it report the error themselves


def run_detectors(ctx: RunContext, only: set[str] | None = None,
                  after: dict[str, Callable[[RunContext], None]] | None = None,
                  progress: Callable[[str], None] | None = None,
                  workers: int | None = None) -> list[DetectorResult]:
    specs = load_all()
    ids = [i for i in specs if only is None or i in only]
    ts = TopologicalSorter({i: {d for d in specs[i].depends_on if d in ids} for i in ids})
    ts.prepare()
    results: dict[str, DetectorResult] = {}
    workers = workers or 4

    def finish(r: DetectorResult) -> None:
        results[r.id] = r
        ctx.signals[r.id] = r.signals
        if after and r.id in after and r.status == "ok":
            after[r.id](ctx)  # e.g. E.09 relationships -> KG, before dependants start
        if progress:
            progress(r.id)
        ts.done(r.id)

    if workers <= 1:
        while ts.is_active():
            for did in ts.get_ready():
                finish(_run_one(ctx, did, results))
    else:
        _preload_heavy_modules()
        # AI tests mostly wait on the model, so they get their own pool and start first; they never queue behind
        # the CPU-bound statistical tests (and those never wait for a slow model call).
        llm_ids = {i for i in ids if specs[i].needs_llm and ctx.llm is not None}
        with ThreadPoolExecutor(max_workers=workers) as pool, ThreadPoolExecutor(max_workers=max(len(llm_ids), 1)) as llm_pool:
            running: dict[Future, str] = {}
            while ts.is_active():
                for did in sorted(ts.get_ready(), key=lambda d: d not in llm_ids):
                    target = llm_pool if did in llm_ids else pool
                    running[target.submit(contextvars.copy_context().run, _run_one, ctx, did, results)] = did
                if not running:
                    break
                finished, _ = wait(running, return_when=FIRST_COMPLETED)
                for fut in finished:
                    running.pop(fut)
                    finish(fut.result())
    return [results[i] for i in sorted(results, key=_sort_key)]


def _sort_key(did: str) -> tuple:
    parts = did.replace(".", " ").split()
    return (parts[0], *[int(p) if p.isdigit() else p for p in parts[1:]])
