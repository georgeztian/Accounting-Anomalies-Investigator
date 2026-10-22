"""Everything a report is rendered from, saved with each run, so a report can be rebuilt without re-running.

`anomaly-report render <run>` reloads this file and renders again: no SEC downloads, no detectors, no AI calls.
Only what rendering reads is kept; live handles (SEC client, AI client, graph database) and bulk data the report
never shows (the raw fact store, the industry cross-section, full filing text, the source facts behind peers'
figures other than their basis) are dropped to keep the file small. A multi-year review saves every year's context the same way; contexts
shared by several anomalies are saved once.
"""

from __future__ import annotations

import dataclasses
import gzip
import pickle
from pathlib import Path

import pandas as pd

from ffi.xbrl.canonical import basis

SNAPSHOT = "run.pkl.gz"
SNAPSHOT_VERSION = 1


def _lighten(memo: dict[int, object]):
    """A function returning the light copy of a context (or document, or peer set), each made once, so objects the
    run shares (documents across years, one context for several anomalies) stay shared in the file."""
    def once(obj, make):
        if id(obj) not in memo:
            memo[id(obj)] = (obj, make(obj))  # the original is held so its id is not reused while saving
        return memo[id(obj)][1]

    def doc(d):
        return once(d, lambda d: dataclasses.replace(d, text="", paragraphs=[], sections={}))

    def peers(ps):
        if ps is None:
            return None
        # a peer's source facts are dropped except those on another basis, which decide whether it is compared at all
        return once(ps, lambda ps: dataclasses.replace(ps, peers=[dataclasses.replace(p, annual=once(p.annual, lambda a: dataclasses.replace(
            a, prov={k: r for k, r in a.prov.items() if basis(k[0], r.concept)}))) if p.annual is not None else p for p in ps.peers]))

    def light(ctx):
        def make(c):
            out = dataclasses.replace(c, kg=None, client=None, llm=None, store=None, xsec=pd.DataFrame(), peers=peers(c.peers),
                                      docs=[doc(d) for d in c.docs], releases=[doc(d) for d in c.releases])
            # the year the company changed shape is read from the fact store, which is not saved: keep it, so a re-rendered
            # report compares the same years as the original
            out.__dict__["_perimeter_break"] = c.perimeter_break
            return out
        return once(ctx, make)
    return light


def save(out) -> Path:
    _light = _lighten({})
    if getattr(out, "kind", "") == "span":
        years = [dataclasses.replace(y, ctx=_light(y.ctx)) for y in out.years]
        data = {"version": SNAPSHOT_VERSION, "kind": "span", "years": years, "span": out.span,
                "contexts": {k: _light(c) for k, c in out.contexts.items()},
                "manifest": out.manifest, "timings": out.timings, "agent_cases": out.agent_cases}
    else:
        data = {"version": SNAPSHOT_VERSION, "ctx": _light(out.ctx), "results": out.results, "score": out.score,
                "manifest": out.manifest, "timings": out.timings, "agent_cases": out.agent_cases}
    path = Path(out.run_dir) / SNAPSHOT
    with gzip.open(path, "wb", compresslevel=6) as fh:
        pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load(run_dir: Path):
    """The saved run as a RunOutput (or SpanOutput). Only open snapshots this tool wrote: unpickling runs code from the file."""
    from ffi.pipeline.run import RunOutput

    with gzip.open(Path(run_dir) / SNAPSHOT, "rb") as fh:
        data = pickle.load(fh)  # noqa: S301 - written by this tool into the user's own reports folder
    if data.get("version") != SNAPSHOT_VERSION:
        raise ValueError(f"saved run format {data.get('version')} is not supported by this version; re-run the analysis")
    if data.get("kind") == "span":
        from ffi.pipeline.span import SpanOutput

        return SpanOutput(data["years"], data["span"], data["contexts"], Path(run_dir), data["manifest"],
                          data.get("timings", {}), data.get("agent_cases", []))
    return RunOutput(data["ctx"], data["results"], data["score"], Path(run_dir), data["manifest"],
                     data.get("timings", {}), data.get("agent_cases", []))
