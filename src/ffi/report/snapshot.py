"""Everything a report is rendered from, saved with each run, so a report can be rebuilt without re-running.

`anomaly-report render <run>` reloads this file and renders again: no SEC downloads, no detectors, no AI calls.
Only what rendering reads is kept; live handles (SEC client, AI client, graph database) and bulk data the report
never shows (the industry cross-section, full filing text) are dropped to keep the file small.
"""

from __future__ import annotations

import dataclasses
import gzip
import pickle
from pathlib import Path

import pandas as pd

SNAPSHOT = "run.pkl.gz"
SNAPSHOT_VERSION = 1


def save(out) -> Path:
    ctx = out.ctx
    light = dataclasses.replace(
        ctx, kg=None, client=None, llm=None, xsec=pd.DataFrame(),
        docs=[dataclasses.replace(d, text="", paragraphs=[], sections={}) for d in ctx.docs],
        releases=[dataclasses.replace(d, text="", paragraphs=[], sections={}) for d in ctx.releases],
    )
    data = {"version": SNAPSHOT_VERSION, "ctx": light, "results": out.results, "score": out.score,
            "manifest": out.manifest, "timings": out.timings, "agent_cases": out.agent_cases}
    path = Path(out.run_dir) / SNAPSHOT
    with gzip.open(path, "wb", compresslevel=6) as fh:
        pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load(run_dir: Path):
    """The saved run as a RunOutput. Only open snapshots this tool wrote: unpickling runs code from the file."""
    from ffi.pipeline.run import RunOutput

    with gzip.open(Path(run_dir) / SNAPSHOT, "rb") as fh:
        data = pickle.load(fh)  # noqa: S301 - written by this tool into the user's own reports folder
    if data.get("version") != SNAPSHOT_VERSION:
        raise ValueError(f"saved run format {data.get('version')} is not supported by this version; re-run the analysis")
    return RunOutput(data["ctx"], data["results"], data["score"], Path(run_dir), data["manifest"],
                     data.get("timings", {}), data.get("agent_cases", []))
