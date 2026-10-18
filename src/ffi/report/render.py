"""Render the report model to HTML (self-contained), Markdown, JSON and optional PDF."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
from jinja2 import Environment, FileSystemLoader, select_autoescape

from ffi.detectors.catalog import explain
from ffi.report.model import build_model

TEMPLATES = Path(__file__).parent / "templates"


def _json_default(o):
    if isinstance(o, np.floating):  # float32 and others that are not Python floats
        return _finite(float(o))
    if isinstance(o, np.integer):
        return int(o)
    return str(o)


def _finite(o):
    """NaN and infinity become null: json.dumps would write them as NaN / Infinity, which is not valid JSON."""
    if isinstance(o, float):
        return o if np.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _finite(v) for k, v in o.items()}
    if isinstance(o, list | tuple):
        return [_finite(v) for v in o]
    return o


def to_json(data) -> str:
    return json.dumps(_finite(data), indent=2, default=_json_default)


def _findings_json(out) -> dict:
    sc = out.score
    return {
        "manifest": out.manifest,
        "risk_index": sc.risk_index,
        "band": sc.band,
        "findings": [_finding_json(f) for f in sc.findings],
        "set_aside": [{**_finding_json(f), "reason": f.set_aside} for f in getattr(sc, "set_aside", [])],
        "signals": [s.to_dict() for r in out.results for s in r.signals],
        "detectors": [{"id": r.id, "name": r.name, "family": r.family, "status": r.status, "reason": r.reason,
                       **({"what": ex.what, "why": ex.why, "how_to_read": ex.read} if (ex := explain(r.id)) else {})}
                      for r in out.results],
    }


def _finding_json(f) -> dict:
    return {
        "id": f.id, "title": f.title, "kind": f.kind, "score": f.score, "priority": f.priority,
        "detectors": f.detectors, "narrative": f.narrative, "narrative_source": f.narrative_source,
        "impact": f.impact.to_dict() if f.impact else None,
        "materiality": f.materiality.to_dict() if f.materiality else None,
        "evidence": [e.to_dict() for e in f.evidence],
        "alternatives": f.alternatives, "questions": f.questions,
        "audit": f.audit.to_dict() if f.audit is not None else None,
        **({"years": f.years, "timeline": f.timeline, "signals": [s.to_dict() for s in f.signals]} if getattr(f, "years", None) else {}),
    }


def _span_json(out) -> dict:
    span = out.span
    return {
        "manifest": out.manifest,
        "periods": span.periods,
        "risk_index": span.risk_index,
        "band": span.band,
        "yearly": span.yearly,
        "area_by_year": span.area_by_year,
        "findings": [_finding_json(f) for f in span.findings],
        "set_aside": [{**_finding_json(f), "reason": f.set_aside} for f in getattr(span, "set_aside", [])],
        "years": [{"period": y.ctx.focus_label, "risk_index": y.score.risk_index, "band": y.score.band,
                   "signals": [s.to_dict() for r in y.results for s in r.signals],
                   "detectors": [{"id": r.id, "status": r.status, "reason": r.reason} for r in y.results]} for y in out.years],
    }


def render(out, formats: tuple[str, ...]) -> dict[str, Path]:
    from ffi.detectors.base import load_all

    load_all()  # test names and descriptions come from the registry (not loaded when re-rendering a saved run)
    multi = getattr(out, "kind", "") == "span"
    if multi:
        from ffi.report.span_model import build_span_model

        model, name, data = build_span_model(out), "report_span", _span_json(out)
    else:
        model, name, data = build_model(out), "report", _findings_json(out)
    pdf = "pdf" in formats and importlib.util.find_spec("xhtml2pdf") is not None
    if "pdf" in formats and not pdf:  # noted before rendering, so the report itself says why there is no PDF
        note = "PDF skipped: install the [pdf] extra (pip install -e .[pdf])."
        if note not in out.ctx.notes:
            out.ctx.notes.append(note)
        model["notes"] = [n for n in model["notes"] if n != model["none_text"]] + [note]
    env = Environment(loader=FileSystemLoader(str(TEMPLATES)), autoescape=select_autoescape(["html", "j2"]))
    files: dict[str, Path] = {}
    d = out.run_dir
    if "html" in formats or "pdf" in formats:
        html = env.get_template(f"{name}.html.j2").render(m=model)
        p = d / "report.html"
        p.write_text(html, encoding="utf-8")
        files["html"] = p
    if "md" in formats:
        md_env = Environment(loader=FileSystemLoader(str(TEMPLATES)), autoescape=False)
        p = d / "report.md"
        p.write_text(md_env.get_template(f"{name}.md.j2").render(m=model), encoding="utf-8")
        files["md"] = p
    if "json" in formats:
        p = d / "report.json"
        p.write_text(to_json(data), encoding="utf-8")
        files["json"] = p
    if pdf:
        from xhtml2pdf import pisa

        p = d / "report.pdf"
        with p.open("wb") as fh:
            pisa.CreatePDF(files["html"].read_text(encoding="utf-8"), dest=fh)
        files["pdf"] = p
    return files
