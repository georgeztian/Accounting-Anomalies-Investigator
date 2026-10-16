"""Render the report model to HTML (self-contained), Markdown, JSON and optional PDF (BLUEPRINT §12.3)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from jinja2 import Environment, FileSystemLoader, select_autoescape

from ffi.detectors.catalog import explain
from ffi.report.model import build_model

TEMPLATES = Path(__file__).parent / "templates"


def _json_default(o):
    if isinstance(o, float | np.floating):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    return str(o)


def _findings_json(out) -> dict:
    sc = out.score
    return {
        "manifest": out.manifest,
        "risk_index": sc.risk_index,
        "band": sc.band,
        "findings": [{
            "id": f.id, "title": f.title, "kind": f.kind, "score": f.score, "priority": f.priority,
            "detectors": f.detectors, "narrative": f.narrative, "narrative_source": f.narrative_source,
            "impact": f.impact.to_dict() if f.impact else None,
            "materiality": f.materiality.to_dict() if f.materiality else None,
            "evidence": [e.to_dict() for e in f.evidence],
            "alternatives": f.alternatives, "questions": f.questions,
            "audit": f.audit.to_dict() if f.audit is not None else None,
        } for f in sc.findings],
        "signals": [s.to_dict() for r in out.results for s in r.signals],
        "detectors": [{"id": r.id, "name": r.name, "family": r.family, "status": r.status, "reason": r.reason,
                       **({"what": ex.what, "why": ex.why, "how_to_read": ex.read} if (ex := explain(r.id)) else {})}
                      for r in out.results],
    }


def render(out, formats: tuple[str, ...]) -> dict[str, Path]:
    env = Environment(loader=FileSystemLoader(str(TEMPLATES)), autoescape=select_autoescape(["html", "j2"]))
    model = build_model(out)
    files: dict[str, Path] = {}
    d = out.run_dir
    if "html" in formats or "pdf" in formats:
        html = env.get_template("report.html.j2").render(m=model)
        p = d / "report.html"
        p.write_text(html, encoding="utf-8")
        files["html"] = p
    if "md" in formats:
        md_env = Environment(loader=FileSystemLoader(str(TEMPLATES)), autoescape=False)
        p = d / "report.md"
        p.write_text(md_env.get_template("report.md.j2").render(m=model), encoding="utf-8")
        files["md"] = p
    if "json" in formats:
        p = d / "report.json"
        p.write_text(json.dumps(_findings_json(out), indent=2, default=_json_default), encoding="utf-8")
        files["json"] = p
    if "pdf" in formats:
        try:
            from xhtml2pdf import pisa
        except ImportError:
            out.ctx.notes.append("PDF skipped: install the [pdf] extra (pip install -e .[pdf]).")
        else:
            p = d / "report.pdf"
            with p.open("wb") as fh:
                pisa.CreatePDF(files["html"].read_text(encoding="utf-8"), dest=fh)
            files["pdf"] = p
    return files
