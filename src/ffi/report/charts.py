"""Inline-SVG charts for the self-contained HTML report.

Colors are CSS custom properties (defined in the report stylesheet for light and dark), using the
reference dataviz palette: series-1 blue = the company, series-2 orange = peer median, gray band =
peer interquartile range. One y-axis per chart; 2px lines; >=8px markers; <title> tooltips on marks.
"""

from __future__ import annotations

import html
import math
from collections.abc import Sequence

import numpy as np

W, H = 640, 240
PAD_L, PAD_R, PAD_T, PAD_B = 64, 96, 16, 32


def _nice_ticks(lo: float, hi: float, n: int = 4) -> list[float]:
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi == lo:
        hi = lo + 1 if np.isfinite(lo) else 1
        lo = lo if np.isfinite(lo) else 0
    span = hi - lo
    raw = span / n
    mag = 10 ** math.floor(math.log10(raw))
    step = min((s * mag for s in (1, 2, 2.5, 5, 10) if s * mag >= raw), default=raw)
    start = math.floor(lo / step) * step
    ticks, t = [], start
    while t <= hi + step * 0.5:
        ticks.append(round(t, 12))
        t += step
    return ticks


def _fmt(v: float, kind: str) -> str:
    if not np.isfinite(v):
        return "n/a"
    if kind == "money":
        a = abs(v)
        s = "-" if v < 0 else ""
        return f"{s}${a / 1e9:.1f}B" if a >= 1e9 else f"{s}${a / 1e6:.0f}M" if a >= 1e6 else f"{s}${a:,.0f}"
    if kind == "pct":
        return f"{v * 100:.1f}%"
    if kind == "days":
        return f"{v:.0f}d"
    return f"{v:.2f}"


def line_chart(title: str, labels: Sequence[str], series: list[dict], kind: str = "num",
               band: tuple[Sequence[float], Sequence[float]] | None = None, aria: str = "") -> str:
    """series: [{name, values, cls: 's1'|'s2', dashed?}] ; band: (low, high) arrays for a shaded range."""
    n = len(labels)
    if n < 2:
        return ""
    vals = [v for s in series for v in s["values"] if v is not None and np.isfinite(v)]
    if band:
        vals += [v for v in list(band[0]) + list(band[1]) if v is not None and np.isfinite(v)]
    if not vals:
        return ""
    lo, hi = min(vals), max(vals)
    if lo > 0 and kind in ("money", "days") and lo / (hi or 1) > 0.3:
        lo = 0.0
    pad = (hi - lo) * 0.08 or abs(hi) * 0.1 or 1
    ticks = _nice_ticks(lo - pad * 0.5, hi + pad * 0.5)
    y0, y1 = ticks[0], ticks[-1]
    pw, ph = W - PAD_L - PAD_R, H - PAD_T - PAD_B

    def X(i: int) -> float:
        return PAD_L + pw * (i / (n - 1))

    def Y(v: float) -> float:
        return PAD_T + ph * (1 - (v - y0) / ((y1 - y0) or 1))

    out = [f'<figure class="chart"><figcaption>{html.escape(title)}</figcaption>',
           f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{html.escape(aria or title)}">']
    for t in ticks:
        out.append(f'<line class="grid" x1="{PAD_L}" x2="{W - PAD_R}" y1="{Y(t):.1f}" y2="{Y(t):.1f}"/>')
        out.append(f'<text class="tick" x="{PAD_L - 8}" y="{Y(t) + 4:.1f}" text-anchor="end">{_fmt(t, kind)}</text>')
    if y0 < 0 < y1:
        out.append(f'<line class="zero" x1="{PAD_L}" x2="{W - PAD_R}" y1="{Y(0):.1f}" y2="{Y(0):.1f}"/>')
    step = max(1, n // 8)
    for i, lab in enumerate(labels):
        if i % step == 0 or i == n - 1:
            out.append(f'<text class="tick" x="{X(i):.1f}" y="{H - 10}" text-anchor="middle">{html.escape(lab)}</text>')
    if band:
        pts_hi = [(X(i), Y(v)) for i, v in enumerate(band[1]) if v is not None and np.isfinite(v)]
        pts_lo = [(X(i), Y(v)) for i, v in enumerate(band[0]) if v is not None and np.isfinite(v)]
        if len(pts_hi) >= 2 and len(pts_lo) >= 2:
            poly = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts_hi + pts_lo[::-1])
            out.append(f'<polygon class="band" points="{poly}"><title>Peer interquartile range</title></polygon>')
    for s in series:
        pts = [(i, v) for i, v in enumerate(s["values"]) if v is not None and np.isfinite(v)]
        if len(pts) < 1:
            continue
        d = " ".join(f"{'M' if k == 0 else 'L'}{X(i):.1f},{Y(v):.1f}" for k, (i, v) in enumerate(pts))
        dash = ' stroke-dasharray="6 4"' if s.get("dashed") else ""
        out.append(f'<path class="line {s["cls"]}" d="{d}"{dash}/>')
        for i, v in pts:
            out.append(f'<circle class="dot {s["cls"]}" cx="{X(i):.1f}" cy="{Y(v):.1f}" r="4">'
                       f'<title>{html.escape(s["name"])}, {html.escape(labels[i])}: {_fmt(v, kind)}</title></circle>')
        li, lv = pts[-1]
        out.append(f'<text class="direct" x="{X(li) + 8:.1f}" y="{Y(lv) + 4:.1f}">{html.escape(s["name"])}</text>')
    out.append("</svg>")
    if len(series) >= 2 or band:
        leg = "".join(f'<span class="key"><i class="sw {s["cls"]}{" dashed" if s.get("dashed") else ""}"></i>{html.escape(s["name"])}</span>' for s in series)
        if band:
            leg += '<span class="key"><i class="sw band"></i>Peer interquartile range</span>'
        out.append(f'<div class="legend">{leg}</div>')
    out.append("</figure>")
    return "".join(out)


def dot_strip(title: str, rows: list[dict]) -> str:
    """rows: [{label, value, peers: [..], ref: float|None, ref_label, kind}] one row per score."""
    rows = [r for r in rows if r.get("value") is not None and np.isfinite(r["value"])]
    if not rows:
        return ""
    rh = 56
    h = 24 + rh * len(rows)
    out = [f'<figure class="chart"><figcaption>{html.escape(title)}</figcaption>',
           f'<svg viewBox="0 0 {W} {h}" role="img" aria-label="{html.escape(title)}">']
    x0, x1 = 170, W - 40
    for k, r in enumerate(rows):
        cy = 24 + rh * k + 20
        vals = [v for v in r["peers"] if np.isfinite(v)] + [r["value"]] + ([r["ref"]] if r.get("ref") is not None else [])
        lo, hi = min(vals), max(vals)
        span = (hi - lo) or 1
        lo -= span * 0.08
        hi += span * 0.08

        def X(v: float, lo=lo, hi=hi) -> float:
            return x0 + (x1 - x0) * (v - lo) / ((hi - lo) or 1)

        out.append(f'<text class="rowlabel" x="0" y="{cy + 4}">{html.escape(r["label"])}</text>')
        out.append(f'<line class="grid" x1="{x0}" x2="{x1}" y1="{cy}" y2="{cy}"/>')
        if r.get("ref") is not None:
            out.append(f'<line class="ref" x1="{X(r["ref"]):.1f}" x2="{X(r["ref"]):.1f}" y1="{cy - 14}" y2="{cy + 14}"><title>{html.escape(r.get("ref_label", "reference"))}</title></line>')
            out.append(f'<text class="tick" x="{X(r["ref"]):.1f}" y="{cy + 28}" text-anchor="middle">{html.escape(r.get("ref_label", ""))}</text>')
        for v in r["peers"]:
            if np.isfinite(v):
                out.append(f'<circle class="dot peer" cx="{X(v):.1f}" cy="{cy}" r="4"><title>Peer: {_fmt(v, r.get("kind", "num"))}</title></circle>')
        out.append(f'<circle class="dot s1 big" cx="{X(r["value"]):.1f}" cy="{cy}" r="6"><title>Company: {_fmt(r["value"], r.get("kind", "num"))}</title></circle>')
        out.append(f'<text class="direct" x="{X(r["value"]):.1f}" y="{cy - 12}" text-anchor="middle">{_fmt(r["value"], r.get("kind", "num"))}</text>')
    out.append("</svg>")
    out.append('<div class="legend"><span class="key"><i class="sw s1"></i>Company</span><span class="key"><i class="sw peer"></i>Peers</span>'
               '<span class="key"><i class="sw ref"></i>Published reference cut-off</span></div></figure>')
    return "".join(out)


def evidence_graph(center: str, groups: list[tuple[str, list[str]]]) -> str:
    """Radial evidence map: finding at the centre, evidence grouped by kind on a ring."""
    items = [(g, lab) for g, labs in groups for lab in labs][:14]
    if not items:
        return ""
    w, h = 640, 340
    cx, cy, r = w / 2, h / 2, 130
    out = [f'<svg class="graph" viewBox="0 0 {w} {h}" role="img" aria-label="Evidence graph for {html.escape(center)}">']
    for k, (g, lab) in enumerate(items):
        a = 2 * math.pi * k / len(items) - math.pi / 2
        x, y = cx + r * math.cos(a), cy + r * 0.9 * math.sin(a)
        out.append(f'<line class="edge" x1="{cx}" y1="{cy}" x2="{x:.1f}" y2="{y:.1f}"/>')
        anchor = "start" if math.cos(a) > 0.2 else "end" if math.cos(a) < -0.2 else "middle"
        tx = x + (10 if anchor == "start" else -10 if anchor == "end" else 0)
        ty = y + (16 if math.sin(a) > 0.5 else -8 if math.sin(a) < -0.5 else 4)
        short = lab if len(lab) <= 34 else lab[:32] + "…"
        out.append(f'<circle class="node {g}" cx="{x:.1f}" cy="{y:.1f}" r="6"><title>{html.escape(lab)}</title></circle>')
        out.append(f'<text class="nodelabel" x="{tx:.1f}" y="{ty:.1f}" text-anchor="{anchor}">{html.escape(short)}</text>')
    out.append(f'<circle class="node center" cx="{cx}" cy="{cy}" r="10"/>')
    out.append("</svg>")
    out.append('<div class="legend"><span class="key"><i class="sw fact"></i>Reported fact</span><span class="key"><i class="sw span"></i>Filing text</span>'
               '<span class="key"><i class="sw derived"></i>Computed value</span><span class="key"><i class="sw event"></i>Event</span></div>')
    return "".join(out)
