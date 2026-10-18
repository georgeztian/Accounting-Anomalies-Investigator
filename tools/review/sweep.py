"""Run every test without the AI over a varied panel of companies and report what looks wrong with the tests.

Different business models, filers and calendars break tests in different ways, so the default panel mixes them:
a manufacturer, a card issuer, a bank, an insurer, a REIT, a 52/53-week retailer, a January year end, a company
without total pre-tax income, a Japanese 20-F filer and a Canadian 40-F filer. For each company and test it reports:

- tests that errored, or skipped for a reason that suggests a bug rather than missing data;
- NaN, infinite or implausible results (days, years, severities out of range; flags with no evidence);
- tests that flag most of the panel (a test that flags everyone measures something other than anomalies);
- cross-source checks: EPS stated in the company's own earnings-release tables against the filed EPS for the same
  period (quarter with quarter, full year with full year), and hand-checked figures in known_values.toml against the
  filed figures. A disagreement means a parsing, period, scale or tagging problem.

    .venv/Scripts/python tools/review/sweep.py [--tickers AAPL,AXP | --wide] [--period FY2019] [--out sweep.md]

Reports go to a temporary folder; reports/ is never touched. Nothing calls the AI.
"""

from __future__ import annotations

import argparse
import math
import sys
import tempfile
import tomllib
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from mine_runs import _bounds_problem  # noqa: E402

PANEL = ["AAPL", "AXP", "JPM", "PGR", "O", "COST", "NVDA", "MCD", "TM", "SHOP"]
# --wide: 42 companies across business models, calendars and filer types, so a test meets many different tagging habits
WIDE_PANEL = PANEL + [
    "MSFT", "INTC", "CRM", "ORCL", "IBM", "TSLA",          # technology; January and June year ends; heavy stock pay
    "WMT", "HD", "TGT", "KO", "NKE", "DIS",                # retail and consumer; 52/53-week and non-December years
    "JNJ", "PFE", "UNH", "ABBV",                           # health care; large acquisitions and divestitures
    "CAT", "GE", "XOM", "UPS", "WM", "BA",                 # industrial and energy; segments, spin-offs, negative equity
    "BAC", "GS", "MET", "BLK",                             # banks, brokers, insurers, asset managers
    "PLD", "NEE",                                          # REIT and utility
    "ASML", "SNOW", "RIVN", "UBER",                        # foreign 20-F filer; loss-making and recently listed companies
]
SUSPECT_SKIP = ("error", "unexpected", "keyerror", "attribute", "index", "nan", "invalid", "traceback")
FLAG_SHARE = 0.6  # a test flagging more than this share of the panel is suspect


def _eps_cross_check(ctx) -> list[str]:
    from ffi.detectors.e_text import filed_eps, release_eps, releases_in_year

    out = []
    for rel in releases_in_year(ctx):
        _, gaap = release_eps(ctx, rel)
        for g in gaap:
            filed = filed_eps(ctx, rel, g.period)
            if math.isfinite(filed) and filed and abs(g.value - filed) / abs(filed) > 0.01:
                out.append(f"release {rel.period_end}: GAAP EPS ${g.value:.2f} ({g.period}) vs filed ${filed:.2f} for the same period")
    return out


def _known_checks(ctx, known: dict) -> list[str]:
    out = []
    for item, v in known.get(ctx.focus_label, {}).items():
        filed = ctx.a(item)
        if not math.isfinite(filed) or abs(filed - v) / abs(v) > 0.005:
            out.append(f"{item} {ctx.focus_label}: filed {filed:,.4g} vs hand-checked {v:,.4g}")
    return out


def sweep(tickers: list[str], period: str | None) -> str:
    from ffi.pipeline.run import RunOptions, run

    known_all = tomllib.loads((Path(__file__).parent / "known_values.toml").read_text(encoding="utf-8"))
    rows, flagged_by = [], defaultdict(set)
    errors, odd, cross, held = [], [], [], []
    with tempfile.TemporaryDirectory() as tmp:
        for t in tickers:
            try:
                res = run(RunOptions(company=t, mode="deterministic", formats=("json",), out_dir=Path(tmp), period=period))
            except Exception as e:  # noqa: BLE001 - one company failing is itself a finding
                errors.append(f"{t}: run failed: {type(e).__name__}: {e}")
                continue
            ctx = res.ctx
            rows.append(f"| {t} | {ctx.focus_label} | {ctx.profile.label if ctx.profile else ''} | {res.score.risk_index:.0f} | "
                        f"{len(res.score.findings)} | {sum(r.status == 'error' for r in res.results)} |")
            for r in res.results:
                if r.status == "error" or (r.status == "skipped" and any(k in (r.reason or "").lower() for k in SUSPECT_SKIP)):
                    errors.append(f"{t} · {r.id}: {r.status}: {r.reason}")
                for s in r.signals:
                    if s.flagged:
                        flagged_by[s.detector_id].add(t)
                    why = _bounds_problem(s)
                    if why:
                        odd.append(f"{t} · {s.detector_id} ({s.period}): {why}. {s.rationale[:160]}")
                    if "implausible" in s.tags:
                        held.append(f"{t} · {s.detector_id} ({s.period}): {s.suppressed_by}")
            cross += [f"{t} · {x}" for x in _eps_cross_check(ctx) + _known_checks(ctx, known_all.get(t, {}))]
    n = len(rows)
    broad = [f"{d}: flags {len(c)} of {n} ({', '.join(sorted(c))})" for d, c in sorted(flagged_by.items())
             if n >= 4 and len(c) / n > FLAG_SHARE]
    out = ["# Detector sweep (no AI)", "", "| Company | Year | Profile | Risk | Findings | Errors |", "|---|---|---|---:|---:|---:|", *rows, ""]
    for title, items in (("Errors and suspicious skips", errors), ("Implausible results", odd),
                         ("Results held back by the plausibility gate (check each input in the filing)", held),
                         ("Tests flagging most of the panel", broad), ("Cross-source disagreements", cross)):
        out += [f"## {title}", ""] + ([f"- {x}" for x in items] or ["- none"]) + [""]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tickers", help="comma-separated tickers (default: the mixed panel)")
    ap.add_argument("--wide", action="store_true", help="the 42-company panel (about an hour on a cold cache)")
    ap.add_argument("--period", help="fiscal year for every company, e.g. FY2024 (default: each one's latest)")
    ap.add_argument("--out", help="write the report to this file instead of printing it")
    a = ap.parse_args()
    text = sweep([t.strip() for t in a.tickers.split(",")] if a.tickers else WIDE_PANEL if a.wide else PANEL, a.period)
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
        print(f"wrote {a.out}")
    else:
        sys.stdout.reconfigure(encoding="utf-8")
        print(text)


if __name__ == "__main__":
    main()
