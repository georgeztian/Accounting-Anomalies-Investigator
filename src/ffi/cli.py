"""Command-line interface (BLUEPRINT §1.1).

    anomaly-report "Apple Inc. (AAPL)" [options]   run the analysis and write a report
    anomaly-report detectors                       list the detector catalogue
    anomaly-report train ...                       offline model training / backtest
    anomaly-report cache-info                      show cache location and size
"""

from __future__ import annotations

import logging
import sys
from datetime import date
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from ffi._env_guard import require_virtualenv

console = Console()
SUBCOMMANDS = {"detectors", "train", "labels", "cache-info", "batch"}
app = typer.Typer(add_completion=False, no_args_is_help=True, help="Accounting-anomaly analysis reports for SEC registrants.")


def _choose(cands: list[dict]) -> int | None:
    if not sys.stdin.isatty():
        return None
    t = Table(title="Which company did you mean?")
    for col in ("#", "Ticker", "Name", "CIK", "Match"):
        t.add_column(col)
    for i, c in enumerate(cands):
        t.add_row(str(i + 1), c["ticker"], c["title"], str(c["cik"]), f"{c['score']:.0f}")
    console.print(t)
    ans = console.input("Number (Enter to cancel): ").strip()
    return int(ans) - 1 if ans.isdigit() and 1 <= int(ans) <= len(cands) else None


@app.command("run")
def run_cmd(
    company: Annotated[str, typer.Argument(help='Company as "Name (TICKER)", a ticker, or a name')],
    years: Annotated[int, typer.Option(help="Fiscal years of history analysed")] = 5,
    period: Annotated[str | None, typer.Option(help="Focus fiscal year, e.g. FY2024 (default: latest 10-K)")] = None,
    as_of: Annotated[str | None, typer.Option("--as-of", help="Point-in-time date YYYY-MM-DD")] = None,
    peers: Annotated[str | None, typer.Option(help="Comma-separated peer tickers (overrides automatic selection)")] = None,
    format: Annotated[str, typer.Option("--format", help="Comma list of html,md,json,pdf")] = "html,md,json",
    out: Annotated[Path, typer.Option(help="Output directory")] = Path("reports"),
    mode: Annotated[str, typer.Option(help="interactive | auto | deterministic")] = "interactive",
    budget_usd: Annotated[float | None, typer.Option("--budget-usd", help="Hard cap on LLM spend (default: ffi.toml [agent] run_budget_usd)")] = None,
    max_cases: Annotated[int | None, typer.Option("--max-cases", help="Findings the agent investigates (default: ffi.toml [agent] max_cases)")] = None,
    resume: Annotated[str | None, typer.Option(help="Resume a previous run id (agent continues from checkpoints)")] = None,
    pause_on: Annotated[str | None, typer.Option("--pause-on", help="Interrupt names at which to stop and save (e.g. review_publish)")] = None,
    offline: Annotated[bool, typer.Option(help="Use cached data only")] = False,
    verbose: Annotated[bool, typer.Option("-v", "--verbose")] = False,
) -> None:
    """Analyse one company and write the report."""
    from ffi.config import get_tuning
    from ffi.edgar.client import EdgarError
    from ffi.pipeline.run import RunOptions, run
    from ffi.resolve.resolver import ResolutionError

    agent_cfg = get_tuning()["agent"]
    budget_usd = budget_usd if budget_usd is not None else float(agent_cfg.get("run_budget_usd", 6.0))
    max_cases = max_cases if max_cases is not None else int(agent_cfg.get("max_cases", 3))
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    if mode not in ("interactive", "auto", "deterministic"):
        raise typer.BadParameter("mode must be interactive, auto or deterministic")
    if mode == "interactive" and not sys.stdin.isatty():
        mode = "auto"
        console.print("[dim]No terminal attached: using --mode auto.[/dim]")
    opts = RunOptions(
        company=company, years=years, period=period, as_of=date.fromisoformat(as_of) if as_of else None,
        peers=[p for p in peers.split(",") if p.strip()] if peers else None,
        formats=tuple(f.strip().lower() for f in format.split(",")), out_dir=out, mode=mode, budget_usd=budget_usd,
        run_id=resume, offline=offline, max_cases=max_cases,
        pause_on=tuple(p.strip() for p in pause_on.split(",")) if pause_on else (),
    )
    from ffi.agent.ui import TerminalUI

    ui = TerminalUI(console, mode=mode, pause_on=opts.pause_on)
    try:
        with console.status("Starting...") as status:
            ui.status = status
            res = run(opts, choose=_choose, progress=lambda s: status.update(f"[bold]{s}[/bold]"), agent_ui=ui)
    except ResolutionError as e:
        console.print(f"[red]Could not resolve the company:[/red] {e}")
        for c in e.candidates:
            console.print(f"  {c['ticker']:8} {c['title']}  (CIK {c['cik']}, match {c['score']:.0f})")
        raise typer.Exit(e.exit_code) from e
    except EdgarError as e:
        console.print(f"[red]SEC EDGAR error:[/red] {e}")
        raise typer.Exit(5) from e
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(6) from e
    _summary(res)


def _summary(res) -> None:
    sc, ctx = res.score, res.ctx
    console.rule(f"[bold]{ctx.company.name} ({ctx.company.ticker}) {ctx.focus_label}")
    console.print(f"Risk index [bold]{sc.risk_index:.0f}[/bold] / 100 ({sc.band})   ·   resolution: {ctx.company.match_note}")
    t = Table(show_header=True, header_style="bold")
    t.add_column("#")
    t.add_column("Finding")
    t.add_column("Strength", justify="right")
    t.add_column("Impact (mid)", justify="right")
    t.add_column("% pre-tax", justify="right")
    from ffi.detectors.stats import fmt_money, fmt_pct

    for i, f in enumerate(sc.findings[:8], 1):
        m = f.materiality
        t.add_row(str(i), f.title, f"{f.score:.2f}", fmt_money(m.impact_mid) if m else "-", fmt_pct(m.pct_pretax) if m and m.pct_pretax is not None else "-")
    if sc.findings:
        console.print(t)
    else:
        console.print("No findings met the reporting threshold.")
    stat = {"ok": 0, "skipped": 0, "n/a": 0, "error": 0}
    for r in res.results:
        stat[r.status] = stat.get(r.status, 0) + 1
    if ctx.profile is not None:
        console.print(f"Profile: {ctx.profile.label} · {ctx.annual_form} filer · amounts in {ctx.currency}")
    console.print(f"Detectors: {stat['ok']} ran, {stat['skipped']} skipped (missing data), {stat['n/a']} not applicable "
                  f"to this profile, {stat['error']} errored. Timings: "
                  + ", ".join(f"{k} {v}s" for k, v in res.timings.items()))
    for kind, p in res.files.items():
        console.print(f"[green]{kind.upper()}[/green] {p.resolve()}")
    if ctx.llm:
        console.print(f"LLM spend ${ctx.llm.spend.usd:.2f} over {ctx.llm.spend.calls} calls.")


@app.command("batch")
def batch_cmd(
    companies: Annotated[Path, typer.Argument(help='Text file: one company per line ("Name (TICKER)", ticker or name); # comments allowed')],
    mode: Annotated[str, typer.Option(help="auto | deterministic (batch runs are never interactive)")] = "deterministic",
    format: Annotated[str, typer.Option("--format")] = "html,json",
    out: Annotated[Path, typer.Option(help="Output directory")] = Path("reports"),
    workers: Annotated[int, typer.Option(help="Companies analysed concurrently (EDGAR traffic shares one rate limit)")] = 2,
    budget_usd: Annotated[float, typer.Option("--budget-usd", help="LLM cap per company")] = 3.0,
) -> None:
    """Analyse many companies in one process and write a ranked summary (CSV + Markdown)."""
    import csv
    import time as _time
    from datetime import datetime

    from ffi.concurrency import map_concurrent
    from ffi.pipeline.run import RunOptions, run

    if mode not in ("auto", "deterministic"):
        raise typer.BadParameter("batch mode must be auto or deterministic")
    lines = [ln.strip() for ln in companies.read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.strip().startswith("#")]
    console.print(f"Analysing {len(lines)} companies with {workers} worker(s)...")

    def one(name: str) -> dict:
        t0 = _time.perf_counter()
        try:
            res = run(RunOptions(company=name, mode=mode, formats=tuple(format.split(",")), out_dir=out, budget_usd=budget_usd))
            sc = res.score
            top = sc.findings[0] if sc.findings else None
            row = {"input": name, "ticker": res.ctx.company.ticker, "company": res.ctx.company.name, "period": res.ctx.focus_label,
                   "risk_index": sc.risk_index, "band": sc.band, "findings": len(sc.findings),
                   "verified": sum(1 for f in sc.findings if f.audit and f.audit.status != "unverified"),
                   "unverified": sum(1 for f in sc.findings if f.audit and f.audit.status == "unverified"),
                   "top_finding": top.title if top else "", "report": str(res.files.get("html") or res.files.get("json") or ""),
                   "seconds": round(_time.perf_counter() - t0, 1), "error": ""}
        except Exception as e:  # noqa: BLE001 - one company failing must not stop the batch
            row = {"input": name, "error": f"{type(e).__name__}: {e}", "seconds": round(_time.perf_counter() - t0, 1)}
        try:
            status = f"{row['risk_index']:.0f} ({row['band']}) in {row['seconds']}s" if not row["error"] else row["error"]
            console.print(f"  {'OK  ' if not row['error'] else 'FAIL'} {name}: {status}", markup=False)
        except Exception:  # noqa: BLE001 - a console encoding problem must never lose a result
            pass
        return row

    results = map_concurrent(one, lines, workers=workers)
    rows = [r if isinstance(r, dict) else {"input": name, "error": f"{type(r).__name__}: {r}"} for name, r in zip(lines, results, strict=True)]
    rows.sort(key=lambda r: -(r.get("risk_index") or -1))
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    fields = ["input", "ticker", "company", "period", "risk_index", "band", "findings", "verified", "unverified", "top_finding", "report", "seconds", "error"]
    csv_path = out / f"batch_{stamp}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows({k: r.get(k, "") for k in fields} for r in rows)
    md = ["| Company | Period | Risk index | Findings (verified / unverified) | Top finding |", "|---|---|---:|---|---|"]
    md += [f"| {r.get('company', r['input'])} ({r.get('ticker', '')}) | {r.get('period', '')} | {r.get('risk_index', 'error')} {r.get('band', '')} | "
           f"{r.get('findings', '')} ({r.get('verified', '')} / {r.get('unverified', '')}) | {r.get('top_finding') or r.get('error', '')} |" for r in rows]
    (out / f"batch_{stamp}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    console.print(f"[green]Summary[/green] {csv_path.resolve()}")


@app.command("detectors")
def detectors_cmd() -> None:
    """List every registered detector."""
    from ffi.detectors.base import load_all

    t = Table("ID", "Family", "Name", "Needs")
    for did, s in load_all().items():
        needs = ", ".join(n for n, v in (("LLM", s.needs_llm), ("text", s.needs_text), ("peers", s.needs_peers)) if v)
        t.add_row(did, s.family, s.name, needs)
    console.print(t)


@app.command("cache-info")
def cache_info() -> None:
    """Show where data is cached and how large the cache is."""
    from ffi.config import get_settings

    root = get_settings().cache_dir
    size = sum(p.stat().st_size for p in root.rglob("*") if p.is_file()) if root.exists() else 0
    console.print(f"Cache: {root}  ({size / 1e6:,.1f} MB)")


@app.command("train")
def train_cmd(
    labels: Annotated[Path, typer.Option(help="CSV with columns cik,fiscal_year,label (1 = misstated year)")],
    controls: Annotated[int, typer.Option(help="Random control firm-years to sample per positive")] = 3,
    backtest: Annotated[bool, typer.Option(help="Walk-forward evaluation instead of a final fit")] = False,
) -> None:
    """Train the supervised misstatement model (D.06) from a labels file."""
    from ffi.training.train import train

    report = train(labels, controls_per_positive=controls, backtest=backtest, console=console)
    console.print(report)


@app.command("labels")
def labels_cmd(
    out: Annotated[Path, typer.Option(help="Output CSV")] = Path("data/labels_4_02.csv"),
    start: Annotated[int, typer.Option(help="First filing year")] = 2016,
    end: Annotated[int, typer.Option(help="Last filing year")] = 2022,
    per_year: Annotated[int, typer.Option(help="Max search hits per year")] = 100,
) -> None:
    """Build starter (weak) labels from 8-K Item 4.02 non-reliance filings via SEC full-text search."""
    from ffi.edgar.client import EdgarClient
    from ffi.training.labels import build_labels

    n = build_labels(EdgarClient(), start, end, out, per_year)
    console.print(f"Wrote {n} labelled firm-years to {out.resolve()}")


def main() -> None:
    require_virtualenv()
    argv = sys.argv[1:]
    if argv and argv[0] not in SUBCOMMANDS and argv[0] != "run" and not argv[0].startswith("-"):
        sys.argv.insert(1, "run")
    app()


if __name__ == "__main__":
    main()
