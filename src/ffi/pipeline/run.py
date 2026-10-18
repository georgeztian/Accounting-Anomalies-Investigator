"""End-to-end run for one company: resolve -> fetch -> normalize -> KG -> detect -> score -> agent -> report."""

from __future__ import annotations

import contextlib
import contextvars
import dataclasses
import json
import logging
import re
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from ffi import __version__
from ffi.config import get_settings, get_tuning
from ffi.context import Event, RunContext, announced_quarter
from ffi.detectors.stats import CURRENCY
from ffi.edgar import endpoints
from ffi.edgar.client import EdgarClient, EdgarError
from ffi.kg import build as kg_build
from ffi.kg.store import DuckNXGraphStore
from ffi.peers.crosssection import build as build_xsec
from ffi.peers.crosssection import calendar_year_of
from ffi.peers.select import PeerSet, select_peers
from ffi.peers.universe import expanding_universe
from ffi.pipeline.detect import DetectorResult, run_detectors
from ffi.profiles import for_company
from ffi.report import snapshot
from ffi.resolve.resolver import CompanyRef, resolve
from ffi.scoring.engine import ScoreResult, score
from ffi.text.document import TextDoc, build_doc, find_auditor
from ffi.xbrl.facts import FactStore, load_companyfacts

log = logging.getLogger(__name__)

EVENT_ITEMS = {"4.01": "AUDITOR_CHANGE", "4.02": "NON_RELIANCE", "5.02": "OFFICER_CHANGE", "2.01": "ACQUISITION"}


@dataclass
class RunOptions:
    company: str
    years: int = 10
    period: str | None = None
    as_of: date | None = None
    peers: list[str] | None = None
    formats: tuple[str, ...] = ("html", "md", "json")
    out_dir: Path = Path("reports")
    mode: str = "interactive"  # interactive | auto | deterministic
    budget_usd: float = 6.0
    run_id: str | None = None
    offline: bool = False
    max_cases: int = 3
    pause_on: tuple[str, ...] = ()
    keep_artifacts: bool = False  # keep the run's graph database and agent checkpoints after a successful run


@dataclass
class RunOutput:
    ctx: RunContext
    results: list[DetectorResult]
    score: ScoreResult
    run_dir: Path
    manifest: dict[str, Any]
    timings: dict[str, float] = field(default_factory=dict)
    agent_cases: list[Any] = field(default_factory=list)
    files: dict[str, Path] = field(default_factory=dict)


class Step:
    """Times each pipeline stage. A stage that runs several times (once per year in a multi-year run) adds up."""

    def __init__(self, timings: dict[str, float], progress: Callable[[str], None] | None, prefix: str = "") -> None:
        self.timings, self.progress, self.prefix = timings, progress, prefix

    def __call__(self, name: str):
        timings, progress, prefix = self.timings, self.progress, self.prefix

        class _Ctx:
            def __enter__(self_inner):
                self_inner.t = time.perf_counter()
                if progress:
                    progress(prefix + name)
                return self_inner

            def __exit__(self_inner, *exc):
                timings[name] = round(timings.get(name, 0.0) + time.perf_counter() - self_inner.t, 2)
                return False

        return _Ctx()


OFFICER_TEXTS = 8  # Item 5.02 filings read to tell which officer changed (the latest in the officer-turnover window)


def load_events(client: EdgarClient, cik: int, filings, as_of: date, focus: pd.Timestamp) -> list[Event]:
    from ffi.concurrency import map_concurrent

    lo = (focus - pd.Timedelta(days=4 * 366)).date()
    out: list[Event] = []
    # the filings C.GOV.03 counts (two years to the focus year end, plus the filing season after it)
    win_lo, win_hi = (focus - pd.Timedelta(days=730)).date(), min(as_of, (focus + pd.Timedelta(days=120)).date())
    officer = [f for f in filings if f.form == "8-K" and "5.02" in (f.items or "").split(",") and win_lo <= f.filing_date <= win_hi]
    texts = dict(zip((f.accession for f in officer[-OFFICER_TEXTS:]),
                     map_concurrent(lambda f: client.get_text(endpoints.document_url(cik, f)), officer[-OFFICER_TEXTS:], workers=4),
                     strict=True))
    for f in filings:
        if not (lo <= f.filing_date <= as_of):
            continue
        url = endpoints.document_url(cik, f)
        if f.form in ("NT 10-K", "NT 10-Q", "NT 20-F"):
            out.append(Event(f"{f.accession}:late", "LATE_FILING", f.filing_date, f.accession, f"{f.form} notification of late filing", url))
        elif f.form in ("10-K/A", "20-F/A", "40-F/A"):
            out.append(Event(f"{f.accession}:amend", "AMENDED_10K", f.filing_date, f.accession, f"Amended annual report ({f.form})", url))
        elif f.form == "8-K":
            for item, etype in EVENT_ITEMS.items():
                if item in (f.items or "").split(","):
                    desc = f"8-K Item {item}"
                    if item == "5.02" and isinstance(texts.get(f.accession), str):  # a failed download stays undescribed
                        desc += _officer_change_kind(texts[f.accession])
                    out.append(Event(f"{f.accession}:{item}", etype, f.filing_date, f.accession, desc, url))
    return out


OFFICER_RE = re.compile(
    r"(?is)(chief financial officer|principal financial officer|chief accounting officer|principal accounting officer|chief executive officer)"
    r".{0,200}?\b(resign\w*|retire\w*|depart\w*|step(?:s|ped)? down|terminat\w*|succeed\w*|successor|appoint\w*|transition\w*)"
    r"|\b(resign\w*|retire\w*|depart\w*|step(?:s|ped)? down|terminat\w*|appoint\w*)\b.{0,200}?"
    r"(chief financial officer|principal financial officer|chief accounting officer|principal accounting officer|chief executive officer)"
)


ITEM_502_TITLE_RE = re.compile(r"(?i)departure of directors or (?:certain|principal) officers;?(?:\s*election of directors;?)?"
                               r"(?:\s*appointment of (?:certain|principal) officers;?)?(?:\s*compensatory arrangements of certain officers\.?)?")


def _officer_change_kind(raw_html: str) -> str:
    """Describe which senior officer an Item 5.02 8-K changes (compensation-only 5.02s mention none)."""
    from ffi.text.document import html_to_lines

    text = " ".join(line for line, _ in html_to_lines(raw_html))[:20000]
    text = ITEM_502_TITLE_RE.sub(" ", text)  # the item's own title says "Appointment of Certain Officers" in every 5.02
    roles = set()
    for m in OFFICER_RE.finditer(text):
        role = (m.group(1) or m.group(4) or "").lower()
        if "financial" in role:
            roles.add("CFO change")
        elif "accounting" in role:
            roles.add("CAO change")
        elif "executive" in role:
            roles.add("CEO change")
    return (" - " + ", ".join(sorted(roles))) if roles else " - no senior finance/CEO departure or appointment language"


def load_docs(client: EdgarClient, cik: int, filings, store_periods: list[pd.Timestamp], focus: pd.Timestamp, as_of: date, n: int = 3,
              cache: dict[str, TextDoc] | None = None) -> list[TextDoc]:
    """The annual reports of the focus year and the n - 1 before it. `cache` (accession -> parsed document) lets the years of
    a multi-year review share each parsed report instead of parsing it up to n times."""
    from ffi.concurrency import map_concurrent

    tenks = [f for f in filings if f.form in ANNUAL_TEXT_FORMS and f.filing_date <= as_of and f.report_date]
    wanted = [p for p in store_periods if p <= focus][-n:]
    picks = []
    for p in wanted:
        f = min(tenks, key=lambda x: abs((pd.Timestamp(x.report_date) - p).days), default=None)
        if f is not None and abs((pd.Timestamp(f.report_date) - p).days) <= 15:
            picks.append(f)

    def load(f):
        if cache is not None and f.accession in cache:
            return cache[f.accession]
        url = endpoints.document_url(cik, f)
        raw = client.get_text(url)
        if f.form == "40-F":  # the AIF, MD&A and audited statements are exhibits to the 40-F cover document
            return build_doc(f.accession, f.form, str(f.report_date), url, raw, _exhibits(client, cik, f, EXHIBITS_40F))
        doc = build_doc(f.accession, f.form, str(f.report_date), url, raw)
        if f.form.startswith("10-K") and len(doc.section_paragraphs("item7")) < 20:
            # MD&A and the statements incorporated by reference from the annual report to shareholders (Exhibit 13)
            appendix = _exhibits(client, cik, f, EXHIBITS_10K)
            if appendix:
                doc = build_doc(f.accession, f.form, str(f.report_date), url, raw, appendix)
        return doc

    docs = []
    for f, d in zip(picks, map_concurrent(load, picks, workers=3), strict=True):
        if isinstance(d, Exception):
            log.warning("annual report text %s unavailable: %s", f.accession, d)
        else:
            docs.append(d)
            if cache is not None:
                cache[f.accession] = d
    return docs


ANNUAL_TEXT_FORMS = ("10-K", "10-KT", "20-F", "40-F")
# (exhibit type prefix, pattern on description or file name, synthetic 10-K section header)
EXHIBITS_10K = [("EX-13", r"", "Item 7. Management Discussion and Analysis (Exhibit 13 annual report)")]
EXHIBITS_40F = [
    ("EX-99", r"annual information form|\baif\b", "Item 1. Business (Annual Information Form)"),
    ("EX-99", r"management.?s discussion|md&a|\bmda\b", "Item 7. Management Discussion and Analysis"),
    ("EX-99", r"financial statements|financial report", "Item 8. Financial Statements"),
]


def _exhibits(client: EdgarClient, cik: int, f, wanted: list[tuple[str, str, str]]) -> list[tuple[str, str]]:
    """Fetch the annual-report exhibits of a filing as (synthetic section header, HTML) pairs."""
    try:
        rows = endpoints.filing_index(client, cik, f)
    except EdgarError:
        return []
    out: list[tuple[str, str]] = []
    for prefix, pat, header in wanted:
        row = next((r for r in rows if r["type"].upper().startswith(prefix) and r["document"].lower().endswith((".htm", ".html"))
                    and re.search(pat, f"{r['description']} {r['document']}", re.I)), None)
        if row is None:
            continue
        try:
            out.append((header, client.get_text(endpoints.document_url(cik, f, row["document"]))))
        except EdgarError as e:
            log.warning("exhibit %s of %s unavailable: %s", row["document"], f.accession, e)
    return out


RELEASE_DAYS_AFTER = 120  # the focus year's results release comes out within this many days of its year end


def load_releases(client: EdgarClient, cik: int, filings, focus: pd.Timestamp, as_of: date, n: int = 6,
                  cache: dict[str, TextDoc] | None = None) -> list[TextDoc]:
    """The last n earnings releases (8-K Item 2.02, Ex. 99) up to the focus year's results: releases about later fiscal
    years are never read, so a past year (--period, or the early years of a multi-year review) sees its own releases."""
    from ffi.concurrency import map_concurrent

    lo, hi = (focus - pd.Timedelta(days=430)).date(), min(as_of, (focus + pd.Timedelta(days=RELEASE_DAYS_AFTER)).date())
    # a release filed within the window can still announce the next year's first quarter (American Express files it
    # about 107 days after its year end); it is later-period information and is never read
    cands = [f for f in filings if f.form == "8-K" and "2.02" in (f.items or "").split(",") and lo <= f.filing_date <= hi
             and announced_quarter(focus, pd.Timestamp(f.report_date or f.filing_date)) <= focus + pd.Timedelta(days=8)][-n:]

    def load(f):
        if cache is not None and f.accession in cache:
            return cache[f.accession]
        rows = endpoints.filing_index(client, cik, f)
        ex = next((r for r in rows if r["type"].upper().startswith("EX-99")), None)
        if not ex or not ex["document"]:
            return None
        url = endpoints.document_url(cik, f, ex["document"])
        doc = build_doc(f.accession, "EX-99", str(f.report_date or f.filing_date), url, client.get_text(url))
        if cache is not None:
            cache[f.accession] = doc
        return doc

    return [d for d in map_concurrent(load, cands, workers=4) if d is not None and not isinstance(d, Exception)]


def point_in_time(focus: pd.Timestamp, filings, as_of: date) -> date:
    """The last day a fiscal year's analysis may read: the end of its filing season (RELEASE_DAYS_AFTER days after the
    year end, or the day its annual report was filed if that is later), never after --as-of. A past year (--period, or
    the earlier years of a multi-year review) is analysed with what was public then: no later restatement, annual or
    quarterly report, earnings release or 8-K event reaches its figures, tests or investigation."""
    season = (focus + pd.Timedelta(days=RELEASE_DAYS_AFTER)).date()
    filed = [f.filing_date for f in filings if f.form in ANNUAL_TEXT_FORMS and f.report_date
             and abs((pd.Timestamp(f.report_date) - focus).days) <= 15]
    return min(as_of, max([season, *([min(filed)] if filed else [])]))


def pick_focus(store: FactStore, period: str | None) -> pd.Timestamp:
    fyes = [p for p in store.fiscal_year_ends()]
    if not fyes:
        raise ValueError("no annual financial data (10-K, 20-F or 40-F XBRL) found for this company")
    if period:
        want = period.upper().replace(" ", "")
        for p in fyes:
            if store.fiscal_label(p) == want:
                return p
        raise ValueError(f"period {period} not found; available: {', '.join(store.fiscal_label(p) for p in fyes[-8:])}")
    return fyes[-1]


@dataclass
class Prepared:
    """What every analysed year shares: the company, its filings and facts, and the AI client."""
    client: EdgarClient
    company: CompanyRef
    filings: list
    cf: Any
    store: FactStore
    profile: Any
    tuning: dict[str, Any]
    as_of: date
    llm: Any
    notes: list[str]
    # shared by the years of a multi-year review: parsed filings by accession, peers' annual panels by CIK
    doc_cache: dict[str, TextDoc] = field(default_factory=dict)
    release_cache: dict[str, TextDoc] = field(default_factory=dict)
    peer_cache: dict[int, Any] = field(default_factory=dict)


@dataclass
class YearAnalysis:
    """One fiscal year analysed: detectors run and scored, before any investigation."""
    ctx: RunContext
    results: list[DetectorResult]
    score: ScoreResult
    kg_counts: dict[str, Any]


def prepare(opts: RunOptions, choose=None, step: Step | None = None) -> Prepared:
    settings = get_settings()
    step = step or Step({}, None)
    as_of = opts.as_of or date.today()
    client = EdgarClient(offline=opts.offline)
    with step("Resolving company"):
        company: CompanyRef = resolve(opts.company, client, choose=choose)
    with step("Loading filings and events"):
        filings = endpoints.filings_table(client, company.cik)
    with step("Normalizing XBRL facts"):
        cf = load_companyfacts(client, company.cik)
        store = FactStore(company.cik, cf.raw, as_of)  # holds only filings public by as_of
        CURRENCY.set(store.currency)  # money in narratives and the report is shown in the reporting currency
        profile = for_company(company.sic, company.annual_form, store.currency, store.taxonomy)
    llm = None
    notes: list[str] = []
    if opts.mode != "deterministic":
        if settings.has_llm:
            from ffi.llm import LLM

            llm = LLM(budget_usd=opts.budget_usd, offline=opts.offline)
        else:
            notes.append("No LLM available (no ANTHROPIC_API_KEY and no logged-in Claude Code CLI): ran in deterministic mode "
                         "(LLM detectors and the investigation agent skipped).")
    return Prepared(client, company, filings, cf, store, profile, get_tuning(), as_of, llm, notes)


def analyse_year(prep: Prepared, opts: RunOptions, focus: pd.Timestamp, run_dir: Path, step: Step,
                 kg_file: str = "kg.duckdb") -> YearAnalysis:
    """Peers, filing text, knowledge graph, detectors and scoring for one fiscal year."""
    client, company, filings, store, as_of, tuning = prep.client, prep.company, prep.filings, prep.store, prep.as_of, prep.tuning
    cutoff = point_in_time(focus, filings, as_of)
    with step("Normalizing XBRL facts"):
        if cutoff < as_of:  # only what was public by the end of the year's filing season
            store = store.restricted(cutoff)
        A0, AL0, Q = store.annual("first"), store.annual("latest"), store.quarterly("first")  # memoized per store
    keep = [p for p in A0.values.index if p <= focus][-(opts.years + 3):]
    A = dataclasses.replace(A0, values=A0.values.loc[keep])
    A_latest = dataclasses.replace(AL0, values=AL0.values.loc[[p for p in AL0.values.index if p in keep]])
    # Filing text, earnings releases and events do not depend on the industry data, so they download in the
    # background while the cross-section and peers are built (all traffic shares one SEC rate limiter).
    background = ThreadPoolExecutor(max_workers=3, thread_name_prefix="ffi-prefetch")
    bg = lambda fn, *a: background.submit(contextvars.copy_context().run, fn, *a)  # noqa: E731
    events_fut = bg(load_events, client, company.cik, filings, cutoff, focus)
    docs_fut = bg(load_docs, client, company.cik, filings, list(A.values.index), focus, cutoff, 3, prep.doc_cache)
    releases_fut = bg(load_releases, client, company.cik, filings, focus, cutoff, 6, prep.release_cache)

    peers: PeerSet | None = None
    xsec = pd.DataFrame()
    industry: set[int] = set()
    digits = 2
    with step("Building industry cross-section and peer set"):
        y = calendar_year_of(focus)
        try:
            xsec = build_xsec(client, None, [y - 3, y - 2, y - 1, y])
            active = set(xsec["cik"].unique()) if not xsec.empty else set()
            if company.sic:
                uni, digits = expanding_universe(client, company.sic, tuning["peers"]["min_candidates"], active)
                industry = set(uni) | {company.cik}
                override = None
                if opts.peers:
                    tick = endpoints.company_tickers(client)
                    override = [int(tick.loc[tick["ticker"] == t.strip().upper(), "cik"].iloc[0]) for t in opts.peers
                                if (tick["ticker"] == t.strip().upper()).any()]
                own = {k: A.get(k, focus) for k in ("REVENUE", "NET_INCOME", "CFO", "COGS", "SGA", "OPERATING_INCOME",
                                                     "GROSS_PROFIT", "TOTAL_ASSETS", "AR", "PPE_NET", "INVENTORY")}
                prior = A.periods[-2] if len(A.periods) > 1 else None
                own["REV_PRIOR"] = A.get("REVENUE", prior) if prior is not None else float("nan")
                peers = select_peers(client, company.cik, sorted(industry), digits,
                                     xsec[xsec["cik"].isin(industry)], focus, tuning["peers"]["count"], as_of, override,
                                     target_row=own, size_comparable=store.currency == "USD", cache=prep.peer_cache)
        except EdgarError as e:
            log.warning("peer data unavailable: %s", e)

    with step("Reading annual-report text and earnings releases"):
        try:
            events, docs, releases = events_fut.result(), docs_fut.result(), releases_fut.result()
        finally:
            background.shutdown(wait=False, cancel_futures=True)
        auditor = find_auditor(docs[-1]) if docs else None

    notes = list(prep.notes)
    for (item, p), factor in sorted(A.rescaled.items(), key=lambda kv: kv[0][1]):
        if p in keep:
            notes.append(f"XBRL data quality: {item.replace('_', ' ').lower()} for {A.label(p)} is tagged in "
                         f"{'millions' if factor == 1e6 else 'thousands'}; rescaled using net income / EPS.")
    kg = DuckNXGraphStore(run_dir / kg_file, as_of=cutoff)
    llm = prep.llm
    ctx = RunContext(company=company, store=store, A=A, A_latest=A_latest, Q=Q, filings=filings, focus=focus, as_of=as_of, cutoff=cutoff,
                     kg=kg, peers=peers, xsec=xsec, industry_ciks=industry, industry_digits=digits, docs=docs, releases=releases,
                     events=events, auditor=auditor, llm=llm, mode=opts.mode if llm else "deterministic", tuning=tuning,
                     notes=notes, client=client, profile=prep.profile, currency=store.currency, annual_form=company.annual_form)

    with step("Building knowledge graph"):
        kg_counts = kg_build.build(ctx)

    with step("Running detectors"):
        results = run_detectors(ctx, after={"E.09": kg_build.add_relationships})

    with step("Scoring and materiality"):
        sc = score(ctx, results)
        kg_build.add_signals(ctx, [s for r in results for s in r.signals])
    return YearAnalysis(ctx, results, sc, kg_counts)


def run(opts: RunOptions, choose=None, progress: Callable[[str], None] | None = None,
        agent_ui: Any = None) -> RunOutput:
    timings: dict[str, float] = {}
    step = Step(timings, progress)
    prep = prepare(opts, choose, step)
    company, store = prep.company, prep.store
    focus = pick_focus(store, opts.period)

    run_id = opts.run_id or f"{company.ticker or company.cik}_{store.fiscal_label(focus)}_{uuid.uuid4().hex[:6]}"
    run_dir = Path(opts.out_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    year = analyse_year(prep, opts, focus, run_dir, step)
    ctx, results, sc = year.ctx, year.results, year.score

    agent_cases: list[Any] = []
    if ctx.llm is not None and sc.findings:
        with step("Investigating top findings (agent)"):
            from ffi.agent.runner import investigate

            agent_cases = investigate(ctx, sc, run_dir, run_id, opts, ui=agent_ui)

    with step("Independent audit of findings"):
        audit_findings(ctx, sc)
    with step("Explaining findings in plain language"):
        from ffi.report.explain import explain_findings

        explain_findings(ctx, sc)

    manifest = build_manifest(ctx, results, sc, run_id, opts, timings, year.kg_counts, prep.cf)
    out = RunOutput(ctx, results, sc, run_dir, manifest, timings, agent_cases)
    with step("Rendering report"):
        from ffi.report.render import render, to_json

        out.files = render(out, opts.formats)
    (run_dir / "manifest.json").write_text(to_json(manifest), encoding="utf-8")
    try:
        snapshot.save(out)  # lets `anomaly-report render` rebuild the report without re-running
    except Exception as e:  # noqa: BLE001 - the report is already written; only re-rendering is lost
        log.warning("could not save the run for re-rendering: %s", e)
    if not opts.keep_artifacts:
        _remove_working_files(run_dir, ctx.kg, agent_cases)
    _remove_superseded_reports(run_dir, company.cik)
    return out


def _remove_superseded_reports(run_dir: Path, cik: int) -> None:
    """Keep one final report per company and fiscal year: once this run's report is written, earlier runs for the same
    company and year are deleted. Folders are matched on name (TICKER_FY2025_xxxxxx) and confirmed by the CIK recorded
    in their manifest, so another company's reports are never touched."""
    import shutil

    prefix = run_dir.name.rsplit("_", 1)[0] + "_"
    for old in run_dir.parent.glob(f"{prefix}*"):
        if old == run_dir or not old.is_dir() or len(old.name) != len(run_dir.name):
            continue
        try:
            same = json.loads((old / "manifest.json").read_text(encoding="utf-8")).get("company", {}).get("cik") == cik
        except (OSError, ValueError):
            # no manifest: a run that failed before finishing; remove it unless it somehow holds a report
            same = not any((old / f"report.{ext}").exists() for ext in ("html", "md", "json"))
        if same:
            shutil.rmtree(old, ignore_errors=True)
            log.info("removed superseded report %s", old.name)


def _remove_working_files(run_dir: Path, kg, cases: list[Any]) -> None:
    """The graph database and agent checkpoints are only working files once the report is written (4-7 MB per run).
    Checkpoints are kept when an investigation paused or failed, so `--resume` can continue it."""
    with contextlib.suppress(Exception):
        kg.con.close()
    targets = [run_dir / "kg.duckdb", run_dir / "kg.duckdb.wal"]
    if not any(isinstance(c, dict) and c.get("status") in ("paused", "error", "running") for c in cases):
        targets.append(run_dir / "checkpoints.sqlite")
    for f in targets:
        try:
            f.unlink(missing_ok=True)
        except OSError as e:
            log.debug("could not remove %s: %s", f, e)


def audit_findings(ctx: RunContext, sc: ScoreResult) -> None:
    """Independent audit with a correction loop (max 3 rounds) for every reported finding."""
    from ffi.audit.deterministic import run_audit
    from ffi.scoring.engine import apply_audit, audit_reported, template_narrative

    def one(f) -> None:
        if not f.narrative:
            f.narrative = template_narrative(f)
        f.audit = run_audit(ctx, f, template_narrative, ledger=f.ledger)

    audit_reported(sc.findings, one)
    apply_audit(sc)


def build_manifest(ctx: RunContext, results, sc: ScoreResult, run_id: str, opts: RunOptions, timings, kg_counts, cf) -> dict:
    from ffi.detectors.base import REGISTRY

    src_hash = cf.sha256  # of the downloaded file (re-serialising the parsed JSON took seconds for large filers)
    return {
        "run_id": run_id,
        "package_version": __version__,
        "company": {"cik": ctx.company.cik, "ticker": ctx.company.ticker, "name": ctx.company.name, "sic": ctx.company.sic,
                    "resolution": ctx.company.match_note, "warnings": ctx.company.warnings,
                    "annual_form": ctx.annual_form, "currency": ctx.currency},
        "profile": ({"key": ctx.profile.key, "label": ctx.profile.label, "not_applicable": ctx.profile.excluded,
                     "excluded_ratios": sorted(ctx.profile.excluded_ratios)} if ctx.profile else None),
        "focus_period": ctx.focus_label, "focus_end": str(ctx.focus.date()), "as_of": str(ctx.as_of),
        "options": {k: (str(v) if isinstance(v, Path | date) else v) for k, v in opts.__dict__.items()},
        "mode": ctx.mode,
        "detectors": {r.id: {"version": REGISTRY[r.id].version, "module": REGISTRY[r.id].fn.__module__, "status": r.status, "reason": r.reason,
                             "seconds": round(r.seconds, 3), "flagged": r.flagged} for r in results},
        "peers": {"method": ctx.peers.method if ctx.peers else None,
                  "ciks": [p.cik for p in ctx.peers.loaded] if ctx.peers else []},
        "kg": kg_counts,
        "sources": {"companyfacts_sha256": src_hash,
                    "documents": [{"accession": d.accession, "url": d.url} for d in ctx.docs + ctx.releases]},
        "llm": ({"backend": ctx.llm.backend, "reasoning_model": ctx.llm.reasoning_model, "extraction_model": ctx.llm.extraction_model,
                 "calls": ctx.llm.spend.calls, "usd": round(ctx.llm.spend.usd, 4), "by_purpose": ctx.llm.spend.by_purpose}
                if ctx.llm else None),
        "risk_index": sc.risk_index, "band": sc.band,
        "audit": {f.id: f.audit.label for f in sc.findings if f.audit is not None},
        "timings_s": timings,
    }
