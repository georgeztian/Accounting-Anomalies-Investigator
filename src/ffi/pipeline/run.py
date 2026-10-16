"""End-to-end run for one company (BLUEPRINT §3): resolve -> fetch -> normalize -> KG -> detect -> score -> agent -> report."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from ffi import __version__
from ffi.config import get_settings, get_tuning
from ffi.context import Event, RunContext
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
from ffi.resolve.resolver import CompanyRef, resolve
from ffi.scoring.engine import ScoreResult, score
from ffi.text.document import TextDoc, build_doc, find_auditor
from ffi.xbrl.facts import FactStore

log = logging.getLogger(__name__)

EVENT_ITEMS = {"4.01": "AUDITOR_CHANGE", "4.02": "NON_RELIANCE", "5.02": "OFFICER_CHANGE", "2.01": "ACQUISITION"}


@dataclass
class RunOptions:
    company: str
    years: int = 5
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
    def __init__(self, timings: dict[str, float], progress: Callable[[str], None] | None) -> None:
        self.timings, self.progress = timings, progress

    def __call__(self, name: str):
        timings, progress = self.timings, self.progress

        class _Ctx:
            def __enter__(self_inner):
                self_inner.t = time.perf_counter()
                if progress:
                    progress(name)
                return self_inner

            def __exit__(self_inner, *exc):
                timings[name] = round(time.perf_counter() - self_inner.t, 2)
                return False

        return _Ctx()


def load_events(client: EdgarClient, cik: int, filings, as_of: date, focus: pd.Timestamp) -> list[Event]:
    from ffi.concurrency import map_concurrent

    lo = (focus - pd.Timedelta(days=4 * 366)).date()
    out: list[Event] = []
    fetched = 0
    officer = [f for f in filings if f.form == "8-K" and "5.02" in (f.items or "").split(",") and lo <= f.filing_date <= as_of][-8:]
    map_concurrent(lambda f: client.get_text(endpoints.document_url(cik, f)), officer, workers=4)  # warm the cache in parallel
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
                    if item == "5.02" and fetched < 8:
                        try:
                            txt = client.get_text(url)
                            fetched += 1
                            desc += _officer_change_kind(txt)
                        except EdgarError:
                            pass
                    out.append(Event(f"{f.accession}:{item}", etype, f.filing_date, f.accession, desc, url))
    return out


OFFICER_RE = re.compile(
    r"(?is)(chief financial officer|principal financial officer|chief accounting officer|principal accounting officer|chief executive officer)"
    r".{0,200}?\b(resign\w*|retire\w*|depart\w*|step(?:s|ped)? down|terminat\w*|succeed\w*|successor|appoint\w*|transition\w*)"
    r"|\b(resign\w*|retire\w*|depart\w*|step(?:s|ped)? down|terminat\w*|appoint\w*)\b.{0,200}?"
    r"(chief financial officer|principal financial officer|chief accounting officer|principal accounting officer|chief executive officer)"
)


def _officer_change_kind(raw_html: str) -> str:
    """Describe which senior officer an Item 5.02 8-K changes (compensation-only 5.02s mention none)."""
    from ffi.text.document import html_to_lines

    text = " ".join(line for line, _ in html_to_lines(raw_html))[:20000]
    roles = set()
    for m in OFFICER_RE.finditer(text):
        role = (m.group(1) or m.group(4) or "").lower()
        if "financial" in role or "accounting" in role:
            roles.add("CFO/CAO change")
        elif "executive" in role:
            roles.add("CEO change")
    return (" - " + ", ".join(sorted(roles))) if roles else " - no senior finance/CEO departure or appointment language"


def load_docs(client: EdgarClient, cik: int, filings, store_periods: list[pd.Timestamp], focus: pd.Timestamp, as_of: date, n: int = 3) -> list[TextDoc]:
    from ffi.concurrency import map_concurrent

    tenks = [f for f in filings if f.form in ANNUAL_TEXT_FORMS and f.filing_date <= as_of and f.report_date]
    wanted = [p for p in store_periods if p <= focus][-n:]
    picks = []
    for p in wanted:
        f = min(tenks, key=lambda x: abs((pd.Timestamp(x.report_date) - p).days), default=None)
        if f is not None and abs((pd.Timestamp(f.report_date) - p).days) <= 15:
            picks.append(f)

    def load(f):
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
            log.warning("10-K text %s unavailable: %s", f.accession, d)
        else:
            docs.append(d)
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


def load_releases(client: EdgarClient, cik: int, filings, focus: pd.Timestamp, as_of: date, n: int = 6) -> list[TextDoc]:
    lo = (focus - pd.Timedelta(days=430)).date()
    from ffi.concurrency import map_concurrent

    cands = [f for f in filings if f.form == "8-K" and "2.02" in (f.items or "") and lo <= f.filing_date <= as_of][-n:]

    def load(f):
        rows = endpoints.filing_index(client, cik, f)
        ex = next((r for r in rows if r["type"].upper().startswith("EX-99")), None)
        if not ex or not ex["document"]:
            return None
        url = endpoints.document_url(cik, f, ex["document"])
        return build_doc(f.accession, "EX-99", str(f.report_date or f.filing_date), url, client.get_text(url))

    return [d for d in map_concurrent(load, cands, workers=4) if d is not None and not isinstance(d, Exception)]


def pick_focus(store: FactStore, period: str | None, as_of: date) -> pd.Timestamp:
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


def run(opts: RunOptions, choose=None, progress: Callable[[str], None] | None = None,
        agent_ui: Any = None) -> RunOutput:
    settings = get_settings()
    tuning = get_tuning()
    timings: dict[str, float] = {}
    step = Step(timings, progress)
    as_of = opts.as_of or date.today()
    client = EdgarClient(offline=opts.offline)

    with step("Resolving company"):
        company: CompanyRef = resolve(opts.company, client, choose=choose)
    with step("Loading filings and events"):
        filings = endpoints.filings_table(client, company.cik)
    with step("Normalizing XBRL facts"):
        cf = endpoints.companyfacts(client, company.cik)
        store = FactStore.from_companyfacts(company.cik, cf, as_of)
        CURRENCY.set(store.currency)  # money in narratives and the report is shown in the reporting currency
        profile = for_company(company.sic, company.annual_form, store.currency, store.taxonomy)
        focus = pick_focus(store, opts.period, as_of)
        A, A_latest, Q = store.annual("first"), store.annual("latest"), store.quarterly("first")
        keep = [p for p in A.values.index if p <= focus][-(opts.years + 3):]
        A.values = A.values.loc[keep]
        A_latest.values = A_latest.values.loc[[p for p in A_latest.values.index if p in keep]]
        events = load_events(client, company.cik, filings, as_of, focus)

    run_id = opts.run_id or f"{company.ticker or company.cik}_{store.fiscal_label(focus)}_{uuid.uuid4().hex[:6]}"
    run_dir = Path(opts.out_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

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
                                     target_row=own, size_comparable=store.currency == "USD")
        except EdgarError as e:
            log.warning("peer data unavailable: %s", e)

    with step("Parsing 10-K text and earnings releases"):
        docs = load_docs(client, company.cik, filings, list(A.values.index), focus, as_of)
        releases = load_releases(client, company.cik, filings, focus, as_of)
        auditor = find_auditor(docs[-1]) if docs else None

    llm = None
    notes: list[str] = []
    if opts.mode != "deterministic":
        if settings.has_llm:
            from ffi.llm import LLM

            llm = LLM(budget_usd=opts.budget_usd, offline=opts.offline)
        else:
            notes.append("No LLM available (no ANTHROPIC_API_KEY and no logged-in Claude Code CLI): ran in deterministic mode "
                         "(LLM detectors and the investigation agent skipped).")

    for (item, p), factor in sorted(A.rescaled.items(), key=lambda kv: kv[0][1]):
        notes.append(f"XBRL data quality: {item.replace('_', ' ').lower()} for {A.label(p)} is tagged in "
                     f"{'millions' if factor == 1e6 else 'thousands'}; rescaled using net income / EPS.")
    kg = DuckNXGraphStore(run_dir / "kg.duckdb", as_of=as_of)
    ctx = RunContext(company=company, store=store, A=A, A_latest=A_latest, Q=Q, filings=filings, focus=focus, as_of=as_of,
                     kg=kg, peers=peers, xsec=xsec, industry_ciks=industry, industry_digits=digits, docs=docs, releases=releases,
                     events=events, auditor=auditor, llm=llm, mode=opts.mode if llm else "deterministic", tuning=tuning,
                     notes=notes, client=client, profile=profile, currency=store.currency, annual_form=company.annual_form)

    with step("Building knowledge graph"):
        kg_counts = kg_build.build(ctx)

    with step("Running detectors"):
        results = run_detectors(ctx, after={"E.09": kg_build.add_relationships})

    with step("Scoring and materiality"):
        sc = score(ctx, results)
        kg_build.add_signals(ctx, [s for r in results for s in r.signals])

    agent_cases: list[Any] = []
    if llm is not None and sc.findings:
        with step("Investigating top findings (agent)"):
            from ffi.agent.runner import investigate

            agent_cases = investigate(ctx, sc, run_dir, run_id, opts, ui=agent_ui)

    with step("Independent audit of findings"):
        audit_findings(ctx, sc)

    manifest = build_manifest(ctx, results, sc, run_id, opts, timings, kg_counts, cf)
    out = RunOutput(ctx, results, sc, run_dir, manifest, timings, agent_cases)
    with step("Rendering report"):
        from ffi.report.render import render

        out.files = render(out, opts.formats)
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return out


def audit_findings(ctx: RunContext, sc: ScoreResult, limit: int = 12) -> None:
    """Independent audit with a correction loop (max 3 rounds) for every reported finding."""
    from ffi.audit.deterministic import run_audit
    from ffi.scoring.engine import apply_audit, template_narrative

    for f in sc.findings[:limit]:
        if not f.narrative:
            f.narrative = template_narrative(f)
        f.audit = run_audit(ctx, f, template_narrative, ledger=f.ledger)
    apply_audit(sc)


def build_manifest(ctx: RunContext, results, sc: ScoreResult, run_id: str, opts: RunOptions, timings, kg_counts, cf) -> dict:
    from ffi.detectors.base import REGISTRY

    src_hash = hashlib.sha256(json.dumps(cf, sort_keys=True).encode()).hexdigest()
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
