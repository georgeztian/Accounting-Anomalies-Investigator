"""Claude subscription (Claude Code CLI) backend and 10-K layout fallbacks."""

import json
import subprocess

import pytest
from pydantic import BaseModel

from ffi import llm as llm_mod
from ffi.text.document import build_doc


class Out(BaseModel):
    revenue_musd: float
    direction: str


def _cli_llm(monkeypatch, tmp_path, stdout: dict, captured: dict):
    def fake_run(args, **kw):
        captured["args"], captured["env"], captured["input"] = args, kw["env"], kw["input"]
        return subprocess.CompletedProcess(args, 0, stdout=json.dumps(stdout), stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-not-leak")
    obj = llm_mod.LLM.__new__(llm_mod.LLM)
    obj.backend, obj.cli, obj.client = "claude-code", "claude", None
    obj.reasoning_model, obj.extraction_model = "claude-opus-5-5", "claude-haiku-5-5"
    obj.cache_dir, obj.budget_usd, obj.offline = tmp_path, 5.0, False
    obj.spend = llm_mod.Spend()
    import threading
    obj._lock = threading.Lock()
    obj._reserved = 0.0
    return obj


def test_claude_code_backend_returns_validated_output(monkeypatch, tmp_path):
    cap: dict = {}
    obj = _cli_llm(monkeypatch, tmp_path, {"is_error": False, "structured_output": {"revenue_musd": 4512, "direction": "up"},
                                           "total_cost_usd": 0.001, "usage": {"input_tokens": 10, "output_tokens": 5}}, cap)
    res = obj.structured(Out, "sys", "Net sales rose to $4,512 million.", purpose="t")
    assert res.revenue_musd == 4512 and obj.spend.calls == 1
    assert "--json-schema" in cap["args"] and cap["args"][cap["args"].index("--tools") + 1] == ""
    assert "ANTHROPIC_API_KEY" not in cap["env"]  # never billed to an API account
    assert cap["input"].startswith("Net sales")
    # cached: a second identical call does not start the CLI again
    cap.clear()
    obj.structured(Out, "sys", "Net sales rose to $4,512 million.", purpose="t")
    assert not cap


def test_claude_code_backend_errors(monkeypatch, tmp_path):
    obj = _cli_llm(monkeypatch, tmp_path, {"is_error": True, "result": "Claude AI usage limit reached, resets 5pm"}, {})
    with pytest.raises(llm_mod.LLMError, match="usage limit"):
        obj.structured(Out, "sys", "x", purpose="t")
    obj = _cli_llm(monkeypatch, tmp_path, {"is_error": False, "structured_output": {"wrong": 1}}, {})
    with pytest.raises(llm_mod.LLMError, match="no parsed output"):
        obj.structured(Out, "sys", "y", purpose="t")


def _html(lines):
    return "".join(f"<p>{t}</p>" for t in lines)


def test_running_page_headers_do_not_truncate_sections():
    body = ["Item 7. Management's Discussion and Analysis of Financial Condition and Results of Operations"]
    for n in range(30):
        body += [f"MD&A paragraph {n} about results."] + (["Item 7"] if n % 10 == 9 else [])
    body += ["Item 8. Financial Statements and Supplementary Data", "Statements."]
    doc = build_doc("a", "10-K", "2025-12-31", "", _html(body))
    assert len([p for p in doc.section_paragraphs("item7") if "MD&A paragraph" in p.text]) == 30


def test_cross_reference_index_format():
    body = ["Contents", "Management's Discussion and Analysis | 7"]
    body += ["RISK FACTORS"] + [f"Risk {n}." for n in range(40)]
    body += ["MANAGEMENT'S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION AND RESULTS OF OPERATIONS |"]
    body += [f"Results paragraph {n}." for n in range(200)]
    body += ["Consolidated Statement of Income"] + [f"Note text {n}." for n in range(60)]
    body += [f"Item {k} | Title {k} | Page {k}" for k in ("1", "1A", "2", "3", "7", "7A", "8", "9A", "15")]
    doc = build_doc("a", "10-K", "2025-12-31", "", _html(body))
    assert len(doc.section_paragraphs("item7")) == 201
    assert doc.section_paragraphs("item8")[1].text == "Note text 0."
    assert len(doc.section_paragraphs("item1a")) == 41


def test_contents_list_without_page_numbers_is_not_the_mdna_section():
    # Intel's 10-K: Items are listed in an index table at the end, and the contents list at the front gives each title
    # and its page number in separate cells, so "Management's Discussion and Analysis |" looked like the MD&A heading
    # and MD&A came out as 5 words (the tone test then read 200 uncertainty words per 1,000)
    body = ["Our Business | 6", "Management's Discussion and Analysis |", "Operating Segment Results | 21",
            "Consolidated Results of Operations | 25", "Liquidity and Capital Resources | 29", "Properties | 32",
            "Quantitative and Qualitative Disclosures About Market Risk | 33", "Risk Factors", "| 37"]
    body += ["Our business paragraph."] * 30
    body += ["Management's Discussion and Analysis"] + [f"Results paragraph {n}." for n in range(120)]
    body += ["Quantitative and Qualitative Disclosures About Market Risk"] + [f"Market risk paragraph {n}." for n in range(30)]
    body += ["Risk Factors"] + [f"Risk {n}." for n in range(60)]
    body += ["Financial Statements and Supplementary Data"] + [f"Note text {n}." for n in range(60)]
    body += [f"Item {k} | Title {k} | Page {k}" for k in ("1", "1A", "2", "3", "7", "7A", "8", "9A", "15")]
    doc = build_doc("a", "10-K", "2025-12-27", "", _html(body))
    mdna = doc.section_paragraphs("item7")
    assert len(mdna) == 121 and mdna[1].text == "Results paragraph 0."


def test_tone_test_skips_when_mdna_was_not_located():
    from types import SimpleNamespace

    import pytest

    from ffi.detectors.base import SkipDetector
    from ffi.detectors.e_text import e05

    short = build_doc("a", "10-K", "2025-12-27", "", _html(["Item 7. Management's Discussion and Analysis", "See page 21.",
                                                           "Item 8. Financial Statements and Supplementary Data", "Statements."]))
    long = build_doc("b", "10-K", "2024-12-28", "", _html(["Item 7. Management's Discussion and Analysis"]
                                                          + ["Results may decline because of risk and uncertainty. " * 5] * 200
                                                          + ["Item 8. Financial Statements and Supplementary Data", "Statements."]))
    with pytest.raises(SkipDetector, match="too short"):
        e05(SimpleNamespace(doc=short, prior_doc=long))
