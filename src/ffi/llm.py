"""Thin wrapper over the Anthropic SDK: structured outputs, response caching, spend tracking, budgets.

The LLM only reads text and writes prose (BLUEPRINT §0 principle 6). Every call returns a
Pydantic object validated against a schema; numbers in it are re-checked by Python callers.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from ffi.config import find_claude_cli, get_settings

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)

# USD per 1M tokens: (input, output). Cache reads ~0.05x, writes 1.25x of input.
PRICES = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
}
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class LLMError(RuntimeError):
    pass


class LLMRefusal(LLMError):
    pass


class BudgetExceeded(LLMError):
    pass


@dataclass
class Spend:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    usd: float = 0.0
    by_purpose: dict[str, float] = field(default_factory=dict)

    def add(self, purpose: str, usd: float, usage) -> None:
        self.calls += 1
        self.input_tokens += getattr(usage, "input_tokens", 0) or 0
        self.output_tokens += getattr(usage, "output_tokens", 0) or 0
        self.cache_read_tokens += getattr(usage, "cache_read_input_tokens", 0) or 0
        self.usd += usd
        self.by_purpose[purpose] = self.by_purpose.get(purpose, 0.0) + usd


def cost_usd(model: str, usage) -> float:
    pin, pout = PRICES.get(model, PRICES["claude-opus-5-5"])
    inp = getattr(usage, "input_tokens", 0) or 0
    out = getattr(usage, "output_tokens", 0) or 0
    cr = getattr(usage, "cache_read_input_tokens", 0) or 0
    cw = getattr(usage, "cache_creation_input_tokens", 0) or 0
    return (inp * pin + out * pout + cr * pin * 0.05 + cw * pin * 1.25) / 1e6


class LLM:
    """Structured-output client with a disk cache (identical requests are free and reproducible)."""

    def __init__(self, cache_dir: Path | None = None, budget_usd: float = 6.0, offline: bool = False) -> None:
        s = get_settings()
        self.backend = s.backend or ("api" if offline else None)
        if self.backend is None:
            raise LLMError("no LLM available: set ANTHROPIC_API_KEY, or install Claude Code and log in with your Claude subscription")
        self.client = None
        self.cli: str | None = None
        if self.backend == "api":
            import anthropic

            self.client = anthropic.Anthropic(api_key=s.anthropic_api_key or "offline", max_retries=3)
        else:
            self.cli = find_claude_cli(s.claude_cli)
        self.reasoning_model = s.reasoning_model
        self.extraction_model = s.extraction_model
        self.cache_dir = (cache_dir or s.cache_dir) / "llm"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.budget_usd = budget_usd
        self.offline = offline
        self.spend = Spend()
        self._lock = threading.Lock()

    def remaining(self) -> float:
        return self.budget_usd - self.spend.usd

    def _key(self, model: str, system: str, prompt: str, schema: type[BaseModel], effort: str) -> str:
        blob = json.dumps([model, system, prompt, schema.model_json_schema(), effort], sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    def structured(
        self,
        schema: type[T],
        system: str,
        prompt: str,
        *,
        purpose: str,
        model: str | None = None,
        effort: str = "medium",
        max_tokens: int = 16000,
        max_usd: float | None = None,
        timeout_s: float | None = None,
    ) -> T:
        model = model or self.extraction_model
        key = self._key(model, system, prompt, schema, effort)
        path = self.cache_dir / key[:2] / f"{key}.json"
        if path.exists():
            return schema.model_validate_json(path.read_text(encoding="utf-8"))
        if self.offline:
            raise LLMError(f"LLM response not cached (offline): {purpose}")
        projected = (len(system) + len(prompt)) / 3.5 * PRICES.get(model, (4, 20))[0] / 1e6 + max_tokens * 0.25 * PRICES.get(model, (4, 20))[1] / 1e6
        if self.remaining() < projected or (max_usd is not None and projected > max_usd):
            raise BudgetExceeded(f"{purpose}: projected ${projected:.3f} exceeds remaining ${self.remaining():.3f}")

        if self.backend == "claude-code":
            parsed = self._via_claude_code(schema, system, prompt, purpose=purpose, model=model, effort=effort, timeout_s=timeout_s)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(parsed.model_dump_json(), encoding="utf-8")
            return parsed

        import anthropic

        try:
            client = self.client.with_options(timeout=timeout_s) if timeout_s else self.client
            resp = client.beta.messages.parse(
                model=model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": prompt}],
                output_format=schema,
                output_config={"effort": effort},
                cache_control={"type": "ephemeral"},
                betas=[FALLBACK_BETA],
                fallbacks="default",
            )
        except anthropic.BadRequestError as e:
            raise LLMError(f"{purpose}: bad request: {e.message}") from e
        except anthropic.APIStatusError as e:
            raise LLMError(f"{purpose}: API error {e.status_code}") from e
        except anthropic.APITimeoutError as e:
            raise LLMError(f"{purpose}: connection error (timeout after {timeout_s}s)") from e
        except anthropic.APIConnectionError as e:
            raise LLMError(f"{purpose}: connection error") from e
        usd = cost_usd(getattr(resp, "model", model) or model, resp.usage)
        with self._lock:
            self.spend.add(purpose, usd, resp.usage)
        if resp.stop_reason == "refusal":
            raise LLMRefusal(f"{purpose}: model declined ({getattr(resp.stop_details, 'category', None)})")
        if resp.stop_reason == "max_tokens":
            raise LLMError(f"{purpose}: output truncated at max_tokens")
        parsed = resp.parsed_output
        if parsed is None:
            raise LLMError(f"{purpose}: no parsed output")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(parsed.model_dump_json(), encoding="utf-8")
        return parsed

    # ------------------------------------------------------------------ Claude Code (subscription) backend
    def _via_claude_code(self, schema: type[T], system: str, prompt: str, *, purpose: str, model: str, effort: str,
                         timeout_s: float | None) -> T:
        """One headless Claude Code turn (`claude -p`) with structured output and no tools.

        It authenticates with the Claude subscription you are logged in with. ANTHROPIC_API_KEY is removed from the
        child's environment so the call is never billed to an API account. It runs in an empty temporary folder,
        with no tools, MCP servers or skills, so the model only reads the prompt and returns the schema."""
        import os
        import subprocess
        import tempfile

        assert self.cli
        env = {k: v for k, v in os.environ.items() if k not in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")}
        with tempfile.TemporaryDirectory(prefix="ffi-llm-") as tmp:
            sys_file = Path(tmp) / "system.txt"
            sys_file.write_text(system, encoding="utf-8")
            args = [self.cli, "-p", "--output-format", "json", "--json-schema", json.dumps(schema.model_json_schema()),
                    "--system-prompt-file", str(sys_file), "--tools", "", "--model", model, "--effort", effort,
                    "--no-session-persistence", "--strict-mcp-config", "--disable-slash-commands"]
            try:
                r = subprocess.run(args, input=prompt, capture_output=True, text=True, encoding="utf-8", cwd=tmp, env=env,
                                   timeout=timeout_s or 600)
            except subprocess.TimeoutExpired as e:
                raise LLMError(f"{purpose}: connection error (timeout after {timeout_s}s)") from e
            except OSError as e:
                raise LLMError(f"{purpose}: could not start Claude Code ({e})") from e
        try:
            out = json.loads(r.stdout)
        except json.JSONDecodeError as e:
            detail = (r.stderr or r.stdout).strip()[:300]
            if "login" in detail.lower() or "auth" in detail.lower():
                raise LLMError(f"{purpose}: Claude Code is not logged in; run `claude` once and log in with your Claude account") from e
            raise LLMError(f"{purpose}: API error 500 (Claude Code returned no JSON: {detail})") from e
        usage = _Usage(**{k: out.get("usage", {}).get(k, 0) or 0 for k in ("input_tokens", "output_tokens",
                                                                          "cache_read_input_tokens", "cache_creation_input_tokens")})
        with self._lock:
            # subscription calls are not billed per token; the notional API price still drives the run budget
            self.spend.add(purpose, float(out.get("total_cost_usd") or cost_usd(model, usage)), usage)
        if out.get("is_error"):
            msg = str(out.get("result") or out.get("subtype") or "")
            status = out.get("api_error_status")
            if "limit" in msg.lower() and ("usage" in msg.lower() or "reset" in msg.lower()):
                raise LLMError(f"{purpose}: Claude subscription usage limit reached ({msg[:160]})")
            raise LLMError(f"{purpose}: API error {status or 500} ({msg[:200]})")
        if out.get("stop_reason") == "refusal":
            raise LLMRefusal(f"{purpose}: model declined")
        data = out.get("structured_output")
        if data is None:
            raise LLMError(f"{purpose}: no parsed output")
        try:
            return schema.model_validate(data)
        except ValueError as e:
            raise LLMError(f"{purpose}: no parsed output (schema validation failed: {str(e)[:200]})") from e


@dataclass
class _Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
