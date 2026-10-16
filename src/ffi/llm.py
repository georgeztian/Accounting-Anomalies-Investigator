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

from ffi.config import get_settings

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
        import anthropic

        s = get_settings()
        if not s.anthropic_api_key and not offline:
            raise LLMError("ANTHROPIC_API_KEY is not set")
        self.client = anthropic.Anthropic(api_key=s.anthropic_api_key or "offline", max_retries=3)
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
