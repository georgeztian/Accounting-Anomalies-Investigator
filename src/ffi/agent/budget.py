"""Per-node budgets and retry policies (BLUEPRINT §13.5)."""

from __future__ import annotations

from dataclasses import dataclass

from langgraph.types import RetryPolicy

from ffi.llm import BudgetExceeded, LLMError, LLMRefusal


@dataclass(frozen=True)
class NodeBudget:
    max_input_chars: int  # ~3.5 chars/token; prompts are clipped to this
    max_output_tokens: int
    max_tool_calls: int
    max_wall_seconds: int
    max_usd: float


DEFAULT_BUDGETS: dict[str, NodeBudget] = {
    "hypothesize": NodeBudget(200_000, 4_000, 0, 120, 0.60),
    "plan_evidence": NodeBudget(200_000, 4_000, 12, 120, 0.60),
    "gather": NodeBudget(0, 0, 40, 240, 0.00),
    "test_alternatives": NodeBudget(400_000, 6_000, 0, 180, 1.20),
    "quantify": NodeBudget(0, 0, 20, 30, 0.00),
    "draft": NodeBudget(350_000, 8_000, 0, 180, 1.00),
    "audit": NodeBudget(350_000, 6_000, 0, 180, 0.80),
}


def transient(exc: Exception) -> bool:
    """Retry only transient LLM failures: never budget, refusal or validation errors."""
    if isinstance(exc, BudgetExceeded | LLMRefusal):
        return False
    if isinstance(exc, LLMError):
        msg = str(exc)
        return any(k in msg for k in ("API error 5", "API error 429", "connection error", "no parsed output", "truncated"))
    return isinstance(exc, TimeoutError | ConnectionError)


LLM_RETRY = RetryPolicy(max_attempts=3, initial_interval=2.0, backoff_factor=2.0, max_interval=30.0, jitter=True, retry_on=transient)
ALT_RETRY = RetryPolicy(max_attempts=2, initial_interval=5.0, retry_on=transient)
TOOL_RETRY = RetryPolicy(max_attempts=4, initial_interval=1.0, max_interval=20.0, retry_on=(TimeoutError, ConnectionError, OSError))
NO_RETRY = RetryPolicy(max_attempts=1)
