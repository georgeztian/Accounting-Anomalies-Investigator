"""Runtime settings: environment variables (.env) plus optional ffi.toml overrides."""

from __future__ import annotations

import tomllib
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(str(PROJECT_ROOT / ".env"), ".env"), env_file_encoding="utf-8", extra="ignore"
    )

    sec_user_agent: str = Field(default="", alias="SEC_USER_AGENT")
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    cache_dir: Path = Field(default=PROJECT_ROOT / ".cache", alias="FFI_CACHE_DIR")
    reasoning_model: str = Field(default="claude-opus-5-5", alias="FFI_REASONING_MODEL")
    extraction_model: str = Field(default="claude-haiku-5-5", alias="FFI_EXTRACTION_MODEL")
    sec_max_rps: float = 8.0

    @property
    def has_llm(self) -> bool:
        return bool(self.anthropic_api_key)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


DEFAULT_TUNING: dict[str, Any] = {
    "peers": {"count": 15, "min_candidates": 30},
    "materiality": {"pretax_pct": 0.05, "revenue_pct": 0.005, "assets_pct": 0.005, "equity_pct": 0.01},
    "scoring": {"fdr_q": 0.10, "flag_severity": 0.5},
    "agent": {"max_cases": 3, "run_budget_usd": 6.0, "recursion_limit": 40},
}


@lru_cache(maxsize=1)
def get_tuning() -> dict[str, Any]:
    """DEFAULT_TUNING deep-merged with ffi.toml at the project root, if present."""
    merged = {k: dict(v) for k, v in DEFAULT_TUNING.items()}
    path = PROJECT_ROOT / "ffi.toml"
    if path.exists():
        with path.open("rb") as fh:
            user = tomllib.load(fh)
        for section, values in user.items():
            merged.setdefault(section, {}).update(values)
    return merged
