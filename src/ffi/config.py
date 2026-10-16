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
    # auto: the API when ANTHROPIC_API_KEY is set, otherwise the Claude Code CLI (Claude subscription login)
    llm_backend: str = Field(default="auto", alias="FFI_LLM_BACKEND")  # auto | api | claude-code
    claude_cli: str = Field(default="", alias="FFI_CLAUDE_CLI")  # path to the claude executable (optional)

    @property
    def backend(self) -> str | None:
        """The LLM backend this run will use, or None when neither is available."""
        if self.llm_backend == "api":
            return "api" if self.anthropic_api_key else None
        if self.llm_backend == "claude-code":
            return "claude-code" if find_claude_cli(self.claude_cli) else None
        if self.anthropic_api_key:
            return "api"
        return "claude-code" if find_claude_cli(self.claude_cli) else None

    @property
    def has_llm(self) -> bool:
        return self.backend is not None


def find_claude_cli(explicit: str = "") -> str | None:
    """The Claude Code CLI: FFI_CLAUDE_CLI, `claude` on PATH, or the binary bundled with the VS Code extension."""
    import os
    import shutil

    if explicit:
        return explicit if Path(explicit).exists() else None
    on_path = shutil.which("claude")
    if on_path:
        return on_path
    exe = "claude.exe" if os.name == "nt" else "claude"
    found: list[Path] = []
    for ext_dir in (Path.home() / ".vscode" / "extensions", Path.home() / ".cursor" / "extensions"):
        if ext_dir.exists():
            found += ext_dir.glob(f"anthropic.claude-code-*/resources/native-binary/{exe}")
    for p in (Path.home() / ".local" / "bin" / exe, Path.home() / ".claude" / "local" / exe):
        if p.exists():
            found.append(p)
    return str(max(found, key=lambda p: p.stat().st_mtime)) if found else None


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
