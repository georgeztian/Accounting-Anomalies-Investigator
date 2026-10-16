"""Refuse to run outside a Python virtual environment."""

from __future__ import annotations

import os
import sys


def in_virtualenv() -> bool:
    return sys.prefix != getattr(sys, "base_prefix", sys.prefix) or bool(os.environ.get("VIRTUAL_ENV"))


def require_virtualenv() -> None:
    if os.environ.get("FFI_ALLOW_SYSTEM_PYTHON") == "1":  # escape hatch for CI containers only
        return
    if not in_virtualenv():
        sys.stderr.write(
            "anomaly-report must run inside a Python virtual environment.\n"
            "Create one with:  python -m venv .venv  and activate it, then  pip install -e .\n"
        )
        raise SystemExit(2)
