"""Terminal handling of human-in-the-loop interrupts (BLUEPRINT §13.4)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from typing import Any

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel

# Policy answers used in --mode auto (and when no terminal is attached)
AUTO_POLICY = {
    "review_plan": {"choice": "approve"},
    "review_explained": {"choice": "close"},
    "review_budget": {"choice": "degrade"},
    "review_publish": {"choice": "approve"},
    "review_sensitive": {"choice": "rewrite"},
}

CHOICES = {
    "review_plan": [("a", "approve"), ("e", "edit"), ("s", "skip")],
    "review_explained": [("c", "close"), ("o", "override")],
    "review_budget": [("r", "raise"), ("n", "narrow"), ("d", "degrade")],
    "review_publish": [("a", "approve"), ("e", "edit"), ("x", "drop")],
    "review_sensitive": [("a", "approve"), ("r", "rewrite"), ("x", "remove")],
}


class PauseRun(Exception):
    """Raised to stop the process at a --pause-on interrupt; state is already checkpointed."""


class TerminalUI:
    def __init__(self, console: Console | None = None, mode: str = "auto", pause_on: tuple[str, ...] = ()) -> None:
        self.console = console or Console()
        self.mode = mode
        self.pause_on = set(pause_on)
        self.status = None  # rich Status, paused while prompting

    def decide(self, payload: dict[str, Any]) -> dict[str, Any]:
        kind = payload["interrupt"]
        if kind in self.pause_on:
            raise PauseRun(kind)
        if self.mode != "interactive" or not sys.stdin.isatty():
            d = dict(AUTO_POLICY[kind])
            d["auto"] = True
            return d
        if self.status:
            self.status.stop()
        try:
            return self._ask(kind, payload)
        finally:
            if self.status:
                self.status.start()

    def _ask(self, kind: str, payload: dict[str, Any]) -> dict[str, Any]:
        title = {"review_plan": "Review investigation plan", "review_explained": "Innocent explanation found",
                 "review_budget": "Budget limit", "review_publish": "Approve finding narrative",
                 "review_sensitive": "Sensitive wording"}[kind]
        body = payload.get("summary", "")
        self.console.print(Panel(escape(body), title=escape(f"{title} · {payload.get('finding', '')}"), border_style="yellow"))
        opts = CHOICES[kind]
        prompt = " · ".join(f"[{k}]{v[len(k):] if v.startswith(k) else ' ' + v}" for k, v in opts)
        while True:
            ans = self.console.input(escape(f"{prompt} > ")).strip().lower()[:1]
            m = dict(opts)
            if ans in m:
                choice = m[ans]
                break
        out: dict[str, Any] = {"choice": choice, "auto": False}
        if choice == "edit":
            out["edited"] = self._edit(payload.get("editable", ""))
        elif choice == "override":
            out["reason"] = self.console.input("Reason for overriding: ").strip()
        elif choice == "raise":
            amt = self.console.input("Raise run budget by USD (default 2): ").strip()
            out["amount"] = float(amt) if amt else 2.0
        return out

    def _edit(self, text: str) -> str:
        editor = os.environ.get("EDITOR") or os.environ.get("VISUAL") or ("notepad" if os.name == "nt" else "vi")
        with tempfile.NamedTemporaryFile("w+", suffix=".txt", delete=False, encoding="utf-8") as fh:
            fh.write(text if isinstance(text, str) else json.dumps(text, indent=2))
            path = fh.name
        try:
            subprocess.call([editor, path])
            with open(path, encoding="utf-8") as fh:
                return fh.read()
        finally:
            os.unlink(path)
