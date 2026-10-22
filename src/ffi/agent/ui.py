"""Terminal handling of human-in-the-loop interrupts."""

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


def stdin_interactive() -> bool:
    """True only when a person can answer prompts on stdin.

    On Windows `isatty()` is also True for the NUL device (what IDE agents and schedulers attach), so the
    console handle is checked with GetConsoleMode, which fails for NUL, pipes and files.
    """
    try:
        if sys.stdin is None or not sys.stdin.isatty():
            return False
        if os.name != "nt":
            return True
        import ctypes
        import msvcrt
        from ctypes import wintypes

        mode = wintypes.DWORD()
        return bool(ctypes.windll.kernel32.GetConsoleMode(msvcrt.get_osfhandle(sys.stdin.fileno()), ctypes.byref(mode)))
    except (OSError, ValueError, AttributeError):
        return False


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
        if self.mode != "interactive" or not stdin_interactive():
            return self._auto(kind)
        if self.status:
            self.status.stop()
        try:
            return self._ask(kind, payload)
        except EOFError:
            # input closed mid-run: answer this and every later interrupt by policy instead of failing the case
            self.mode = "auto"
            self.console.print("[dim]Input closed: continuing with --mode auto.[/dim]")
            return self._auto(kind)
        finally:
            if self.status:
                self.status.start()

    @staticmethod
    def _auto(kind: str) -> dict[str, Any]:
        return {**AUTO_POLICY[kind], "auto": True}

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
            try:
                out["amount"] = max(0.0, float(amt)) if amt else 2.0
            except ValueError:  # a typo must not fail the case
                out["amount"] = 2.0
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
