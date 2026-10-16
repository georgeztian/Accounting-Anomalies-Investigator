"""Bounded thread-pool helpers. EDGAR traffic stays within SEC limits because every client shares one rate limiter."""

from __future__ import annotations

import contextvars
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from typing import TypeVar

T = TypeVar("T")
R = TypeVar("R")


def map_concurrent(fn: Callable[[T], R], items: Iterable[T], workers: int = 6) -> list[R | Exception]:
    """Apply fn to items concurrently, preserving order. Exceptions are returned in place, not raised."""
    seq = list(items)

    def safe(it: T) -> R | Exception:
        try:
            return fn(it)
        except Exception as e:  # noqa: BLE001 - the caller decides per item
            return e

    if len(seq) <= 1 or workers <= 1:
        return [safe(it) for it in seq]
    # each task runs in a copy of the caller's context, so per-run settings (e.g. the reporting currency) carry over
    jobs = [(contextvars.copy_context(), it) for it in seq]
    with ThreadPoolExecutor(max_workers=min(workers, len(seq))) as pool:
        return list(pool.map(lambda job: job[0].run(safe, job[1]), jobs))
