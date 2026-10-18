"""SEC EDGAR HTTP client: User-Agent enforcement, rate limiting, retries and an on-disk cache."""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential_jitter

from ffi.config import get_settings

log = logging.getLogger(__name__)


class EdgarError(RuntimeError):
    pass


class OfflineCacheMiss(EdgarError):
    pass


def _is_transient(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (429, 500, 502, 503, 504)
    return isinstance(exc, (httpx.TransportError, httpx.TimeoutException))


class _RateLimiter:
    def __init__(self, max_per_second: float) -> None:
        self.min_interval = 1.0 / max_per_second
        self._lock = threading.Lock()
        self._last = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            delay = self._last + self.min_interval - now
            if delay > 0:
                time.sleep(delay)
            self._last = time.monotonic()


_LIMITERS: dict[float, _RateLimiter] = {}
_LIMITERS_LOCK = threading.Lock()


def shared_limiter(max_per_second: float) -> _RateLimiter:
    """One limiter per process: concurrent threads and multiple clients together stay under SEC's 10 req/s."""
    with _LIMITERS_LOCK:
        if max_per_second not in _LIMITERS:
            _LIMITERS[max_per_second] = _RateLimiter(max_per_second)
        return _LIMITERS[max_per_second]


class EdgarClient:
    """All SEC traffic goes through here. Responses are cached by URL hash.

    `max_age_s` controls revalidation: None = cache forever (immutable filing documents),
    a number = refetch when the cached copy is older than that (indexes, companyfacts).
    """

    def __init__(self, cache_dir: Path | None = None, offline: bool = False) -> None:
        settings = get_settings()
        self.offline = offline
        self.cache_dir = (cache_dir or settings.cache_dir) / "edgar"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        ua = settings.sec_user_agent.strip()
        if not offline and ("@" not in ua or len(ua) < 8):
            raise EdgarError(
                "SEC_USER_AGENT is not set. SEC requires a descriptive User-Agent with a contact email, "
                'e.g. SEC_USER_AGENT="Jane Doe jane@example.com" in your .env file.'
            )
        self._limiter = shared_limiter(settings.sec_max_rps)
        self._http = httpx.Client(
            headers={"User-Agent": ua or "offline", "Accept-Encoding": "gzip, deflate"},
            timeout=httpx.Timeout(60.0, connect=15.0),
            follow_redirects=True,
        )

    def _path(self, url: str) -> Path:
        h = hashlib.sha256(url.encode()).hexdigest()
        return self.cache_dir / h[:2] / h

    @retry(
        retry=retry_if_exception(_is_transient),
        stop=stop_after_attempt(4),
        wait=wait_exponential_jitter(initial=1, max=20),
        reraise=True,
    )
    def _fetch(self, url: str) -> bytes:
        self._limiter.wait()
        resp = self._http.get(url)
        resp.raise_for_status()
        return resp.content

    def fetch(self, url: str) -> bytes:
        """Download without caching (for responses the caller stores in a processed form)."""
        if self.offline:
            raise OfflineCacheMiss(f"Not in cache (offline mode): {url}")
        try:
            return self._fetch(url)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                raise EdgarError(f"404 Not Found: {url}") from exc
            raise

    def get_bytes(self, url: str, max_age_s: float | None = None) -> bytes:
        path = self._path(url)
        if path.exists():
            fresh = max_age_s is None or (time.time() - path.stat().st_mtime) < max_age_s
            if fresh or self.offline:
                return path.read_bytes()
        if self.offline:
            raise OfflineCacheMiss(f"Not in cache (offline mode): {url}")
        try:
            data = self._fetch(url)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                raise EdgarError(f"404 Not Found: {url}") from exc
            if path.exists():  # stale copy beats nothing
                return path.read_bytes()
            raise
        tmp = path.with_suffix(f".{threading.get_ident()}.tmp")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_bytes(data)
            tmp.replace(path)
        except OSError as e:  # a cache file held open by another process (a sync client, an antivirus scan): not cached
            log.debug("could not cache %s: %s", url, e)
            with contextlib.suppress(OSError):
                tmp.unlink(missing_ok=True)
        return data

    def evict(self, url: str) -> None:
        """Drop a cached response the caller has stored in processed form. A file another process holds open (a sync
        client such as Dropbox, an antivirus scan) stays until a later run: eviction only saves disk space."""
        try:
            self._path(url).unlink(missing_ok=True)
        except OSError as e:
            log.debug("could not evict %s: %s", url, e)

    def get_json(self, url: str, max_age_s: float | None = None) -> Any:
        return json.loads(self.get_bytes(url, max_age_s))

    def get_text(self, url: str, max_age_s: float | None = None) -> str:
        raw = self.get_bytes(url, max_age_s)
        for enc in ("utf-8", "latin-1"):
            try:
                return raw.decode(enc)
            except UnicodeDecodeError:
                continue
        return raw.decode("utf-8", errors="replace")

    @staticmethod
    def sha256(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    def close(self) -> None:
        self._http.close()
