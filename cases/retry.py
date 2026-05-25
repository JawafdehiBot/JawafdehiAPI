"""Retry policies with exponential backoff for enrichment pipeline operations."""

from __future__ import annotations

import logging
import random
import time as _time
from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")

logger = logging.getLogger(__name__)

# Default jitter factor (0-20% of wait time).
_JITTER = 0.2


def retry_with_backoff(
    fn: Callable[..., T],
    *args: Any,
    max_retries: int = 3,
    base_seconds: float = 1.0,
    max_seconds: float = 120.0,
    jitter: float = _JITTER,
    retryable_exceptions: tuple[type[Exception], ...] = (Exception,),
    on_retry: Callable[[Exception, int, float], None] | None = None,
    **kwargs: Any,
) -> T:
    """Call *fn(*args, **kwargs)* with exponential backoff.

    Args:
        fn: Callable to invoke.
        max_retries: Maximum number of retries (default 3).
        base_seconds: Initial backoff wait (default 1s).
        max_seconds: Maximum backoff wait (default 120s).
        jitter: Random jitter factor applied to wait time (default 0.2).
        retryable_exceptions: Exception types that trigger a retry.
        on_retry: Optional callback invoked before each retry with
                  (exception, attempt_number, wait_seconds).

    Returns:
        The return value of *fn*.

    Raises:
        The last caught exception after exhausting retries.
    """
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 2):  # 1 initial + N retries
        try:
            return fn(*args, **kwargs)
        except retryable_exceptions as exc:
            last_exc = exc
            if attempt > max_retries:
                raise
            wait = min(base_seconds * (2 ** (attempt - 1)), max_seconds)
            wait *= 1.0 + random.uniform(-jitter, jitter)  # noqa: S311 — not crypto
            if on_retry is not None:
                on_retry(exc, attempt, wait)
            else:
                logger.warning(
                    "Retry %d/%d after %.1fs for %s: %s",
                    attempt,
                    max_retries,
                    wait,
                    fn.__name__,
                    exc,
                )
            _time.sleep(wait)
    # Should not be reached; appease type checker.
    assert last_exc is not None
    raise last_exc


def retryable_http_status(status: int) -> bool:
    """Return True for HTTP status codes that are safe to retry."""
    return status in (429, 502, 503, 504)


def retryable_network_error(exc: Exception) -> bool:
    """Return True for common transient network errors."""
    name = type(exc).__name__
    return name in (
        "Timeout",
        "TimeoutError",
        "ConnectionError",
        "ConnectionResetError",
        "ConnectionRefusedError",
        "RemoteDisconnected",
        "ReadTimeout",
        "ConnectTimeout",
        "ChunkedEncodingError",
    )
