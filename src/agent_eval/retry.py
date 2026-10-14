"""Retry with exponential backoff for LLM API calls.

Model providers enforce per-plan concurrency/rate limits: under parallel
evaluation workers, some requests will come back as 429 even when nothing is
wrong. The harness must absorb that with backoff instead of failing the run.
"""

from __future__ import annotations

import random
import time
from typing import Callable, TypeVar

T = TypeVar("T")

# HTTP statuses worth retrying: rate limits, transient server-side failures.
# 4xx like 400/401/403 are permanent — retrying would just burn quota.
RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}

# Connection-level errors carry no status_code; classify by exception name so
# we don't need to import litellm/openai exception classes (optional deps).
RETRYABLE_NAME_PARTS = (
    "ratelimit",
    "timeout",
    "serviceunavailable",
    "internalserver",
    "apiconnection",
    "connectionerror",
)


def is_retryable(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    if status is not None:
        try:
            return int(status) in RETRYABLE_STATUS
        except (TypeError, ValueError):
            return False
    name = type(exc).__name__.lower()
    return any(part in name for part in RETRYABLE_NAME_PARTS)


def call_with_retries(
    fn: Callable[[], T],
    *,
    max_retries: int = 6,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Call fn, retrying transient failures with exponential backoff + jitter.

    Non-retryable errors (e.g. auth) raise immediately; retryable errors raise
    after `max_retries` failed retries. Total worst-case added latency is
    roughly 2*max_delay*log2-ish seconds — bounded, so CI cannot hang forever.
    """
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as e:
            if attempt >= max_retries or not is_retryable(e):
                raise
            delay = min(max_delay, base_delay * 2**attempt) * (0.5 + random.random())
            sleep(delay)
    raise AssertionError("unreachable")  # pragma: no cover
