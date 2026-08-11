"""Per-provider retry layer, ported from omnisearch ``retry_with_backoff``.

Retries a coroutine up to ``max_retries`` times with p-retry's
``calculateDelay`` backoff (factor 2, min 2000 ms, max 5000 ms, randomized).
For the single retry the fan-out uses, the realized delay is [2000, 4000) ms --
the 5000 ms cap never binds because factor**0 is 1 and the random multiplier
tops out just under 2.

Only ``ProviderError(PROVIDER_ERROR)`` -- in practice HTTP 5xx -- is retried;
rate-limit, auth, and bad-input errors surface immediately. Cancellation (an
asyncio task cancel from the fan-out deadline) is a ``CancelledError``
(``BaseException``), so it propagates through the sleep and the call and is NOT
caught by the retry classifier -- the deadline aborts an in-flight retry.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable

from omnifetch.fetch.shared.types import ErrorType, ProviderError

_MIN_TIMEOUT_MS = 2000
_MAX_TIMEOUT_MS = 5000
_FACTOR = 2


def _is_retryable(error: BaseException) -> bool:
    """ProviderError retries only for PROVIDER_ERROR; untyped errors retry."""
    if isinstance(error, ProviderError):
        return error.error_type is ErrorType.PROVIDER_ERROR
    return True


def _calculate_delay_ms(retry_index: int, rng: Callable[[], float]) -> int:
    """Replicate p-retry ``calculateDelay`` for the given retry (1-based)."""
    random_multiplier = rng() + 1
    attempt = max(1, retry_index)
    scaled = random_multiplier * _MIN_TIMEOUT_MS * (_FACTOR ** (attempt - 1))
    timeout = int(scaled + 0.5)
    return min(timeout, _MAX_TIMEOUT_MS)


async def retry_with_backoff(
    func: Callable[[], Awaitable[object]],
    *,
    max_retries: int = 1,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    rng: Callable[[], float] = random.random,
) -> object:
    """Run ``func`` with up to ``max_retries`` retries on retryable failures."""
    attempt = 0
    while True:
        try:
            return await func()
        except Exception as error:
            attempt += 1
            if attempt > max_retries or not _is_retryable(error):
                raise
            await sleep(_calculate_delay_ms(attempt, rng) / 1000)
