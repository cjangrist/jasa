"""Retry layer: classification, jitter range, budget, cancellation."""

from __future__ import annotations

import asyncio

import pytest

from jasa.search.retry import (
    _calculate_delay_ms,
    _is_retryable,
    retry_with_backoff,
)
from omnifetch.fetch.shared.types import ErrorType, ProviderError


async def _no_sleep(_seconds: float) -> None:
    return None


async def test_succeeds_first_try() -> None:
    calls: list[int] = []

    async def func() -> str:
        calls.append(1)
        return "ok"

    assert await retry_with_backoff(func, sleep=_no_sleep) == "ok"
    assert len(calls) == 1


async def test_retries_on_provider_error_then_succeeds() -> None:
    calls: list[int] = []
    sleeps: list[float] = []

    async def func() -> str:
        calls.append(1)
        if len(calls) < 2:
            raise ProviderError(ErrorType.PROVIDER_ERROR, "boom", "p")
        return "ok"

    async def sleep(seconds: float) -> None:
        sleeps.append(seconds)

    assert await retry_with_backoff(func, sleep=sleep) == "ok"
    assert len(calls) == 2
    assert len(sleeps) == 1
    assert 2.0 <= sleeps[0] < 4.0


async def test_no_retry_on_rate_limit() -> None:
    calls: list[int] = []

    async def func() -> None:
        calls.append(1)
        raise ProviderError(ErrorType.RATE_LIMIT, "rl", "p")

    with pytest.raises(ProviderError):
        await retry_with_backoff(func, sleep=_no_sleep)
    assert len(calls) == 1


async def test_no_retry_on_api_error() -> None:
    calls: list[int] = []

    async def func() -> None:
        calls.append(1)
        raise ProviderError(ErrorType.API_ERROR, "api", "p")

    with pytest.raises(ProviderError):
        await retry_with_backoff(func, sleep=_no_sleep)
    assert len(calls) == 1


async def test_untyped_error_retries_then_exhausts() -> None:
    calls: list[int] = []

    async def func() -> None:
        calls.append(1)
        raise ValueError("kaboom")

    with pytest.raises(ValueError):
        await retry_with_backoff(func, sleep=_no_sleep, max_retries=1)
    assert len(calls) == 2


async def test_cancellation_propagates_through_sleep() -> None:
    async def func() -> None:
        raise ProviderError(ErrorType.PROVIDER_ERROR, "x", "p")

    async def sleep(_seconds: float) -> None:
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await retry_with_backoff(func, sleep=sleep)


def test_calculate_delay_realized_range() -> None:
    assert _calculate_delay_ms(1, lambda: 0.0) == 2000
    assert _calculate_delay_ms(1, lambda: 0.5) == 3000
    assert _calculate_delay_ms(1, lambda: 0.999) == 3998


def test_is_retryable_classification() -> None:
    assert _is_retryable(ProviderError(ErrorType.PROVIDER_ERROR, "x", "p"))
    assert not _is_retryable(ProviderError(ErrorType.RATE_LIMIT, "x", "p"))
    assert not _is_retryable(ProviderError(ErrorType.API_ERROR, "x", "p"))
    assert not _is_retryable(ProviderError(ErrorType.INVALID_INPUT, "x", "p"))
    assert _is_retryable(ValueError("x"))
