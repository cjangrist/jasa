"""Fan-out: parallel dispatch, retry, deadline cancellation, canonical order."""

from __future__ import annotations

import asyncio

import pytest

from jasa.search.fanout import (
    _FanoutKnobs,
    dispatch_to_providers,
    DispatchResult,
)
from jasa.search.providers.base import SearchProvider, SearchRequest
from jasa.search.ranking import SearchResult
from omnifetch.fetch.shared.types import ErrorType, ProviderError


async def _no_sleep(_seconds: float) -> None:
    return None


def _result(provider: str, url: str) -> SearchResult:
    return SearchResult(
        title=url, url=url, snippet="s", source_provider=provider
    )


class FakeProvider(SearchProvider):
    """Controllable provider: ok / error / flaky / slow."""

    name = "fake"
    secret_env = "FAKE_API_KEY"
    base_url = "https://fake.example.com"
    default_timeout_s = 1.0

    def __init__(
        self,
        name: str,
        *,
        ok: list[SearchResult] | None = None,
        error: ProviderError | None = None,
        delay: float = 0.0,
        flaky: bool = False,
    ) -> None:
        self.name = name
        self._ok = ok or []
        self._error = error
        self._delay = delay
        self._flaky = flaky
        self.calls = 0

    async def search(self, request: SearchRequest) -> list[SearchResult]:
        self.calls += 1
        if self._delay:
            await asyncio.sleep(self._delay)
        if self._flaky and self.calls == 1 and self._error is not None:
            raise self._error
        if self._error is not None and not self._flaky:
            raise self._error
        return list(self._ok)


async def test_all_succeed_in_registry_order() -> None:
    alpha = FakeProvider(
        "alpha", ok=[_result("alpha", "https://a.com/1")], delay=0.01
    )
    beta = FakeProvider("beta", ok=[_result("beta", "https://b.com/1")])
    result = await dispatch_to_providers(
        {"alpha": alpha, "beta": beta},
        "q",
        knobs=_FanoutKnobs(retry_sleep=_no_sleep),
    )
    assert list(result.results_by_provider.keys()) == ["alpha", "beta"]
    assert [s.provider for s in result.providers_succeeded] == ["alpha", "beta"]
    assert result.providers_failed == []


async def test_flaky_retries_then_succeeds() -> None:
    p = FakeProvider(
        "p",
        ok=[_result("p", "u")],
        error=ProviderError(ErrorType.PROVIDER_ERROR, "x", "p"),
        flaky=True,
    )
    sleeps: list[float] = []

    async def spy(seconds: float) -> None:
        sleeps.append(seconds)

    result = await dispatch_to_providers(
        {"p": p}, "q", knobs=_FanoutKnobs(retry_sleep=spy)
    )
    assert p.calls == 2
    assert len(sleeps) == 1
    assert [s.provider for s in result.providers_succeeded] == ["p"]


async def test_rate_limit_fails_without_retry() -> None:
    p = FakeProvider(
        "p",
        error=ProviderError(
            ErrorType.RATE_LIMIT, "Rate limit exceeded for p", "p"
        ),
    )
    result = await dispatch_to_providers(
        {"p": p}, "q", knobs=_FanoutKnobs(retry_sleep=_no_sleep)
    )
    assert p.calls == 1
    assert result.providers_succeeded == []
    assert "Rate limit" in result.providers_failed[0].error


async def test_all_fail() -> None:
    a = FakeProvider("a", error=ProviderError(ErrorType.API_ERROR, "e", "a"))
    b = FakeProvider(
        "b", error=ProviderError(ErrorType.PROVIDER_ERROR, "e", "b")
    )
    result = await dispatch_to_providers(
        {"a": a, "b": b}, "q", knobs=_FanoutKnobs(retry_sleep=_no_sleep)
    )
    assert {f.provider for f in result.providers_failed} == {"a", "b"}
    assert result.providers_succeeded == []


async def test_no_providers() -> None:
    assert await dispatch_to_providers({}, "q") == DispatchResult({}, [], [])


async def test_empty_result_is_success() -> None:
    p = FakeProvider("p", ok=[])
    result = await dispatch_to_providers(
        {"p": p}, "q", knobs=_FanoutKnobs(retry_sleep=_no_sleep)
    )
    assert result.results_by_provider["p"] == []
    assert [s.provider for s in result.providers_succeeded] == ["p"]


async def test_retry_budget_is_two_attempts() -> None:
    p = FakeProvider(
        "p", error=ProviderError(ErrorType.PROVIDER_ERROR, "x", "p")
    )
    await dispatch_to_providers(
        {"p": p}, "q", knobs=_FanoutKnobs(retry_sleep=_no_sleep)
    )
    assert p.calls == 2


async def test_deadline_marks_pending_failed() -> None:
    slow = FakeProvider("slow", ok=[_result("slow", "u")], delay=0.05)
    result = await dispatch_to_providers({"slow": slow}, "q", timeout_ms=10)
    assert result.providers_succeeded == []
    failure = result.providers_failed[0]
    assert failure.provider == "slow"
    assert failure.error == "Timed out (fanout deadline 10ms)"
    assert failure.duration_ms == 10
    assert failure.deadline_exceeded is True


async def test_deadline_cancels_inflight_retry_sleep() -> None:
    p = FakeProvider(
        "p", error=ProviderError(ErrorType.PROVIDER_ERROR, "x", "p")
    )

    async def slow_retry(_seconds: float) -> None:
        await asyncio.sleep(0.05)

    result = await dispatch_to_providers(
        {"p": p}, "q", timeout_ms=10, knobs=_FanoutKnobs(retry_sleep=slow_retry)
    )
    assert p.calls == 1
    assert (
        result.providers_failed[0].error == "Timed out (fanout deadline 10ms)"
    )
    assert result.providers_failed[0].deadline_exceeded is True


async def test_non_provider_error_is_isolated() -> None:
    """A provider raising AttributeError must not crash the whole search."""

    class BadProvider(SearchProvider):
        name = "bad"
        secret_env = "BAD"
        base_url = ""
        default_timeout_s = 1.0

        def __init__(self) -> None:
            pass

        async def search(self, request: SearchRequest) -> list[SearchResult]:
            raise AttributeError("item.get on None")

    good = FakeProvider("good", ok=[_result("good", "https://g.com/1")])
    result = await dispatch_to_providers(
        {"bad": BadProvider(), "good": good},
        "q",
        knobs=_FanoutKnobs(retry_sleep=_no_sleep),
    )
    assert [s.provider for s in result.providers_succeeded] == ["good"]
    assert result.providers_failed[0].provider == "bad"
    assert "AttributeError" in result.providers_failed[0].error


async def test_deadline_one_settled_one_pending() -> None:
    fast = FakeProvider("fast", ok=[_result("fast", "u")])
    slow = FakeProvider("slow", ok=[_result("slow", "u")], delay=0.05)
    result = await dispatch_to_providers(
        {"fast": fast, "slow": slow}, "q", timeout_ms=20
    )
    assert [s.provider for s in result.providers_succeeded] == ["fast"]
    assert [f.provider for f in result.providers_failed] == ["slow"]


async def test_cancelling_dispatch_cleans_up_provider_tasks() -> None:
    started = asyncio.Event()
    cancelled = asyncio.Event()

    class BlockingProvider(FakeProvider):
        async def search(self, request: SearchRequest) -> list[SearchResult]:
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
            return []

    dispatch = asyncio.create_task(
        dispatch_to_providers({"p": BlockingProvider("p")}, "q")
    )
    await started.wait()
    dispatch.cancel()
    with pytest.raises(asyncio.CancelledError):
        await dispatch
    assert cancelled.is_set()


async def test_timed_out_provider_await_cancellation_propagates() -> None:
    first_cancellation = asyncio.Event()

    class CancellationDeferringProvider(FakeProvider):
        async def search(self, request: SearchRequest) -> list[SearchResult]:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                first_cancellation.set()
                await asyncio.Event().wait()
            return []

    dispatch = asyncio.create_task(
        dispatch_to_providers(
            {"p": CancellationDeferringProvider("p")}, "q", timeout_ms=1
        )
    )
    await asyncio.wait_for(first_cancellation.wait(), timeout=1)
    dispatch.cancel()
    with pytest.raises(asyncio.CancelledError):
        await dispatch
