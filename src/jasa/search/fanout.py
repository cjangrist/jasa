"""Parallel provider fan-out, ported from omnisearch ``web_search_fanout``.

Dispatches a query to every active provider in parallel, each wrapped in the
retry layer, fuses results in canonical registry order (the approved §18
divergence from the source's completion order, making aggregation
deterministic), and returns an immutable snapshot.

A caller ``timeout_ms`` is a GLOBAL fan-out deadline (not per-provider). When it
fires, in-flight tasks are cancelled and AWAITED (strictly better than the
source, which abandons them), pending providers are marked failed exactly once
with the verbatim deadline string, and the snapshot is frozen. No provider's
failure short-circuits a sibling. The retry sleep / RNG / clock are injected via
``_FanoutKnobs`` so tests are deterministic and never wait.
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import cast

from jasa.search.providers.base import SearchProvider, SearchRequest
from jasa.search.ranking import SearchResult
from jasa.search.retry import retry_with_backoff
from omnifetch.fetch.shared.types import ProviderError

_PER_PROVIDER_LIMIT = 20
_RETRY_MAX_RETRIES = 1

_RetrySleep = Callable[[float], Awaitable[None]]
_Rng = Callable[[], float]
_Clock = Callable[[], float]


@dataclass(frozen=True, slots=True)
class ProviderSuccess:
    """A provider that completed within the deadline."""

    provider: str
    duration_ms: int


@dataclass(frozen=True, slots=True)
class ProviderFailure:
    """A provider that failed or was still pending at the deadline."""

    provider: str
    error: str
    duration_ms: int


@dataclass(frozen=True, slots=True)
class DispatchResult:
    """Immutable snapshot of a fan-out dispatch."""

    results_by_provider: Mapping[str, list[SearchResult]]
    providers_succeeded: list[ProviderSuccess]
    providers_failed: list[ProviderFailure]


@dataclass(frozen=True, slots=True)
class _Outcome:
    name: str
    succeeded: bool
    results: list[SearchResult]
    error: str
    duration_ms: int


@dataclass(frozen=True, slots=True)
class _FanoutKnobs:
    """Injectable retry sleep / RNG / clock for deterministic tests."""

    retry_sleep: _RetrySleep = asyncio.sleep
    retry_rng: _Rng = random.random
    clock: _Clock = time.monotonic


def _elapsed_ms(start: float, now: float) -> int:
    return int((now - start) * 1000)


def _deadline_message(timeout_ms: int) -> str:
    return f"Timed out (fanout deadline {timeout_ms}ms)"


async def _run_one(
    name: str,
    provider: SearchProvider,
    query: str,
    limit: int,
    knobs: _FanoutKnobs,
) -> _Outcome:
    """Run one provider; absorb any exception as a failure outcome."""
    start = knobs.clock()
    try:
        results = cast(
            "list[SearchResult]",
            await retry_with_backoff(
                lambda: provider.search(
                    SearchRequest(query=query, limit=limit)
                ),
                max_retries=_RETRY_MAX_RETRIES,
                sleep=knobs.retry_sleep,
                rng=knobs.retry_rng,
            ),
        )
    except ProviderError as error:
        return _Outcome(
            name, False, [], str(error), _elapsed_ms(start, knobs.clock())
        )
    except Exception as error:
        return _Outcome(
            name,
            False,
            [],
            f"{type(error).__name__}: {error}",
            _elapsed_ms(start, knobs.clock()),
        )
    return _Outcome(
        name, True, list(results), "", _elapsed_ms(start, knobs.clock())
    )


async def dispatch_to_providers(
    providers: Mapping[str, SearchProvider],
    query: str,
    *,
    per_provider_limit: int = _PER_PROVIDER_LIMIT,
    timeout_ms: int | None = None,
    knobs: _FanoutKnobs | None = None,
) -> DispatchResult:
    """Dispatch ``query`` to all providers in parallel; snapshot the result."""
    resolved_knobs = knobs if knobs is not None else _FanoutKnobs()
    active = list(providers.items())
    if not active:
        return DispatchResult({}, [], [])
    tasks: dict[str, asyncio.Task[_Outcome]] = {
        name: asyncio.create_task(
            _run_one(name, provider, query, per_provider_limit, resolved_knobs)
        )
        for name, provider in active
    }
    try:
        if timeout_ms is not None and timeout_ms > 0:
            await asyncio.wait(tasks.values(), timeout=timeout_ms / 1000)
        else:
            await asyncio.gather(*tasks.values())
    except BaseException:
        for task in tasks.values():
            task.cancel()
        await asyncio.gather(*tasks.values(), return_exceptions=True)
        raise

    results_by_provider: dict[str, list[SearchResult]] = {}
    succeeded: list[ProviderSuccess] = []
    failed: list[ProviderFailure] = []
    for name, _provider in active:
        task = tasks[name]
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                current_task = cast(
                    "asyncio.Task[object]", asyncio.current_task()
                )
                if current_task.cancelling():
                    raise
                deadline = timeout_ms or 0
                failed.append(
                    ProviderFailure(name, _deadline_message(deadline), deadline)
                )
                continue
        outcome = task.result()
        if outcome.succeeded:
            results_by_provider[name] = outcome.results
            succeeded.append(ProviderSuccess(name, outcome.duration_ms))
        else:
            failed.append(
                ProviderFailure(name, outcome.error, outcome.duration_ms)
            )
    return DispatchResult(results_by_provider, succeeded, failed)
