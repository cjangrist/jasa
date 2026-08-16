"""Shared deterministic usage-runtime fixtures."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import cast

import httpx

from jasa.usage.runtime import _CACHE_KEY, UsageRuntime
from omnifetch.cache import CacheBackend
from omnifetch.fetch.shared.config import ProviderSecrets


@dataclass(slots=True)
class UsageCache:
    """Small controllable cache implementing the omnifetch protocol."""

    value: object | None = None
    get_error: Exception | None = None
    set_error: Exception | None = None
    set_result: bool = True
    get_calls: int = 0
    set_calls: list[tuple[str, object, int]] = field(default_factory=list)

    async def get(self, key: str) -> object | None:
        """Return or fail one usage-cache read."""
        assert key == _CACHE_KEY
        self.get_calls += 1
        if self.get_error is not None:
            raise self.get_error
        return self.value

    async def set(self, key: str, value: object, ttl_seconds: int) -> bool:
        """Record, retain, or fail one usage-cache write."""
        if self.set_error is not None:
            raise self.set_error
        self.set_calls.append((key, value, ttl_seconds))
        if self.set_result:
            self.value = value
        return self.set_result

    async def delete(self, _key: str) -> bool:
        """Delete the retained value."""
        self.value = None
        return True

    async def is_ready(self) -> bool:
        """Report this deterministic cache as ready."""
        return True

    async def close(self) -> None:
        """Close the no-resource test cache."""
        return None


def build_usage_runtime(
    client: httpx.AsyncClient,
    *,
    secrets: dict[str, str] | None = None,
    cache: UsageCache | None = None,
    clock: Callable[[], float] = lambda: 1_000.0,
    ttl_seconds: int = 600,
) -> UsageRuntime:
    """Build a usage runtime with deterministic secrets, time, and storage."""
    return UsageRuntime(
        client=client,
        cache=cast(CacheBackend, cache or UsageCache()),
        secrets=ProviderSecrets(secrets or {}),
        ttl_seconds=ttl_seconds,
        clock=clock,
    )
