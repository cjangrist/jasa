"""In-memory cache backend for tests and stdio development.

Process-local; entries respect the TTL via an injectable clock so expiry is
testable. Not shared across replicas (use the redis backend for that).
"""

from __future__ import annotations

import time
from collections.abc import Callable


class MemoryCache:
    """A process-local string store with TTL expiry."""

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        """Bind an injectable clock so expiry is deterministic in tests."""
        self._store: dict[str, tuple[str, float]] = {}
        self._clock = clock

    async def get(self, key: str) -> str | None:
        """Return the value if present and unexpired, else None."""
        entry = self._store.get(key)
        if entry is None:
            return None
        value, expires_at = entry
        if expires_at <= self._clock():
            self._store.pop(key, None)
            return None
        return value

    async def set(self, key: str, value: str, ttl_seconds: int) -> None:
        """Store ``value`` with a TTL relative to the clock."""
        self._store[key] = (value, self._clock() + ttl_seconds)

    async def close(self) -> None:
        """Drop all entries."""
        self._store.clear()
