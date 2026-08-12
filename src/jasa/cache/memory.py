"""In-memory cache backend for tests and stdio development.

Process-local; entries respect the TTL via an injectable clock so expiry is
testable. Not shared across replicas (use the redis backend for that).
"""

from __future__ import annotations

import time
from collections.abc import Callable

DEFAULT_MAX_ENTRIES = 1_000


class MemoryCache:
    """A process-local string store with TTL expiry."""

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.monotonic,
        max_entries: int = DEFAULT_MAX_ENTRIES,
    ) -> None:
        """Bind a clock and maximum entry count for deterministic eviction."""
        if max_entries < 1:
            raise ValueError("max_entries must be at least 1")
        self._store: dict[str, tuple[str, float]] = {}
        self._clock = clock
        self._max_entries = max_entries

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
        """Store with TTL, evicting expired then oldest entries as needed."""
        now = self._clock()
        expired = [
            stored_key
            for stored_key, (_, expires_at) in self._store.items()
            if expires_at <= now
        ]
        for stored_key in expired:
            self._store.pop(stored_key, None)
        if key not in self._store and len(self._store) >= self._max_entries:
            self._store.pop(next(iter(self._store)))
        self._store[key] = (value, now + ttl_seconds)

    async def close(self) -> None:
        """Drop all entries."""
        self._store.clear()
