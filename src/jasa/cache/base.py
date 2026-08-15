r"""Cache protocol, key construction, and the complete-fanout write gate.

The 36-hour TTL constant is the direct-call default; composed execution passes
the configured search TTL. The cache key is
``hash_key('search:', query + suffixes)`` via omnifetch's SHA-256 helper, with a
``\0sqf=true`` suffix when the quality filter is skipped (raw mode) and a
``\0gnd=true`` suffix when grounding is active for the call.
``include_snippets`` and ``timeout_ms`` are deliberately NOT in the key.

The write gate caches ONLY a complete fan-out: at least one provider succeeded,
zero failed, and grounding (if active) had no transient failure. A partial or
transient result cached for 36 hours would mask upstream recovery -- the gate is
the poisoning guard.
"""

from __future__ import annotations

from typing import cast, Protocol, runtime_checkable

from omnifetch.fetch.shared.util import hash_key

KEY_PREFIX = "search:"
TTL_SECONDS = 129_600
_RAW_SUFFIX = "\0sqf=true"
_GROUNDED_SUFFIX = "\0gnd=true"


@runtime_checkable
class CacheBackend(Protocol):
    """A get/set/close string store behind one protocol."""

    async def get(self, key: str) -> str | None:
        """Return the stored value, or None on miss / unreadable entry."""

    async def set(self, key: str, value: str, ttl_seconds: int) -> None:
        """Store ``value`` under ``key`` with a TTL; never raise."""

    async def close(self) -> None:
        """Release resources."""


def make_cache_key(
    query: str,
    *,
    skip_quality_filter: bool = False,
    grounding: bool = False,
) -> str:
    """Return the SHA-256 cache key for the query + mode suffixes."""
    value = query
    if skip_quality_filter:
        value += _RAW_SUFFIX
    if grounding:
        value += _GROUNDED_SUFFIX
    return cast(str, hash_key(KEY_PREFIX, value))


def should_cache(
    *,
    providers_succeeded: int,
    providers_failed: int,
    want_grounding: bool,
    transient_failures: int = 0,
) -> bool:
    """Return True only for a complete, non-transient fan-out."""
    grounding_complete = (not want_grounding) or transient_failures == 0
    return (
        providers_succeeded > 0 and providers_failed == 0 and grounding_complete
    )
