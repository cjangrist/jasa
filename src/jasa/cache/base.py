r"""Cache protocol, v4 search identity, and complete-fanout write gate.

Search keys hash canonical JSON containing the exact query, quality-filter and
grounding modes, ordered active providers, and grounding semantics fingerprint.
The versioned namespace invalidates legacy identities without a destructive
cache clear. ``include_snippets`` and ``timeout_ms`` remain outside the identity
because the full result is cached before transport shaping.

``KEY_PREFIX`` moves with the record schema version rather than independently.
During a rolling deploy two generations share one backend, so a schema bump
alone would leave both addressing the same keys while each rejects the other's
records -- every request would miss, overwrite, and re-run a paid fan-out.
Separate namespaces let the generations coexist and let the old entries expire
on their own TTL.

The write gate caches ONLY a complete fan-out: at least one provider succeeded,
zero failed, and grounding (if active) had no transient failure. A partial or
transient result cached for the configured TTL would mask upstream recovery --
the gate is the poisoning guard.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import cast, Protocol, runtime_checkable

from omnifetch.fetch.shared.util import hash_key

KEY_PREFIX = "jasa:search:v4:"
TTL_SECONDS = 129_600


@dataclass(frozen=True, slots=True)
class SearchCacheIdentity:
    """Every semantic input that can change a complete cached search."""

    query: str
    skip_quality_filter: bool
    grounding: bool
    providers: tuple[str, ...]
    grounding_fingerprint: str | None


@runtime_checkable
class CacheBackend(Protocol):
    """A get/set/close string store behind one protocol."""

    async def get(self, key: str) -> object | None:
        """Return the stored value, or None on miss / unreadable entry."""

    async def set(self, key: str, value: str, ttl_seconds: int) -> bool | None:
        """Store with a TTL; false reports a fail-open backend rejection."""

    async def close(self) -> None:
        """Release resources."""


def make_cache_key(identity: SearchCacheIdentity) -> str:
    """Return a hash-only key for the canonical search identity."""
    canonical = json.dumps(
        asdict(identity),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return cast(str, hash_key(KEY_PREFIX, canonical))


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
