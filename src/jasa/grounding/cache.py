"""Strict success-only cache contract for individual grounding LLM calls.

Keys hash the *page and question* -- the canonical fetch URL plus the query --
together with every setting that can change the answer. They deliberately do
not hash the fetched content.

Content keying is the obvious design and it is the wrong one. The same page
reaches this stage as different bytes routinely: a different provider wins the
fetch race, or the waterfall order changes, and the identical page arrives in
different markdown. Under content keying every accepted snippet for that page
becomes unreachable at that moment -- not stale, not wrong, simply unaddressable
-- and the LLM call that produced it is bought again. Inserting fastCRW ahead of
Firecrawl did exactly that to the whole cache.

The URL is canonicalized with ``omnifetch.tools.fetch.cache_identity_url``, the
same function the fetch cache keys on, so both caches agree on which spellings
are one page rather than each deciding separately.

What content keying bought for free was invalidation: new bytes, new key. That
is replaced by a bounded lifetime, and the bound has to respect omnifetch's own
judgment about which pages go stale fast. A homepage's fetch entry lives five
minutes because it is a rolling index; letting a snippet written from it live a
full day would pin a masthead that the layer underneath deliberately refuses to
hold. ``grounding_cache_ttl_seconds`` therefore clamps volatile URLs to the
same short lifetime. For an ordinary page the fetch entry outlives the snippet
many times over, so URL keying adds no staleness the pipeline did not already
have.

Values bind only the irreversible identity digest to the accepted snippet, so
queries, URLs, fetched content, titles, and prompts are not retained. Reads and
writes fail open without exposing request or key material.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import cast, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from jasa.cache.base import CacheBackend
from jasa.grounding.detectors import (
    detect_grounded_sentinel,
    FENCE_REPAIR_SUFFIX,
    grounding_detector_semantics,
    repair_unbalanced_fence,
)
from jasa.grounding.prompts import (
    GROUNDING_MAX_TOKENS,
    grounding_prompt_semantics,
    SNIPPET_MAX_CHARS,
)
from jasa.grounding.waterfall import (
    grounding_chain_semantics,
    GroundingChain,
)
from jasa.logging import get_logger
from jasa.observability.metrics import emit_grounding_cache_metric
from omnifetch.fetch.shared.util import hash_key
from omnifetch.tools.fetch import is_volatile_fetch_url

_LOGGER = get_logger("grounding.cache")

MIN_SNIPPET_CHARS = 1
TEMPERATURE = 0.2
TOP_P = 0.9
FREQUENCY_PENALTY = 0.3
GROUNDING_CACHE_KEY_PREFIX = "jasa:grounding:v2:"
GROUNDING_CACHE_SEMANTICS_VERSION: Literal[3] = 3
_GROUNDING_CACHE_SCHEMA_VERSION: Literal[2] = 2
_STRICT_RECORD_CONFIG = ConfigDict(extra="forbid", strict=True, frozen=True)

GroundingCacheEvent = Literal[
    "hit",
    "miss",
    "write",
    "read_skipped",
    "write_skipped",
    "read_error",
    "write_error",
    "coalesced",
]


@dataclass(frozen=True, slots=True)
class GroundingCacheIdentity:
    """Every effective input that can change an accepted LLM snippet.

    ``url`` and ``query`` replace the effective user message. They are what the
    snippet is *about*; the fetched bytes are one rendering of the first, and
    keying on a rendering discards reusable work every time the rendering
    changes. ``url`` must be the canonical fetch identity, so the two caches
    partition traffic identically.

    The query stays in the identity because a grounded snippet is written to
    answer it: the same page asked two different questions must not collapse
    onto one entry. Sharing the fetch key outright would do exactly that.

    ``prompt_fingerprint`` carries the prompt template, truncation marker,
    system-prompt digest, and content cap. Those all used to reach the key
    implicitly, through the message text they shaped; with the message gone
    they have to be named, or changing the cap would silently reuse snippets
    written from a differently truncated page.

    ``llm_chain`` is the whole ordered waterfall rather than the one tier that
    happened to answer, because the chain is the unit of substitutability: any
    tier in it may serve a request, so its accepted output is reusable for the
    identical request and a swapped chain starts a fresh namespace.

    The fetched title is absent by design. It is derived from content, so
    including it would reintroduce exactly the content sensitivity this
    identity exists to remove -- a page that changed only its title would
    re-buy every snippet written from it.
    """

    url: str
    query: str
    prompt_fingerprint: str
    llm_chain: tuple[tuple[str, str], ...]
    temperature: float
    top_p: float
    frequency_penalty: float
    max_tokens: int
    postprocess_fingerprint: str
    semantics_version: Literal[3]


@dataclass(frozen=True, slots=True)
class GroundingCacheWrite:
    """One accepted LLM result awaiting a fail-open cache write."""

    key: str
    identity: GroundingCacheIdentity
    snippet: str


class _GroundingCacheRecord(BaseModel):
    """Versioned envelope binding exact LLM inputs to accepted output."""

    model_config = _STRICT_RECORD_CONFIG

    schema_version: Literal[2]
    identity_digest: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )
    snippet: str = Field(
        min_length=MIN_SNIPPET_CHARS,
        max_length=SNIPPET_MAX_CHARS + len(FENCE_REPAIR_SUFFIX),
    )


def _grounding_postprocess_fingerprint() -> str:
    """Hash accepted-output bounds and detector/repair semantics."""
    identity = {
        "detectors": grounding_detector_semantics(),
        "min_snippet_chars": MIN_SNIPPET_CHARS,
        "snippet_max_chars": SNIPPET_MAX_CHARS,
    }
    canonical = json.dumps(identity, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _grounding_prompt_fingerprint(max_content_chars: int) -> str:
    """Hash everything that shapes the message built from a page.

    The content cap belongs here rather than beside the page: it decides how
    much of the page the model was shown, so a snippet written under one cap is
    not reusable under another. It used to reach the key by truncating the
    message that was hashed; with the message out of the identity that path is
    gone and the cap has to be named explicitly.
    """
    identity = {
        "max_content_chars": max_content_chars,
        "prompts": grounding_prompt_semantics(),
    }
    canonical = json.dumps(identity, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def grounding_cache_identity(
    url: str,
    query: str,
    max_content_chars: int,
    chain: GroundingChain,
) -> GroundingCacheIdentity:
    """Build the API-key-free identity for one page-and-query snippet.

    ``url`` must already be the canonical fetch identity; this function does
    not canonicalize, because doing so here would let a caller pass a raw URL
    and silently key apart from the fetch entry for the same page.
    """
    return GroundingCacheIdentity(
        url=url,
        query=query,
        prompt_fingerprint=_grounding_prompt_fingerprint(max_content_chars),
        llm_chain=grounding_chain_semantics(chain),
        temperature=TEMPERATURE,
        top_p=TOP_P,
        frequency_penalty=FREQUENCY_PENALTY,
        max_tokens=GROUNDING_MAX_TOKENS,
        postprocess_fingerprint=_grounding_postprocess_fingerprint(),
        semantics_version=GROUNDING_CACHE_SEMANTICS_VERSION,
    )


def grounding_cache_ttl_seconds(
    url: str,
    configured_seconds: int,
    volatile_seconds: int,
) -> int:
    """Return how long a snippet written from this URL may be reused.

    Keying on the page instead of its bytes gives up the free invalidation that
    content keying provided, so the lifetime has to carry that weight. A
    homepage is the case where it matters: omnifetch holds its fetched content
    for minutes because it is a rolling index, and a snippet written from that
    index is exactly as perishable. Reusing one for the configured day would
    republish a masthead the fetch layer had already thrown away twice over.

    The clamp only ever shortens. An operator who sets a volatile lifetime
    longer than the ordinary one means everything to be fresher, not homepages
    to become the most durable entries in the cache -- the same reading
    omnifetch applies to its own pair of fetch TTLs.
    """
    if not is_volatile_fetch_url(url):
        return configured_seconds
    return min(volatile_seconds, configured_seconds)


def make_grounding_cache_key(identity: GroundingCacheIdentity) -> str:
    """Return a hash-only key for the canonical grounding identity."""
    canonical = json.dumps(
        asdict(identity),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return cast(str, hash_key(GROUNDING_CACHE_KEY_PREFIX, canonical))


def _identity_digest(identity: GroundingCacheIdentity) -> str:
    """Return the irreversible digest shared by the key and value envelope."""
    return make_grounding_cache_key(identity).removeprefix(
        GROUNDING_CACHE_KEY_PREFIX
    )


def _serialize_grounding_cache(
    identity: GroundingCacheIdentity,
    snippet: str,
) -> str:
    """Serialize one accepted grounding result into the strict v2 envelope."""
    record = _GroundingCacheRecord(
        schema_version=_GROUNDING_CACHE_SCHEMA_VERSION,
        identity_digest=_identity_digest(identity),
        snippet=snippet,
    )
    return record.model_dump_json()


def _deserialize_grounding_cache(
    record: object,
    identity: GroundingCacheIdentity,
) -> str | None:
    """Return a validated cached snippet or None for incompatible data."""
    try:
        cached = _GroundingCacheRecord.model_validate(record)
    except ValidationError:
        return None
    if cached.identity_digest != _identity_digest(identity):
        return None
    if repair_unbalanced_fence(cached.snippet) != cached.snippet:
        return None
    if detect_grounded_sentinel(cached.snippet) is not None:
        return None
    return cached.snippet


def record_grounding_cache_event(
    event: GroundingCacheEvent,
    error_type: str | None = None,
) -> None:
    """Log one bounded grounding-cache event without request material."""
    if error_type is None:
        _LOGGER.debug("Grounding cache event=%s", event)
        emit_grounding_cache_metric(event=event)
        return
    _LOGGER.warning("Grounding cache event=%s error_type=%s", event, error_type)
    emit_grounding_cache_metric(event=event, error_type=error_type)


async def read_grounding_cache(
    cache: CacheBackend,
    key: str,
    identity: GroundingCacheIdentity,
) -> str | None:
    """Read and strictly validate one grounding cache entry fail-open."""
    try:
        raw = await cache.get(key)
    except Exception as error:
        record_grounding_cache_event("read_error", type(error).__name__)
        return None
    if raw is None or not isinstance(raw, str | bytes | bytearray):
        record_grounding_cache_event("miss")
        return None
    try:
        record = _GroundingCacheRecord.model_validate_json(raw)
    except ValidationError:
        record_grounding_cache_event("miss")
        return None
    snippet = _deserialize_grounding_cache(record, identity)
    record_grounding_cache_event("hit" if snippet is not None else "miss")
    return snippet


async def write_grounding_cache(
    cache: CacheBackend,
    pending: GroundingCacheWrite,
    ttl_seconds: int,
) -> None:
    """Store one accepted grounding result without failing that result."""
    try:
        value = _serialize_grounding_cache(
            pending.identity,
            pending.snippet,
        )
        stored = await cache.set(pending.key, value, ttl_seconds)
    except Exception as error:
        record_grounding_cache_event("write_error", type(error).__name__)
        return
    if stored is False:
        record_grounding_cache_event("write_error", "BackendRejected")
        return
    record_grounding_cache_event("write")
