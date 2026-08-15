"""Strict success-only cache contract for individual grounding LLM calls.

Keys hash the exact effective LLM request and post-processing semantics. Values
bind only that irreversible digest to the accepted snippet and fetched title in
a strict v1 envelope, so queries, fetched content, and prompts are not retained.
Reads and writes fail open without exposing request or key material.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import cast, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from jasa.cache.base import CacheBackend
from jasa.config import GroundingSettings
from jasa.grounding.detectors import (
    detect_grounded_sentinel,
    FENCE_REPAIR_SUFFIX,
    grounding_detector_semantics,
    repair_unbalanced_fence,
)
from jasa.grounding.prompts import (
    GROUNDING_MAX_TOKENS,
    SNIPPET_MAX_CHARS,
    SYSTEM_PROMPT_SHA256,
)
from jasa.logging import get_logger
from omnifetch.fetch.shared.util import hash_key

_LOGGER = get_logger("grounding.cache")

MIN_SNIPPET_CHARS = 1
FETCHED_TITLE_MAX_CHARS = 2000
TEMPERATURE = 0.2
TOP_P = 0.9
FREQUENCY_PENALTY = 0.3
GROUNDING_CACHE_KEY_PREFIX = "jasa:grounding:v1:"
GROUNDING_CACHE_SEMANTICS_VERSION: Literal[1] = 1
_GROUNDING_CACHE_SCHEMA_VERSION: Literal[1] = 1
_STRICT_RECORD_CONFIG = ConfigDict(extra="forbid", strict=True, frozen=True)

GroundingCacheEvent = Literal[
    "hit",
    "miss",
    "write",
    "read_skipped",
    "write_skipped",
    "read_error",
    "write_error",
]


@dataclass(frozen=True, slots=True)
class GroundingCacheIdentity:
    """Every effective input that can change an accepted LLM snippet."""

    user_message: str
    system_prompt_sha256: str
    llm_base_url: str
    llm_model: str
    temperature: float
    top_p: float
    frequency_penalty: float
    max_tokens: int
    postprocess_fingerprint: str
    semantics_version: Literal[1]


@dataclass(frozen=True, slots=True)
class GroundingCacheWrite:
    """One accepted LLM result awaiting a fail-open cache write."""

    key: str
    identity: GroundingCacheIdentity
    snippet: str
    fetched_title: str


class _GroundingOutputRecord(BaseModel):
    """Strict accepted output required to rebuild a grounded result."""

    model_config = _STRICT_RECORD_CONFIG

    snippet: str = Field(
        min_length=MIN_SNIPPET_CHARS,
        max_length=SNIPPET_MAX_CHARS + len(FENCE_REPAIR_SUFFIX),
    )
    fetched_title: str = Field(max_length=FETCHED_TITLE_MAX_CHARS)


class _GroundingCacheRecord(BaseModel):
    """Versioned envelope binding exact LLM inputs to accepted output."""

    model_config = _STRICT_RECORD_CONFIG

    schema_version: Literal[1]
    identity_digest: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )
    output: _GroundingOutputRecord


def _grounding_postprocess_fingerprint() -> str:
    """Hash accepted-output bounds and detector/repair semantics."""
    identity = {
        "detectors": grounding_detector_semantics(),
        "min_snippet_chars": MIN_SNIPPET_CHARS,
        "snippet_max_chars": SNIPPET_MAX_CHARS,
    }
    canonical = json.dumps(identity, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def grounding_cache_identity(
    user_message: str,
    config: GroundingSettings,
) -> GroundingCacheIdentity:
    """Build the API-key-free identity for one effective LLM request."""
    return GroundingCacheIdentity(
        user_message=user_message,
        system_prompt_sha256=SYSTEM_PROMPT_SHA256,
        llm_base_url=config.llm_base_url,
        llm_model=config.llm_model,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        frequency_penalty=FREQUENCY_PENALTY,
        max_tokens=GROUNDING_MAX_TOKENS,
        postprocess_fingerprint=_grounding_postprocess_fingerprint(),
        semantics_version=GROUNDING_CACHE_SEMANTICS_VERSION,
    )


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
    fetched_title: str,
) -> str:
    """Serialize one accepted grounding result into the strict v1 envelope."""
    record = _GroundingCacheRecord(
        schema_version=_GROUNDING_CACHE_SCHEMA_VERSION,
        identity_digest=_identity_digest(identity),
        output=_GroundingOutputRecord(
            snippet=snippet,
            fetched_title=fetched_title,
        ),
    )
    return record.model_dump_json()


def _deserialize_grounding_cache(
    record: object,
    identity: GroundingCacheIdentity,
    fetched_title: str,
) -> str | None:
    """Return a validated cached snippet or None for incompatible data."""
    try:
        cached = _GroundingCacheRecord.model_validate(record)
    except ValidationError:
        return None
    output = cached.output
    if cached.identity_digest != _identity_digest(identity):
        return None
    if output.fetched_title != fetched_title:
        return None
    if repair_unbalanced_fence(output.snippet) != output.snippet:
        return None
    if detect_grounded_sentinel(output.snippet) is not None:
        return None
    return output.snippet


def record_grounding_cache_event(
    event: GroundingCacheEvent,
    error_type: str | None = None,
) -> None:
    """Log one bounded grounding-cache event without request material."""
    if error_type is None:
        _LOGGER.debug("Grounding cache event=%s", event)
        return
    _LOGGER.warning("Grounding cache event=%s error_type=%s", event, error_type)


async def read_grounding_cache(
    cache: CacheBackend,
    key: str,
    identity: GroundingCacheIdentity,
    fetched_title: str,
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
    snippet = _deserialize_grounding_cache(record, identity, fetched_title)
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
            pending.fetched_title,
        )
        stored = await cache.set(pending.key, value, ttl_seconds)
    except Exception as error:
        record_grounding_cache_event("write_error", type(error).__name__)
        return
    if stored is False:
        record_grounding_cache_event("write_error", "BackendRejected")
        return
    record_grounding_cache_event("write")
