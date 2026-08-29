"""Grounding service: mocked fetch + LLM, outcome classification."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import logging
from dataclasses import asdict, replace
from typing import cast, Literal

import httpx
import pytest
import respx

from jasa.cache.base import CacheBackend
from jasa.cache.memory import MemoryCache
from jasa.config import DEFAULT_GROUNDING_CACHE_TTL_SECONDS, GroundingSettings
from jasa.grounding.cache import (
    _deserialize_grounding_cache,
    _serialize_grounding_cache,
    FREQUENCY_PENALTY,
    grounding_cache_identity,
    GROUNDING_CACHE_KEY_PREFIX,
    grounding_cache_ttl_seconds,
    make_grounding_cache_key,
    MIN_SNIPPET_CHARS,
    TEMPERATURE,
    TOP_P,
)
from jasa.grounding.detectors import (
    FENCE_REPAIR_SUFFIX,
    grounding_detector_semantics,
)
from jasa.grounding.flights import GroundingFlightRegistry
from jasa.grounding.prompts import (
    build_grounded_user_message,
    GROUNDING_MAX_TOKENS,
    grounding_prompt_semantics,
    SNIPPET_MAX_CHARS,
)
from jasa.grounding.service import (
    _TierResponse,
    ground_results,
    grounding_semantic_fingerprint,
    GROUNDING_SEMANTICS_VERSION,
    GroundingContext,
    MIN_CONTENT_CHARS,
)
from jasa.grounding.waterfall import grounding_chain_semantics
from jasa.search.ranking import RankedWebResult
from jasa.server import _fetch_cache_identity
from omnifetch.cache import build_cache_backend
from omnifetch.tools.fetch import cache_identity_url
from tests.conftest import (
    grounding_engine,
    single_tier_waterfall,
    tier,
    tier_answer,
)

_SETTINGS = GroundingSettings()
_CHAIN = single_tier_waterfall(_SETTINGS).chain
_KEY = "cerebras-test"
_LLM_URL = "https://api.cerebras.ai/v1/chat/completions"
_SENTINEL_SUBSTRING_LIMIT = int(
    grounding_detector_semantics()["sentinel_substring_max_chars"]  # type: ignore[call-overload]
)


class _FetchResult:
    def __init__(self, content: str, title: str = "") -> None:
        self.content = content
        self.title = title


class _RecordingCache(MemoryCache):
    def __init__(self) -> None:
        super().__init__()
        self.read_keys: list[str] = []
        self.write_calls: list[tuple[str, str, int]] = []

    async def get(self, key: str) -> str | None:
        self.read_keys.append(key)
        return await super().get(key)

    async def set(self, key: str, value: str, ttl_seconds: int) -> None:
        self.write_calls.append((key, value, ttl_seconds))
        await super().set(key, value, ttl_seconds)


class _BrokenCache:
    async def get(self, key: str) -> object | None:
        raise RuntimeError("read failed")

    async def set(self, key: str, value: str, ttl_seconds: int) -> bool:
        raise RuntimeError("write failed")

    async def close(self) -> None:
        return None


class _RejectingCache:
    async def get(self, key: str) -> None:
        return None

    async def set(self, key: str, value: str, ttl_seconds: int) -> bool:
        return False

    async def close(self) -> None:
        return None


class _SlowWriteCache:
    def __init__(self) -> None:
        self.write_started = asyncio.Event()
        self.write_cancelled = False

    async def get(self, key: str) -> None:
        return None

    async def set(self, key: str, value: str, ttl_seconds: int) -> bool:
        self.write_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.write_cancelled = True
            raise
        return True

    async def close(self) -> None:
        return None


class _SlowReadCache:
    def __init__(self) -> None:
        self.read_started = asyncio.Event()
        self.read_cancelled = False

    async def get(self, key: str) -> None:
        self.read_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.read_cancelled = True
            raise

    async def set(self, key: str, value: str, ttl_seconds: int) -> bool:
        return True

    async def close(self) -> None:
        return None


class _BlockingWriteCache:
    def __init__(self) -> None:
        self.write_count = 0
        self.active_writes = 0
        self.max_active_writes = 0
        self.first_write_started = asyncio.Event()
        self.release_writes = asyncio.Event()

    async def get(self, key: str) -> None:
        return None

    async def set(self, key: str, value: str, ttl_seconds: int) -> bool:
        self.write_count += 1
        self.active_writes += 1
        self.max_active_writes = max(
            self.max_active_writes,
            self.active_writes,
        )
        self.first_write_started.set()
        try:
            await self.release_writes.wait()
        finally:
            self.active_writes -= 1
        return True

    async def close(self) -> None:
        return None


def _result(url: str) -> RankedWebResult:
    return RankedWebResult("t", url, ["agg"], ["p"], 0.1)


def _ctx(
    cache: CacheBackend | None = None,
    *,
    api_key: str = _KEY,
    settings: GroundingSettings = _SETTINGS,
    cache_ttl_seconds: int = DEFAULT_GROUNDING_CACHE_TTL_SECONDS,
) -> tuple[GroundingContext, httpx.AsyncClient]:
    """Return a grounding context with a dummy engine + fresh client."""
    client = httpx.AsyncClient()
    return (
        GroundingContext(
            engine=grounding_engine(),
            client=client,
            cache=cache if cache is not None else MemoryCache(),
            cache_write_semaphore=asyncio.Semaphore(settings.concurrency),
            flights=GroundingFlightRegistry(),
            waterfall=single_tier_waterfall(settings, api_key),
            config=settings,
            cache_ttl_seconds=cache_ttl_seconds,
        ),
        client,
    )


def _llm_ok(text: str | None) -> httpx.Response:
    return httpx.Response(
        200, json={"choices": [{"message": {"content": text}}]}
    )


def test_grounding_semantic_fingerprint_covers_output_inputs() -> None:
    identity = {
        "detectors": grounding_detector_semantics(),
        "frequency_penalty": FREQUENCY_PENALTY,
        "llm_chain": grounding_chain_semantics(_CHAIN),
        "max_content_chars": _SETTINGS.max_content_chars,
        "max_tokens": GROUNDING_MAX_TOKENS,
        "min_content_chars": MIN_CONTENT_CHARS,
        "min_snippet_chars": MIN_SNIPPET_CHARS,
        "prompts": grounding_prompt_semantics(),
        "semantics_version": GROUNDING_SEMANTICS_VERSION,
        "snippet_max_chars": SNIPPET_MAX_CHARS,
        "temperature": TEMPERATURE,
        "top_n": _SETTINGS.top_n,
        "top_p": TOP_P,
    }
    canonical = json.dumps(identity, separators=(",", ":"), sort_keys=True)
    expected = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    assert grounding_semantic_fingerprint(_SETTINGS, _CHAIN) == expected
    other_endpoint = (tier("p", "https://other", _SETTINGS.llm_model),)
    other_model = (tier("p", _SETTINGS.llm_base_url, "other"),)
    variants = {
        grounding_semantic_fingerprint(_SETTINGS, other_endpoint),
        grounding_semantic_fingerprint(_SETTINGS, other_model),
        grounding_semantic_fingerprint(_SETTINGS, _CHAIN + other_model),
        grounding_semantic_fingerprint(
            _SETTINGS.model_copy(update={"max_content_chars": 500}), _CHAIN
        ),
        grounding_semantic_fingerprint(
            _SETTINGS.model_copy(update={"top_n": 3}), _CHAIN
        ),
    }
    assert expected not in variants
    assert len(variants) == 5


def test_grounding_cache_key_hashes_every_effective_llm_input() -> None:
    identity = grounding_cache_identity(
        "https://example.com/page", "private query", 48_000, _CHAIN
    )
    key = make_grounding_cache_key(identity)
    variants = (
        replace(identity, url="https://example.com/other"),
        replace(identity, query="other query"),
        replace(identity, prompt_fingerprint="other prompt"),
        replace(
            identity,
            llm_chain=(("https://other.example/v1", "other-model"),),
        ),
        replace(identity, llm_chain=identity.llm_chain * 2),
        replace(identity, temperature=0.3),
        replace(identity, top_p=0.8),
        replace(identity, frequency_penalty=0.4),
        replace(identity, max_tokens=1024),
        replace(identity, postprocess_fingerprint="other semantics"),
        replace(identity, semantics_version=cast(Literal[4], 5)),
    )

    assert key.startswith(GROUNDING_CACHE_KEY_PREFIX)
    assert len(key) == len(GROUNDING_CACHE_KEY_PREFIX) + 64
    assert "private query" not in key
    assert "example.com" not in key
    assert make_grounding_cache_key(identity) == key
    assert len({make_grounding_cache_key(item) for item in variants}) == 11
    assert key not in {make_grounding_cache_key(item) for item in variants}
    assert "api_key" not in asdict(identity)


def test_the_same_page_rerendered_keeps_its_key() -> None:
    """The defect this identity exists to fix.

    The same page arrives as different bytes whenever a different provider
    wins the fetch race -- which is what inserting a provider into the
    waterfall does to every URL at once. Under content keying each rendering
    was a separate entry, so every accepted snippet became unaddressable and
    the LLM call behind it was bought again. Nothing about the page or the
    question changed, so nothing about the key may either.
    """

    def key(content: str) -> str:
        assert build_grounded_user_message(
            "query", "Title", content, 48_000
        ) != build_grounded_user_message("query", "Title", "other", 48_000)
        return make_grounding_cache_key(
            grounding_cache_identity(
                "https://example.com/page", "query", 48_000, _CHAIN
            )
        )

    assert key("# Heading\n\ntext") == key("Heading\n===\n\ntext")


def test_grounding_cache_key_separates_page_question_and_content_cap() -> None:
    """URL keying must not collapse distinct requests onto one entry.

    A grounded snippet answers a query about a page, so the page alone cannot
    be the key -- sharing the fetch key outright would serve one query's
    snippet to another. The content cap belongs here too: it decides how much
    of the page the model saw, and it used to reach the key only by truncating
    the message that was hashed.
    """

    def key(url: str, query: str, max_chars: int = 48_000) -> str:
        return make_grounding_cache_key(
            grounding_cache_identity(url, query, max_chars, _CHAIN)
        )

    base = key("https://example.com/page", "query")
    assert base != key("https://example.com/other", "query")
    assert base != key("https://example.com/page", "other query")
    assert base != key("https://example.com/page", "query", 1000)


def test_grounding_url_identity_matches_the_fetch_cache_identity() -> None:
    """Both caches must agree on which spellings are one page.

    Jasa injects its own canonicalizer into the omnifetch engine, so the
    agreement is only real if grounding keys on the canonicalizer's output
    rather than on the URL as given. Two spellings the fetch cache folds into
    one entry must reach one grounding entry as well, or the derived cache
    misses on pages the fetch cache is actively serving.
    """
    engine = grounding_engine()

    def key(url: str) -> str:
        return make_grounding_cache_key(
            grounding_cache_identity(
                cache_identity_url(engine, url), "query", 48_000, _CHAIN
            )
        )

    assert key("https://example.com/page/") == key("https://example.com/page")
    assert key("  https://example.com/page  ") == key(
        "https://example.com/page"
    )
    assert key("https://example.com/page") != key("https://example.com/else")


def test_volatile_urls_keep_the_short_fetch_lifetime() -> None:
    """A snippet is exactly as perishable as the page it was written from.

    Omnifetch holds a homepage for minutes because it is a rolling index.
    Content keying used to invalidate that snippet for free -- new masthead,
    new bytes, new key. URL keying removes that, so the lifetime has to carry
    it, or a front page would be republished for a full day after the fetch
    layer had already discarded it many times over.
    """
    assert (
        grounding_cache_ttl_seconds(
            "https://example.com/", 86_400, 864_000, 300
        )
        == 300
    )
    assert (
        grounding_cache_ttl_seconds("https://example.com", 86_400, 864_000, 300)
        == 300
    )
    assert (
        grounding_cache_ttl_seconds(
            "https://example.com/article", 86_400, 864_000, 300
        )
        == 86_400
    )


def test_a_snippet_never_outlives_the_page_it_describes() -> None:
    """The fetch lifetime is a ceiling, not a comfortable assumption.

    At the shipped defaults a page is held ten days and a snippet one, so the
    ceiling never binds. The two are configured independently, though, and an
    operator who shortens the fetch TTL below the grounding TTL would otherwise
    keep serving a snippet describing a page this deployment has already
    stopped believing in -- the next fetch may return something else entirely.
    """
    assert (
        grounding_cache_ttl_seconds(
            "https://example.com/article", 86_400, 60, 300
        )
        == 60
    )
    assert (
        grounding_cache_ttl_seconds("https://example.com/", 86_400, 60, 300)
        == 60
    )


def test_volatile_lifetime_never_exceeds_the_configured_one() -> None:
    """Shortening the main TTL means everything fresher, not homepages last."""
    assert (
        grounding_cache_ttl_seconds("https://example.com/", 60, 864_000, 300)
        == 60
    )


def test_grounding_cache_record_is_strict_and_identity_bound() -> None:
    identity = grounding_cache_identity(
        "https://example.com/page", "private query", 48_000, _CHAIN
    )
    serialized = _serialize_grounding_cache(identity, "accepted")
    valid = cast(dict[str, object], json.loads(serialized))

    assert _deserialize_grounding_cache(valid, identity) == "accepted"
    assert "private query" not in serialized
    assert "example.com" not in serialized
    assert identity.prompt_fingerprint not in serialized

    legacy_v1 = {
        "schema_version": 1,
        "identity_digest": valid["identity_digest"],
        "output": {"snippet": "accepted", "fetched_title": "Title"},
    }
    wrong_version = {**copy.deepcopy(valid), "schema_version": 3}
    top_extra = {**copy.deepcopy(valid), "unexpected": True}
    identity_drift = copy.deepcopy(valid)
    identity_drift["identity_digest"] = "0" * 64
    malformed_digest = copy.deepcopy(valid)
    malformed_digest["identity_digest"] = "not-a-digest"
    wrong_type = {**copy.deepcopy(valid), "snippet": 7}
    empty = {**copy.deepcopy(valid), "snippet": ""}
    overlong = {
        **copy.deepcopy(valid),
        "snippet": "x" * (SNIPPET_MAX_CHARS + len(FENCE_REPAIR_SUFFIX) + 1),
    }
    unbalanced = {**copy.deepcopy(valid), "snippet": "```python"}
    sentinel = {**copy.deepcopy(valid), "snippet": "[no usable content]"}

    cases = (
        legacy_v1,
        wrong_version,
        top_extra,
        identity_drift,
        malformed_digest,
        wrong_type,
        empty,
        overlong,
        unbalanced,
        sentinel,
        "not a record",
    )
    assert all(
        _deserialize_grounding_cache(case, identity) is None for case in cases
    )


async def test_grounded(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_fetch(engine: object, url: str) -> _FetchResult:
        return _FetchResult("Real content. " * 20, "Title")

    monkeypatch.setattr("jasa.grounding.service.execute_web_fetch", fake_fetch)
    cache = _RecordingCache()
    ctx, client = _ctx(cache, cache_ttl_seconds=321)
    with respx.mock:
        route = respx.post(_LLM_URL).mock(
            return_value=_llm_ok("Grounded snippet.")
        )
        pairs, stats = await ground_results(
            "q", [_result("https://x.com/a")], ctx
        )
        cached_pairs, cached_stats = await ground_results(
            "q", [_result("https://x.com/a")], ctx
        )
    assert pairs[0][1] == "grounded"
    assert pairs[0][0].snippet_source == "grounded"
    assert pairs[0][0].title == "Title"
    assert stats.grounded_count == 1
    assert cached_pairs == pairs
    assert cached_stats == stats
    assert route.call_count == 1
    assert len(cache.read_keys) == 3
    assert len(cache.write_calls) == 1
    assert cache.write_calls[0][2] == 321
    assert (
        json.loads(route.calls.last.request.content)["max_tokens"]
        == GROUNDING_MAX_TOKENS
    )
    await client.aclose()


async def test_a_homepage_snippet_is_written_with_the_short_lifetime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The clamp has to reach the write, not merely exist.

    `grounding_cache_ttl_seconds` being correct proves nothing if the write
    site still passes the configured value straight through, so this asserts
    the TTL the cache was actually handed for a rolling index.
    """

    async def fake_fetch(engine: object, url: str) -> _FetchResult:
        return _FetchResult("Real content. " * 20, "Title")

    monkeypatch.setattr("jasa.grounding.service.execute_web_fetch", fake_fetch)
    cache = _RecordingCache()
    ctx, client = _ctx(cache, cache_ttl_seconds=86_400)
    ctx = replace(ctx, volatile_cache_ttl_seconds=300)
    with respx.mock:
        respx.post(_LLM_URL).mock(return_value=_llm_ok("Grounded snippet."))
        await ground_results("q", [_result("https://x.com/")], ctx)

    assert [call[2] for call in cache.write_calls] == [300]
    await client.aclose()


async def test_grounding_cache_identity_excludes_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_fetch(engine: object, url: str) -> _FetchResult:
        return _FetchResult("Real content. " * 20, "Title")

    monkeypatch.setattr("jasa.grounding.service.execute_web_fetch", fake_fetch)
    cache = _RecordingCache()
    first_context, first_client = _ctx(cache, api_key="first-secret")
    second_context, second_client = _ctx(cache, api_key="second-secret")
    with respx.mock:
        route = respx.post(_LLM_URL).mock(return_value=_llm_ok("Grounded"))
        first, _ = await ground_results("q", [_result("u")], first_context)
        second, _ = await ground_results("q", [_result("u")], second_context)

    assert first == second
    assert route.call_count == 1
    assert route.calls[0].request.headers["authorization"] == (
        "Bearer first-secret"
    )
    assert all("first-secret" not in value for _, value, _ in cache.write_calls)
    await first_client.aclose()
    await second_client.aclose()


async def test_runtime_cachelib_backend_reuses_grounding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_fetch(engine: object, url: str) -> _FetchResult:
        return _FetchResult("Real content. " * 20, "Title")

    monkeypatch.setattr("jasa.grounding.service.execute_web_fetch", fake_fetch)
    cache = build_cache_backend(
        "memory",
        disk_path="",
        redis_url="",
        max_entries=10,
    )
    context, client = _ctx(cast(CacheBackend, cache))
    with respx.mock:
        route = respx.post(_LLM_URL).mock(return_value=_llm_ok("Grounded"))
        first, _ = await ground_results("q", [_result("u")], context)
        second, _ = await ground_results("q", [_result("u")], context)

    assert first == second
    assert route.call_count == 1
    await cache.close()
    await client.aclose()


async def test_invalid_cached_grounding_continues_to_llm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    content = "Real content. " * 20
    title = "Title"

    async def fake_fetch(engine: object, url: str) -> _FetchResult:
        return _FetchResult(content, title)

    monkeypatch.setattr("jasa.grounding.service.execute_web_fetch", fake_fetch)
    cache = _RecordingCache()
    key = make_grounding_cache_key(
        grounding_cache_identity(
            _fetch_cache_identity("u"),
            "q",
            _SETTINGS.max_content_chars,
            _CHAIN,
        )
    )
    await cache.set(key, "not json", 60)
    cache.write_calls.clear()
    context, client = _ctx(cache)
    with respx.mock:
        route = respx.post(_LLM_URL).mock(return_value=_llm_ok("Recovered"))
        pairs, stats = await ground_results("q", [_result("u")], context)

    assert pairs[0][0].snippets == ["Recovered"]
    assert pairs[0][1] == "grounded"
    assert stats.grounded_count == 1
    assert stats.transient_failures == 0
    assert route.call_count == 1
    assert cache.read_keys == [key, key]
    assert len(cache.write_calls) == 1
    await client.aclose()


async def test_grounding_cache_exceptions_fail_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_fetch(engine: object, url: str) -> _FetchResult:
        return _FetchResult("Real content. " * 20, "Title")

    monkeypatch.setattr("jasa.grounding.service.execute_web_fetch", fake_fetch)
    context, client = _ctx(cast(CacheBackend, _BrokenCache()))
    with respx.mock:
        route = respx.post(_LLM_URL).mock(return_value=_llm_ok("Grounded"))
        pairs, stats = await ground_results("q", [_result("u")], context)

    assert pairs[0][1] == "grounded"
    assert pairs[0][0].snippets == ["Grounded"]
    assert stats.transient_failures == 0
    assert route.call_count == 1
    await client.aclose()


async def test_an_oversized_title_no_longer_blocks_reuse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A page with a huge title is cacheable like any other.

    The record used to persist the fetched title under a 2000-character bound,
    so a page whose title exceeded it failed validation on every write: the
    snippet was never stored and the LLM call was repeated for the life of the
    page. The title is no longer part of the record -- it is derived from
    content, which is exactly what this identity stopped keying on -- so the
    bound, and the permanent miss it caused, are both gone. The title still
    reaches the caller from the live fetch.
    """
    oversized_title = "x" * 2001

    async def fake_fetch(engine: object, url: str) -> _FetchResult:
        return _FetchResult("Real content. " * 20, oversized_title)

    monkeypatch.setattr("jasa.grounding.service.execute_web_fetch", fake_fetch)
    cache = _RecordingCache()
    context, client = _ctx(cache)
    with respx.mock:
        route = respx.post(_LLM_URL).mock(return_value=_llm_ok("Grounded"))
        first, _ = await ground_results("q", [_result("u")], context)
        second, _ = await ground_results("q", [_result("u")], context)

    assert first == second
    assert first[0][0].title == oversized_title
    assert route.call_count == 1
    assert len(cache.write_calls) == 1
    assert oversized_title not in cache.write_calls[0][1]
    await client.aclose()


async def test_slow_grounding_cache_read_continues_to_llm(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def fake_fetch(engine: object, url: str) -> _FetchResult:
        return _FetchResult("Real content. " * 20, "Title")

    monkeypatch.setattr("jasa.grounding.service.execute_web_fetch", fake_fetch)
    cache = _SlowReadCache()
    context, client = _ctx(cast(CacheBackend, cache))
    with (
        respx.mock,
        caplog.at_level(
            logging.DEBUG,
            logger="jasa.grounding.cache",
        ),
    ):
        route = respx.post(_LLM_URL).mock(return_value=_llm_ok("Grounded"))
        pairs, stats = await ground_results("q", [_result("u")], context)

    assert cache.read_started.is_set()
    assert cache.read_cancelled is True
    assert route.call_count == 1
    assert pairs[0][1] == "grounded"
    assert stats.transient_failures == 0
    assert "Grounding cache event=read_skipped" in caplog.messages
    assert not any("event=read_error" in message for message in caplog.messages)
    await client.aclose()


async def test_slow_cache_read_reserves_remaining_pipeline_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_fetch(engine: object, url: str) -> _FetchResult:
        await asyncio.sleep(0.3)
        return _FetchResult("Real content. " * 20, "Title")

    monkeypatch.setattr("jasa.grounding.service.execute_web_fetch", fake_fetch)
    cache = _SlowReadCache()
    settings = GroundingSettings(per_url_deadline_ms=500)
    context, client = _ctx(cast(CacheBackend, cache), settings=settings)
    with respx.mock:
        route = respx.post(_LLM_URL).mock(return_value=_llm_ok("Grounded"))
        pairs, stats = await ground_results("q", [_result("u")], context)

    assert cache.read_cancelled is True
    assert route.call_count == 1
    assert pairs[0][1] == "grounded"
    assert stats.transient_failures == 0
    await client.aclose()


async def test_grounding_cache_rejection_never_claims_reuse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_fetch(engine: object, url: str) -> _FetchResult:
        return _FetchResult("Real content. " * 20, "Title")

    monkeypatch.setattr("jasa.grounding.service.execute_web_fetch", fake_fetch)
    context, client = _ctx(cast(CacheBackend, _RejectingCache()))
    with respx.mock:
        route = respx.post(_LLM_URL).mock(return_value=_llm_ok("Grounded"))
        first, _ = await ground_results("q", [_result("u")], context)
        second, _ = await ground_results("q", [_result("u")], context)

    assert first == second
    assert route.call_count == 2
    await client.aclose()


async def test_slow_grounding_cache_write_preserves_paid_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_fetch(engine: object, url: str) -> _FetchResult:
        return _FetchResult("Real content. " * 20, "Title")

    monkeypatch.setattr("jasa.grounding.service.execute_web_fetch", fake_fetch)
    cache = _SlowWriteCache()
    settings = GroundingSettings(per_url_deadline_ms=100)
    context, client = _ctx(cast(CacheBackend, cache), settings=settings)
    with respx.mock:
        respx.post(_LLM_URL).mock(return_value=_llm_ok("Grounded"))
        pairs, stats = await ground_results("q", [_result("u")], context)

    assert pairs[0][1] == "grounded"
    assert pairs[0][0].snippets == ["Grounded"]
    assert stats.transient_failures == 0
    assert cache.write_started.is_set()
    assert cache.write_cancelled is True
    await client.aclose()


async def test_caller_cancellation_propagates_during_grounding_cache_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_fetch(engine: object, url: str) -> _FetchResult:
        return _FetchResult("Real content. " * 20, "Title")

    monkeypatch.setattr("jasa.grounding.service.execute_web_fetch", fake_fetch)
    cache = _SlowWriteCache()
    context, client = _ctx(cast(CacheBackend, cache))
    with respx.mock:
        respx.post(_LLM_URL).mock(return_value=_llm_ok("Grounded"))
        task = asyncio.create_task(ground_results("q", [_result("u")], context))
        await cache.write_started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert cache.write_cancelled is True
    await client.aclose()


async def test_grounding_cache_writes_do_not_hold_worker_slots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm_call_count = 0
    second_llm_called = asyncio.Event()

    async def fake_fetch(engine: object, url: str) -> _FetchResult:
        return _FetchResult(f"{url} content. " * 20, "Title")

    async def fake_llm_call(*args: object) -> _TierResponse:
        nonlocal llm_call_count
        llm_call_count += 1
        if llm_call_count == 2:
            second_llm_called.set()
        return tier_answer("Grounded")

    monkeypatch.setattr("jasa.grounding.service.execute_web_fetch", fake_fetch)
    monkeypatch.setattr(
        "jasa.grounding.service._call_grounding_tier", fake_llm_call
    )
    cache = _BlockingWriteCache()
    settings = GroundingSettings(concurrency=1)
    context, client = _ctx(cast(CacheBackend, cache), settings=settings)
    first_task = asyncio.create_task(
        ground_results("q", [_result("a")], context)
    )
    second_task = asyncio.create_task(
        ground_results("q", [_result("b")], context)
    )
    async with asyncio.timeout(1):
        await cache.first_write_started.wait()
        await second_llm_called.wait()
    assert cache.write_count == 1
    assert cache.max_active_writes == 1
    cache.release_writes.set()
    first, second = await asyncio.gather(first_task, second_task)

    assert [first[0][0][1], second[0][0][1]] == ["grounded", "grounded"]
    assert first[1].grounded_count == 1
    assert second[1].grounded_count == 1
    assert cache.write_count == 2
    assert cache.max_active_writes == 1
    await client.aclose()


async def test_fetch_exhausted(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_fetch(engine: object, url: str) -> None:
        raise RuntimeError("connect failed")

    monkeypatch.setattr("jasa.grounding.service.execute_web_fetch", fake_fetch)
    cache = _RecordingCache()
    ctx, client = _ctx(cache)
    pairs, stats = await ground_results("q", [_result("u")], ctx)
    assert pairs[0][1] == "fallback:fetch_exhausted"
    assert stats.transient_failures == 0
    assert cache.write_calls == []
    await client.aclose()


async def test_fetch_too_short(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_fetch(engine: object, url: str) -> _FetchResult:
        return _FetchResult("short")

    monkeypatch.setattr("jasa.grounding.service.execute_web_fetch", fake_fetch)
    cache = _RecordingCache()
    ctx, client = _ctx(cache)
    pairs, _ = await ground_results("q", [_result("u")], ctx)
    assert pairs[0][1] == "fallback:fetch_too_short"
    assert cache.write_calls == []
    await client.aclose()


async def test_fetch_junk(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_fetch(engine: object, url: str) -> _FetchResult:
        return _FetchResult("subscribe to continue reading" + "x" * 100)

    monkeypatch.setattr("jasa.grounding.service.execute_web_fetch", fake_fetch)
    cache = _RecordingCache()
    ctx, client = _ctx(cache)
    pairs, _ = await ground_results("q", [_result("u")], ctx)
    assert pairs[0][1] == "fallback:fetch_junk"
    assert cache.write_calls == []
    await client.aclose()


async def test_llm_sentinel(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_fetch(engine: object, url: str) -> _FetchResult:
        return _FetchResult("Real content. " * 20)

    monkeypatch.setattr("jasa.grounding.service.execute_web_fetch", fake_fetch)
    cache = _RecordingCache()
    ctx, client = _ctx(cache)
    with respx.mock:
        respx.post(_LLM_URL).mock(return_value=_llm_ok("[no usable content]"))
        pairs, _ = await ground_results("q", [_result("u")], ctx)
    assert pairs[0][1] == "fallback:llm_sentinel"
    assert cache.write_calls == []
    await client.aclose()


async def test_overlong_snippet_repairs_fence_after_truncation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_fetch(engine: object, url: str) -> _FetchResult:
        return _FetchResult("Real content. " * 20)

    monkeypatch.setattr("jasa.grounding.service.execute_web_fetch", fake_fetch)
    overlong = "x" * (SNIPPET_MAX_CHARS - 20) + "\n```python\n" + "y" * 100
    assert len(overlong) > SNIPPET_MAX_CHARS
    ctx, client = _ctx()
    with respx.mock:
        respx.post(_LLM_URL).mock(return_value=_llm_ok(overlong))
        pairs, _ = await ground_results("q", [_result("u")], ctx)
    snippet = pairs[0][0].snippets[0]
    assert len(snippet) <= SNIPPET_MAX_CHARS
    assert snippet.endswith("\n```")
    assert snippet.count("```") == 2
    await client.aclose()


async def test_llm_error_is_transient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_fetch(engine: object, url: str) -> _FetchResult:
        return _FetchResult("Real content. " * 20)

    monkeypatch.setattr("jasa.grounding.service.execute_web_fetch", fake_fetch)
    cache = _RecordingCache()
    ctx, client = _ctx(cache)
    with respx.mock:
        respx.post(_LLM_URL).mock(return_value=httpx.Response(500))
        pairs, stats = await ground_results("q", [_result("u")], ctx)
    assert pairs[0][1] == "fallback:llm_error"
    assert stats.transient_failures == 1
    assert cache.write_calls == []
    await client.aclose()


async def test_pipeline_timeout_is_transient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_fetch(engine: object, url: str) -> _FetchResult:
        await asyncio.sleep(0.15)
        return _FetchResult("content")

    monkeypatch.setattr("jasa.grounding.service.execute_web_fetch", fake_fetch)
    settings = GroundingSettings(per_url_deadline_ms=100)
    client = httpx.AsyncClient()
    cache = _RecordingCache()
    ctx = GroundingContext(
        engine=grounding_engine(),
        client=client,
        cache=cache,
        cache_write_semaphore=asyncio.Semaphore(settings.concurrency),
        flights=GroundingFlightRegistry(),
        waterfall=single_tier_waterfall(settings, _KEY),
        config=settings,
    )
    pairs, stats = await ground_results("q", [_result("u")], ctx)
    assert pairs[0][1] == "fallback:pipeline_timeout"
    assert stats.transient_failures == 1
    assert cache.write_calls == []
    await client.aclose()


async def test_llm_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_fetch(engine: object, url: str) -> _FetchResult:
        return _FetchResult("Real content. " * 20)

    monkeypatch.setattr("jasa.grounding.service.execute_web_fetch", fake_fetch)
    cache = _RecordingCache()
    ctx, client = _ctx(cache)
    with respx.mock:
        respx.post(_LLM_URL).mock(return_value=_llm_ok(""))
        pairs, _ = await ground_results("q", [_result("u")], ctx)
    assert pairs[0][1] == "fallback:llm_empty"
    assert cache.write_calls == []
    await client.aclose()


async def test_llm_null_content_is_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_fetch(engine: object, url: str) -> _FetchResult:
        return _FetchResult("Real content. " * 20)

    monkeypatch.setattr("jasa.grounding.service.execute_web_fetch", fake_fetch)
    cache = _RecordingCache()
    ctx, client = _ctx(cache)
    with respx.mock:
        respx.post(_LLM_URL).mock(return_value=_llm_ok(None))
        pairs, _ = await ground_results("q", [_result("u")], ctx)
    assert pairs[0][1] == "fallback:llm_empty"
    assert cache.write_calls == []
    await client.aclose()


async def test_worker_rejection_is_transient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def reject(*_args: object) -> None:
        raise RuntimeError("worker failed")

    monkeypatch.setattr("jasa.grounding.service._fetch_and_prepare", reject)
    cache = _RecordingCache()
    ctx, client = _ctx(cache)
    pairs, stats = await ground_results("q", [_result("u")], ctx)
    assert pairs[0][1] == "fallback:worker_rejected"
    assert stats.transient_failures == 1
    assert cache.write_calls == []
    await client.aclose()


async def test_fetch_error_not_transient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_fetch(engine: object, url: str) -> None:
        raise ValueError("unexpected")

    monkeypatch.setattr("jasa.grounding.service.execute_web_fetch", fake_fetch)
    cache = _RecordingCache()
    ctx, client = _ctx(cache)
    pairs, stats = await ground_results("q", [_result("u")], ctx)
    assert pairs[0][1] == "fallback:fetch_exhausted"
    assert stats.transient_failures == 0
    assert cache.write_calls == []
    await client.aclose()


async def test_default_grounding_prefetches_only_top_20_of_50(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processed_urls: list[str] = []

    async def fake_fetch(engine: object, url: str) -> _FetchResult:
        processed_urls.append(url)
        return _FetchResult("short")

    monkeypatch.setattr("jasa.grounding.service.execute_web_fetch", fake_fetch)
    settings = GroundingSettings()
    client = httpx.AsyncClient()
    ctx = GroundingContext(
        engine=grounding_engine(),
        client=client,
        cache=MemoryCache(),
        cache_write_semaphore=asyncio.Semaphore(settings.concurrency),
        flights=GroundingFlightRegistry(),
        waterfall=single_tier_waterfall(settings, _KEY),
        config=settings,
    )
    results = [_result(f"https://{index}.example") for index in range(50)]
    pairs, stats = await ground_results("q", results, ctx)
    assert len(pairs) == 20
    assert stats.total_urls == 20
    assert processed_urls == [f"https://{index}.example" for index in range(20)]
    await client.aclose()


async def test_expired_stage_budget_keeps_finished_snippets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A URL that finished must survive a stage deadline a sibling blew.

    This is the regression guard for the failure that made grounding look
    absent in production: the whole stage was cancelled on expiry, so pages
    that had already been fetched and rewritten by the LLM were discarded
    along with the one page still in flight.
    """

    async def fake_fetch(engine: object, url: str) -> _FetchResult:
        if url.endswith("slow"):
            await asyncio.Event().wait()
        return _FetchResult("c" * 200, "Title")

    monkeypatch.setattr("jasa.grounding.service.execute_web_fetch", fake_fetch)
    monkeypatch.setattr(
        "jasa.grounding.service.MIN_WORKER_BUDGET_SECONDS", 0.05
    )
    ctx, client = _ctx()
    results = [
        _result("https://a.example/fast"),
        _result("https://b.example/slow"),
    ]
    with respx.mock:
        respx.post(_LLM_URL).mock(return_value=_llm_ok("grounded text"))
        deadline_at = asyncio.get_running_loop().time() + 0.4
        pairs, stats = await ground_results("q", results, ctx, deadline_at)

    by_url = {result.url: (result, outcome) for result, outcome in pairs}
    assert by_url["https://a.example/fast"][1] == "grounded"
    assert by_url["https://a.example/fast"][0].snippets == ["grounded text"]
    assert by_url["https://b.example/slow"][1] == "fallback:pipeline_timeout"
    assert stats.grounded_count == 1
    assert stats.total_urls == 2
    assert stats.outcomes["grounded"] == 1
    await client.aclose()


async def test_failed_grounding_marks_the_result_as_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_fetch(engine: object, url: str) -> _FetchResult:
        return _FetchResult("short")

    monkeypatch.setattr("jasa.grounding.service.execute_web_fetch", fake_fetch)
    ctx, client = _ctx()

    pairs, _stats = await ground_results(
        "q", [_result("https://a.example")], ctx
    )

    result, outcome = pairs[0]
    assert outcome == "fallback:fetch_too_short"
    assert result.snippet_source == "fallback"
    assert result.snippets == ["agg"]
    await client.aclose()


async def test_worker_declines_when_the_budget_is_already_spent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A worker that reaches the front of the queue too late must not pay."""
    fetched: list[str] = []

    async def fake_fetch(engine: object, url: str) -> _FetchResult:
        fetched.append(url)
        return _FetchResult("c" * 200, "Title")

    monkeypatch.setattr("jasa.grounding.service.execute_web_fetch", fake_fetch)
    ctx, client = _ctx()
    expired = asyncio.get_running_loop().time() - 1.0

    pairs, stats = await ground_results(
        "q", [_result("https://a.example")], ctx, expired
    )

    assert pairs[0][1] == "fallback:pipeline_timeout"
    assert fetched == []
    assert stats.grounded_count == 0
    await client.aclose()


async def test_worker_crash_is_classified_without_escaping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def exploding_worker(execution: object) -> None:
        raise RuntimeError("worker exploded")

    monkeypatch.setattr("jasa.grounding.service._ground_one", exploding_worker)
    ctx, client = _ctx()

    pairs, stats = await ground_results(
        "q", [_result("https://a.example")], ctx
    )

    assert pairs[0][1] == "fallback:worker_rejected"
    assert stats.transient_failures == 1
    await client.aclose()


async def test_outer_cancellation_drains_every_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cancelled_fetches: list[str] = []

    async def fake_fetch(engine: object, url: str) -> _FetchResult:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled_fetches.append(url)
            raise
        raise AssertionError("unreachable")

    monkeypatch.setattr("jasa.grounding.service.execute_web_fetch", fake_fetch)
    ctx, client = _ctx()
    task = asyncio.create_task(
        ground_results("q", [_result("https://a.example")], ctx)
    )
    await asyncio.sleep(0.05)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert cancelled_fetches == ["https://a.example"], (
        "the in-flight worker was not cancelled by the outer cancellation"
    )
    await client.aclose()


def test_result_host_tolerates_an_unparseable_url() -> None:
    from jasa.grounding.service import _result_host

    assert _result_host("https://[oops") == "(unparseable)"
    assert _result_host("mailto:someone@example.com") == "(unparseable)"
    assert _result_host("https://example.com/path") == "example.com"


async def test_worker_cancelled_while_queued_is_classified_not_lost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A worker still waiting for a slot at the deadline gets an outcome.

    Queued workers hold no deadline of their own, so the stage cancels them.
    Harvesting must turn that cancellation into a pipeline timeout rather than
    dropping the row.
    """

    async def fake_fetch(engine: object, url: str) -> _FetchResult:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    monkeypatch.setattr("jasa.grounding.service.execute_web_fetch", fake_fetch)
    monkeypatch.setattr(
        "jasa.grounding.service.MIN_WORKER_BUDGET_SECONDS", 0.01
    )
    settings = GroundingSettings(concurrency=1, top_n=2)
    client = httpx.AsyncClient()
    ctx = GroundingContext(
        engine=grounding_engine(),
        client=client,
        cache=MemoryCache(),
        cache_write_semaphore=asyncio.Semaphore(1),
        flights=GroundingFlightRegistry(),
        waterfall=single_tier_waterfall(settings, _KEY),
        config=settings,
    )
    results = [_result("https://a.example"), _result("https://b.example")]

    deadline_at = asyncio.get_running_loop().time() + 0.2
    pairs, stats = await ground_results("q", results, ctx, deadline_at)

    assert len(pairs) == 2
    assert [outcome for _result, outcome in pairs] == [
        "fallback:pipeline_timeout",
        "fallback:pipeline_timeout",
    ]
    assert stats.transient_failures == 2
    assert all(r.snippet_source == "fallback" for r, _o in pairs)
    await client.aclose()


def test_trim_truncated_snippet_cuts_back_to_a_complete_sentence() -> None:
    from jasa.grounding.detectors import (
        trim_truncated_snippet,
        TRUNCATION_TRIM_MAX_CHARS,
    )

    cut = "First fact. Second fact. Then the model stopped mid-thought with"
    assert trim_truncated_snippet(cut) == "First fact. Second fact."
    # A fenced snippet is left alone; repair_unbalanced_fence owns that case.
    fenced = "Intro. ```sql\nSELECT 1\n``` trailing words that were cut"
    assert trim_truncated_snippet(fenced) == fenced
    # No boundary at all: keep the evidence rather than gutting the snippet.
    assert trim_truncated_snippet("no boundary here") == "no boundary here"
    # Trimming away a long tail of real evidence is worse than a rough ending.
    long_tail = "Short. " + ("x" * (TRUNCATION_TRIM_MAX_CHARS + 50))
    assert trim_truncated_snippet(long_tail) == long_tail


async def test_token_ceiling_generation_is_trimmed_before_acceptance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A generation stopped at max_tokens must not publish a fragment.

    The character cap cannot do this: it only fires above 2000 characters,
    while a capped generation stops wherever the tokens ran out.
    """

    async def fake_fetch(engine: object, url: str) -> _FetchResult:
        return _FetchResult("c" * 400, "Title")

    monkeypatch.setattr("jasa.grounding.service.execute_web_fetch", fake_fetch)
    ctx, client = _ctx()
    body = {
        "choices": [
            {
                "message": {
                    "content": "Skip scan needs B-tree. Enable it with",
                },
                "finish_reason": "length",
            }
        ]
    }
    with respx.mock:
        respx.post(_LLM_URL).mock(return_value=httpx.Response(200, json=body))
        pairs, stats = await ground_results(
            "q", [_result("https://a.example")], ctx
        )

    result, outcome = pairs[0]
    assert outcome == "grounded"
    assert result.snippets == ["Skip scan needs B-tree."]
    assert stats.grounded_count == 1
    await client.aclose()


async def test_completed_generation_is_never_trimmed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_fetch(engine: object, url: str) -> _FetchResult:
        return _FetchResult("c" * 400, "Title")

    monkeypatch.setattr("jasa.grounding.service.execute_web_fetch", fake_fetch)
    ctx, client = _ctx()
    text = (
        "Skip scan needs B-tree. Trailing clause without a period\n"
        "Coverage: answers x; does NOT cover y."
    )
    body = {
        "choices": [{"message": {"content": text}, "finish_reason": "stop"}]
    }
    with respx.mock:
        respx.post(_LLM_URL).mock(return_value=httpx.Response(200, json=body))
        pairs, _stats = await ground_results(
            "q", [_result("https://a.example")], ctx
        )

    assert pairs[0][0].snippets == [text]
    await client.aclose()


def test_max_tokens_covers_the_prompt_length_contract() -> None:
    """The token ceiling must be able to reach the contract the prompt sets.

    The prompt permits 2000 characters plus a Coverage line of up to 200 more,
    and requires the snippet to be written in the query's language. CJK runs
    near one character per token -- the worst ratio the contract must survive.
    A ceiling below it silently truncates long snippets in exactly the
    languages the live-testing rule insists on covering.
    """
    from jasa.grounding.prompts import (
        GROUNDING_MAX_TOKENS,
        SNIPPET_MAX_CHARS,
        WORST_CASE_CHARS_PER_TOKEN,
    )

    reachable_chars = GROUNDING_MAX_TOKENS * WORST_CASE_CHARS_PER_TOKEN
    assert reachable_chars >= SNIPPET_MAX_CHARS


def test_snippet_cap_leaves_room_for_the_coverage_line() -> None:
    """The cap must fit the prompt's contract, not just the snippet body.

    The prompt caps the body at 2000 characters and then requires a closing
    Coverage line of up to 200 more. A 2000-character cap severed that line
    mid-sentence on exactly the snippets whose pages had the most to say.
    """
    from jasa.grounding.prompts import (
        COVERAGE_LINE_MAX_CHARS,
        COVERAGE_SEPARATOR_MAX_CHARS,
        SNIPPET_BODY_MAX_CHARS,
        SNIPPET_MAX_CHARS,
    )

    assert SNIPPET_MAX_CHARS == (
        SNIPPET_BODY_MAX_CHARS
        + COVERAGE_SEPARATOR_MAX_CHARS
        + COVERAGE_LINE_MAX_CHARS
    )


async def test_character_cap_also_trims_to_a_clean_sentence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An over-long generation is trimmed, not chopped mid-word."""
    from jasa.grounding.prompts import SNIPPET_MAX_CHARS

    async def fake_fetch(engine: object, url: str) -> _FetchResult:
        return _FetchResult("c" * 400, "Title")

    monkeypatch.setattr("jasa.grounding.service.execute_web_fetch", fake_fetch)
    ctx, client = _ctx()
    sentence = "Fact number one is stated here. "
    overlong = sentence * (SNIPPET_MAX_CHARS // len(sentence) + 4)
    assert len(overlong) > SNIPPET_MAX_CHARS
    body = {"choices": [{"message": {"content": overlong}}]}
    with respx.mock:
        respx.post(_LLM_URL).mock(return_value=httpx.Response(200, json=body))
        pairs, _stats = await ground_results(
            "q", [_result("https://a.example")], ctx
        )

    emitted = pairs[0][0].snippets[0]
    assert pairs[0][1] == "grounded"
    assert len(emitted) <= SNIPPET_MAX_CHARS
    assert emitted.endswith("here.")
    await client.aclose()


async def test_character_cap_preserves_coverage_after_cut_code_fence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An overlong fenced body cannot consume the final coverage line."""
    from jasa.grounding.prompts import SNIPPET_MAX_CHARS

    async def fake_fetch(engine: object, url: str) -> _FetchResult:
        return _FetchResult("c" * 400, "Title")

    monkeypatch.setattr("jasa.grounding.service.execute_web_fetch", fake_fetch)
    ctx, client = _ctx()
    coverage = "Coverage: answers x; does NOT cover y."
    overlong = "```text\n" + "x" * SNIPPET_MAX_CHARS + f"\n```\n\n{coverage}"
    body = {
        "choices": [{"message": {"content": overlong}, "finish_reason": "stop"}]
    }
    with respx.mock:
        respx.post(_LLM_URL).mock(return_value=httpx.Response(200, json=body))
        pairs, _stats = await ground_results(
            "q", [_result("https://a.example")], ctx
        )

    emitted = pairs[0][0].snippets[0]
    assert pairs[0][1] == "grounded"
    assert emitted.endswith(coverage)
    assert emitted.count("```") == 2
    assert len(emitted) <= SNIPPET_MAX_CHARS
    await client.aclose()


async def test_character_cap_discards_excess_coverage_separator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Whitespace before coverage cannot force an otherwise short body cut."""
    from jasa.grounding.prompts import SNIPPET_MAX_CHARS

    async def fake_fetch(engine: object, url: str) -> _FetchResult:
        return _FetchResult("c" * 400, "Title")

    monkeypatch.setattr("jasa.grounding.service.execute_web_fetch", fake_fetch)
    ctx, client = _ctx()
    coverage = "Coverage: answers x; does NOT cover y."
    overlong = "Short body.\n" + "\n" * SNIPPET_MAX_CHARS + coverage
    body = {
        "choices": [{"message": {"content": overlong}, "finish_reason": "stop"}]
    }
    with respx.mock:
        respx.post(_LLM_URL).mock(return_value=httpx.Response(200, json=body))
        pairs, _stats = await ground_results(
            "q", [_result("https://a.example")], ctx
        )

    assert pairs[0][0].snippets == [f"Short body.\n\n{coverage}"]
    await client.aclose()


async def test_empty_selection_returns_empty_without_raising() -> None:
    """``asyncio.wait`` rejects an empty set; the stage must absorb that."""
    ctx, client = _ctx()

    pairs, stats = await ground_results("q", [], ctx)

    assert pairs == []
    assert stats.total_urls == 0
    assert stats.grounded_count == 0
    assert dict(stats.outcomes) == {}
    await client.aclose()


async def test_late_worker_declines_at_the_front_of_the_queue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A worker that waited out the stage must not pay for a doomed fetch.

    The pre-queue budget check says nothing about the time left once the
    worker reaches the front, which is where the spend actually happens.
    """
    fetched: list[str] = []
    release_first = asyncio.Event()

    async def fake_fetch(engine: object, url: str) -> _FetchResult:
        fetched.append(url)
        if url.endswith("/slow"):
            await release_first.wait()
        return _FetchResult("c" * 200, "Title")

    monkeypatch.setattr("jasa.grounding.service.execute_web_fetch", fake_fetch)
    monkeypatch.setattr(
        "jasa.grounding.service.MIN_WORKER_BUDGET_SECONDS", 0.30
    )
    settings = GroundingSettings(concurrency=1, top_n=2)
    client = httpx.AsyncClient()
    ctx = GroundingContext(
        engine=grounding_engine(),
        client=client,
        cache=MemoryCache(),
        cache_write_semaphore=asyncio.Semaphore(1),
        flights=GroundingFlightRegistry(),
        waterfall=single_tier_waterfall(settings, _KEY),
        config=settings,
    )
    results = [
        _result("https://a.example/slow"),
        _result("https://b.example/x"),
    ]

    async def free_the_slot() -> None:
        await asyncio.sleep(0.20)
        release_first.set()

    asyncio.get_running_loop().create_task(free_the_slot())
    deadline_at = asyncio.get_running_loop().time() + 0.35
    with respx.mock:
        respx.post(_LLM_URL).mock(return_value=_llm_ok("grounded"))
        pairs, _stats = await ground_results("q", results, ctx, deadline_at)

    assert fetched == ["https://a.example/slow"], (
        "the second worker paid for a fetch it had no budget to use"
    )
    assert pairs[1][1] == "fallback:pipeline_timeout"
    await client.aclose()


async def test_trimming_never_manufactures_a_sentinel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Post-processing reads the model's verdict; it must not create one.

    Substring sentinel matching applies only below 200 normalized characters,
    so trimming a longer answer that merely quotes a bracketed phrase could
    otherwise flip a paid, valid snippet to a sentinel fallback.
    """

    async def fake_fetch(engine: object, url: str) -> _FetchResult:
        return _FetchResult("c" * 400, "Title")

    monkeypatch.setattr("jasa.grounding.service.execute_web_fetch", fake_fetch)
    ctx, client = _ctx()
    quoted = (
        "The vendor documentation states that anonymous access returns the "
        "string [login required] in the response body, which callers must "
        "treat as an authentication failure rather than as page content, and "
        "the guide goes on at length about the retry behaviour involved here. "
    )
    text = quoted + "Then the generation was cut off mid-clause and"
    assert len(text) > _SENTINEL_SUBSTRING_LIMIT
    body = {
        "choices": [{"message": {"content": text}, "finish_reason": "length"}]
    }
    with respx.mock:
        respx.post(_LLM_URL).mock(return_value=httpx.Response(200, json=body))
        pairs, stats = await ground_results(
            "q", [_result("https://a.example")], ctx
        )

    result, outcome = pairs[0]
    assert outcome == "grounded", "trimming manufactured a sentinel verdict"
    assert result.snippets[0].endswith("here.")
    assert stats.grounded_count == 1
    await client.aclose()


@pytest.mark.parametrize(
    ("cut", "expected"),
    [
        ("First. Second. Then cut off", "First. Second."),
        ("完整句子。次の文。そして切れた", "完整句子。次の文。"),
        (
            "彼は「終わった。」と言った。まだ切れ",
            "彼は「終わった。」と言った。",
        ),
        ("“完整句子。” 然后被截断", "“完整句子。”"),
        ("Done!' Then cut", "Done!'"),
        ("終わり？ そして切れた", "終わり？"),  # noqa: RUF001
    ],
)
def test_trim_keeps_closing_marks_with_their_terminator(
    cut: str, expected: str
) -> None:
    """A trim must not strand the opening half of a quotation.

    Every closing mark that can legally follow a sentence terminator is
    consumed with it, in both the ASCII and the full-width branch.
    """
    from jasa.grounding.detectors import trim_truncated_snippet

    assert trim_truncated_snippet(cut) == expected


def test_snippet_cap_leaves_room_for_the_coverage_separator() -> None:
    """A maximal body plus a maximal Coverage line needs its newline too.

    Without the separator in the cap, the longest valid output loses its final
    character, reads as cut, and has the whole Coverage line trimmed away --
    the defect the larger cap exists to fix.
    """
    from jasa.grounding.prompts import (
        COVERAGE_LINE_MAX_CHARS,
        COVERAGE_SEPARATOR_MAX_CHARS,
        SNIPPET_BODY_MAX_CHARS,
        SNIPPET_MAX_CHARS,
    )

    body = "b" * SNIPPET_BODY_MAX_CHARS
    coverage = "Coverage: answers x; does NOT cover y.".ljust(
        COVERAGE_LINE_MAX_CHARS, "."
    )
    longest_valid = f"{body}\n{coverage}"
    assert len(longest_valid) <= SNIPPET_MAX_CHARS
    assert COVERAGE_SEPARATOR_MAX_CHARS >= 1


async def test_a_trim_that_would_read_as_a_sentinel_is_abandoned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The stored snippet must survive its own cache round-trip.

    The value is re-checked against the sentinel detector on read, so a trim
    that pushes a quoting snippet under the substring threshold would be
    written and then refused forever, repeating the paid call every time.
    """

    async def fake_fetch(engine: object, url: str) -> _FetchResult:
        return _FetchResult("c" * 400, "Title")

    monkeypatch.setattr("jasa.grounding.service.execute_web_fetch", fake_fetch)
    cache = _RecordingCache()
    ctx, client = _ctx(cache)
    quoted = (
        "Anonymous access returns [login required] in the body, which callers "
        "must treat as an authentication failure rather than page content. "
    )
    text = quoted + "x" * (_SENTINEL_SUBSTRING_LIMIT - len(quoted) + 40)
    assert len(text) > _SENTINEL_SUBSTRING_LIMIT
    body = {
        "choices": [{"message": {"content": text}, "finish_reason": "length"}]
    }
    with respx.mock:
        respx.post(_LLM_URL).mock(return_value=httpx.Response(200, json=body))
        pairs, _stats = await ground_results(
            "q", [_result("https://a.example")], ctx
        )

    result, outcome = pairs[0]
    assert outcome == "grounded"
    stored = json.loads(cache.write_calls[0][1])["snippet"]
    assert stored == result.snippets[0]
    assert (
        _deserialize_grounding_cache(
            json.loads(cache.write_calls[0][1]),
            grounding_cache_identity(
                _fetch_cache_identity("https://a.example"),
                "q",
                _SETTINGS.max_content_chars,
                ctx.waterfall.chain,
            ),
        )
        == stored
    ), "the accepted snippet was refused by its own cache read"
    await client.aclose()
