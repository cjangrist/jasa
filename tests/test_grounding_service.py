"""Grounding service: mocked fetch + LLM, outcome classification."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
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
    FETCHED_TITLE_MAX_CHARS,
    FREQUENCY_PENALTY,
    grounding_cache_identity,
    GROUNDING_CACHE_KEY_PREFIX,
    make_grounding_cache_key,
    MIN_SNIPPET_CHARS,
    TEMPERATURE,
    TOP_P,
)
from jasa.grounding.detectors import grounding_detector_semantics
from jasa.grounding.prompts import (
    build_grounded_user_message,
    GROUNDING_MAX_TOKENS,
    grounding_prompt_semantics,
    SNIPPET_MAX_CHARS,
)
from jasa.grounding.service import (
    ground_results,
    grounding_semantic_fingerprint,
    GROUNDING_SEMANTICS_VERSION,
    GroundingContext,
    MIN_CONTENT_CHARS,
)
from jasa.search.ranking import RankedWebResult
from omnifetch.cache import build_cache_backend

_SETTINGS = GroundingSettings()
_KEY = "cerebras-test"
_LLM_URL = "https://api.cerebras.ai/v1/chat/completions"


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
    def __init__(self, expected_writes: int) -> None:
        self.expected_writes = expected_writes
        self.write_count = 0
        self.all_writes_started = asyncio.Event()
        self.release_writes = asyncio.Event()

    async def get(self, key: str) -> None:
        return None

    async def set(self, key: str, value: str, ttl_seconds: int) -> bool:
        self.write_count += 1
        if self.write_count == self.expected_writes:
            self.all_writes_started.set()
        await self.release_writes.wait()
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
            engine=object(),
            client=client,
            cache=cache if cache is not None else MemoryCache(),
            api_key=api_key,
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
        "llm_base_url": _SETTINGS.llm_base_url,
        "llm_model": _SETTINGS.llm_model,
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

    assert grounding_semantic_fingerprint(_SETTINGS) == expected
    variants = {
        grounding_semantic_fingerprint(
            _SETTINGS.model_copy(update={"llm_base_url": "https://other"})
        ),
        grounding_semantic_fingerprint(
            _SETTINGS.model_copy(update={"llm_model": "other"})
        ),
        grounding_semantic_fingerprint(
            _SETTINGS.model_copy(update={"max_content_chars": 500})
        ),
        grounding_semantic_fingerprint(
            _SETTINGS.model_copy(update={"top_n": 3})
        ),
    }
    assert expected not in variants
    assert len(variants) == 4


def test_grounding_cache_key_hashes_every_effective_llm_input() -> None:
    identity = grounding_cache_identity("private effective input", _SETTINGS)
    key = make_grounding_cache_key(identity)
    variants = (
        replace(identity, user_message="other input"),
        replace(identity, system_prompt_sha256="other prompt"),
        replace(identity, llm_base_url="https://other.example/v1"),
        replace(identity, llm_model="other-model"),
        replace(identity, temperature=0.3),
        replace(identity, top_p=0.8),
        replace(identity, frequency_penalty=0.4),
        replace(identity, max_tokens=1024),
        replace(identity, postprocess_fingerprint="other semantics"),
        replace(identity, semantics_version=cast(Literal[1], 2)),
    )

    assert key.startswith(GROUNDING_CACHE_KEY_PREFIX)
    assert len(key) == len(GROUNDING_CACHE_KEY_PREFIX) + 64
    assert "private effective input" not in key
    assert make_grounding_cache_key(identity) == key
    assert len({make_grounding_cache_key(item) for item in variants}) == 10
    assert key not in {make_grounding_cache_key(item) for item in variants}
    assert "api_key" not in asdict(identity)


def test_grounding_cache_key_uses_exact_truncated_user_message() -> None:
    max_chars = 100
    base_content = "a" * max_chars

    def key(query: str, title: str, content: str) -> str:
        message = build_grounded_user_message(
            query,
            title,
            content,
            max_chars,
        )
        return make_grounding_cache_key(
            grounding_cache_identity(message, _SETTINGS)
        )

    base_key = key("query", "Title", base_content)
    assert base_key != key("other query", "Title", base_content)
    assert base_key != key("query", "Other title", base_content)
    assert base_key != key("query", "Title", "b" + base_content[1:])
    assert key("query", "Title", base_content + "first suffix") == key(
        "query", "Title", base_content + "second suffix"
    )


def test_grounding_cache_record_is_strict_and_identity_bound() -> None:
    identity = grounding_cache_identity("effective input", _SETTINGS)
    serialized = _serialize_grounding_cache(identity, "accepted", "Title")
    valid = cast(dict[str, object], json.loads(serialized))

    assert _deserialize_grounding_cache(valid, identity, "Title") == "accepted"
    assert "effective input" not in serialized
    assert identity.system_prompt_sha256 not in serialized

    legacy = copy.deepcopy(cast(dict[str, object], valid["output"]))
    wrong_version = {**copy.deepcopy(valid), "schema_version": 2}
    top_extra = {**copy.deepcopy(valid), "unexpected": True}
    identity_drift = copy.deepcopy(valid)
    identity_drift["identity_digest"] = "0" * 64
    malformed_digest = copy.deepcopy(valid)
    malformed_digest["identity_digest"] = "not-a-digest"
    output_extra = copy.deepcopy(valid)
    cast(dict[str, object], output_extra["output"])["extra"] = True
    wrong_type = copy.deepcopy(valid)
    cast(dict[str, object], wrong_type["output"])["snippet"] = 7
    empty = copy.deepcopy(valid)
    cast(dict[str, object], empty["output"])["snippet"] = ""
    overlong = copy.deepcopy(valid)
    cast(dict[str, object], overlong["output"])["snippet"] = "x" * 2005
    wrong_title = copy.deepcopy(valid)
    cast(dict[str, object], wrong_title["output"])["fetched_title"] = "Other"
    overlong_title = copy.deepcopy(valid)
    cast(dict[str, object], overlong_title["output"])["fetched_title"] = "x" * (
        FETCHED_TITLE_MAX_CHARS + 1
    )
    unbalanced = copy.deepcopy(valid)
    cast(dict[str, object], unbalanced["output"])["snippet"] = "```python"
    sentinel = copy.deepcopy(valid)
    cast(dict[str, object], sentinel["output"])["snippet"] = (
        "[no usable content]"
    )

    cases = (
        legacy,
        wrong_version,
        top_extra,
        identity_drift,
        malformed_digest,
        output_extra,
        wrong_type,
        empty,
        overlong,
        wrong_title,
        overlong_title,
        unbalanced,
        sentinel,
        "not a record",
    )
    assert all(
        _deserialize_grounding_cache(case, identity, "Title") is None
        for case in cases
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
            "q", [_result("https://x.com")], ctx
        )
        cached_pairs, cached_stats = await ground_results(
            "q", [_result("https://x.com")], ctx
        )
    assert pairs[0][1] == "grounded"
    assert pairs[0][0].snippet_source == "grounded"
    assert pairs[0][0].title == "Title"
    assert stats.grounded_count == 1
    assert cached_pairs == pairs
    assert cached_stats == stats
    assert route.call_count == 1
    assert len(cache.read_keys) == 2
    assert len(cache.write_calls) == 1
    assert cache.write_calls[0][2] == 321
    assert (
        json.loads(route.calls.last.request.content)["max_tokens"]
        == GROUNDING_MAX_TOKENS
    )
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
    message = build_grounded_user_message(
        "q", title, content, _SETTINGS.max_content_chars
    )
    key = make_grounding_cache_key(grounding_cache_identity(message, _SETTINGS))
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
    assert cache.read_keys == [key]
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


async def test_overlong_fetched_title_skips_cache_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    oversized_title = "x" * (FETCHED_TITLE_MAX_CHARS + 1)

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
    assert route.call_count == 2
    assert cache.write_calls == []
    await client.aclose()


async def test_slow_grounding_cache_read_continues_to_llm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_fetch(engine: object, url: str) -> _FetchResult:
        return _FetchResult("Real content. " * 20, "Title")

    monkeypatch.setattr("jasa.grounding.service.execute_web_fetch", fake_fetch)
    cache = _SlowReadCache()
    context, client = _ctx(cast(CacheBackend, cache))
    with respx.mock:
        route = respx.post(_LLM_URL).mock(return_value=_llm_ok("Grounded"))
        pairs, stats = await ground_results("q", [_result("u")], context)

    assert cache.read_started.is_set()
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
    async def fake_fetch(engine: object, url: str) -> _FetchResult:
        return _FetchResult("Real content. " * 20, "Title")

    monkeypatch.setattr("jasa.grounding.service.execute_web_fetch", fake_fetch)
    cache = _BlockingWriteCache(expected_writes=2)
    settings = GroundingSettings(concurrency=1)
    context, client = _ctx(cast(CacheBackend, cache), settings=settings)
    with respx.mock:
        route = respx.post(_LLM_URL).mock(return_value=_llm_ok("Grounded"))
        task = asyncio.create_task(
            ground_results("q", [_result("a"), _result("b")], context)
        )
        async with asyncio.timeout(1):
            await cache.all_writes_started.wait()
        assert route.call_count == 2
        cache.release_writes.set()
        pairs, stats = await task

    assert [outcome for _, outcome in pairs] == ["grounded", "grounded"]
    assert stats.grounded_count == 2
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
    overlong = "x" * 1980 + "\n```python\n" + "y" * 100
    ctx, client = _ctx()
    with respx.mock:
        respx.post(_LLM_URL).mock(return_value=_llm_ok(overlong))
        pairs, _ = await ground_results("q", [_result("u")], ctx)
    snippet = pairs[0][0].snippets[0]
    assert len(snippet) == 2004
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
        engine=object(),
        client=client,
        cache=cache,
        api_key=_KEY,
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

    monkeypatch.setattr("jasa.grounding.service._fetch_and_ground", reject)
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


async def test_total_urls_counts_only_processed_top_n(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processed_urls: list[str] = []

    async def fake_fetch(engine: object, url: str) -> _FetchResult:
        processed_urls.append(url)
        return _FetchResult("short")

    monkeypatch.setattr("jasa.grounding.service.execute_web_fetch", fake_fetch)
    settings = GroundingSettings(top_n=2)
    client = httpx.AsyncClient()
    ctx = GroundingContext(
        engine=object(),
        client=client,
        cache=MemoryCache(),
        api_key=_KEY,
        config=settings,
    )
    results = [_result(f"https://{index}.example") for index in range(4)]
    pairs, stats = await ground_results("q", results, ctx)
    assert len(pairs) == 2
    assert stats.total_urls == 2
    assert processed_urls == ["https://0.example", "https://1.example"]
    await client.aclose()
