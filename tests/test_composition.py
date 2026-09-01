"""Composition: mount omnifetch in-process (§13.6 core invariants)."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
import respx
from fastmcp import Client
from starlette.testclient import TestClient

import jasa.server as server_module
import omnifetch.tools.fetch as fetch_module
from jasa.config import load_config
from jasa.server import (
    _build_cache,
    _fetch_cache_identity,
    build_composition,
    build_composition_async,
)
from jasa.usage import UsageRuntime
from omnifetch.cache import CacheBackend
from omnifetch.fetch.engine.race import (
    FetchExhaustionDetails,
    FetchRaceResult,
    ProviderAttemptFailure,
)
from omnifetch.fetch.shared.types import (
    ErrorType,
    FetchResult,
    ProviderError,
)
from omnifetch.server import build_server as build_omnifetch_server
from omnifetch.tools.fetch import execute_web_fetch


def _client_of(engine: Any) -> httpx.AsyncClient:
    """Return the engine's HTTP client (the shared client, by identity)."""
    return cast(httpx.AsyncClient, engine.client)


async def test_composed_tool_set_excludes_hello() -> None:
    composition = await build_composition_async(load_config())
    tools = await composition.server.list_tools()
    assert {tool.name for tool in tools} == {"web_search", "web_fetch"}


async def test_web_search_publishes_safe_tool_annotations() -> None:
    composition = await build_composition_async(load_config())
    tools = await composition.server.list_tools()
    search_tool = next(tool for tool in tools if tool.name == "web_search")

    assert search_tool.title == "Web Search (multi-provider RRF)"
    assert search_tool.annotations is not None
    assert search_tool.annotations.read_only_hint is True
    assert search_tool.annotations.destructive_hint is False
    assert search_tool.annotations.idempotent_hint is True
    assert search_tool.annotations.open_world_hint is True


async def test_web_search_publishes_strict_dereferenced_output_schema() -> None:
    composition = await build_composition_async(load_config())
    async with Client(composition.server) as client:
        tools = await client.list_tools()
    search_tool = next(tool for tool in tools if tool.name == "web_search")
    schema = search_tool.output_schema

    assert schema is not None
    assert "$ref" not in str(schema)
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert schema["required"] == [
        "query",
        "total_duration_ms",
        "providers_succeeded",
        "providers_failed",
        "grounding",
        "truncation",
        "web_results",
    ]
    properties = schema["properties"]
    assert properties["query"]["type"] == "string"
    assert properties["total_duration_ms"]["type"] == "integer"

    success = properties["providers_succeeded"]["items"]
    assert success["additionalProperties"] is False
    assert success["required"] == ["provider", "duration_ms"]
    assert success["properties"]["duration_ms"]["type"] == "integer"

    failure = properties["providers_failed"]["items"]
    assert failure["additionalProperties"] is False
    assert failure["required"] == ["provider", "error", "duration_ms"]

    grounding = properties["grounding"]
    assert grounding["additionalProperties"] is False
    assert grounding["required"] == [
        "requested",
        "attempted",
        "grounded",
        "outcomes",
    ]
    assert grounding["properties"]["outcomes"]["additionalProperties"] == {
        "type": "integer"
    }

    truncation = properties["truncation"]
    assert truncation["additionalProperties"] is False
    assert truncation["required"] == ["total_before", "kept", "rescued"]

    result = properties["web_results"]["items"]
    assert result["additionalProperties"] is False
    assert result["required"] == [
        "title",
        "url",
        "source_providers",
        "score",
        "snippet_source",
    ]
    assert result["properties"]["source_providers"]["items"] == {
        "type": "string"
    }
    assert result["properties"]["score"]["type"] == "number"
    assert result["properties"]["snippet_source"]["enum"] == [
        "aggregated",
        "grounded",
        "fallback",
    ]
    assert result["properties"]["snippets"]["anyOf"][0] == {
        "items": {"type": "string"},
        "type": "array",
    }


async def test_mounted_fetch_exhaustion_is_a_structured_mcp_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def missing_url(*_args: object, **_kwargs: object) -> FetchRaceResult:
        raise ProviderError(
            ErrorType.NOT_FOUND,
            "No provider returned content; reported not found",
            "waterfall",
            details=FetchExhaustionDetails(
                providers_attempted=("tavily",),
                providers_failed=(
                    ProviderAttemptFailure(
                        "tavily",
                        "Tavily extraction failed: 404 page not found",
                        4,
                        ErrorType.NOT_FOUND,
                    ),
                ),
            ),
        )

    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    monkeypatch.setattr(fetch_module, "run_fetch_race", missing_url)
    composition = await build_composition_async(load_config())

    async with Client(composition.server) as client:
        result = await client.call_tool(
            "web_fetch", {"url": "https://example.test/missing"}
        )

    assert result.is_error is False
    assert result.data.status == "not_found"
    assert result.data.providers_attempted == ["tavily"]
    assert result.data.providers_failed[0].error_type == "NOT_FOUND"


@pytest.mark.parametrize("legacy_value", ["true", "false"])
async def test_removed_compat_fetch_flag_does_not_change_tool_set(
    monkeypatch: pytest.MonkeyPatch, legacy_value: str
) -> None:
    monkeypatch.setenv("JASA_COMPAT_FETCH_TOOL", legacy_value)
    composition = await build_composition_async(load_config())
    tools = await composition.server.list_tools()
    assert {tool.name for tool in tools} == {"web_search", "web_fetch"}


async def test_say_hello_present_when_exposed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JASA_EXPOSE_HELLO", "true")
    composition = await build_composition_async(load_config())
    tools = await composition.server.list_tools()
    assert "say_hello" in {tool.name for tool in tools}


def test_single_shared_client_identity() -> None:
    composition = build_composition(load_config())
    assert _client_of(composition.engine) is composition.client


def test_single_shared_cache_identity_and_fetch_ttl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JASA_FETCH_CACHE_TTL_SECONDS", "321")
    monkeypatch.setenv("JASA_VOLATILE_FETCH_CACHE_TTL_SECONDS", "123")
    composition = build_composition(load_config())
    assert composition.engine.cache is composition.cache
    assert composition.engine.fetch_cache_ttl_seconds == 321
    assert composition.engine.volatile_fetch_cache_ttl_seconds == 123
    assert composition.engine.owns_cache is False
    assert composition.engine.owns_client is False


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("https://:one@example.com/p", "https://:two@example.com/p"),
        ("https://:one@example.com/p", "https://example.com/p"),
        ("https://alice:k@example.com/p", "https://bob:k@example.com/p"),
        ("https://user@example.com/p", "https://example.com/p"),
        ("https://faß.de/p", "https://fass.de/p"),
        ("https://ex.com/p", "https://xn--fa-hia.de/p"),
        ("https://example.com/x?", "https://example.com/x"),
        ("https://example.com/x?#f", "https://example.com/x"),
    ],
)
def test_unsafe_folds_keep_distinct_cache_identities(
    first: str, second: str
) -> None:
    """A fetch entry is content, so only provably-equal URLs may share one."""
    assert _fetch_cache_identity(first) != _fetch_cache_identity(second)


@pytest.mark.parametrize(
    ("spelling", "canonical"),
    [
        ("https://example.com/x/", "https://example.com/x"),
        ("https://EXAMPLE.com/x", "https://example.com/x"),
        ("https://example.com:443/x", "https://example.com/x"),
        ("https://example.com/a/../x", "https://example.com/x"),
        ("https://example.com/x#frag", "https://example.com/x"),
    ],
)
def test_safe_spellings_still_fold(spelling: str, canonical: str) -> None:
    assert _fetch_cache_identity(spelling) == canonical


def test_unparseable_url_keeps_its_own_identity() -> None:
    assert _fetch_cache_identity("http://[::1") == "http://[::1"


async def test_engine_uses_jasa_url_canonicalization() -> None:
    composition = await build_composition_async(load_config())
    try:
        spellings = (
            "HTTPS://Example.COM:443/a/../page#section",
            "https://example.com/page",
        )
        assert {
            composition.engine.canonicalize_cache_url(url) for url in spellings
        } == {_fetch_cache_identity(spellings[0])}
    finally:
        await composition.cache.close()
        await composition.client.aclose()


async def test_url_spellings_share_one_paid_fetch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One page spelled many ways is bought once, and sent as asked.

    The first request is deliberately the non-canonical one, so forwarding
    ``normalize_url(url)`` to the provider instead of the URL the caller gave
    would fail here rather than passing quietly.
    """
    races: list[str] = []

    async def counting_race(
        _dispatcher: object,
        url: str,
        *,
        provider: str | None = None,
        skip_providers: object = (),
    ) -> FetchRaceResult:
        races.append(url)
        return FetchRaceResult(
            requested_url=url,
            total_duration_ms=5,
            provider_used="fake",
            providers_attempted=("fake",),
            providers_failed=(),
            result=FetchResult(
                url=url,
                title="Title",
                content="Page content long enough to count. " * 5,
                source_provider="fake",
                metadata={},
            ),
        )

    requested = "https://EXAMPLE.com:443/a/../x/#fragment"
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    monkeypatch.setattr(fetch_module, "run_fetch_race", counting_race)
    composition = await build_composition_async(load_config())
    engine = composition.engine
    try:
        first = await execute_web_fetch(engine, requested)
        second = await execute_web_fetch(engine, "https://example.com/x")
        await execute_web_fetch(engine, "https://example.com/x/")
        await execute_web_fetch(engine, "https://example.com:443/x")
    finally:
        await composition.cache.close()
        await composition.client.aclose()

    assert races == [requested]
    assert first.content == second.content


def test_search_runtime_owns_composition_resources_by_identity() -> None:
    composition = build_composition(load_config())

    with TestClient(composition.server.http_app()):
        assert composition.search.providers is composition.providers
        assert composition.search.cache is composition.cache
        assert composition.usage.cache is composition.cache
        assert composition.usage.client is composition.client


def test_parent_health_route_wins() -> None:
    composition = build_composition(load_config())
    with TestClient(composition.server.http_app()) as client:
        body = client.get("/health").json()
    assert "search" in body and "fetch" in body
    assert body["status"] == "unavailable"


def test_child_web_fetch_route_is_gated_off() -> None:
    composition = build_composition(load_config())
    with TestClient(composition.server.http_app()) as client:
        response = client.post("/web_fetch", json={"url": "https://x.com"})
    assert response.status_code == 404


def test_own_engine_false_without_engine_raises() -> None:
    with pytest.raises(ValueError):
        build_omnifetch_server(own_engine=False)


def test_lifespan_closes_shared_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    telemetry_shutdowns: list[bool] = []
    monkeypatch.setattr(
        "jasa.server.shutdown_telemetry",
        lambda: telemetry_shutdowns.append(True),
    )
    composition = build_composition(load_config())
    with TestClient(composition.server.http_app()):
        pass
    assert composition.client.is_closed
    assert telemetry_shutdowns == [True]


def test_lifespan_closes_cache_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    close_order: list[str] = []
    cache = AsyncMock(spec=CacheBackend)
    cache.is_ready.return_value = True
    cache.close.side_effect = lambda: close_order.append("cache")
    monkeypatch.setattr(server_module, "_build_cache", lambda _config: cache)
    usage_close = AsyncMock(side_effect=lambda: close_order.append("usage"))
    monkeypatch.setattr(UsageRuntime, "close", usage_close)
    composition = build_composition(load_config())
    original_close = composition.client.aclose

    async def close_client() -> None:
        close_order.append("client")
        await original_close()

    client_close = AsyncMock(side_effect=close_client)
    monkeypatch.setattr(composition.client, "aclose", client_close)

    with TestClient(composition.server.http_app()):
        pass

    cache.is_ready.assert_awaited_once()
    usage_close.assert_awaited_once()
    cache.close.assert_awaited_once()
    client_close.assert_awaited_once()
    assert close_order == ["usage", "cache", "client"]
    assert composition.client.is_closed


@pytest.mark.parametrize("probe_raises", [False, True])
def test_lifespan_unready_cache_fails_open_and_closes_resources(
    monkeypatch: pytest.MonkeyPatch, probe_raises: bool
) -> None:
    cache = AsyncMock(spec=CacheBackend)
    if probe_raises:
        cache.is_ready.side_effect = RuntimeError("probe failed")
    else:
        cache.is_ready.return_value = False
    monkeypatch.setattr(server_module, "_build_cache", lambda _config: cache)
    composition = build_composition(load_config())

    with TestClient(composition.server.http_app()) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["cache"]["ready"] is False
    cache.is_ready.assert_awaited_once()
    cache.close.assert_awaited_once()
    assert composition.client.is_closed


def test_assembly_failure_rolls_back_resources_synchronously(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = AsyncMock(spec=CacheBackend)
    client = httpx.AsyncClient()
    original_close = client.aclose
    client_close = AsyncMock(side_effect=original_close)
    monkeypatch.setattr(client, "aclose", client_close)
    monkeypatch.setattr(server_module, "_build_shared_client", lambda: client)
    monkeypatch.setattr(server_module, "_build_cache", lambda _config: cache)
    monkeypatch.setattr(
        server_module,
        "_build_parent_server",
        MagicMock(side_effect=RuntimeError("parent assembly failed")),
    )

    with pytest.raises(RuntimeError, match="parent assembly failed"):
        build_composition(load_config())

    cache.close.assert_awaited_once()
    client_close.assert_awaited_once()
    assert client.is_closed


async def test_async_assembly_failure_rolls_back_on_owning_event_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owning_loop = asyncio.get_running_loop()
    cache = AsyncMock(spec=CacheBackend)
    client = httpx.AsyncClient()
    original_close = client.aclose

    async def close_client() -> None:
        assert asyncio.get_running_loop() is owning_loop
        await original_close()

    client_close = AsyncMock(side_effect=close_client)
    monkeypatch.setattr(client, "aclose", client_close)
    monkeypatch.setattr(server_module, "_build_shared_client", lambda: client)
    monkeypatch.setattr(server_module, "_build_cache", lambda _config: cache)
    monkeypatch.setattr(
        server_module,
        "_build_parent_server",
        MagicMock(side_effect=RuntimeError("parent assembly failed")),
    )

    with pytest.raises(RuntimeError, match="parent assembly failed"):
        await build_composition_async(load_config())

    cache.close.assert_awaited_once()
    client_close.assert_awaited_once()
    assert client.is_closed


async def test_sync_composition_rejects_active_loop_before_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    build_client = MagicMock()
    monkeypatch.setattr(server_module, "_build_shared_client", build_client)

    with pytest.raises(
        RuntimeError, match="await build_composition_async instead"
    ):
        build_composition(load_config())

    build_client.assert_not_called()


def test_cache_construction_failure_still_closes_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = httpx.AsyncClient()
    original_close = client.aclose
    client_close = AsyncMock(side_effect=original_close)
    monkeypatch.setattr(client, "aclose", client_close)
    monkeypatch.setattr(server_module, "_build_shared_client", lambda: client)
    monkeypatch.setattr(
        server_module,
        "_build_cache",
        MagicMock(side_effect=RuntimeError("cache construction failed")),
    )

    with pytest.raises(RuntimeError, match="cache construction failed"):
        build_composition(load_config())

    client_close.assert_awaited_once()
    assert client.is_closed


def test_rollback_failure_does_not_mask_assembly_error(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    cache = AsyncMock(spec=CacheBackend)
    cache.close.side_effect = OSError("cache close failed")
    client = httpx.AsyncClient()
    monkeypatch.setattr(server_module, "_build_shared_client", lambda: client)
    monkeypatch.setattr(server_module, "_build_cache", lambda _config: cache)
    monkeypatch.setattr(
        server_module,
        "_build_parent_server",
        MagicMock(side_effect=RuntimeError("parent assembly failed")),
    )

    with (
        caplog.at_level(logging.WARNING, logger="jasa.server"),
        pytest.raises(RuntimeError, match="parent assembly failed"),
    ):
        build_composition(load_config())

    assert client.is_closed
    assert "Parent resource rollback failed (OSError)" in caplog.messages


async def test_async_rollback_failure_is_observed(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    cache = AsyncMock(spec=CacheBackend)
    cache.close.side_effect = OSError("cache close failed")
    client = httpx.AsyncClient()
    original_close = client.aclose

    async def close_client() -> None:
        await original_close()

    monkeypatch.setattr(client, "aclose", close_client)
    monkeypatch.setattr(server_module, "_build_shared_client", lambda: client)
    monkeypatch.setattr(server_module, "_build_cache", lambda _config: cache)
    monkeypatch.setattr(
        server_module,
        "_build_parent_server",
        MagicMock(side_effect=RuntimeError("parent assembly failed")),
    )

    with (
        caplog.at_level(logging.WARNING, logger="jasa.server"),
        pytest.raises(RuntimeError, match="parent assembly failed"),
    ):
        await build_composition_async(load_config())

    assert client.is_closed
    assert "Parent resource rollback failed (OSError)" in caplog.messages


@pytest.mark.parametrize("backend", ["memory", "disk", "redis"])
def test_build_cache_delegates_backend_selection(
    monkeypatch: pytest.MonkeyPatch, backend: str
) -> None:
    expected = AsyncMock(spec=CacheBackend)
    captured: dict[str, object] = {}

    def fake_build(selected: str, **kwargs: object) -> CacheBackend:
        captured["backend"] = selected
        captured.update(kwargs)
        return cast(CacheBackend, expected)

    monkeypatch.setattr(server_module, "build_cache_backend", fake_build)
    monkeypatch.setenv("JASA_CACHE_BACKEND", backend)
    monkeypatch.setenv("JASA_DISK_CACHE_PATH", "/cache")
    monkeypatch.setenv("JASA_REDIS_URL", "redis://cache:6379/0")
    monkeypatch.setenv("JASA_CACHE_MAX_ENTRIES", "123")

    assert _build_cache(load_config().cache) is expected
    assert captured == {
        "backend": backend,
        "disk_path": "/cache",
        "redis_url": "redis://cache:6379/0",
        "max_entries": 123,
    }


def test_build_cache_redis_requires_jasa_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JASA_CACHE_BACKEND", "redis")
    with pytest.raises(ValueError, match="JASA_REDIS_URL is required"):
        _build_cache(load_config().cache)


async def test_web_search_callable_through_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    progress_updates: list[tuple[float, float | None, str | None]] = []

    async def record_progress(
        progress: float,
        total: float | None,
        message: str | None,
    ) -> None:
        progress_updates.append((progress, total, message))

    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    composition = await build_composition_async(load_config())
    with respx.mock:
        route = respx.post("https://api.tavily.com/search").mock(
            return_value=httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "title": "T",
                            "url": "https://x.com",
                            "content": "c" * 320,
                            "score": 0.9,
                        }
                    ]
                },
            )
        )
        async with Client(composition.server) as client:
            result = await client.call_tool(
                "web_search",
                {"query": "test"},
                progress_handler=record_progress,
            )
            first_call_progress = list(progress_updates)
            progress_updates.clear()
            cached = await client.call_tool(
                "web_search",
                {"query": "test"},
                progress_handler=record_progress,
            )
    payload = result.structured_content
    assert payload is not None
    assert payload["web_results"][0]["url"] == "https://x.com"
    assert payload["web_results"][0]["snippet_source"] == "aggregated"
    assert cached.structured_content == payload
    assert route.call_count == 1
    assert [update[0] for update in first_call_progress] == [0, 10, 35, 90, 100]
    assert all(update[1] == 100 for update in first_call_progress)
    assert first_call_progress[1][2] == "Searching 1 providers"
    assert progress_updates == [
        (0, 100, "Checking search cache"),
        (100, 100, "Search complete from cache: 1 results"),
    ]
