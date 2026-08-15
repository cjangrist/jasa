"""Composition: mount omnifetch in-process (§13.6 core invariants)."""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import AsyncMock

import httpx
import pytest
import respx
from fastmcp import Client
from starlette.testclient import TestClient

import jasa.server as server_module
from jasa.config import load_config
from jasa.server import _build_cache, build_composition
from omnifetch.cache import CacheBackend
from omnifetch.server import build_server as build_omnifetch_server


def _client_of(engine: Any) -> httpx.AsyncClient:
    """Return the engine's HTTP client (the shared client, by identity)."""
    return cast(httpx.AsyncClient, engine.client)


async def test_composed_tool_set_excludes_hello() -> None:
    composition = build_composition(load_config())
    tools = await composition.server.list_tools()
    assert {tool.name for tool in tools} == {"web_search", "web_fetch"}


@pytest.mark.parametrize("legacy_value", ["true", "false"])
async def test_removed_compat_fetch_flag_does_not_change_tool_set(
    monkeypatch: pytest.MonkeyPatch, legacy_value: str
) -> None:
    monkeypatch.setenv("JASA_COMPAT_FETCH_TOOL", legacy_value)
    composition = build_composition(load_config())
    tools = await composition.server.list_tools()
    assert {tool.name for tool in tools} == {"web_search", "web_fetch"}


async def test_say_hello_present_when_exposed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JASA_EXPOSE_HELLO", "true")
    composition = build_composition(load_config())
    tools = await composition.server.list_tools()
    assert "say_hello" in {tool.name for tool in tools}


def test_single_shared_client_identity() -> None:
    composition = build_composition(load_config())
    assert _client_of(composition.engine) is composition.client


def test_single_shared_cache_identity_and_fetch_ttl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JASA_FETCH_CACHE_TTL_SECONDS", "321")
    composition = build_composition(load_config())
    assert composition.engine.cache is composition.cache
    assert composition.engine.fetch_cache_ttl_seconds == 321
    assert composition.engine.owns_cache is False
    assert composition.engine.owns_client is False


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
    cache = AsyncMock(spec=CacheBackend)
    cache.is_ready.return_value = True
    monkeypatch.setattr(server_module, "_build_cache", lambda _config: cache)
    composition = build_composition(load_config())
    original_close = composition.client.aclose
    client_close = AsyncMock(side_effect=original_close)
    monkeypatch.setattr(composition.client, "aclose", client_close)

    with TestClient(composition.server.http_app()):
        pass

    cache.is_ready.assert_awaited_once()
    cache.close.assert_awaited_once()
    client_close.assert_awaited_once()
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
    assert cache.is_ready.await_count == 2
    cache.close.assert_awaited_once()
    assert composition.client.is_closed


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
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    composition = build_composition(load_config())
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
            result = await client.call_tool("web_search", {"query": "test"})
            cached = await client.call_tool("web_search", {"query": "test"})
    assert isinstance(result.data, dict)
    assert result.data["web_results"][0]["url"] == "https://x.com"
    assert cached.data == result.data
    assert route.call_count == 1
