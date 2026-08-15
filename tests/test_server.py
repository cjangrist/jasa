"""Health payload, status derivation, and the aggregate route."""

from __future__ import annotations

import asyncio
import importlib
import importlib.metadata
from importlib.metadata import PackageNotFoundError
from typing import Any, cast
from unittest.mock import AsyncMock

import httpx
import pytest
from fastmcp import FastMCP
from starlette.testclient import TestClient

import jasa.server as server_module
from jasa import __version__
from jasa.cache.memory import MemoryCache
from jasa.config import load_config
from jasa.search.service import (
    SearchFlightRegistry,
    SearchOptions,
    SearchOutcome,
    SearchRuntime,
)
from jasa.server import (
    build_health_payload,
    build_server,
    CacheReadiness,
    derive_status,
    grounding_enabled,
    register_health_route,
    register_web_search_tool,
)
from omnifetch.cache import CacheBackend


class _ToolServer:
    def __init__(self) -> None:
        self.function: Any = None

    def tool(self, **_kwargs: object) -> Any:
        def register(function: Any) -> Any:
            self.function = function
            return function

        return register


def _search_runtime() -> SearchRuntime:
    config = load_config()
    return SearchRuntime(
        providers={},
        cache=MemoryCache(),
        cache_ttl_seconds=config.cache.search_ttl_seconds,
        flights=SearchFlightRegistry(),
    )


def test_derive_status_three_states() -> None:
    assert derive_status(0, 0) == "unavailable"
    assert derive_status(1, 0) == "degraded"
    assert derive_status(0, 1) == "degraded"
    assert derive_status(1, 1) == "ok"
    assert derive_status(3, 2) == "ok"


def test_grounding_enabled_logic() -> None:
    assert grounding_enabled("off", "key") is False
    assert grounding_enabled("auto", None) is False
    assert grounding_enabled("auto", "") is False
    assert grounding_enabled("auto", "key") is True
    assert grounding_enabled("on", "key") is True


def test_health_payload_shape() -> None:
    payload = build_health_payload(
        search_providers=["tavily"],
        fetch_providers=["jina"],
        grounding_on=True,
        cache_backend="disk",
        cache_ready=True,
    )
    assert payload["status"] == "ok"
    assert payload["version"]
    assert payload["search"] == {"providers": ["tavily"], "count": 1}
    assert payload["fetch"] == {"providers": ["jina"], "count": 1}
    assert payload["grounding_enabled"] is True
    assert payload["cache"] == {"backend": "disk", "ready": True}


@pytest.mark.parametrize("path", ["/health", "/"])
def test_aggregate_health_route_zero_providers(path: str) -> None:
    server = build_server(load_config())
    with TestClient(server.http_app()) as client:
        response = client.get(path)
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "unavailable"
    assert body["search"] == {"providers": [], "count": 0}
    assert body["fetch"] == {"providers": [], "count": 0}
    assert body["grounding_enabled"] is False
    assert body["cache"] == {"backend": "memory", "ready": True}


def test_build_server_reads_env_when_config_omitted() -> None:
    server = build_server()
    with TestClient(server.http_app()) as client:
        assert client.get("/health").status_code == 200


def test_health_route_with_injected_providers() -> None:
    server = FastMCP(name="test")
    cache = AsyncMock(spec=CacheBackend)
    cache.is_ready.return_value = False
    register_health_route(
        server,
        load_config(),
        search_providers=["tavily", "brave"],
        fetch_providers=[],
        readiness=CacheReadiness(cache),
    )
    with TestClient(server.http_app()) as client:
        body = client.get("/health").json()
    assert body["status"] == "degraded"
    assert body["search"]["providers"] == ["tavily", "brave"]
    assert body["search"]["count"] == 2
    assert body["fetch"]["providers"] == []
    assert body["cache"]["ready"] is False
    cache.is_ready.assert_awaited_once()


def test_health_route_probes_actual_cache_readiness() -> None:
    server = FastMCP(name="test")
    cache = AsyncMock(spec=CacheBackend)
    cache.is_ready.return_value = False
    register_health_route(
        server, load_config(), readiness=CacheReadiness(cache)
    )

    with TestClient(server.http_app()) as client:
        body = client.get("/health").json()

    assert body["cache"] == {"backend": "memory", "ready": False}
    cache.is_ready.assert_awaited_once()


async def test_cache_readiness_reuses_result_until_refresh_boundary() -> None:
    ticks = [100.0]
    cache = AsyncMock(spec=CacheBackend)
    cache.is_ready.side_effect = [True, False]
    readiness = CacheReadiness(cache, refresh_seconds=5, clock=lambda: ticks[0])

    assert await readiness.current() is True
    ticks[0] = 104.999
    assert await readiness.current() is True
    ticks[0] = 105.0
    assert await readiness.current() is False
    assert cache.is_ready.await_count == 2


async def test_cache_readiness_coalesces_concurrent_probes() -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    cache = AsyncMock(spec=CacheBackend)

    async def probe() -> bool:
        started.set()
        await release.wait()
        return True

    cache.is_ready.side_effect = probe
    readiness = CacheReadiness(cache)
    first = asyncio.create_task(readiness.current())
    await started.wait()
    second = asyncio.create_task(readiness.current())
    await asyncio.sleep(0)
    release.set()

    assert list(await asyncio.gather(first, second)) == [True, True]
    cache.is_ready.assert_awaited_once()


async def test_cache_readiness_probe_timeout_fails_open() -> None:
    cache = AsyncMock(spec=CacheBackend)

    async def never_ready() -> bool:
        await asyncio.Event().wait()
        return True

    cache.is_ready.side_effect = never_ready
    readiness = CacheReadiness(cache, timeout_seconds=0.001)

    assert await readiness.current() is False
    cache.is_ready.assert_awaited_once()


def test_version_falls_back_when_package_metadata_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(_distribution_name: str) -> str:
        raise PackageNotFoundError("jasa")

    with monkeypatch.context() as scoped:
        scoped.setattr(importlib.metadata, "version", missing)
        reloaded = importlib.reload(server_module)
        assert __version__ == reloaded._VERSION
    importlib.reload(server_module)


async def test_explicit_grounding_requires_cerebras_key() -> None:
    server = _ToolServer()
    client = httpx.AsyncClient()
    register_web_search_tool(
        cast(FastMCP, server),
        search=_search_runtime(),
        engine=object(),
        client=client,
        config=load_config(),
    )
    with pytest.raises(ValueError, match="requires CEREBRAS_API_KEY"):
        await server.function("q", grounded_snippets=True)
    await client.aclose()


async def test_grounding_context_is_passed_to_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, SearchOptions] = {}

    async def fake_run_search(
        providers: object,
        cache: object,
        query: str,
        *,
        options: SearchOptions,
    ) -> SearchOutcome:
        captured["options"] = options
        return SearchOutcome(query, 0, [], [], [])

    monkeypatch.setenv("CEREBRAS_API_KEY", "test-key")
    monkeypatch.setenv("JASA_SEARCH_CACHE_TTL_SECONDS", "321")
    monkeypatch.setattr("jasa.server.run_search", fake_run_search)
    server = _ToolServer()
    client = httpx.AsyncClient()
    engine = object()
    search = _search_runtime()
    register_web_search_tool(
        cast(FastMCP, server),
        search=search,
        engine=engine,
        client=client,
        config=load_config(),
    )
    response = await server.function("q", grounded_snippets=True)
    grounding = captured["options"].grounding
    assert response["query"] == "q"
    assert grounding is not None
    assert grounding.engine is engine
    assert grounding.api_key == "test-key"
    assert captured["options"].cache_ttl_seconds == 321
    assert captured["options"].flights is search.flights
    await client.aclose()
