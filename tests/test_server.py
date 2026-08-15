"""Health payload, status derivation, and the aggregate route."""

from __future__ import annotations

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
from jasa.search.service import SearchOptions, SearchOutcome
from jasa.server import (
    build_health_payload,
    build_server,
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
        cache=cache,
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
    register_health_route(server, load_config(), cache=cache)

    with TestClient(server.http_app()) as client:
        body = client.get("/health").json()

    assert body["cache"] == {"backend": "memory", "ready": False}
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
        providers={},
        cache=MemoryCache(),
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
    monkeypatch.setattr("jasa.server.run_search", fake_run_search)
    server = _ToolServer()
    client = httpx.AsyncClient()
    engine = object()
    register_web_search_tool(
        cast(FastMCP, server),
        providers={},
        cache=MemoryCache(),
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
    await client.aclose()
