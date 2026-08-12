"""Composition: mount omnifetch in-process (§13.6 core invariants)."""

from __future__ import annotations

from typing import Any, cast

import httpx
import pytest
import respx
from fastmcp import Client
from starlette.testclient import TestClient

from jasa.config import load_config
from jasa.server import _build_cache, build_composition
from omnifetch.server import build_server as build_omnifetch_server


def _client_of(engine: Any) -> httpx.AsyncClient:
    """Return the engine's HTTP client (the shared client, by identity)."""
    return cast(httpx.AsyncClient, engine.client)


async def test_composed_tool_set_excludes_hello() -> None:
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


def test_build_cache_selects_backend() -> None:
    assert _build_cache(
        type("C", (), {"backend": "memory"})()
    ).__class__.__name__ == ("MemoryCache")
    assert (
        _build_cache(
            type("C", (), {"backend": "disk", "disk_path": ".cache/jasa"})()
        ).__class__.__name__
        == "DiskCache"
    )
    with pytest.raises(ValueError):
        _build_cache(type("C", (), {"backend": "redis"})())


async def test_web_search_callable_through_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    composition = build_composition(load_config())
    with respx.mock:
        respx.post("https://api.tavily.com/search").mock(
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
    assert isinstance(result.data, dict)
    assert result.data["web_results"][0]["url"] == "https://x.com"
