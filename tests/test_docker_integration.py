"""Container and real-Redis integration for the deployable cache contract.

Marked ``docker_integration``; runs only when ``JASA_RUN_DOCKER_TESTS=1`` (the
CI Docker job). The image probe verifies non-root startup and aggregate health.
The Redis case proves readiness, expiry, process recreation, and reuse through
REST search/fetch, researcher, MCP search/fetch, and internal grounding.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

import httpx
import pytest
from fastmcp import Client

import omnifetch.tools.fetch as fetch_module
from jasa.cache.base import make_cache_key, SearchCacheIdentity
from jasa.config import load_config
from jasa.grounding.service import grounding_semantic_fingerprint
from jasa.search.providers.base import SearchRequest
from jasa.search.providers.tavily import TavilyProvider
from jasa.search.ranking import SearchResult
from jasa.server import build_composition_async, Composition
from omnifetch.cache import CacheBackend
from omnifetch.fetch.engine.race import FetchRaceResult
from omnifetch.fetch.shared.types import FetchResult

docker = pytest.importorskip("docker")

_REDIS_IMAGE = (
    "redis@sha256:"
    "987c376c727652f99625c7d205a1cba3cb2c53b92b0b62aade2bd48ee1593232"
)
_MATRIX_QUERY = "redis cache surface matrix"
_MATRIX_GROUNDED_QUERY = "redis grounding cache surface matrix"
_MATRIX_URL = "https://example.test/redis-cache-matrix"


@pytest.mark.docker_integration
def test_container_health_unavailable() -> None:
    """Build the image, run it, and assert the aggregate health route."""
    if not os.environ.get("JASA_RUN_DOCKER_TESTS"):
        pytest.skip("set JASA_RUN_DOCKER_TESTS=1 to run container integration")

    client = docker.from_env()
    root = Path(__file__).resolve().parents[1]
    client.images.build(path=str(root), tag="jasa:test", rm=True)
    container = client.containers.run(
        "jasa:test", detach=True, ports={"8000/tcp": None}
    )
    try:
        container.reload()
        port = int(container.ports["8000/tcp"][0]["HostPort"])
        body = _wait_for_health(port)
        assert body["status"] == "unavailable"
        assert body["search"]["count"] == 0
        assert body["fetch"]["count"] == 0
        assert _container_user_is_nonroot(container)
    finally:
        container.remove(force=True)


@pytest.mark.docker_integration
def test_redis_cache_startup_round_trip_and_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prove Jasa starts on, uses, reports, and closes a real Redis cache."""
    if not os.environ.get("JASA_RUN_DOCKER_TESTS"):
        pytest.skip("set JASA_RUN_DOCKER_TESTS=1 to run container integration")

    client = docker.from_env()
    container = client.containers.run(
        _REDIS_IMAGE,
        detach=True,
        ports={"6379/tcp": None},
    )
    try:
        container.reload()
        port = int(container.ports["6379/tcp"][0]["HostPort"])
        monkeypatch.setenv("JASA_CACHE_BACKEND", "redis")
        monkeypatch.setenv("JASA_REDIS_URL", f"redis://127.0.0.1:{port}/0")
        monkeypatch.setenv("TAVILY_API_KEY", "test-key")
        monkeypatch.setenv("CEREBRAS_API_KEY", "test-cerebras-key")
        calls = {"search": 0, "fetch": 0, "llm": 0}
        _install_redis_matrix_upstreams(monkeypatch, calls)
        asyncio.run(_assert_redis_surface_matrix(calls))
    finally:
        container.remove(force=True)


def _install_redis_matrix_upstreams(
    monkeypatch: pytest.MonkeyPatch,
    calls: dict[str, int],
) -> None:
    """Replace paid boundaries while preserving all cache orchestration."""

    async def search(
        _provider: TavilyProvider,
        _request: SearchRequest,
    ) -> list[SearchResult]:
        calls["search"] += 1
        return [
            SearchResult(
                "Redis matrix",
                _MATRIX_URL,
                "Useful Redis cache evidence. " * 12,
                "tavily",
                0.9,
            )
        ]

    async def fetch(
        _dispatcher: object,
        url: str,
        *,
        provider: str | None = None,
        skip_providers: Iterable[str] = (),
    ) -> FetchRaceResult:
        assert provider is None and tuple(skip_providers) == ()
        calls["fetch"] += 1
        return _redis_fetch_race(url)

    async def llm(*_args: object) -> str:
        calls["llm"] += 1
        return "Grounded Redis cache evidence."

    monkeypatch.setattr(TavilyProvider, "search", search)
    monkeypatch.setattr(fetch_module, "run_fetch_race", fetch)
    monkeypatch.setattr("jasa.grounding.service._llm_call", llm)


def _redis_fetch_race(url: str) -> FetchRaceResult:
    """Return a deterministic successful fetch for the Redis matrix."""
    return FetchRaceResult(
        requested_url=url,
        total_duration_ms=7,
        provider_used="tavily",
        providers_attempted=("tavily",),
        providers_failed=(),
        result=FetchResult(
            url=url,
            title="Redis cache matrix",
            content="# Redis\n\n" + "Persistent cache content. " * 20,
            source_provider="tavily",
            metadata={"controlled": True},
        ),
    )


def _redis_grounded_search_key(composition: Composition) -> str:
    """Return the outer grounded-search key, leaving inner entries intact."""
    config = load_config()
    return make_cache_key(
        SearchCacheIdentity(
            _MATRIX_GROUNDED_QUERY,
            False,
            True,
            tuple(composition.providers),
            grounding_semantic_fingerprint(config.grounding),
        )
    )


async def _assert_first_redis_process(composition: Composition) -> None:
    """Populate Redis through REST, MCP, researcher, fetch, and grounding."""
    transport = httpx.ASGITransport(app=composition.server.http_app())
    async with (
        Client(composition.server) as mcp_client,
        httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as rest_client,
    ):
        search = await rest_client.post(
            "/search", json={"query": _MATRIX_QUERY}
        )
        researcher = await rest_client.get(
            "/researcher", params={"query": _MATRIX_QUERY}
        )
        await mcp_client.call_tool(
            "web_search",
            {"query": _MATRIX_QUERY, "grounded_snippets": False},
        )
        fetch = await rest_client.post("/fetch", json={"url": _MATRIX_URL})
        await mcp_client.call_tool("web_fetch", {"url": _MATRIX_URL})
        grounded = await mcp_client.call_tool(
            "web_search",
            {"query": _MATRIX_GROUNDED_QUERY, "grounded_snippets": True},
        )
        health = await rest_client.get("/health")
        resource = await mcp_client.read_resource("jasa://providers/status")
        assert await composition.cache.set("jasa:test:expiring", "value", 1)
        assert await composition.cache.delete(
            _redis_grounded_search_key(composition)
        )

    assert search.json()[0]["link"] == _MATRIX_URL
    assert researcher.json()[0]["href"] == _MATRIX_URL
    assert fetch.json()["source_provider"] == "tavily"
    assert grounded.data["web_results"][0]["snippet_source"] == "grounded"
    assert health.json()["cache"] == {"backend": "redis", "ready": True}
    assert json.loads(resource[0].text)["cache"]["ready"] is True


async def _assert_second_redis_process(composition: Composition) -> None:
    """Prove stored values survive Jasa recreation and one TTL expires."""
    transport = httpx.ASGITransport(app=composition.server.http_app())
    async with (
        Client(composition.server) as mcp_client,
        httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as rest_client,
    ):
        raw = await mcp_client.call_tool(
            "web_search",
            {"query": _MATRIX_QUERY, "grounded_snippets": False},
        )
        fetch = await mcp_client.call_tool("web_fetch", {"url": _MATRIX_URL})
        grounded = await mcp_client.call_tool(
            "web_search",
            {"query": _MATRIX_GROUNDED_QUERY, "grounded_snippets": True},
        )
        health = await rest_client.get("/health")
        expired = await composition.cache.get("jasa:test:expiring")

    assert raw.data["web_results"][0]["url"] == _MATRIX_URL
    assert fetch.data.source_provider == "tavily"
    assert grounded.data["web_results"][0]["snippet_source"] == "grounded"
    assert health.json()["cache"] == {"backend": "redis", "ready": True}
    assert expired is None


async def _assert_redis_surface_matrix(calls: dict[str, int]) -> None:
    """Exercise and recreate the whole Jasa process around one Redis."""
    first = await build_composition_async(load_config())
    await _wait_for_cache_readiness(first.cache)
    await _assert_cache_round_trip(first.cache)
    await _assert_first_redis_process(first)
    assert calls == {"search": 2, "fetch": 1, "llm": 1}
    assert first.client.is_closed

    await asyncio.sleep(1.1)
    second = await build_composition_async(load_config())
    await _wait_for_cache_readiness(second.cache)
    await _assert_second_redis_process(second)
    assert calls == {"search": 3, "fetch": 1, "llm": 1}
    assert second.client.is_closed


async def _wait_for_cache_readiness(
    cache: CacheBackend, timeout: float = 30.0
) -> None:
    """Wait until the ephemeral Redis backend answers its readiness probe."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if await cache.is_ready():
            return
        await asyncio.sleep(0.1)
    raise AssertionError("Redis cache never became ready")


async def _assert_cache_round_trip(cache: CacheBackend) -> None:
    """Write and read one value through Jasa's selected shared backend."""
    assert await cache.set("jasa:test:redis", "value", 30) is True
    assert await cache.get("jasa:test:redis") == "value"


def _wait_for_health(port: int, timeout: float = 30.0) -> dict[str, Any]:
    """Poll the health route until it answers or the timeout elapses."""
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/health", timeout=2
            ) as response:
                return cast(
                    dict[str, Any], json.loads(response.read().decode())
                )
        except (urllib.error.URLError, OSError, ValueError) as error:
            last_error = error
            time.sleep(0.5)
    raise AssertionError(f"health route never became ready: {last_error}")


def _container_user_is_nonroot(container: object) -> bool:
    """Return True when the entrypoint runs as the non-root app user."""
    exit_code, output = container.exec_run("id -u")  # type: ignore[attr-defined]
    uid = output.decode().strip() if isinstance(output, bytes) else str(output)
    return exit_code == 0 and uid == "10001"
