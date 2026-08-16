"""End-to-end cachelib matrix across Jasa's public cache consumers."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import httpx
import pytest
from cachelib import file as cachelib_file
from cachelib import simple as cachelib_simple
from fastmcp import Client

import omnifetch.tools.fetch as fetch_module
from jasa.cache.base import make_cache_key, SearchCacheIdentity
from jasa.config import load_config
from jasa.grounding.service import grounding_semantic_fingerprint
from jasa.search.providers.base import SearchRequest
from jasa.search.providers.tavily import TavilyProvider
from jasa.search.ranking import SearchResult
from jasa.server import build_composition_async, Composition
from omnifetch.cache import build_cache_backend, CachelibBackend
from omnifetch.fetch.engine.race import FetchRaceResult
from omnifetch.fetch.shared.types import FetchResult

_QUERY = "cache surface matrix"
_GROUNDED_QUERY = "grounding cache surface matrix"
_URL = "https://example.test/cache-matrix"


def _configure_backend(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    backend: str,
) -> None:
    """Select one real cachelib backend and one dual-family provider."""
    monkeypatch.setenv("JASA_CACHE_BACKEND", backend)
    monkeypatch.setenv("JASA_DISK_CACHE_PATH", str(tmp_path / "cache"))
    monkeypatch.setenv("JASA_CACHE_MAX_ENTRIES", "100")
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")


def _install_controlled_upstreams(
    monkeypatch: pytest.MonkeyPatch,
    calls: dict[str, int],
) -> None:
    """Replace paid boundaries with deterministic, counted operations."""

    async def search(
        _provider: TavilyProvider,
        _request: SearchRequest,
    ) -> list[SearchResult]:
        calls["search"] += 1
        return [
            SearchResult(
                title="Cache matrix",
                url=_URL,
                snippet="Useful cache matrix evidence. " * 12,
                source_provider="tavily",
                score=0.9,
            )
        ]

    async def fetch(
        _dispatcher: object,
        url: str,
        *,
        provider: str | None = None,
        skip_providers: Iterable[str] = (),
    ) -> FetchRaceResult:
        assert provider is None
        assert tuple(skip_providers) == ()
        calls["fetch"] += 1
        return _fetch_race(url)

    monkeypatch.setattr(TavilyProvider, "search", search)
    monkeypatch.setattr(fetch_module, "run_fetch_race", fetch)


def _fetch_race(url: str) -> FetchRaceResult:
    """Return one successful upstream-equivalent fetch race."""
    return FetchRaceResult(
        requested_url=url,
        total_duration_ms=7,
        provider_used="tavily",
        providers_attempted=("tavily",),
        providers_failed=(),
        result=FetchResult(
            url=url,
            title="Fetched cache matrix",
            content="# Cache matrix\n\n" + "Persistent fetched content. " * 20,
            source_provider="tavily",
            metadata={"controlled": True},
        ),
    )


async def _exercise_public_surfaces(composition: Composition) -> None:
    """Call MCP, REST, and researcher paths sharing one search/fetch cache."""
    transport = httpx.ASGITransport(app=composition.server.http_app())
    async with (
        Client(composition.server) as mcp_client,
        httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as rest_client,
    ):
        mcp_search = await mcp_client.call_tool(
            "web_search",
            {"query": _QUERY, "grounded_snippets": False},
        )
        rest_search = await rest_client.post("/search", json={"query": _QUERY})
        researcher = await rest_client.get(
            "/researcher", params={"query": _QUERY}
        )
        rest_fetch = await rest_client.post("/fetch", json={"url": _URL})
        mcp_fetch = await mcp_client.call_tool("web_fetch", {"url": _URL})

    assert mcp_search.data["web_results"][0]["url"] == _URL
    assert rest_search.json()[0]["link"] == _URL
    assert researcher.json()[0]["href"] == _URL
    assert rest_fetch.json()["source_provider"] == "tavily"
    assert mcp_fetch.data.source_provider == "tavily"


async def _exercise_cached_mcp_reads(composition: Composition) -> None:
    """Read both public capabilities after application recreation."""
    async with Client(composition.server) as client:
        search = await client.call_tool(
            "web_search",
            {"query": _QUERY, "grounded_snippets": False},
        )
        fetch = await client.call_tool("web_fetch", {"url": _URL})
    assert search.data["web_results"][0]["url"] == _URL
    assert fetch.data.source_provider == "tavily"


@pytest.mark.parametrize(
    ("backend", "persists"),
    [("memory", False), ("disk", True)],
)
async def test_public_surface_reuse_and_recreation_matrix(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    backend: str,
    persists: bool,
) -> None:
    calls = {"search": 0, "fetch": 0}
    _configure_backend(monkeypatch, tmp_path, backend)
    _install_controlled_upstreams(monkeypatch, calls)

    first = await build_composition_async(load_config())
    assert isinstance(first.cache, CachelibBackend)
    await _exercise_public_surfaces(first)
    assert calls == {"search": 1, "fetch": 1}

    second = await build_composition_async(load_config())
    await _exercise_cached_mcp_reads(second)
    expected_calls = 1 if persists else 2
    assert calls == {"search": expected_calls, "fetch": expected_calls}


@pytest.mark.parametrize(
    ("backend", "persists"),
    [("memory", False), ("disk", True)],
)
async def test_cachelib_backend_readiness_persistence_and_expiry_matrix(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    backend: str,
    persists: bool,
) -> None:
    ticks = [1_000.0]
    monkeypatch.setattr(cachelib_simple, "time", lambda: ticks[0])
    monkeypatch.setattr(cachelib_file, "time", lambda: ticks[0])
    disk_path = str(tmp_path / backend)
    first = build_cache_backend(
        backend,
        disk_path=disk_path,
        redis_url="",
        max_entries=10,
    )
    assert await first.is_ready() is True
    assert await first.set("matrix:persistent", {"value": 1}, 30) is True
    await first.close()

    second = build_cache_backend(
        backend,
        disk_path=disk_path,
        redis_url="",
        max_entries=10,
    )
    expected = {"value": 1} if persists else None
    assert await second.get("matrix:persistent") == expected
    assert await second.set("matrix:expiring", "value", 1) is True
    assert await second.get("matrix:expiring") == "value"
    ticks[0] += 2
    assert await second.get("matrix:expiring") is None
    await second.close()


def _grounded_search_key(composition: Composition) -> str:
    """Build the outer search key whose removal exposes grounding reuse."""
    config = load_config()
    identity = SearchCacheIdentity(
        query=_GROUNDED_QUERY,
        skip_quality_filter=False,
        grounding=True,
        providers=tuple(composition.providers),
        grounding_fingerprint=grounding_semantic_fingerprint(config.grounding),
    )
    return make_cache_key(identity)


async def _call_grounded_search(client: Client[Any]) -> None:
    """Call the real MCP grounding path and verify grounded output."""
    result = await client.call_tool(
        "web_search",
        {"query": _GROUNDED_QUERY, "grounded_snippets": True},
    )
    row = result.data["web_results"][0]
    assert row["snippet_source"] == "grounded"
    assert row["snippets"] == ["Grounded cache matrix evidence."]


@pytest.mark.parametrize(
    ("backend", "persists"),
    [("memory", False), ("disk", True)],
)
async def test_grounding_reuse_and_recreation_matrix(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    backend: str,
    persists: bool,
) -> None:
    calls = {"search": 0, "fetch": 0, "llm": 0}
    _configure_backend(monkeypatch, tmp_path, backend)
    monkeypatch.setenv("CEREBRAS_API_KEY", "test-cerebras-key")
    _install_controlled_upstreams(monkeypatch, calls)

    async def llm(*_args: object) -> str:
        calls["llm"] += 1
        return "Grounded cache matrix evidence."

    monkeypatch.setattr("jasa.grounding.service._llm_call", llm)
    first = await build_composition_async(load_config())
    async with Client(first.server) as first_client:
        await _call_grounded_search(first_client)
        await _call_grounded_search(first_client)
        assert await first.cache.delete(_grounded_search_key(first)) is True
        await _call_grounded_search(first_client)
        assert calls == {"search": 2, "fetch": 1, "llm": 1}
        assert await first.cache.delete(_grounded_search_key(first)) is True

    second = await build_composition_async(load_config())
    async with Client(second.server) as second_client:
        await _call_grounded_search(second_client)
    expected_fetch_and_llm = 1 if persists else 2
    assert calls == {
        "search": 3,
        "fetch": expected_fetch_and_llm,
        "llm": expected_fetch_and_llm,
    }
