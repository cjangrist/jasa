"""FastMCP server assembly for jasa -- in-process composition of omnifetch.

One process, one shared ``httpx.AsyncClient``, one omnifetch ``Engine``. The
omnifetch child is mounted unnamespaced (its tool keeps the name ``web_fetch``);
its ``say_hello`` reference tool is suppressed unless ``JASA_EXPOSE_HELLO`` is
set. jasa owns the parent ``/health`` route; the child's ``/web_fetch`` REST
mirror is forced off (composed-mode security, §3.5). The parent lifespan closes
the single shared client; the child declines ownership so it never closes it.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, replace
from importlib.metadata import PackageNotFoundError, version

import httpx
from fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse

from jasa.cache.base import CacheBackend
from jasa.cache.disk import DiskCache
from jasa.cache.memory import MemoryCache
from jasa.config import AppConfig, CacheSettings, load_config
from jasa.grounding.service import GroundingContext
from jasa.logging import get_logger
from jasa.rest import register_provider_resources, register_rest_routes
from jasa.schemas import WebSearchInput
from jasa.search.providers import load_search_providers
from jasa.search.providers.base import SearchProvider
from jasa.search.service import run_search, SearchOptions
from jasa.telemetry import shutdown_telemetry
from jasa.tools.web_search import format_web_search_response
from omnifetch.config import AppConfig as OmnifetchAppConfig
from omnifetch.config import load_config as load_omnifetch_config
from omnifetch.fetch.shared.config import ProviderSecrets
from omnifetch.server import (
    build_engine,
)
from omnifetch.server import (
    build_server as build_omnifetch_server,
)

_LOGGER = get_logger("server")
_NAME = "jasa"
try:
    _VERSION = version("jasa")
except PackageNotFoundError:
    from jasa import __version__

    _VERSION = __version__
_INSTRUCTIONS = (
    "Jasa MCP server. Multi-provider web search with RRF ranking, optional "
    "grounded snippets, and an in-process omnifetch fetch tool."
)
_HTTP_MAX_CONNECTIONS = 100
_HTTP_MAX_KEEPALIVE_CONNECTIONS = 40
_HELLO_TOOL = "say_hello"
_WEB_SEARCH_TOOL = "web_search"
_WEB_SEARCH_DESCRIPTION = (
    "Search the web across multiple providers, fuse results with RRF ranking,"
    " and optionally ground snippets against fetched page content."
)


def derive_status(search_count: int, fetch_count: int) -> str:
    """Return the three-state health status from per-family active counts."""
    if search_count > 0 and fetch_count > 0:
        return "ok"
    if search_count > 0 or fetch_count > 0:
        return "degraded"
    return "unavailable"


def grounding_enabled(mode: str, cerebras_key: str | None) -> bool:
    """Return whether grounding is active for the given mode and key."""
    if mode == "off":
        return False
    return bool(cerebras_key)


def build_health_payload(
    *,
    search_providers: list[str],
    fetch_providers: list[str],
    grounding_on: bool,
    cache_backend: str,
    cache_ready: bool,
) -> dict[str, object]:
    """Build the aggregate health body as a pure function of its inputs."""
    search_count = len(search_providers)
    fetch_count = len(fetch_providers)
    return {
        "status": derive_status(search_count, fetch_count),
        "version": _VERSION,
        "search": {"providers": search_providers, "count": search_count},
        "fetch": {"providers": fetch_providers, "count": fetch_count},
        "grounding_enabled": grounding_on,
        "cache": {"backend": cache_backend, "ready": cache_ready},
    }


def _build_shared_client() -> httpx.AsyncClient:
    """Construct the single process-wide HTTP client."""
    limits = httpx.Limits(
        max_connections=_HTTP_MAX_CONNECTIONS,
        max_keepalive_connections=_HTTP_MAX_KEEPALIVE_CONNECTIONS,
    )
    return httpx.AsyncClient(http2=True, follow_redirects=True, limits=limits)


def _build_cache(config: CacheSettings) -> CacheBackend:
    """Select a cache backend from configuration."""
    backend = config.backend
    if backend == "memory":
        return MemoryCache()
    if backend == "disk":
        return DiskCache(config.disk_path)
    raise ValueError(f"Unsupported cache backend: {backend}")


def _cache_ready(backend: str) -> bool:
    """Cheap readiness flag for the health route (memory/disk are ready)."""
    return backend in ("memory", "disk")


def _omnifetch_child_config() -> OmnifetchAppConfig:
    """Build the omnifetch child config, REST fetch mirror forced off."""
    base = load_omnifetch_config()
    forced_server = base.server.model_copy(update={"rest_web_fetch": False})
    return replace(base, server=forced_server)


def register_health_route(
    server: FastMCP,
    config: AppConfig,
    *,
    search_providers: list[str] | None = None,
    fetch_providers: list[str] | None = None,
    cache_ready: bool | None = None,
) -> None:
    """Register the parent-owned aggregate ``/health`` and ``/`` routes."""
    resolved_search = search_providers if search_providers is not None else []
    resolved_fetch = fetch_providers if fetch_providers is not None else []
    resolved_cache_ready = (
        cache_ready
        if cache_ready is not None
        else _cache_ready(config.cache.backend)
    )

    async def health(_request: Request) -> JSONResponse:
        payload = build_health_payload(
            search_providers=resolved_search,
            fetch_providers=resolved_fetch,
            grounding_on=grounding_enabled(
                config.grounding.mode, os.getenv("CEREBRAS_API_KEY")
            ),
            cache_backend=config.cache.backend,
            cache_ready=resolved_cache_ready,
        )
        return JSONResponse(payload)

    for path in ("/health", "/"):
        server.custom_route(path, methods=["GET"], include_in_schema=False)(
            health
        )


def register_web_search_tool(
    server: FastMCP,
    *,
    providers: Mapping[str, SearchProvider],
    cache: CacheBackend,
    engine: object,
    client: httpx.AsyncClient,
    config: AppConfig,
) -> None:
    """Register the ``web_search`` tool, a thin adapter over ``run_search``."""

    @server.tool(name=_WEB_SEARCH_TOOL, description=_WEB_SEARCH_DESCRIPTION)
    async def web_search(
        query: str,
        timeout_ms: int | None = None,
        include_snippets: bool = True,
        grounded_snippets: bool | None = None,
    ) -> dict[str, object]:
        validated = WebSearchInput(
            query=query,
            timeout_ms=timeout_ms,
            include_snippets=include_snippets,
            grounded_snippets=grounded_snippets,
        )
        cerebras_key = os.getenv("CEREBRAS_API_KEY", "")
        if validated.grounded_snippets is True and not cerebras_key:
            raise ValueError(
                "grounded_snippets=true requires CEREBRAS_API_KEY to be set"
            )
        want_grounding = (
            validated.grounded_snippets
            if validated.grounded_snippets is not None
            else bool(cerebras_key) and config.grounding.mode != "off"
        )
        grounding_ctx = None
        if want_grounding and cerebras_key:
            grounding_ctx = GroundingContext(
                engine=engine,
                client=client,
                api_key=cerebras_key,
                config=config.grounding,
            )
        options = SearchOptions(
            timeout_ms=validated.timeout_ms or 30000,
            include_snippets=validated.include_snippets,
            want_grounding=want_grounding,
            grounding=grounding_ctx,
        )
        outcome = await run_search(
            providers, cache, validated.query, options=options
        )
        return format_web_search_response(
            outcome, include_snippets=validated.include_snippets
        )


@dataclass(frozen=True, slots=True)
class Composition:
    """The assembled process: server plus the shared resources it owns."""

    server: FastMCP
    client: httpx.AsyncClient
    engine: object
    providers: Mapping[str, SearchProvider]
    cache: CacheBackend


def build_composition(config: AppConfig | None = None) -> Composition:
    """Assemble the composed jasa server and its shared resources."""
    app_config = load_config() if config is None else config
    client = _build_shared_client()
    secrets = ProviderSecrets.from_env()
    providers = load_search_providers(secrets, client)
    cache = _build_cache(app_config.cache)
    omnifetch_config = _omnifetch_child_config()
    engine = build_engine(omnifetch_config, client=client)
    child = build_omnifetch_server(
        config=omnifetch_config, engine=engine, own_engine=False
    )

    @contextlib.asynccontextmanager
    async def lifespan(_server: FastMCP) -> AsyncIterator[None]:
        try:
            yield
        finally:
            await cache.close()
            await client.aclose()
            shutdown_telemetry()

    _LOGGER.info("Building server %r (version %s).", _NAME, _VERSION)
    server: FastMCP = FastMCP(
        name=_NAME,
        version=_VERSION,
        instructions=_INSTRUCTIONS,
        strict_input_validation=True,
        mask_error_details=True,
        lifespan=lifespan,
    )
    register_health_route(
        server,
        app_config,
        search_providers=list(providers),
        fetch_providers=list(engine.unified.active_names),
        cache_ready=_cache_ready(app_config.cache.backend),
    )
    register_web_search_tool(
        server,
        providers=providers,
        cache=cache,
        engine=engine,
        client=client,
        config=app_config,
    )
    server.mount(child)
    if not app_config.composition.expose_hello:
        server.disable(names={_HELLO_TOOL})
    register_rest_routes(server, providers, cache, engine)
    register_provider_resources(
        server, list(providers), list(engine.unified.active_names), app_config
    )
    _LOGGER.info("Server %r ready.", _NAME)
    return Composition(
        server=server,
        client=client,
        engine=engine,
        providers=providers,
        cache=cache,
    )


def build_server(config: AppConfig | None = None) -> FastMCP:
    """Construct and return the composed FastMCP server."""
    return build_composition(config).server
