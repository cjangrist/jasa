"""FastMCP server assembly for jasa -- in-process composition of omnifetch.

One process, one shared ``httpx.AsyncClient``, one shared cachelib backend, and
one omnifetch ``Engine``. The child is mounted unnamespaced, so its tool keeps
the name ``web_fetch``. Its ``say_hello`` reference tool is suppressed unless
``JASA_EXPOSE_HELLO`` is set. Jasa owns the parent ``/health`` route; the
child's ``/web_fetch`` REST mirror is forced off (composed-mode security,
§3.5). The parent lifespan checks and closes the shared cache and client; the
child declines ownership of both.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import time
from collections.abc import (
    AsyncIterator,
    Callable,
    Mapping,
)
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version

import httpx
from fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse

from jasa.config import (
    AppConfig,
    CacheSettings,
    DEFAULT_FETCH_CACHE_TTL_SECONDS,
    load_config,
)
from jasa.grounding.service import GroundingContext
from jasa.logging import get_logger
from jasa.rest import register_provider_resources, register_rest_routes
from jasa.schemas import WebSearchInput
from jasa.search.providers import load_search_providers
from jasa.search.providers.base import SearchProvider
from jasa.search.service import (
    run_search,
    SearchFlightRegistry,
    SearchOptions,
    SearchRuntime,
)
from jasa.telemetry import shutdown_telemetry
from jasa.tools.web_search import format_web_search_response
from omnifetch.cache import build_cache_backend
from omnifetch.cache import CacheBackend as SharedCacheBackend
from omnifetch.config import AppConfig as OmnifetchAppConfig
from omnifetch.config import ServerSettings as OmnifetchServerSettings
from omnifetch.config import TelemetrySettings as OmnifetchTelemetrySettings
from omnifetch.fetch.engine.runtime import Engine
from omnifetch.fetch.shared.config import ProviderSecrets
from omnifetch.server import build_engine
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
_CACHE_READINESS_REFRESH_SECONDS = 5.0
_CACHE_READINESS_TIMEOUT_SECONDS = 1.0
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


def _build_cache(config: CacheSettings) -> SharedCacheBackend:
    """Build the selected shared cachelib backend."""
    if config.backend == "redis" and not config.redis_url.strip():
        raise ValueError(
            "JASA_REDIS_URL is required when JASA_CACHE_BACKEND=redis"
        )
    return build_cache_backend(
        config.backend,
        disk_path=config.disk_path,
        redis_url=config.redis_url,
        max_entries=config.max_entries,
    )


async def _cache_is_ready(
    cache: SharedCacheBackend,
    timeout_seconds: float = _CACHE_READINESS_TIMEOUT_SECONDS,
) -> bool:
    """Probe cache readiness without allowing a cache fault to escape."""
    try:
        async with asyncio.timeout(timeout_seconds):
            return bool(await cache.is_ready())
    except Exception as error:
        _LOGGER.warning(
            "Cache readiness probe failed (%s)", type(error).__name__
        )
        return False


@dataclass(slots=True)
class CacheReadiness:
    """Bounded, coalesced, briefly cached backend readiness state."""

    cache: SharedCacheBackend
    refresh_seconds: float = _CACHE_READINESS_REFRESH_SECONDS
    timeout_seconds: float = _CACHE_READINESS_TIMEOUT_SECONDS
    clock: Callable[[], float] = time.monotonic
    _ready: bool = field(default=False, init=False)
    _checked_at: float | None = field(default=None, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    async def current(self) -> bool:
        """Return recent readiness, running at most one bounded fresh probe."""
        now = self.clock()
        if self._checked_at is not None and (
            now - self._checked_at < self.refresh_seconds
        ):
            return self._ready
        async with self._lock:
            now = self.clock()
            if self._checked_at is not None and (
                now - self._checked_at < self.refresh_seconds
            ):
                return self._ready
            self._ready = await _cache_is_ready(
                self.cache, self.timeout_seconds
            )
            self._checked_at = self.clock()
            return self._ready


async def _close_parent_resources(
    cache: SharedCacheBackend | None,
    client: httpx.AsyncClient,
) -> None:
    """Close partially assembled parent-owned resources in full."""
    try:
        if cache is not None:
            await cache.close()
    finally:
        await client.aclose()


def _omnifetch_child_config(
    secrets: ProviderSecrets,
    *,
    fetch_cache_ttl_seconds: int = DEFAULT_FETCH_CACHE_TTL_SECONDS,
) -> OmnifetchAppConfig:
    """Build child config without exposing omnifetch runtime env knobs."""
    return OmnifetchAppConfig(
        server=OmnifetchServerSettings.model_construct(
            rest_web_fetch=False,
            fetch_cache_ttl_seconds=fetch_cache_ttl_seconds,
        ),
        telemetry=OmnifetchTelemetrySettings.model_construct(),
        providers=secrets,
    )


def register_health_route(
    server: FastMCP,
    config: AppConfig,
    *,
    search_providers: list[str] | None = None,
    fetch_providers: list[str] | None = None,
    readiness: CacheReadiness,
) -> None:
    """Register the parent-owned aggregate ``/health`` and ``/`` routes."""
    resolved_search = search_providers if search_providers is not None else []
    resolved_fetch = fetch_providers if fetch_providers is not None else []

    async def health(_request: Request) -> JSONResponse:
        payload = build_health_payload(
            search_providers=resolved_search,
            fetch_providers=resolved_fetch,
            grounding_on=grounding_enabled(
                config.grounding.mode, os.getenv("CEREBRAS_API_KEY")
            ),
            cache_backend=config.cache.backend,
            cache_ready=await readiness.current(),
        )
        return JSONResponse(payload)

    for path in ("/health", "/"):
        server.custom_route(path, methods=["GET"], include_in_schema=False)(
            health
        )


def register_web_search_tool(
    server: FastMCP,
    *,
    search: SearchRuntime,
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
                cache=search.cache,
                api_key=cerebras_key,
                config=config.grounding,
                cache_ttl_seconds=config.cache.grounding_ttl_seconds,
            )
        options = SearchOptions(
            timeout_ms=validated.timeout_ms or 30000,
            include_snippets=validated.include_snippets,
            want_grounding=want_grounding,
            grounding=grounding_ctx,
            cache_ttl_seconds=search.cache_ttl_seconds,
            flights=search.flights,
        )
        outcome = await run_search(
            search.providers,
            search.cache,
            validated.query,
            options=options,
        )
        return format_web_search_response(
            outcome, include_snippets=validated.include_snippets
        )


@dataclass(frozen=True, slots=True)
class Composition:
    """The assembled process: server plus the shared resources it owns."""

    server: FastMCP
    client: httpx.AsyncClient
    engine: Engine
    providers: Mapping[str, SearchProvider]
    cache: SharedCacheBackend
    search: SearchRuntime


def _build_lifespan(
    cache: SharedCacheBackend,
    client: httpx.AsyncClient,
    readiness: CacheReadiness,
) -> Callable[[FastMCP], AbstractAsyncContextManager[None]]:
    """Build the parent lifespan that owns shared resource shutdown."""

    @contextlib.asynccontextmanager
    async def lifespan(_server: FastMCP) -> AsyncIterator[None]:
        try:
            if not await readiness.current():
                _LOGGER.warning(
                    "Cache backend is not ready; continuing without cache."
                )
            yield
        finally:
            try:
                await cache.close()
            finally:
                try:
                    await client.aclose()
                finally:
                    shutdown_telemetry()

    return lifespan


def _build_runtime(
    app_config: AppConfig,
    client: httpx.AsyncClient,
    cache: SharedCacheBackend,
) -> tuple[dict[str, SearchProvider], Engine, FastMCP]:
    """Build provider registries, borrowed engine, and mounted child."""
    secrets = ProviderSecrets.from_env()
    providers = load_search_providers(secrets, client)
    omnifetch_config = _omnifetch_child_config(
        secrets,
        fetch_cache_ttl_seconds=app_config.cache.fetch_ttl_seconds,
    )
    engine = build_engine(omnifetch_config, client=client, cache=cache)
    child = build_omnifetch_server(
        config=omnifetch_config, engine=engine, own_engine=False
    )
    return providers, engine, child


def _build_parent_server(
    app_config: AppConfig,
    *,
    client: httpx.AsyncClient,
    cache: SharedCacheBackend,
    readiness: CacheReadiness,
    search: SearchRuntime,
    engine: Engine,
    child: FastMCP,
) -> FastMCP:
    """Register the parent surfaces and mount the borrowed child server."""
    _LOGGER.info("Building server %r (version %s).", _NAME, _VERSION)
    server: FastMCP = FastMCP(
        name=_NAME,
        version=_VERSION,
        instructions=_INSTRUCTIONS,
        strict_input_validation=True,
        mask_error_details=True,
        lifespan=_build_lifespan(cache, client, readiness),
    )
    search_names = list(search.providers)
    fetch_names = list(engine.unified.active_names)
    register_health_route(
        server,
        app_config,
        search_providers=search_names,
        fetch_providers=fetch_names,
        readiness=readiness,
    )
    register_web_search_tool(
        server,
        search=search,
        engine=engine,
        client=client,
        config=app_config,
    )
    server.mount(child)
    if not app_config.composition.expose_hello:
        server.disable(names={_HELLO_TOOL})
    register_rest_routes(
        server,
        search,
        engine,
    )
    register_provider_resources(
        server, search_names, fetch_names, app_config, readiness
    )
    _LOGGER.info("Server %r ready.", _NAME)
    return server


async def build_composition_async(
    config: AppConfig | None = None,
) -> Composition:
    """Assemble the composition with same-loop transactional rollback."""
    app_config = load_config() if config is None else config
    client = _build_shared_client()
    cache: SharedCacheBackend | None = None
    try:
        cache = _build_cache(app_config.cache)
        providers, engine, child = _build_runtime(app_config, client, cache)
        readiness = CacheReadiness(cache)
        search = SearchRuntime(
            providers=providers,
            cache=cache,
            cache_ttl_seconds=app_config.cache.search_ttl_seconds,
            flights=SearchFlightRegistry(),
        )
        server = _build_parent_server(
            app_config,
            client=client,
            cache=cache,
            readiness=readiness,
            search=search,
            engine=engine,
            child=child,
        )
        return Composition(server, client, engine, providers, cache, search)
    except BaseException:
        try:
            await _close_parent_resources(cache, client)
        except BaseException as error:
            _LOGGER.warning(
                "Parent resource rollback failed (%s)", type(error).__name__
            )
        raise


def build_composition(config: AppConfig | None = None) -> Composition:
    """Synchronously assemble outside an active event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(build_composition_async(config))
    raise RuntimeError(
        "build_composition cannot run inside an active event loop; "
        "await build_composition_async instead"
    )


def build_server(config: AppConfig | None = None) -> FastMCP:
    """Construct and return the composed FastMCP server."""
    return build_composition(config).server
