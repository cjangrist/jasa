"""FastMCP server assembly for jasa -- in-process composition of omnifetch.

One process, one shared ``httpx.AsyncClient``, one shared cachelib backend,
one grounding flight registry, and one omnifetch ``Engine``. The engine is
given a cache identity built on Jasa's own ``normalize_url``, so the fetch
cache and search-result dedup agree on which URL spellings are the same page
rather than paying twice for a trailing slash. Credential-bearing and
non-ASCII-host URLs are excluded from that fold; see
``_fetch_cache_identity``. The child is
mounted unnamespaced, so its tool keeps the name ``web_fetch``. Its
``say_hello`` reference tool is suppressed unless
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
from urllib.parse import urlsplit

import httpx
from fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from jasa.assets import (
    build_icons,
    FAVICON_ICO_ROUTE,
    FAVICON_MEDIA_TYPE,
    FAVICON_PNG_ROUTE,
    ICON_MEDIA_TYPE,
    ICON_ROUTE,
    ICON_SIZES,
    read_favicon,
    read_icon,
)
from jasa.config import (
    AppConfig,
    CacheSettings,
    DEFAULT_FETCH_CACHE_TTL_SECONDS,
    DEFAULT_VOLATILE_FETCH_CACHE_TTL_SECONDS,
    load_config,
)
from jasa.grounding.flights import GroundingFlightRegistry
from jasa.grounding.service import GroundingContext
from jasa.grounding.waterfall import (
    grounding_credential_envs,
    GroundingChain,
    load_grounding_waterfall,
    resolve_grounding_waterfall,
)
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
from jasa.search.urls import normalize_url
from jasa.telemetry import shutdown_telemetry
from jasa.tools.web_search import format_web_search_response
from jasa.usage import UsageRefreshMiddleware, UsageRuntime
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


def grounding_enabled(mode: str, available_tiers: int) -> bool:
    """Return whether grounding is active for the mode and credentials."""
    if mode == "off":
        return False
    return available_tiers > 0


def available_grounding_tiers(chain: GroundingChain) -> int:
    """Count the waterfall tiers whose credential is present right now."""
    return len(resolve_grounding_waterfall(chain, os.environ).chain)


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


def _fetch_cache_identity(url: str) -> str:
    """Return one fetch URL's cache identity, folding only where it is safe.

    ``normalize_url`` exists for search dedup, where merging two spellings
    costs at most a duplicate row. A fetch entry is content, so the same fold
    decides whose response a later caller receives. Two cases are therefore
    left unfolded and keyed verbatim, exactly as they were before:

    Userinfo, because ``normalize_url`` tests the username for truthiness and
    so drops a password-only credential, mapping ``https://:one@host/p``,
    ``https://:two@host/p``, and the unauthenticated URL onto one entry -- one
    caller's private page answering another's request for the whole TTL.

    A non-ASCII host, because the IDNA 2003 mapping behind ``normalize_url``
    folds ``faß.de`` onto ``fass.de`` while the HTTP client treats them as the
    separate origins they are.

    An empty but present query, because ``urlsplit`` keeps no record of the
    delimiter and ``normalize_url`` re-emits one only for a truthy query, so
    ``https://host/x?`` folds onto ``https://host/x`` while the provider is
    still sent the request target that was asked for.

    Under-folding only costs a second fetch; over-folding hands one URL's
    content to another. Anything not provably one page therefore keys verbatim.
    """
    try:
        parts = urlsplit(url)
        has_userinfo = parts.username is not None or parts.password is not None
        host = None if has_userinfo else parts.hostname
    except ValueError:
        return url
    if has_userinfo:
        return url
    if host is not None and not host.isascii():
        return url
    if not parts.query and "?" in url.split("#", 1)[0]:
        return url
    return normalize_url(url)


def _omnifetch_child_config(
    secrets: ProviderSecrets,
    *,
    fetch_cache_ttl_seconds: int = DEFAULT_FETCH_CACHE_TTL_SECONDS,
    volatile_fetch_cache_ttl_seconds: int = (
        DEFAULT_VOLATILE_FETCH_CACHE_TTL_SECONDS
    ),
) -> OmnifetchAppConfig:
    """Build child config without exposing omnifetch runtime env knobs."""
    return OmnifetchAppConfig(
        server=OmnifetchServerSettings.model_construct(
            rest_web_fetch=False,
            fetch_cache_ttl_seconds=fetch_cache_ttl_seconds,
            volatile_fetch_cache_ttl_seconds=volatile_fetch_cache_ttl_seconds,
        ),
        telemetry=OmnifetchTelemetrySettings.model_construct(),
        providers=secrets,
    )


def register_icon_routes(server: FastMCP) -> None:
    """Register the parent-owned icon and favicon routes.

    The icon is served from the server's own origin as well as declared in
    ``serverInfo.icons``, because a client that renders one may look in either
    place: the specified field, or a favicon at the origin. Both answer from the
    same packaged bytes. ``?size=`` selects a declared square; anything else
    falls back to the largest, so a stale or hand-written link still resolves to
    an image rather than an error.

    The requested size is matched as a string rather than parsed as a number.
    ``str.isdigit`` is true for characters ``int`` refuses, such as ``²``, and
    ``int`` also rejects a decimal string beyond its conversion limit -- either
    would turn this public route's documented fallback into a 500.
    """
    icons = {str(size): read_icon(size) for size in ICON_SIZES}
    largest = icons[str(max(ICON_SIZES))]
    favicon_bytes = read_favicon()

    async def icon(request: Request) -> Response:
        body = icons.get(request.query_params.get("size", ""), largest)
        return Response(content=body, media_type=ICON_MEDIA_TYPE)

    async def favicon_ico(_request: Request) -> Response:
        return Response(content=favicon_bytes, media_type=FAVICON_MEDIA_TYPE)

    for path in (ICON_ROUTE, FAVICON_PNG_ROUTE):
        server.custom_route(path, methods=["GET"], include_in_schema=False)(
            icon
        )
    server.custom_route(
        FAVICON_ICO_ROUTE, methods=["GET"], include_in_schema=False
    )(favicon_ico)


def register_health_route(
    server: FastMCP,
    config: AppConfig,
    *,
    search_providers: list[str] | None = None,
    fetch_providers: list[str] | None = None,
    readiness: CacheReadiness,
    grounding_chain: GroundingChain = (),
) -> None:
    """Register the parent-owned aggregate ``/health`` and ``/`` routes."""
    resolved_search = search_providers if search_providers is not None else []
    resolved_fetch = fetch_providers if fetch_providers is not None else []

    async def health(_request: Request) -> JSONResponse:
        payload = build_health_payload(
            search_providers=resolved_search,
            fetch_providers=resolved_fetch,
            grounding_on=grounding_enabled(
                config.grounding.mode,
                available_grounding_tiers(grounding_chain),
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
    grounding_chain: GroundingChain,
) -> None:
    """Register the ``web_search`` tool, a thin adapter over ``run_search``."""
    grounding_cache_write_semaphore = asyncio.Semaphore(
        config.grounding.concurrency
    )
    grounding_flights = GroundingFlightRegistry()
    grounding_credentials = ", ".join(
        grounding_credential_envs(grounding_chain)
    )

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
        waterfall = resolve_grounding_waterfall(grounding_chain, os.environ)
        if validated.grounded_snippets is True and not waterfall.chain:
            raise ValueError(
                "grounded_snippets=true requires a grounding waterfall "
                f"credential ({grounding_credentials}) to be set"
            )
        want_grounding = (
            validated.grounded_snippets
            if validated.grounded_snippets is not None
            else bool(waterfall.chain) and config.grounding.mode != "off"
        )
        grounding_ctx = None
        if want_grounding and waterfall.chain:
            grounding_ctx = GroundingContext(
                engine=engine,
                client=client,
                cache=search.cache,
                cache_write_semaphore=grounding_cache_write_semaphore,
                flights=grounding_flights,
                waterfall=waterfall,
                config=config.grounding,
                cache_ttl_seconds=config.cache.grounding_ttl_seconds,
            )
        options = SearchOptions(
            timeout_ms=validated.timeout_ms or config.search.timeout_ms,
            fanout_timeout_ms=config.search.fanout_timeout_ms,
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
    usage: UsageRuntime


def _build_lifespan(
    cache: SharedCacheBackend,
    client: httpx.AsyncClient,
    readiness: CacheReadiness,
    usage: UsageRuntime,
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
                await usage.close()
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
) -> tuple[dict[str, SearchProvider], Engine, FastMCP, ProviderSecrets]:
    """Build provider registries, borrowed engine, and mounted child."""
    secrets = ProviderSecrets.from_env()
    providers = load_search_providers(secrets, client)
    omnifetch_config = _omnifetch_child_config(
        secrets,
        fetch_cache_ttl_seconds=app_config.cache.fetch_ttl_seconds,
        volatile_fetch_cache_ttl_seconds=(
            app_config.cache.volatile_fetch_ttl_seconds
        ),
    )
    engine = build_engine(
        omnifetch_config,
        client=client,
        cache=cache,
        canonicalize_cache_url=_fetch_cache_identity,
    )
    child = build_omnifetch_server(
        config=omnifetch_config, engine=engine, own_engine=False
    )
    return providers, engine, child, secrets


def _build_parent_server(
    app_config: AppConfig,
    *,
    client: httpx.AsyncClient,
    cache: SharedCacheBackend,
    readiness: CacheReadiness,
    search: SearchRuntime,
    usage: UsageRuntime,
    engine: Engine,
    child: FastMCP,
) -> FastMCP:
    """Register the parent surfaces and mount the borrowed child server."""
    _LOGGER.info("Building server %r (version %s).", _NAME, _VERSION)
    public_url = app_config.server.public_url.strip()
    server: FastMCP = FastMCP(
        name=_NAME,
        version=_VERSION,
        instructions=_INSTRUCTIONS,
        icons=build_icons(public_url),
        website_url=public_url or None,
        strict_input_validation=True,
        mask_error_details=True,
        lifespan=_build_lifespan(cache, client, readiness, usage),
    )
    register_icon_routes(server)
    search_names = list(search.providers)
    fetch_names = list(engine.unified.active_names)
    grounding_chain = load_grounding_waterfall(app_config.grounding)
    register_health_route(
        server,
        app_config,
        search_providers=search_names,
        fetch_providers=fetch_names,
        readiness=readiness,
        grounding_chain=grounding_chain,
    )
    register_web_search_tool(
        server,
        search=search,
        engine=engine,
        client=client,
        config=app_config,
        grounding_chain=grounding_chain,
    )
    server.add_middleware(UsageRefreshMiddleware(usage))
    server.mount(child)
    if not app_config.composition.expose_hello:
        server.disable(names={_HELLO_TOOL})
    register_rest_routes(
        server,
        search,
        engine,
        usage,
    )
    register_provider_resources(
        server,
        search_names,
        fetch_names,
        app_config,
        readiness,
        grounding_chain=grounding_chain,
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
        providers, engine, child, secrets = _build_runtime(
            app_config, client, cache
        )
        readiness = CacheReadiness(cache)
        search = SearchRuntime(
            providers=providers,
            cache=cache,
            cache_ttl_seconds=app_config.cache.search_ttl_seconds,
            flights=SearchFlightRegistry(),
        )
        usage = UsageRuntime(
            client=client,
            cache=cache,
            secrets=secrets,
            ttl_seconds=app_config.cache.usage_ttl_seconds,
        )
        server = _build_parent_server(
            app_config,
            client=client,
            cache=cache,
            readiness=readiness,
            search=search,
            usage=usage,
            engine=engine,
            child=child,
        )
        return Composition(
            server, client, engine, providers, cache, search, usage
        )
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
