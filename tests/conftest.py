"""Shared test fixtures.

The environment-isolation fixture purges the UNION of both servers' secret sets
plus every ``JASA_``/``OMNIFETCH_``/``OTEL_`` server setting before each test,
so the developer's real local environment never leaks into the test run.
Explicit test-control flags remain available. The provider-registry invariant
test (Phase 3) asserts every provider-required secret name is a member of
``SECRET_ENV_NAMES``.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from types import SimpleNamespace

import httpx
import pytest

import jasa.grounding.cache as cache_module
from jasa.cache.base import CacheBackend
from jasa.config import GroundingSettings
from jasa.grounding.flights import GroundingFlightRegistry
from jasa.grounding.service import GroundingContext
from jasa.search.providers import KNOWN_SEARCH_SECRET_ENVS
from jasa.search.ranking import RankedWebResult

# Search-provider secrets (the jasa search family) -- the canonical set is the
# single source of truth in the providers package.
_SEARCH_SECRET_ENV = frozenset(KNOWN_SEARCH_SECRET_ENVS)

# Fetch-provider secrets (the mounted omnifetch family), transcribed from
# omnifetch's .env.example at the pinned commit. Expanded from the
# providers-registry characterization manifest in Phase 3.
_FETCH_SECRET_ENV = frozenset(
    {
        "TAVILY_API_KEY",
        "FIRECRAWL_API_KEY",
        "JINA_API_KEY",
        "YOU_API_KEY",
        "BRIGHT_DATA_API_KEY",
        "BRIGHT_DATA_ZONE",
        "LINKUP_API_KEY",
        "DIFFBOT_TOKEN",
        "SOCIAVAULT_API_KEY",
        "SPIDER_CLOUD_API_TOKEN",
        "SCRAPFLY_API_KEY",
        "SCRAPEGRAPHAI_API_KEY",
        "SCRAPE_DO_API_TOKEN",
        "SCRAPELESS_API_KEY",
        "OPENGRAPH_IO_API_KEY",
        "SCRAPINGBEE_API_KEY",
        "SCRAPERAPI_API_KEY",
        "ZYTE_API_KEY",
        "SCRAPINGANT_API_KEY",
        "OXYLABS_WEB_SCRAPER_USERNAME",
        "OXYLABS_WEB_SCRAPER_PASSWORD",
        "OLOSTEP_API_KEY",
        "DECODO_WEB_SCRAPING_API_KEY",
        "SCRAPPEY_API_KEY",
        "LEADMAGIC_API_KEY",
        "SERPAPI_API_KEY",
        "SUPADATA_API_KEY",
        "GITHUB_API_KEY",
        "KIMI_API_KEY",
    }
)

_GROUNDING_SECRET_ENV = frozenset({"CEREBRAS_API_KEY"})
_LEGACY_AUTH_ENV = frozenset({"OPENWEBUI_API_KEY", "OMNISEARCH_API_KEY"})
_DELETED_ENV = frozenset({"OMNIFETCH_ENDPOINT"})

SECRET_ENV_NAMES = (
    _SEARCH_SECRET_ENV
    | _FETCH_SECRET_ENV
    | _GROUNDING_SECRET_ENV
    | _LEGACY_AUTH_ENV
    | _DELETED_ENV
)

_SETTING_PREFIXES = ("JASA_", "OMNIFETCH_", "OTEL_")
_TEST_CONTROL_ENV_NAMES = frozenset({"JASA_RUN_DOCKER_TESTS"})


@pytest.fixture(autouse=True)
def isolate_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Purge every setting and secret for a clean test environment."""
    for name in list(os.environ):
        is_server_configuration = (
            name.startswith(_SETTING_PREFIXES) or name in SECRET_ENV_NAMES
        )
        if is_server_configuration and name not in _TEST_CONTROL_ENV_NAMES:
            monkeypatch.delenv(name, raising=False)


@pytest.fixture
async def http_client() -> AsyncIterator[httpx.AsyncClient]:
    """A fresh shared HTTP client for provider tests (respx-interceptable)."""
    async with httpx.AsyncClient() as client:
        yield client


@dataclass(slots=True)
class GroundingFlightHarness:
    """Build common flight inputs and close every test client on teardown."""

    events: list[dict[str, object]] = field(default_factory=list)
    _clients: list[httpx.AsyncClient] = field(default_factory=list)

    @staticmethod
    def result(url: str) -> RankedWebResult:
        """Return one stable aggregated search result."""
        return RankedWebResult("title", url, ["aggregate"], ["provider"], 0.1)

    @staticmethod
    def fetch_result(content: str, title: str = "Title") -> SimpleNamespace:
        """Return the minimal successful fetch shape used by grounding."""
        return SimpleNamespace(content=content, title=title)

    def context(
        self,
        cache: CacheBackend,
        flights: GroundingFlightRegistry,
        *,
        settings: GroundingSettings | None = None,
    ) -> GroundingContext:
        """Return a context whose client is owned by this harness."""
        resolved_settings = settings or GroundingSettings()
        client = httpx.AsyncClient()
        self._clients.append(client)
        return GroundingContext(
            engine=object(),
            client=client,
            cache=cache,
            cache_write_semaphore=asyncio.Semaphore(
                resolved_settings.concurrency
            ),
            flights=flights,
            api_key="test-key",
            config=resolved_settings,
        )

    @staticmethod
    async def wait_for_event(event: asyncio.Event) -> None:
        """Bound a synchronization wait so a regression fails promptly."""
        async with asyncio.timeout(1):
            await event.wait()

    @staticmethod
    async def wait_until(predicate: Callable[[], bool]) -> None:
        """Bound polling for an observable asynchronous condition."""
        async with asyncio.timeout(1):
            while not predicate():
                await asyncio.sleep(0)

    async def close(self) -> None:
        """Close all clients even when the test body raises an assertion."""
        await asyncio.gather(*(client.aclose() for client in self._clients))


@pytest.fixture
async def grounding_flights(
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[GroundingFlightHarness]:
    """Yield a shared harness with cache metrics captured and teardown armed."""
    harness = GroundingFlightHarness()
    monkeypatch.setattr(
        cache_module,
        "emit_grounding_cache_metric",
        lambda **fields: harness.events.append(fields),
    )
    try:
        yield harness
    finally:
        await harness.close()
