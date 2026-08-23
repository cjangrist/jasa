"""Shared test fixtures.

The environment-isolation fixture purges the UNION of both servers' secret sets,
the search adapters' optional setting names, and every
``JASA_``/``OMNIFETCH_``/``OTEL_`` server setting before each test, so the
developer's real local environment never leaks into the test run. Explicit
test-control flags remain available. The provider-registry invariant test
(Phase 3) asserts every provider-required secret name is a member of
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
from jasa.grounding.service import _TierResponse, GroundingContext
from jasa.grounding.waterfall import (
    GroundingChain,
    GroundingTier,
    load_grounding_waterfall,
    resolve_grounding_waterfall,
    ResolvedGroundingWaterfall,
)
from jasa.search.providers import (
    KNOWN_SEARCH_SECRET_ENVS,
    KNOWN_SEARCH_SETTING_ENVS,
)
from jasa.search.ranking import RankedWebResult

# Search-provider secrets (the jasa search family) -- the canonical set is the
# single source of truth in the providers package.
_SEARCH_SECRET_ENV = frozenset(KNOWN_SEARCH_SECRET_ENVS)

# Optional adapter settings. They activate nothing, but a developer's real
# gateway or model override must not reach a test either.
_SEARCH_SETTING_ENV = frozenset(KNOWN_SEARCH_SETTING_ENVS)

# Fetch-provider secrets (the mounted omnifetch family), transcribed from
# omnifetch's .env.example at the pinned commit. Expanded from the
# providers-registry characterization manifest in Phase 3.
_FETCH_SECRET_ENV = frozenset(
    {
        "TAVILY_API_KEY",
        "CRW_API_KEY",
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
_PURGED_ENV_NAMES = SECRET_ENV_NAMES | _SEARCH_SETTING_ENV
PRIMARY_TIER_ENV = "CEREBRAS_API_KEY"


def tier_answer(text: str, truncated: bool = False) -> _TierResponse:
    """Build what a patched ``_call_grounding_tier`` must hand the waterfall.

    The waterfall reads the stop reason alongside the text so a generation cut
    off at its token ceiling is not mistaken for a complete answer, so a double
    returning a bare string no longer satisfies the contract.
    """
    return _TierResponse(text, truncated)


def tier(
    name: str,
    base_url: str,
    model: str,
    *,
    api_key_env: str = PRIMARY_TIER_ENV,
    timeout_ms: int = 60000,
) -> GroundingTier:
    """Build one waterfall tier without going through the YAML loader."""
    return GroundingTier(
        name=name,
        base_url=base_url,
        model=model,
        timeout_ms=timeout_ms,
        api_key_env=api_key_env,
    )


def resolved_waterfall(
    chain: GroundingChain, *keys: str
) -> ResolvedGroundingWaterfall:
    """Pair an exact chain with one credential per distinct tier env."""
    envs = dict.fromkeys(entry.api_key_env for entry in chain)
    supplied = keys or ("test-key",)
    api_keys = {
        env: supplied[index % len(supplied)] for index, env in enumerate(envs)
    }
    return ResolvedGroundingWaterfall(chain=chain, api_keys=api_keys)


def resolved_grounding_chain(config: GroundingSettings) -> GroundingChain:
    """Return the chain the composition builds for the current environment."""
    return resolve_grounding_waterfall(
        load_grounding_waterfall(config), os.environ
    ).chain


def single_tier_waterfall(
    settings: GroundingSettings, api_key: str = "test-key"
) -> ResolvedGroundingWaterfall:
    """Return the one-tier chain matching pre-waterfall single-model runs."""
    return resolved_waterfall(
        (
            tier(
                "primary",
                settings.llm_base_url,
                settings.llm_model,
                timeout_ms=settings.llm_timeout_ms,
            ),
        ),
        api_key,
    )


@pytest.fixture(autouse=True)
def isolate_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Purge every setting and secret for a clean test environment."""
    for name in list(os.environ):
        is_server_configuration = (
            name.startswith(_SETTING_PREFIXES) or name in _PURGED_ENV_NAMES
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
            waterfall=single_tier_waterfall(resolved_settings),
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
