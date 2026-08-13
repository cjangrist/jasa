"""Search-provider registry in canonical fan-out order.

No singleton, no atomic-swap initialization, no idempotency gate -- those
Cloudflare-isolate-sharing patterns have no purpose in a Python process (port
plan §6). ``load_search_providers`` returns only the providers whose secret is
configured, instantiated in canonical order.
"""

from __future__ import annotations

import httpx

from jasa.search.providers.base import SearchProvider
from jasa.search.providers.brave import BraveProvider
from jasa.search.providers.exa import ExaProvider
from jasa.search.providers.firecrawl import FirecrawlProvider
from jasa.search.providers.kagi import KagiProvider
from jasa.search.providers.linkup import LinkupProvider
from jasa.search.providers.parallel import ParallelProvider
from jasa.search.providers.perplexity import PerplexityProvider
from jasa.search.providers.serpapi import SerpapiProvider
from jasa.search.providers.serper import SerperProvider
from jasa.search.providers.tavily import TavilyProvider
from jasa.search.providers.you import YouProvider
from omnifetch.fetch.shared.config import ProviderSecrets

# Canonical fan-out order (omnisearch providers/unified/web_search.ts:24-35).
# Adapters are appended here as each lands, in this exact order.
PROVIDER_CLASSES: tuple[type[SearchProvider], ...] = (
    TavilyProvider,
    BraveProvider,
    KagiProvider,
    ExaProvider,
    FirecrawlProvider,
    PerplexityProvider,
    SerpapiProvider,
    LinkupProvider,
    YouProvider,
    ParallelProvider,
    SerperProvider,
)

CANONICAL_PROVIDER_ORDER: tuple[str, ...] = tuple(
    cls.name for cls in PROVIDER_CLASSES
)

# The canonical search-provider secret env-var names in registry order. The test
# environment-isolation purge and the registry-gating invariant both read this
# single source, so the two cannot drift.
KNOWN_SEARCH_SECRET_ENVS: tuple[str, ...] = (
    "TAVILY_API_KEY",
    "BRAVE_API_KEY",
    "KAGI_API_KEY",
    "EXA_API_KEY",
    "FIRECRAWL_API_KEY",
    "PERPLEXITY_API_KEY",
    "SERPAPI_API_KEY",
    "LINKUP_API_KEY",
    "YOU_API_KEY",
    "PARALLEL_API_KEY",
    "SERPER_API_KEY",
)


def load_search_providers(
    secrets: ProviderSecrets, client: httpx.AsyncClient
) -> dict[str, SearchProvider]:
    """Instantiate the providers whose secret is configured, in order."""
    active: dict[str, SearchProvider] = {}
    for provider_cls in PROVIDER_CLASSES:
        api_key = secrets.get(provider_cls.secret_env)
        if api_key:
            active[provider_cls.name] = provider_cls(api_key, client)
    return active
