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
from jasa.search.providers.claude import ClaudeProvider
from jasa.search.providers.codex import CodexProvider
from jasa.search.providers.ddgs import DDGSProvider
from jasa.search.providers.exa import ExaProvider
from jasa.search.providers.firecrawl import FirecrawlProvider
from jasa.search.providers.kagi import KagiProvider
from jasa.search.providers.keenable import KeenableProvider
from jasa.search.providers.linkup import LinkupProvider
from jasa.search.providers.ollama import OllamaProvider
from jasa.search.providers.parallel import ParallelProvider
from jasa.search.providers.perplexity import PerplexityProvider
from jasa.search.providers.serpapi import SerpapiProvider
from jasa.search.providers.serper import SerperProvider
from jasa.search.providers.tavily import TavilyProvider
from jasa.search.providers.you import YouProvider
from jasa.search.providers.zai import ZaiProvider
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
    ClaudeProvider,
    CodexProvider,
    ZaiProvider,
    DDGSProvider,
    OllamaProvider,
    KeenableProvider,
)

CANONICAL_PROVIDER_ORDER: tuple[str, ...] = tuple(
    cls.name for cls in PROVIDER_CLASSES
)

# The canonical search-provider secret env-var names in registry order. The test
# environment-isolation purge and the registry-gating invariant both read this
# single source, so the two cannot drift.
KNOWN_SEARCH_SECRET_ENVS: tuple[str, ...] = tuple(
    provider_cls.secret_env for provider_cls in PROVIDER_CLASSES
)

# Optional provider-native deployment knobs (gateway base URLs and model ids)
# declared by the adapters themselves. They activate nothing on their own, and
# the environment-isolation fixture and the .env.example parity test read this
# same source so the three cannot drift.
KNOWN_SEARCH_SETTING_ENVS: tuple[str, ...] = tuple(
    dict.fromkeys(
        env_name
        for provider_cls in PROVIDER_CLASSES
        for env_name in provider_cls.setting_envs
    )
)


def load_search_providers(
    secrets: ProviderSecrets, client: httpx.AsyncClient
) -> dict[str, SearchProvider]:
    """Instantiate the providers whose secret is configured, in order.

    Optional settings come from the same immutable environment snapshot that
    gates the registry, so an adapter can never observe a different
    environment than the one that activated it.
    """
    active: dict[str, SearchProvider] = {}
    for provider_cls in PROVIDER_CLASSES:
        api_key = secrets.get(provider_cls.secret_env)
        if api_key:
            settings = {
                env_name: value
                for env_name in provider_cls.setting_envs
                if (value := secrets.get(env_name))
            }
            active[provider_cls.name] = provider_cls(api_key, client, settings)
    return active
