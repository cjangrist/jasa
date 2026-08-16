"""Usage-probe registry; each provider lands in its own pull request."""

from collections.abc import Mapping

from jasa.usage.base import UsageProbe
from jasa.usage.providers.diffbot import DIFFBOT_USAGE_PROBE
from jasa.usage.providers.firecrawl import FIRECRAWL_USAGE_PROBE
from jasa.usage.providers.github import GITHUB_USAGE_PROBE
from jasa.usage.providers.kimi import KIMI_USAGE_PROBE
from jasa.usage.providers.linkup import LINKUP_USAGE_PROBE
from jasa.usage.providers.olostep import OLOSTEP_USAGE_PROBE
from jasa.usage.providers.scrapegraphai import SCRAPEGRAPHAI_USAGE_PROBE
from jasa.usage.providers.scrapingant import SCRAPINGANT_USAGE_PROBE
from jasa.usage.providers.scrapingbee import SCRAPINGBEE_USAGE_PROBE
from jasa.usage.providers.serpapi import SERPAPI_USAGE_PROBE
from jasa.usage.providers.serper import SERPER_USAGE_PROBE
from jasa.usage.providers.tavily import TAVILY_USAGE_PROBE
from jasa.usage.providers.you import YOU_USAGE_PROBE

PROVIDER_USAGE_PROBES: Mapping[str, UsageProbe] = {
    "tavily": TAVILY_USAGE_PROBE,
    "firecrawl": FIRECRAWL_USAGE_PROBE,
    "github": GITHUB_USAGE_PROBE,
    "scrapingant": SCRAPINGANT_USAGE_PROBE,
    "scrapingbee": SCRAPINGBEE_USAGE_PROBE,
    "serpapi": SERPAPI_USAGE_PROBE,
    "serper": SERPER_USAGE_PROBE,
    "diffbot": DIFFBOT_USAGE_PROBE,
    "kimi": KIMI_USAGE_PROBE,
    "linkup": LINKUP_USAGE_PROBE,
    "you": YOU_USAGE_PROBE,
    "olostep": OLOSTEP_USAGE_PROBE,
    "scrapegraphai": SCRAPEGRAPHAI_USAGE_PROBE,
}

__all__ = ["PROVIDER_USAGE_PROBES"]
