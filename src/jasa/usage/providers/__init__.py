"""Usage-probe registry; each provider lands in its own pull request."""

from collections.abc import Mapping

from jasa.usage.base import UsageProbe
from jasa.usage.providers.tavily import TAVILY_USAGE_PROBE

PROVIDER_USAGE_PROBES: Mapping[str, UsageProbe] = {
    "tavily": TAVILY_USAGE_PROBE,
}

__all__ = ["PROVIDER_USAGE_PROBES"]
