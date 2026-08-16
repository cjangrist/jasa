"""Tavily usage probe."""

from __future__ import annotations

import httpx

from jasa.usage.base import JsonValue, request_usage_json, UsageProbe
from omnifetch.fetch.shared.config import ProviderSecrets
from omnifetch.fetch.shared.util import validate_api_key


async def fetch_tavily_usage(
    client: httpx.AsyncClient,
    secrets: ProviderSecrets,
) -> dict[str, JsonValue]:
    """Return Tavily's free raw usage response."""
    api_key = validate_api_key(secrets.get("TAVILY_API_KEY"), "tavily")
    return await request_usage_json(
        client,
        secrets,
        "GET",
        "https://api.tavily.com/usage",
        headers={"Authorization": f"Bearer {api_key}"},
    )


TAVILY_USAGE_PROBE = UsageProbe(("TAVILY_API_KEY",), fetch_tavily_usage)
