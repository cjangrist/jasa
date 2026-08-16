"""SerpAPI usage probe sharing its key with search and fetch adapters."""

from __future__ import annotations

import httpx

from jasa.usage.base import JsonValue, request_usage_json, UsageProbe
from omnifetch.fetch.shared.config import ProviderSecrets
from omnifetch.fetch.shared.util import validate_api_key


async def fetch_serpapi_usage(
    client: httpx.AsyncClient,
    secrets: ProviderSecrets,
) -> dict[str, JsonValue]:
    """Return SerpAPI's free raw account and quota response."""
    api_key = validate_api_key(secrets.get("SERPAPI_API_KEY"), "serpapi")
    return await request_usage_json(
        client,
        secrets,
        "GET",
        "https://serpapi.com/account.json",
        params={"api_key": api_key},
    )


SERPAPI_USAGE_PROBE = UsageProbe(("SERPAPI_API_KEY",), fetch_serpapi_usage)
