"""ScrapingAnt usage probe sharing its key with the fetch adapter."""

from __future__ import annotations

import httpx

from jasa.usage.base import JsonValue, request_usage_json, UsageProbe
from omnifetch.fetch.shared.config import ProviderSecrets
from omnifetch.fetch.shared.util import validate_api_key


async def fetch_scrapingant_usage(
    client: httpx.AsyncClient,
    secrets: ProviderSecrets,
) -> dict[str, JsonValue]:
    """Return ScrapingAnt's free raw subscription and credit response."""
    api_key = validate_api_key(
        secrets.get("SCRAPINGANT_API_KEY"), "scrapingant"
    )
    return await request_usage_json(
        client,
        secrets,
        "GET",
        "https://api.scrapingant.com/v2/usage",
        params={"x-api-key": api_key},
    )


SCRAPINGANT_USAGE_PROBE = UsageProbe(
    ("SCRAPINGANT_API_KEY",), fetch_scrapingant_usage
)
