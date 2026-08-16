"""ScrapingBee usage probe sharing its key with the fetch adapter."""

from __future__ import annotations

import httpx

from jasa.usage.base import JsonValue, request_usage_json, UsageProbe
from omnifetch.fetch.shared.config import ProviderSecrets
from omnifetch.fetch.shared.util import validate_api_key


async def fetch_scrapingbee_usage(
    client: httpx.AsyncClient,
    secrets: ProviderSecrets,
) -> dict[str, JsonValue]:
    """Return ScrapingBee's free raw credit and concurrency response."""
    api_key = validate_api_key(
        secrets.get("SCRAPINGBEE_API_KEY"), "scrapingbee"
    )
    return await request_usage_json(
        client,
        secrets,
        "GET",
        "https://app.scrapingbee.com/api/v1/usage",
        headers={"Authorization": f"Bearer {api_key}"},
    )


SCRAPINGBEE_USAGE_PROBE = UsageProbe(
    ("SCRAPINGBEE_API_KEY",), fetch_scrapingbee_usage
)
