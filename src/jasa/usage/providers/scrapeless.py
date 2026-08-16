"""Scrapeless usage probe sharing its key with the fetch adapter."""

from __future__ import annotations

import httpx

from jasa.usage.base import JsonValue, request_usage_json, UsageProbe
from omnifetch.fetch.shared.config import ProviderSecrets
from omnifetch.fetch.shared.util import validate_api_key


async def fetch_scrapeless_usage(
    client: httpx.AsyncClient,
    secrets: ProviderSecrets,
) -> dict[str, JsonValue]:
    """Return Scrapeless's free raw credit and subscription response."""
    api_key = validate_api_key(secrets.get("SCRAPELESS_API_KEY"), "scrapeless")
    return await request_usage_json(
        client,
        secrets,
        "GET",
        "https://api.scrapeless.com/api/v1/me",
        headers={
            "Accept": "application/json",
            "x-api-token": api_key,
        },
    )


SCRAPELESS_USAGE_PROBE = UsageProbe(
    ("SCRAPELESS_API_KEY",), fetch_scrapeless_usage
)
