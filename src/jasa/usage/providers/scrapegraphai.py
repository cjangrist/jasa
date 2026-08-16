"""ScrapeGraphAI usage probe sharing its key with the fetch adapter."""

from __future__ import annotations

import httpx

from jasa.usage.base import JsonValue, request_usage_json, UsageProbe
from omnifetch.fetch.shared.config import ProviderSecrets
from omnifetch.fetch.shared.util import validate_api_key


async def fetch_scrapegraphai_usage(
    client: httpx.AsyncClient,
    secrets: ProviderSecrets,
) -> dict[str, JsonValue]:
    """Return ScrapeGraphAI's free raw credit and job-quota response."""
    api_key = validate_api_key(
        secrets.get("SCRAPEGRAPHAI_API_KEY"), "scrapegraphai"
    )
    return await request_usage_json(
        client,
        secrets,
        "GET",
        "https://v2-api.scrapegraphai.com/api/credits",
        headers={
            "Accept": "application/json",
            "SGAI-APIKEY": api_key,
        },
    )


SCRAPEGRAPHAI_USAGE_PROBE = UsageProbe(
    ("SCRAPEGRAPHAI_API_KEY",), fetch_scrapegraphai_usage
)
