"""Spider usage probe sharing its token with the fetch adapter."""

from __future__ import annotations

import httpx

from jasa.usage.base import JsonValue, request_usage_json, UsageProbe
from omnifetch.fetch.shared.config import ProviderSecrets
from omnifetch.fetch.shared.util import validate_api_key


async def fetch_spider_usage(
    client: httpx.AsyncClient,
    secrets: ProviderSecrets,
) -> dict[str, JsonValue]:
    """Return Spider's free raw credit-balance response."""
    api_token = validate_api_key(
        secrets.get("SPIDER_CLOUD_API_TOKEN"), "spider"
    )
    return await request_usage_json(
        client,
        secrets,
        "GET",
        "https://api.spider.cloud/data/credits",
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {api_token}",
        },
    )


SPIDER_USAGE_PROBE = UsageProbe(("SPIDER_CLOUD_API_TOKEN",), fetch_spider_usage)
