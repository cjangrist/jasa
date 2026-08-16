"""Firecrawl usage probe sharing its key with search and fetch adapters."""

from __future__ import annotations

import httpx

from jasa.usage.base import JsonValue, request_usage_json, UsageProbe
from omnifetch.fetch.shared.config import ProviderSecrets
from omnifetch.fetch.shared.util import validate_api_key


async def fetch_firecrawl_usage(
    client: httpx.AsyncClient,
    secrets: ProviderSecrets,
) -> dict[str, JsonValue]:
    """Return Firecrawl's free raw team credit-usage response."""
    api_key = validate_api_key(secrets.get("FIRECRAWL_API_KEY"), "firecrawl")
    return await request_usage_json(
        client,
        secrets,
        "GET",
        "https://api.firecrawl.dev/v2/team/credit-usage",
        headers={"Authorization": f"Bearer {api_key}"},
    )


FIRECRAWL_USAGE_PROBE = UsageProbe(
    ("FIRECRAWL_API_KEY",), fetch_firecrawl_usage
)
