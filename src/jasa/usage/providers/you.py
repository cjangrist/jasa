"""You.com usage probe sharing its key with search and fetch adapters."""

from __future__ import annotations

import httpx

from jasa.usage.base import JsonValue, request_usage_json, UsageProbe
from omnifetch.fetch.shared.config import ProviderSecrets
from omnifetch.fetch.shared.util import validate_api_key


async def fetch_you_usage(
    client: httpx.AsyncClient,
    secrets: ProviderSecrets,
) -> dict[str, JsonValue]:
    """Return You.com's free raw account-balance response."""
    api_key = validate_api_key(secrets.get("YOU_API_KEY"), "you")
    return await request_usage_json(
        client,
        secrets,
        "GET",
        "https://api.you.com/v1/billing/account_balance",
        headers={
            "Accept": "application/json",
            "X-API-Key": api_key,
        },
    )


YOU_USAGE_PROBE = UsageProbe(("YOU_API_KEY",), fetch_you_usage)
