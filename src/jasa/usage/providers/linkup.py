"""Linkup usage probe sharing its key with search and fetch adapters."""

from __future__ import annotations

import httpx

from jasa.usage.base import JsonValue, request_usage_json, UsageProbe
from omnifetch.fetch.shared.config import ProviderSecrets
from omnifetch.fetch.shared.util import validate_api_key


async def fetch_linkup_usage(
    client: httpx.AsyncClient,
    secrets: ProviderSecrets,
) -> dict[str, JsonValue]:
    """Return Linkup's free raw credit-balance response."""
    api_key = validate_api_key(secrets.get("LINKUP_API_KEY"), "linkup")
    return await request_usage_json(
        client,
        secrets,
        "GET",
        "https://api.linkup.so/v1/credits/balance",
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )


LINKUP_USAGE_PROBE = UsageProbe(("LINKUP_API_KEY",), fetch_linkup_usage)
