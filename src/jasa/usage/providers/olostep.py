"""Olostep usage probe sharing its key with the fetch adapter."""

from __future__ import annotations

import httpx

from jasa.usage.base import JsonValue, request_usage_json, UsageProbe
from omnifetch.fetch.shared.config import ProviderSecrets
from omnifetch.fetch.shared.util import validate_api_key


async def fetch_olostep_usage(
    client: httpx.AsyncClient,
    secrets: ProviderSecrets,
) -> dict[str, JsonValue]:
    """Return Olostep's free raw credit and subscription response."""
    api_key = validate_api_key(secrets.get("OLOSTEP_API_KEY"), "olostep")
    return await request_usage_json(
        client,
        secrets,
        "GET",
        "https://api.olostep.com/user/credits/info",
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )


OLOSTEP_USAGE_PROBE = UsageProbe(("OLOSTEP_API_KEY",), fetch_olostep_usage)
