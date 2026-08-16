"""Supadata usage probe sharing its key with the fetch adapter."""

from __future__ import annotations

import httpx

from jasa.usage.base import JsonValue, request_usage_json, UsageProbe
from omnifetch.fetch.shared.config import ProviderSecrets
from omnifetch.fetch.shared.util import validate_api_key


async def fetch_supadata_usage(
    client: httpx.AsyncClient,
    secrets: ProviderSecrets,
) -> dict[str, JsonValue]:
    """Return Supadata's free raw plan and credit-usage response."""
    api_key = validate_api_key(secrets.get("SUPADATA_API_KEY"), "supadata")
    return await request_usage_json(
        client,
        secrets,
        "GET",
        "https://api.supadata.ai/v1/me",
        headers={
            "Accept": "application/json",
            "x-api-key": api_key,
        },
    )


SUPADATA_USAGE_PROBE = UsageProbe(("SUPADATA_API_KEY",), fetch_supadata_usage)
