"""SociaVault usage probe sharing its key with the fetch adapter."""

from __future__ import annotations

import httpx

from jasa.usage.base import JsonValue, request_usage_json, UsageProbe
from omnifetch.fetch.shared.config import ProviderSecrets
from omnifetch.fetch.shared.util import validate_api_key


async def fetch_sociavault_usage(
    client: httpx.AsyncClient,
    secrets: ProviderSecrets,
) -> dict[str, JsonValue]:
    """Return SociaVault's free raw credit and subscription response."""
    api_key = validate_api_key(secrets.get("SOCIAVAULT_API_KEY"), "sociavault")
    return await request_usage_json(
        client,
        secrets,
        "GET",
        "https://api.sociavault.com/v1/credits",
        headers={
            "Accept": "application/json",
            "X-API-Key": api_key,
        },
    )


SOCIAVAULT_USAGE_PROBE = UsageProbe(
    ("SOCIAVAULT_API_KEY",), fetch_sociavault_usage
)
