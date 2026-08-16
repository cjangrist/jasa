"""Serper usage probe sharing its key with the search adapter."""

from __future__ import annotations

import httpx

from jasa.usage.base import JsonValue, request_usage_json, UsageProbe
from omnifetch.fetch.shared.config import ProviderSecrets
from omnifetch.fetch.shared.util import validate_api_key


async def fetch_serper_usage(
    client: httpx.AsyncClient,
    secrets: ProviderSecrets,
) -> dict[str, JsonValue]:
    """Return Serper's free raw account balance and rate limit response."""
    api_key = validate_api_key(secrets.get("SERPER_API_KEY"), "serper")
    return await request_usage_json(
        client,
        secrets,
        "GET",
        "https://google.serper.dev/account",
        headers={"X-API-KEY": api_key},
    )


SERPER_USAGE_PROBE = UsageProbe(("SERPER_API_KEY",), fetch_serper_usage)
