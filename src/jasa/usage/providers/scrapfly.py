"""Scrapfly usage probe sharing its key with fetch adapters."""

from __future__ import annotations

import httpx

from jasa.usage.base import JsonValue, request_usage_json, UsageProbe
from omnifetch.fetch.shared.config import ProviderSecrets
from omnifetch.fetch.shared.util import validate_api_key


async def fetch_scrapfly_usage(
    client: httpx.AsyncClient,
    secrets: ProviderSecrets,
) -> dict[str, JsonValue]:
    """Return Scrapfly's free raw account, project, and usage response."""
    api_key = validate_api_key(secrets.get("SCRAPFLY_API_KEY"), "scrapfly")
    return await request_usage_json(
        client,
        secrets,
        "GET",
        "https://api.scrapfly.io/account",
        params={"key": api_key},
        headers={"Accept": "application/json"},
    )


SCRAPFLY_USAGE_PROBE = UsageProbe(("SCRAPFLY_API_KEY",), fetch_scrapfly_usage)
