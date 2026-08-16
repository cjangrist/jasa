"""Kimi usage probe sharing its key with the fetch adapter."""

from __future__ import annotations

import httpx

from jasa.usage.base import JsonValue, request_usage_json, UsageProbe
from omnifetch.fetch.shared.config import ProviderSecrets
from omnifetch.fetch.shared.util import validate_api_key


async def fetch_kimi_usage(
    client: httpx.AsyncClient,
    secrets: ProviderSecrets,
) -> dict[str, JsonValue]:
    """Return Kimi Code's free raw weekly and rolling-window usage."""
    api_key = validate_api_key(secrets.get("KIMI_API_KEY"), "kimi")
    return await request_usage_json(
        client,
        secrets,
        "GET",
        "https://api.kimi.com/coding/v1/usages",
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )


KIMI_USAGE_PROBE = UsageProbe(("KIMI_API_KEY",), fetch_kimi_usage)
