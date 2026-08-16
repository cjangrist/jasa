"""Diffbot usage probe sharing its token with the fetch adapter."""

from __future__ import annotations

import httpx

from jasa.usage.base import JsonValue, request_usage_json, UsageProbe
from omnifetch.fetch.shared.config import ProviderSecrets
from omnifetch.fetch.shared.util import validate_api_key


async def fetch_diffbot_usage(
    client: httpx.AsyncClient,
    secrets: ProviderSecrets,
) -> dict[str, JsonValue]:
    """Return Diffbot's free raw account plan and usage response."""
    token = validate_api_key(secrets.get("DIFFBOT_TOKEN"), "diffbot")
    return await request_usage_json(
        client,
        secrets,
        "GET",
        "https://api.diffbot.com/v4/account",
        params={"token": token},
        headers={"Accept": "application/json"},
    )


DIFFBOT_USAGE_PROBE = UsageProbe(("DIFFBOT_TOKEN",), fetch_diffbot_usage)
