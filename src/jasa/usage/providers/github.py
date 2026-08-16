"""GitHub usage probe sharing its token with the fetch adapter."""

from __future__ import annotations

import httpx

from jasa.usage.base import JsonValue, request_usage_json, UsageProbe
from omnifetch.fetch.shared.config import ProviderSecrets
from omnifetch.fetch.shared.util import validate_api_key


async def fetch_github_usage(
    client: httpx.AsyncClient,
    secrets: ProviderSecrets,
) -> dict[str, JsonValue]:
    """Return GitHub's free raw authenticated rate-limit response."""
    api_key = validate_api_key(secrets.get("GITHUB_API_KEY"), "github")
    return await request_usage_json(
        client,
        secrets,
        "GET",
        "https://api.github.com/rate_limit",
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {api_key}",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )


GITHUB_USAGE_PROBE = UsageProbe(("GITHUB_API_KEY",), fetch_github_usage)
