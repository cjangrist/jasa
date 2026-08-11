"""Kagi search provider, ported from omnisearch ``providers/search/kagi``.

GETs the Kagi search endpoint with an ``Authorization: Bot <key>`` header.
Operators are re-rendered into the query EXCEPT ``file_type`` and dates, which
become the dedicated ``file_type`` and ``time_range`` query params (after, then
before, comma-joined). The response ``rank`` field is deliberately NOT mapped to
score -- array position carries the rank for downstream RRF.
"""

from __future__ import annotations

from urllib.parse import urlencode

from jasa.search.operators import (
    apply_search_operators,
    build_query_with_operators,
    parse_search_operators,
)
from jasa.search.providers.base import SearchProvider, SearchRequest
from jasa.search.ranking import SearchResult

_DEFAULT_LIMIT = 20


class KagiProvider(SearchProvider):
    """Kagi web-search adapter."""

    name = "kagi"
    secret_env = "KAGI_API_KEY"
    base_url = "https://kagi.com/api/v0"
    default_timeout_s = 20.0

    async def search(self, request: SearchRequest) -> list[SearchResult]:
        """Validate the key, split operators, GET, and map results."""
        api_key = self._validated_key()
        search_params = apply_search_operators(
            parse_search_operators(request.query)
        )
        query = build_query_with_operators(
            search_params,
            list(request.include_domains),
            list(request.exclude_domains),
            {"exclude_file_type": True, "exclude_dates": True},
        )
        params = [
            ("q", query),
            ("limit", str(request.limit or _DEFAULT_LIMIT)),
        ]
        file_type = search_params.get("file_type")
        if file_type:
            params.append(("file_type", str(file_type)))
        date_after = search_params.get("date_after")
        date_before = search_params.get("date_before")
        if date_after or date_before:
            time_range = []
            if date_after:
                time_range.append(f"after:{date_after}")
            if date_before:
                time_range.append(f"before:{date_before}")
            params.append(("time_range", ",".join(time_range)))
        url = f"{self.base_url}/search?{urlencode(params)}"
        data = await self._fetch(
            url,
            method="GET",
            headers={
                "Authorization": f"Bot {api_key}",
                "Accept": "application/json",
            },
            timeout_s=self.default_timeout_s,
        )
        items = data.get("data") if isinstance(data, dict) else None
        return [
            SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=item.get("snippet", ""),
                source_provider=self.name,
            )
            for item in (items or [])
        ]
