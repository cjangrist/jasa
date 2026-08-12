"""Kagi search provider using the current JSON Search API contract.

POSTs the query and inline lens to the Kagi search endpoint with Bearer auth.
Domain, file-type, and date operators become documented lens fields; remaining
operators stay in the query. Array position carries rank for downstream RRF.
"""

from __future__ import annotations

from typing import cast

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
    base_url = "https://kagi.com/api/v1"
    default_timeout_s = 20.0

    async def search(self, request: SearchRequest) -> list[SearchResult]:
        """Validate the key, build a JSON request, POST, and map results."""
        api_key = self._validated_key()
        search_params = apply_search_operators(
            parse_search_operators(request.query)
        )
        query_params = {
            name: value
            for name, value in search_params.items()
            if name not in {"include_domains", "exclude_domains"}
        }
        query = build_query_with_operators(
            query_params,
            options={"exclude_file_type": True, "exclude_dates": True},
        )
        lens: dict[str, object] = {}
        include_domains = [
            *request.include_domains,
            *cast(list[str], search_params.get("include_domains", [])),
        ]
        exclude_domains = [
            *request.exclude_domains,
            *cast(list[str], search_params.get("exclude_domains", [])),
        ]
        if include_domains:
            lens["sites_included"] = include_domains
        if exclude_domains:
            lens["sites_excluded"] = exclude_domains
        file_type = search_params.get("file_type")
        if file_type:
            lens["file_type"] = str(file_type)
        date_after = search_params.get("date_after")
        date_before = search_params.get("date_before")
        if date_after:
            lens["time_after"] = str(date_after)
        if date_before:
            lens["time_before"] = str(date_before)
        body: dict[str, object] = {
            "query": query,
            "limit": request.limit or _DEFAULT_LIMIT,
        }
        if lens:
            body["lens"] = lens
        data = await self._fetch(
            f"{self.base_url}/search",
            method="POST",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            json=body,
            timeout_s=self.default_timeout_s,
        )
        response_data = data.get("data") if isinstance(data, dict) else None
        items = (
            response_data.get("search")
            if isinstance(response_data, dict)
            else None
        )
        return [
            SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=item.get("snippet", ""),
                source_provider=self.name,
            )
            for item in (items or [])
        ]
