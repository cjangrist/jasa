"""DeepSeek's native, index-backed Search API (not a chat completion)."""

from __future__ import annotations

import math

from jasa.search.providers.base import SearchProvider, SearchRequest
from jasa.search.ranking import SearchResult
from omnifetch.fetch.shared.types import ErrorType, ProviderError

_MAX_QUERY_LENGTH = 128
_MAX_COUNT = 50
_MAX_SCORE = 3


class DeepseekProvider(SearchProvider):
    """Search DeepSeek's index with intact query syntax and native scores."""

    name = "deepseek"
    secret_env = "DEEPSEEK_API_KEY"
    base_url = "https://api.deepseek.com"
    default_timeout_s = 20.0

    async def search(self, request: SearchRequest) -> list[SearchResult]:
        """Validate the complete query, POST once, and map source summaries."""
        api_key = self._validated_key()
        # Preserve the caller's syntax, including OR, quotes and date precision.
        parts = [request.query]
        if request.include_domains:
            parts.append(
                " OR ".join(
                    f"site:{domain}" for domain in request.include_domains
                )
            )
        parts.extend(f"-site:{domain}" for domain in request.exclude_domains)
        query = " ".join(parts)
        if not query.strip() or len(query) > _MAX_QUERY_LENGTH:
            raise ProviderError(
                ErrorType.INVALID_INPUT,
                "DeepSeek query must contain 1 to 128 Unicode characters "
                "and cannot be all whitespace (including domain filters)",
                self.name,
            )
        data = await self._fetch(
            f"{self.base_url}/v1/tools/search",
            method="POST",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "query": query,
                "count": max(1, min(request.limit, _MAX_COUNT)),
                "need_content": False,
            },
            timeout_s=self.default_timeout_s,
        )
        rows = data.get("results") if isinstance(data, dict) else None
        if not isinstance(rows, list):
            return []
        return [
            SearchResult(
                title=_text(row.get("title")) or url,
                url=url,
                snippet=_text(row.get("summary")) or _text(row.get("snippet")),
                source_provider=self.name,
                score=_score(row.get("rerank_score")),
            )
            for row in rows
            if isinstance(row, dict) and (url := _text(row.get("url")))
        ]


def _text(value: object) -> str:
    """Keep string fields without coercing malformed upstream values."""
    return value if isinstance(value, str) else ""


def _score(value: object) -> float | None:
    """Keep only documented finite scores; booleans are not scores."""
    if (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and 0 <= value <= _MAX_SCORE
        and math.isfinite(value)
    ):
        return float(value)
    return None
