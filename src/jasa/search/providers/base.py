"""Shared base for jasa search-provider adapters.

Each adapter validates its key, parses search operators, builds a provider-
specific request, issues it through omnifetch's shared HTTP core, and maps the
response into ``SearchResult`` objects. Errors propagate as omnifetch's
``ProviderError`` so the fan-out retry classification and the single
process-wide error taxonomy stay unified (port plan §4.5, §5.1).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import cast

import httpx

from jasa.search.ranking import SearchResult
from omnifetch.fetch.shared.http import http_json
from omnifetch.fetch.shared.types import ErrorType, ProviderError
from omnifetch.fetch.shared.util import validate_api_key


@dataclass(frozen=True, slots=True)
class SearchRequest:
    """The subset of search input every provider consumes."""

    query: str
    limit: int = 20
    include_domains: tuple[str, ...] = ()
    exclude_domains: tuple[str, ...] = ()


class SearchProvider(ABC):
    """Abstract adapter; subclasses set the class attrs and implement search."""

    name: str
    secret_env: str
    base_url: str
    default_timeout_s: float

    def __init__(self, api_key: str, client: httpx.AsyncClient) -> None:
        """Store the API key and the shared HTTP client for this provider."""
        self._api_key = api_key
        self._client = client

    def _validated_key(self) -> str:
        """Return the quote-stripped key, raising INVALID_INPUT if absent."""
        return cast(str, validate_api_key(self._api_key, self.name))

    async def _fetch(self, url: str, **kwargs: object) -> object:
        """Issue a typed request via the shared HTTP core.

        ``ProviderError`` (the status/transport/size/JSON errors raised by the
        shared core) is re-raised unchanged; any other exception is wrapped as
        an ``API_ERROR`` with the upstream ``Failed to fetch search results``
        text.
        """
        try:
            return await http_json(self._client, self.name, url, **kwargs)
        except ProviderError:
            raise
        except Exception as error:
            raise ProviderError(
                ErrorType.API_ERROR,
                f"Failed to fetch search results: {error}",
                self.name,
            ) from error

    @abstractmethod
    async def search(self, request: SearchRequest) -> list[SearchResult]:
        """Run the search and return normalized results; raise on failure."""
