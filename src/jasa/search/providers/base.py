"""Shared base for jasa search-provider adapters.

Each adapter validates its key, parses search operators, builds a provider-
specific request, issues it through omnifetch's shared HTTP core, and maps the
response into ``SearchResult`` objects. Errors propagate as omnifetch's
``ProviderError`` so the fan-out retry classification and the single
process-wide error taxonomy stay unified (port plan §4.5, §5.1).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import cast

import httpx

from jasa.search.ranking import SearchResult
from omnifetch.fetch.shared.http import http_json
from omnifetch.fetch.shared.types import ErrorType, ProviderError
from omnifetch.fetch.shared.util import validate_api_key


@dataclass(frozen=True, slots=True)
class SearchRequest:
    """The subset of search input every provider consumes.

    ``limit`` matches the fan-out's per-provider limit. The fan-out always
    passes it explicitly, so the default only governs a direct adapter caller;
    keeping the two equal stops such a caller from being silently capped
    below what the registry asks for in production. ``reference_datetime`` is
    the one wall-clock snapshot shared by cache eligibility and every provider
    attempt in a search.
    """

    query: str
    limit: int = 30
    include_domains: tuple[str, ...] = ()
    exclude_domains: tuple[str, ...] = ()
    reference_datetime: datetime | None = None


class SearchProvider(ABC):
    """Abstract adapter; subclasses set the class attrs and implement search.

    ``setting_envs`` names optional provider-native deployment knobs (a gateway
    base URL, a model id) that the registry resolves from the same environment
    snapshot it gates the adapter on. They are configuration, never
    credentials, so they are neither required nor redacted.

    An LLM-mediated adapter names its model setting in ``model_env`` and its
    shipped default in ``default_model``. ``model_id()`` resolves the pair, so
    the request body and the search-cache identity read one value and a model
    changed through the environment can never replay results cached for the
    previous one.
    """

    name: str
    secret_env: str
    base_url: str
    default_timeout_s: float
    setting_envs: tuple[str, ...] = ()
    model_env: str | None = None
    default_model: str | None = None

    def __init__(
        self,
        api_key: str,
        client: httpx.AsyncClient,
        settings: Mapping[str, str] | None = None,
    ) -> None:
        """Store the key, the shared HTTP client, and resolved settings."""
        self._api_key = api_key
        self._client = client
        self._settings: Mapping[str, str] = dict(settings or {})

    def _setting(self, env_name: str, default: str) -> str:
        """Return the configured value for one setting, else ``default``."""
        return self._settings.get(env_name) or default

    def model_id(self) -> str | None:
        """Return the effective model id, or None for a non-LLM adapter."""
        if self.model_env is None or self.default_model is None:
            return None
        return self._setting(self.model_env, self.default_model)

    def cache_semantics(self) -> str | None:
        """Return the configuration that shapes this adapter's results.

        The search service keys cached outcomes on it, so changing the value
        through the environment never replays results produced under the old
        one. The default is the effective model; an adapter with another
        result-shaping setting appends it. It must not raise: a malformed
        setting fails that adapter's request, not the whole search's key.
        """
        return self.model_id()

    def _validated_key(self) -> str:
        """Return the quote-stripped key, raising INVALID_INPUT if absent."""
        normalized = cast(str, validate_api_key(self._api_key, self.name))
        if not normalized:
            raise ProviderError(
                ErrorType.INVALID_INPUT,
                f"API key not found for {self.name}",
                self.name,
            )
        return normalized

    def _redact_secret(self, message: str) -> str:
        """Remove raw and quote-stripped provider credentials from text."""
        candidates = {
            self._api_key,
            self._api_key.strip().strip('"').strip("'"),
        }
        redacted = message
        for candidate in candidates:
            if candidate:
                redacted = redacted.replace(candidate, "[REDACTED]")
        return redacted

    def allows_cache(
        self,
        query: str,
        *,
        reference_datetime: datetime | None = None,
    ) -> bool:
        """Return whether an aggregate result for ``query`` may be cached."""
        return True

    async def _fetch(self, url: str, **kwargs: object) -> object:
        """Issue a typed request via the shared HTTP core.

        ``ProviderError`` (the status/transport/size/JSON errors raised by the
        shared core) is re-raised unchanged; any other exception is wrapped as
        an ``API_ERROR`` with the upstream ``Failed to fetch search results``
        text.
        """
        try:
            return await http_json(self._client, self.name, url, **kwargs)
        except ProviderError as error:
            message = self._redact_secret(str(error))
            if message == str(error):
                raise
            raise ProviderError(
                error.error_type, message, error.provider, error.details
            ) from None
        except Exception as error:
            raise ProviderError(
                ErrorType.API_ERROR,
                self._redact_secret(f"Failed to fetch search results: {error}"),
                self.name,
            ) from None

    @abstractmethod
    async def search(self, request: SearchRequest) -> list[SearchResult]:
        """Run the search and return normalized results; raise on failure."""
