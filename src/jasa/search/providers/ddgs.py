"""DuckDuckGo text search routed through the Scrapfly scrape API.

DuckDuckGo's legacy html endpoint rejects direct datacenter traffic with a
202 anomaly challenge, so the adapter sends one GET to
``https://api.scrapfly.io/scrape`` whose ``url`` parameter targets
``https://html.duckduckgo.com/html/?q=...`` and lets Scrapfly fetch the page
with browser-grade fingerprints from its shared pool. The JSON envelope's
``result.content`` HTML is parsed with the same selectors the DDGS library's
DuckDuckGo engine uses, and its redirect links are decoded back to their
target URLs.
"""

from __future__ import annotations

from collections.abc import Mapping
from urllib.parse import parse_qs, quote_plus, urlencode, urlparse

import lxml.html

from jasa.search.operators import (
    apply_search_operators,
    build_query_with_operators,
    parse_search_operators,
)
from jasa.search.providers.base import SearchProvider, SearchRequest
from jasa.search.ranking import SearchResult
from omnifetch.fetch.shared.types import ErrorType, ProviderError

_SCRAPE_PATH = "/scrape"
_DDG_HTML_URL = "https://html.duckduckgo.com/html/"
_DEFAULT_LIMIT = 20
_COUNTRY = "us"
_ITEMS_XPATH = "//div[contains(@class, 'body')]"
_TITLE_XPATH = ".//h2//text()"
_HREF_XPATH = "./a/@href"
_BODY_XPATH = "./a//text()"
_AD_HREF_PREFIX = "https://duckduckgo.com/y.js?"
_HTTP_OK = 200
_MALFORMED_MESSAGE = "Scrapfly returned a malformed scrape envelope"
_UPSTREAM_MESSAGE = "DuckDuckGo search did not return a successful page"


class DDGSProvider(SearchProvider):
    """DuckDuckGo search adapter proxied through the Scrapfly scrape API."""

    name = "ddgs"
    secret_env = "SCRAPFLY_API_KEY"
    base_url = "https://api.scrapfly.io"
    default_timeout_s = 15.0

    async def search(self, request: SearchRequest) -> list[SearchResult]:
        """Scrape one DuckDuckGo html GET through Scrapfly and map rows."""
        api_key = self._validated_key()
        query = _build_query(request)
        params = [
            ("key", api_key),
            ("url", f"{_DDG_HTML_URL}?q={quote_plus(query)}"),
            ("country", _COUNTRY),
        ]
        url = f"{self.base_url}{_SCRAPE_PATH}?{urlencode(params)}"
        envelope = await self._fetch(
            url, method="GET", timeout_s=self.default_timeout_s
        )
        content = _extract_content(envelope, self.name)
        return _map_results(content, request.limit or _DEFAULT_LIMIT, self.name)


def _build_query(request: SearchRequest) -> str:
    """Re-render all Jasa operators for DuckDuckGo's text query."""
    search_params = apply_search_operators(
        parse_search_operators(request.query)
    )
    return build_query_with_operators(
        search_params,
        list(request.include_domains),
        list(request.exclude_domains),
    )


def _extract_content(envelope: object, provider: str) -> str:
    """Return the scraped HTML, typing envelope and upstream failures."""
    result = envelope.get("result") if isinstance(envelope, Mapping) else None
    if not isinstance(result, Mapping):
        raise ProviderError(ErrorType.API_ERROR, _MALFORMED_MESSAGE, provider)
    status_code = result.get("status_code")
    if status_code != _HTTP_OK:
        raise ProviderError(
            ErrorType.PROVIDER_ERROR,
            f"{_UPSTREAM_MESSAGE} (HTTP {status_code})",
            provider,
        )
    content = result.get("content")
    return content if isinstance(content, str) else ""


def _decode_href(href: str) -> str:
    """Resolve one DuckDuckGo redirect link to its target URL."""
    candidate = f"https:{href}" if href.startswith("//") else href
    parsed = urlparse(candidate)
    host = parsed.hostname or ""
    is_duckduckgo = host == "duckduckgo.com" or host.endswith(".duckduckgo.com")
    if is_duckduckgo and parsed.path.startswith("/l/"):
        targets = parse_qs(parsed.query).get("uddg")
        if targets:
            return targets[0]
    return candidate


def _map_results(content: str, limit: int, provider: str) -> list[SearchResult]:
    """Parse the html-endpoint markup and enforce the requested result cap."""
    if not content.strip():
        return []
    document = lxml.html.fromstring(content.encode("utf-8"))
    rows = []
    for item in document.xpath(_ITEMS_XPATH):
        hrefs = item.xpath(_HREF_XPATH)
        url = _decode_href(hrefs[0]) if hrefs else ""
        if not url or url.startswith(_AD_HREF_PREFIX):
            continue
        title = "".join(item.xpath(_TITLE_XPATH)).strip()
        rows.append(
            SearchResult(
                title=title or url,
                url=url,
                snippet="".join(item.xpath(_BODY_XPATH)).strip(),
                source_provider=provider,
            )
        )
    return rows[:limit]
