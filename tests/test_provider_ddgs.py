"""DDGS provider: Scrapfly-routed request, HTML mapping, and error behavior."""

from __future__ import annotations

import logging
from urllib.parse import quote_plus

import httpx
import pytest
import respx

from jasa.logging import configure_logging
from jasa.search.providers.base import SearchRequest
from jasa.search.providers.ddgs import DDGSProvider
from jasa.search.ranking import SearchResult
from omnifetch.fetch.shared.types import ErrorType, ProviderError

SCRAPFLY_URL = "https://api.scrapfly.io/scrape"
_DDG_HTML_URL = "https://html.duckduckgo.com/html/"
_KEY = "scp-test-key"

_RESULT_HTML = """
<html><body>
<div class="result results_links results_links_deep web-result">
  <div class="links_main links_deep result__body">
    <h2 class="result__title"><a class="result__a"
      href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fone&amp;rut=abc"
      >Example One</a></h2>
    <a class="result__snippet"
      href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fone&amp;rut=abc"
      >Snippet one.</a>
  </div>
</div>
<div class="result results_links results_links_deep web-result">
  <div class="links_main links_deep result__body">
    <h2 class="result__title"><a class="result__a"
      href="https://direct.example/page">Direct</a></h2>
    <a class="result__snippet" href="https://direct.example/page"
      >Snippet direct.</a>
  </div>
</div>
</body></html>
"""


def _ok(html_text: str, status_code: int = 200) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "result": {
                "status_code": status_code,
                "content": html_text,
                "success": status_code == 200,
            }
        },
    )


async def test_exact_outbound_request_and_redirect_decoding(
    http_client: httpx.AsyncClient,
) -> None:
    with respx.mock:
        route = respx.get(SCRAPFLY_URL).mock(return_value=_ok(_RESULT_HTML))
        results = await DDGSProvider(_KEY, http_client).search(
            SearchRequest(query="hello world", limit=8)
        )
        params = route.calls.last.request.url.params
    assert params["key"] == _KEY
    assert params["country"] == "us"
    assert params["url"] == f"{_DDG_HTML_URL}?q=hello+world"
    assert results == [
        SearchResult(
            title="Example One",
            url="https://example.com/one",
            snippet="Snippet one.",
            source_provider="ddgs",
        ),
        SearchResult(
            title="Direct",
            url="https://direct.example/page",
            snippet="Snippet direct.",
            source_provider="ddgs",
        ),
    ]


async def test_operators_and_domain_filters_are_re_rendered(
    http_client: httpx.AsyncClient,
) -> None:
    with respx.mock:
        route = respx.get(SCRAPFLY_URL).mock(return_value=_ok(""))
        await DDGSProvider(_KEY, http_client).search(
            SearchRequest(
                query="site:b.com filetype:pdf foo -site:c.com",
                include_domains=("a.com",),
                exclude_domains=("d.com",),
            )
        )
        target = route.calls.last.request.url.params["url"]
    rendered = (
        "foo site:a.com OR site:b.com -site:d.com -site:c.com filetype:pdf"
    )
    assert target == f"{_DDG_HTML_URL}?q={quote_plus(rendered)}"


async def test_ad_rows_linkless_rows_and_empty_titles_are_handled(
    http_client: httpx.AsyncClient,
) -> None:
    page = """
    <html><body>
    <div class="result__body">
      <h2>Ad</h2>
      <a href="https://duckduckgo.com/y.js?ad_domain=x.example">Ad text</a>
    </div>
    <div class="result__body">
      <h2>No link</h2>
    </div>
    <div class="result__body">
      <a href="//duckduckgo.com/l/?rut=zzz">Bare redirect snippet</a>
    </div>
    <div class="result__body">
      <h2></h2>
      <a href="https://notitle.example">No title snippet</a>
    </div>
    </body></html>
    """
    with respx.mock:
        respx.get(SCRAPFLY_URL).mock(return_value=_ok(page))
        results = await DDGSProvider(_KEY, http_client).search(
            SearchRequest(query="q")
        )
    assert results == [
        SearchResult(
            title="https://duckduckgo.com/l/?rut=zzz",
            url="https://duckduckgo.com/l/?rut=zzz",
            snippet="Bare redirect snippet",
            source_provider="ddgs",
        ),
        SearchResult(
            title="https://notitle.example",
            url="https://notitle.example",
            snippet="No title snippet",
            source_provider="ddgs",
        ),
    ]


async def test_default_limit_applies_and_caps_rows(
    http_client: httpx.AsyncClient,
) -> None:
    rows = "".join(
        f'<div class="result__body"><h2>T{index}</h2>'
        f'<a href="https://x{index}.example">S{index}</a></div>'
        for index in range(25)
    )
    with respx.mock:
        respx.get(SCRAPFLY_URL).mock(return_value=_ok(f"<html>{rows}</html>"))
        results = await DDGSProvider(_KEY, http_client).search(
            SearchRequest(query="q", limit=0)
        )
    assert len(results) == 20
    assert results[-1].url == "https://x19.example"


async def test_non_mapping_envelope_is_an_api_error(
    http_client: httpx.AsyncClient,
) -> None:
    with respx.mock:
        respx.get(SCRAPFLY_URL).mock(return_value=httpx.Response(200, json=[]))
        with pytest.raises(ProviderError) as captured:
            await DDGSProvider(_KEY, http_client).search(
                SearchRequest(query="q")
            )
    assert captured.value.error_type is ErrorType.API_ERROR
    assert (
        str(captured.value) == "Scrapfly returned a malformed scrape envelope"
    )
    assert captured.value.provider == "ddgs"


async def test_non_mapping_result_is_an_api_error(
    http_client: httpx.AsyncClient,
) -> None:
    with respx.mock:
        respx.get(SCRAPFLY_URL).mock(
            return_value=httpx.Response(200, json={"result": "nope"})
        )
        with pytest.raises(ProviderError) as captured:
            await DDGSProvider(_KEY, http_client).search(
                SearchRequest(query="q")
            )
    assert captured.value.error_type is ErrorType.API_ERROR
    assert (
        str(captured.value) == "Scrapfly returned a malformed scrape envelope"
    )


async def test_upstream_non_200_is_a_retryable_provider_error(
    http_client: httpx.AsyncClient,
) -> None:
    with respx.mock:
        respx.get(SCRAPFLY_URL).mock(
            return_value=_ok("<html>anomaly</html>", status_code=202)
        )
        with pytest.raises(ProviderError) as captured:
            await DDGSProvider(_KEY, http_client).search(
                SearchRequest(query="q")
            )
    assert captured.value.error_type is ErrorType.PROVIDER_ERROR
    assert (
        str(captured.value)
        == "DuckDuckGo search did not return a successful page (HTTP 202)"
    )


async def test_non_string_content_is_an_empty_success(
    http_client: httpx.AsyncClient,
) -> None:
    with respx.mock:
        respx.get(SCRAPFLY_URL).mock(
            return_value=httpx.Response(
                200, json={"result": {"status_code": 200, "content": None}}
            )
        )
        results = await DDGSProvider(_KEY, http_client).search(
            SearchRequest(query="q")
        )
    assert results == []


async def test_scrapfly_500_is_a_provider_error(
    http_client: httpx.AsyncClient,
) -> None:
    with respx.mock:
        respx.get(SCRAPFLY_URL).mock(
            return_value=httpx.Response(500, json={"message": "boom"})
        )
        with pytest.raises(ProviderError) as captured:
            await DDGSProvider(_KEY, http_client).search(
                SearchRequest(query="q")
            )
    assert captured.value.error_type is ErrorType.PROVIDER_ERROR
    assert str(captured.value) == "ddgs API internal error (500): boom"


async def test_scrapfly_429_is_a_rate_limit(
    http_client: httpx.AsyncClient,
) -> None:
    with respx.mock:
        respx.get(SCRAPFLY_URL).mock(
            return_value=httpx.Response(429, json={"message": "quota"})
        )
        with pytest.raises(ProviderError) as captured:
            await DDGSProvider(_KEY, http_client).search(
                SearchRequest(query="q")
            )
    assert captured.value.error_type is ErrorType.RATE_LIMIT


async def test_missing_key_raises_invalid_input(
    http_client: httpx.AsyncClient,
) -> None:
    with pytest.raises(ProviderError) as captured:
        await DDGSProvider("", http_client).search(SearchRequest(query="q"))
    assert captured.value.error_type is ErrorType.INVALID_INPUT
    assert str(captured.value) == "API key not found for ddgs"


async def test_api_key_redacted_from_error_and_logs(
    http_client: httpx.AsyncClient, caplog: pytest.LogCaptureFixture
) -> None:
    configure_logging("DEBUG")
    caplog.set_level(logging.DEBUG)
    with respx.mock:
        respx.get(SCRAPFLY_URL).mock(
            return_value=httpx.Response(500, json={"message": "x"})
        )
        with pytest.raises(ProviderError) as captured:
            await DDGSProvider(_KEY, http_client).search(
                SearchRequest(query="q")
            )
    assert _KEY not in caplog.text
    assert _KEY not in str(captured.value)


async def test_api_key_redacted_from_transport_error(
    http_client: httpx.AsyncClient,
) -> None:
    request = httpx.Request("GET", f"{SCRAPFLY_URL}?key={_KEY}")
    with respx.mock:
        respx.get(SCRAPFLY_URL).mock(
            side_effect=httpx.ConnectError(
                f"failed request {request.url}", request=request
            )
        )
        with pytest.raises(ProviderError) as captured:
            await DDGSProvider(_KEY, http_client).search(
                SearchRequest(query="q")
            )
    assert _KEY not in str(captured.value)
    assert "[REDACTED]" in str(captured.value)
