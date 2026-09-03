"""SearXNG-compatible REST search contract and output formats."""

from __future__ import annotations

from datetime import date
from typing import Any, Literal
from urllib.parse import urlparse
from xml.etree import ElementTree

import pytest
from starlette.testclient import TestClient

from jasa.config import load_config
from jasa.search.fanout import ProviderFailure, ProviderSuccess
from jasa.search.ranking import RankedWebResult
from jasa.search.service import SearchError, SearchOptions, SearchOutcome
from jasa.searxng import (
    _jasa_query,
    _validate_parameters,
    _xml_text,
    SearxngParameters,
)
from jasa.server import build_composition


def _outcome(result_count: int = 2) -> SearchOutcome:
    results = [
        RankedWebResult(
            title=f"Title <{index}>",
            url=f"https://host{index}.example/path?x=1&y=2",
            snippets=[f"Snippet {index}", "second sentence"],
            source_providers=(
                ["tavily", "brave"] if index == 0 else ["tavily"]
            ),
            score=1 / (index + 1),
            snippet_source="aggregated",
        )
        for index in range(result_count)
    ]
    if results:
        results[-1].source_providers = []
    return SearchOutcome(
        query="provider query",
        total_duration_ms=12,
        providers_succeeded=[ProviderSuccess("tavily", 5)],
        providers_failed=[ProviderFailure("brave", "timed out", 7)],
        web_results=results,
    )


def _install_search(
    monkeypatch: pytest.MonkeyPatch,
    outcome: SearchOutcome | None = None,
    error: SearchError | None = None,
) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    async def fake_run_search(
        providers: object,
        cache: object,
        query: str,
        *,
        options: SearchOptions,
    ) -> SearchOutcome:
        captured.update(
            providers=providers,
            cache=cache,
            query=query,
            options=options,
        )
        if error is not None:
            raise error
        return outcome if outcome is not None else _outcome()

    monkeypatch.setattr("jasa.searxng.run_search", fake_run_search)
    return captured


def test_get_json_matches_current_searxng_and_openwebui_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _install_search(monkeypatch)
    composition = build_composition(load_config())
    with TestClient(composition.server.http_app()) as client:
        response = client.get(
            "/searchxng",
            params={
                "q": "python asyncio",
                "format": "json",
                "pageno": "1",
                "safesearch": "1",
                "language": "all",
                "time_range": "",
                "categories": "general",
                "theme": "simple",
                "image_proxy": "0",
            },
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    payload = response.json()
    assert set(payload) == {
        "query",
        "results",
        "answers",
        "corrections",
        "infoboxes",
        "suggestions",
        "unresponsive_engines",
    }
    assert payload["query"] == "python asyncio"
    assert payload["answers"] == []
    assert payload["unresponsive_engines"] == [["brave", "timed out"]]
    first = payload["results"][0]
    assert first["url"].startswith("https://host0.example/")
    assert first["title"] == "Title <0>"
    assert first["content"] == "Snippet 0 second sentence"
    assert first["engine"] == "tavily"
    assert first["engines"] == ["tavily", "brave"]
    assert first["positions"] == [1, 1]
    assert first["parsed_url"][1] == "host0.example"
    assert payload["results"][-1]["engine"] == "jasa"
    assert captured["query"] == "python asyncio"
    options = captured["options"]
    assert options.timeout_ms == 30000
    assert options.cache_ttl_seconds == composition.search.cache_ttl_seconds
    assert options.flights is composition.search.flights


def test_post_form_overrides_query_and_paginates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _install_search(monkeypatch, _outcome(25))
    composition = build_composition(load_config())
    with TestClient(composition.server.http_app()) as client:
        response = client.post(
            "/searchxng?q=query-value&pageno=1",
            data={
                "q": "form value",
                "format": "json",
                "pageno": "2",
                "language": "fr",
                "time_range": "day",
                "safesearch": "2",
                "categories": "news",
                "theme": "simple",
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["query"] == "form value"
    assert len(payload["results"]) == 5
    assert payload["results"][0]["title"] == "Title <20>"
    assert payload["results"][0]["positions"] == [1]
    assert captured["query"].startswith("form value lang:fr after:")


@pytest.mark.parametrize(
    ("time_range", "expected_date"),
    [
        ("day", "2026-09-02"),
        ("week", "2026-08-27"),
        ("month", "2026-08-04"),
        ("year", "2025-09-03"),
    ],
)
def test_time_ranges_translate_to_search_operators(
    time_range: str, expected_date: str
) -> None:
    parameters = SearxngParameters(
        query="query",
        output_format="json",
        language="en-US",
        page_number=1,
        time_range=time_range,
        safe_search=0,
        categories="general",
        theme="simple",
    )

    assert _jasa_query(parameters, date(2026, 9, 3)) == (
        f"query lang:en-US after:{expected_date}"
    )


@pytest.mark.parametrize("language", ["all", "auto"])
def test_language_defaults_do_not_change_query(language: str) -> None:
    parameters = SearxngParameters(
        query="query",
        output_format="json",
        language=language,
        page_number=1,
        time_range=None,
        safe_search=0,
        categories="general",
        theme="simple",
    )
    assert _jasa_query(parameters, date(2026, 9, 3)) == "query"


def test_query_language_operator_wins_over_parameter() -> None:
    parameters = SearxngParameters(
        query="query lang:de",
        output_format="json",
        language="fr",
        page_number=1,
        time_range=None,
        safe_search=0,
        categories="general",
        theme="simple",
    )
    assert _jasa_query(parameters, date(2026, 9, 3)) == "query lang:de"


def test_query_after_operator_wins_over_time_range_parameter() -> None:
    parameters = SearxngParameters(
        query="query after:2020-01-01",
        output_format="json",
        language="all",
        page_number=1,
        time_range="day",
        safe_search=0,
        categories="general",
        theme="simple",
    )
    assert _jasa_query(parameters, date(2026, 9, 3)) == (
        "query after:2020-01-01"
    )


def test_csv_output_matches_searxng_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_search(monkeypatch)
    with TestClient(
        build_composition(load_config()).server.http_app()
    ) as client:
        response = client.post(
            "/searchxng", data={"q": "csv query", "format": "csv"}
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/csv")
    assert response.headers["content-disposition"] == (
        "attachment;Filename=searx.csv"
    )
    rows = response.text.splitlines()
    assert rows[0] == "title,url,content,host,engine,score,type"
    assert rows[1].startswith(
        "Title <0>,https://host0.example/path?x=1&y=2,"
        "Snippet 0 second sentence,host0.example,tavily,1.0,result"
    )


def test_rss_output_is_parseable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_search(monkeypatch)
    with TestClient(
        build_composition(load_config()).server.http_app()
    ) as client:
        response = client.get(
            "/searchxng", params={"q": "rss query", "format": "rss"}
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/xml")
    root = ElementTree.fromstring(response.content)
    assert root.tag == "rss"
    namespace = {"opensearch": "http://a9.com/-/spec/opensearch/1.1/"}
    assert root.findtext("channel/title") == "SearXNG search: rss query"
    assert root.findtext("channel/link") == (
        "http://testserver/searchxng?q=rss+query"
    )
    assert (
        root.findtext("channel/opensearch:startIndex", namespaces=namespace)
        == "1"
    )
    query = root.find("channel/opensearch:Query", namespaces=namespace)
    assert query is not None
    assert query.attrib == {
        "role": "request",
        "searchTerms": "rss query",
        "startPage": "1",
    }
    assert root.findtext("channel/item/title") == "Title <0>"
    result_link = root.findtext("channel/item/link")
    assert result_link is not None
    parsed_result_link = urlparse(result_link)
    assert parsed_result_link.scheme == "https"
    assert parsed_result_link.hostname == "host0.example"


@pytest.mark.parametrize("output_format", ["", "unsupported"])
def test_default_and_unknown_formats_render_escaped_html(
    monkeypatch: pytest.MonkeyPatch, output_format: str
) -> None:
    _install_search(monkeypatch)
    parameters = {"q": "html query"}
    if output_format:
        parameters["format"] = output_format
    with TestClient(
        build_composition(load_config()).server.http_app()
    ) as client:
        response = client.get("/searchxng", params=parameters)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert '<form method="get" action="/searchxng">' in response.text
    assert "Title &lt;0&gt;" in response.text
    assert "x=1&amp;y=2" in response.text


def test_empty_html_search_renders_form_without_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _install_search(monkeypatch)
    with TestClient(
        build_composition(load_config()).server.http_app()
    ) as client:
        response = client.get("/searchxng")

    assert response.status_code == 200
    assert "Jasa search" in response.text
    assert captured == {}


@pytest.mark.parametrize(
    ("output_format", "content_type"),
    [
        ("json", "application/json"),
        ("csv", "application/csv"),
        ("rss", "text/xml"),
    ],
)
def test_empty_machine_search_uses_format_specific_error(
    output_format: str, content_type: str
) -> None:
    with TestClient(
        build_composition(load_config()).server.http_app()
    ) as client:
        response = client.get("/searchxng", params={"format": output_format})

    assert response.status_code == 400
    assert response.headers["content-type"].startswith(content_type)
    if output_format == "json":
        assert response.json() == {"error": "No query"}
    elif output_format == "csv":
        assert response.text == ""
    else:
        assert (
            ElementTree.fromstring(response.content).findtext(
                "channel/item/description"
            )
            == "No query"
        )


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("pageno", "zero"),
        ("pageno", "0"),
        ("safesearch", "strict"),
        ("safesearch", "3"),
        ("language", "english-US"),
        ("time_range", "decade"),
    ],
)
def test_invalid_parameters_return_json_error(name: str, value: str) -> None:
    with TestClient(
        build_composition(load_config()).server.http_app()
    ) as client:
        response = client.get(
            "/searchxng",
            params={"q": "query", "format": "json", name: value},
        )

    assert response.status_code == 400
    assert response.json()["error"].startswith("Invalid value for parameter")


def test_invalid_html_parameter_renders_error() -> None:
    with TestClient(
        build_composition(load_config()).server.http_app()
    ) as client:
        response = client.get(
            "/searchxng", params={"q": "query", "pageno": "0"}
        )
    assert response.status_code == 400
    assert "Invalid value for parameter pageno" in response.text


@pytest.mark.parametrize(
    ("parameters", "message"),
    [
        ({}, "No query"),
        ({"q": "x" * 2001}, "query exceeds 2000 characters"),
    ],
)
def test_query_validation_boundary(
    parameters: dict[str, str], message: str
) -> None:
    assert _validate_parameters(parameters, "json") == message


def test_post_can_take_query_from_url_with_empty_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _install_search(monkeypatch)
    with TestClient(
        build_composition(load_config()).server.http_app()
    ) as client:
        response = client.post("/searchxng?q=url-query&format=json")
    assert response.status_code == 200
    assert captured["query"] == "url-query"


def test_post_rejects_non_form_body() -> None:
    with TestClient(
        build_composition(load_config()).server.http_app()
    ) as client:
        response = client.post(
            "/searchxng?format=json",
            json={"q": "not accepted"},
        )
    assert response.status_code == 400
    assert response.json()["error"].startswith("POST body must use")


def test_post_content_type_error_honors_form_body_format() -> None:
    with TestClient(
        build_composition(load_config()).server.http_app()
    ) as client:
        response = client.post(
            "/searchxng",
            content="q=query&format=json",
            headers={"content-type": "text/plain"},
        )
    assert response.status_code == 400
    assert response.json()["error"].startswith("POST body must use")


def test_post_accepts_content_type_whitespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_search(monkeypatch)
    with TestClient(
        build_composition(load_config()).server.http_app()
    ) as client:
        response = client.post(
            "/searchxng",
            content="q=query&format=json",
            headers={
                "content-type": (
                    "application/x-www-form-urlencoded ; charset=utf-8"
                )
            },
        )
    assert response.status_code == 200


def test_rss_removes_xml_control_characters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outcome = _outcome(1)
    outcome.web_results[0].title = "Title\x00"
    outcome.web_results[0].snippets = ["Snippet\x01"]
    _install_search(monkeypatch, outcome)
    with TestClient(
        build_composition(load_config()).server.http_app()
    ) as client:
        response = client.get(
            "/searchxng", params={"q": "query\x02", "format": "rss"}
        )
    root = ElementTree.fromstring(response.content)
    assert root.findtext("channel/title") == "SearXNG search: query"
    assert root.findtext("channel/item/title") == "Title"
    assert root.findtext("channel/item/description") == "Snippet"


def test_xml_text_preserves_every_xml_character_range() -> None:
    value = (
        "\x00\t\n\r\x1f\x20\ud7ff\ud800\udfff"
        "\ue000\ufffd\ufffe\U00010000\U0010ffff"
    )

    assert _xml_text(value) == (
        "\t\n\r\x20\ud7ff\ue000\ufffd\U00010000\U0010ffff"
    )


def test_post_rejects_invalid_utf8_form() -> None:
    with TestClient(
        build_composition(load_config()).server.http_app()
    ) as client:
        response = client.post(
            "/searchxng?format=json",
            content=b"q=\xff",
            headers={"content-type": "application/x-www-form-urlencoded"},
        )
    assert response.status_code == 400
    assert response.json() == {"error": "invalid form encoding"}


def test_post_body_cap_returns_413() -> None:
    with TestClient(
        build_composition(load_config()).server.http_app()
    ) as client:
        response = client.post(
            "/searchxng?format=json",
            content="q=" + "x" * 70000,
            headers={"content-type": "application/x-www-form-urlencoded"},
        )
    assert response.status_code == 413
    assert response.json() == {"error": "request body too large"}


def test_auth_is_shared_with_other_rest_routes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JASA_API_KEY", "secret")
    captured = _install_search(monkeypatch)
    with TestClient(
        build_composition(load_config()).server.http_app()
    ) as client:
        response = client.get(
            "/searchxng?q=query&format=json",
            headers={"authorization": "Bearer wrong"},
        )
    assert response.status_code == 401
    assert response.json() == {"error": "unauthorized"}
    assert captured == {}


@pytest.mark.parametrize(
    ("kind", "status_code"),
    [
        ("no_providers", 503),
        ("all_failed", 502),
        ("deadline_exceeded", 504),
    ],
)
def test_search_errors_keep_jasa_status_mapping(
    monkeypatch: pytest.MonkeyPatch,
    kind: Literal["no_providers", "all_failed", "deadline_exceeded"],
    status_code: int,
) -> None:
    error = SearchError("search failed", kind=kind)
    _install_search(monkeypatch, error=error)
    with TestClient(
        build_composition(load_config()).server.http_app()
    ) as client:
        response = client.get(
            "/searchxng", params={"q": "query", "format": "json"}
        )
    assert response.status_code == status_code
    assert response.json() == {"error": "search failed"}
