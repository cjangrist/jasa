"""SearXNG-compatible search API over Jasa's shared search runtime.

The route accepts SearXNG GET query parameters and form-encoded POST bodies,
then renders HTML, JSON, CSV, or RSS without creating another search client or
provider path. Jasa is a general-web instance, so category and presentation
preferences are accepted while language, page, and time-range semantics are
applied to the underlying search.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta, UTC
from html import escape
from io import StringIO
from typing import Any
from urllib.parse import parse_qsl, ParseResult, urlencode, urlparse
from xml.etree import ElementTree

from fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from jasa.auth import is_authorized
from jasa.search.ranking import RankedWebResult
from jasa.search.service import (
    run_search,
    SearchError,
    SearchOptions,
    SearchOutcome,
    SearchRuntime,
)
from jasa.usage import UsageRuntime

_BODY_LIMIT_BYTES = 65536
_QUERY_LIMIT_CHARS = 2000
_PAGE_SIZE = 20
_SEARCH_TIMEOUT_MS = 30000
_OUTPUT_FORMATS = frozenset({"html", "json", "csv", "rss"})
_TIME_RANGES = frozenset({"day", "week", "month", "year"})
_LANGUAGE_CODE = re.compile(r"^[a-z]{2,3}(?:-[a-zA-Z]{2})?$")
_LANGUAGE_OPERATOR = re.compile(r"(?:^|\s)(?:lang|language):[^\s]+")
_AFTER_OPERATOR = re.compile(r"after:(\d{4}(?:-\d{2}(?:-\d{2})?)?)")
_TIME_RANGE_DAYS = {"day": 1, "week": 7, "month": 30, "year": 365}
_OPENSEARCH_NAMESPACE = "http://a9.com/-/spec/opensearch/1.1/"


@dataclass(frozen=True, slots=True)
class SearxngParameters:
    """Validated search controls accepted by a SearXNG endpoint."""

    query: str
    output_format: str
    language: str
    page_number: int
    time_range: str | None
    safe_search: int
    categories: str
    theme: str


def _output_format(parameters: dict[str, str]) -> str:
    requested = parameters.get("format", "html")
    return requested if requested in _OUTPUT_FORMATS else "html"


async def _read_form_body(request: Request) -> bytes | Response:
    chunks: list[bytes] = []
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > _BODY_LIMIT_BYTES:
            return _error_response(
                _output_format(dict(request.query_params)),
                "request body too large",
                413,
                request,
            )
        chunks.append(chunk)
    return b"".join(chunks)


async def _request_parameters(
    request: Request,
) -> dict[str, str] | Response:
    parameters = dict(request.query_params)
    if request.method != "POST":
        return parameters
    body = await _read_form_body(request)
    if isinstance(body, Response):
        return body
    try:
        form_parameters = dict(
            parse_qsl(body.decode("utf-8"), keep_blank_values=True)
        )
    except UnicodeDecodeError:
        return _error_response(
            _output_format(parameters), "invalid form encoding", 400, request
        )
    content_type = (
        request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    )
    if body and content_type != "application/x-www-form-urlencoded":
        return _error_response(
            _output_format({**parameters, **form_parameters}),
            "POST body must use application/x-www-form-urlencoded",
            400,
            request,
        )
    parameters.update(form_parameters)
    return parameters


def _parse_integer(
    parameters: dict[str, str], name: str, default: int
) -> int | str:
    raw_value = parameters.get(name, str(default))
    if not raw_value.isascii() or not raw_value.isdecimal():
        return f"Invalid value for parameter {name}: {raw_value}"
    try:
        return int(raw_value)
    except ValueError:
        return f"Invalid value for parameter {name}: {raw_value}"


def _validate_parameters(
    parameters: dict[str, str], output_format: str
) -> SearxngParameters | str:
    query = parameters.get("q", "").strip()
    if not query or len(query) > _QUERY_LIMIT_CHARS:
        return "No query" if not query else "query exceeds 2000 characters"
    page_number = _parse_integer(parameters, "pageno", 1)
    if isinstance(page_number, str) or page_number < 1:
        return (
            page_number
            if isinstance(page_number, str)
            else "Invalid value for parameter pageno: 0"
        )
    safe_search = _parse_integer(parameters, "safesearch", 0)
    if isinstance(safe_search, str) or safe_search not in (0, 1, 2):
        return (
            safe_search
            if isinstance(safe_search, str)
            else f"Invalid value for parameter safesearch: {safe_search}"
        )
    language = parameters.get("language", "all")
    if language != "auto" and not _LANGUAGE_CODE.fullmatch(language):
        return f"Invalid value for parameter language: {language}"
    time_range = parameters.get("time_range")
    time_range = None if time_range in (None, "", "None") else time_range
    if time_range is not None and time_range not in _TIME_RANGES:
        return f"Invalid value for parameter time_range: {time_range}"
    return SearxngParameters(
        query=query,
        output_format=output_format,
        language=language,
        page_number=page_number,
        time_range=time_range,
        safe_search=safe_search,
        categories=parameters.get("categories", "general"),
        theme=parameters.get("theme", "simple"),
    )


def _jasa_query(
    parameters: SearxngParameters, today: date | None = None
) -> str:
    filters: list[str] = []
    if parameters.language not in ("all", "auto") and not (
        _LANGUAGE_OPERATOR.search(parameters.query)
    ):
        filters.append(f"lang:{parameters.language}")
    if parameters.time_range is not None and not _AFTER_OPERATOR.search(
        parameters.query
    ):
        current_date = today or datetime.now(UTC).date()
        days = _TIME_RANGE_DAYS[parameters.time_range]
        filters.append(f"after:{current_date - timedelta(days=days):%Y-%m-%d}")
    return " ".join((parameters.query, *filters))


def _parse_result_url(value: str) -> ParseResult | None:
    """Parse provider URL data without failing on malformed input."""
    try:
        return urlparse(value)
    except ValueError:
        return None


def _result_payload(result: RankedWebResult, position: int) -> dict[str, Any]:
    providers = result.source_providers or ["jasa"]
    parsed_url = _parse_result_url(result.url)
    return {
        "url": result.url,
        "engine": providers[0],
        "parsed_url": (
            list(parsed_url)
            if parsed_url is not None
            else ["", "", "", "", "", ""]
        ),
        "template": "default.html",
        "title": result.title,
        "content": " ".join(result.snippets),
        "img_src": "",
        "iframe_src": "",
        "audio_src": "",
        "thumbnail": "",
        "publishedDate": None,
        "pubdate": "",
        "length": None,
        "views": "",
        "author": "",
        "metadata": "",
        "priority": "",
        "engines": providers,
        "open_group": False,
        "close_group": False,
        "positions": [position for _provider in providers],
        "score": result.score,
        "category": "general",
    }


def _page_results(
    outcome: SearchOutcome, page_number: int
) -> list[dict[str, Any]]:
    start = (page_number - 1) * _PAGE_SIZE
    page = outcome.web_results[start : start + _PAGE_SIZE]
    return [
        _result_payload(result, index)
        for index, result in enumerate(page, start=1)
    ]


def _json_response(
    query: str,
    results: list[dict[str, Any]],
    outcome: SearchOutcome,
) -> JSONResponse:
    return JSONResponse(
        {
            "query": query,
            "results": results,
            "answers": [],
            "corrections": [],
            "infoboxes": [],
            "suggestions": [],
            "unresponsive_engines": [
                [failure.provider, failure.error]
                for failure in outcome.providers_failed
            ],
        }
    )


def _csv_cell(value: object) -> object:
    """Prefix formula-leading strings so spreadsheets treat them as text."""
    if isinstance(value, str) and value.startswith(("=", "+", "-", "@")):
        return f"'{value}"
    return value


def _csv_result_row(result: dict[str, Any]) -> dict[str, object]:
    """Select and sanitize the SearXNG CSV fields for one result."""
    row: dict[str, object] = {
        "title": result["title"],
        "url": result["url"],
        "content": result["content"],
        "host": result["parsed_url"][1],
        "engine": result["engine"],
        "score": result["score"],
        "type": "result",
    }
    return {name: _csv_cell(value) for name, value in row.items()}


def _csv_response(
    results: list[dict[str, Any]],
    status_code: int = 200,
    include_header: bool = True,
) -> Response:
    stream = StringIO(newline="")
    fieldnames = ("title", "url", "content", "host", "engine", "score", "type")
    writer = csv.DictWriter(
        stream, fieldnames=fieldnames, extrasaction="ignore"
    )
    if include_header:
        writer.writeheader()
    writer.writerows(map(_csv_result_row, results))
    return Response(
        stream.getvalue(),
        status_code=status_code,
        media_type="application/csv",
        headers={"Content-Disposition": "attachment;Filename=searx.csv"},
    )


def _rss_response(
    request: Request,
    query: str,
    results: list[dict[str, Any]],
    page_number: int = 1,
    error: tuple[int, str] | None = None,
) -> Response:
    status_code, error_message = error or (200, None)
    ElementTree.register_namespace("opensearch", _OPENSEARCH_NAMESPACE)
    rss = ElementTree.Element("rss", {"version": "2.0"})
    channel = ElementTree.SubElement(rss, "channel")
    safe_query = _xml_text(query)
    search_parameters = {"q": query}
    query_key = request.query_params.get("key")
    if query_key is not None:
        search_parameters["key"] = query_key
    search_url = str(request.url.replace(query=urlencode(search_parameters)))
    ElementTree.SubElement(
        channel, "title"
    ).text = f"SearXNG search: {safe_query}"
    ElementTree.SubElement(channel, "link").text = search_url
    ElementTree.SubElement(
        channel, "description"
    ).text = f'Search results for "{safe_query}" - Jasa'
    ElementTree.SubElement(
        channel, f"{{{_OPENSEARCH_NAMESPACE}}}startIndex"
    ).text = str((page_number - 1) * _PAGE_SIZE + 1)
    ElementTree.SubElement(
        channel,
        f"{{{_OPENSEARCH_NAMESPACE}}}Query",
        {
            "role": "request",
            "searchTerms": safe_query,
            "startPage": str(page_number),
        },
    )
    if error_message is not None:
        item = ElementTree.SubElement(channel, "item")
        ElementTree.SubElement(item, "title").text = "Error"
        ElementTree.SubElement(item, "description").text = _xml_text(
            error_message
        )
    for result in results:
        item = ElementTree.SubElement(channel, "item")
        ElementTree.SubElement(item, "title").text = _xml_text(
            str(result["title"])
        )
        ElementTree.SubElement(item, "type").text = "result"
        ElementTree.SubElement(item, "link").text = _xml_text(
            str(result["url"])
        )
        ElementTree.SubElement(item, "description").text = _xml_text(
            str(result["content"])
        )
    content = ElementTree.tostring(rss, encoding="utf-8", xml_declaration=True)
    return Response(content, status_code=status_code, media_type="text/xml")


def _xml_text(value: str) -> str:
    return "".join(
        character
        for character in value
        if character in "\t\n\r"
        or "\u0020" <= character <= "\ud7ff"
        or "\ue000" <= character <= "\ufffd"
        or "\U00010000" <= character <= "\U0010ffff"
    )


def _html_result_item(result: dict[str, Any]) -> str:
    """Render a result without linking provider-supplied unsafe URI schemes."""
    url = str(result["url"])
    parsed_url = _parse_result_url(url)
    safe_url = (
        url
        if parsed_url is not None
        and parsed_url.scheme.lower() in {"http", "https"}
        and parsed_url.hostname is not None
        else None
    )
    title = escape(str(result["title"]))
    heading = (
        f"<h2>{title}</h2>"
        if safe_url is None
        else f'<h2><a href="{escape(safe_url, quote=True)}">{title}</a></h2>'
    )
    return f"<li>{heading}<p>{escape(str(result['content']))}</p></li>"


def _html_response(
    request: Request,
    query: str,
    results: list[dict[str, Any]],
    status_code: int = 200,
    error_message: str | None = None,
) -> Response:
    error = "" if error_message is None else f"<p>{escape(error_message)}</p>"
    items = "".join(map(_html_result_item, results))
    query_key = request.query_params.get("key")
    hidden_key = (
        ""
        if query_key is None
        else (
            '<input type="hidden" name="key" '
            f'value="{escape(query_key, quote=True)}">'
        )
    )
    document = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        "<title>Jasa search</title></head><body><main><h1>Jasa search</h1>"
        f'<form method="get" action="/searchxng"><input name="q" '
        f'value="{escape(query, quote=True)}">{hidden_key}'
        "<button>Search</button></form>"
        f"{error}<ol>{items}</ol></main></body></html>"
    )
    return Response(document, status_code=status_code, media_type="text/html")


def _error_response(
    output_format: str,
    message: str,
    status_code: int,
    request: Request,
) -> Response:
    if output_format == "json":
        return JSONResponse({"error": message}, status_code=status_code)
    if output_format == "csv":
        return _csv_response([], status_code, include_header=False)
    if output_format == "rss":
        return _rss_response(request, "", [], error=(status_code, message))
    return _html_response(request, "", [], status_code, error_message=message)


def _search_error_status(error: SearchError) -> int:
    if error.kind == "no_providers":
        return 503
    if error.kind == "deadline_exceeded":
        return 504
    return 502


def _success_response(
    request: Request,
    parameters: SearxngParameters,
    outcome: SearchOutcome,
) -> Response:
    results = _page_results(outcome, parameters.page_number)
    if parameters.output_format == "json":
        return _json_response(parameters.query, results, outcome)
    if parameters.output_format == "csv":
        return _csv_response(results)
    if parameters.output_format == "rss":
        return _rss_response(
            request,
            parameters.query,
            results,
            page_number=parameters.page_number,
        )
    return _html_response(request, parameters.query, results)


async def _execute_searchxng(
    request: Request, search: SearchRuntime, usage: UsageRuntime
) -> Response:
    request_parameters = await _request_parameters(request)
    if isinstance(request_parameters, Response):
        return request_parameters
    output_format = _output_format(request_parameters)
    if not request_parameters.get("q", "").strip():
        if output_format == "html":
            return _html_response(request, "", [])
        return _error_response(output_format, "No query", 400, request)
    parameters = _validate_parameters(request_parameters, output_format)
    if isinstance(parameters, str):
        return _error_response(output_format, parameters, 400, request)
    usage.trigger_refresh()
    options = SearchOptions(
        timeout_ms=_SEARCH_TIMEOUT_MS,
        cache_ttl_seconds=search.cache_ttl_seconds,
        flights=search.flights,
    )
    try:
        outcome = await run_search(
            search.providers,
            search.cache,
            _jasa_query(parameters),
            options=options,
        )
    except SearchError as error:
        return _error_response(
            parameters.output_format,
            str(error),
            _search_error_status(error),
            request,
        )
    return _success_response(request, parameters, outcome)


def register_searxng_route(
    server: FastMCP, search: SearchRuntime, usage: UsageRuntime
) -> None:
    """Register the SearXNG-compatible GET/form-POST search endpoint."""

    @server.custom_route(
        "/searchxng", methods=["GET", "POST"], include_in_schema=False
    )
    async def rest_searchxng(request: Request) -> Response:
        if not is_authorized(request):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        return await _execute_searchxng(request, search, usage)
