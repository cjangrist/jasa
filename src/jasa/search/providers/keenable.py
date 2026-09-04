"""Keenable search provider using the v1 Search API.

POSTs one JSON request with ``X-API-Key`` authentication and always requests
Keenable's maximum of fifty results. One inclusive domain and date operators
use native fields; ambiguous domain policies and unsupported operators stay
rendered. A neutral wildcard keeps operator-only searches non-empty without
silently discarding structural filters.
"""

from __future__ import annotations

import calendar
import re
from typing import cast

from jasa.search.operators import (
    apply_search_operators,
    build_query_with_operators,
    parse_search_operators,
)
from jasa.search.providers.base import SearchProvider, SearchRequest
from jasa.search.ranking import SearchResult

_MAX_RESULTS = 50
_SEARCH_PATH = "/v1/search"
_MIN_MONTH = 1
_MAX_MONTH = 12
_YEAR_PATTERN = re.compile(r"\d{4}")
_YEAR_MONTH_PATTERN = re.compile(r"(\d{4})-(\d{2})")
_QUOTED_CLAUSE_PATTERN = re.compile(
    r"(?<![\w/:-])(?P<operator>-?site|filetype|ext|intitle|inurl|inbody|inpage|"
    r'lang(?:uage)?|loc(?:ation)?|before|after):"'
    r'(?P<operator_value>[^"]+)"|'
    r'(?<!\S)(?P<term>[+-])"(?P<term_value>[^"]+)"|'
    r'"(?P<exact>[^"]+)"'
)
_DATE_OPERATOR_PATTERN = re.compile(
    r"(?<![\w/:-])(before|after):"
    r"(\d+(?:min|h|d|mo|y)|"
    r"\d{4}(?:-\d{2}(?:-\d{2}(?:T\d{2}:\d{2}:\d{2}"
    r"(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?)?)?)?)(?=$|[^\w./:+-])"
)
_TOKEN_PREFIX_PATTERN = re.compile(r"[\w.:/-]*\Z")
_WHITESPACE_PATTERN = re.compile(r"\s")
_FILTER_WRAPPER_PATTERN = re.compile(r"^[\s,;|()\[\]{}]*$")
_NATIVE_OPERATOR_TYPES = frozenset({"after", "before"})
_QUOTED_OPERATOR_TYPES = {
    "-site": "exclude_site",
    "site": "site",
    "filetype": "filetype",
    "ext": "ext",
    "intitle": "intitle",
    "inurl": "inurl",
    "inbody": "inbody",
    "inpage": "inpage",
    "lang": "language",
    "language": "language",
    "loc": "location",
    "location": "location",
    "before": "before",
    "after": "after",
}
_UNQUOTED_VALUE_TYPES = {"site", "exclude_site", "before", "after"}


class KeenableProvider(SearchProvider):
    """Keenable web-search adapter."""

    name = "keenable"
    secret_env = "KEENABLE_API_KEY"
    base_url = "https://api.keenable.ai"
    default_timeout_s = 20.0

    async def search(self, request: SearchRequest) -> list[SearchResult]:
        """Validate the key, POST native filters, and map ranked hits."""
        api_key = self._validated_key()
        data = await self._fetch(
            f"{self.base_url}{_SEARCH_PATH}",
            method="POST",
            headers={
                "X-API-Key": api_key,
                "Content-Type": "application/json",
            },
            json=_build_body(request),
            timeout_s=self.default_timeout_s,
        )
        return _map_results(data, self.name)


def _build_body(request: SearchRequest) -> dict[str, object]:
    """Build native filters without allowing them to empty the query."""
    search_params = _parse_search_params(request.query)
    include_domains = _distinct_domains(
        request.include_domains, search_params, "include_domains"
    )
    exclude_domains = _distinct_domains(
        request.exclude_domains, search_params, "exclude_domains"
    )
    use_structural_site = len(include_domains) == 1
    date_after = search_params.get("date_after")
    date_before = search_params.get("date_before")
    query_params = {
        name: value
        for name, value in search_params.items()
        if name
        not in {
            "include_domains",
            "exclude_domains",
            "date_after",
            "date_before",
        }
    }
    query = build_query_with_operators(
        query_params,
        None if use_structural_site else include_domains,
        exclude_domains,
    ).strip()
    has_native_filter = use_structural_site or bool(date_after or date_before)
    if not query or (
        has_native_filter and _FILTER_WRAPPER_PATTERN.fullmatch(query)
    ):
        query = "*"
    body: dict[str, object] = {"query": query, "max_results": _MAX_RESULTS}
    if use_structural_site:
        body["site"] = include_domains[0]
    if date_after:
        body["published_after"] = str(date_after)
    if date_before:
        body["published_before"] = str(date_before)
    return body


def _distinct_domains(
    request_domains: tuple[str, ...],
    search_params: dict[str, object],
    field_name: str,
) -> list[str]:
    """Merge direct and parsed domains while preserving first-seen order."""
    parsed_domains = cast(list[str], search_params.get(field_name, []))
    return list(dict.fromkeys((*request_domains, *parsed_domains)))


def _parse_search_params(query: str) -> dict[str, object]:
    """Parse special clauses without losing their original source order."""
    parts = _partition_special_clauses(query)
    query_without_special_clauses = "".join(
        part if isinstance(part, str) else part[1] for part in parts
    )
    parsed = parse_search_operators(
        query_without_special_clauses,
        excluded_types=_NATIVE_OPERATOR_TYPES,
        preserve_source_order=True,
    )
    parsed["operators"] = _ordered_operators(parts)
    search_params = apply_search_operators(parsed)
    for operator_type in ("after", "before"):
        field_name = f"date_{operator_type}"
        value = search_params.get(field_name)
        if isinstance(value, str):
            search_params[field_name] = _normalize_date_bound(
                operator_type, value
            )
    return search_params


def _partition_special_clauses(
    query: str,
) -> list[str | tuple[dict[str, str], str]]:
    """Partition quoted and native-date clauses with removal text."""
    parts: list[str | tuple[dict[str, str], str]] = []
    cursor = 0
    for match in _QUOTED_CLAUSE_PATTERN.finditer(query):
        parts.extend(_partition_native_dates(query[cursor : match.start()]))
        parts.append((_quoted_operator(match), " "))
        cursor = match.end()
    parts.extend(_partition_native_dates(query[cursor:]))
    return parts


def _partition_native_dates(
    text: str,
) -> list[str | tuple[dict[str, str], str]]:
    """Partition eligible native dates while retaining nested URL tokens."""
    parts: list[str | tuple[dict[str, str], str]] = []
    cursor = 0
    for match in _DATE_OPERATOR_PATTERN.finditer(text):
        token_prefix = _TOKEN_PREFIX_PATTERN.search(text[: match.start()])
        if ":" in cast(re.Match[str], token_prefix).group():
            continue
        parts.append(text[cursor : match.start()])
        parts.append(({"type": match.group(1), "value": match.group(2)}, ""))
        cursor = match.end()
    parts.append(text[cursor:])
    return parts


def _quoted_operator(match: re.Match[str]) -> dict[str, str]:
    """Classify one balanced quoted clause as a shared search operator."""
    if operator_name := match.group("operator"):
        operator_type = _QUOTED_OPERATOR_TYPES[operator_name]
        value = str(match.group("operator_value"))
        if (
            operator_type not in _UNQUOTED_VALUE_TYPES
            and _WHITESPACE_PATTERN.search(value)
        ):
            value = f'"{value}"'
    elif sign := match.group("term"):
        operator_type = "force_include" if sign == "+" else "exclude_term"
        value = f'"{match.group("term_value")}"'
    else:
        operator_type = "exact"
        value = str(match.group("exact"))
    return {"type": operator_type, "value": value}


def _ordered_operators(
    parts: list[str | tuple[dict[str, str], str]],
) -> list[dict[str, str]]:
    """Return generic and special operators in their original clause order."""
    ordered: list[dict[str, str]] = []
    for part in parts:
        if isinstance(part, str):
            parsed = parse_search_operators(
                part,
                excluded_types=_NATIVE_OPERATOR_TYPES,
                preserve_source_order=True,
            )
            ordered.extend(cast(list[dict[str, str]], parsed["operators"]))
        else:
            ordered.append(part[0])
    return ordered


def _normalize_date_bound(operator_type: str, value: str) -> str:
    """Expand Jasa's partial dates to Keenable-valid inclusive bounds."""
    if _YEAR_PATTERN.fullmatch(value):
        suffix = "01-01" if operator_type == "after" else "12-31"
        return f"{value}-{suffix}"
    if match := _YEAR_MONTH_PATTERN.fullmatch(value):
        year, month = map(int, match.groups())
        if not _MIN_MONTH <= month <= _MAX_MONTH:
            return value
        day = (
            1
            if operator_type == "after"
            else calendar.monthrange(year, month)[1]
        )
        return f"{value}-{day:02d}"
    return value


def _map_results(data: object, provider_name: str) -> list[SearchResult]:
    """Map well-formed result objects and ignore unusable vendor rows."""
    results = data.get("results") if isinstance(data, dict) else None
    if not isinstance(results, list):
        return []
    return [
        SearchResult(
            title=_text(item.get("title")) or url,
            url=url,
            snippet=(
                _text(item.get("snippet")) or _text(item.get("description"))
            ),
            source_provider=provider_name,
        )
        for item in results
        if isinstance(item, dict) and (url := _text(item.get("url")))
    ]


def _text(value: object) -> str:
    """Return a string field verbatim, or empty for any other JSON type."""
    return value if isinstance(value, str) else ""
