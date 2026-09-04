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
from datetime import date, datetime
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
_MAX_YEAR = 9999
_YEAR_PATTERN = re.compile(r"\d{4}")
_YEAR_MONTH_PATTERN = re.compile(r"(\d{4})-(\d{2})")
_RELATIVE_DATE_PATTERN = re.compile(r"\d+(?:min|h|d|mo|y)\Z")
_SITE_VALUE_PATTERN = re.compile(
    r"(?=.{1,253}\Z)"
    r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?"
    r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)*\Z"
)
_DATE_VALUE_SOURCE = (
    r"\d+(?:min|h|d|mo|y)|"
    r"\d{4}(?:-\d{2}(?:-\d{2}(?:T\d{2}:\d{2}:\d{2}"
    r"(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?)?)?)?"
)
_DATE_VALUE_PATTERN = re.compile(rf"(?:{_DATE_VALUE_SOURCE})\Z")
_QUOTED_CLAUSE_PATTERN = re.compile(
    r"(?<!\w)(?P<operator>-?(?:site|filetype|ext|intitle|inurl|inbody|"
    r'inpage|lang(?:uage)?|loc(?:ation)?|before|after)):"'
    r'(?P<operator_value>[^"]+)"(?P<operator_suffix>[\w./:+-]+)?|'
    r'(?<!\S)(?P<term>[+-])"(?P<term_value>[^"]+)"|'
    r'"(?P<exact>[^"]+)"(?P<exact_suffix>[^\s,;|()\[\]{}+]+)?'
)
_DATE_OPERATOR_PATTERN = re.compile(
    rf"(?<!\w)(?P<date_operator>before|after):"
    rf"(?P<date_value>{_DATE_VALUE_SOURCE})(?=$|[^\w./:+-])"
)
_SITE_OPERATOR_PATTERN = re.compile(
    r"(?<!\w)(?P<site_operator>-?site):"
    r'(?P<site_value>[^\s,;|()\[\]{}"]+)(?=$|[\s,;|()\[\]{}])'
)
_GENERIC_OPERATOR_PATTERN = re.compile(
    r"(?<!\w)"
    r"(?P<generic_operator>-?(?:filetype|ext|intitle|inurl|inbody|inpage|"
    r"lang(?:uage)?|loc(?:ation)?)):"
    r'(?P<generic_value>[^\s,;|()\[\]{}"]+)(?=$|[\s,;|()\[\]{}])'
)
_NEGATED_DATE_PATTERN = re.compile(
    r"(?<![^\s,;|()\[\]{}])"
    r'(?P<negated_date>-(?:before|after):[^\s,;|()\[\]{}"]+)'
)
_TOKEN_PREFIX_PATTERN = re.compile(r"[^\s,;|)\]}]*\Z")
_OPERATOR_PREFIX_WRAPPERS = frozenset(",;|()[]{}+")
_OPERATOR_PREFIX_BLOCKERS = frozenset(":/?=&")
_WHITESPACE_PATTERN = re.compile(r"\s")
_FILTER_WRAPPER_PATTERN = re.compile(r"^[\s,;|()\[\]{}+]*$")
_GENERIC_OPERATOR_TYPES = {
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
}
_PARTITIONED_OPERATOR_TYPES = frozenset(
    {
        "after",
        "before",
        "site",
        "exclude_site",
        *_GENERIC_OPERATOR_TYPES.values(),
    }
)
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
_DATE_OPERATOR_TYPES = frozenset({"after", "before"})
_SITE_OPERATOR_TYPES = frozenset({"site", "exclude_site"})

_ClausePart = str | tuple[dict[str, str] | None, str]


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
    use_structural_site = len(include_domains) == 1 and bool(
        _SITE_VALUE_PATTERN.fullmatch(include_domains[0])
    )
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
    parsed = _parse_clause_parts(parts)
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
) -> list[_ClausePart]:
    """Partition quoted, native, and protected literal clauses."""
    parts: list[_ClausePart] = []
    cursor = 0
    for match in _QUOTED_CLAUSE_PATTERN.finditer(query):
        operator = _quoted_operator(match)
        if match.group("operator") and _has_ambiguous_operator_prefix(
            query, match.start()
        ):
            literal_start = max(
                cursor, _operator_token_start(query, match.start())
            )
            parts.extend(
                _partition_unquoted_clauses(query[cursor:literal_start])
            )
            parts.append((None, query[literal_start : match.end()]))
            cursor = match.end()
            continue
        parts.extend(_partition_unquoted_clauses(query[cursor : match.start()]))
        replacement = "" if operator is not None else match.group()
        parts.append((operator, replacement))
        cursor = match.end()
    parts.extend(_partition_unquoted_clauses(query[cursor:]))
    return parts


def _partition_unquoted_clauses(
    text: str,
) -> list[_ClausePart]:
    """Partition native filters and protect ambiguous date literals."""
    matches = sorted(
        (
            *_DATE_OPERATOR_PATTERN.finditer(text),
            *_SITE_OPERATOR_PATTERN.finditer(text),
            *_GENERIC_OPERATOR_PATTERN.finditer(text),
            *_NEGATED_DATE_PATTERN.finditer(text),
        ),
        key=lambda match: match.start(),
    )
    parts: list[_ClausePart] = []
    cursor = 0
    for match in matches:
        if match.start() < cursor:
            continue
        has_ambiguous_suffix = match.lastgroup in {
            "site_value",
            "generic_value",
        } and _has_ambiguous_operator_suffix(text, match.end())
        if match.lastgroup != "negated_date" and (
            _has_ambiguous_operator_prefix(text, match.start())
            or has_ambiguous_suffix
        ):
            literal_start = max(
                cursor, _operator_token_start(text, match.start())
            )
            parts.append(text[cursor:literal_start])
            parts.append((None, text[literal_start : match.end()]))
            cursor = match.end()
            continue
        if match.lastgroup == "date_value":
            operator, replacement = _unquoted_date_operator(match)
        elif match.lastgroup == "site_value":
            operator, replacement = _unquoted_site_operator(match)
        elif match.lastgroup == "generic_value":
            operator_name = str(match.group("generic_operator"))
            if operator_name.startswith("-"):
                operator = None
                replacement = match.group()
            else:
                operator = {
                    "type": _GENERIC_OPERATOR_TYPES[operator_name],
                    "value": str(match.group("generic_value")),
                }
                replacement = ""
        else:
            operator = None
            replacement = str(match.group("negated_date"))
        parts.append(text[cursor : match.start()])
        parts.append((operator, replacement))
        cursor = match.end()
    parts.append(text[cursor:])
    return parts


def _unquoted_date_operator(
    match: re.Match[str],
) -> tuple[dict[str, str] | None, str]:
    """Promote a calendar-valid date clause or preserve it literally."""
    value = str(match.group("date_value"))
    if not _is_valid_date_bound(value):
        return None, match.group()
    return {
        "type": str(match.group("date_operator")),
        "value": value,
    }, ""


def _unquoted_site_operator(
    match: re.Match[str],
) -> tuple[dict[str, str] | None, str]:
    """Promote a clean domain clause or preserve it literally."""
    value = str(match.group("site_value"))
    if not _SITE_VALUE_PATTERN.fullmatch(value):
        return None, match.group()
    operator_name = str(match.group("site_operator"))
    return {
        "type": "exclude_site" if operator_name.startswith("-") else "site",
        "value": value,
    }, ""


def _has_ambiguous_operator_prefix(text: str, position: int) -> bool:
    """Return whether an operator-like clause is nested in another token."""
    if position:
        previous = text[position - 1]
        if not previous.isspace() and previous not in _OPERATOR_PREFIX_WRAPPERS:
            return True
    token_prefix = _TOKEN_PREFIX_PATTERN.search(text[:position])
    prefix = cast(re.Match[str], token_prefix).group()
    return any(marker in prefix for marker in _OPERATOR_PREFIX_BLOCKERS)


def _operator_token_start(text: str, position: int) -> int:
    """Return the start of the punctuation-bearing token at ``position``."""
    token_prefix = _TOKEN_PREFIX_PATTERN.search(text[:position])
    return position - len(cast(re.Match[str], token_prefix).group())


def _has_ambiguous_operator_suffix(text: str, position: int) -> bool:
    """Return whether a delimiter splits an operator-like token value."""
    if position >= len(text):
        return False
    boundary = text[position]
    if boundary in "([{":
        return True
    if boundary not in ")]}" or position + 1 >= len(text):
        return False
    following = text[position + 1]
    return not following.isspace() and following not in ",;|()[]{}+"


def _quoted_operator(match: re.Match[str]) -> dict[str, str] | None:
    """Classify one balanced quoted clause as a shared search operator."""
    if operator_name := match.group("operator"):
        if operator_name not in _QUOTED_OPERATOR_TYPES:
            return None
        operator_type = _QUOTED_OPERATOR_TYPES[operator_name]
        value = str(match.group("operator_value"))
        if match.group("operator_suffix"):
            return None
        if operator_type in _DATE_OPERATOR_TYPES and (
            not _DATE_VALUE_PATTERN.fullmatch(value)
            or not _is_valid_date_bound(value)
        ):
            return None
        if operator_type in _SITE_OPERATOR_TYPES and (
            _WHITESPACE_PATTERN.search(value)
            or not _SITE_VALUE_PATTERN.fullmatch(value)
        ):
            return None
        if operator_type not in _DATE_OPERATOR_TYPES | _SITE_OPERATOR_TYPES:
            value = f'"{value}"'
    elif sign := match.group("term"):
        operator_type = "force_include" if sign == "+" else "exclude_term"
        value = f'"{match.group("term_value")}"'
    else:
        if match.group("exact_suffix"):
            return None
        operator_type = "exact"
        value = str(match.group("exact"))
    return {"type": operator_type, "value": value}


def _parse_clause_parts(parts: list[_ClausePart]) -> dict[str, object]:
    """Parse base text and operators from the same isolated source parts."""
    base_parts: list[str] = []
    ordered: list[dict[str, str]] = []
    for part in parts:
        if isinstance(part, str):
            parsed = parse_search_operators(
                part,
                excluded_types=_PARTITIONED_OPERATOR_TYPES,
                preserve_source_order=True,
            )
            base_query = str(parsed["base_query"])
            if not base_query and part.isspace():
                base_query = " "
            elif part[:1].isspace():
                base_query = f" {base_query}"
            if base_query and part[-1:].isspace() and not part.isspace():
                base_query = f"{base_query} "
            _append_base_part(base_parts, base_query)
            ordered.extend(cast(list[dict[str, str]], parsed["operators"]))
        else:
            operator, replacement = part
            _append_base_part(base_parts, replacement)
            if operator is not None:
                ordered.append(operator)
    base_query = "".join(base_parts).strip()
    return {"base_query": base_query, "operators": ordered}


def _append_base_part(base_parts: list[str], value: str) -> None:
    """Append text while collapsing only whitespace crossing part edges."""
    if "".join(base_parts)[-1:].isspace() and value[:1].isspace():
        value = value[1:]
    base_parts.append(value)


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


def _is_valid_date_bound(value: str) -> bool:
    """Return whether a syntactically recognized bound is calendar-valid."""
    if _RELATIVE_DATE_PATTERN.fullmatch(value):
        return True
    if _YEAR_PATTERN.fullmatch(value):
        return 1 <= int(value) <= _MAX_YEAR
    if match := _YEAR_MONTH_PATTERN.fullmatch(value):
        year, month = map(int, match.groups())
        try:
            date(year, month, 1)
        except ValueError:
            return False
        return True
    try:
        if "T" in value:
            datetime.fromisoformat(value.replace("Z", "+00:00"))
        else:
            date.fromisoformat(value)
    except ValueError:
        return False
    return True


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
