r"""Search operator parsing, ported from omnisearch ``search_operators.ts``.

Converts advanced query syntax (``site:``, ``filetype:``, ...) into structured
params consumed by the Brave, Kagi, and Tavily providers.

Two patterns use the JS variable-width lookbehind ``(?<=^|\s)``; Python ``re``
requires fixed-width lookbehind, so they are rewritten as the consuming prefix
``(?:^|\s)``. The final whitespace collapse makes the two equivalent. Validated
against ``tests/fixtures/golden/operators.json``.
"""

from __future__ import annotations

import re
from typing import cast

# (type, pattern) in the exact upstream object-iteration order; the order is
# load-bearing because patterns are applied sequentially to a mutating string.
_OPERATOR_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("exclude_site", re.compile(r"-site:([^\s]+)")),
    ("site", re.compile(r"(?<!-)site:([^\s]+)")),
    ("filetype", re.compile(r"filetype:([^\s]+)")),
    ("ext", re.compile(r"ext:([^\s]+)")),
    ("intitle", re.compile(r"intitle:([^\s]+)")),
    ("inurl", re.compile(r"inurl:([^\s]+)")),
    ("inbody", re.compile(r'inbody:"?([^"\s]+)"?')),
    ("inpage", re.compile(r'inpage:"?([^"\s]+)"?')),
    ("language", re.compile(r"(?:lang|language):([^\s]+)")),
    ("location", re.compile(r"(?:loc|location):([^\s]+)")),
    ("before", re.compile(r"before:(\d{4}(?:-\d{2}(?:-\d{2})?)?)")),
    ("after", re.compile(r"after:(\d{4}(?:-\d{2}(?:-\d{2})?)?)")),
    ("exact", re.compile(r'"([^"]+)"')),
    ("force_include", re.compile(r"(?:^|\s)\+([^\s]+)")),
    ("exclude_term", re.compile(r"(?:^|\s)-([^\s:]+)")),
]

_SINGLE_VALUE_FIELDS = {
    "filetype": "file_type",
    "ext": "file_type",
    "intitle": "title_filter",
    "inurl": "url_filter",
    "inbody": "body_filter",
    "inpage": "page_filter",
    "language": "language",
    "location": "location",
    "before": "date_before",
    "after": "date_after",
}

_ARRAY_FIELDS = {
    "site": "include_domains",
    "exclude_site": "exclude_domains",
    "exact": "exact_phrases",
    "force_include": "force_include_terms",
    "exclude_term": "exclude_terms",
}

_WHITESPACE_RUN = re.compile(r"\s+")


def _strip_pattern(
    text: str,
    pattern: re.Pattern[str],
    op_type: str,
    operators: list[dict[str, str]],
) -> str:
    """Remove every match of ``pattern`` from ``text``, recording operators."""

    def replace(match: re.Match[str]) -> str:
        operators.append({"type": op_type, "value": match.group(1)})
        return ""

    return pattern.sub(replace, text)


def parse_search_operators(
    query: str, *, excluded_types: frozenset[str] = frozenset()
) -> dict[str, object]:
    """Parse ``query``, optionally leaving selected operator types intact."""
    modified = query
    operators: list[dict[str, str]] = []
    for op_type, pattern in _OPERATOR_PATTERNS:
        if op_type in excluded_types:
            continue
        modified = _strip_pattern(modified, pattern, op_type, operators)
    base_query = _WHITESPACE_RUN.sub(" ", modified).strip()
    return {"base_query": base_query, "operators": operators}


def apply_search_operators(parsed: dict[str, object]) -> dict[str, object]:
    """Fold parsed operators into a flat ``SearchParams`` mapping."""
    params: dict[str, object] = {"query": parsed["base_query"]}
    operators = cast(list[dict[str, str]], parsed["operators"])
    for operator in operators:
        op_type = operator["type"]
        value = operator["value"]
        single_field = _SINGLE_VALUE_FIELDS.get(op_type)
        if single_field is not None:
            params[single_field] = value
            continue
        array_field = _ARRAY_FIELDS.get(op_type)
        if array_field is not None:
            existing = params.get(array_field)
            if isinstance(existing, list):
                existing.append(value)
            else:
                params[array_field] = [value]
    return params


def _str_list(value: object) -> list[str]:
    """Coerce a SearchParams array field to a list of strings."""
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def build_query_with_operators(
    search_params: dict[str, object],
    additional_include_domains: list[str] | None = None,
    additional_exclude_domains: list[str] | None = None,
    options: dict[str, bool] | None = None,
) -> str:
    """Re-render a query string from structured params.

    ``exclude_file_type`` and ``exclude_dates`` (the Kagi options) drop the
    filetype and date clauses, which Kagi takes as dedicated parameters.
    """
    resolved_options = options or {}
    filters: list[str] = []

    includes = [
        *(additional_include_domains or []),
        *_str_list(search_params.get("include_domains")),
    ]
    if includes:
        filters.append(" OR ".join(f"site:{domain}" for domain in includes))

    excludes = [
        *(additional_exclude_domains or []),
        *_str_list(search_params.get("exclude_domains")),
    ]
    filters.extend(f"-site:{domain}" for domain in excludes)

    file_type = search_params.get("file_type")
    if file_type and not resolved_options.get("exclude_file_type"):
        filters.append(f"filetype:{file_type}")
    if search_params.get("title_filter"):
        filters.append(f"intitle:{search_params['title_filter']}")
    if search_params.get("url_filter"):
        filters.append(f"inurl:{search_params['url_filter']}")
    if search_params.get("body_filter"):
        filters.append(f"inbody:{search_params['body_filter']}")
    if search_params.get("page_filter"):
        filters.append(f"inpage:{search_params['page_filter']}")
    if search_params.get("language"):
        filters.append(f"lang:{search_params['language']}")
    if search_params.get("location"):
        filters.append(f"loc:{search_params['location']}")
    if search_params.get("date_before") and not resolved_options.get(
        "exclude_dates"
    ):
        filters.append(f"before:{search_params['date_before']}")
    if search_params.get("date_after") and not resolved_options.get(
        "exclude_dates"
    ):
        filters.append(f"after:{search_params['date_after']}")
    filters.extend(
        f'"{phrase}"'
        for phrase in _str_list(search_params.get("exact_phrases"))
    )
    filters.extend(
        f"+{term}"
        for term in _str_list(search_params.get("force_include_terms"))
    )
    filters.extend(
        f"-{term}" for term in _str_list(search_params.get("exclude_terms"))
    )

    query = search_params["query"]
    return f"{query} {' '.join(filters)}" if filters else str(query)
