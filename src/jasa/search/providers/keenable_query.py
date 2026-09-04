"""Request-body assembly for Keenable search queries."""

from __future__ import annotations

import re
from typing import cast

from jasa.search.operators import (
    apply_search_operators,
    build_query_with_operators,
    parse_search_operators,
)
from jasa.search.providers.base import SearchRequest
from jasa.search.providers.keenable_partition import (
    ClausePart,
    partition_special_clauses,
    PARTITIONED_OPERATOR_TYPES,
)
from jasa.search.providers.keenable_validation import (
    is_clean_site_value,
    normalize_date_bound,
)

KEENABLE_MAX_RESULTS = 50
_FILTER_WRAPPER_PATTERN = re.compile(r"^[\s,;|()\[\]{}+]*$")


def build_keenable_body(request: SearchRequest) -> dict[str, object]:
    """Build native filters without allowing them to empty the query."""
    search_params = _parse_search_params(request.query)
    include_domains = _distinct_domains(
        request.include_domains, search_params, "include_domains"
    )
    exclude_domains = _distinct_domains(
        request.exclude_domains, search_params, "exclude_domains"
    )
    use_structural_site = len(include_domains) == 1 and is_clean_site_value(
        include_domains[0]
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
        options={"group_include_domains": True},
    ).strip()
    has_native_filter = use_structural_site or bool(date_after or date_before)
    if not query or (
        has_native_filter and _FILTER_WRAPPER_PATTERN.fullmatch(query)
    ):
        query = "*"
    body: dict[str, object] = {
        "query": query,
        "max_results": KEENABLE_MAX_RESULTS,
    }
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
    parsed = _parse_clause_parts(partition_special_clauses(query))
    search_params = apply_search_operators(parsed)
    for operator_type in ("after", "before"):
        field_name = f"date_{operator_type}"
        value = search_params.get(field_name)
        if isinstance(value, str):
            search_params[field_name] = normalize_date_bound(
                operator_type, value
            )
    return search_params


def _parse_clause_parts(parts: list[ClausePart]) -> dict[str, object]:
    """Parse base text and operators from the same isolated source parts."""
    base_parts: list[str] = []
    ordered: list[dict[str, str]] = []
    for part in parts:
        if isinstance(part, str):
            parsed = parse_search_operators(
                part,
                excluded_types=PARTITIONED_OPERATOR_TYPES,
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
    return {"base_query": "".join(base_parts).strip(), "operators": ordered}


def _append_base_part(base_parts: list[str], value: str) -> None:
    """Append text while collapsing only whitespace crossing part edges."""
    if not value:
        return
    if base_parts and base_parts[-1][-1:].isspace() and value[:1].isspace():
        value = value.lstrip()
    if value:
        base_parts.append(value)
