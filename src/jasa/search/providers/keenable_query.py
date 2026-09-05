"""Request-body assembly for Keenable search queries."""

from __future__ import annotations

import re
from datetime import datetime, UTC
from typing import cast

from jasa.search.operators import (
    apply_search_operators,
    build_query_with_operators,
    parse_search_operators,
)
from jasa.search.providers.base import SearchRequest
from jasa.search.providers.keenable_partition import (
    ClausePart,
    LITERAL_INCLUDE_SITE_TYPE,
    partition_special_clauses,
    PARTITIONED_OPERATOR_TYPES,
)
from jasa.search.providers.keenable_validation import (
    is_clean_site_value,
    is_relative_date_bound,
    normalize_date_bound,
    resolve_date_bound,
)

KEENABLE_MAX_RESULTS = 50
_FILTER_WRAPPER_PATTERN = re.compile(r"^[\s,;|()\[\]{}+]*$")


def has_promoted_relative_date(
    query: str, *, reference_datetime: datetime | None = None
) -> bool:
    """Return whether any validated native date clause moves over time."""
    reference = reference_datetime or datetime.now(UTC)
    parsed = _parse_clause_parts(
        partition_special_clauses(query, reference_datetime=reference)
    )
    operators = cast(list[dict[str, str]], parsed["operators"])
    return any(
        operator["type"] in {"after", "before"}
        and is_relative_date_bound(operator["value"])
        for operator in operators
    )


def build_keenable_body(
    request: SearchRequest, *, reference_datetime: datetime | None = None
) -> dict[str, object]:
    """Build native filters without allowing them to empty the query."""
    reference = reference_datetime or datetime.now(UTC)
    search_params = _parse_search_params(request.query, reference)
    include_domains = _distinct_domains(
        request.include_domains, search_params, "include_domains"
    )
    exclude_domains = _distinct_domains(
        request.exclude_domains, search_params, "exclude_domains"
    )
    has_literal_include_site = bool(
        search_params.get("has_literal_include_site")
    )
    use_structural_site = (
        not has_literal_include_site
        and len(include_domains) == 1
        and is_clean_site_value(include_domains[0])
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
            "has_literal_include_site",
        }
    }
    has_site_query_filter = bool(exclude_domains) or (
        bool(include_domains) and not use_structural_site
    )
    if has_site_query_filter and _FILTER_WRAPPER_PATTERN.fullmatch(
        str(query_params.get("query", ""))
    ):
        query_params["query"] = ""
    query = build_query_with_operators(
        query_params,
        None if use_structural_site else include_domains,
        exclude_domains,
        options={"group_include_domains": True},
    ).strip()
    has_native_filter = use_structural_site or bool(date_after or date_before)
    if has_native_filter and (
        not query or _FILTER_WRAPPER_PATTERN.fullmatch(query)
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
    if any(
        isinstance(bound, str) and is_relative_date_bound(bound)
        for bound in (date_after, date_before)
    ):
        body["query_time"] = _format_query_time(reference)
    return body


def _format_query_time(reference_datetime: datetime) -> str:
    """Format the shared reference at Keenable's relative-date precision."""
    return (
        reference_datetime.replace(second=0, microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _distinct_domains(
    request_domains: tuple[str, ...],
    search_params: dict[str, object],
    field_name: str,
) -> list[str]:
    """Merge direct and parsed domains while preserving first-seen order."""
    parsed_domains = cast(list[str], search_params.get(field_name, []))
    distinct_domains: list[str] = []
    seen_identities: set[str] = set()
    for domain in (*request_domains, *parsed_domains):
        identity = domain.casefold() if is_clean_site_value(domain) else domain
        if identity in seen_identities:
            continue
        seen_identities.add(identity)
        distinct_domains.append(domain)
    return distinct_domains


def _parse_search_params(
    query: str, reference_datetime: datetime
) -> dict[str, object]:
    """Parse special clauses without losing their original source order."""
    parsed = _parse_clause_parts(
        partition_special_clauses(query, reference_datetime=reference_datetime)
    )
    operators = cast(list[dict[str, str]], parsed["operators"])
    parsed["operators"] = _strictest_date_operators(
        operators, reference_datetime
    )
    search_params = apply_search_operators(parsed)
    search_params["has_literal_include_site"] = any(
        operator["type"] == LITERAL_INCLUDE_SITE_TYPE for operator in operators
    )
    for operator_type in ("after", "before"):
        field_name = f"date_{operator_type}"
        value = search_params.get(field_name)
        if isinstance(value, str):
            search_params[field_name] = normalize_date_bound(
                operator_type, value
            )
    return search_params


def _strictest_date_operators(
    operators: list[dict[str, str]],
    reference_datetime: datetime,
) -> list[dict[str, str]]:
    """Keep the latest after and earliest before bound from one query."""
    selected: dict[str, tuple[int, datetime]] = {}
    for index, operator in enumerate(operators):
        operator_type = operator["type"]
        if operator_type not in {"after", "before"}:
            continue
        resolved = cast(
            datetime,
            resolve_date_bound(
                operator_type,
                operator["value"],
                reference_datetime=reference_datetime,
            ),
        )
        current = selected.get(operator_type)
        is_stricter = current is None or (
            resolved > current[1]
            if operator_type == "after"
            else resolved < current[1]
        )
        if is_stricter:
            selected[operator_type] = (index, resolved)
    selected_indexes = {item[0] for item in selected.values()}
    return [
        operator
        for index, operator in enumerate(operators)
        if operator["type"] not in {"after", "before"}
        or index in selected_indexes
    ]


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
