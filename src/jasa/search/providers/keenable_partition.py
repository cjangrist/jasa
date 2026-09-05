"""Quote-aware token partitioning for Keenable search queries."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, UTC
from typing import cast

from jasa.search.providers.keenable_validation import (
    DATE_VALUE_PATTERN,
    DATE_VALUE_SOURCE,
    is_clean_site_value,
    is_valid_date_bound,
)

_MIN_SITE_ALTERNATIVES = 2
_MIN_GLUED_SIGN_POSITION = 2
_QUOTED_CLAUSE_PATTERN = re.compile(
    r"(?<!\w)(?P<operator>-?(?:site|filetype|ext|intitle|inurl|inbody|"
    r'inpage|lang(?:uage)?|loc(?:ation)?|before|after)):"'
    r'(?P<operator_value>[^\"]+)"(?P<operator_suffix>[^\s,;|()\[\]{}+]+)?|'
    r'(?<![^\s,;|()\[\]{}])(?P<term>[+-])"(?P<term_value>[^\"]+)"'
    r"(?P<term_suffix>[^\s,;|()\[\]{}+]+)?|"
    r'"(?P<exact>[^\"]+)"(?P<exact_suffix>[^\s,;|()\[\]{}+]+)?'
)
_DATE_OPERATOR_PATTERN = re.compile(
    rf"(?<!\w)(?P<date_operator>before|after):"
    rf"(?P<date_value>{DATE_VALUE_SOURCE})"
    r"(?=$|[\s,;|()\[\]{}+])"
)
_SITE_OPERATOR_PATTERN = re.compile(
    r"(?<!\w)(?P<site_operator>-?site):"
    r"(?P<site_value>[^\s,;|()\[\]{}\"]+)(?=$|[\s,;|()\[\]{}])"
)
_GENERIC_OPERATOR_PATTERN = re.compile(
    r"(?<!\w)"
    r"(?P<generic_operator>-?(?:filetype|ext|intitle|inurl|inbody|inpage|"
    r"lang(?:uage)?|loc(?:ation)?)):"
    r"(?P<generic_value>[^\s,;|()\[\]{}\"]+)(?=$|[\s,;|()\[\]{}])"
)
_SITE_ALTERNATIVE_VALUE_SOURCE = r'(?:(?:"[^\"]+")|[^\s,;|()\[\]{}\"]+)'
_SITE_ALTERNATIVE_MEMBER_SOURCE = rf"site:{_SITE_ALTERNATIVE_VALUE_SOURCE}"
_SITE_ALTERNATIVE_PATTERN = re.compile(
    rf"(?P<site_alternatives>"
    rf"\(\s*{_SITE_ALTERNATIVE_MEMBER_SOURCE}"
    rf"(?:\s+OR\s+{_SITE_ALTERNATIVE_MEMBER_SOURCE})+\s*\)|"
    rf"{_SITE_ALTERNATIVE_MEMBER_SOURCE}"
    rf"(?:\s+OR\s+{_SITE_ALTERNATIVE_MEMBER_SOURCE})+)"
)
_BOOLEAN_GROUP_PATTERN = re.compile(
    r"(?P<boolean_group>\([^()]*\b(?:AND|OR|NOT)\b[^()]*\))",
    re.IGNORECASE,
)
_SITE_ALTERNATIVE_MEMBER_PATTERN = re.compile(
    r'site:(?:"(?P<quoted_site_value>[^\"]+)"|'
    r"(?P<site_alternative_value>[^\s,;|()\[\]{}\"]+))"
)
_NEGATED_DATE_PATTERN = re.compile(
    r"(?<![^\s,;|()\[\]{}])"
    r"(?P<negated_date>-(?:before|after):[^\s,;|()\[\]{}\"]+)"
)
_NEGATED_EMPTY_OPERATOR_PATTERN = re.compile(
    r"(?<![^\s,;|()\[\]{}])"
    r"(?P<negated_empty>-(?:site|filetype|ext|intitle|inurl|inbody|"
    r"inpage|lang(?:uage)?|loc(?:ation)?|before|after):)"
    r"(?=$|[\s,;|()\[\]{}+])"
)
_LOGICAL_TOKEN_PATTERN = re.compile(r'(?:[^\"\s]+|"[^\"]*")+')
_BLOCKER_TOKEN_CLAUSE_PATTERNS = (
    _QUOTED_CLAUSE_PATTERN,
    _DATE_OPERATOR_PATTERN,
    _SITE_OPERATOR_PATTERN,
    _GENERIC_OPERATOR_PATTERN,
    _NEGATED_DATE_PATTERN,
    _NEGATED_EMPTY_OPERATOR_PATTERN,
)
_TOKEN_PREFIX_PATTERN = re.compile(r"[^\s,;|)\]}]*\Z")
_TOKEN_SUFFIX_PATTERN = re.compile(r"[^\s,;|]*")
_OPERATOR_PREFIX_WRAPPERS = frozenset(",;|()[]{}+")
_OPERATOR_PREFIX_BLOCKERS = frozenset(":/?=&")
_WHITESPACE_PATTERN = re.compile(r"\s")
_BOOLEAN_TOKEN_PATTERN = re.compile(
    r"(?<![^\s,;|()\[\]{}+])(?P<boolean>AND|OR|NOT)"
    r"(?=$|[\s,;|()\[\]{}+])",
    re.IGNORECASE,
)
_BOOLEAN_WORDS = frozenset({"and", "or", "not"})
_BOOLEAN_BOUNDARIES = frozenset(",;|()[]{}+")
_BOOLEAN_LEFT_GAP = frozenset("([{+")
_BOOLEAN_RIGHT_GAP = frozenset(")]}+")
_SEPARATOR_ONLY_PATTERN = re.compile(r"^[\s,;|+]*$")
_WRAPPER_PAIRS = {"(": ")", "[": "]", "{": "}"}
_WRAPPER_CLOSERS = frozenset(_WRAPPER_PAIRS.values())
_WRAPPER_CHARACTERS = frozenset((*_WRAPPER_PAIRS, *_WRAPPER_CLOSERS))
GENERIC_OPERATOR_TYPES = {
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
PARTITIONED_OPERATOR_TYPES = frozenset(
    {
        "after",
        "before",
        "site",
        "exclude_site",
        "exact",
        "force_include",
        "exclude_term",
        *GENERIC_OPERATOR_TYPES.values(),
    }
)
LITERAL_INCLUDE_SITE_TYPE = "literal_include_site"
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

ClausePart = str | tuple[dict[str, str] | None, str]


@dataclass(frozen=True, slots=True)
class _QueryStructure:
    """One-pass structural facts shared by every clause decision."""

    text: str
    depths: tuple[int, ...]
    inside_quotes: tuple[bool, ...]
    protected_wrapper_context: tuple[bool, ...]
    minimum_pipe_depth: int
    boolean_depth_before: tuple[int, ...]
    boolean_depth_after: tuple[int, ...]
    uppercase_boolean_depth_before: tuple[int, ...]
    uppercase_boolean_depth_after: tuple[int, ...]


def partition_special_clauses(
    query: str, *, reference_datetime: datetime | None = None
) -> list[ClausePart]:
    """Partition quoted, native, and protected literal clauses."""
    quote_positions = frozenset(_unescaped_quote_positions(query))
    if _has_escaped_quote(query, quote_positions) or _has_malformed_wrappers(
        query, quote_positions
    ):
        return [(None, query)]
    reference = reference_datetime or datetime.now(UTC)
    structure = _query_structure(query, quote_positions)
    unmatched_quote_position = _unmatched_quote_position(quote_positions)
    partitionable_end = (
        _unmatched_literal_start(query, unmatched_quote_position)
        if unmatched_quote_position is not None
        else len(query)
    )
    partitionable_query = query[:partitionable_end]
    alternative_groups = _count_site_alternative_groups(
        partitionable_query, structure
    )
    if unmatched_quote_position is not None and alternative_groups:
        alternative_groups += 1
    parts = _partition_blocker_tokens(
        partitionable_query,
        alternative_groups,
        structure=structure,
        query_offset=0,
        reference_datetime=reference,
    )
    if unmatched_quote_position is not None:
        parts.append((None, query[partitionable_end:]))
    return _collapse_emptied_scaffolding(parts)


def _count_site_alternative_groups(
    query: str, structure: _QueryStructure
) -> int:
    """Count promotable site-alternative groups in the complete query."""
    return len(
        {
            (match.start(), match.end())
            for match in _SITE_ALTERNATIVE_PATTERN.finditer(query)
            if not structure.inside_quotes[match.start()]
            if _site_alternative_operators(match.group()) is not None
        }
    )


def _partition_blocker_tokens(
    query: str,
    total_alternative_groups: int,
    *,
    structure: _QueryStructure,
    query_offset: int,
    reference_datetime: datetime,
) -> list[ClausePart]:
    """Protect whole URL-like tokens before parsing individual clauses."""
    parts: list[ClausePart] = []
    cursor = 0
    for match in _LOGICAL_TOKEN_PATTERN.finditer(query):
        if not _blocker_token_requires_literal(match.group()):
            continue
        parts.extend(
            _partition_site_alternatives(
                query[cursor : match.start()],
                total_alternative_groups,
                structure=structure,
                query_offset=query_offset + cursor,
                reference_datetime=reference_datetime,
            )
        )
        parts.append((None, match.group()))
        cursor = match.end()
    parts.extend(
        _partition_site_alternatives(
            query[cursor:],
            total_alternative_groups,
            structure=structure,
            query_offset=query_offset + cursor,
            reference_datetime=reference_datetime,
        )
    )
    return parts


def _blocker_token_requires_literal(token: str) -> bool:
    """Return whether structural clauses sit inside a URL/custom token."""
    found_clause, residue = _blocker_token_residue(token)
    return found_clause and any(
        marker in residue for marker in _OPERATOR_PREFIX_BLOCKERS
    )


def _blocker_token_residue(token: str) -> tuple[bool, str]:
    """Remove recognized clauses and return their surrounding token text."""
    matches = sorted(
        (
            match
            for pattern in _BLOCKER_TOKEN_CLAUSE_PATTERNS
            for match in pattern.finditer(token)
        ),
        key=lambda match: match.start(),
    )
    cursor = 0
    residue: list[str] = []
    for match in matches:
        if match.start() < cursor:
            continue
        residue.append(token[cursor : match.start()])
        cursor = match.end()
    if cursor == 0:
        return False, token
    residue.append(token[cursor:])
    return True, "".join(residue)


def _partition_site_alternatives(
    query: str,
    total_alternative_groups: int,
    *,
    structure: _QueryStructure,
    query_offset: int,
    reference_datetime: datetime,
) -> list[ClausePart]:
    """Consume site alternatives without leaving Boolean scaffolding."""
    parts: list[ClausePart] = []
    cursor = 0
    has_multiple_alternative_groups = total_alternative_groups > 1
    matches = sorted(
        (
            *_BOOLEAN_GROUP_PATTERN.finditer(query),
            *_SITE_ALTERNATIVE_PATTERN.finditer(query),
        ),
        key=lambda match: match.start(),
    )
    for match in matches:
        if match.start() < cursor:
            continue
        operators = _site_alternative_operators(match.group())
        first_site = match.start() + match.group().find("site:")
        ambiguous = structure.inside_quotes[query_offset + match.start()]
        ambiguous = ambiguous or _has_ambiguous_operator_prefix(
            query, first_site
        )
        ambiguous = ambiguous or _has_token_continuation(query, match.end())
        ambiguous = (
            ambiguous
            or structure.protected_wrapper_context[query_offset + match.start()]
        )
        ambiguous = ambiguous or _has_boolean_neighbor(
            structure,
            query_offset + match.start(),
            query_offset + match.end(),
            uppercase_only=True,
        )
        ambiguous = ambiguous or has_multiple_alternative_groups
        if operators is None or ambiguous:
            literal_start = max(
                cursor, _operator_token_start(query, match.start())
            )
            literal_end = _operator_token_end(query, match.end())
            parts.extend(
                _partition_quoted_clauses(
                    query[cursor:literal_start],
                    structure=structure,
                    query_offset=query_offset + cursor,
                    reference_datetime=reference_datetime,
                )
            )
            literal_operator = (
                _literal_include_site_operator("")
                if _SITE_ALTERNATIVE_MEMBER_PATTERN.search(match.group())
                else None
            )
            parts.append((literal_operator, query[literal_start:literal_end]))
            cursor = literal_end
            continue
        clause_start, clause_end = _native_clause_bounds(
            query,
            _clause_start_with_plus(query, match.start()),
            match.end(),
        )
        replacement = _native_clause_replacement(
            query, clause_start, clause_end
        )
        parts.extend(
            _partition_quoted_clauses(
                query[cursor:clause_start],
                structure=structure,
                query_offset=query_offset + cursor,
                reference_datetime=reference_datetime,
            )
        )
        parts.extend((operator, "") for operator in operators)
        if replacement:
            parts.append(replacement)
        cursor = clause_end
    parts.extend(
        _partition_quoted_clauses(
            query[cursor:],
            structure=structure,
            query_offset=query_offset + cursor,
            reference_datetime=reference_datetime,
        )
    )
    return parts


def _partition_quoted_clauses(
    query: str,
    *,
    structure: _QueryStructure,
    query_offset: int,
    reference_datetime: datetime,
) -> list[ClausePart]:
    """Partition balanced quoted clauses and remaining unquoted text."""
    parts: list[ClausePart] = []
    cursor = 0
    for match in _QUOTED_CLAUSE_PATTERN.finditer(query):
        if match.start() < cursor:
            continue
        operator = _quoted_operator(match, reference_datetime)
        has_nested_prefix = _has_ambiguous_operator_prefix(query, match.start())
        has_nested_prefix = has_nested_prefix or _has_glued_quoted_sign(
            query, match.start()
        )
        has_nested_prefix = (
            has_nested_prefix
            or (
                structure.protected_wrapper_context[
                    query_offset + match.start()
                ]
            )
        )
        if has_nested_prefix:
            literal_start = max(
                cursor, _operator_token_start(query, match.start())
            )
            literal_end = _operator_token_end(query, match.end())
            parts.extend(
                _partition_unquoted_clauses(
                    query[cursor:literal_start],
                    structure=structure,
                    query_offset=query_offset + cursor,
                    reference_datetime=reference_datetime,
                )
            )
            parts.append(
                (
                    _literalized_operator(operator),
                    query[literal_start:literal_end],
                )
            )
            cursor = literal_end
            continue
        if operator is not None and operator["type"] not in (
            _DATE_OPERATOR_TYPES | {"site", LITERAL_INCLUDE_SITE_TYPE}
        ):
            operator = None
        if (
            operator is not None
            and operator["type"] in _DATE_OPERATOR_TYPES | _SITE_OPERATOR_TYPES
            and _has_ambiguous_operator_suffix(query, match.end())
        ):
            operator = _literalized_operator(operator)
        if operator is not None and _has_boolean_neighbor(
            structure,
            query_offset + match.start(),
            query_offset + match.end(),
            uppercase_only=operator["type"]
            in _DATE_OPERATOR_TYPES | _SITE_OPERATOR_TYPES,
        ):
            operator = _literalized_operator(operator)
        is_promoted_native = operator is not None and operator["type"] in (
            _DATE_OPERATOR_TYPES | _SITE_OPERATOR_TYPES
        )
        clause_start = match.start()
        clause_end = match.end()
        if is_promoted_native:
            clause_start, clause_end = _native_clause_bounds(
                query, clause_start, clause_end
            )
        parts.extend(
            _partition_unquoted_clauses(
                query[cursor:clause_start],
                structure=structure,
                query_offset=query_offset + cursor,
                reference_datetime=reference_datetime,
            )
        )
        replacement = match.group() if not is_promoted_native else ""
        if is_promoted_native and _native_clause_replacement(
            query, clause_start, clause_end
        ):
            replacement = " "
        parts.append((operator, replacement))
        cursor = clause_end
    parts.extend(
        _partition_unquoted_clauses(
            query[cursor:],
            structure=structure,
            query_offset=query_offset + cursor,
            reference_datetime=reference_datetime,
        )
    )
    return parts


def _site_alternative_operators(text: str) -> list[dict[str, str]] | None:
    """Return clean site operators from one Boolean alternative sequence."""
    if _SITE_ALTERNATIVE_PATTERN.fullmatch(text) is None:
        return None
    values = [
        str(match.group("quoted_site_value"))
        if match.group("quoted_site_value") is not None
        else str(match.group("site_alternative_value"))
        for match in _SITE_ALTERNATIVE_MEMBER_PATTERN.finditer(text)
    ]
    if len(values) < _MIN_SITE_ALTERNATIVES or any(
        not is_clean_site_value(value) for value in values
    ):
        return None
    return [{"type": "site", "value": value} for value in values]


def _partition_unquoted_clauses(
    text: str,
    *,
    structure: _QueryStructure,
    query_offset: int,
    reference_datetime: datetime,
) -> list[ClausePart]:
    """Partition native filters and protect ambiguous date literals."""
    matches = sorted(
        (
            *_DATE_OPERATOR_PATTERN.finditer(text),
            *_SITE_OPERATOR_PATTERN.finditer(text),
            *_GENERIC_OPERATOR_PATTERN.finditer(text),
            *_NEGATED_DATE_PATTERN.finditer(text),
            *_NEGATED_EMPTY_OPERATOR_PATTERN.finditer(text),
        ),
        key=lambda match: match.start(),
    )
    parts: list[ClausePart] = []
    cursor = 0
    for match in matches:
        if match.start() < cursor:
            continue
        has_ambiguous_suffix = match.lastgroup in {
            "date_value",
            "site_value",
            "generic_value",
        } and _has_ambiguous_operator_suffix(text, match.end())
        has_boolean_neighbor = match.lastgroup in {
            "date_value",
            "site_value",
            "generic_value",
        } and _has_boolean_neighbor(
            structure,
            query_offset + match.start(),
            query_offset + match.end(),
            uppercase_only=match.lastgroup in {"date_value", "site_value"},
        )
        if match.lastgroup != "negated_date" and (
            _has_ambiguous_operator_prefix(text, match.start())
            or structure.protected_wrapper_context[query_offset + match.start()]
            or has_ambiguous_suffix
            or has_boolean_neighbor
        ):
            literal_start = max(
                cursor, _operator_token_start(text, match.start())
            )
            literal_end = _operator_token_end(text, match.end())
            parts.append(text[cursor:literal_start])
            parts.append(
                (
                    _literal_site_operator_from_match(match),
                    text[literal_start:literal_end],
                )
            )
            cursor = literal_end
            continue
        if match.lastgroup == "date_value":
            operator, replacement = _unquoted_date_operator(
                match, reference_datetime
            )
        elif match.lastgroup == "site_value":
            operator, replacement = _unquoted_site_operator(match)
        elif match.lastgroup == "generic_value":
            operator = None
            replacement = match.group()
        elif match.lastgroup == "negated_date":
            operator = None
            replacement = str(match.group("negated_date"))
        else:
            operator = None
            replacement = str(match.group("negated_empty"))
        clause_start = match.start()
        clause_end = match.end()
        if operator is not None:
            clause_start, clause_end = _native_clause_bounds(
                text, clause_start, clause_end
            )
            if _native_clause_replacement(text, clause_start, clause_end):
                replacement = " "
        parts.append(text[cursor:clause_start])
        parts.append((operator, replacement))
        cursor = clause_end
    parts.append(text[cursor:])
    return parts


def _is_extracted(part: ClausePart) -> bool:
    """Return whether a promoted clause left no literal text."""
    return not isinstance(part, str) and part[0] is not None and not part[1]


def _collapse_emptied_scaffolding(parts: list[ClausePart]) -> list[ClausePart]:
    """Drop wrappers and separators emptied by native extraction."""
    working = list(parts)
    while True:
        collapsed = _collapse_one_group(working)
        if collapsed is None:
            return _collapse_separator_runs(working)
        working = collapsed


def _collapse_one_group(parts: list[ClausePart]) -> list[ClausePart] | None:
    """Collapse the innermost wrapper pair emptied by extraction."""
    for opening_index, part in enumerate(parts[:-1]):
        if not isinstance(part, str) or not part.rstrip().endswith(
            ("(", "[", "{")
        ):
            continue
        opening = part.rstrip()[-1]
        closing = _WRAPPER_PAIRS[opening]
        found_extraction = False
        for closing_index in range(opening_index + 1, len(parts)):
            candidate = parts[closing_index]
            if _is_extracted(candidate):
                found_extraction = True
                continue
            if not isinstance(candidate, str):
                break
            if _SEPARATOR_ONLY_PATTERN.match(candidate):
                continue
            if found_extraction and candidate.lstrip().startswith(closing):
                updated = list(parts)
                head = part.rstrip()
                updated[opening_index] = head[:-1] + part[len(head) :]
                tail = candidate.lstrip()
                updated[closing_index] = (
                    candidate[: len(candidate) - len(tail)] + tail[1:]
                )
                return updated
            break
    return None


def _collapse_separator_runs(parts: list[ClausePart]) -> list[ClausePart]:
    """Blank separators stranded beside extracted clauses."""
    updated = list(parts)
    for index, part in enumerate(updated):
        if not isinstance(part, str) or not part.strip(" \t\n\r\f\v"):
            continue
        if not _SEPARATOR_ONLY_PATTERN.match(part):
            continue
        before = index > 0 and _is_extracted(updated[index - 1])
        after = index + 1 < len(updated) and _is_extracted(updated[index + 1])
        literal_before = index > 0 and _has_literal_content(updated[index - 1])
        literal_after = index + 1 < len(updated) and _has_literal_content(
            updated[index + 1]
        )
        if (before and literal_after) or (after and literal_before):
            continue
        if before or after:
            updated[index] = " " if part.strip() != part else ""
    return updated


def _has_literal_content(part: ClausePart) -> bool:
    """Return whether a neighboring part contributes query text."""
    value = part if isinstance(part, str) else part[1]
    return bool(value and _SEPARATOR_ONLY_PATTERN.fullmatch(value) is None)


def _unquoted_date_operator(
    match: re.Match[str],
    reference_datetime: datetime,
) -> tuple[dict[str, str] | None, str]:
    """Promote an API-valid date clause or preserve it literally."""
    value = str(match.group("date_value"))
    operator_type = str(match.group("date_operator"))
    if not is_valid_date_bound(
        operator_type, value, reference_datetime=reference_datetime
    ):
        return None, match.group()
    return {"type": operator_type, "value": value}, ""


def _unquoted_site_operator(
    match: re.Match[str],
) -> tuple[dict[str, str] | None, str]:
    """Promote a clean domain clause or preserve it literally."""
    value = str(match.group("site_value"))
    operator_name = str(match.group("site_operator"))
    if operator_name.startswith("-"):
        return None, match.group()
    if not is_clean_site_value(value):
        return _literal_include_site_operator(value), match.group()
    return {"type": "site", "value": value}, ""


def _has_ambiguous_operator_prefix(text: str, position: int) -> bool:
    """Return whether an operator-like clause is nested in another token."""
    plus_cursor = position
    while plus_cursor and text[plus_cursor - 1] == "+":
        plus_cursor -= 1
    if position - plus_cursor > 1:
        return True
    if position:
        previous = text[position - 1]
        if not previous.isspace() and previous not in _OPERATOR_PREFIX_WRAPPERS:
            return True
    prefix = text[_operator_token_start(text, position) : position]
    return any(
        marker in prefix for marker in _OPERATOR_PREFIX_BLOCKERS
    ) or _has_signed_wrapper_left(text, position)


def _has_signed_wrapper_left(text: str, position: int) -> bool:
    """Return whether a signed wrapper sequence ends at ``position``."""
    cursor = position
    while cursor and text[cursor - 1].isspace():
        cursor -= 1
    found_wrapper = False
    while cursor and text[cursor - 1] in _WRAPPER_PAIRS:
        found_wrapper = True
        cursor -= 1
    while cursor and text[cursor - 1].isspace():
        cursor -= 1
    return found_wrapper and cursor > 0 and text[cursor - 1] in "+-"


def _literal_include_site_operator(value: str) -> dict[str, str]:
    """Build an internal marker for a literal positive site clause."""
    return {"type": LITERAL_INCLUDE_SITE_TYPE, "value": value}


def _literalized_operator(
    operator: dict[str, str] | None,
) -> dict[str, str] | None:
    """Retain an internal marker when a positive site becomes literal."""
    if operator is not None and operator["type"] == LITERAL_INCLUDE_SITE_TYPE:
        return operator
    if operator is None or operator["type"] != "site":
        return None
    return _literal_include_site_operator(operator["value"])


def _literal_site_operator_from_match(
    match: re.Match[str],
) -> dict[str, str] | None:
    """Mark an ambiguous positive unquoted site clause as literal."""
    if match.lastgroup != "site_value":
        return None
    operator_name = str(match.group("site_operator"))
    if operator_name.startswith("-"):
        return None
    return _literal_include_site_operator(str(match.group("site_value")))


def _operator_token_start(text: str, position: int) -> int:
    """Return the start of the punctuation-bearing token at ``position``."""
    while position and (
        not text[position - 1].isspace() and text[position - 1] not in ",;|)]}"
    ):
        position -= 1
    return position


def _operator_token_end(text: str, position: int) -> int:
    """Return the end of a punctuation-bearing token at ``position``."""
    token_suffix = _TOKEN_SUFFIX_PATTERN.match(text, position)
    return position + len(cast(re.Match[str], token_suffix).group())


def _clause_start_with_plus(text: str, position: int) -> int:
    """Include adjacent unary-plus wrappers in a promoted clause."""
    original_position = position
    while position and text[position - 1] == "+":
        position -= 1
    if not position:
        return position
    preceding = text[position - 1]
    if preceding.isspace() or preceding in ",;|()[]{}":
        return position
    return original_position


def _native_clause_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    """Consume wrappers and one separator emptied by native extraction."""
    start = _clause_start_with_plus(text, start)
    while True:
        left = start
        right = end
        while left and text[left - 1].isspace():
            left -= 1
        while right < len(text) and text[right].isspace():
            right += 1
        if not left or right >= len(text):
            break
        opening = text[left - 1]
        if _WRAPPER_PAIRS.get(opening) != text[right]:
            break
        start = left - 1
        end = right + 1
    left = start
    right = end
    while left and text[left - 1].isspace():
        left -= 1
    while right < len(text) and text[right].isspace():
        right += 1
    if left and text[left - 1] in ",;":
        start = left - 1
    elif right < len(text) and text[right] in ",;":
        end = right + 1
    return start, end


def _native_clause_replacement(text: str, start: int, end: int) -> str:
    """Separate substantive neighbors when extraction consumes punctuation."""
    if (
        start
        and end < len(text)
        and text[end - 1] in ",;"
        and not text[start - 1].isspace()
        and not text[end].isspace()
        and text[start - 1] not in _WRAPPER_CHARACTERS
        and text[end] not in _WRAPPER_CHARACTERS
    ):
        return " "
    return ""


def _unescaped_quote_positions(text: str) -> list[int]:
    """Return quote positions not escaped by an odd backslash run."""
    positions: list[int] = []
    backslash_run = 0
    for position, character in enumerate(text):
        if character == "\\":
            backslash_run += 1
            continue
        if character == '"' and backslash_run % 2 == 0:
            positions.append(position)
        backslash_run = 0
    return positions


def _has_escaped_quote(text: str, quote_positions: frozenset[int]) -> bool:
    """Return whether any quote follows an odd backslash run."""
    return text.count('"') != len(quote_positions)


def _unmatched_quote_position(
    quote_positions: frozenset[int],
) -> int | None:
    """Return the unmatched opening quote, if the text has one."""
    return max(quote_positions) if len(quote_positions) % 2 else None


def _unmatched_literal_start(text: str, position: int) -> int:
    """Include a token directly attached to an unmatched opening quote."""
    while position and (
        not text[position - 1].isspace() and text[position - 1] not in ",;|+"
    ):
        position -= 1
    return position


def _has_malformed_wrappers(text: str, quote_positions: frozenset[int]) -> bool:
    """Return whether unquoted wrappers are unbalanced or misnested."""
    stack: list[str] = []
    inside_quote = False
    for position, character in enumerate(text):
        if position in quote_positions:
            inside_quote = not inside_quote
        elif not inside_quote and character in _WRAPPER_PAIRS:
            stack.append(character)
        elif (
            not inside_quote
            and character in _WRAPPER_CLOSERS
            and (not stack or _WRAPPER_PAIRS[stack.pop()] != character)
        ):
            return True
    return bool(stack)


def _query_structure(
    text: str, quote_positions: frozenset[int]
) -> _QueryStructure:
    """Index quote state, wrapper depth, Booleans, and pipes once."""
    depths: list[int] = []
    inside_quotes: list[bool] = []
    protected_wrapper_context: list[bool] = []
    pipe_positions: list[int] = []
    protected_wrapper_stack: list[bool] = []
    protected_wrapper_depth = 0
    depth = 0
    inside_quote = False
    literal_pipe_positions = _literal_pipe_positions(text)
    for position, character in enumerate(text):
        depths.append(depth)
        inside_quotes.append(inside_quote)
        protected_wrapper_context.append(protected_wrapper_depth > 0)
        if position in quote_positions:
            inside_quote = not inside_quote
        elif not inside_quote and character in _WRAPPER_PAIRS:
            depth += 1
            is_protected_wrapper = (
                protected_wrapper_depth > 0
                or _has_protected_wrapper_prefix(text, position)
            )
            protected_wrapper_stack.append(is_protected_wrapper)
            protected_wrapper_depth += int(is_protected_wrapper)
        elif not inside_quote and character in _WRAPPER_CLOSERS:
            depth = max(0, depth - 1)
            protected_wrapper_depth -= int(protected_wrapper_stack.pop())
        elif (
            not inside_quote
            and character == "|"
            and position not in literal_pipe_positions
        ):
            pipe_positions.append(position)
    depths.append(depth)
    inside_quotes.append(inside_quote)
    protected_wrapper_context.append(protected_wrapper_depth > 0)
    boolean_tokens = tuple(
        (
            match.start(),
            depths[match.start()],
            str(match.group("boolean")).isupper(),
        )
        for match in _BOOLEAN_TOKEN_PATTERN.finditer(text)
        if not inside_quotes[match.start()]
    )
    boolean_depth_before, boolean_depth_after = _minimum_token_depth_boundaries(
        len(text),
        tuple(
            (position, token_depth)
            for position, token_depth, _ in boolean_tokens
        ),
    )
    uppercase_depth_before, uppercase_depth_after = (
        _minimum_token_depth_boundaries(
            len(text),
            tuple(
                (position, token_depth)
                for position, token_depth, uppercase in boolean_tokens
                if uppercase
            ),
        )
    )
    return _QueryStructure(
        text,
        tuple(depths),
        tuple(inside_quotes),
        tuple(protected_wrapper_context),
        min(
            (depths[position] for position in pipe_positions),
            default=len(text) + 1,
        ),
        boolean_depth_before,
        boolean_depth_after,
        uppercase_depth_before,
        uppercase_depth_after,
    )


def _literal_pipe_positions(text: str) -> frozenset[int]:
    """Return pipes belonging to URL-like or custom-token residue."""
    positions: set[int] = set()
    for match in _LOGICAL_TOKEN_PATTERN.finditer(text):
        _, residue = _blocker_token_residue(match.group())
        if not any(marker in residue for marker in _OPERATOR_PREFIX_BLOCKERS):
            continue
        positions.update(
            match.start() + offset
            for offset, character in enumerate(match.group())
            if character == "|"
        )
    return frozenset(positions)


def _has_protected_wrapper_prefix(text: str, position: int) -> bool:
    """Return whether a wrapper belongs to a signed or fielded expression."""
    while position and text[position - 1].isspace():
        position -= 1
    if position and text[position - 1] in "+-":
        return True
    prefix = text[_operator_token_start(text, position) : position]
    return any(marker in prefix for marker in _OPERATOR_PREFIX_BLOCKERS)


def _minimum_token_depth_boundaries(
    text_length: int, tokens: tuple[tuple[int, int], ...]
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Build O(1) minimum-depth lookups outside a clause span."""
    sentinel = text_length + 1
    depths_at_position = dict(tokens)
    before: list[int] = []
    minimum_depth = sentinel
    for position in range(text_length + 1):
        before.append(minimum_depth)
        minimum_depth = min(
            minimum_depth, depths_at_position.get(position, sentinel)
        )
    after = [sentinel] * (text_length + 1)
    minimum_depth = sentinel
    for position in range(text_length, -1, -1):
        minimum_depth = min(
            minimum_depth, depths_at_position.get(position, sentinel)
        )
        after[position] = minimum_depth
    return tuple(before), tuple(after)


def _has_ambiguous_operator_suffix(text: str, position: int) -> bool:
    """Return whether a delimiter splits an operator-like token value."""
    if position >= len(text):
        return False
    boundary = text[position]
    if boundary in "([{+":
        return True
    if boundary not in ")]}":
        return False
    while position < len(text) and text[position] in ")]}":
        position += 1
    if position >= len(text):
        return False
    following = text[position]
    return not following.isspace() and following not in ",;|()[]{}+"


def _has_token_continuation(text: str, position: int) -> bool:
    """Return whether a grouped clause continues in the same token."""
    if position >= len(text):
        return False
    following = text[position]
    return not following.isspace() and following not in ",;|()[]{}+"


def _has_glued_quoted_sign(text: str, position: int) -> bool:
    """Return whether an exact quote follows a plus inside a custom token."""
    if position < _MIN_GLUED_SIGN_POSITION or text[position - 1] != "+":
        return False
    preceding = text[position - 2]
    return not preceding.isspace() and preceding not in ",;|()[]{}"


def _has_boolean_neighbor(
    structure: _QueryStructure,
    start: int,
    end: int,
    *,
    uppercase_only: bool = False,
) -> bool:
    """Return whether lifting a clause could escape Boolean scope."""
    text = structure.text
    clause_depth = structure.depths[start]
    if structure.minimum_pipe_depth <= clause_depth:
        return True
    if _has_adjacent_boolean(text, start, end):
        return True
    before = (
        structure.uppercase_boolean_depth_before
        if uppercase_only
        else structure.boolean_depth_before
    )
    after = (
        structure.uppercase_boolean_depth_after
        if uppercase_only
        else structure.boolean_depth_after
    )
    return before[start] <= clause_depth or after[end] <= clause_depth


def _has_adjacent_boolean(text: str, start: int, end: int) -> bool:
    """Return whether a Boolean word directly neighbors one clause."""
    return _boolean_on_left(text, start) or _boolean_on_right(text, end)


def _boolean_on_left(text: str, position: int) -> bool:
    """Recognize a left Boolean without copying the query prefix."""
    cursor = position
    while cursor and (
        text[cursor - 1].isspace() or text[cursor - 1] in _BOOLEAN_LEFT_GAP
    ):
        cursor -= 1
    token_end = cursor
    while cursor and text[cursor - 1].isalpha():
        cursor -= 1
    token = text[cursor:token_end].casefold()
    boundary = cursor == 0 or _is_boolean_boundary(text[cursor - 1])
    return token in _BOOLEAN_WORDS and boundary


def _boolean_on_right(text: str, position: int) -> bool:
    """Recognize a right Boolean without copying the query suffix."""
    cursor = position
    while cursor < len(text) and (
        text[cursor].isspace() or text[cursor] in _BOOLEAN_RIGHT_GAP
    ):
        cursor += 1
    token_start = cursor
    while cursor < len(text) and text[cursor].isalpha():
        cursor += 1
    token = text[token_start:cursor].casefold()
    boundary = cursor == len(text) or _is_boolean_boundary(text[cursor])
    return token in _BOOLEAN_WORDS and boundary


def _is_boolean_boundary(character: str) -> bool:
    """Return whether one character can delimit a Boolean token."""
    return character.isspace() or character in _BOOLEAN_BOUNDARIES


def _quoted_operator(
    match: re.Match[str], reference_datetime: datetime
) -> dict[str, str] | None:
    """Classify one balanced quoted clause as a shared search operator."""
    if operator_name := match.group("operator"):
        operator_type = _QUOTED_OPERATOR_TYPES.get(operator_name, "")
        value = str(match.group("operator_value"))
        invalid_operator = not operator_type or bool(
            match.group("operator_suffix")
        )
        invalid_date = operator_type in _DATE_OPERATOR_TYPES and (
            not DATE_VALUE_PATTERN.fullmatch(value)
            or not is_valid_date_bound(
                operator_type,
                value,
                reference_datetime=reference_datetime,
            )
        )
        invalid_site = operator_type in _SITE_OPERATOR_TYPES and (
            _WHITESPACE_PATTERN.search(value) or not is_clean_site_value(value)
        )
        if invalid_operator or invalid_date or invalid_site:
            if operator_name == "site":
                return _literal_include_site_operator(value)
            return None
        if operator_type not in _DATE_OPERATOR_TYPES | _SITE_OPERATOR_TYPES:
            value = f'"{value}"'
    elif sign := match.group("term"):
        operator_type = "force_include" if sign == "+" else "exclude_term"
        if match.group("term_suffix"):
            operator_type = ""
        value = f'"{match.group("term_value")}"'
    else:
        operator_type = "" if match.group("exact_suffix") else "exact"
        value = str(match.group("exact"))
    if not operator_type:
        return None
    return {"type": operator_type, "value": value}
