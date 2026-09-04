"""Quote-aware token partitioning for Keenable search queries."""

from __future__ import annotations

import re
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
    r'(?P<operator_value>[^\"]+)"(?P<operator_suffix>[\w./:+-]+)?|'
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
    r"(?P<boolean_group>\([^()]*\b(?:AND|OR|NOT)\b[^()]*\))"
)
_SITE_ALTERNATIVE_MEMBER_PATTERN = re.compile(
    r'site:(?:"(?P<quoted_site_value>[^\"]+)"|'
    r"(?P<site_alternative_value>[^\s,;|()\[\]{}\"]+))"
)
_NEGATED_DATE_PATTERN = re.compile(
    r"(?<![^\s,;|()\[\]{}])"
    r"(?P<negated_date>-(?:before|after):[^\s,;|()\[\]{}\"]+)"
)
_LOGICAL_TOKEN_PATTERN = re.compile(r'(?:[^\"\s,;|]+|"[^\"]*")+')
_BLOCKER_TOKEN_CLAUSE_PATTERNS = (
    _QUOTED_CLAUSE_PATTERN,
    _DATE_OPERATOR_PATTERN,
    _SITE_OPERATOR_PATTERN,
    _GENERIC_OPERATOR_PATTERN,
    _NEGATED_DATE_PATTERN,
)
_TOKEN_PREFIX_PATTERN = re.compile(r"[^\s,;|)\]}]*\Z")
_TOKEN_SUFFIX_PATTERN = re.compile(r"[^\s,;|]*")
_UNMATCHED_QUOTE_PREFIX_PATTERN = re.compile(r"[^\s,;|]*\Z")
_OPERATOR_PREFIX_WRAPPERS = frozenset(",;|()[]{}+")
_OPERATOR_PREFIX_BLOCKERS = frozenset(":/?=&")
_WHITESPACE_PATTERN = re.compile(r"\s")
_BOOLEAN_LEFT_PATTERN = re.compile(
    r"(?<![^\s,;|()\[\]{}+])(?:AND|OR|NOT)"
    r"(?=$|[\s,;|()\[\]{}+])[\s(\[\]{+]*\Z"
)
_BOOLEAN_RIGHT_PATTERN = re.compile(
    r"\A[\s)\]}+]*(?:AND|OR|NOT)(?=$|[\s,;|()\[\]{}+])"
)
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
        *GENERIC_OPERATOR_TYPES.values(),
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

ClausePart = str | tuple[dict[str, str] | None, str]


def partition_special_clauses(query: str) -> list[ClausePart]:
    """Partition quoted, native, and protected literal clauses."""
    if (unmatched_quote := _unmatched_quote_position(query)) is not None:
        literal_start = _unmatched_quote_literal_start(query, unmatched_quote)
        return [
            *partition_special_clauses(query[:literal_start]),
            (None, query[literal_start:]),
        ]
    return _partition_blocker_tokens(query)


def _partition_blocker_tokens(query: str) -> list[ClausePart]:
    """Protect whole URL-like tokens before parsing individual clauses."""
    parts: list[ClausePart] = []
    cursor = 0
    for match in _LOGICAL_TOKEN_PATTERN.finditer(query):
        if not _blocker_token_requires_literal(match.group()):
            continue
        parts.extend(
            _partition_site_alternatives(query[cursor : match.start()])
        )
        parts.append((None, match.group()))
        cursor = match.end()
    parts.extend(_partition_site_alternatives(query[cursor:]))
    return parts


def _blocker_token_requires_literal(token: str) -> bool:
    """Return whether structural clauses sit inside a URL/custom token."""
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
        return False
    residue.append(token[cursor:])
    return any(
        marker in "".join(residue) for marker in _OPERATOR_PREFIX_BLOCKERS
    )


def _partition_site_alternatives(query: str) -> list[ClausePart]:
    """Consume site alternatives without leaving Boolean scaffolding."""
    parts: list[ClausePart] = []
    cursor = 0
    alternative_spans = {
        (match.start(), match.end())
        for match in _SITE_ALTERNATIVE_PATTERN.finditer(query)
        if _site_alternative_operators(match.group()) is not None
    }
    has_multiple_alternative_groups = len(alternative_spans) > 1
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
        ambiguous = _is_inside_quote(query, match.start())
        ambiguous = ambiguous or _has_ambiguous_operator_prefix(
            query, first_site
        )
        ambiguous = ambiguous or _has_token_continuation(query, match.end())
        ambiguous = ambiguous or _has_boolean_neighbor(
            query, match.start(), match.end()
        )
        ambiguous = ambiguous or has_multiple_alternative_groups
        if operators is None or ambiguous:
            literal_start = max(
                cursor, _operator_token_start(query, match.start())
            )
            literal_end = _operator_token_end(query, match.end())
            parts.extend(_partition_quoted_clauses(query[cursor:literal_start]))
            parts.append((None, query[literal_start:literal_end]))
            cursor = literal_end
            continue
        clause_start = _clause_start_with_plus(query, match.start())
        parts.extend(_partition_quoted_clauses(query[cursor:clause_start]))
        parts.extend((operator, "") for operator in operators)
        cursor = match.end()
    parts.extend(_partition_quoted_clauses(query[cursor:]))
    return parts


def _partition_quoted_clauses(query: str) -> list[ClausePart]:
    """Partition balanced quoted clauses and remaining unquoted text."""
    parts: list[ClausePart] = []
    cursor = 0
    for match in _QUOTED_CLAUSE_PATTERN.finditer(query):
        if match.start() < cursor:
            continue
        operator = _quoted_operator(match)
        has_nested_prefix = _has_ambiguous_operator_prefix(query, match.start())
        has_nested_prefix = has_nested_prefix or _has_glued_quoted_sign(
            query, match.start()
        )
        if has_nested_prefix:
            literal_start = max(
                cursor, _operator_token_start(query, match.start())
            )
            literal_end = _operator_token_end(query, match.end())
            parts.extend(
                _partition_unquoted_clauses(query[cursor:literal_start])
            )
            parts.append((None, query[literal_start:literal_end]))
            cursor = literal_end
            continue
        if operator is not None and _has_boolean_neighbor(
            query, match.start(), match.end()
        ):
            operator = None
        parts.extend(_partition_unquoted_clauses(query[cursor : match.start()]))
        replacement = "" if operator is not None else match.group()
        parts.append((operator, replacement))
        cursor = match.end()
    parts.extend(_partition_unquoted_clauses(query[cursor:]))
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


def _partition_unquoted_clauses(text: str) -> list[ClausePart]:
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
        } and _has_boolean_neighbor(text, match.start(), match.end())
        if match.lastgroup != "negated_date" and (
            _has_ambiguous_operator_prefix(text, match.start())
            or has_ambiguous_suffix
            or has_boolean_neighbor
        ):
            literal_start = max(
                cursor, _operator_token_start(text, match.start())
            )
            literal_end = _operator_token_end(text, match.end())
            parts.append(text[cursor:literal_start])
            parts.append((None, text[literal_start:literal_end]))
            cursor = literal_end
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
                    "type": GENERIC_OPERATOR_TYPES[operator_name],
                    "value": str(match.group("generic_value")),
                }
                replacement = ""
        else:
            operator = None
            replacement = str(match.group("negated_date"))
        clause_start = (
            _clause_start_with_plus(text, match.start())
            if operator is not None
            else match.start()
        )
        parts.append(text[cursor:clause_start])
        parts.append((operator, replacement))
        cursor = match.end()
    parts.append(text[cursor:])
    return parts


def _unquoted_date_operator(
    match: re.Match[str],
) -> tuple[dict[str, str] | None, str]:
    """Promote an API-valid date clause or preserve it literally."""
    value = str(match.group("date_value"))
    operator_type = str(match.group("date_operator"))
    if not is_valid_date_bound(operator_type, value):
        return None, match.group()
    return {"type": operator_type, "value": value}, ""


def _unquoted_site_operator(
    match: re.Match[str],
) -> tuple[dict[str, str] | None, str]:
    """Promote a clean domain clause or preserve it literally."""
    value = str(match.group("site_value"))
    if not is_clean_site_value(value):
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


def _unmatched_quote_position(text: str) -> int | None:
    """Return the unmatched opening quote, if the text has one."""
    quote_positions = [
        position for position, character in enumerate(text) if character == '"'
    ]
    return quote_positions[-1] if len(quote_positions) % 2 else None


def _unmatched_quote_literal_start(text: str, position: int) -> int:
    """Return the start of the token attached to an unmatched quote."""
    token_prefix = _UNMATCHED_QUOTE_PREFIX_PATTERN.search(text[:position])
    return position - len(cast(re.Match[str], token_prefix).group())


def _is_inside_quote(text: str, position: int) -> bool:
    """Return whether a position occurs inside a balanced quoted phrase."""
    return text[:position].count('"') % 2 == 1


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


def _has_boolean_neighbor(text: str, start: int, end: int) -> bool:
    """Return whether a clause participates in a Boolean expression."""
    return bool(
        _BOOLEAN_LEFT_PATTERN.search(text[:start])
        or _BOOLEAN_RIGHT_PATTERN.search(text[end:])
    )


def _quoted_operator(match: re.Match[str]) -> dict[str, str] | None:
    """Classify one balanced quoted clause as a shared search operator."""
    if operator_name := match.group("operator"):
        operator_type = _QUOTED_OPERATOR_TYPES.get(operator_name, "")
        value = str(match.group("operator_value"))
        invalid_operator = not operator_type or bool(
            match.group("operator_suffix")
        )
        invalid_date = operator_type in _DATE_OPERATOR_TYPES and (
            not DATE_VALUE_PATTERN.fullmatch(value)
            or not is_valid_date_bound(operator_type, value)
        )
        invalid_site = operator_type in _SITE_OPERATOR_TYPES and (
            _WHITESPACE_PATTERN.search(value) or not is_clean_site_value(value)
        )
        if invalid_operator or invalid_date or invalid_site:
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
