"""Pure hostname and date validation for Keenable search filters."""

from __future__ import annotations

import calendar
import re
from datetime import date, datetime

DATE_VALUE_SOURCE = (
    r"\d+(?:min|h|d|mo|y)|"
    r"\d{4}(?:-\d{2}(?:-\d{2}(?:T\d{2}:\d{2}:\d{2}"
    r"(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?)?)?)?"
)
DATE_VALUE_PATTERN = re.compile(rf"(?:{DATE_VALUE_SOURCE})\Z")

_MAX_YEAR = 9999
_YEAR_PATTERN = re.compile(r"\d{4}")
_YEAR_MONTH_PATTERN = re.compile(r"(\d{4})-(\d{2})")
_RELATIVE_DATE_PATTERN = re.compile(r"\d+(?:min|h|d|mo|y)\Z")
_SITE_VALUE_PATTERN = re.compile(
    r"(?=.{1,253}\Z)"
    r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?"
    r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)*\Z"
)


def is_clean_site_value(value: str) -> bool:
    """Return whether a value is a clean hostname accepted as native site."""
    return bool(_SITE_VALUE_PATTERN.fullmatch(value))


def normalize_date_bound(operator_type: str, value: str) -> str:
    """Expand Jasa's partial dates to Keenable-valid inclusive bounds."""
    if _YEAR_PATTERN.fullmatch(value):
        suffix = "01-01" if operator_type == "after" else "12-31"
        return f"{value}-{suffix}"
    if match := _YEAR_MONTH_PATTERN.fullmatch(value):
        year, month = map(int, match.groups())
        day = (
            1
            if operator_type == "after"
            else calendar.monthrange(year, month)[1]
        )
        return f"{value}-{day:02d}"
    return value


def is_valid_date_bound(value: str) -> bool:
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
