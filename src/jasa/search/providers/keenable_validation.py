"""Pure hostname and date validation for Keenable search filters."""

from __future__ import annotations

import calendar
import re
from datetime import date, datetime, time, UTC

DATE_VALUE_SOURCE = (
    r"\d+(?:min|h|d|mo|y)|"
    r"\d{4}(?:-\d{2}(?:-\d{2}(?:T\d{2}:\d{2}:\d{2}"
    r"(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?)?)?)?"
)
DATE_VALUE_PATTERN = re.compile(rf"(?:{DATE_VALUE_SOURCE})\Z")

_YEAR_PATTERN = re.compile(r"\d{4}")
_YEAR_MONTH_PATTERN = re.compile(r"(\d{4})-(\d{2})")
_RELATIVE_DATE_PATTERN = re.compile(r"(?P<amount>\d+)(?:min|h|d|mo|y)\Z")
_TIMEZONE_OFFSET_PATTERN = re.compile(
    r"[+-](?P<hours>\d{2}):(?P<minutes>\d{2})\Z"
)
_MAXIMUM_TIMEZONE_OFFSET_HOURS = 23
_MAXIMUM_TIMEZONE_OFFSET_MINUTES = 59
_MINIMUM_ABSOLUTE_DATE = date(1970, 1, 1)
_MAXIMUM_ABSOLUTE_DATE = date(2149, 6, 5)
_MINIMUM_ABSOLUTE_DATETIME = datetime.combine(
    _MINIMUM_ABSOLUTE_DATE, time.min, UTC
)
_MAXIMUM_ABSOLUTE_DATETIME = datetime.combine(
    _MAXIMUM_ABSOLUTE_DATE, time.max, UTC
)
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


def is_valid_date_bound(operator_type: str, value: str) -> bool:
    """Return whether a bound is accepted by Keenable's live API."""
    if relative_match := _RELATIVE_DATE_PATTERN.fullmatch(value):
        return int(relative_match.group("amount")) > 0
    if offset_match := _TIMEZONE_OFFSET_PATTERN.search(value):
        if int(offset_match.group("hours")) > _MAXIMUM_TIMEZONE_OFFSET_HOURS:
            return False
        if (
            int(offset_match.group("minutes"))
            > _MAXIMUM_TIMEZONE_OFFSET_MINUTES
        ):
            return False
    try:
        normalized_value = normalize_date_bound(operator_type, value)
        if "T" in normalized_value:
            parsed_datetime = datetime.fromisoformat(
                normalized_value.replace("Z", "+00:00")
            )
            if parsed_datetime.tzinfo is None:
                parsed_datetime = parsed_datetime.replace(tzinfo=UTC)
            else:
                parsed_datetime = parsed_datetime.astimezone(UTC)
        else:
            parsed_date = date.fromisoformat(normalized_value)
            parsed_datetime = datetime.combine(parsed_date, time.min, UTC)
    except ValueError:
        return False
    return (
        _MINIMUM_ABSOLUTE_DATETIME
        <= parsed_datetime
        <= _MAXIMUM_ABSOLUTE_DATETIME
    )
