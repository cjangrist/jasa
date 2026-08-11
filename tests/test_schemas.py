"""web_search input model validation (§4.1 contract)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jasa.schemas import WebSearchInput


def test_valid_input_defaults() -> None:
    validated = WebSearchInput(query="hello")
    assert validated.query == "hello"
    assert validated.timeout_ms is None
    assert validated.include_snippets is True
    assert validated.grounded_snippets is None


def test_missing_query_rejected() -> None:
    with pytest.raises(ValidationError):
        WebSearchInput()  # type: ignore[call-arg]


def test_oversized_query_rejected() -> None:
    with pytest.raises(ValidationError):
        WebSearchInput(query="x" * 2001)


def test_extra_property_rejected() -> None:
    with pytest.raises(ValidationError):
        WebSearchInput(query="q", surprise="nope")  # type: ignore[call-arg]


def test_non_positive_timeout_rejected() -> None:
    with pytest.raises(ValidationError):
        WebSearchInput(query="q", timeout_ms=0)
