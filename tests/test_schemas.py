"""web_search input model validation (§4.1 contract)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jasa.schemas import (
    WebSearchGrounding,
    WebSearchInput,
    WebSearchProviderFailure,
    WebSearchProviderSuccess,
    WebSearchResponse,
    WebSearchResult,
    WebSearchTruncation,
)


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


def test_output_models_are_strict_frozen_and_serialize_omissions() -> None:
    response = WebSearchResponse(
        query="q",
        total_duration_ms=7,
        providers_succeeded=[
            WebSearchProviderSuccess(provider="good", duration_ms=3)
        ],
        providers_failed=[
            WebSearchProviderFailure(
                provider="bad",
                error="unavailable",
                duration_ms=4,
            )
        ],
        grounding=WebSearchGrounding(
            requested=False,
            attempted=0,
            grounded=0,
            outcomes={},
        ),
        truncation=WebSearchTruncation(
            total_before=1,
            kept=1,
            rescued=0,
        ),
        web_results=[
            WebSearchResult(
                title="Title",
                url="https://example.test",
                source_providers=["good"],
                score=0.5,
                snippet_source="aggregated",
            )
        ],
    )

    assert "snippets" not in response.model_dump()["web_results"][0]
    with pytest.raises(ValidationError):
        response.query = "changed"
    with pytest.raises(ValidationError):
        WebSearchProviderSuccess(
            provider="good",
            duration_ms=3,
            surprise=True,
        )  # type: ignore[call-arg]
