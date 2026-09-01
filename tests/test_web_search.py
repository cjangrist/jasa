"""web_search tool: end-to-end execution and MCP response formatting."""

from __future__ import annotations

from jasa.cache.memory import MemoryCache
from jasa.config import DEFAULT_SEARCH_MAX_RESULTS
from jasa.search.fanout import _FanoutKnobs, ProviderFailure
from jasa.search.providers.base import SearchProvider, SearchRequest
from jasa.search.ranking import RankedWebResult, SearchResult
from jasa.search.service import SearchOutcome
from jasa.tools.web_search import execute_web_search, format_web_search_response


async def _no_sleep(_seconds: float) -> None:
    return None


_KNOBS = _FanoutKnobs(retry_sleep=_no_sleep)


class Fake(SearchProvider):
    name = "fake"
    secret_env = "FAKE"
    base_url = ""
    default_timeout_s = 1.0

    def __init__(self, name: str, results: list[SearchResult]) -> None:
        self.name = name
        self._results = results

    async def search(self, request: SearchRequest) -> list[SearchResult]:
        return list(self._results)


async def test_execute_end_to_end_shapes_response() -> None:
    a = Fake("a", [SearchResult("t", "https://a.com/1", "s" * 320, "a", 0.9)])
    response = await execute_web_search(
        {"a": a}, MemoryCache(), "query", knobs=_KNOBS
    )
    assert response.query == "query"
    assert response.providers_succeeded[0].provider == "a"
    assert response.web_results[0].url == "https://a.com/1"
    assert response.web_results[0].snippets == ["s" * 320]
    assert response.web_results[0].score > 0
    assert response.web_results[0].snippet_source == "aggregated"
    assert response.truncation.total_before == 1


def test_format_strips_snippets_when_disabled() -> None:
    outcome = SearchOutcome(
        query="q",
        total_duration_ms=10,
        providers_succeeded=[],
        providers_failed=[],
        web_results=[RankedWebResult("t", "u", ["s"], ["a"], 0.5, "grounded")],
    )
    response = format_web_search_response(outcome, include_snippets=False)
    result = response.web_results[0]
    assert "snippets" not in result.model_dump()
    assert result.snippet_source == "grounded"


def test_format_keeps_snippets_when_enabled() -> None:
    outcome = SearchOutcome(
        query="q",
        total_duration_ms=10,
        providers_succeeded=[],
        providers_failed=[],
        web_results=[RankedWebResult("t", "u", ["s"], ["a"], 0.5)],
    )
    response = format_web_search_response(outcome, include_snippets=True)
    assert response.web_results[0].snippets == ["s"]
    assert response.web_results[0].snippet_source == "aggregated"


def test_format_includes_provider_failures() -> None:
    outcome = SearchOutcome(
        query="q",
        total_duration_ms=10,
        providers_succeeded=[],
        providers_failed=[ProviderFailure("p", "err", 5)],
        web_results=[],
    )
    response = format_web_search_response(outcome)
    assert response.providers_failed[0].model_dump() == {
        "provider": "p",
        "error": "err",
        "duration_ms": 5,
    }


def test_format_defaults_to_configured_search_result_ceiling() -> None:
    outcome = SearchOutcome(
        query="q",
        total_duration_ms=10,
        providers_succeeded=[],
        providers_failed=[],
        web_results=[
            RankedWebResult(
                str(index),
                f"https://same.example/{index}",
                ["s"],
                ["a"],
                1 / (index + 1),
            )
            for index in range(DEFAULT_SEARCH_MAX_RESULTS + 1)
        ],
    )
    response = format_web_search_response(outcome)
    assert len(response.web_results) == DEFAULT_SEARCH_MAX_RESULTS
    assert response.truncation.model_dump() == {
        "total_before": DEFAULT_SEARCH_MAX_RESULTS + 1,
        "kept": DEFAULT_SEARCH_MAX_RESULTS,
        "rescued": 0,
    }


def test_format_honors_custom_search_result_ceiling() -> None:
    outcome = SearchOutcome(
        query="q",
        total_duration_ms=10,
        providers_succeeded=[],
        providers_failed=[],
        web_results=[
            RankedWebResult(
                str(index),
                f"https://same.example/{index}",
                ["s"],
                ["a"],
                1 / (index + 1),
            )
            for index in range(3)
        ],
    )

    response = format_web_search_response(outcome, max_results=2)

    assert len(response.web_results) == 2
    assert response.truncation.kept == 2
