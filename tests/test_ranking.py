"""RRF ranking + truncation parity, validated against the golden TS fixtures."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, cast

from jasa.search.operators import apply_search_operators
from jasa.search.ranking import (
    _apply_quality_filters,
    _host_of,
    rank_and_merge,
    RankedWebResult,
    SearchResult,
    truncate_web_results,
)

_FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "golden"
_LONG = "a" * 320


def _result(
    provider: str, url: str, snippet: str = _LONG, score: float | None = None
) -> SearchResult:
    return SearchResult(url, url, snippet, provider, score)


def _summarize(ranked: list[RankedWebResult]) -> list[dict[str, Any]]:
    return [
        {
            "url": result.url,
            "score": result.score,
            "source_providers": result.source_providers,
            "snippets": result.snippets,
        }
        for result in ranked
    ]


def _assert_ranked(
    actual: list[dict[str, Any]], expected: list[dict[str, Any]]
) -> None:
    assert len(actual) == len(expected)
    for got, want in zip(actual, expected, strict=True):
        assert got["url"] == want["url"], (got, want)
        assert got["source_providers"] == want["source_providers"], got["url"]
        assert got["snippets"] == want["snippets"], got["url"]
        assert math.isclose(got["score"], want["score"], rel_tol=1e-12), got[
            "url"
        ]


def _load(name: str) -> dict[str, Any]:
    return cast(
        dict[str, Any],
        json.loads((_FIXTURE_DIR / f"{name}.json").read_text(encoding="utf-8")),
    )


def test_stability_scoreless_order() -> None:
    ranked = rank_and_merge(
        {
            "alpha": [
                _result("alpha", "https://a.com/1"),
                _result("alpha", "https://a.com/2"),
                _result("alpha", "https://a.com/3"),
            ],
            "beta": [
                _result("beta", "https://b.com/1"),
                _result("beta", "https://b.com/2"),
            ],
        },
        "test",
        skip_quality_filter=True,
    )
    _assert_ranked(
        _summarize(ranked), _load("ranking_stability_scoreless")["output"]
    )


def test_scored_single_provider_order() -> None:
    ranked = rank_and_merge(
        {
            "tavily": [
                _result("tavily", "https://t.com/1", _LONG, 0.9),
                _result("tavily", "https://t.com/2", _LONG, 0.5),
                _result("tavily", "https://t.com/3", _LONG, 0.1),
            ]
        },
        "test",
        skip_quality_filter=True,
    )
    _assert_ranked(
        _summarize(ranked), _load("ranking_scored_single_provider")["output"]
    )


def test_dedup_merge() -> None:
    ranked = rank_and_merge(
        {
            "alpha": [
                _result("alpha", "https://shared.com/page", "alpha snippet one")
            ],
            "beta": [
                _result("beta", "https://shared.com/page", "beta snippet two")
            ],
        },
        "test",
        skip_quality_filter=True,
    )
    _assert_ranked(_summarize(ranked), _load("ranking_dedup_merge")["output"])


def test_quality_filter_keeps_via_golden() -> None:
    ranked = rank_and_merge(
        {
            "alpha": [
                _result("alpha", "https://keep.com/multi-a", "x" * 50),
                _result("alpha", "https://keep.com/long-a", "y" * 320),
            ],
            "beta": [_result("beta", "https://keep.com/multi-a", "z" * 50)],
        },
        "test",
    )
    _assert_ranked(
        _summarize(ranked), _load("ranking_quality_filter")["output"]
    )


def test_same_provider_duplicate_url_dedups_internally() -> None:
    ranked = rank_and_merge(
        {
            "alpha": [
                _result("alpha", "https://a.com/x", "s1"),
                _result("alpha", "https://a.com/x", "s1"),
            ]
        },
        "test",
        skip_quality_filter=True,
    )
    assert len(ranked) == 1
    assert ranked[0].source_providers == ["alpha"]
    assert ranked[0].snippets == ["s1"]
    assert math.isclose(ranked[0].score, 1 / (60 + 1), rel_tol=1e-12)


def test_same_provider_normalized_duplicates_contribute_once() -> None:
    ranked = rank_and_merge(
        {
            "alpha": [
                _result("alpha", "https://a.com/x/", "highest", 0.9),
                _result("alpha", "https://a.com/x", "lower", 0.5),
            ]
        },
        "test",
        skip_quality_filter=True,
    )
    assert len(ranked) == 1
    assert ranked[0].url == "https://a.com/x/"
    assert ranked[0].snippets == ["highest"]
    assert math.isclose(ranked[0].score, 1 / (60 + 1), rel_tol=1e-12)


def test_quality_filter_drops_low_score() -> None:
    result = RankedWebResult("t", "https://x.com/a", ["s"], ["only"], 0.005)
    assert _apply_quality_filters([result]) == []


def test_quality_filter_drops_short_single_provider() -> None:
    result = RankedWebResult("t", "https://x.com/a", ["short"], ["only"], 0.02)
    assert _apply_quality_filters([result]) == []


def test_quality_filter_keeps_result_without_snippet() -> None:
    result = RankedWebResult("t", "https://x.com/a", [], ["only"], 0.02)
    assert _apply_quality_filters([result]) == [result]


def test_unknown_operator_type_is_ignored() -> None:
    params = apply_search_operators(
        {"base_query": "x", "operators": [{"type": "mystery", "value": "v"}]}
    )
    assert params == {"query": "x"}


def test_truncate_rescues_distinct_hosts() -> None:
    ranked = rank_and_merge(
        {f"p{i}": [_result(f"p{i}", f"https://h{i}.com/x")] for i in range(22)},
        "test",
        skip_quality_filter=True,
    )
    truncated = truncate_web_results(ranked, 20)
    fixture = _load("truncate_rescue_distinct_hosts")
    assert truncated.truncation.total_before == fixture["total_input"]
    assert truncated.truncation.kept == fixture["truncation"]["kept"]
    assert truncated.truncation.rescued == fixture["truncation"]["rescued"]
    assert [r.url for r in truncated.results] == fixture["output_urls"]


def test_truncate_no_rescue_same_host() -> None:
    ranked = rank_and_merge(
        {"big": [_result("big", f"https://same.com/p{i}") for i in range(25)]},
        "test",
        skip_quality_filter=True,
    )
    truncated = truncate_web_results(ranked, 20)
    fixture = _load("truncate_no_rescue_same_host")
    assert truncated.truncation.total_before == fixture["total_input"]
    assert truncated.truncation.rescued == 0
    assert [r.url for r in truncated.results] == fixture["output_urls"]


def test_truncate_below_top_n_is_passthrough() -> None:
    small = [
        RankedWebResult("t", f"https://h{i}.com/x", ["s"], ["p"], 0.1)
        for i in range(5)
    ]
    truncated = truncate_web_results(small, 20)
    assert truncated.results == small
    assert truncated.truncation == type(truncated.truncation)(5, 5, 0)


def test_host_of_unparseable_returns_none() -> None:
    assert _host_of("not a url") is None


def test_host_of_invalid_ipv6_returns_none() -> None:
    assert _host_of("http://[::1") is None
