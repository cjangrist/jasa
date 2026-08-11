"""Snippet collapse parity, validated against the golden TS fixture."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from jasa.search.ranking import RankedWebResult
from jasa.search.snippets import (
    build_bigrams,
    collapse_snippets,
    jaccard,
    select_best_snippet,
    sentence_merge,
)

_FIXTURE = (
    Path(__file__).resolve().parent / "fixtures" / "golden" / "snippets.json"
)


def _cases() -> list[dict[str, Any]]:
    data = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    return list(data["cases"])


@pytest.mark.parametrize(
    "case", _cases(), ids=[str(c["name"]) for c in _cases()]
)
def test_collapse_matches_golden(case: dict[str, Any]) -> None:
    result = RankedWebResult(
        title="",
        url="",
        snippets=list(case["input"]),
        source_providers=[],
        score=0.0,
    )
    collapsed = collapse_snippets([result], str(case["query"]))
    assert collapsed[0].snippets == case["output"], case["name"]


def test_jaccard_empty_union_is_nan_like_javascript() -> None:
    assert math.isnan(jaccard(frozenset(), frozenset()))
    assert math.isnan(jaccard(build_bigrams(["solo"]), build_bigrams(["one"])))


def test_single_word_snippets_return_primary() -> None:
    result = RankedWebResult(
        title="",
        url="",
        snippets=["alpha", "beta"],
        source_providers=[],
        score=0.0,
    )
    assert collapse_snippets([result], "query")[0].snippets == ["alpha"]


def test_select_best_with_single_candidate() -> None:
    assert select_best_snippet(["only one here"], "q") == "only one here"


def test_select_best_diverse_but_no_sentences_returns_primary() -> None:
    assert select_best_snippet(["ab cd ef", "xy zw uv"], "q") == "ab cd ef"


def test_sentence_merge_exceeding_budget_returns_empty() -> None:
    long_sentence = "this is a sentence far longer than the tiny budget allows"
    assert sentence_merge([long_sentence], 10) == ""


def test_sentence_merge_no_bigram_break_returns_empty() -> None:
    assert sentence_merge(["abcdefghijklmnopqrstuvwxyz"], 500) == ""


def test_sentence_merge_dedupes_near_identical() -> None:
    first = "the quick brown fox jumps over the lazy dog here"
    second = "the quick brown fox jumps over the lazy dog now"
    assert sentence_merge([first, second], 500) == first
