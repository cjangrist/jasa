"""Grounding detectors + prompt parity (pure-function tests)."""

from __future__ import annotations

import hashlib
from pathlib import Path

from jasa.grounding.detectors import (
    complete_coverage_line,
    detect_grounded_junk,
    detect_grounded_sentinel,
    FENCE_REPAIR_SUFFIX,
    grounding_detector_semantics,
    has_complete_coverage_line,
    repair_unbalanced_fence,
)
from jasa.grounding.prompts import (
    build_grounded_user_message,
    COVERAGE_LINE_MAX_CHARS,
    SYSTEM_PROMPT,
)

_GOLDEN_PROMPT = (
    Path(__file__).resolve().parent
    / "fixtures/golden/grounded_system_prompt.txt"
)


def test_system_prompt_hash_is_pinned() -> None:
    """The verbatim system prompt's SHA-256 must match the pinned value."""
    assert _GOLDEN_PROMPT.read_text(encoding="utf-8") == SYSTEM_PROMPT
    assert (
        hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()
        == "07c7e74757f0515bbb4aaa193c7e71fd2990915b0dc64155e30f91b662862799"
    )


def test_build_user_message_truncates_long_content() -> None:
    message = build_grounded_user_message("q", "t", "x" * 3000, 100)
    assert "Query: q" in message
    assert "Page title: t" in message
    assert "[content truncated]" in message
    assert len(message) < 3000


def test_build_user_message_uses_untitled() -> None:
    message = build_grounded_user_message("q", "", "content", 1000)
    assert "(untitled)" in message


def test_junk_empty_body() -> None:
    assert detect_grounded_junk("") == "empty_body"


def test_junk_tight_always_fires() -> None:
    assert detect_grounded_junk("blah subscribe to continue reading blah")
    assert detect_grounded_junk("please enable javascript and retry")


def test_junk_ambiguous_fires_on_short_body() -> None:
    assert detect_grounded_junk("access denied") == "pattern:access denied"


def test_junk_ambiguous_does_not_fire_on_long_body() -> None:
    long_body = "a" * 3001 + " access denied"
    assert detect_grounded_junk(long_body) is None


def test_junk_clean_content_is_none() -> None:
    assert (
        detect_grounded_junk("This is a genuine article about Python.") is None
    )


def test_sentinel_exact_match() -> None:
    assert (
        detect_grounded_sentinel("[no usable content]") == "[no usable content]"
    )
    assert detect_grounded_sentinel("[login required]") == "[login required]"


def test_sentinel_normalized_match() -> None:
    assert (
        detect_grounded_sentinel("**[page not found]**.") == "[page not found]"
    )
    assert (
        detect_grounded_sentinel('"[navigation only]"') == "[navigation only]"
    )


def test_sentinel_long_prose_is_not_sentinel() -> None:
    long = "x" * 201 + " [search results page]"
    assert detect_grounded_sentinel(long) is None


def test_fence_repair_closes_unbalanced() -> None:
    snippet = "Code:\n```python\nprint(1)\n"
    repaired = repair_unbalanced_fence(snippet)
    assert repaired.count("```") % 2 == 0
    assert repaired.endswith(FENCE_REPAIR_SUFFIX)
    assert grounding_detector_semantics()["fence_repair_suffix"] == (
        FENCE_REPAIR_SUFFIX
    )


def test_fence_repair_leaves_balanced() -> None:
    snippet = "```\ncode\n```"
    assert repair_unbalanced_fence(snippet) == snippet


def test_sentinel_not_detected_on_normal_snippet() -> None:
    assert detect_grounded_sentinel("A normal snippet about the topic.") is None


def test_complete_coverage_line_must_end_a_normal_snippet() -> None:
    valid = "Body.\nCoverage: answers x; does NOT cover y."
    assert has_complete_coverage_line(valid) is True
    assert has_complete_coverage_line("Body without coverage.") is False
    assert has_complete_coverage_line(valid + "\nTrailing text.") is False
    too_long = "Coverage: answers x; does NOT cover y.".ljust(
        COVERAGE_LINE_MAX_CHARS + 1, "."
    )
    assert has_complete_coverage_line(f"Body.\n{too_long}") is False
    assert complete_coverage_line(valid) == (
        "Coverage: answers x; does NOT cover y."
    )


def test_sentinel_prose_framed_substring() -> None:
    """Tier 2: sentinel inside a short framing sentence (not exact match)."""
    assert (
        detect_grounded_sentinel("body: [no usable content] end")
        == "[no usable content]"
    )
