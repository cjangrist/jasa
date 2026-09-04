"""Search operator parity, validated against the golden TS fixture."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from jasa.search.operators import (
    apply_search_operators,
    build_query_with_operators,
    parse_search_operators,
)

_FIXTURE = (
    Path(__file__).resolve().parent / "fixtures" / "golden" / "operators.json"
)


def _cases() -> list[dict[str, Any]]:
    data = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    return list(data["cases"])


@pytest.mark.parametrize(
    "case", _cases(), ids=[str(c["name"]) for c in _cases()]
)
def test_operators_match_golden(case: dict[str, Any]) -> None:
    parsed = parse_search_operators(str(case["input"]))
    assert parsed == case["parsed"], f"parsed {case['name']}"
    applied = apply_search_operators(parsed)
    assert applied == case["applied"], f"applied {case['name']}"
    assert build_query_with_operators(applied) == case["built_default"], (
        f"built_default {case['name']}"
    )
    assert (
        build_query_with_operators(
            applied, options={"exclude_file_type": True, "exclude_dates": True}
        )
        == case["built_kagi"]
    ), f"built_kagi {case['name']}"


def test_parser_can_leave_selected_operator_types_intact() -> None:
    parsed = parse_search_operators(
        "custom:after:2025 before:2024 query",
        excluded_types=frozenset({"after", "before"}),
    )
    assert parsed == {
        "base_query": "custom:after:2025 before:2024 query",
        "operators": [],
    }
