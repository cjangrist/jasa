"""URL normalization parity, validated against the golden TS fixture."""

from __future__ import annotations

import encodings.idna
import json
from pathlib import Path

import pytest

from jasa.search.urls import normalize_url

_FIXTURE = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "golden"
    / "url_normalization.json"
)


def _golden_cases() -> list[tuple[str, str, str]]:
    data = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    return [(c["name"], c["input"], c["output"]) for c in data["cases"]]


@pytest.mark.parametrize(
    ("name", "raw", "expected"),
    _golden_cases(),
    ids=[case[0] for case in _golden_cases()],
)
def test_normalize_url_matches_golden(
    name: str, raw: str, expected: str
) -> None:
    assert normalize_url(raw) == expected, f"case {name!r}"


def test_punycode_fallback_on_encoding_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(label: str) -> bytes:
        raise UnicodeError("forced")

    monkeypatch.setattr(encodings.idna, "ToASCII", boom)
    assert normalize_url("https://Example.com/x") == "https://example.com/x"


def test_hostname_value_error_returns_raw(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BrokenParts:
        scheme = "https"

        @property
        def hostname(self) -> str:
            raise ValueError("invalid host")

    monkeypatch.setattr("jasa.search.urls.urlsplit", lambda _raw: BrokenParts())
    assert normalize_url("https://bad.example") == "https://bad.example"


async def test_malformed_ipv6_url() -> None:
    assert normalize_url("http://[::1") == "http://[::1"
