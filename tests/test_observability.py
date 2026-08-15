"""Observability: metrics emission never fails."""

from __future__ import annotations

import logging

import pytest

from jasa.observability.metrics import (
    emit_request_metric,
    emit_search_cache_metric,
    emit_search_metric,
)


class _BrokenString:
    def __str__(self) -> str:
        raise RuntimeError("format failed")


def test_emit_search_metric_does_not_raise() -> None:
    emit_search_metric(mode="raw", total_duration_ms=10, cache_hit=False)


def test_emit_request_metric_does_not_raise() -> None:
    emit_request_metric(route="/search", status=200, duration_ms=5)


def test_emit_search_cache_metric_does_not_raise() -> None:
    emit_search_cache_metric(event="hit")


def test_emit_search_metric_swallows_format_error() -> None:
    emit_search_metric(value=_BrokenString())


def test_emit_request_metric_swallows_format_error() -> None:
    emit_request_metric(value=_BrokenString())


def test_emit_search_cache_metric_swallows_format_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.DEBUG, logger="jasa.metrics"):
        emit_search_cache_metric(value=_BrokenString())
