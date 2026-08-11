"""Opt-in OpenTelemetry tracing activation."""

from __future__ import annotations

import sys

import pytest

from jasa.config import TelemetrySettings
from jasa.telemetry import configure_telemetry


def test_disabled_when_exporter_empty() -> None:
    assert configure_telemetry(TelemetrySettings()) is False


def test_disabled_when_exporter_none() -> None:
    settings = TelemetrySettings(otel_traces_exporter="none")
    assert configure_telemetry(settings) is False


def test_disabled_when_sdk_disabled() -> None:
    settings = TelemetrySettings(
        otel_sdk_disabled=True, otel_traces_exporter="console"
    )
    assert configure_telemetry(settings) is False


def test_console_exporter_activates() -> None:
    settings = TelemetrySettings(otel_traces_exporter="console")
    assert configure_telemetry(settings) is True


def test_otlp_http_exporter_activates() -> None:
    settings = TelemetrySettings(
        otel_traces_exporter="otlp",
        otel_exporter_otlp_protocol="http/protobuf",
        otel_exporter_otlp_endpoint="http://collector:4318",
    )
    assert configure_telemetry(settings) is True


def test_otlp_grpc_exporter_activates() -> None:
    settings = TelemetrySettings(
        otel_traces_exporter="otlp",
        otel_exporter_otlp_protocol="grpc",
        otel_exporter_otlp_endpoint="http://collector:4317",
    )
    assert configure_telemetry(settings) is True


def test_missing_sdk_returns_false(monkeypatch: pytest.MonkeyPatch) -> None:
    for module_name in list(sys.modules):
        if module_name == "opentelemetry" or module_name.startswith(
            "opentelemetry."
        ):
            monkeypatch.setitem(sys.modules, module_name, None)
    settings = TelemetrySettings(otel_traces_exporter="console")
    assert configure_telemetry(settings) is False
