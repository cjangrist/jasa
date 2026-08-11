"""Opt-in OpenTelemetry tracing (a no-op by default).

Installs an SDK ``TracerProvider`` only when ``OTEL_TRACES_EXPORTER`` selects
an exporter; the SDK imports are lazy (the optional ``telemetry`` extra).
Configured once at bootstrap so both the jasa server and the mounted omnifetch
child emit spans through the same provider.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from jasa.config import TelemetrySettings
from jasa.logging import get_logger

if TYPE_CHECKING:
    from opentelemetry.sdk.trace.export import SpanExporter

_LOGGER = get_logger("telemetry")
_DISABLED_EXPORTERS = frozenset({"", "none"})


def _build_otlp_exporter(endpoint: str, protocol: str) -> SpanExporter:
    """Construct an OTLP span exporter for the configured wire protocol."""
    target_endpoint = endpoint or None
    if protocol == "grpc":
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter as GrpcSpanExporter,
        )

        return GrpcSpanExporter(endpoint=target_endpoint)
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
        OTLPSpanExporter as HttpSpanExporter,
    )

    return HttpSpanExporter(endpoint=target_endpoint)


def configure_telemetry(settings: TelemetrySettings) -> bool:
    """Install an OpenTelemetry SDK tracer provider when tracing is enabled.

    Returns ``True`` when tracing was activated, ``False`` when it remained a
    no-op (disabled by configuration, or the optional ``telemetry`` extra is not
    installed).
    """
    exporter_name = settings.otel_traces_exporter
    if settings.otel_sdk_disabled or exporter_name in _DISABLED_EXPORTERS:
        _LOGGER.info(
            "Telemetry disabled (OTEL_TRACES_EXPORTER=%r); tracing is a no-op.",
            settings.otel_traces_exporter,
        )
        return False
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import (
            BatchSpanProcessor,
            ConsoleSpanExporter,
            SimpleSpanProcessor,
        )
    except ImportError:
        _LOGGER.error(
            "Telemetry requested but the OpenTelemetry SDK is missing. "
            "Install the extra:  uv sync --extra telemetry."
        )
        return False

    resource = Resource.create({"service.name": settings.otel_service_name})
    provider = TracerProvider(resource=resource)
    if exporter_name == "console":
        provider.add_span_processor(
            SimpleSpanProcessor(ConsoleSpanExporter(out=sys.stderr))
        )
    else:
        exporter = _build_otlp_exporter(
            settings.otel_exporter_otlp_endpoint,
            settings.otel_exporter_otlp_protocol,
        )
        provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    _LOGGER.info(
        "Telemetry enabled (exporter=%s, service=%s).",
        exporter_name,
        settings.otel_service_name,
    )
    return True
