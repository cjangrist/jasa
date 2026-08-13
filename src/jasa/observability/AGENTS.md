# AGENTS.md — `src/jasa/observability/`

This package is the lightweight metrics surface. Distributed tracing bootstrap
lives one level up in `telemetry.py`; colored application logging lives in
`logging.py`.

## Files and behavior

- `metrics.py` exposes `emit_search_metric(**fields)` and
  `emit_request_metric(**fields)`.
- `__init__.py` marks the package scope.

Both functions format key/value fields only at DEBUG level and swallow every
error. Instrumentation must never fail or delay a user request. Do not put
secrets, raw authorization headers, or full environment mappings into fields.

The current facade has no external exporter and no durable metric sink. If an
exporter is added, retain a no-op/fail-open default and test its absence.

## Tests

`tests/test_observability.py` verifies normal emission and formatting failure.
Telemetry SDK/exporter behavior is covered separately in
`tests/test_telemetry.py`.
