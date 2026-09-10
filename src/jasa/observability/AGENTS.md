# AGENTS.md — `src/jasa/observability/`

This package owns lightweight metrics and optional durable `web_search` request
traces. OpenTelemetry bootstrap lives one level up in `telemetry.py`; colored
application logging lives in `logging.py`.

## Files and behavior

- `metrics.py` exposes `emit_search_metric(**fields)` and
  `emit_request_metric(**fields)`, plus bounded
  `emit_search_cache_metric(**fields)` and
  `emit_grounding_cache_metric(**fields)` events.
- `__init__.py` marks the package scope.
- `traces.py` owns request-local trace state, provider records, and shared
  HTTP-client event hooks.
- `trace_delivery.py` redacts and serializes completed records, then delivers
  them through a bounded queue to a generic S3-compatible endpoint.

All functions format key/value fields only at DEBUG level and swallow every
error. Search and grounding cache events are `hit`, `miss`, `write`,
`read_skipped`, `write_skipped`, `read_error`, `write_error`, and `coalesced`;
error events may include only the exception class. Deadline-skipped reads and
writes use their respective `*_skipped` event without an error field because the
cache backend did not fail. Instrumentation must never fail or delay a user
request. Do not put queries, fetched content, grounded output, cache keys,
secrets, raw authorization headers, or full environment mappings into fields.

Metrics have no external exporter or durable sink. Request tracing is disabled
by default and fail-open when enabled. Search completion may only take a bounded
in-memory snapshot and call the sink's non-awaiting `put_nowait` path. JSON
encoding, S3 client construction, signing, DNS, TLS, and object delivery belong
in `asyncio.to_thread`. Queue saturation increments an in-memory counter on the
request path and reports the aggregate from the delivery worker. Decoded bodies
are capped per HTTP call, per trace, and across all accepted queue entries.

Trace object keys remain
`<prefix>/tool=web_search/date=YYYY-MM-DD/hour=HH/trace_id=<uuid>.json`.
Preserve the legacy document shape, recursively redact credential-bearing
names, scrub captured credential values from the whole document, and never put
S3 destination credentials into trace content or application logs. The one
composition-owned provider-secret snapshot supplies the scrub set so cache-hit
traces cannot bypass value redaction.

## Tests

`tests/test_observability.py` verifies normal metric emission and formatting
failure. `tests/test_request_tracing.py` owns trace shape, redaction, HTTP
capture, queue, thread, and search-context behavior. Telemetry SDK/exporter
behavior is covered separately in `tests/test_telemetry.py`.
