# AGENTS.md — `src/jasa/observability/`

This package owns lightweight metrics and optional durable `web_search` and
public `web_fetch` request traces. OpenTelemetry bootstrap lives one level up
in `telemetry.py`; colored
application logging lives in `logging.py`.

## Files and behavior

- `metrics.py` exposes `emit_search_metric(**fields)` and
  `emit_request_metric(**fields)`, plus bounded
  `emit_search_cache_metric(**fields)` and
  `emit_grounding_cache_metric(**fields)` events.
- `__init__.py` marks the package scope.
- `traces.py` owns request-local trace state, fetch middleware, provider
  records, and shared HTTP-client event hooks.
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
by default and fail-open when enabled. Search or fetch completion reserves one
bounded submission slot and schedules snapshot preparation without awaiting it. Bounded
copying, JSON encoding, S3 client construction and closure, signing, DNS, TLS,
and object delivery belong in `asyncio.to_thread`. Queue saturation increments
an in-memory counter on the request path and reports the aggregate from the
delivery worker. Decoded bodies are capped per HTTP call and per trace; complete
snapshots and all accepted queue entries have independent byte caps.

Trace object keys are
`<prefix>/tool=<web_search|web_fetch>/date=YYYY-MM-DD/hour=HH/trace_id=<uuid>.json`.
Preserve the legacy document shape, recursively redact credential-bearing
names, scrub captured credential values from the whole document, and never put
S3 destination credentials into trace content or application logs. The one
composition-owned provider-secret snapshot supplies the scrub set so cache-hit
traces cannot bypass value redaction. Dynamic secret discovery scans raw
provider input/output, decisions, both HTTP directions, and the final result
before field redaction, then scrubs duplicates from every string. Incomplete
streams retain any bounded partial body and report truncation; preloaded HTTPX
content is already decoded. A non-cacheable search waiter records the
`in_process_flight` strategy and no provider calls of its own, so durable traces
do not misclassify a shared result as another provider fan-out.
Sensitive container fields contribute every nested string leaf to the scrub
set. URL detection ignores surrounding whitespace and scheme casing, fragments
are removed, and signed-URL credential and signature parameters are redacted.
Oversized
snapshots preserve the document contract, add `trace_truncated=true`, and bound
all retained provider output, decision details, HTTP data, and final results.
Fetch snapshots retain provider-attempt metadata before bounded page content.
Omnifetch responses retain their origin-provider evidence across persistent
cache hits and in-process flight replays. Fetch documents therefore label the
provider sections as `returned_result_origin` and current-request execution as
`unknown`; consumers must not treat those sections as billing evidence for the
current request.

## Tests

`tests/test_observability.py` verifies normal metric emission and formatting
failure. `tests/test_request_tracing.py` owns search shape, redaction, HTTP
capture, queue, thread, and search-context behavior.
`tests/test_fetch_request_tracing.py` owns fetch shape, middleware, transport,
provider evidence, and event-loop isolation. Telemetry SDK/exporter behavior is
covered separately in `tests/test_telemetry.py`.
