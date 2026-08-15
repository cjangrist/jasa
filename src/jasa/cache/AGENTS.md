# AGENTS.md — `src/jasa/cache/`

Search caching owns a string-store protocol, versioned semantic identities,
strict records, and a completeness gate. Grounding owns its v1 records in
`grounding/cache.py`; omnifetch owns fetch v1 records. Runtime storage uses
omnifetch's shared cachelib adapter, which Jasa configures once and injects into
all three surfaces.

## Files

- `base.py` — `CacheBackend`, `SearchCacheIdentity`, `make_cache_key`,
  `should_cache`, the v2 key prefix, and the default 129,600-second TTL.
- `memory.py` — legacy-compatible process-local store; no longer runtime-selected.
- `disk.py` — legacy-compatible JSON-file store; no longer runtime-selected;
  filesystem work runs in worker threads so caller deadlines stay enforceable,
  with directory-and-event-loop shared fair lock stripes owned by shielded
  operation tasks so ordering remains intact across instances after caller
  cancellation. Per-stripe admission bounds retained work and rejects overflow
  fail-open without queuing callers; shutdown drains admitted operations without
  clearing persisted entries.
- `__init__.py` — package marker and scope description.

## Key and write semantics

The key is `jasa:search:v2:` plus SHA-256 of compact, sorted-key JSON. Identity
contains the exact query, raw/quality-filter mode, grounded mode, ordered active
provider names, and a grounding-semantics fingerprint when a context exists.
The fingerprint covers prompt/version, model, base URL, content cap, top-N, and
generation constants without including the API key. `include_snippets` and
`timeout_ms` do not affect the key because the full result is cached before
transport formatting.

Values use a schema-v2 envelope containing the exact identity and complete
outcome. Every nested field is strict and extra-forbidden. Legacy,
wrong-version, malformed, wrong-type, unexpected-field, identity-mismatched,
and query-mismatched records are misses. A readable outcome must have zero
failures, successes exactly matching the ordered identity providers, and result
attribution limited to those providers.

`should_cache()` requires at least one success, zero provider failures, and no
transient grounding failures. Preserve this poisoning guard: a partial upstream
outage must not become the response for the configured search TTL.

## Shared runtime backend

`server._build_cache()` delegates memory, disk, and Redis selection to
`omnifetch.cache.build_cache_backend()`. The resulting async adapter dispatches
cachelib's synchronous work off the event loop, fails open, supports real
readiness probes, and is closed exactly once by the parent lifespan. The same
object is `Composition.cache`, `Composition.engine.cache`,
`SearchRuntime.cache`, and every request-scoped `GroundingContext.cache`.

The local `MemoryCache` and `DiskCache` remain import-compatible for callers
that used them directly. Preserve their tests, but do not restore them to
runtime selection.

The search-facing protocol accepts backend-native `object | None` reads and
`bool | None` writes because the runtime cachelib adapter can reject an
operation without raising. Search treats a false write as a fail-open
`write_error`; it never wakes waiters into an assumed hit.

## Tests

`tests/test_cache.py` owns backend/key/gate behavior;
`tests/test_service.py` owns cache orchestration and failure swallowing;
`tests/test_search_coalescing.py` owns concurrent miss behavior. Run:

```bash
conda run -n base uv run pytest tests/test_cache.py tests/test_service.py \
  tests/test_search_coalescing.py
```
