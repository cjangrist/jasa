# AGENTS.md — `src/jasa/cache/`

Search caching is intentionally small: a string-store protocol, deterministic
keys, a completeness gate, and memory/disk implementations. Fetch caching is
owned by omnifetch and is not configured through this package in composed mode.

## Files

- `base.py` — `CacheBackend`, `make_cache_key`, `should_cache`, key prefix, and
  the fixed 129,600-second TTL.
- `memory.py` — process-local bounded insertion-order store with monotonic TTL.
- `disk.py` — one JSON file per key, wall-clock expiry, atomic temp-file replace.
- `__init__.py` — package marker and scope description.

## Key and write semantics

The key is `search:` plus SHA-256 of the query. Raw mode adds `\0sqf=true`;
grounding adds `\0gnd=true`. `include_snippets` and `timeout_ms` do not affect
the key because the full result is cached before transport formatting.

`should_cache()` requires at least one success, zero provider failures, and no
transient grounding failures. Preserve this poisoning guard: a partial upstream
outage must not become the response for 36 hours.

## Backend guarantees

- Reads degrade to a miss on corrupt, expired, legacy, or missing data.
- Writes never fail a user request.
- Disk writes fsync a temporary file and atomically replace the destination.
- Failed atomic replacement preserves the old entry and cleans its temp file.
- `MemoryCache` evicts expired entries first, then its oldest live entry.
- Redis is reserved in config but rejected at startup; do not pretend it works.

## Tests

`tests/test_cache.py` owns backend/key/gate behavior;
`tests/test_service.py` owns cache orchestration and failure swallowing. Run:

```bash
conda run -n base uv run pytest tests/test_cache.py tests/test_service.py
```
