# AGENTS.md — `src/jasa/cache/`

Search caching is intentionally small: a string-store protocol, deterministic
keys, and a completeness gate. Runtime storage uses omnifetch's shared cachelib
adapter, which Jasa configures once and injects into the composed fetch engine.

## Files

- `base.py` — `CacheBackend`, `make_cache_key`, `should_cache`, key prefix, and
  the fixed 129,600-second TTL.
- `memory.py` — legacy-compatible process-local store; no longer runtime-selected.
- `disk.py` — legacy-compatible JSON-file store; no longer runtime-selected.
- `__init__.py` — package marker and scope description.

## Key and write semantics

The key is `search:` plus SHA-256 of the query. Raw mode adds `\0sqf=true`;
grounding adds `\0gnd=true`. `include_snippets` and `timeout_ms` do not affect
the key because the full result is cached before transport formatting.

`should_cache()` requires at least one success, zero provider failures, and no
transient grounding failures. Preserve this poisoning guard: a partial upstream
outage must not become the response for 36 hours.

## Shared runtime backend

`server._build_cache()` delegates memory, disk, and Redis selection to
`omnifetch.cache.build_cache_backend()`. The resulting async adapter dispatches
cachelib's synchronous work off the event loop, fails open, supports real
readiness probes, and is closed exactly once by the parent lifespan. The same
object is `Composition.cache` and `Composition.engine.cache`.

The local `MemoryCache` and `DiskCache` remain import-compatible for callers
that used them directly. Preserve their tests, but do not restore them to
runtime selection.

## Tests

`tests/test_cache.py` owns backend/key/gate behavior;
`tests/test_service.py` owns cache orchestration and failure swallowing. Run:

```bash
conda run -n base uv run pytest tests/test_cache.py tests/test_service.py
```
