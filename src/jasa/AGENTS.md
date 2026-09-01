# AGENTS.md — `src/jasa/`

This directory owns the Jasa process: bootstrap, configuration, search,
grounding, caching, REST/MCP surfaces, composition, logging, and telemetry.
Fetch adapters and waterfall execution come from the locked omnifetch Git
source. Its declaration tracks GitHub `main`; `uv.lock` freezes the commit.

## File map

| File / directory | Responsibility                                                                      |
| ---------------- | ----------------------------------------------------------------------------------- |
| `__init__.py`    | Version and lazy `build_server` export; keep imports light.                         |
| `__main__.py`    | CLI and startup order: dotenv, config, validate, log, uvloop, telemetry, serve.     |
| `config.py`      | Frozen Pydantic settings grouped into `AppConfig`.                                  |
| `assets.py`      | Packaged icons and the `serverInfo.icons` declaration built from them.               |
| `auth.py`        | REST API-key precedence and constant-time bearer/query comparison.                  |
| `logging.py`     | Rich stderr logging under the `jasa` namespace.                                     |
| `schemas.py`     | Strict Pydantic MCP input and output schemas for `web_search`.                      |
| `server.py`      | Parent assembly, child mount, shared client/cache/engine, health, MCP registration. |
| `rest.py`        | `/search`, `/fetch`, `/usage`, `/researcher`, body caps and error mapping.           |
| `telemetry.py`   | Lazy opt-in OpenTelemetry setup and shutdown.                                       |
| `assets/`        | Square PNGs and a favicon, shipped in the wheel and the image.                       |
| `cache/`         | Search keys/gate and compatibility stores; server selects cachelib/Redis.           |
| `grounding/`     | Fetch-to-LLM snippet pipeline, prompt, detectors, outcomes.                         |
| `observability/` | Fail-open metric facade.                                                            |
| `search/`        | Provider adapters, fan-out, retry, ranking, normalization, service.                 |
| `tools/`         | MCP execution/response adapters.                                                    |
| `usage/`         | Provider-native quota probes, redaction, shared cache, refresh middleware.          |

## Composition ownership

`build_composition_async()` is the awaitable architectural center;
`build_composition()` is its synchronous, pre-event-loop wrapper. They must
maintain:

- one shared `httpx.AsyncClient` with HTTP/2, redirects, and bounded pools;
- one immutable provider-secret snapshot;
- Jasa search adapters loaded in canonical order;
- one cachelib memory, filesystem, or Redis backend shared by both families;
- request-scoped grounding contexts borrowing that same backend and the
  configured successful-grounding TTL plus one registration-owned cache-write
  semaphore and grounding flight registry shared across requests;
- one `SearchRuntime` sharing the provider map, cache, configured search TTL,
  and process-local miss-flight registry across MCP and REST;
- one `UsageRuntime` borrowing the same client, cache, and secret snapshot,
  with a process-local refresh task shared by REST and both MCP tools;
- one omnifetch engine built with the shared client and shared cache;
- one mounted omnifetch child with `own_engine=False`;
- child REST fetch disabled and `say_hello` hidden by default;
- parent-owned `/`, `/health`, REST routes, MCP resources, cache readiness, and
  lifespan cleanup.

Do not add a second connection pool, external fetch endpoint, or duplicate fetch
implementation to work around composition issues.

## Public surfaces

### MCP

- `web_search` is registered in `server.py`; input validation comes from
  `schemas.py`, execution from `search/service.py`, response shaping from
  `tools/web_search.py`, and MCP safety hints mark it read-only,
  non-destructive, idempotent, and open-world. Its injected FastMCP `Context`
  feeds best-effort protocol progress into the shared search service without
  adding a public tool argument. Its Pydantic return annotation publishes a
  strict, dereferenced `outputSchema` and returns the same shape through MCP
  `structuredContent`.
- `web_fetch` is registered by the mounted child and uses the same engine as
  grounding and REST fetch.
- Resources: `jasa://providers/status` and
  `jasa://providers/{provider}/info` are registered in `rest.py`.

### HTTP

- `/` and `/health`: aggregate provider/cache/grounding status.
- `/icon.png`, `/favicon.png`, `/favicon.ico`: packaged brand assets. `?size=`
  selects a declared square; anything else falls back to the largest, so a
  stale link resolves to an image rather than an error.
- `/search`: compact search results, default 20, `raw` quality-filter bypass.
- `/fetch`: full fetch result with status mapping and a 30-second outer timeout.
- `/usage`: cleaned provider-native quota snapshots with a 30-second timeout;
  returns 503 if shutdown has already closed the usage runtime.
- `/researcher`: GET/POST snippet response compatible with GPT-Researcher.
- `/mcp`: FastMCP Streamable HTTP endpoint. HTTP startup adds standard CORS
  preflight handling and exposes `Mcp-Session-Id` for browser MCP clients.

REST auth is open when no configured alias resolves. `JASA_API_KEY` wins over
`OPENWEBUI_API_KEY`, then `OMNISEARCH_API_KEY`. The shared guard accepts either
`Authorization: Bearer ...` or `?key=...` on all four routes; bearer auth is
preferred because URLs are frequently logged.

## Configuration checklist

When adding a setting:

1. Add an explicit `validation_alias` in the owning settings class.
2. Add its secret-free default/name to `.env.example`.
3. Update `tests/test_config.py`; exact set equality must pass.
4. If Compose consumes it directly, add the substitution or env-file behavior
   and include it in the README table.
5. Update this guide or the nearest subsystem guide.

Provider secrets intentionally do not become Pydantic settings. Search and
fetch registries read the same immutable `ProviderSecrets` snapshot.

## Error boundaries

- Provider adapters raise omnifetch's shared `ProviderError` taxonomy.
- Fan-out captures provider errors and unexpected exceptions independently.
- `SearchError(kind=no_providers|all_failed|deadline_exceeded)` maps to REST
  503, 502, and 504 respectively.
- FastMCP masks unhandled error detail; deliberate user errors should be clear
  before reaching that boundary.
- Cache, telemetry shutdown, and metrics fail open; provider execution does not.
- Grounding-cache reads and writes fail open; only accepted LLM output is stored,
  and a write deadline never downgrades that paid success to a fallback.
- An expired grounding budget cancels only the URLs still in flight. Every URL
  that already produced a snippet keeps it; the rest are reported through
  `snippet_source: "fallback"` and the response's `grounding.outcomes`.
- Grounding waiters release the fetch/LLM worker slot, retain their own absolute
  per-URL deadline, reread cache after the leader write, and retry independently
  after every non-cacheable leader outcome.
- Search waiters reread cache after a flight. They never receive a leader result
  directly, so partial, failed, and cache-write-rejected outcomes cannot leak as
  synthetic hits.

## Focused tests

| Change                    | Test files                                                      |
| ------------------------- | --------------------------------------------------------------- |
| Bootstrap/config          | `test_bootstrap.py`, `test_config.py`                           |
| Composition/lifecycle     | `test_composition.py`, `test_server.py`                         |
| REST/auth/resources       | `test_rest.py`                                                  |
| Usage/quota snapshots     | `test_usage.py`, `test_usage_cache.py`                          |
| MCP schemas/format        | `test_schemas.py`, `test_web_search.py`                         |
| Logging/telemetry/metrics | `test_logging.py`, `test_telemetry.py`, `test_observability.py` |
| Packaging                 | `test_package.py`                                               |

After focused tests, run the entire suite because composition and environment
isolation are cross-cutting.
