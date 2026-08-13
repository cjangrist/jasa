# AGENTS.md — `src/jasa/`

This directory owns the Jasa process: bootstrap, configuration, search,
grounding, caching, REST/MCP surfaces, composition, logging, and telemetry.
Fetch adapters and waterfall execution come from the pinned omnifetch package.

## File map

| File / directory | Responsibility                                                                      |
| ---------------- | ----------------------------------------------------------------------------------- |
| `__init__.py`    | Version and lazy `build_server` export; keep imports light.                         |
| `__main__.py`    | CLI and startup order: dotenv, config, validate, log, uvloop, telemetry, serve.     |
| `config.py`      | Frozen Pydantic settings grouped into `AppConfig`.                                  |
| `auth.py`        | REST API-key precedence and constant-time bearer/query comparison.                  |
| `logging.py`     | Rich stderr logging under the `jasa` namespace.                                     |
| `schemas.py`     | Strict Pydantic MCP input schema for `web_search`.                                  |
| `server.py`      | Parent assembly, child mount, shared client/cache/engine, health, MCP registration. |
| `rest.py`        | `/search`, `/fetch`, `/researcher`, body caps, error mapping, provider resources.   |
| `telemetry.py`   | Lazy opt-in OpenTelemetry setup and shutdown.                                       |
| `cache/`         | Search-cache protocol, keys, write gate, memory and disk stores.                    |
| `grounding/`     | Fetch-to-LLM snippet pipeline, prompt, detectors, outcomes.                         |
| `observability/` | Fail-open metric facade.                                                            |
| `search/`        | Provider adapters, fan-out, retry, ranking, normalization, service.                 |
| `tools/`         | MCP execution/response adapters.                                                    |

## Composition ownership

`build_composition()` is the architectural center. It must maintain:

- one shared `httpx.AsyncClient` with HTTP/2, redirects, and bounded pools;
- one immutable provider-secret snapshot;
- Jasa search adapters loaded in canonical order;
- one memory or disk search cache;
- one omnifetch engine built with the shared client;
- one mounted omnifetch child with `own_engine=False`;
- child REST fetch disabled and `say_hello` hidden by default;
- parent-owned `/`, `/health`, REST routes, MCP resources, and lifespan cleanup.

Do not add a second connection pool, external fetch endpoint, or duplicate fetch
implementation to work around composition issues.

## Public surfaces

### MCP

- `web_search` is registered in `server.py`; input validation comes from
  `schemas.py`, execution from `search/service.py`, response shaping from
  `tools/web_search.py`.
- `web_fetch` is registered by the mounted child and uses the same engine as
  grounding and REST fetch.
- Resources: `jasa://providers/status` and
  `jasa://providers/{provider}/info` are registered in `rest.py`.

### HTTP

- `/` and `/health`: aggregate provider/cache/grounding status.
- `/search`: compact search results, default 20, `raw` quality-filter bypass.
- `/fetch`: full fetch result with status mapping and a 30-second outer timeout.
- `/researcher`: GET/POST snippet response compatible with GPT-Researcher.
- `/mcp/`: FastMCP Streamable HTTP endpoint.

REST auth is open when no configured alias resolves. `JASA_API_KEY` wins over
`OPENWEBUI_API_KEY`, then `OMNISEARCH_API_KEY`.

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
- `SearchError(kind=no_providers|all_failed)` is mapped by REST.
- FastMCP masks unhandled error detail; deliberate user errors should be clear
  before reaching that boundary.
- Cache, telemetry shutdown, and metrics fail open; provider execution does not.

## Focused tests

| Change                    | Test files                                                      |
| ------------------------- | --------------------------------------------------------------- |
| Bootstrap/config          | `test_bootstrap.py`, `test_config.py`                           |
| Composition/lifecycle     | `test_composition.py`, `test_server.py`                         |
| REST/auth/resources       | `test_rest.py`                                                  |
| MCP schemas/format        | `test_schemas.py`, `test_web_search.py`                         |
| Logging/telemetry/metrics | `test_logging.py`, `test_telemetry.py`, `test_observability.py` |
| Packaging                 | `test_package.py`                                               |

After focused tests, run the entire suite because composition and environment
isolation are cross-cutting.
