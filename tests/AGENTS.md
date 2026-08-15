# AGENTS.md — `tests/`

The normal suite is hermetic: no live provider APIs, no required local `.env`,
and no Docker unless explicitly enabled. Coverage must be 100% for both lines
and branches.

## Test map

| Area                     | Files                                                                      |
| ------------------------ | -------------------------------------------------------------------------- |
| Bootstrap/config/package | `test_bootstrap.py`, `test_config.py`, `test_package.py`                   |
| Composition/server/REST  | `test_composition.py`, `test_server.py`, `test_rest.py`, `test_schemas.py` |
| Search orchestration     | `test_fanout.py`, `test_retry.py`, `test_service.py`, `test_search_coalescing.py`, `test_web_search.py` |
| Search algorithms        | `test_operators.py`, `test_ranking.py`, `test_snippets.py`, `test_urls.py` |
| Providers                | `test_provider_*.py`, `test_providers.py`                                  |
| Grounding/cache          | `test_grounding.py`, `test_grounding_service.py`, `test_cache.py`          |
| Operations               | `test_logging.py`, `test_telemetry.py`, `test_observability.py`            |
| Container                | `test_docker_integration.py` (opt-in marker)                               |
| Source parity            | `fixtures/golden/`                                                         |

## Environment isolation

`conftest.py` automatically removes every `JASA_`, `OMNIFETCH_`, and `OTEL_`
setting plus the union of search/fetch/grounding/auth secret names before each
test. This prevents a developer's real `.env` or shell from activating paid
providers. `JASA_RUN_DOCKER_TESTS` is the only retained test-control flag.

When adding a secret, update the registry source and parity test so isolation
cannot drift. Never weaken the fixture to make an environment-dependent test
pass.

## Testing patterns

- Use `respx` to assert exact outbound HTTP requests and return controlled
  provider responses.
- Use injected clocks, RNGs, sleeps, fake providers, and in-memory FastMCP
  clients for deterministic async behavior.
- Assert error type, message, provider attribution, and credential redaction.
- Put source-port parity cases in JSON goldens; keep test logic generic over the
  cases.
- Use `TestClient` for HTTP routes and `fastmcp.Client` for MCP behavior.
- Close explicitly created clients or exercise the composition lifespan.

## Commands

```bash
conda run -n base uv run pytest
conda run -n base uv run pytest tests/test_service.py -q
JASA_RUN_DOCKER_TESTS=1 conda run -n base uv run pytest \
  -m docker_integration --no-cov
```

Do not add paid-provider tests to normal pytest or a GitHub Actions workflow.
Those belong in the manual `scripts/run_provider_integration.py` path.
