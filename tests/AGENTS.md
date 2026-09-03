# AGENTS.md — `tests/`

The normal suite is hermetic: no live provider APIs, no required local `.env`,
and no Docker unless explicitly enabled. Coverage must be 100% for both lines
and branches.

## Test map

| Area                     | Files                                                                      |
| ------------------------ | -------------------------------------------------------------------------- |
| Bootstrap/config/package | `test_bootstrap.py`, `test_config.py`, `test_package.py`                   |
| Composition/server/REST  | `test_composition.py`, `test_server.py`, `test_rest.py`, `test_searxng.py`, `test_schemas.py`, `test_usage.py` |
| Usage cache/runtime      | `test_usage_cache.py`; shared fixtures in `usage_helpers.py`                          |
| Search orchestration     | `test_fanout.py`, `test_retry.py`, `test_service.py`, `test_search_coalescing.py`, `test_web_search.py` |
| Search algorithms        | `test_operators.py`, `test_ranking.py`, `test_snippets.py`, `test_urls.py` |
| Providers                | `test_provider_*.py`, `test_providers.py`                                  |
| Grounding/cache          | `test_grounding.py`, `test_grounding_service.py`, `test_grounding_coalescing.py`, `test_grounding_flight_failures.py`, `test_grounding_flight_deadlines.py`, `test_grounding_waterfall.py`, `test_cache.py`, `test_cache_surface_matrix.py` |
| Brand assets/icon        | `test_assets.py`                                                           |
| Operations               | `test_logging.py`, `test_telemetry.py`, `test_observability.py`            |
| Container/real Redis     | `test_docker_integration.py` (opt-in marker)                               |
| Source parity            | `fixtures/golden/`                                                         |

## Environment isolation

`conftest.py` automatically removes every `JASA_`, `OMNIFETCH_`, and `OTEL_`
setting, the union of search/fetch/grounding/auth secret names, and the search
adapters' declared `setting_envs` before each test. This prevents a developer's
real `.env` or shell from activating paid providers or retargeting an adapter
at a local gateway. `JASA_RUN_DOCKER_TESTS` is the only retained test-control
flag.

When adding a secret or an adapter setting, update the registry source and
parity test so isolation cannot drift. Never weaken the fixture to make an
environment-dependent test pass.

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
- Derive an expectation from the constant that owns it rather than restating it
  as a literal. A literal that silently stops exercising its case is worse than
  no test, because it keeps reporting success.
- Assert something that can actually fail. Searching a log record's message for
  a traceback, or a task's `repr()` for a function name, guards nothing.
- A patched `_call_grounding_tier` must return `tests.conftest.tier_answer(...)`;
  the waterfall reads a stop reason alongside the text, so a bare string no
  longer satisfies the contract.
- Scale a timing constant down with `monkeypatch` instead of waiting out a real
  budget. Tests that consume their whole deadline leave no slack and flake.

A green suite is necessary and not sufficient. Mocked providers cannot show a
stage that runs out of budget, a page that never returns, a tier that answers
with an empty body, or a generation cut off at its token ceiling -- each of
those has shipped past a fully passing run. The root `AGENTS.md` requires live
Docker verification with 2 varied queries before every PR -- except for a
fetch-only change, which takes the single-credit `web_fetch` path described
there rather than paying for two search fan-outs.

## Commands

```bash
conda run -n base uv run pytest
conda run -n base uv run pytest tests/test_service.py -q
JASA_RUN_DOCKER_TESTS=1 conda run -n base uv run pytest \
  -m docker_integration --no-cov
```

Do not add paid-provider tests to normal pytest or a GitHub Actions workflow.
Those belong in the manual `scripts/run_provider_integration.py` path.
