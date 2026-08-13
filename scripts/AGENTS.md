# AGENTS.md — `scripts/`

This directory contains manual operational harnesses, not unit-test helpers.
The only script, `run_provider_integration.py`, isolates one provider inside the
Compose service and makes exactly one paid REST or MCP request.

## `run_provider_integration.py`

Execution flow:

1. Parse an explicit family, provider, and surface.
2. Validate the case against `INTEGRATION_CASES`.
3. Read only the selected provider's credential names from local `.env`.
4. Remove every other known secret from the child environment.
5. Pass selected values to Compose through an in-memory file descriptor.
6. Force-recreate the service and wait for health.
7. Confirm `/health` shows only the expected provider(s) in that family.
8. Make one paid request and validate provider attribution.

`--invalid-credential` substitutes a fixed non-secret value to test error
reporting. `--no-build` reuses `jasa:local` after the first run. Fetch providers
with shared secrets may activate multiple fetch adapters; `expected_active_names`
accounts for that and the request skips all except the selected target.

## Supported cases

The authoritative list is `INTEGRATION_CASES`. Search currently covers Exa,
Kagi, Linkup, Parallel, Perplexity, Serper, Tavily, and You.com. Fetch currently
covers Diffbot, GitHub, Jina, Linkup, Oxylabs, ScrapeGraphAI, Scrapeless,
ScrapingAnt, ScrapingBee, SociaVault, Tavily, and You.com.

## Commands

```bash
conda run -n base uv run python scripts/run_provider_integration.py \
  --family search --provider tavily --surface rest

conda run -n base uv run python scripts/run_provider_integration.py \
  --family fetch --provider jina --surface mcp --no-build
```

## Change rules

- Keep one paid request per process invocation.
- Keep secrets out of argv, logs, temporary disk files, and Git.
- Preserve provider isolation verification before the paid call.
- Preserve REST and MCP attribution checks.
- A provider case belongs in `INTEGRATION_CASES` only after a real run works.
- The POSIX file-descriptor transport is intentional. If adding Windows/macOS
  support, solve secret transport at the root rather than weakening isolation.

## Tests

Unit coverage for helper behavior belongs in existing tests if script logic is
changed. Then run the script with its prior successful arguments, followed by
`conda run -n base uv run pytest` and pre-commit.
