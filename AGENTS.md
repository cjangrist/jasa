# AGENTS.md — repository navigation hub

Jasa is a Python FastMCP server that owns multi-provider search and composes the
separately versioned `omnifetch` fetch engine in-process. This is the first file
to read before changing the repository. Each child directory has a narrower
`AGENTS.md`; follow the nearest guide after locating the owning subsystem.

## Fast orientation

| Question                             | Start here                              | Then read                                 |
| ------------------------------------ | --------------------------------------- | ----------------------------------------- |
| How is the process assembled?        | `src/jasa/server.py`                    | `src/jasa/AGENTS.md`                      |
| How does a search execute?           | `src/jasa/search/service.py`            | `src/jasa/search/AGENTS.md`               |
| How is a provider added?             | `src/jasa/search/providers/__init__.py` | `src/jasa/search/providers/AGENTS.md`     |
| Where are ranking and dedup defined? | `src/jasa/search/ranking.py`            | `urls.py`, `snippets.py`                  |
| Where is fetch implemented?          | pinned `omnifetch` dependency           | composition notes in `src/jasa/AGENTS.md` |
| How are snippets grounded?           | `src/jasa/grounding/service.py`         | `src/jasa/grounding/AGENTS.md`            |
| Which HTTP routes exist?             | `src/jasa/rest.py`                      | `src/jasa/server.py`                      |
| Which MCP tools exist?               | `src/jasa/server.py`                    | `src/jasa/tools/AGENTS.md`                |
| How is configuration loaded?         | `src/jasa/config.py`                    | `.env.example`, `tests/test_config.py`    |
| How are images/releases published?   | `.github/workflows/`                    | `.github/workflows/AGENTS.md`             |
| How are live providers tested?       | `scripts/run_provider_integration.py`   | `scripts/AGENTS.md`                       |
| Which test owns a behavior?          | `tests/AGENTS.md`                       | nearest `test_*.py`                       |

## Runtime in one minute

`jasa.__main__.main()` loads `.env`, resolves immutable settings, validates
startup, configures stderr logging, installs uvloop when available, enables
optional telemetry, and calls `build_server()`.

`build_composition()` creates exactly one `httpx.AsyncClient`, snapshots provider
secrets, constructs Jasa's search registry, selects the search cache, builds an
omnifetch `Engine` with the same client, and mounts the omnifetch FastMCP child.
Jasa registers `web_search`; the child supplies `web_fetch`. The child does not
own the shared client and its standalone `/web_fetch` route is disabled. The
parent lifespan closes cache, client, and telemetry.

Search path:

```text
MCP/REST -> run_search -> cache read -> parallel provider dispatch
         -> selective retry -> deterministic RRF/dedup/snippet collapse
         -> optional fetch+Cerebras grounding -> complete-result cache write
         -> transport-specific formatting
```

Fetch path:

```text
MCP/REST/grounding -> omnifetch.execute_web_fetch -> domain breakers
                   -> tiered provider waterfall/races -> quality gate
                   -> attributed result/failures
```

## Repository layout

| Path                 | Ownership                                                            |
| -------------------- | -------------------------------------------------------------------- |
| `README.md`          | User-facing product, setup, API, operations, and contributor guide.  |
| `.env.example`       | Exact tested set of supported runtime/env-file names; never secrets. |
| `pyproject.toml`     | Package metadata, exact dependency pins, tools, coverage gate.       |
| `uv.lock`            | Reproducible resolution. Change only through `uv lock`/`uv sync`.    |
| `Dockerfile`         | Multi-stage, non-root production image.                              |
| `docker-compose.yml` | Local deployment and persistent disk cache.                          |
| `.github/`           | CI, release, and multi-platform image automation.                    |
| `scripts/`           | Manual one-provider paid integration harness.                        |
| `src/jasa/`          | Application code. See `src/jasa/AGENTS.md`.                          |
| `tests/`             | Unit, parity, transport, packaging, and opt-in Docker tests.         |
| `tmp/`               | Ignored temporary/reference/review material. Never commit.           |
| `trash/`             | Ignored recoverable deletions. Never commit.                         |

## Load-bearing invariants

- One process, one shared async HTTP client, one omnifetch engine. Do not add an
  internal REST call or `OMNIFETCH_ENDPOINT`.
- Search providers are Jasa-owned; fetch providers and waterfall logic are
  omnifetch-owned. Fix fetch behavior upstream and update the full-SHA pin.
- Provider-native secret names have no `JASA_` prefix. Shared names can enable
  both search and fetch adapters.
- `.env.example` must exactly equal all supported settings, auth aliases,
  Compose substitutions, search secrets, and omnifetch fetch secrets. The
  equality test is `tests/test_config.py`.
- A populated `.env` is local-only. Never print or commit secret values.
- stdout belongs to MCP stdio JSON-RPC; application logs go to stderr.
- Search aggregation is deterministic in registry order even when providers
  finish out of order.
- Only complete, non-transient search outcomes are cached for 36 hours.
- `web_search` returns top 30 plus tail rescues. REST `/search` defaults to 20;
  `/researcher` returns 10.
- Omnifetch's `say_hello` is disabled unless `JASA_EXPOSE_HELLO=true`.
- Parent `/health` wins; child standalone `/web_fetch` is always off in composed
  mode.
- Unit tests require 100% line and branch coverage. Live provider calls are
  manual and never part of normal pytest/CI.

## Change routing

### Add or modify a search provider

1. Read every file in `src/jasa/search/providers/` and its `AGENTS.md`.
2. Implement the adapter using the shared HTTP/error base.
3. Add it to `PROVIDER_CLASSES` in canonical order and its secret to
   `KNOWN_SEARCH_SECRET_ENVS`.
4. Add focused request/mapping/error/redaction tests.
5. Add the secret-free name to `.env.example`; the parity test must remain an
   exact set equality.
6. Add a manual integration case only after a real REST or MCP run succeeds.
7. Update the provider tables in `README.md` and the local agents guide.

### Modify search semantics

- Fan-out, deadlines, cancellation: `fanout.py` and `test_fanout.py`.
- Retry classification/backoff: `retry.py` and `test_retry.py`.
- Operators: `operators.py` plus `fixtures/golden/operators.json`.
- URL equivalence: `urls.py` plus `fixtures/golden/url_normalization.json`.
- RRF, quality gate, tail rescue: `ranking.py` plus ranking/truncation goldens.
- Snippet selection: `snippets.py` plus `fixtures/golden/snippets.json`.
- Cache/read/write orchestration: `service.py` and `test_service.py`.

### Modify a public surface

- MCP input model: `src/jasa/schemas.py`.
- MCP registration and composition: `src/jasa/server.py`.
- MCP response formatting: `src/jasa/tools/web_search.py`.
- REST validation/status/shape: `src/jasa/rest.py`.
- Auth: `src/jasa/auth.py`.
- Update README contract examples and test both MCP and REST paths.

### Modify fetch behavior

Do not copy or patch omnifetch internals in Jasa. Work in the omnifetch
repository, test/release it, update the full commit SHA in `pyproject.toml`, run
`uv lock`, then run Jasa's entire suite and Docker integration. Changes limited
to how Jasa composes the child belong in `src/jasa/server.py`.

## Required local checks

Use Anaconda base for Python/uv commands:

```bash
conda run -n base uv sync --frozen --extra telemetry
conda run -n base uv run pytest
conda run -n base uv run ruff format --check
conda run -n base uv run ruff check
conda run -n base uv run mypy
conda run -n base uv run pre-commit run --all-files
conda run -n base uv build
```

For Docker/runtime changes:

```bash
JASA_RUN_DOCKER_TESTS=1 conda run -n base uv run pytest \
  -m docker_integration --no-cov
```

For workflow changes, run `actionlint`. For Markdown, verify internal links,
code fences, documented paths, and that every tracked directory retains an
`AGENTS.md`.

## Documentation maintenance

- README is for users/operators and should explain value before internals.
- AGENTS files are precise navigation indexes, not general prose. Keep paths,
  constants, provider counts, and commands synchronized with code.
- When a file moves or a behavior changes, update the nearest `AGENTS.md` in the
  same commit.
- Do not create speculative roadmap docs. Document current code and explicitly
  label reserved/unimplemented settings.
- Generated caches, `.venv`, `.git`, `dist`, `tmp`, and `trash` are not project
  directories and do not receive AGENTS files.
