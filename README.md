# jasa

A multi-provider web-search MCP server that composes
[`omnifetch`](https://github.com/cjangrist/omnifetch) **in-process**, ported from
the TypeScript/Cloudflare-Workers service
[`omnisearch`](https://github.com/cjangrist/omnisearch) (branch
`remove-answer-and-externalize-fetch`) into Python on FastMCP.

One process exposes two model-visible tools:

- **`web_search`** — parallel fan-out to up to 10 search providers, RRF merge
  with URL dedup, snippet collapse, tail rescue, quality filtering, optional
  grounded snippets, and a 36-hour result cache with a completeness gate.
- **`web_fetch`** — the mounted omnifetch tool: a multi-provider fetch waterfall
  with domain breakers, quality gating, and provider racing.

The grounded-snippet stage and the REST fetch route call the **same omnifetch
engine object directly as a coroutine**. There is no `OMNIFETCH_ENDPOINT`, no
REST client for omnifetch, and no second process.

> **Status:** Initial feature-complete port. Search and fetch providers,
> deterministic fan-out and ranking, grounded snippets, caching, REST routes,
> telemetry, packaging, and the container runtime are implemented and covered
> by the test suite.

## Install

```bash
uv sync --extra telemetry   # telemetry extra is optional
```

`omnifetch` is consumed as an in-process composition target through a PEP 508
Git dependency pinned by full commit SHA in `pyproject.toml`. It is **never**
installed from PyPI — the bare name `omnifetch` on PyPI is unrelated.

## Run

```bash
uv run jasa                                  # stdio (MCP over stdin/stdout)
uv run jasa --transport http --port 8000     # streamable HTTP
docker compose up -d --build --wait          # container from local .env
```

## Configuration

Copy `.env.example` to `.env`. Server settings use the `JASA_` prefix; telemetry
uses the standard `OTEL_` names. Provider secrets keep **provider-native names
with no prefix** — five of them (`TAVILY`, `FIRECRAWL`, `LINKUP`, `YOU`,
`SERPAPI`) enable a provider in **both** the jasa search family and the mounted
omnifetch fetch family. The server starts with zero providers so `/health` can
explain the state; a search call with none configured fails with a specific
configuration error.

Both `uv run jasa` and Docker Compose load the same repository-local `.env`.
Compose also persists the disk cache in the `jasa-cache` named volume. Keep
`.env` local and uncommitted; `.env.example` is the complete secret-free
template.

### Composed mode

jasa constructs omnifetch's configuration from the process environment.
Provider credentials stay live, while standalone `OMNIFETCH_` process settings
are intentionally not exposed because jasa supplies the shared client and
runtime. Jasa always forces the mounted child's REST mirror off: jasa owns the
authenticated fetch surface at `POST /fetch`.

## Health

`GET /health` (and `GET /`) returns an aggregate body — overall status
(`ok` / `degraded` / `unavailable`), version, the search and fetch provider
families as separate lists with counts, whether grounding is enabled, and the
cache backend with readiness. It never calls a paid provider API.

## Test

```bash
uv run pytest                               # unit + coverage (no live APIs)
JASA_RUN_DOCKER_TESTS=1 uv run pytest -m docker_integration --no-cov
uv run ruff format --check                  # formatting
uv run ruff check && uv run mypy            # lint + strict types
uv build                                    # source distribution + wheel
```

## License

MIT.
