# Jasa

[![Quality](https://github.com/cjangrist/jasa/actions/workflows/quality.yml/badge.svg)](https://github.com/cjangrist/jasa/actions/workflows/quality.yml)
[![Unit tests](https://github.com/cjangrist/jasa/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/cjangrist/jasa/actions/workflows/unit-tests.yml)
[![Docker tests](https://github.com/cjangrist/jasa/actions/workflows/docker-tests.yml/badge.svg)](https://github.com/cjangrist/jasa/actions/workflows/docker-tests.yml)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python)](https://www.python.org/)
[![MCP](https://img.shields.io/badge/MCP-FastMCP-green)](https://gofastmcp.com/)
[![Container](https://img.shields.io/badge/GHCR-multi--arch-blue?logo=docker)](https://github.com/users/cjangrist/packages/container/package/jasa)
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

> One MCP server for resilient web research: multi-engine search, deterministic
> ranking, source-backed snippets, and a deep URL-fetch waterfall.

The name is literal: Jasa is just another search aggregator—one built to make
several providers behave like a dependable whole.

Jasa gives agents two dependable primitives:

- `web_search` asks every configured search provider in parallel, merges their
  blind spots with Reciprocal Rank Fusion, deduplicates URLs, consolidates
  snippets, and keeps the top 30 high-signal results plus eligible tail rescues.
- `web_fetch` turns a public URL into clean content through the in-process
  [omnifetch](https://github.com/cjangrist/omnifetch) waterfall, including
  domain-aware routes for GitHub, YouTube, and social media.

The result is a single Python process, one shared HTTP connection pool, one
shared cache backend, one configuration surface, and no internal network hop
between search and fetch.
Use it over MCP, call the REST compatibility routes, run it locally, or pull
the AMD64/ARM64 container.

## Contents

- [Why Jasa](#why-jasa)
- [Architecture](#architecture)
- [Quick start](#quick-start)
- [Connect an MCP client](#connect-an-mcp-client)
- [REST API](#rest-api)
- [Configuration](#configuration)
- [Ranking, caching, and failure semantics](#ranking-caching-and-failure-semantics)
- [Health and operations](#health-and-operations)
- [Repository map](#repository-map)
- [Development](#development)
- [Security notes](#security-notes)
- [Releases](#releases)

## Why Jasa

| Concern          | Single-provider integration               | Jasa                                                                                       |
| ---------------- | ----------------------------------------- | ------------------------------------------------------------------------------------------ |
| Search coverage  | One index and one ranking model           | Up to 11 search engines fan out concurrently                                               |
| Result quality   | Provider-native order and duplicate links | Deterministic RRF, URL normalization, snippet collapse, quality filtering, and tail rescue |
| Snippet trust    | Search-engine excerpts                    | Optional snippets regenerated from fetched page content                                    |
| URL extraction   | One scraper succeeds or the request fails | 27 fetch adapters behind domain breakers and a tiered waterfall                            |
| Failure behavior | One outage breaks the tool                | Per-provider isolation, selective retry, and partial-result reporting                      |
| Latency and cost | Repeat every upstream call                | Success-only search, fetch, and grounding caches                                            |
| Integration      | Separate search and fetch services        | One FastMCP server and one shared `httpx` client                                           |
| Operations       | Provider calls needed to inspect state    | Free `/health` probe with active providers and cache readiness                             |

Jasa is designed for agentic research rather than a human-facing search page.
Its outputs preserve provider attribution, failures, timings, ranking scores,
and truncation metadata so an agent can decide whether it has enough evidence
or should search/fetch again.

## Architecture

```text
MCP client                         REST client
    |                                  |
    | /mcp/                            | /search, /fetch, /researcher
    +----------------+-----------------+
                     |
              FastMCP parent server
                     |
        +------------+-------------+
        |                          |
   web_search                  web_fetch
        |                          |
 search provider fan-out       mounted omnifetch
        |                      fetch waterfall
 retry + deadline                  |
        |                    domain breakers +
 RRF + URL dedup              provider racing
        |
 snippet collapse + quality filter
        |
 optional fetch -> per-key flight -> grounding cache -> Cerebras on miss
        |
 complete-result cache
```

Search, fetch, and grounding share one cachelib memory, filesystem, or Redis
backend.

The parent server mounts omnifetch directly. The grounded-snippet stage and
`POST /fetch` invoke the same engine object as `web_fetch`; they do not call a
second service. Omnifetch's standalone runtime settings are intentionally
ignored in composed mode, and its standalone REST mirror is disabled because
Jasa owns the authenticated REST surface.

## Quick start

### Run from source

Requirements: Linux or macOS, Python 3.11+, Git, and
[`uv`](https://docs.astral.sh/uv/). Windows users should run the published
container or use WSL; the source dependency set includes uvloop, which does not
publish native Windows wheels.

```bash
git clone https://github.com/cjangrist/jasa.git
cd jasa
uv sync --extra telemetry
cp .env.example .env
# Add at least one search or fetch provider key to .env.
uv run jasa --transport http --host 127.0.0.1 --port 8000
```

Verify the process without spending provider credits:

```bash
curl -fsS http://127.0.0.1:8000/health
```

The default command, `uv run jasa`, uses MCP over stdio. HTTP mode exposes MCP
at `http://127.0.0.1:8000/mcp/` and the REST routes described below.

### Run with Docker Compose

```bash
cp .env.example .env
# Configure provider keys in .env.
docker compose up -d --build --wait
curl -fsS http://127.0.0.1:8000/health
```

Compose reads the same `.env` as the local process. Select the `disk` backend to
persist entries in the `jasa-cache` volume. Override the host binding with
`JASA_DOCKER_HOST`/`JASA_DOCKER_PORT`, or point Compose at another local env
file with `JASA_ENV_FILE`.

Upgrading an existing Compose deployment: Compose now follows
`JASA_CACHE_BACKEND` from `.env` instead of forcing filesystem storage. Set
`JASA_CACHE_BACKEND=disk` to retain the persistent cache behavior and continue
using the mounted `jasa-cache` volume.

### Run the published image

```bash
docker run --rm -p 8000:8000 --env-file .env ghcr.io/cjangrist/jasa:latest
```

Published tags include:

- `latest` for the newest successful main or stable-tag build;
- `sha-<full-commit>` for immutable main builds;
- `0`, `0.1`, and `0.1.0`-style tags for stable releases.

Every published manifest supports `linux/amd64` and `linux/arm64`, including
Apple Silicon Macs running Docker Desktop.

## Connect an MCP client

For a local stdio client:

```json
{
  "mcpServers": {
    "jasa": {
      "command": "uv",
      "args": ["--directory", "/absolute/path/to/jasa", "run", "jasa"]
    }
  }
}
```

For a running HTTP server:

```json
{
  "mcpServers": {
    "jasa": {
      "url": "http://127.0.0.1:8000/mcp/"
    }
  }
}
```

### MCP tool: `web_search`

| Input               | Type                 | Default   | Meaning                                                                |
| ------------------- | -------------------- | --------- | ---------------------------------------------------------------------- |
| `query`             | string, 1-2000 chars | required  | Search query; advanced operators are supported by compatible providers |
| `timeout_ms`        | positive integer     | 30000     | Global cache, fan-out, coalescing, and grounding budget                 |
| `include_snippets`  | boolean              | `true`    | Include consolidated snippets in each result                           |
| `grounded_snippets` | boolean or null      | automatic | Regenerate top-result snippets when a grounding key is configured      |

The response contains `providers_succeeded`, `providers_failed`, total timing,
truncation counts, and `web_results`. Each result carries its contributing
providers and RRF score. MCP keeps the top 30 results plus eligible tail
rescues from previously unseen hosts.

```json
{
  "query": "python structured concurrency",
  "providers_succeeded": [
    { "provider": "brave", "duration_ms": 412 },
    { "provider": "tavily", "duration_ms": 683 }
  ],
  "providers_failed": [],
  "total_duration_ms": 697,
  "truncation": { "total_before": 34, "kept": 31, "rescued": 1 },
  "web_results": [
    {
      "title": "...",
      "url": "https://example.com/article",
      "snippets": ["..."],
      "source_providers": ["tavily", "brave"],
      "score": 0.0325
    }
  ]
}
```

### MCP tool: `web_fetch`

| Input            | Type                   | Default  | Meaning                                                                |
| ---------------- | ---------------------- | -------- | ---------------------------------------------------------------------- |
| `url`            | string, 1-2000 chars   | required | Public URL to extract                                                  |
| `skip_providers` | string or string array | empty    | Skip known-bad providers and request a comparison result when possible |

The response includes clean content, the winning provider, every provider
attempted, attributed failures, total duration, and optional alternative
results. If the content is incomplete or wrong, repeat the URL with the prior
`source_provider` in `skip_providers`.

## REST API

HTTP mode offers thin compatibility routes over the same execution paths.

| Route          | Method              | Purpose                                               | Default result count |
| -------------- | ------------------- | ----------------------------------------------------- | -------------------- |
| `/health`, `/` | `GET`               | Free liveness/readiness and active-provider inventory | n/a                  |
| `/search`      | `POST`              | Search results shaped as `{link,title,snippet}`       | 20; `0` returns all  |
| `/fetch`       | `POST`              | Full omnifetch result                                 | one primary result   |
| `/usage`       | `GET`               | Cached provider-native usage and quota snapshots      | n/a                  |
| `/researcher`  | `GET`, `POST`       | GPT-Researcher-compatible `{href,body}` snippets      | 10                   |
| `/mcp/`        | MCP Streamable HTTP | FastMCP tools and resources                           | tool-specific        |

Search:

```bash
curl -fsS http://127.0.0.1:8000/search \
  -H 'content-type: application/json' \
  -H "authorization: Bearer $JASA_API_KEY" \
  -d '{"query":"site:docs.python.org asyncio TaskGroup","count":10}'
```

Set `raw: true` in the request body to bypass the search quality filter.
`count` is clamped to 0-100. REST searches have a 30-second fan-out deadline.

Fetch:

```bash
curl -fsS http://127.0.0.1:8000/fetch \
  -H 'content-type: application/json' \
  -H "authorization: Bearer $JASA_API_KEY" \
  -d '{"url":"https://example.com","skip_providers":["tavily"]}'
```

Researcher compatibility:

```bash
curl -fsS 'http://127.0.0.1:8000/researcher?query=python+free+threading' \
  -H "authorization: Bearer $JASA_API_KEY"
```

Usage and quota status:

```bash
curl -fsS http://127.0.0.1:8000/usage \
  -H "authorization: Bearer $JASA_API_KEY"
```

`/usage` enumerates every registered search and fetch provider. Tavily,
Firecrawl, GitHub, ScrapingAnt, ScrapingBee, SerpAPI, Serper, Diffbot, Kimi,
Linkup, You.com, Olostep, ScrapeGraphAI, Scrapeless, Scrapfly, and Scrappey are
the currently integrated usage sources, along with SociaVault and Spider;
configured providers without an integration remain explicit as
`not_implemented`.
Successful responses keep the provider's native JSON fields under `raw`, with
credentials and account identities recursively redacted. Provider failures are
isolated and reported beside the other records. GitHub uses its unmetered
authenticated rate-limit endpoint; ScrapingAnt, ScrapingBee, and Kimi use their
free usage endpoints; SerpAPI, Serper, Diffbot, Linkup, You.com, Olostep,
ScrapeGraphAI, and Scrapfly use their free account, balance, or credit endpoints.
Scrapeless uses its free user-info endpoint for account balance and subscription
plan data.
Scrappey uses its free remaining-balance endpoint for the fetch provider's
request balance.
SociaVault uses its free credits endpoint for balance and subscription status.
Spider uses its free credits endpoint for balance and refill/payment state.

The endpoint reuses the shared memory, filesystem, or Redis cache for 10
minutes by default. A cache miss coalesces into one refresh for `/usage` callers.
Normal REST and MCP search/fetch requests only trigger a refresh check in the
background, so quota APIs never add latency to their execution paths. The full
`/usage` cache-read, refresh, and cache-write path has a 30-second deadline.

The header may be omitted when no authentication alias is configured.

REST bodies are capped at 64 KiB. Queries and URLs are capped at 2000
characters. Error status codes distinguish invalid input (`400`), auth
failure (`401`), body size (`413`), rate limiting (`429`), not found (`404`),
upstream exhaustion (`502`), unavailable usage or no configured search provider
(`503`), and usage, search, or fetch deadline expiry (`504`).

## Configuration

Copy `.env.example` to `.env`. It is the tested, complete, secret-free runtime
contract: variables consumed by Jasa, Docker Compose, and every registered
search or fetch provider. A real `.env` is local-only and ignored by Git.

### Server and storage

| Variable                             | Default        | Description                                                   |
| ------------------------------------ | -------------- | ------------------------------------------------------------- |
| `JASA_TRANSPORT`                     | `stdio`        | `stdio`, `http`, or `sse`                                     |
| `JASA_HOST`                          | `127.0.0.1`    | Bind address for HTTP/SSE                                     |
| `JASA_PORT`                          | `8000`         | Bind port                                                     |
| `JASA_LOG_LEVEL`                     | `INFO`         | Package log level                                             |
| `JASA_UVLOOP`                        | `auto`         | `auto`/`on` uses uvloop; `off` uses the asyncio default       |
| `JASA_CACHE_BACKEND`                 | `memory`       | `memory`, `disk`, or `redis`                                  |
| `JASA_DISK_CACHE_PATH`               | `.cache/jasa`  | Filesystem-cache directory                                    |
| `JASA_REDIS_URL`                     | empty          | Required Redis URL when the Redis backend is selected         |
| `JASA_CACHE_MAX_ENTRIES`             | `10000`        | Maximum memory/filesystem entries                             |
| `JASA_SEARCH_CACHE_TTL_SECONDS`      | `129600`       | Complete successful-search TTL                                |
| `JASA_FETCH_CACHE_TTL_SECONDS`       | `86400`        | Successful fetch TTL                                         |
| `JASA_GROUNDING_CACHE_TTL_SECONDS`   | `86400`        | Accepted grounding-output TTL                                 |
| `JASA_USAGE_CACHE_TTL_SECONDS`       | `600`          | Provider usage/quota snapshot TTL                             |
| `JASA_EXPOSE_HELLO`                  | `false`        | Expose omnifetch's reference `say_hello` tool                 |
| `JASA_ENV_FILE`                      | empty          | Compose-only path to a local env file                         |
| `JASA_DOCKER_HOST`                   | `127.0.0.1`    | Compose port-publish host                                     |
| `JASA_DOCKER_PORT`                   | `8000`         | Compose port-publish port                                     |

### REST authentication

Set `JASA_API_KEY` to require a bearer token on `/search`, `/fetch`, `/usage`,
and `/researcher`. If it is empty, those routes are open. Legacy
`OPENWEBUI_API_KEY` and `OMNISEARCH_API_KEY` aliases remain supported, but
`JASA_API_KEY` has precedence. Token comparison is constant-time. Every guarded
route also accepts `?key=...` for compatibility; prefer the bearer header
because query strings are commonly retained in proxy and access logs.

### Search providers

Configure any subset. A missing key disables only that adapter.

| Variable             | Provider         | Notes                                                    |
| -------------------- | ---------------- | -------------------------------------------------------- |
| `TAVILY_API_KEY`     | Tavily           | Native scores; shared with fetch                         |
| `BRAVE_API_KEY`      | Brave Search     | Supports the full rendered operator vocabulary           |
| `KAGI_API_KEY`       | Kagi             | Operators become Kagi lens fields where possible         |
| `EXA_API_KEY`        | Exa              | Auto search with inline page text                        |
| `FIRECRAWL_API_KEY`  | Firecrawl        | Shared with fetch                                        |
| `PERPLEXITY_API_KEY` | Perplexity Sonar | Uses structured search results, then citations fallback  |
| `SERPAPI_API_KEY`    | SerpAPI          | Google Light; shared with YouTube fetch                  |
| `LINKUP_API_KEY`     | Linkup           | Native include/exclude domain filters; shared with fetch |
| `YOU_API_KEY`        | You.com          | Shared with fetch                                        |
| `PARALLEL_API_KEY`   | Parallel         | Advanced search mode                                     |
| `SERPER_API_KEY`     | Serper           | Google organic results                                   |

Search operators include `site:`, `-site:`, `filetype:`, `ext:`, `intitle:`,
`inurl:`, `inbody:`, `inpage:`, `lang:`, `loc:`, `before:`, `after:`, quoted
phrases, `+required`, and `-excluded`. Adapter capabilities differ: Brave and
Serper re-render the complete query, Kagi maps supported fields to a lens,
Tavily extracts domain filters, and other providers receive the raw query.

### Fetch providers

The mounted fetch engine registers 27 adapters. Shared keys such as Tavily,
Firecrawl, Linkup, You.com, and SerpAPI can activate both families.

| Providers                                      | Environment variables                                                                  |
| ---------------------------------------------- | -------------------------------------------------------------------------------------- |
| Tavily, Firecrawl, Jina, You.com               | `TAVILY_API_KEY`, `FIRECRAWL_API_KEY`, `JINA_API_KEY`, `YOU_API_KEY`                   |
| Bright Data                                    | `BRIGHT_DATA_API_KEY`; optional `BRIGHT_DATA_ZONE`                                     |
| Linkup, Diffbot, Olostep                       | `LINKUP_API_KEY`, `DIFFBOT_TOKEN`, `OLOSTEP_API_KEY`                                   |
| Scrapfly, Scrape.do, Decodo                    | `SCRAPFLY_API_KEY`, `SCRAPE_DO_API_TOKEN`, `DECODO_WEB_SCRAPING_API_KEY`               |
| Scrapeless, ScrapeGraphAI                      | `SCRAPELESS_API_KEY`, `SCRAPEGRAPHAI_API_KEY`                                          |
| ScrapingBee, ScrapingAnt, ScraperAPI, Scrappey | `SCRAPINGBEE_API_KEY`, `SCRAPINGANT_API_KEY`, `SCRAPERAPI_API_KEY`, `SCRAPPEY_API_KEY` |
| Oxylabs                                        | `OXYLABS_WEB_SCRAPER_USERNAME` and `OXYLABS_WEB_SCRAPER_PASSWORD`                      |
| Zyte, Spider, LeadMagic, OpenGraph.io          | `ZYTE_API_KEY`, `SPIDER_CLOUD_API_TOKEN`, `LEADMAGIC_API_KEY`, `OPENGRAPH_IO_API_KEY`  |
| GitHub, Supadata, SociaVault                   | `GITHUB_API_KEY`, `SUPADATA_API_KEY`, `SOCIAVAULT_API_KEY`                             |
| SerpAPI                                        | `SERPAPI_API_KEY`                                                                      |
| Kimi                                           | `KIMI_API_KEY` and `SCRAPFLY_API_KEY`                                                  |

The current waterfall checks domain breakers for GitHub, YouTube, and social
sites before the general tiers. General extraction starts with Tavily,
Firecrawl, and Kimi; races several capable middle tiers; then proceeds through
the long fallback group. Results that are empty, suspiciously short, paywalled,
or challenge pages are rejected so the next provider can try.

### Grounded snippets

Set `CEREBRAS_API_KEY` to let MCP `web_search` regenerate snippets for the top
ranked pages from fetched content. The stage uses the same fetch engine, a
bounded worker pool, junk-page detection, strict per-URL deadlines, and a
query-grounded prompt. Accepted LLM outputs are cached independently of the
complete search for `JASA_GROUNDING_CACHE_TTL_SECONDS`; failures fall back to
the aggregated search snippet and never enter that cache. Concurrent identical
misses share one in-process LLM request when its accepted output can be stored.

| Variable                             | Default                                                         |
| ------------------------------------ | --------------------------------------------------------------- |
| `JASA_GROUNDING_MODE`                | `auto`; `on` requires a key; `off` disables automatic grounding |
| `JASA_GROUNDING_CONCURRENCY`         | `10`                                                            |
| `JASA_GROUNDING_PER_URL_DEADLINE_MS` | `7500`                                                          |
| `JASA_GROUNDING_TOP_N`               | `20`                                                            |
| `JASA_GROUNDING_LLM_BASE_URL`        | `https://api.cerebras.ai/v1`                                    |
| `JASA_GROUNDING_LLM_MODEL`           | `gpt-oss-120b`                                                  |
| `JASA_GROUNDING_LLM_TIMEOUT_MS`      | `60000`                                                         |
| `JASA_GROUNDING_MAX_CONTENT_CHARS`   | `24000`                                                         |

`grounded_snippets=true` on an individual MCP request explicitly opts into
grounding and overrides `JASA_GROUNDING_MODE=off`. Omit the tool argument or
set it to `false` when an operator-level `off` should remain effective for a
client.

### OpenTelemetry

Tracing is a no-op unless `OTEL_TRACES_EXPORTER` is `console` or `otlp`. Use
the `telemetry` extra and set the standard `OTEL_SERVICE_NAME`,
`OTEL_EXPORTER_OTLP_ENDPOINT`, and `OTEL_EXPORTER_OTLP_PROTOCOL` variables as
needed. `OTEL_SDK_DISABLED=true` wins over exporter settings.

## Ranking, caching, and failure semantics

Search result order is deterministic:

1. Each provider's results are sorted by native score where one exists.
2. Equivalent URLs are normalized and deduplicated.
3. Each provider contributes `1 / (60 + rank)` to the URL's RRF score.
4. Complementary snippets are selected or sentence-merged within a 500-character
   budget.
5. Thin single-provider and very-low-score entries are filtered unless raw mode
   is requested.
6. The top set is truncated, with strong results from new hosts eligible for
   tail rescue.

Search results are cached for `JASA_SEARCH_CACHE_TTL_SECONDS` (129,600 seconds,
or 36 hours, by default). Versioned, hash-only keys distinguish the exact query,
raw and grounded modes, ordered active providers, and grounding semantics.
Strict nested records treat legacy, malformed, or incompatible values as misses.
A write occurs only when at least one provider succeeds, no provider fails, and
grounding has no transient failures. This completeness gate prevents a temporary
outage from poisoning the cache for the configured TTL.

Concurrent identical search misses coalesce around one provider fan-out in each
Jasa process. Waiters reread the shared cache after the leader finishes; if the
leader fails or produces a partial result that cannot be cached, a waiter becomes
the next leader instead of reusing an unsafe result. Redis shares stored entries
between replicas, but this in-flight coordination is intentionally process-local.
Every caller retains its original timeout budget across cache I/O, coalesced
waiting, fan-out, grounding, and retries after a non-cacheable leader. Slow
cache reads fail at that deadline; slow cache writes fail open so a completed
search can return and release its waiters without extra delay.
DEBUG logs and the metric facade report bounded `hit`, `miss`, `write`,
`read_skipped`, `write_skipped`, `read_error`, `write_error`, and `coalesced`
events without including query or cache-key material. Deadline skips are not
backend errors.

The selected backend applies uniformly to search, fetch, grounding, and usage:

| Backend  | Lifetime and ownership                                                                                     |
| -------- | ---------------------------------------------------------------------------------------------------------- |
| `memory` | Process-local; a new Jasa process starts empty.                                                           |
| `disk`   | Survives Jasa restarts at `JASA_DISK_CACHE_PATH` (image default `/home/app/.cache/jasa`); Compose mounts that default in a named volume, so overriding the path also requires changing the volume target. |
| `redis`  | Shared through `JASA_REDIS_URL`; persistence and backups belong to the operator's Redis deployment.     |

All three use cachelib behind the same asynchronous, fail-open adapter. Backend
readiness is checked without exposing paths, URLs, credentials, keys, or cached
values.

Successful fetches are cached for `JASA_FETCH_CACHE_TTL_SECONDS`. Fetch failures
and invalid cached payloads remain misses. Keys hash the URL and provider
controls, and concurrent identical misses coalesce to one upstream operation in
each process. Memory is process-local, filesystem storage survives restarts, and
Redis shares entries across replicas; single-flight coordination is not
distributed across replicas.

Provider-native usage snapshots are cached at `jasa:usage:v1` for
`JASA_USAGE_CACHE_TTL_SECONDS`. Records include the exact ordered provider
catalog, configured-provider set, and schema fingerprint, so incompatible
deployments refresh instead of reusing stale shapes. Provider-call and cache
failures remain isolated and fail open; normal search and fetch work never
waits for a usage refresh.

Successful grounding LLM outputs are cached independently for
`JASA_GROUNDING_CACHE_TTL_SECONDS`. The v1 hash-only identity covers the exact
effective query/title/truncated-content message, system prompt, model endpoint,
model, generation parameters, and post-processing semantics without including
the API key. Strict records retain only that irreversible digest, an accepted
snippet, and an exact fetched title of at most 2000 characters—not the query,
fetched content, or prompt. An oversized title skips only the cache write.
Fetch failures, short or junk pages, LLM errors, empty output,
sentinels, timeouts, and worker rejection are never written. Cache read/write
faults remain fail-open; reads use at most 250 milliseconds and half the
remaining per-URL budget. A slow cache write cannot downgrade an already
accepted paid LLM result or hold a fetch/LLM worker slot, and a separate bound
shared across searches limits concurrent writes. Concurrent identical grounding
misses coalesce around one process-local LLM flight. Waiters release their
fetch/LLM worker slot, retain their own per-URL deadline, and reread storage
after the leader's bounded cache write. If the leader is cancelled, times out,
returns an error/empty/sentinel result, or cannot store its accepted output, a
waiter becomes the next leader instead of reusing an unsafe result. Grounding
cache logs and metrics use the same bounded event names as search, including
`coalesced`, without query, content, output, credential, or cache-key material.

Provider failures are isolated. Transient `PROVIDER_ERROR` failures receive one
backoff retry; auth, rate-limit, not-found, and invalid-input failures do not.
The final MCP response preserves each failure instead of hiding partial health.

## Health and operations

`GET /health` never calls a paid API. It reports:

- `ok` when search and fetch both have an active provider;
- `degraded` when only one family is configured;
- `unavailable` when neither family is configured;
- active provider names/counts, grounding state, package version, and a live
  cache-backend readiness result sampled at most once every five seconds.

Logs use Rich formatting on stderr so stdio JSON-RPC on stdout stays valid.
Set `JASA_LOG_LEVEL=DEBUG` for request and cache diagnostics. Upstream secrets
are redacted by the shared HTTP layer. Usage-probe warnings retain only the
provider name and upstream HTTP status or exception class. Never log environment
mappings or local `.env` contents.

## Repository map

```text
jasa/
├── AGENTS.md                       # agent navigation hub and invariants
├── README.md                       # user and operator guide
├── .env.example                    # complete, secret-free config contract
├── docker-compose.yml              # local container + optional cache volume
├── Dockerfile                      # non-root multi-stage image
├── pyproject.toml                  # package metadata, pins, tool configuration
├── uv.lock                         # reproducible dependency graph
├── scripts/
│   └── run_provider_integration.py # one-provider, one-paid-call manual harness
├── src/jasa/
│   ├── __main__.py                 # dotenv -> config -> logging -> telemetry -> serve
│   ├── config.py                   # immutable typed settings
│   ├── server.py                   # parent/child assembly and shared resources
│   ├── rest.py                     # /search, /fetch, /usage, /researcher
│   ├── auth.py                     # constant-time REST bearer auth
│   ├── cache/                      # search keys/gate + compatibility stores
│   ├── grounding/                  # fetch -> detect -> LLM snippet pipeline
│   ├── observability/              # fail-open metric facade
│   ├── search/                     # fan-out, retry, RRF, snippets, URL normalization
│   │   └── providers/              # 11 search API adapters and registry
│   ├── usage/                      # usage cache/runtime + one provider probe per PR
│   └── tools/                      # MCP response adapters
└── tests/                          # 100% line/branch unit suite + opt-in Docker test
```

Every directory has an `AGENTS.md` with a file-by-file map, local invariants,
and the fastest test commands for that scope. Agents should begin at the root
[`AGENTS.md`](AGENTS.md), then follow the nearest nested guide.

## Development

Use the Anaconda base environment for project commands when working in this
repository:

```bash
conda run -n base uv sync --frozen --extra telemetry
conda run -n base uv run pytest
conda run -n base uv run ruff format --check
conda run -n base uv run ruff check
conda run -n base uv run mypy
conda run -n base uv run pre-commit run --all-files
conda run -n base uv build
```

The unit suite makes no live provider calls and enforces 100% line and branch
coverage. The Docker integration is opt-in:

```bash
JASA_RUN_DOCKER_TESTS=1 conda run -n base uv run pytest \
  -m docker_integration --no-cov
```

Manual provider integrations deliberately make one paid request per run and
isolate the container to the selected provider:

```bash
conda run -n base uv run python scripts/run_provider_integration.py \
  --family search --provider tavily --surface rest

conda run -n base uv run python scripts/run_provider_integration.py \
  --family fetch --provider jina --surface mcp --no-build
```

Add `--invalid-credential` to verify provider-attributed error behavior. Live
integrations are manual and are not part of the unit workflow.

### Where to make common changes

| Change                      | Start here                                               | Required companion work                                                           |
| --------------------------- | -------------------------------------------------------- | --------------------------------------------------------------------------------- |
| Add a search provider       | `src/jasa/search/providers/`                             | Registry tuple, env example, adapter tests, manual integration case when verified |
| Change ranking/dedup        | `src/jasa/search/ranking.py`, `urls.py`, `snippets.py`   | Update or add golden fixtures and parity tests                                    |
| Change fan-out/retry        | `src/jasa/search/fanout.py`, `retry.py`                  | Deadline, cancellation, and cache-completeness tests                              |
| Change grounding            | `src/jasa/grounding/`                                    | Prompt hash/golden and transient-cache-gate tests                                 |
| Change REST/MCP contracts   | `src/jasa/rest.py`, `server.py`, `schemas.py`, `tools/`  | Both transport surfaces, auth, validation, and README examples                    |
| Change fetch behavior       | Update the pinned omnifetch dependency                   | Preserve the composed-mode boundary in `server.py`                                |
| Add an environment variable | Owning settings/registry                                 | `.env.example` parity test and Compose wiring if relevant                         |
| Change container/release    | `Dockerfile`, `docker-compose.yml`, `.github/workflows/` | Docker integration, actionlint, multi-arch publishing behavior                    |

## Security notes

- Keep `.env` local; only `.env.example` belongs in Git.
- Set `JASA_API_KEY` before exposing REST routes outside a trusted network.
- MCP transport is not guarded by the REST bearer helper; place remote MCP
  behind an authenticated gateway when required.
- The process runs as UID 10001 in the production image.
- Provider keys are passed directly to their upstream APIs. Review provider
  data-retention terms for sensitive research queries.
- Dependencies and GitHub Actions are exact-pinned, and omnifetch is resolved
  from a full Git commit SHA. The unrelated `omnifetch` package name on PyPI is
  never used.

## Releases

Stable Git tags use `vMAJOR.MINOR.PATCH`. The GitHub Release workflow requires
the tag to match `src/jasa/__init__.py`, then builds the wheel/source
distribution and creates the release. The independent container workflow
derives `MAJOR`, `MAJOR.MINOR`, and `MAJOR.MINOR.PATCH` image aliases from that
Git tag; pushes to `main` publish `latest` plus an immutable full-SHA tag.

## License

[MIT](LICENSE) © 2026 CJ Angrist.
