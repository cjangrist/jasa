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

- `web_search` asks every configured search provider in parallel,
  merges their blind spots with Reciprocal Rank Fusion, deduplicates URLs, consolidates
  snippets, and keeps the top 50 high-signal results plus eligible tail rescues.
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
| Search coverage  | One index and one ranking model           | 17 search providers, including Keenable, Ollama, and DuckDuckGo through Scrapfly            |
| Result quality   | Provider-native order and duplicate links | Deterministic RRF, URL normalization, snippet collapse, quality filtering, and tail rescue |
| Snippet trust    | Search-engine excerpts                    | Optional snippets regenerated from fetched page content                                    |
| URL extraction   | One scraper succeeds or the request fails | 28 fetch adapters behind domain breakers and a tiered waterfall                            |
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
    | /mcp                             | /search, /fetch, /researcher
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
 optional fetch -> per-key flight -> grounding cache -> LLM waterfall on miss
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
# Optionally add provider keys to expand search coverage or enable fetch.
uv run jasa --transport http --host 127.0.0.1 --port 8000
```

Verify the process without spending provider credits:

```bash
curl -fsS http://127.0.0.1:8000/health
```

The default command, `uv run jasa`, uses MCP over stdio. HTTP mode exposes MCP
at `http://127.0.0.1:8000/mcp` and the REST routes described below.

### Run with Docker Compose

```bash
cp .env.example .env
# Optionally configure provider keys in .env.
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
- `0`, `0.7`, and `0.7.0`-style tags for stable releases.

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
      "url": "http://127.0.0.1:8000/mcp"
    }
  }
}
```

HTTP mode answers browser preflight requests and exposes `Mcp-Session-Id` to
cross-origin MCP clients. Reverse proxies must pass `X-Forwarded-Proto`; set
Uvicorn's `FORWARDED_ALLOW_IPS` to the trusted proxy address or network so
trailing-slash redirects retain the public HTTPS scheme.

### MCP tool: `web_search`

| Input               | Type                 | Default   | Meaning                                                                |
| ------------------- | -------------------- | --------- | ---------------------------------------------------------------------- |
| `query`             | string, 1-2000 chars | required  | Search query; advanced operators are supported by compatible providers |
| `timeout_ms`        | positive integer     | `JASA_SEARCH_TIMEOUT_MS` | Global cache, fan-out, coalescing, and grounding budget |
| `include_snippets`  | boolean              | `true`    | Include consolidated snippets in each result                           |
| `grounded_snippets` | boolean or null      | automatic | Regenerate top-result snippets when a grounding key is configured      |

MCP advertises `web_search` as read-only, non-destructive, idempotent, and
open-world so clients can safely select it without inferring side effects from
the description.

Clients that include an MCP `progressToken` receive standard progress
notifications while a search checks cache, fans out to providers, ranks, and
grounds results. Grounding emits a small set of completion milestones rather
than one event per URL. Clients without progress support see the same response;
each update is capped at 100 milliseconds and notification failures never fail
the search.

Leave `timeout_ms` unset unless latency matters more than snippet quality. A
short caller deadline is spent by the provider fan-out first, leaving grounding
too little time to finish the page fetch and LLM call it has already paid for.

The response contains `providers_succeeded`, `providers_failed`, total timing,
a `grounding` block, truncation counts, and `web_results`. Each result carries
its contributing providers, RRF score, and `snippet_source`. MCP keeps
`JASA_SEARCH_MAX_RESULTS` results (50 by default) plus eligible tail rescues
from previously unseen hosts. Only the first `JASA_GROUNDING_TOP_N` results
(20 by default) are fetched and grounded; the remaining rows retain aggregated
provider snippets.

`tools/list` publishes a strict, fully dereferenced `outputSchema` for this
shape. Every fixed object rejects unknown properties, numeric fields and arrays
carry their concrete JSON types, and `snippet_source` is an explicit enum.
FastMCP validates and returns the same data in MCP `structuredContent`; clients
can consume it without parsing the backward-compatible text content.

`snippet_source` is always present and answers where the snippet came from:

| Value        | Meaning                                                          |
| ------------ | ---------------------------------------------------------------- |
| `aggregated` | Merged provider snippets; grounding did not reach this result    |
| `grounded`   | Rewritten from fetched page content                              |
| `fallback`   | Grounding attempted this result and failed; aggregated text kept |

The `grounding` block reports the stage as a whole, so a caller never has to
infer from a successful response whether grounding actually ran. `outcomes`
names the reason for every shortfall.

```json
{
  "query": "python structured concurrency",
  "providers_succeeded": [
    { "provider": "brave", "duration_ms": 412 },
    { "provider": "tavily", "duration_ms": 683 }
  ],
  "providers_failed": [],
  "total_duration_ms": 697,
  "grounding": {
    "requested": true,
    "attempted": 20,
    "grounded": 18,
    "outcomes": { "grounded": 18, "fallback:fetch_junk": 2 }
  },
  "truncation": { "total_before": 34, "kept": 31, "rescued": 1 },
  "web_results": [
    {
      "title": "...",
      "url": "https://example.com/article",
      "snippets": ["..."],
      "source_providers": ["tavily", "brave"],
      "score": 0.0325,
      "snippet_source": "grounded"
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

`status` is `success` when content was returned. A provider-local 404, including
Tavily extraction 404s, is recorded while the remaining waterfall continues.
If every eligible provider is exhausted, MCP still returns a normal structured
result: `not_found` when at least one provider reported that the URL was
missing, or `unavailable` otherwise. `providers_attempted`, `providers_failed`,
and `message` explain the outcome; invalid tool inputs remain MCP errors.

## REST API

HTTP mode offers thin compatibility routes over the same execution paths.

| Route          | Method              | Purpose                                               | Default result count |
| -------------- | ------------------- | ----------------------------------------------------- | -------------------- |
| `/health`, `/` | `GET`               | Free liveness/readiness and active-provider inventory | n/a                  |
| `/search`      | `POST`              | Search results shaped as `{link,title,snippet}`       | 20; `0` returns all  |
| `/searchxng`   | `GET`, `POST`       | SearXNG-compatible HTML, JSON, CSV, or RSS search     | 20 per page          |
| `/fetch`       | `POST`              | Full omnifetch result                                 | one primary result   |
| `/usage`       | `GET`               | Cached provider-native usage and quota snapshots      | n/a                  |
| `/researcher`  | `GET`, `POST`       | GPT-Researcher-compatible `{href,body}` snippets      | 10                   |
| `/mcp`         | MCP Streamable HTTP | FastMCP tools and resources                           | tool-specific        |

Search:

```bash
curl -fsS http://127.0.0.1:8000/search \
  -H 'content-type: application/json' \
  -H "authorization: Bearer $JASA_API_KEY" \
  -d '{"query":"site:docs.python.org asyncio TaskGroup","count":10}'
```

Set `raw: true` in the request body to bypass the search quality filter.
`count` is clamped to 0-100. REST searches have a 30-second fan-out deadline.

SearXNG compatibility:

```bash
curl -fsS --get http://127.0.0.1:8000/searchxng \
  --data-urlencode 'q=python asyncio' \
  --data 'format=json&language=en-US&safesearch=1'
```

`/searchxng` follows the SearXNG Search API: GET parameters belong in the URL,
POST parameters use `application/x-www-form-urlencoded`, and `format` accepts
`json`, `csv`, or `rss` while an omitted format returns simple HTML. It accepts
`q`, `categories`, `language`, `pageno`, `time_range`, `safesearch`, and
`theme`; `week` is accepted alongside SearXNG's documented `day`, `month`, and
`year` ranges. Language and time range become Jasa search operators, which
providers honor on a best-effort basis, and page numbers select successive
20-result windows from the ranked result set. Result positions restart at 1 on
every page, matching SearXNG; pages beyond Jasa's aggregated pool are empty.

Jasa exposes one `general` web category and its providers do not share a
portable safe-search control, so other category names, presentation settings,
and `safesearch` are accepted as instance preferences without changing provider
selection. JSON uses SearXNG's `query`/`number_of_results`/`results` envelope,
where `number_of_results` counts the full ranked pool before pagination. Each
result carries at least `url`, `title`, `content`, `engine`, `engines`,
`positions`, `score`, and `category`, making the route directly consumable by
Open WebUI's SearXNG adapter. Point that adapter at the full `/searchxng` URL.
When REST auth is enabled, Open WebUI does not send an authorization header;
configure the URL as
`https://example.test/searchxng?key=YOUR_JASA_API_KEY` instead.

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
Linkup, You.com, Olostep, ScrapeGraphAI, Scrapeless, Scrapfly, Scrappey,
SociaVault, Spider, and Supadata are the currently integrated usage sources;
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
Supadata uses its free account endpoint for plan and credit-usage data.

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
| `JASA_PUBLIC_URL`                    | empty          | Externally reachable `https://` origin; advertised, never bound |
| `JASA_CACHE_BACKEND`                 | `memory`       | `memory`, `disk`, or `redis`                                  |
| `JASA_DISK_CACHE_PATH`               | `.cache/jasa`  | Filesystem-cache directory                                    |
| `JASA_REDIS_URL`                     | empty          | Required Redis URL when the Redis backend is selected         |
| `JASA_CACHE_MAX_ENTRIES`             | `10000`        | Maximum memory/filesystem entries                             |
| `JASA_SEARCH_TIMEOUT_MS`             | `58000`        | Whole-request budget when a caller names no deadline          |
| `JASA_SEARCH_FANOUT_TIMEOUT_MS`      | `25000`        | Fan-out's share of that budget; the rest is left for grounding |
| `JASA_SEARCH_MAX_RESULTS`            | `50`           | MCP ranked rows before eligible tail rescues                  |
| `JASA_SEARCH_CACHE_TTL_SECONDS`      | `129600`       | Complete successful-search TTL                                |
| `JASA_FETCH_CACHE_TTL_SECONDS`       | `864000`       | Successful fetch TTL (10 days)                               |
| `JASA_VOLATILE_FETCH_CACHE_TTL_SECONDS` | `300`       | Homepage TTL; capped by the row above                        |
| `JASA_GROUNDING_CACHE_TTL_SECONDS`   | `86400`        | Accepted grounding-output TTL                                 |
| `JASA_USAGE_CACHE_TTL_SECONDS`       | `600`          | Provider usage/quota snapshot TTL                             |
| `JASA_EXPOSE_HELLO`                  | `false`        | Expose omnifetch's reference `say_hello` tool                 |
| `JASA_ENV_FILE`                      | empty          | Compose-only path to a local env file                         |
| `JASA_DOCKER_HOST`                   | `127.0.0.1`    | Compose port-publish host                                     |
| `JASA_DOCKER_PORT`                   | `8000`         | Compose port-publish port                                     |

### REST authentication

Set `JASA_API_KEY` to require a bearer token on `/search`, `/searchxng`,
`/fetch`, `/usage`, and `/researcher`. If it is empty, those routes are open.
Legacy `OPENWEBUI_API_KEY` and `OMNISEARCH_API_KEY` aliases remain supported,
but `JASA_API_KEY` has precedence. Token comparison is constant-time. Every
guarded route also accepts `?key=...` for compatibility; prefer the bearer
header because query strings are commonly retained in proxy and access logs.

### Search providers

Configure any subset of providers; a missing key disables only that adapter.

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
| `ANTHROPIC_AUTH_TOKEN` | Claude         | Anthropic server-tool search with cited source excerpts  |
| `OPENAI_API_KEY`     | Codex            | OpenAI hosted web search; cited URLs without excerpts    |
| `Z_AI_API_KEY`       | Z.AI             | GLM server-tool search; distinct index, capped at 10 results |
| `SCRAPFLY_API_KEY`   | DDGS             | DuckDuckGo html search via the Scrapfly scrape API; shared with fetch |
| `OLLAMA_API_KEY`     | Ollama Web Search | Hosted search API; always requests 10 results            |
| `KEENABLE_API_KEY`   | Keenable         | Native site/date filters; always requests 50 results     |

The three LLM-mediated adapters accept optional non-secret settings: a
`*_BASE_URL` selecting the endpoint and a `*_SEARCH_MODEL` selecting the model
that runs there. They activate nothing on their own, and none needs
configuration beyond its credential. Claude and Codex default to this project's
own gateway; Z.AI defaults to the vendor directly, at `api.z.ai`, because no
gateway fronts it.

Z.AI reaches search through a chat completion carrying a server-side tool, and
its upstream honours a result `count` only up to ten, so it contributes ten
results where other providers contribute thirty. It earns its place on index
diversity rather than volume. Generation is capped at one token because the
adapter reads the tool's own result array and never the model's prose.

Ollama's hosted search API always requests ten ranked results per fan-out.
Its API exposes no structural search filters, so Jasa preserves supported
operators in the query text.

Keenable's Search API always requests its maximum of fifty ranked results per
fan-out. It receives one inclusive domain through its native `site` field and
maps `after:` / `before:` to publication-date bounds. Multiple inclusive
domains, excluded domains, and unsupported operators remain in the query.
Those date bounds accept Keenable's full dates, ISO timestamps, and relative
deltas such as `7d`; relative values must be positive and resolve inside the
same accepted date window. One UTC reference instant governs validation,
strict-bound selection, and contradiction checks for the whole request. Jasa's
year and year-month shorthand expands to inclusive full-date bounds before the
request. Only clean hostnames and bounds inside Keenable's `1970-01-01` through
`2149-06-05` window become native fields; malformed, escaped, pipe-scoped, or
out-of-range values and unsupported operators remain literal query text in
their source positions.

Jasa exposes DDGS as one provider covering only DuckDuckGo text search. The
adapter GETs DuckDuckGo's html endpoint through the Scrapfly scrape API —
direct datacenter requests now meet a 202 anomaly challenge — and decodes
DuckDuckGo's redirect links back to their target URLs.

| Variable              | Default                     | Purpose                                    |
| --------------------- | --------------------------- | ------------------------------------------ |
| `ANTHROPIC_BASE_URL`  | `https://ai.angrist.net`    | Messages-compatible endpoint for Claude    |
| `CLAUDE_SEARCH_MODEL` | `claude-haiku-4-5-20251001` | Model that drives Claude's web-search tool |
| `OPENAI_BASE_URL`     | `https://ai.angrist.net/v1` | Responses-compatible endpoint for Codex    |
| `CODEX_SEARCH_MODEL`  | `gpt-5.6-luna`              | Model that drives Codex's web-search tool  |
| `Z_AI_BASE_URL`       | `https://api.z.ai/api/coding/paas/v4` | Chat-completions endpoint for Z.AI |
| `ZAI_SEARCH_MODEL`    | `glm-4.6`                   | Model that drives Z.AI's web-search tool   |

> **These two adapters default to a third-party endpoint.** Unless you override
> `ANTHROPIC_BASE_URL` / `OPENAI_BASE_URL`, a configured `ANTHROPIC_AUTH_TOKEN`
> or `OPENAI_API_KEY` is sent to `ai.angrist.net` rather than to Anthropic or
> OpenAI. Every other adapter in this repository talks to its vendor directly.
> Set the endpoint below before configuring either credential if that is not
> what you want.

To call the vendors directly, set `ANTHROPIC_BASE_URL=https://api.anthropic.com`
or `OPENAI_BASE_URL=https://api.openai.com/v1`. The model follows the endpoint:
retargeting Codex without naming a model falls back to the official `gpt-5.6`
rather than sending a gateway-only id to OpenAI, and Claude's default id is
served by both. Name a model explicitly with `CLAUDE_SEARCH_MODEL` or
`CODEX_SEARCH_MODEL` to override that.

Claude sends both `x-api-key` and `Authorization: Bearer`, so a provider-native
API key and a gateway bearer token each authenticate. Codex reports the sources
it used as citations without excerpts, so its results carry no snippet and rely
on other providers, or on grounded snippets, for text.

All three model settings — `CLAUDE_SEARCH_MODEL`, `CODEX_SEARCH_MODEL`, and
`ZAI_SEARCH_MODEL` — name an id the vendor eventually retires or renames; the
defaults are reviewed against the published model lists each release.

Search operators include `site:`, `-site:`, `filetype:`, `ext:`, `intitle:`,
`inurl:`, `inbody:`, `inpage:`, `lang:`, `loc:`, `before:`, `after:`, quoted
phrases, `+required`, and `-excluded`. Adapter capabilities differ: Brave,
DDGS, Ollama, Serper, and Z.AI re-render the complete query, Kagi maps supported
fields to a lens, Keenable maps one site and date bounds, Tavily, Claude, and
Codex extract domain filters, and other providers receive the raw query. Z.AI
re-renders everything because its upstream accepts domain and recency filters
and then ignores them, so sending one structurally would silently drop it.

### Fetch providers

The mounted fetch engine registers 28 fetch adapters. Shared keys such as
Tavily, Firecrawl, Linkup, You.com, and SerpAPI can activate both families.

| Providers                                      | Environment variables                                                                  |
| ---------------------------------------------- | -------------------------------------------------------------------------------------- |
| Tavily, fastCRW, Firecrawl, Jina, You.com      | `TAVILY_API_KEY`, `CRW_API_KEY`, `FIRECRAWL_API_KEY`, `JINA_API_KEY`, `YOU_API_KEY`    |
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
fastCRW, Firecrawl, and Kimi; races several capable middle tiers; then
proceeds through the long fallback group. Results that are empty, suspiciously
short, paywalled, or challenge pages are rejected so the next provider can try.
A provider's `NOT_FOUND` result is likewise local to that attempt; it never
short-circuits a sibling or a later tier.

### Grounded snippets

Set a non-blank credential named by an active waterfall tier to let MCP
`web_search` regenerate snippets for the top-ranked pages from fetched content.
For the packaged chain below that means `CEREBRAS_API_KEY` or `OPENAI_API_KEY`;
a custom chain names whatever variables its tiers declare. The stage uses the same
fetch engine, a bounded worker pool, junk-page detection, strict per-URL
deadlines, and a query-grounded prompt. Accepted LLM outputs are cached
independently of the complete search for `JASA_GROUNDING_CACHE_TTL_SECONDS`;
failures fall back to the aggregated search snippet and never enter that cache.
Concurrent identical misses share one in-process LLM request when its accepted
output can be stored.

#### The LLM waterfall

Grounding pays for a page fetch before it can call an LLM, so a single
rate-limited endpoint would otherwise discard work that was already billed.
The snippet call therefore walks an ordered chain of OpenAI-compatible
chat-completions endpoints, spending that fetch once:

| Tier | Endpoint                    | Model                       | Credential         |
| ---- | --------------------------- | --------------------------- | ------------------ |
| 1    | `https://api.cerebras.ai/v1`| `gpt-oss-120b`              | `CEREBRAS_API_KEY` |
| 2    | `https://ai.angrist.net/v1` | `gpt-5.6-luna`              | `OPENAI_API_KEY`   |
| 3    | `https://ai.angrist.net/v1` | `claude-haiku-4-5-20251001` | `OPENAI_API_KEY`   |
| 4    | `https://ai.angrist.net/v1` | `glm-5.3`                   | `OPENAI_API_KEY`   |

A tier advances on a transport failure, a non-2xx status, an error object in a
200 body, an unreadable response shape, empty text, or an explicit stop whose
normal snippet omits the required final `Coverage:` line. A sentinel does not
advance: it is the model's judgment about the fetched page rather than a
failure. Credentials are resolved per request, not at boot: a tier whose
credential is unset when a search runs is dropped from that search's chain, and
one exported later joins the chain on the next search without a restart. The
chain therefore runs on whatever is configured at the time, and grounding stays
available while any one credential remains. The whole chain shares the single
per-URL deadline.

Because any tier may serve a request, accepted output is cached against the
whole ordered chain rather than the tier that answered. Editing the chain
therefore starts a fresh cache namespace instead of serving snippets the new
chain would not have produced.

The chain ships as `src/jasa/grounding/waterfall.yaml`. Copy it, edit
`base_url`, `model`, `api_key_env`, or `timeout_ms`, and point
`JASA_GROUNDING_WATERFALL_PATH` at the copy to swap providers without
rebuilding. A tier that omits `base_url`, `model`, or `timeout_ms` inherits the
matching `JASA_GROUNDING_LLM_*` setting, which is how tier 1 stays under
environment control. A malformed file fails startup rather than silently
disabling grounding.

| Variable                             | Default                                                         |
| ------------------------------------ | --------------------------------------------------------------- |
| `JASA_GROUNDING_MODE`                | `auto`; `on` requires a key; `off` disables automatic grounding |
| `JASA_GROUNDING_CONCURRENCY`         | `20`; match `TOP_N` so the page set resolves in one wave        |
| `JASA_GROUNDING_PER_URL_DEADLINE_MS` | `30000`; covers the page fetch *and* every LLM tier behind it   |
| `JASA_GROUNDING_TOP_N`               | `20`                                                            |
| `JASA_GROUNDING_LLM_BASE_URL`        | `https://api.cerebras.ai/v1`                                    |
| `JASA_GROUNDING_LLM_MODEL`           | `gpt-oss-120b`                                                  |
| `JASA_GROUNDING_LLM_TIMEOUT_MS`      | `25000`; the bound for one tier, not for the chain              |
| `JASA_GROUNDING_WATERFALL_PATH`      | empty; the packaged `waterfall.yaml`                            |
| `JASA_GROUNDING_MAX_CONTENT_CHARS`   | `48000`                                                         |

These deadlines are deliberately generous. A grounded snippet costs a page
fetch plus at least one LLM completion, and both are billed before any deadline
can fire, so a tight budget does not save money -- it pays for work and then
throws it away. A tier's timeout is an upper bound rather than a claim on the
budget: each attempt also yields a minimum slice to every tier still queued
behind it, so a first tier that hangs cannot leave the fallbacks unreachable.

An expired budget cancels only the URLs still in flight. Every URL that already
produced a snippet keeps it, and the ones that did not are reported through
`snippet_source: "fallback"` and the response's `grounding.outcomes`. Once the
fan-out has produced usable ranked rows, grounding exhaustion never converts
the whole search into an error; the paid aggregate result is returned.

Grounding runs after the fan-out, so it can only use time the fan-out left
behind. `JASA_SEARCH_FANOUT_TIMEOUT_MS` bounds the fan-out inside the request
budget for exactly that reason; raising it or lowering `JASA_SEARCH_TIMEOUT_MS`
narrows the window grounding has to work in.

The 58-second default sits below the 60-second timeout MCP clients commonly
ship with. With the 25-second fan-out cap it leaves roughly the configured
30-second per-URL grounding deadline plus fixed harvest and response reserves.
The client timeout is still the true ceiling: a client that gives up
mid-request abandons every fetch and completion the server already paid for.
Raise `JASA_SEARCH_TIMEOUT_MS` only alongside the client's own timeout.

Keep `JASA_GROUNDING_CONCURRENCY` equal to `JASA_GROUNDING_TOP_N`. A smaller
value splits the page set into waves, and the later waves begin so close to the
deadline that they time out after paying for their fetches. Matching the two
costs no additional LLM calls.

`grounded_snippets=true` on an individual MCP request explicitly opts into
grounding and overrides `JASA_GROUNDING_MODE=off`. Omit the tool argument or
set it to `false` when an operator-level `off` should remain effective for a
client.

### Server icon

Jasa ships its own icon and declares it in `serverInfo.icons`, the field the MCP
specification added in revision 2025-11-25 (SEP-973). The same packaged images
are served from the server's own origin:

| Route          | Contents                                                  |
| -------------- | ---------------------------------------------------------- |
| `/icon.png`    | 256×256 PNG; `?size=48` and `?size=128` select smaller ones |
| `/favicon.png` | The same handler, under the conventional name              |
| `/favicon.ico` | Multi-resolution ICO, 16 through 256                       |

With `JASA_PUBLIC_URL` unset the icon is inlined as a `data:` URI, so a server
that cannot name its own address still advertises one. Setting it switches to
served URLs, advertises every size, and fills in `serverInfo.websiteUrl`.

When set, it must be an **`https://` origin and nothing more**: a host (name or
IP literal) with an optional port, and no path, query, fragment, or
credentials. The icon routes are served at the origin root, so a value like
`https://example.com/mcp` would advertise `/mcp/icon.png` and every client
would get a 404. A value that fails these checks stops startup with a message
naming the reason, rather than silently reverting to the inline icon — a
silent fallback is indistinguishable from success and would leave a typo
looking like it worked.

**What today's clients actually do is a separate matter**, and worth knowing
before assuming this changes anything you can see:

- **ChatGPT** takes the icon from its own connector/app settings rather than
  from the server, so set it there. The exact place has moved between releases;
  see OpenAI's current
  [developer mode and MCP connectors guide](https://help.openai.com/en/articles/12584461-developer-mode-and-full-mcp-connectors-in-chatgpt).
  Supply a square image.
- **Claude** resolves a favicon for the *registrable root domain* of the
  connector URL through Google's favicon service. It reads neither the spec
  field nor a favicon served by the MCP server itself. If Jasa runs on a
  subdomain, the icon shown belongs to the parent domain; on shared hosting
  (`*.example-platform.app`) no per-deployment icon is possible today.

Both behaviours are tracked upstream and may change. Declaring the icon costs
nothing and is what a conforming client will read once support lands.

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

When Keenable is active, a query containing relative `after:` / `before:`
syntax bypasses this aggregate cache. Those windows move with wall-clock time,
so reusing the exact raw query for 36 hours would return stale results. Other
queries retain the normal completeness and TTL policy.

Concurrent identical search misses coalesce around one provider fan-out in each
Jasa process. Waiters reread the shared cache after the leader finishes; if the
leader fails or produces a partial result that cannot be cached, a waiter becomes
the next leader instead of reusing an unsafe result. Redis shares stored entries
between replicas, but this in-flight coordination is intentionally process-local.
Every caller retains its original timeout budget across cache I/O, coalesced
waiting, fan-out, grounding, and retries after a non-cacheable leader. That
budget is split before it is spent: the fan-out is bounded by
`JASA_SEARCH_FANOUT_TIMEOUT_MS` so grounding inherits time rather than whatever
the slowest provider left behind. Slow cache reads fail at the deadline; slow
cache writes fail open so a completed search can return and release its waiters
without extra delay.

Provider work still pending at the fan-out deadline is cancelled concurrently.
Cancellation cleanup has a short bounded grace and repeats cancellation once;
a broken provider that ignores both requests is logged and retired in the
background instead of holding the entire MCP connection open.

An expired grounding budget cancels only the URLs still in flight. Every URL
that already produced a snippet keeps it, because that snippet cost a page
fetch and an LLM completion that were billed long before the deadline arrived.
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

Successful fetches are cached for `JASA_FETCH_CACHE_TTL_SECONDS`, ten days by
default: a published page does not change, so re-fetching it across searches is
money spent for the same bytes. Site homepages are the exception. A bare host
such as `https://cnn.com` is a rolling index rather than a document, so any URL
whose path is empty or `/` expires on `JASA_VOLATILE_FETCH_CACHE_TTL_SECONDS`
instead, five minutes by default; that value is capped by the full TTL, so
shortening the main one keeps everything fresher rather than making homepages
the stalest entries. Homepages are still cached and coalesced, so a burst of
searches inside the window costs one fetch rather than one each.

Fetch failures and invalid cached payloads remain misses. Keys hash the URL
and provider controls, and concurrent identical misses coalesce to one
upstream operation in each process. The URL is folded to Jasa's canonical
form first, so `/x` and `/x/`, a default port, host casing, a fragment, and
a dot segment all share one entry rather than each buying the page again;
the URL sent to a provider is still the one that was asked for. Memory is process-local, filesystem storage survives restarts, and
Redis shares entries across replicas; single-flight coordination is not
distributed across replicas.

Provider-native usage snapshots are cached at `jasa:usage:v1` for
`JASA_USAGE_CACHE_TTL_SECONDS`. Records include the exact ordered provider
catalog, configured-provider set, and schema fingerprint, so incompatible
deployments refresh instead of reusing stale shapes. Provider-call and cache
failures remain isolated and fail open; normal search and fetch work never
waits for a usage refresh.

Successful grounding LLM outputs are cached independently for
`JASA_GROUNDING_CACHE_TTL_SECONDS`. The v2 hash-only identity covers the
canonical fetch URL, the query, a prompt fingerprint (template, truncation
marker, system-prompt digest, and content cap), the ordered model endpoints and
models, generation parameters, and post-processing semantics without including
the API key. It deliberately does not cover the fetched page content: the same
URL arrives as different markdown whenever a different fetch provider wins its
race, and keying on that rendering made every accepted snippet unaddressable
the moment it changed—so reordering the fetch waterfall re-bought every
grounding call. The URL comes from the same canonicalizer the fetch cache keys
on, so both caches agree on which spellings are one page.

Because the identity no longer changes when a page does, the lifetime carries
invalidation instead: a snippet written from a homepage is clamped to
`JASA_VOLATILE_FETCH_CACHE_TTL_SECONDS`, matching the short lifetime omnifetch
gives the rolling index it was written from. For an ordinary page the fetch
entry outlives the snippet many times over, so nothing gets staler than before.

Strict records retain only the irreversible digest and an accepted snippet—not
the query, URL, fetched content, title, or prompt. Fetch failures, short or
junk pages, LLM errors, empty output,
sentinels, timeouts, and worker rejection are never written. Cache read/write
faults remain fail-open; reads use at most 250 milliseconds and half the
remaining per-URL budget. A slow cache write cannot downgrade an already
accepted paid LLM result or hold a fetch/LLM worker slot, and a separate bound
shared across searches limits concurrent writes. Concurrent misses on the same
page and query coalesce around one process-local LLM flight. Waiters release their
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
- `degraded` when only one family is active;
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
│   └── run_provider_integration.py # one-provider, one-live-call manual harness
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
│   │   └── providers/              # 17 search adapters and registry
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

Manual provider integrations deliberately make one live target request per run
and isolate the selected credentialed provider:

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
| Change fetch behavior       | Merge omnifetch, then refresh its locked Git source      | Preserve the composed-mode boundary in `server.py`                                |
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
- Dependencies and GitHub Actions are exact-pinned. Omnifetch tracks its
  GitHub `main` branch in `pyproject.toml`, while `uv.lock` records the exact
  resolved commit rather than installing the unrelated PyPI package. Refresh
  it with `uv lock --upgrade-package omnifetch` after upstream merges.

## Releases

Stable Git tags use `vMAJOR.MINOR.PATCH`. The GitHub Release workflow requires
the tag to match `src/jasa/__init__.py`, then builds the wheel/source
distribution and creates the release. The independent container workflow
derives `MAJOR`, `MAJOR.MINOR`, and `MAJOR.MINOR.PATCH` image aliases from that
Git tag; pushes to `main` publish `latest` plus an immutable full-SHA tag.

## License

[MIT](LICENSE) © 2026 CJ Angrist.
