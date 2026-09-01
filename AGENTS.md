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
| Where is fetch implemented?          | locked `omnifetch` Git source            | composition notes in `src/jasa/AGENTS.md` |
| How are snippets grounded?           | `src/jasa/grounding/service.py`         | `src/jasa/grounding/AGENTS.md`            |
| Which HTTP routes exist?             | `src/jasa/rest.py`                      | `src/jasa/server.py`                      |
| How are provider quotas collected?   | `src/jasa/usage/runtime.py`             | `src/jasa/usage/AGENTS.md`                |
| Which MCP tools exist?               | `src/jasa/server.py`                    | `src/jasa/tools/AGENTS.md`                |
| How is configuration loaded?         | `src/jasa/config.py`                    | `.env.example`, `tests/test_config.py`    |
| How are images/releases published?   | `.github/workflows/`                    | `.github/workflows/AGENTS.md`             |
| How are live providers tested?       | `scripts/run_provider_integration.py`   | `scripts/AGENTS.md`                       |
| Which test owns a behavior?          | `tests/AGENTS.md`                       | nearest `test_*.py`                       |

## Runtime in one minute

`jasa.__main__.main()` loads `.env`, resolves immutable settings, validates
startup, configures stderr logging, installs uvloop when available, enables
optional telemetry, and calls `build_server()`.

`build_composition_async()` creates exactly one `httpx.AsyncClient`, snapshots
provider secrets, constructs Jasa's search registry, builds one shared cachelib
backend, injects the client and cache into an omnifetch `Engine`, and mounts the
child. `build_composition()` is the synchronous pre-event-loop wrapper.
Jasa registers `web_search`; the child supplies `web_fetch`. The child does not
own the shared client and its standalone `/web_fetch` route is disabled. One
usage runtime borrows the shared client/cache, and parent middleware observes
both tools. The parent lifespan closes usage work, cache, client, and telemetry.

Search path:

```text
MCP/REST -> shared SearchRuntime -> cache read -> per-key miss coalescing
         -> parallel provider dispatch
         -> selective retry -> deterministic RRF/dedup/snippet collapse
         -> quality filter
             |-> MCP only: fetch -> flight/cache -> LLM waterfall -|
             |-> REST: no grounding ---------------------------|
         -> complete-result cache write -> transport-specific formatting
```

Fetch path:

```text
MCP/REST/grounding -> omnifetch.execute_web_fetch -> cache read
                   -> domain breakers -> tiered provider waterfall/races
                   -> quality gate -> successful cache write
                   -> attributed result/failures
```

## First five minutes on a task

1. Run `git status --short --branch`; preserve unrelated user changes.
2. Read the root files that define the task's contract (`README.md`,
   `pyproject.toml`, `.env.example`, Compose/workflow config as relevant).
3. Follow the fast-orientation table to the nearest `AGENTS.md` and read every
   file in that directory before editing.
4. Locate the focused test in `tests/AGENTS.md`; run it once before changing
   behavior when a regression is suspected.
5. Make the smallest owning-layer change, run the focused test, then the full
   required checks before committing.

Useful fact commands:

```bash
conda run -n base uv run jasa --help
conda run -n base uv tree --depth 1
conda run -n base uv run pytest --collect-only -q
docker compose config --quiet
```

## Repository layout

| Path                 | Ownership                                                            |
| -------------------- | -------------------------------------------------------------------- |
| `README.md`          | User-facing product, setup, API, operations, and contributor guide.  |
| `.env.example`       | Exact tested, secret-free runtime and provider configuration contract. |
| `pyproject.toml`     | Package metadata, exact dependency pins, tools, coverage gate.       |
| `uv.lock`            | Reproducible resolution. Change only through `uv lock`/`uv sync`.    |
| `Dockerfile`         | Multi-stage, non-root production image.                              |
| `docker-compose.yml` | Local deployment and optional persistent disk-cache volume.          |
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
  omnifetch-owned. Fix fetch behavior upstream, merge it, then refresh the
  locked source with `uv lock --upgrade-package omnifetch`.
- Provider-native secret names have no `JASA_` prefix. Shared names can enable
  both search and fetch adapters; DDGS searches DuckDuckGo through the shared
  `SCRAPFLY_API_KEY`.
- A search adapter may declare optional non-secret settings in `setting_envs`
  (gateway base URL, model id). The registry resolves them from the same
  environment snapshot it gates on, and they never activate an adapter.
- `.env.example` must exactly equal all supported settings, auth aliases,
  Compose substitutions, search secrets, search adapter settings, and
  omnifetch fetch secrets. The equality and empty-secret tests are in
  `tests/test_config.py`.
- A populated `.env` is local-only. Never print or commit secret values.
- stdout belongs to MCP stdio JSON-RPC; application logs go to stderr.
- Search aggregation is deterministic in registry order even when providers
  finish out of order.
- Only complete, non-transient search outcomes are cached, using
  `JASA_SEARCH_CACHE_TTL_SECONDS` (36 hours by default).
- Search cache v4 keys scope exact query, raw/grounded mode, ordered providers,
  and grounding semantics; strict versioned records make incompatible data a
  miss. Bump the version when the fan-out starts producing a materially
  different result set for an unchanged key, so a deploy is not shadowed by
  entries the new policy would never have produced.
- MCP, `/search`, and `/researcher` share one process-local search flight
  registry. Concurrent identical misses dispatch once when the leader writes a
  complete result; waiters retry independently after non-cacheable outcomes.
- MCP `web_search` reports protocol-native progress when the caller supplies a
  `progressToken`: cache lookup, provider fan-out, ranking, bounded grounding
  completion milestones, and completion. Reporting is best-effort, bounded,
  query-free, and absent from REST; it never changes cache identity or outcome.
- Successful composed fetches use the same backend as search and honor
  `JASA_FETCH_CACHE_TTL_SECONDS` (10 days by default) and coalesce concurrent
  identical misses to one upstream fetch; omnifetch runtime variables remain
  ignored. A URL whose path is empty or `/` is a rolling homepage and expires
  on the shorter `JASA_VOLATILE_FETCH_CACHE_TTL_SECONDS` instead, which the
  full TTL caps. Both are passed to the child through
  `_omnifetch_child_config`; the policy itself is omnifetch-owned.
- The engine is built with Jasa's `normalize_url` as its cache identity,
  so a trailing slash, host casing, a default port, a fragment, or a dot
  segment share one fetch entry instead of buying the page again. Search
  dedup and the fetch cache therefore agree on what one page is. The URL
  sent to a provider is still the one asked for; only the key is folded.
  `_fetch_cache_identity` refuses the fold for a credential-bearing or
  non-ASCII-host URL and keys it verbatim: `normalize_url` drops
  password-only userinfo and uses IDNA 2003, both of which would let one
  origin's content answer another's request. Do not pass `normalize_url`
  to the engine directly.
- Successful individual grounding outputs use `jasa:grounding:v2:` records on
  that backend, keyed on the canonical fetch URL and the query rather than on
  fetched content, so a page re-rendered by a different fetch provider still
  reuses its snippet. Every fallback is excluded. The lifetime starts at
  `JASA_GROUNDING_CACHE_TTL_SECONDS` and is clamped down by the fetch TTL, and
  by the volatile fetch TTL for homepages, so a snippet never outlives the page
  it describes.
- Provider-native usage responses use `jasa:usage:v1`, default to a 10-minute
  TTL, redact credentials/account identities, and refresh asynchronously after
  search/fetch requests. Tavily, Firecrawl, GitHub, ScrapingAnt, ScrapingBee,
  SerpAPI, Serper, Diffbot, Kimi, Linkup, You.com, Olostep, ScrapeGraphAI,
  Scrapeless, Scrapfly, Scrappey, SociaVault, Spider, and Supadata are the
  currently integrated usage probes.
- Grounding contexts share one process-local flight registry. Misses on the
  same canonical page and query coalesce through the leader's cache write --
  including two renderings of that page from different fetch providers, whose
  effective LLM messages differ. Waiters keep their own per-URL deadline and
  retry independently after non-cacheable output.
- The grounding LLM call is an ordered waterfall declared in
  `src/jasa/grounding/waterfall.yaml` (override with
  `JASA_GROUNDING_WATERFALL_PATH`). The already-billed fetch is spent once and
  reused across tiers; grounding is enabled when any tier's credential is set.
  Accepted output is keyed by the whole chain, so editing the chain starts a
  fresh grounding cache namespace.
- One request budget, split before it is spent. `JASA_SEARCH_TIMEOUT_MS` is the
  whole-request deadline when a caller names none; `JASA_SEARCH_FANOUT_TIMEOUT_MS`
  bounds the fan-out inside it so the stages behind it inherit time rather than
  whatever the slowest provider left. A fan-out handed the entire budget starves
  grounding, which then pays for LLM calls it has no time to finish.
- Once any provider has produced usable ranked rows, later grounding budget
  exhaustion is a successful degraded search, never `SearchError`. Attempted
  rows keep their aggregate snippets, become `snippet_source=fallback`, and
  report `fallback:pipeline_timeout`; the transient outcome blocks caching.
- A zero deadline means expired, not absent. Only `None` waives one. Treating
  zero as "no deadline" hands an unbounded fan-out to precisely the caller whose
  budget just ran out.
- Grounding never discards a URL that finished. The stage owns its deadline and
  harvests each worker separately; an expired budget cancels only the workers
  still running. A page fetch and at least one LLM completion are billed before
  any deadline can fire, so abandoning finished work spends money and returns
  nothing. The one exception is the hard backstop in
  `_ground_under_backstop`, which sits beyond the stage deadline and does
  forfeit everything: it exists only for a stage that fails to honour the
  deadline it was given, and reaching it is a bug rather than a normal outcome.
  Anything that could make it reachable in normal operation -- an unbounded
  drain, a grace period wider than the gap -- is the defect, not the backstop.
- The 58-second default is sized against the request timeout MCP clients ship
  with, commonly 60 seconds, leaving the 25-second fan-out, roughly 30 seconds
  for the configured grounding waterfall, and response overhead. The client
  timeout is the real ceiling: a client that gives up
  mid-request abandons everything the server already paid for, which is worse
  than returning what finished. Raise `JASA_SEARCH_TIMEOUT_MS` only alongside
  the client's own timeout.
- `web_search` returns `JASA_SEARCH_MAX_RESULTS` rows (50 by default) plus tail
  rescues. Only the first `JASA_GROUNDING_TOP_N` rows (20 by default) are
  fetched and grounded. REST `/search` defaults to 20; `/researcher` returns
  10.
- In the MCP `web_search` response specifically, every result carries a
  `snippet_source` (`aggregated`, `grounded`, or `fallback`) and the response
  carries a `grounding` block. A successful response says nothing about whether
  grounding ran; an MCP client must never have to infer it. The REST shapes are
  deliberately narrower -- `/search` returns `link`/`title`/`snippet` and
  `/researcher` returns `href`/`body` -- and neither carries these fields.
- Omnifetch's `say_hello` is disabled unless `JASA_EXPOSE_HELLO=true`.
- Parent `/health` wins; child standalone `/web_fetch` is always off in composed
  mode.
- A fetch provider's `NOT_FOUND`, including Tavily extraction 404s, is
  provider-local evidence and never aborts the general waterfall. Exhausted MCP
  fetches return structured `not_found` or `unavailable` outcomes with attempt
  evidence; invalid inputs remain tool errors. REST and internal grounding keep
  provider exceptions for their existing status/fallback boundaries.
- Unit tests require 100% line and branch coverage. Live provider calls are
  manual and never part of normal pytest/CI.

## Change routing

### Add or modify a search provider

1. Read every file in `src/jasa/search/providers/` and its `AGENTS.md`.
2. Implement the adapter using the shared HTTP/error base.
3. Add it to `PROVIDER_CLASSES` in canonical order. Its non-`None` secret is
   derived into `KNOWN_SEARCH_SECRET_ENVS`; add optional knobs to `setting_envs`.
4. Add focused request/mapping/error/redaction tests.
5. Add secrets/settings to `.env.example`; a credential-free provider adds no
   placeholder. The parity test must remain an exact set equality.
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

### Debug a provider quickly

- Missing from `/health`: verify the exact provider-native secret name and, for
  multi-secret fetch providers, that every required value is non-empty.
- Present but failing: start with the provider's focused test and shared
  `ProviderError` category. Reject whitespace-only or empty quoted credentials
  yourself: the current shared snapshot can list them as active before an
  adapter rejects them or sends a blank key. Auth and rate limits intentionally
  do not retry.
- Search request is wrong: inspect that adapter's operator strategy; adapters do
  not all consume operators the same way.
- Fetch provider is never tried: inspect omnifetch's breaker/waterfall topology,
  active names, earlier successful tiers, and `skip_providers` filtering.
- Reproduce one live call with `scripts/run_provider_integration.py`; it verifies
  isolation before spending the one request.

### Modify fetch behavior

Do not copy or patch omnifetch internals in Jasa. Work in the omnifetch
repository, test and merge it, run `uv lock --upgrade-package omnifetch`, then
run Jasa's entire suite and Docker integration. `pyproject.toml` tracks
omnifetch's GitHub `main`; `uv.lock` freezes the resolved commit for each build.
Changes limited to how Jasa composes the child belong in `src/jasa/server.py`.

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

### Live Docker verification before every PR

A green unit suite is necessary and not sufficient. Mocked providers cannot
show a stage that runs out of budget, a page that never returns, a tier that
answers with an empty body, or a generation cut off at its token ceiling --
every one of those has shipped past a fully passing suite. Before opening a
PR, run the container and drive it yourself:

```bash
docker compose up -d --build --wait
curl -fsS http://127.0.0.1:8000/health
```

Then issue **2 varied live queries** through `mcp-inspector --cli` against
`http://127.0.0.1:8000/mcp`, using default tool arguments. Every live
`web_search` fans out to every configured paid provider and then buys up to
`top_n` grounding completions, so this sweep costs real money on every run and
is deliberately small.

Spend those two calls on the axis the change actually touches -- typically one
happy path and one failure path -- rather than two that vary decoratively.
Grounding behavior is content-dependent, so when a change touches grounding
itself, make the pair differ in content class (for instance one short English
technical query and one long non-English prose query).

A fetch-side change does not need the search sweep at all: drive `web_fetch`,
or the provider directly, which costs a single provider credit instead of a
full fan-out.

For each call, check:

- `grounding.grounded` against `grounding.attempted`, and what
  `grounding.outcomes` names for the shortfall;
- `snippet_source` on every result (`grounded`/`aggregated`/`fallback`);
- wall-clock time against the client's own request timeout, which is commonly
  60 seconds and is a hard ceiling no server budget can exceed;
- container logs for `Grounding complete`, tier advances, and token-ceiling
  warnings;
- snippet endings, for text stopping mid-sentence or a missing `Coverage:`
  line.

Report the query set and the per-query results in the PR. "Tests pass" is not
a substitute for having watched it work.

## Documentation maintenance

- README is for users/operators and should explain value before internals.
- AGENTS files are precise navigation indexes, not general prose. Keep paths,
  constants, provider counts, and commands synchronized with code.
- When a file moves or a behavior changes, update the nearest `AGENTS.md` in the
  same commit.
- Do not create speculative roadmap docs. Document only current behavior.
- Generated caches, `.venv`, `.git`, `dist`, `tmp`, and `trash` are not project
  directories and do not receive AGENTS files.
