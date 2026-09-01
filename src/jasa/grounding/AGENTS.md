# AGENTS.md — `src/jasa/grounding/`

Grounding replaces aggregated engine snippets with query-specific evidence
written from fetched page content. It is optional, MCP-search only, and reuses
the in-process omnifetch engine.

## Files

| File                | Responsibility                                                                                    |
| ------------------- | ------------------------------------------------------------------------------------------------- |
| `cache.py`          | URL+query identities, strict v2 records, per-URL TTL, fail-open reads/writes, bounded cache logs.  |
| `flights.py`        | Process-local miss registry, cancellation-safe leader ownership, and shielded waiter primitive.   |
| `service.py`        | Bounded top-N workers, per-URL deadline, fetch, LLM waterfall, outcome classification, stats.      |
| `waterfall.py`      | Strict YAML tier document, settings inheritance, credential resolution, chain semantics.          |
| `waterfall.yaml`    | The shipped ordered tier chain; swappable via `JASA_GROUNDING_WATERFALL_PATH`.                    |
| `detectors.py`      | Pre-LLM junk detection, post-LLM sentinel detection, unbalanced-fence repair.                     |
| `prompts.py`        | Loads the packaged system prompt and builds the user message.                                     |
| `system_prompt.txt` | Exact snippet-writing contract; SHA-256 pinned by tests.                                          |
| `__init__.py`       | Package marker.                                                                                   |

## Pipeline

For each top result, `ground_results()`:

1. acquires the concurrency semaphore;
2. applies the per-URL timeout, clamped to the shared stage deadline, to the
   whole pipeline;
3. calls omnifetch `execute_web_fetch` with the shared engine;
4. rejects bodies shorter than 50 chars or matching junk patterns;
5. truncates page content to the configured character budget;
6. joins or leads the process-local flight for the canonical URL and query;
7. reads the strict grounding v2 cache as leader;
8. walks the credentialed waterfall on a miss, stopping at the first tier that
   returns text and capping each attempt by its own budget, by what remains,
   and by the minimum slice owed to each tier still queued behind it;
9. caps the snippet at 2200 chars (2000 of body plus the mandatory
   Coverage line), reads the sentinel verdict from the model's own text,
   trims a cut generation to a whole sentence, and repairs a cut code fence;
10. writes only accepted output with the configured grounding TTL;
11. releases waiters after that write and preserves input order and outcomes.

`ground_results` then harvests every worker independently: completed URLs keep
their snippets and only the stragglers are cancelled. An optional progress
reporter emits the first completion, roughly quarterly milestones, and the
final completion; callback failures are debug-only and never change outcomes.

Durable fetch/junk/sentinel/empty fallbacks retain the aggregate snippet.
`llm_error`, `pipeline_timeout`, and `worker_rejected` are transient and block
the search cache write.

## Invariants

- `system_prompt.txt` changes require an intentional hash update in
  `tests/test_grounding.py` and careful output review.
- `cache.py` owns individual LLM reuse; `service.py` owns when the pipeline may
  read or write it. Do not reconstruct cache identities outside `cache.py`.
- Detector tables, prompt templates, system-prompt content, and accepted-output
  constants invalidate the fingerprint automatically. Algorithm-only semantic
  changes require bumping `GROUNDING_SEMANTICS_VERSION` and, when prior
  individual outputs are not reusable, `GROUNDING_CACHE_SEMANTICS_VERSION`.
- Page content is data, never instructions. Preserve the prompt-injection
  defense in the prompt.
- Ambiguous junk phrases fire only on short bodies to avoid prose false
  positives; tight patterns always fire.
- Deadlines cover fetch plus LLM, not just the HTTP generation call. They are
  sized generously on purpose: the fetch and at least one completion are billed
  before any deadline can fire, so a tight budget buys work and discards it.
- An expired stage budget must never discard a finished URL. `ground_results`
  owns its deadline and harvests each worker separately; the caller passes a
  deadline down instead of wrapping the stage in a timeout. Wrapping it made a
  single slow URL throw away every snippet its siblings had paid an LLM for.
- The stage budget is re-checked after the worker semaphore is acquired, not
  only before the queue is joined. With `concurrency` below `top_n` a worker
  can wait out most of the stage behind its siblings, so a worker reaching the
  front with less than `MIN_WORKER_BUDGET_SECONDS` left declines instead of
  paying for a fetch it cannot use.
- No tier may consume the whole remaining budget while tiers are still queued
  behind it. The first tier inherits an environment timeout sized for a lone
  endpoint and would otherwise make the fallbacks unreachable in exactly the
  outage they exist to cover. The reserve is advisory: when too little remains
  for every tier, the current tier still gets everything left.
- The output list corresponds only to the configured top-N; the caller merges
  results by URL and leaves the rest unchanged.
- Grounding failures must never erase a valid search-engine snippet. They do
  relabel it `snippet_source: "fallback"`, which changes no snippet text and
  lets a client separate a failed attempt from a result grounding never
  reached.
- A complete final `Coverage:` line survives the output cap. When an overlong
  body contains a code fence, cap and repair the body first, then append the
  intact line; never slice the combined output through its final contract.
- Every grounded search logs one summary line naming the per-outcome counts,
  escalated to WARNING when nothing was grounded. A response body alone cannot
  distinguish a grounded search that produced nothing from an ungrounded one.
- Grounding cache keys are `jasa:grounding:v2:` plus SHA-256 of canonical JSON.
  They cover the canonical fetch URL, the query, a prompt fingerprint (template,
  truncation marker, system-prompt digest, content cap), the whole ordered
  `(base_url, model)` chain, generation constants, and post-processing
  semantics; API keys and per-tier names/timeouts are absent.
- The identity keys on the page, never on the page's bytes. The same URL
  reaches this stage as different markdown whenever a different provider wins
  the fetch race, and content keying made every such rendering a separate
  entry -- so reordering the fetch waterfall silently orphaned every accepted
  snippet and re-bought the LLM call behind it. Do not add fetched content, the
  fetched title, or the effective user message back into the identity; all
  three are renderings, not identities.
- The URL in the identity must come from
  `omnifetch.tools.fetch.cache_identity_url`, the same helper the fetch cache
  keys on. Keying on the raw URL instead lets the two caches fold spellings
  differently, which reintroduces misses on pages the fetch cache is serving.
- The query stays in the identity. A grounded snippet answers a question about
  a page, so sharing the fetch key outright would serve one query's snippet to
  another.
- Content keying supplied invalidation for free; the TTL now carries it. The
  aim is to keep a snippet from long outliving the page it describes -- bounded,
  not eliminated, per the invariant below -- so `grounding_cache_ttl_seconds`
  clamps the configured grounding TTL down by the fetch TTL, and by the volatile
  fetch TTL for a homepage. At the shipped
  defaults neither clamp binds, but the TTLs are configured independently and
  inverting them would otherwise serve a snippet for a page the deployment has
  already stopped believing in. Every clamp only ever shortens, matching
  omnifetch's reading of its own TTL pair.
- The clamp bounds a snippet's *duration*, not its absolute expiry, because the
  fetch entry's remaining lifetime is not observable from here -- omnifetch
  returns a `FetchResponse`, not an expiry. A snippet written from a fetch hit
  that was nearly expired therefore starts a full interval of its own, so the
  worst case is roughly twice the intended freshness window: a homepage snippet
  can describe content up to 600 seconds old against a 300-second fetch
  lifetime. Closing that needs omnifetch to expose entry expiry so the write can
  cap against an absolute deadline; do not close it by putting content back into
  the identity, which is the defect this design exists to remove.
- That clamp governs the grounding entry only. The enclosing search cache holds
  a whole `SearchOutcome` for `JASA_SEARCH_CACHE_TTL_SECONDS` and short-circuits
  ahead of this stage, so an identical repeated query still reuses its snippets
  for that longer window. This predates URL keying -- the search cache returns
  before grounding runs regardless of how grounding is keyed -- and changing it
  is a search-cache decision, not a grounding one.
- Two distinct URLs returning identical bytes no longer coalesce. That fold was
  incidental to content keying and unsafe: it let one page's snippet answer for
  another whenever their bodies coincided, as short error pages routinely do.
- The chain, not the answering tier, is the cache identity: any tier may serve
  a request, so its accepted output is reusable, and a swapped chain starts a
  fresh namespace. Tier names and timeouts cannot change accepted text, so they
  must never enter an identity.
- A tier advances on transport failure, non-2xx status, an in-body `error`, an
  unreadable response shape, text that is blank once stripped, or an explicit
  `stop` response whose normal snippet omits the required final `Coverage:`
  line. A sentinel never advances; it is a judgment about the page. An
  exhausted chain reports the last tier's failure kind, so `llm_error` and
  `llm_empty` keep their existing meaning.
- A sentinel verdict is read from the model's own text, before any trimming.
  Substring sentinel matching applies only to short snippets, so shortening a
  long answer that merely quotes a bracketed phrase would manufacture a verdict
  the model never gave from page content an author controls.
- Cached output rechecks only whether the accepted snippet is a standalone
  sentinel. Framed-substring detection belongs to the raw model response;
  rerunning that length-sensitive heuristic after capping or whitespace
  normalization can manufacture a verdict that the model never gave.
- Whitespace-only output is empty output. Accepting it would replace a valid
  aggregated snippet with blanks, which the no-erasure invariant forbids.
- Each attempt is wrapped in `asyncio.timeout` as well as passed to httpx,
  because the httpx budget bounds each connection phase separately and would
  let one slow tier consume the budget the tiers behind it need.
- An effective `base_url` must be an absolute http(s) URL with a connectable
  authority, no query or fragment, and no userinfo. The check runs on the
  inherited value, so a bad `JASA_GROUNDING_LLM_BASE_URL` fails as loudly as a
  bad file entry. Rejection messages name the tier and never repeat the URL:
  the values most likely to be rejected are the ones carrying a credential.
- `waterfall.py` never sees a credential. Tiers name an environment variable;
  `resolve_grounding_waterfall` drops uncredentialed tiers and returns the keys
  separately. Do not add a secret to `GroundingTier`, and reject one inlined in
  a `base_url`: that field is hashed into the cache identity and the search
  fingerprint.
- Configuration is static, credentials are live. The chain is parsed once in
  `_build_parent_server` and passed explicitly to the registrars; only the
  credential filter re-reads `os.environ`, per request and per status read.
  A credential exported after boot therefore joins the chain on the next
  search, which is what `JASA_GROUNDING_MODE=auto` means. Docs must describe
  per-request resolution, never boot-time freezing.
- A malformed, unreadable, or unversioned waterfall file raises at composition.
  Grounding must never silently disable itself because of a bad config file.
- Grounding records are strict, versioned, digest-bound, and contain only the
  irreversible identity digest and the accepted snippet. Queries, URLs, fetched
  content, titles, effective messages, and prompts are not retained. The title
  reaches the caller from the live fetch instead; persisting it once bounded it
  at 2000 characters, so a page with a longer title failed validation on every
  write and repeated its LLM call forever.
- Only the `grounded` path writes. Fetch/junk/sentinel/empty/error/timeout and
  worker-rejected outcomes remain misses on the next call.
- Cache reads and writes fail open. A write shares the absolute per-URL deadline,
  but write expiry returns the accepted result rather than reclassifying it as a
  pipeline timeout.
- The pre-claim read plus leader race-closing reread share at most 250
  milliseconds and half the remaining per-URL budget. Best-effort writes
  release the fetch/LLM semaphore, use a separate registration-owned
  concurrency bound shared across searches, and retain the same absolute
  deadline.
- A grounding cache hit still reports the normal `grounded` outcome so stats and
  the complete-search poisoning guard retain their meaning.
- One registration-owned `GroundingFlightRegistry` is shared across search
  requests. Misses on the same page and query issue one LLM call when the
  leader writes a reusable result. Waiters release worker slots, shield the leader completion,
  keep their original deadline, and reread cache before electing a new leader.
- Error, empty, sentinel, timeout, cancellation, rejected-write, and unexpected
  leader paths release every waiter and remove the flight. A non-cacheable
  result is never handed directly to waiters; they retry independently.
- Grounding cache logs and metrics contain only bounded event/error-type fields.
  Never include query, content, snippet, API key, or hash key material.
- `grounding_semantic_fingerprint()` hashes detector/prompt/version semantics,
  model, base URL, content cap, top-N, and generation constants for the final
  search-cache key; it never accepts the API key.

## Tests

```bash
conda run -n base uv run pytest \
  tests/test_grounding.py tests/test_grounding_service.py \
  tests/test_grounding_coalescing.py tests/test_grounding_flight_failures.py \
  tests/test_grounding_flight_deadlines.py tests/test_grounding_waterfall.py \
  tests/test_service.py
```

Then run the entire suite. Waterfall composition and credential resolution
reach the server, REST, startup-validation, and cache-matrix paths, and the
100% line and branch gate only means anything over the whole run:

```bash
conda run -n base uv run pytest
```
