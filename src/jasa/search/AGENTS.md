# AGENTS.md — `src/jasa/search/`

This is Jasa's core: provider dispatch, selective retry, deterministic merge,
URL deduplication, snippet selection, quality filtering, optional grounding,
and cache orchestration. Pure algorithms are kept separate from network code.

## File map and call graph

```text
service.py run_search
├── cache/base.py read key
├── fanout.py dispatch_to_providers
│   ├── providers/<adapter>.py search
│   └── retry.py retry_with_backoff
├── ranking.py rank_and_merge
│   ├── urls.py normalize_url
│   └── snippets.py collapse_snippets
├── grounding/service.py ground_results (optional)
└── cache/base.py completeness gate + write
```

| File           | Responsibility                                                                       |
| -------------- | ------------------------------------------------------------------------------------ |
| `service.py`   | Single MCP/REST execution path, errors, cache read/write, grounding insertion.       |
| `fanout.py`    | Concurrent provider tasks, global deadline, cancellation cleanup, immutable outcome. |
| `retry.py`     | One randomized backoff retry for transient provider errors.                          |
| `ranking.py`   | RRF merge, quality filter, top-N truncation, tail rescue, result models.             |
| `urls.py`      | WHATWG-like URL canonicalization for dedup keys.                                     |
| `snippets.py`  | Entity cleanup, n-gram scoring, near-duplicate removal, sentence merge.              |
| `operators.py` | Parse advanced syntax and re-render structured search parameters.                    |
| `providers/`   | 14 adapters, all env-gated by provider-native secrets.                              |
| `__init__.py`  | Declares the pure-algorithm package boundary.                                        |

## Fan-out invariants

- Start every active provider concurrently.
- Preserve registry order in final maps and success/failure arrays; completion
  order must not change ranking.
- Give each provider 30 results by default and at most one transient retry.
- A caller deadline is global. Cancel and await pending tasks, mark each once
  with a structured deadline flag plus the exact deadline message, and never
  let late tasks mutate the result.
- Cancellation of the whole dispatch propagates after cleaning child tasks.
- Unexpected exceptions are attributed to one provider, not allowed to crash
  siblings.

## Ranking invariants

- Stable-sort a provider's results by native score, descending.
- A provider contributes at most once per normalized URL.
- RRF constant is 60; contribution is `1/(60 + rank)` with rank starting at 1.
- First-seen provider supplies title/URL; all distinct snippets/providers merge.
- Quality filter drops scores below 0.01 and thin single-provider snippets below
  50 chars; empty-snippet results remain eligible.
- Tail rescue only admits strong results from hosts absent in the top set.
- MCP truncation is 30 in `tools/web_search.py`; algorithm default remains 20.

## Cache and grounding

Cache stores the full ranked outcome before transport truncation and snippet
omission. It never stores no-provider, all-failed, partial-provider, or transient
grounding outcomes. Grounding runs after rank/quality and before the write. The
search service writes with `JASA_SEARCH_CACHE_TTL_SECONDS` (36 hours by default)
and owns only search keys; omnifetch owns successful fetch keys on the same
injected backend, while grounding owns success-only LLM-output keys on it.
Grounding cache hits remain normal `grounded` outcomes, so a complete grounded
search is still eligible for the outer search cache. Search v3 keys include
exact query, both mode flags, ordered active providers, and grounding semantics;
strict versioned records turn legacy, malformed, extra-field, wrong-type, and
identity-mismatched data into misses.

Composition owns one `SearchRuntime` and `SearchFlightRegistry` shared by MCP,
`/search`, and `/researcher`. After an initial miss, one caller leads each exact
identity while shielded waiters await completion and then reread the cache. A
newly elected leader rechecks the cache while holding its flight so a stale
yielding miss cannot escape coalescing. A leader always releases the flight,
including on cancellation or unexpected errors. If its outcome is not cacheable
or its write fails, waiters compete to lead a fresh search rather than sharing
that outcome. This is in-process coalescing only; Redis does not make flights
distributed. Each caller captures one absolute budget before its first cache
read; cache I/O, coalesced waiting, and any later leader retry consume that same
budget rather than resetting it. An expired read fails the request, while an
expired write fails open so a completed provider outcome can return and release
its flight immediately. Caller deadline exhaustion, including during grounding,
is distinct from provider exhaustion and maps to a REST 504 rather than the
`all_failed` 502 boundary.

## Golden parity

Operator parsing, ranking, truncation, snippet selection, and URL normalization
are pinned to JSON fixtures under `tests/fixtures/golden/`. When semantics
change intentionally, update the algorithm and the relevant fixture together,
then explain the divergence from the source behavior.

## Focused tests

```bash
conda run -n base uv run pytest \
  tests/test_fanout.py tests/test_retry.py tests/test_service.py \
  tests/test_search_coalescing.py \
  tests/test_operators.py tests/test_ranking.py tests/test_snippets.py \
  tests/test_urls.py tests/test_web_search.py
```
