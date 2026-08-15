# AGENTS.md — `src/jasa/grounding/`

Grounding replaces aggregated engine snippets with query-specific evidence
written from fetched page content. It is optional, MCP-search only, and reuses
the in-process omnifetch engine.

## Files

| File                | Responsibility                                                                                    |
| ------------------- | ------------------------------------------------------------------------------------------------- |
| `cache.py`          | Hash-only LLM identities, strict v1 records, fail-open reads/writes, bounded cache logs.           |
| `service.py`        | Bounded top-N workers, per-URL deadline, fetch, Cerebras call, outcome classification, stats.      |
| `detectors.py`      | Pre-LLM junk detection, post-LLM sentinel detection, unbalanced-fence repair.                     |
| `prompts.py`        | Loads the packaged system prompt and builds the user message.                                     |
| `system_prompt.txt` | Exact snippet-writing contract; SHA-256 pinned by tests.                                          |
| `__init__.py`       | Package marker.                                                                                   |

## Pipeline

For each top result, `ground_results()`:

1. acquires the concurrency semaphore;
2. applies the per-URL timeout to the whole pipeline;
3. calls omnifetch `execute_web_fetch` with the shared engine;
4. rejects bodies shorter than 50 chars or matching junk patterns;
5. truncates page content to the configured character budget;
6. reads the strict grounding v1 cache by the exact effective LLM input;
7. calls the OpenAI-compatible Cerebras chat-completions endpoint on a miss;
8. caps the snippet at 2000 chars, repairs a cut code fence, and rejects
   sentinel responses;
9. writes only accepted output with the configured grounding TTL;
10. preserves input order and records one of nine outcomes.

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
- Deadlines cover fetch plus LLM, not just the HTTP generation call.
- The output list corresponds only to the configured top-N; the caller merges
  results by URL and leaves the rest unchanged.
- Grounding failures must never erase a valid search-engine snippet.
- Grounding cache keys are `jasa:grounding:v1:` plus SHA-256 of canonical JSON.
  They cover the exact user message, prompt digest, model endpoint/model,
  generation constants, and post-processing semantics; API keys are absent.
- Grounding records are strict, versioned, digest-bound, and contain only the
  irreversible identity digest, accepted snippet, and exact fetched title.
  Queries, fetched content, effective messages, and prompts are not retained.
- Only the `grounded` path writes. Fetch/junk/sentinel/empty/error/timeout and
  worker-rejected outcomes remain misses on the next call.
- Cache reads and writes fail open. A write shares the absolute per-URL deadline,
  but write expiry returns the accepted result rather than reclassifying it as a
  pipeline timeout.
- Cache reads have an independent 250-millisecond fail-open bound. Best-effort
  writes release the fetch/LLM semaphore before using the remaining deadline.
- A grounding cache hit still reports the normal `grounded` outcome so stats and
  the complete-search poisoning guard retain their meaning.
- Grounding misses are not coalesced; simultaneous identical inputs can each
  call the LLM.
- `grounding_semantic_fingerprint()` hashes detector/prompt/version semantics,
  model, base URL, content cap, top-N, and generation constants for the final
  search-cache key; it never accepts the API key.

## Tests

```bash
conda run -n base uv run pytest \
  tests/test_grounding.py tests/test_grounding_service.py tests/test_service.py
```
