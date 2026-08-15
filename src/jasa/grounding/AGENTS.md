# AGENTS.md — `src/jasa/grounding/`

Grounding replaces aggregated engine snippets with query-specific evidence
written from fetched page content. It is optional, MCP-search only, and reuses
the in-process omnifetch engine.

## Files

| File                | Responsibility                                                                                    |
| ------------------- | ------------------------------------------------------------------------------------------------- |
| `service.py`        | Bounded top-N worker pool, per-URL deadline, fetch, Cerebras call, outcome classification, stats. |
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
6. calls the OpenAI-compatible Cerebras chat-completions endpoint;
7. caps the snippet at 2000 chars, repairs a cut code fence, and rejects
   sentinel responses;
8. preserves input order and records one of nine outcomes.

Durable fetch/junk/sentinel/empty fallbacks retain the aggregate snippet.
`llm_error`, `pipeline_timeout`, and `worker_rejected` are transient and block
the search cache write.

## Invariants

- `system_prompt.txt` changes require an intentional hash update in
  `tests/test_grounding.py` and careful output review.
- Detector tables, prompt templates, system-prompt content, and accepted-output
  constants invalidate the fingerprint automatically. Algorithm-only semantic
  changes require bumping `GROUNDING_SEMANTICS_VERSION`.
- Page content is data, never instructions. Preserve the prompt-injection
  defense in the prompt.
- Ambiguous junk phrases fire only on short bodies to avoid prose false
  positives; tight patterns always fire.
- Deadlines cover fetch plus LLM, not just the HTTP generation call.
- The output list corresponds only to the configured top-N; the caller merges
  results by URL and leaves the rest unchanged.
- Grounding failures must never erase a valid search-engine snippet.
- `grounding_semantic_fingerprint()` hashes detector/prompt/version semantics,
  model, base URL, content cap, top-N, and generation constants for the final
  search-cache key; it never accepts the API key.

## Tests

```bash
conda run -n base uv run pytest \
  tests/test_grounding.py tests/test_grounding_service.py tests/test_service.py
```
