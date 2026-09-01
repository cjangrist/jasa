# AGENTS.md — `src/jasa/tools/`

Tools are thin transport adapters. Business logic belongs in execution
services so MCP and REST cannot drift.

## Files

- `web_search.py` calls `run_search()` and formats an MCP dictionary. It keeps
  `JASA_SEARCH_MAX_RESULTS` ranked rows (50 by default) plus tail rescues and
  applies `include_snippets` after retrieval, preserving provider successes,
  failures, timings, scores, snippet source, the grounding report, and
  truncation metadata.
- `__init__.py` marks the adapter package.

`web_search` registration and validation live in `../server.py` and
`../schemas.py`. `web_fetch` is mounted from omnifetch; do not add a duplicate
Jasa adapter. Its MCP-only terminal exhaustion response has `status` equal to
`not_found` or `unavailable` and includes provider-attributed evidence; invalid
inputs remain tool errors. REST and grounding retain their own error mapping.

The parent registration advertises `web_search` as read-only, non-destructive,
idempotent, and open-world through MCP `ToolAnnotations`.

## Invariants

- `JASA_SEARCH_MAX_RESULTS` is MCP-only. Do not change algorithm or REST
  defaults when changing it.
- Cache stores full results before formatting, so snippet omission and
  truncation never poison another caller's response.
- Public field names are part of the MCP contract.
- The `grounding` block is always present, including when grounding never ran,
  and `snippet_source` is always set on every result. A caller must never have
  to infer from a successful response whether grounding happened: `attempted`
  is zero only when the stage did not run, and `outcomes` names the reason for
  every shortfall.
- Keep wrapper functions async and dependency-injected for in-memory tests.
- This layer formats search-cache output. Omnifetch caches successful
  `web_fetch` responses before this transport boundary.

## Tests

`tests/test_web_search.py` owns end-to-end formatting and truncation;
`tests/test_schemas.py`, `test_server.py`, and `test_composition.py` own tool
registration and validation.
