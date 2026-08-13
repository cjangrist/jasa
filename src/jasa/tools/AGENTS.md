# AGENTS.md — `src/jasa/tools/`

Tools are thin transport adapters. Business logic belongs in execution
services so MCP and REST cannot drift.

## Files

- `web_search.py` calls `run_search()` and formats an MCP dictionary. It keeps
  the top 30 ranked rows plus tail rescues and applies `include_snippets` after
  retrieval, preserving provider successes, failures, timings, scores, snippet
  source, and truncation metadata.
- `__init__.py` marks the adapter package.

`web_search` registration and validation live in `../server.py` and
`../schemas.py`. `web_fetch` is mounted from omnifetch; do not add a duplicate
Jasa adapter.

## Invariants

- `DEFAULT_MCP_TOP_N = 30` is MCP-only. Do not change algorithm or REST defaults
  when changing it.
- Cache stores full results before formatting, so snippet omission and
  truncation never poison another caller's response.
- Public field names are part of the MCP contract.
- Keep wrapper functions async and dependency-injected for in-memory tests.
- This layer formats the search cache output; it does not cache `web_fetch`.

## Tests

`tests/test_web_search.py` owns end-to-end formatting and truncation;
`tests/test_schemas.py`, `test_server.py`, and `test_composition.py` own tool
registration and validation.
