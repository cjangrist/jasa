# AGENTS.md — `src/jasa/usage/`

This package owns provider-native usage/quota snapshots. It never normalizes
vendor billing fields: successful upstream JSON dictionaries are returned under
`raw` after recursive credential and account-identity redaction.

## Files

- `base.py` — probe contract, bounded raw JSON request helper, and redaction.
- `runtime.py` — complete provider enumeration, concurrent collection, shared
  10-minute cache reuse, process-local refresh coalescing, and MCP middleware.
- `providers/` — one free provider usage adapter per module and pull request.
- `__init__.py` — public runtime exports.

## Invariants

- `/usage` lists every registered search and fetch adapter, even when missing,
  unconfigured, unsupported, or not implemented yet.
- `/usage` bounds the complete cache/probe/write path at 30 seconds and returns
  HTTP 504 on expiry; the shielded refresh may finish for later callers.
- The refresh task has its own 30-second deadline, so a hung cache read releases
  the singleton and a later request can retry without a restart. Cache writes
  have a one-second fail-open deadline after the local snapshot is complete.
- Normal search/fetch requests trigger refresh checks in the background and do
  not wait for usage APIs.
- One in-process task owns a miss. Redis/filesystem/memory stores the snapshot
  at `jasa:usage:v1` for `JASA_USAGE_CACHE_TTL_SECONDS` (600 by default).
- Cache records bind the ordered catalog and configured-provider set, so a
  differently configured process cannot reuse an incompatible snapshot.
- Provider calls run concurrently, fail independently, and never gate search or
  fetch execution.
- Probe failures log only the provider name and HTTP status or exception class;
  cleaned error data remains in the provider record.
- Shutdown cancels active refresh work and prevents a later cache miss from
  creating a task against already closing shared resources.
- Raw provider dictionaries retain their upstream field names and values except
  credentials and account identities, which become `[REDACTED]` recursively.
- GitHub's probe uses the unmetered authenticated rate-limit endpoint and
  exposes its provider-native resource quota dictionaries only for fetch.
- SerpAPI's free account endpoint supplies one shared search/fetch record with
  native plan, monthly usage, remaining-search, and hourly-rate fields.
- Serper's free account endpoint supplies its native balance and rate-limit
  fields for the search provider.
- ScrapingAnt's free usage endpoint supplies native plan, billing-window, and
  credit fields for the fetch provider.
- ScrapingBee's free usage endpoint supplies native credit, concurrency, and
  subscription-renewal fields for the fetch provider.
- Diffbot's free account endpoint supplies native plan, status, credit, and
  daily usage fields for the fetch provider.
- Kimi Code's free usage endpoint supplies native weekly and rolling-window
  quota fields for the fetch provider; fetch also requires Scrapfly.
- Linkup's free credit-balance endpoint supplies its native remaining balance
  for the shared search and fetch provider.
- You.com's free account-balance endpoint supplies the native billing-entity
  type and remaining credit balance in cents for search and fetch.
- Olostep's free credit-info endpoint supplies native credit-lot and active
  subscription fields for the fetch provider.
- ScrapeGraphAI's free credits endpoint supplies native balance, plan, and
  crawl/monitor job-quota fields for the fetch provider.
- Scrapeless's free user-info endpoint supplies native account balance and
  subscription plan fields for the fetch provider.
- Scrapfly's free account endpoint supplies native project limits,
  subscription details, and scrape/schedule/spider usage for fetch providers.
- Every provider integration gets its own PR, module, mocked request/redaction
  test, and registry entry.

## Tests

Run `conda run -n base uv run pytest tests/test_usage.py tests/test_rest.py
tests/test_server.py tests/test_composition.py` before the full suite.
