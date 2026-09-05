# AGENTS.md — `src/jasa/search/providers/`

Seventeen search adapters normalize unrelated upstream APIs into
`SearchResult(title, url, snippet, source_provider, score?)`. The registry
adds adapters with a non-empty provider-native secret and preserves the
canonical tuple order used by deterministic fan-out and RRF.

## Registry and base

- `__init__.py` defines `PROVIDER_CLASSES`, `CANONICAL_PROVIDER_ORDER`,
  `KNOWN_SEARCH_SECRET_ENVS`, `KNOWN_SEARCH_SETTING_ENVS`, and
  `load_search_providers()`.
- `base.py` defines `SearchRequest` and abstract `SearchProvider`. `_fetch()`
  delegates to omnifetch's shared HTTP JSON helper so size limits, status
  mapping, transport errors, and the shared `ProviderError` taxonomy stay
  consistent. It also redacts raw and quote-stripped credentials.
- `setting_envs` declares optional non-secret deployment knobs. The registry
  resolves them from the same environment snapshot it gates on and passes them
  to the constructor; `_setting()` reads one with a fallback. A setting never
  activates an adapter, and `KNOWN_SEARCH_SETTING_ENVS` is derived from the
  registry so `.env.example` parity and the test isolation fixture cannot
  drift.

## Provider matrix

| Module / name                  | Secret               | API shape                    | Operator/domain behavior                                         |
| ------------------------------ | -------------------- | ---------------------------- | ---------------------------------------------------------------- |
| `tavily.py` / `tavily`         | `TAVILY_API_KEY`     | POST `/search`; native score | Parses operators; sends include/exclude domains structurally.    |
| `brave.py` / `brave`           | `BRAVE_API_KEY`      | GET Brave Web Search         | Re-renders all supported operators into `q`.                     |
| `kagi.py` / `kagi`             | `KAGI_API_KEY`       | POST JSON Search v1          | Maps domains, file type, and dates into a lens.                  |
| `exa.py` / `exa`               | `EXA_API_KEY`        | POST auto search with text   | Raw query; native domain arrays; dual auth headers.              |
| `firecrawl.py` / `firecrawl`   | `FIRECRAWL_API_KEY`  | POST v2 search               | Raw query; `success:false` is an API error; reads `data.web`.    |
| `perplexity.py` / `perplexity` | `PERPLEXITY_API_KEY` | Sonar chat completions       | Prefers `search_results`, falls back to citation URLs.           |
| `serpapi.py` / `serpapi`       | `SERPAPI_API_KEY`    | GET `google_light`           | Raw query; credential is a query parameter and must be redacted. |
| `linkup.py` / `linkup`         | `LINKUP_API_KEY`     | POST v1 standard search      | Native include/exclude domains; keeps text results only.         |
| `you.py` / `you`               | `YOU_API_KEY`        | POST JSON Search             | Raw query; joins snippet arrays; ignores news.                   |
| `parallel.py` / `parallel`     | `PARALLEL_API_KEY`   | POST advanced search         | Domain policy nested under `advanced_settings`.                  |
| `serper.py` / `serper`         | `SERPER_API_KEY`     | POST Google search           | Re-renders operators; maps organic results.                      |
| `claude.py` / `claude`         | `ANTHROPIC_AUTH_TOKEN` | POST Messages web-search tool | Domains become the tool's exclusive allow/block list; rest re-rendered. |
| `codex.py` / `codex`           | `OPENAI_API_KEY`     | POST Responses web-search tool | Domains become both `filters` lists; rest re-rendered; cited URLs de-tracked. |
| `zai.py` / `zai`               | `Z_AI_API_KEY`       | POST GLM chat completions with web-search tool | Re-renders every operator; upstream filters are accepted but ignored. Reads the tool's `web_search` array, caps `count` at 10, and caps generation at one token. |
| `ddgs.py` / `ddgs`             | `SCRAPFLY_API_KEY`   | GET Scrapfly scrape API      | Re-renders every operator; scrapes DuckDuckGo's html endpoint and decodes its redirect links. |
| `ollama.py` / `ollama`         | `OLLAMA_API_KEY`     | POST hosted web search       | Re-renders every operator; always requests the provider maximum of 10. |
| `keenable.py` / `keenable`     | `KEENABLE_API_KEY`   | POST Search API v1           | Uses one clean native site plus positive relative or calendar-valid absolute dates that resolve inside the live API's 1970-01-01 through 2149-06-05 window; preserves ambiguous/unsupported syntax in place, bypasses aggregate caching for relative dates, and requests the maximum of 50. Query assembly, token partitioning, and filter validation live in the adjacent `keenable_query.py`, `keenable_partition.py`, and `keenable_validation.py` modules. |

## Adapter contract

1. Set non-empty `name`, `base_url`, and timeout class attrs. Set `secret_env`
   to a provider-native name and declare optional deployment knobs in
   `setting_envs`.
2. Credentialed adapters call `_validated_key()` before any request.
3. Use `_fetch()` instead of direct `httpx` calls.
4. Map a missing result collection to an empty successful list unless the
   provider contract has an explicit failure flag.
5. Set `source_provider` to the registered name for every row.
6. Preserve native score when useful; otherwise leave it `None` and let array
   order carry rank.
7. Never include a credential in exception text or logs.
8. Do not retry inside an adapter; `fanout.py` owns retry policy.
9. Override `allows_cache()` only when the provider's response semantics move
   independently of the exact query, such as Keenable's relative date window.
   The search service passes the same `reference_datetime` to cache eligibility
   and the provider's `SearchRequest`; use it instead of sampling another clock.

## Fast failure diagnosis

| Symptom                       | Check first                                                              |
| ----------------------------- | ------------------------------------------------------------------------ |
| Not listed by `/health`       | Secret spelling, unset/empty value, and registry membership.             |
| Listed but auth fails         | Whitespace-only/empty quoted value, entitlement, and vendor header.      |
| `INVALID_INPUT` before HTTP   | Empty/quoted key validation or malformed tool input.                     |
| `API_ERROR` 401/403           | Credential validity, product entitlement, and vendor auth header.        |
| `RATE_LIMIT`                  | Provider quota; this category intentionally does not retry.              |
| `PROVIDER_ERROR`              | Transport/5xx path; fan-out retries it once.                             |
| Successful empty list         | Vendor response collection path or provider-side no-results response.    |
| Result filtered after success | Snippet length, score, URL, and quality filter in `ranking.py`.          |

The current shared secret snapshot tests raw truthiness. A whitespace-only or
empty quoted value can therefore appear in `/health` before request-time
validation rejects it or the upstream receives a blank key. Treat that state as
misconfiguration; never use quoted placeholders in `.env`.

## Adding a provider

- Read all adapters to select the closest request/response pattern.
- Add the module, then append the class in `__init__.py` at the intentional
  canonical position. Its non-`None` secret is derived into the known set.
- Add `.env.example` only for a secret or setting, and add the README provider
  entry with a documented default for every declared setting.
- Add focused tests for exact outbound request, mapping, empty/missing data,
  auth/rate-limit/5xx behavior as applicable, missing key, and redaction.
- Update the environment-isolation invariant through the registry source, not a
  second hand-maintained list.
- Verify a real provider with the one-request integration harness before adding
  its `INTEGRATION_CASES` entry.

## Tests

Provider tests are named `tests/test_provider_<name>.py`; registry invariants
are in `tests/test_providers.py`. Tavily has the broad reference error matrix.
Run all with:

```bash
conda run -n base uv run pytest tests/test_provider_*.py tests/test_providers.py
```
