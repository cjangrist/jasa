# AGENTS.md — `src/jasa/usage/providers/`

Each module performs only a provider's documented free usage/quota call and
returns the cleaned raw JSON dictionary through `base.request_usage_json()`.

Add exactly one provider per pull request: create its module, register one
`UsageProbe` in `__init__.py`, document whether credentials are shared with the
runtime provider, and test the exact request plus redaction and error response.
Never substitute a metered search, fetch, or generation call for a missing
usage API.

## Current probes

| Module           | Credential                | Runtime sharing  | Free endpoint               |
| ---------------- | ------------------------- | ---------------- | --------------------------- |
| `tavily.py`      | `TAVILY_API_KEY`          | Search and fetch | `GET /usage`                |
| `firecrawl.py`   | `FIRECRAWL_API_KEY`       | Search and fetch | `GET /v2/team/credit-usage` |
| `github.py`      | `GITHUB_API_KEY`          | Fetch            | `GET /rate_limit`           |
| `scrapingant.py` | `SCRAPINGANT_API_KEY`     | Fetch            | `GET /v2/usage`             |
| `scrapingbee.py` | `SCRAPINGBEE_API_KEY`     | Fetch            | `GET /api/v1/usage`         |
| `serpapi.py`     | `SERPAPI_API_KEY`         | Search and fetch | `GET /account.json`         |
| `serper.py`      | `SERPER_API_KEY`          | Search           | `GET /account`              |
| `diffbot.py`     | `DIFFBOT_TOKEN`           | Fetch            | `GET /v4/account`           |
| `kimi.py`        | `KIMI_API_KEY`            | Fetch + Scrapfly | `GET /coding/v1/usages`     |
| `linkup.py`      | `LINKUP_API_KEY`          | Search and fetch | `GET /v1/credits/balance`   |
