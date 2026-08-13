# AGENTS.md — `tests/fixtures/golden/`

Golden JSON files pin pure search behavior against the ported TypeScript
semantics. Tests load every case and compare exact structure/order, using a
tight tolerance only for floating-point RRF scores.

## Fixture ownership

| Fixture                               | Owning code                   | Owning test         |
| ------------------------------------- | ----------------------------- | ------------------- |
| `operators.json`                      | `search/operators.py`         | `test_operators.py` |
| `url_normalization.json`              | `search/urls.py`              | `test_urls.py`      |
| `snippets.json`                       | `search/snippets.py`          | `test_snippets.py`  |
| `ranking_stability_scoreless.json`    | `search/ranking.py`           | `test_ranking.py`   |
| `ranking_scored_single_provider.json` | `search/ranking.py`           | `test_ranking.py`   |
| `ranking_dedup_merge.json`            | `search/ranking.py`           | `test_ranking.py`   |
| `ranking_quality_filter.json`         | `search/ranking.py`           | `test_ranking.py`   |
| `truncate_rescue_distinct_hosts.json` | `search/ranking.py`           | `test_ranking.py`   |
| `truncate_no_rescue_same_host.json`   | `search/ranking.py`           | `test_ranking.py`   |
| `grounded_system_prompt.txt`          | `grounding/system_prompt.txt` | `test_grounding.py` |

## Editing protocol

Read the entire fixture, owning implementation, and owning test first. Add the
smallest case that distinguishes the intended behavior. Preserve case names and
ordering where possible so diffs remain diagnostic. Prompt changes require the
packaged prompt and golden copy to remain byte-for-byte aligned and the pinned
SHA-256 to be intentionally updated.

## Tests

```bash
conda run -n base uv run pytest \
  tests/test_operators.py tests/test_urls.py tests/test_snippets.py \
  tests/test_ranking.py tests/test_grounding.py
```
