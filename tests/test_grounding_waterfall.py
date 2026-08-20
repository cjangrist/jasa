"""Grounding waterfall: YAML contract, credential resolution, tier advance."""

from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import pytest
import respx

from jasa.cache.memory import MemoryCache
from jasa.config import GroundingSettings
from jasa.grounding.cache import grounding_cache_identity
from jasa.grounding.flights import GroundingFlightRegistry
from jasa.grounding.service import (
    _extract_tier_snippet,
    _run_grounding_waterfall,
    ground_results,
    GroundingContext,
    GroundingTierError,
)
from jasa.grounding.waterfall import (
    grounding_chain_semantics,
    grounding_credential_envs,
    GroundingChain,
    load_grounding_waterfall,
    resolve_grounding_waterfall,
    waterfall_path,
)
from jasa.search.ranking import RankedWebResult
from tests.conftest import resolved_waterfall, tier

_PRIMARY = "https://primary.example/v1"
_BACKUP = "https://backup.example/v1"
_LAST = "https://last.example/v1"
_PRIMARY_URL = f"{_PRIMARY}/chat/completions"
_BACKUP_URL = f"{_BACKUP}/chat/completions"
_LAST_URL = f"{_LAST}/chat/completions"
_CONTENT = "Real page content that clears the minimum. " * 5


class _FetchResult:
    def __init__(self, content: str, title: str = "Title") -> None:
        self.content = content
        self.title = title


def _result(url: str = "https://example.com/a") -> RankedWebResult:
    return RankedWebResult("t", url, ["agg"], ["p"], 0.1)


def _ok(text: str | None) -> httpx.Response:
    return httpx.Response(
        200, json={"choices": [{"message": {"content": text}}]}
    )


def _chain(*envs: str) -> GroundingChain:
    urls = (_PRIMARY, _BACKUP, _LAST)
    names = ("primary", "backup", "last")
    return tuple(
        tier(names[index], urls[index], f"model-{index}", api_key_env=env)
        for index, env in enumerate(envs)
    )


def _context(
    chain: GroundingChain,
    cache: MemoryCache,
    *keys: str,
    settings: GroundingSettings | None = None,
) -> tuple[GroundingContext, httpx.AsyncClient]:
    resolved = settings or GroundingSettings()
    client = httpx.AsyncClient()
    context = GroundingContext(
        engine=object(),
        client=client,
        cache=cache,
        cache_write_semaphore=asyncio.Semaphore(resolved.concurrency),
        flights=GroundingFlightRegistry(),
        waterfall=resolved_waterfall(chain, *keys),
        config=resolved,
    )
    return context, client


@pytest.fixture
def fetch_once(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Count fetches so a waterfall never re-pays for the same page."""
    fetched: list[str] = []

    async def fake_fetch(engine: object, url: str) -> _FetchResult:
        fetched.append(url)
        return _FetchResult(_CONTENT)

    monkeypatch.setattr("jasa.grounding.service.execute_web_fetch", fake_fetch)
    return fetched


def _write_waterfall(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    return path


def test_packaged_waterfall_declares_the_shipped_chain() -> None:
    chain = load_grounding_waterfall(GroundingSettings())

    assert [entry.name for entry in chain] == [
        "cerebras",
        "luna",
        "haiku",
        "glm",
    ]
    assert [entry.model for entry in chain] == [
        "gpt-oss-120b",
        "gpt-5.6-luna",
        "claude-haiku-4-5-20251001",
        "glm-5.3",
    ]
    assert [entry.api_key_env for entry in chain] == [
        "CEREBRAS_API_KEY",
        "OPENAI_API_KEY",
        "OPENAI_API_KEY",
        "OPENAI_API_KEY",
    ]
    assert chain[0].base_url == "https://api.cerebras.ai/v1"
    assert {entry.base_url for entry in chain[1:]} == {
        "https://ai.angrist.net/v1"
    }


def test_first_tier_inherits_the_llm_settings() -> None:
    settings = GroundingSettings(
        llm_base_url="https://elsewhere.example/v1/",
        llm_model="inherited-model",
        llm_timeout_ms=12345,
    )

    chain = load_grounding_waterfall(settings)

    assert chain[0].base_url == "https://elsewhere.example/v1"
    assert chain[0].model == "inherited-model"
    assert chain[0].timeout_ms == 12345
    assert chain[1].model == "gpt-5.6-luna"
    assert chain[1].timeout_ms == 20000


def test_configured_path_replaces_the_packaged_chain(tmp_path: Path) -> None:
    path = _write_waterfall(
        tmp_path / "wf.yaml",
        "version: 1\n"
        "tiers:\n"
        "  - name: only\n"
        "    api_key_env: OTHER_KEY\n"
        "    base_url: https://only.example/v1\n"
        "    model: only-model\n",
    )
    settings = GroundingSettings(waterfall_path=str(path))

    assert waterfall_path(settings) == path
    chain = load_grounding_waterfall(settings)
    assert [entry.name for entry in chain] == ["only"]
    assert chain[0].api_key_env == "OTHER_KEY"


def test_blank_configured_path_uses_the_packaged_file() -> None:
    settings = GroundingSettings(waterfall_path="   ")

    assert waterfall_path(settings).name == "waterfall.yaml"


def test_unreadable_waterfall_fails_startup(tmp_path: Path) -> None:
    settings = GroundingSettings(waterfall_path=str(tmp_path / "missing.yaml"))

    with pytest.raises(ValueError, match="could not be read: FileNotFound"):
        load_grounding_waterfall(settings)


def test_unparseable_waterfall_fails_startup(tmp_path: Path) -> None:
    path = _write_waterfall(tmp_path / "bad.yaml", "version: 1\ntiers: [oops\n")
    settings = GroundingSettings(waterfall_path=str(path))

    with pytest.raises(ValueError, match="could not be read: ParserError"):
        load_grounding_waterfall(settings)


@pytest.mark.parametrize(
    "body",
    [
        "version: 2\ntiers:\n  - name: a\n    api_key_env: K\n",
        "version: 1\ntiers: []\n",
        "version: 1\ntiers:\n  - name: a\n",
        "version: 1\ntiers:\n  - name: a\n    api_key_env: K\n    extra: 1\n",
        "version: 1\ntiers:\n  - name: ''\n    api_key_env: K\n",
        "version: 1\ntiers:\n  - name: a\n    api_key_env: K\n"
        "    timeout_ms: 10\n",
        "tiers:\n  - name: a\n    api_key_env: K\n",
    ],
)
def test_invalid_waterfall_document_fails_startup(
    tmp_path: Path, body: str
) -> None:
    path = _write_waterfall(tmp_path / "invalid.yaml", body)
    settings = GroundingSettings(waterfall_path=str(path))

    with pytest.raises(ValueError, match="is not a valid v1 document"):
        load_grounding_waterfall(settings)


@pytest.mark.parametrize(
    ("base_url", "reason"),
    [
        ("https://", "needs an absolute http"),
        ("ai.angrist.net/v1", "needs an absolute http"),
        ("ftp://host/v1", "needs an absolute http"),
        ("/v1", "needs an absolute http"),
        ("https:///v1", "needs an absolute http"),
        ("https://host/v1?trace=1", "no query or fragment"),
        ("https://host/v1#anchor", "no query or fragment"),
        ("https://host/v1?a=1#b", "no query or fragment"),
    ],
)
def test_unreachable_tier_endpoint_fails_startup(
    tmp_path: Path, base_url: str, reason: str
) -> None:
    path = _write_waterfall(
        tmp_path / "wf.yaml",
        "version: 1\n"
        "tiers:\n"
        "  - name: broken\n"
        "    api_key_env: K\n"
        f"    base_url: '{base_url}'\n",
    )
    settings = GroundingSettings(waterfall_path=str(path))

    with pytest.raises(ValueError, match=reason):
        load_grounding_waterfall(settings)


def test_unreachable_inherited_endpoint_fails_startup() -> None:
    settings = GroundingSettings(llm_base_url="https://")

    with pytest.raises(ValueError, match="tier 'cerebras' needs an absolute"):
        load_grounding_waterfall(settings)


def test_resolved_api_keys_cannot_be_mutated() -> None:
    resolved = resolve_grounding_waterfall(
        _chain("ONLY_KEY"), {"ONLY_KEY": "live"}
    )

    with pytest.raises(TypeError):
        resolved.api_keys["ONLY_KEY"] = "swapped"  # type: ignore[index]


def test_resolution_drops_tiers_without_a_credential() -> None:
    chain = _chain("FIRST_KEY", "SECOND_KEY", "FIRST_KEY")

    resolved = resolve_grounding_waterfall(
        chain, {"FIRST_KEY": " live ", "SECOND_KEY": "   "}
    )

    assert [entry.name for entry in resolved.chain] == ["primary", "last"]
    assert resolved.api_keys == {"FIRST_KEY": "live"}


def test_resolution_without_any_credential_is_empty() -> None:
    resolved = resolve_grounding_waterfall(_chain("ONLY_KEY"), {})

    assert resolved.chain == ()
    assert resolved.api_keys == {}


def test_resolved_waterfall_repr_hides_credentials() -> None:
    resolved = resolve_grounding_waterfall(
        _chain("ONLY_KEY"), {"ONLY_KEY": "super-secret"}
    )

    assert "super-secret" not in repr(resolved)


def test_chain_semantics_ignore_names_and_timeouts() -> None:
    relabelled = (
        tier("renamed", _PRIMARY, "model-0", timeout_ms=999000),
        tier("also-renamed", _BACKUP, "model-1", timeout_ms=1000),
    )

    assert grounding_chain_semantics(
        _chain("A", "B")
    ) == grounding_chain_semantics(relabelled)
    assert grounding_chain_semantics(relabelled) == (
        (_PRIMARY, "model-0"),
        (_BACKUP, "model-1"),
    )


def test_credential_envs_are_distinct_and_ordered() -> None:
    chain = _chain("SECOND_KEY", "FIRST_KEY", "SECOND_KEY")

    assert grounding_credential_envs(chain) == ("SECOND_KEY", "FIRST_KEY")


@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        ("not a mapping", "malformed_body"),
        ({"error": {"code": "1214"}}, "body_error"),
        ({}, "no_choices"),
        ({"choices": []}, "no_choices"),
        ({"choices": "nope"}, "no_choices"),
        ({"choices": ["nope"]}, "no_message"),
        ({"choices": [{}]}, "no_message"),
        ({"choices": [{"message": "nope"}]}, "no_message"),
        ({"choices": [{"message": {"content": 7}}]}, "no_content"),
    ],
)
def test_unreadable_tier_payload_advances(payload: object, reason: str) -> None:
    with pytest.raises(GroundingTierError, match=reason):
        _extract_tier_snippet(payload)


def test_null_tier_content_reads_as_empty() -> None:
    assert (
        _extract_tier_snippet({"choices": [{"message": {"content": None}}]})
        == ""
    )


async def test_rate_limited_tier_advances_without_refetching(
    fetch_once: list[str],
) -> None:
    context, client = _context(
        _chain("FIRST_KEY", "SECOND_KEY"),
        MemoryCache(),
        "first-secret",
        "second-secret",
    )
    with respx.mock:
        primary = respx.post(_PRIMARY_URL).mock(
            return_value=httpx.Response(429, json={"error": "slow down"})
        )
        backup = respx.post(_BACKUP_URL).mock(return_value=_ok("Backup text"))
        pairs, stats = await ground_results("q", [_result()], context)

    assert pairs[0][1] == "grounded"
    assert pairs[0][0].snippets == ["Backup text"]
    assert stats.grounded_count == 1
    assert stats.transient_failures == 0
    assert primary.call_count == 1
    assert backup.call_count == 1
    assert len(fetch_once) == 1
    assert backup.calls[0].request.headers["authorization"] == (
        "Bearer second-secret"
    )
    await client.aclose()


async def test_in_body_gateway_error_advances(fetch_once: list[str]) -> None:
    context, client = _context(
        _chain("FIRST_KEY", "SECOND_KEY"), MemoryCache(), "k1", "k2"
    )
    with respx.mock:
        respx.post(_PRIMARY_URL).mock(
            return_value=httpx.Response(
                200, json={"error": {"code": "1214", "message": "nope"}}
            )
        )
        respx.post(_BACKUP_URL).mock(return_value=_ok("Recovered"))
        pairs, _ = await ground_results("q", [_result()], context)

    assert pairs[0][1] == "grounded"
    assert pairs[0][0].snippets == ["Recovered"]
    await client.aclose()


async def test_empty_tier_output_advances(fetch_once: list[str]) -> None:
    context, client = _context(
        _chain("FIRST_KEY", "SECOND_KEY"), MemoryCache(), "k1", "k2"
    )
    with respx.mock:
        respx.post(_PRIMARY_URL).mock(return_value=_ok(""))
        respx.post(_BACKUP_URL).mock(return_value=_ok("Second tier text"))
        pairs, _ = await ground_results("q", [_result()], context)

    assert pairs[0][1] == "grounded"
    assert pairs[0][0].snippets == ["Second tier text"]
    await client.aclose()


async def test_whitespace_only_output_never_erases_the_aggregate(
    fetch_once: list[str],
) -> None:
    context, client = _context(
        _chain("FIRST_KEY", "SECOND_KEY"), MemoryCache(), "k1", "k2"
    )
    with respx.mock:
        primary = respx.post(_PRIMARY_URL).mock(return_value=_ok("  \n\t "))
        backup = respx.post(_BACKUP_URL).mock(return_value=_ok("Real text"))
        pairs, _ = await ground_results("q", [_result()], context)

    assert pairs[0][1] == "grounded"
    assert pairs[0][0].snippets == ["Real text"]
    assert primary.call_count == 1
    assert backup.call_count == 1
    await client.aclose()


async def test_whitespace_from_every_tier_keeps_the_aggregate(
    fetch_once: list[str],
) -> None:
    context, client = _context(_chain("FIRST_KEY"), MemoryCache(), "k1")
    with respx.mock:
        primary = respx.post(_PRIMARY_URL).mock(return_value=_ok("   "))
        pairs, stats = await ground_results("q", [_result()], context)

    assert pairs[0][1] == "fallback:llm_empty"
    assert pairs[0][0].snippets == ["agg"]
    assert stats.grounded_count == 0
    assert primary.call_count == 1
    await client.aclose()


async def test_slow_tier_is_cut_off_at_its_own_budget(
    fetch_once: list[str],
) -> None:
    settings = GroundingSettings(per_url_deadline_ms=30000)
    chain = (
        tier("primary", _PRIMARY, "m0", api_key_env="K1", timeout_ms=1000),
        tier("backup", _BACKUP, "m1", api_key_env="K2", timeout_ms=20000),
    )
    context, client = _context(
        chain, MemoryCache(), "k1", "k2", settings=settings
    )

    reached: list[str] = []

    async def never_answers(request: httpx.Request) -> httpx.Response:
        reached.append("primary")
        await asyncio.sleep(30)
        return _ok("too late")

    started = asyncio.get_running_loop().time()
    with respx.mock:
        respx.post(_PRIMARY_URL).mock(side_effect=never_answers)
        backup = respx.post(_BACKUP_URL).mock(
            return_value=_ok("Backup answered")
        )
        pairs, _ = await ground_results("q", [_result()], context)
    elapsed = asyncio.get_running_loop().time() - started

    assert pairs[0][1] == "grounded"
    assert pairs[0][0].snippets == ["Backup answered"]
    assert reached == ["primary"]
    assert backup.call_count == 1
    assert elapsed < 5
    await client.aclose()


async def test_every_tier_empty_reports_llm_empty(
    fetch_once: list[str],
) -> None:
    context, client = _context(
        _chain("FIRST_KEY", "SECOND_KEY"), MemoryCache(), "k1", "k2"
    )
    with respx.mock:
        respx.post(_PRIMARY_URL).mock(return_value=_ok(""))
        respx.post(_BACKUP_URL).mock(return_value=_ok(None))
        pairs, stats = await ground_results("q", [_result()], context)

    assert pairs[0][1] == "fallback:llm_empty"
    assert pairs[0][0].snippets == ["agg"]
    assert stats.transient_failures == 0
    await client.aclose()


async def test_exhausted_chain_reports_transient_llm_error(
    fetch_once: list[str],
) -> None:
    cache = MemoryCache()
    context, client = _context(
        _chain("FIRST_KEY", "SECOND_KEY", "THIRD_KEY"), cache, "k1", "k2", "k3"
    )
    with respx.mock:
        for url in (_PRIMARY_URL, _BACKUP_URL, _LAST_URL):
            respx.post(url).mock(return_value=httpx.Response(429))
        pairs, stats = await ground_results("q", [_result()], context)

    assert pairs[0][1] == "fallback:llm_error"
    assert pairs[0][0].snippets == ["agg"]
    assert stats.transient_failures == 1
    await client.aclose()


async def test_sentinel_does_not_advance(fetch_once: list[str]) -> None:
    context, client = _context(
        _chain("FIRST_KEY", "SECOND_KEY"), MemoryCache(), "k1", "k2"
    )
    with respx.mock:
        primary = respx.post(_PRIMARY_URL).mock(
            return_value=_ok("[no usable content]")
        )
        backup = respx.post(_BACKUP_URL).mock(return_value=_ok("Never used"))
        pairs, stats = await ground_results("q", [_result()], context)

    assert pairs[0][1] == "fallback:llm_sentinel"
    assert primary.call_count == 1
    assert backup.call_count == 0
    assert stats.transient_failures == 0
    await client.aclose()


async def test_backup_output_is_cached_for_the_whole_chain(
    fetch_once: list[str],
) -> None:
    cache = MemoryCache()
    chain = _chain("FIRST_KEY", "SECOND_KEY")
    first, first_client = _context(chain, cache, "k1", "k2")
    second, second_client = _context(chain, cache, "k1", "k2")
    with respx.mock:
        primary = respx.post(_PRIMARY_URL).mock(
            return_value=httpx.Response(429)
        )
        backup = respx.post(_BACKUP_URL).mock(return_value=_ok("Reusable"))
        await ground_results("q", [_result()], first)
        pairs, _ = await ground_results("q", [_result()], second)

    assert pairs[0][1] == "grounded"
    assert pairs[0][0].snippets == ["Reusable"]
    assert primary.call_count == 1
    assert backup.call_count == 1
    assert len(fetch_once) == 2
    await first_client.aclose()
    await second_client.aclose()


async def test_swapped_chain_does_not_reuse_the_previous_namespace(
    fetch_once: list[str],
) -> None:
    cache = MemoryCache()
    first, first_client = _context(_chain("FIRST_KEY"), cache, "k1")
    second, second_client = _context(
        _chain("FIRST_KEY", "SECOND_KEY"), cache, "k1", "k2"
    )
    with respx.mock:
        primary = respx.post(_PRIMARY_URL).mock(return_value=_ok("Original"))
        await ground_results("q", [_result()], first)
        await ground_results("q", [_result()], second)

    assert primary.call_count == 2
    await first_client.aclose()
    await second_client.aclose()


async def test_spent_budget_stops_the_chain_before_another_call() -> None:
    cache = MemoryCache()
    context, client = _context(_chain("FIRST_KEY"), cache, "k1")
    message = "irrelevant"
    identity = grounding_cache_identity(message, context.waterfall.chain)
    prepared = type(
        "_Prepared", (), {"identity": identity, "result": _result()}
    )()
    with respx.mock:
        route = respx.post(_PRIMARY_URL).mock(return_value=_ok("unused"))
        outcome = await _run_grounding_waterfall(
            prepared,
            context,
            asyncio.get_running_loop().time() - 1,
        )

    assert outcome.snippet is None
    assert outcome.failure == "fallback:llm_error"
    assert route.call_count == 0
    await client.aclose()


async def test_tier_timeout_caps_the_remaining_budget(
    fetch_once: list[str],
) -> None:
    settings = GroundingSettings(per_url_deadline_ms=30000)
    chain = (
        tier("primary", _PRIMARY, "model-0", api_key_env="K", timeout_ms=1000),
    )
    context, client = _context(chain, MemoryCache(), "k1", settings=settings)
    seen: list[float | None] = []

    with respx.mock:

        async def capture(request: httpx.Request) -> httpx.Response:
            seen.append(request.extensions.get("timeout", {}).get("read"))
            return _ok("Bounded")

        respx.post(_PRIMARY_URL).mock(side_effect=capture)
        pairs, _ = await ground_results("q", [_result()], context)

    assert pairs[0][1] == "grounded"
    assert seen and seen[0] == pytest.approx(1.0, abs=0.05)
    await client.aclose()
