"""Provider usage snapshots, provider probes, caching, and refresh hooks."""

from __future__ import annotations

import asyncio
import copy
import logging
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
import respx
from fastmcp import Client
from fastmcp.exceptions import ToolError
from starlette.testclient import TestClient

import jasa.usage.runtime as runtime_module
from jasa.config import load_config
from jasa.server import build_composition, build_composition_async
from jasa.usage.base import (
    clean_provider_value,
    MAX_RESPONSE_BYTES,
    REDACTED,
    request_usage_json,
    UsageProbe,
    UsageResponseError,
)
from jasa.usage.providers.firecrawl import fetch_firecrawl_usage
from jasa.usage.providers.github import fetch_github_usage
from jasa.usage.providers.serpapi import fetch_serpapi_usage
from jasa.usage.providers.serper import fetch_serper_usage
from jasa.usage.providers.tavily import fetch_tavily_usage
from jasa.usage.runtime import (
    UsageRefreshMiddleware,
    UsageRuntime,
)
from omnifetch.fetch.shared.config import ProviderSecrets
from tests.usage_helpers import build_usage_runtime


async def test_tavily_exact_request_retains_cleaned_raw_shape(
    http_client: httpx.AsyncClient,
) -> None:
    secret = '"tvly-secret"'
    with respx.mock:
        route = respx.get("https://api.tavily.com/usage").mock(
            return_value=httpx.Response(
                200,
                json={
                    "account": {
                        "accountId": "acct-1",
                        "email": "owner@example.com",
                        "plan": "pro",
                    },
                    "key": {"usage": 7, "limit": 100},
                    "note": "1 credential tvly-secret must disappear",
                    "allowedNetworks": ["192.0.2.0/24"],
                    "nullable": None,
                },
            )
        )
        raw = await fetch_tavily_usage(
            http_client,
            ProviderSecrets({"TAVILY_API_KEY": secret, "SHLVL": "1"}),
        )

    assert route.call_count == 1
    assert route.calls[0].request.headers["Authorization"] == (
        "Bearer tvly-secret"
    )
    assert raw == {
        "account": {
            "accountId": REDACTED,
            "email": REDACTED,
            "plan": "pro",
        },
        "key": {"usage": 7, "limit": 100},
        "note": f"1 credential {REDACTED} must disappear",
        "allowedNetworks": [REDACTED],
        "nullable": None,
    }


async def test_firecrawl_exact_request_retains_native_raw_shape(
    http_client: httpx.AsyncClient,
) -> None:
    secret = '"fc-secret"'
    with respx.mock:
        route = respx.get(
            "https://api.firecrawl.dev/v2/team/credit-usage"
        ).mock(
            return_value=httpx.Response(
                200,
                json={
                    "success": True,
                    "data": {
                        "remainingCredits": 900,
                        "planCredits": 1000,
                        "billingPeriodStart": "2026-08-01T00:00:00Z",
                        "billingPeriodEnd": "2026-08-31T23:59:59Z",
                    },
                },
            )
        )
        raw = await fetch_firecrawl_usage(
            http_client,
            ProviderSecrets({"FIRECRAWL_API_KEY": secret}),
        )

    assert route.call_count == 1
    assert route.calls[0].request.headers["Authorization"] == (
        "Bearer fc-secret"
    )
    assert raw == {
        "success": True,
        "data": {
            "remainingCredits": 900,
            "planCredits": 1000,
            "billingPeriodStart": "2026-08-01T00:00:00Z",
            "billingPeriodEnd": "2026-08-31T23:59:59Z",
        },
    }


async def test_github_exact_request_retains_native_rate_limits(
    http_client: httpx.AsyncClient,
) -> None:
    secret = '"github-secret"'
    with respx.mock:
        route = respx.get("https://api.github.com/rate_limit").mock(
            return_value=httpx.Response(
                200,
                json={
                    "resources": {
                        "core": {
                            "limit": 5000,
                            "remaining": 4999,
                            "reset": 1_787_000_000,
                            "used": 1,
                        },
                        "search": {
                            "limit": 30,
                            "remaining": 30,
                            "reset": 1_787_000_000,
                            "used": 0,
                        },
                    },
                    "rate": {
                        "limit": 5000,
                        "remaining": 4999,
                        "reset": 1_787_000_000,
                        "used": 1,
                    },
                },
            )
        )
        raw = await fetch_github_usage(
            http_client,
            ProviderSecrets({"GITHUB_API_KEY": secret}),
        )

    request = route.calls[0].request
    assert route.call_count == 1
    assert request.headers["Accept"] == "application/vnd.github+json"
    assert request.headers["Authorization"] == "Bearer github-secret"
    assert request.headers["X-GitHub-Api-Version"] == "2022-11-28"
    assert raw == {
        "resources": {
            "core": {
                "limit": 5000,
                "remaining": 4999,
                "reset": 1_787_000_000,
                "used": 1,
            },
            "search": {
                "limit": 30,
                "remaining": 30,
                "reset": 1_787_000_000,
                "used": 0,
            },
        },
        "rate": {
            "limit": 5000,
            "remaining": 4999,
            "reset": 1_787_000_000,
            "used": 1,
        },
    }


async def test_serpapi_exact_request_cleans_native_account_response(
    http_client: httpx.AsyncClient,
) -> None:
    secret = '"serpapi-secret"'
    with respx.mock:
        route = respx.get("https://serpapi.com/account.json").mock(
            return_value=httpx.Response(
                200,
                json={
                    "account_id": "account-1",
                    "api_key": "serpapi-secret",
                    "account_email": "owner@example.com",
                    "account_status": "Active",
                    "plan_name": "Developer Plan",
                    "searches_per_month": 5000,
                    "plan_searches_left": 1200,
                    "total_searches_left": 1250,
                    "this_month_usage": 3800,
                    "account_rate_limit_per_hour": 1000,
                },
            )
        )
        raw = await fetch_serpapi_usage(
            http_client,
            ProviderSecrets({"SERPAPI_API_KEY": secret}),
        )

    request = route.calls[0].request
    assert route.call_count == 1
    assert request.url.params["api_key"] == "serpapi-secret"
    assert raw == {
        "account_id": REDACTED,
        "api_key": REDACTED,
        "account_email": REDACTED,
        "account_status": "Active",
        "plan_name": "Developer Plan",
        "searches_per_month": 5000,
        "plan_searches_left": 1200,
        "total_searches_left": 1250,
        "this_month_usage": 3800,
        "account_rate_limit_per_hour": 1000,
    }


async def test_serper_exact_request_retains_native_account_response(
    http_client: httpx.AsyncClient,
) -> None:
    secret = '"serper-secret"'
    with respx.mock:
        route = respx.get("https://google.serper.dev/account").mock(
            return_value=httpx.Response(
                200,
                json={"balance": 2499, "rateLimit": 50},
            )
        )
        raw = await fetch_serper_usage(
            http_client,
            ProviderSecrets({"SERPER_API_KEY": secret}),
        )

    assert route.call_count == 1
    assert route.calls[0].request.headers["X-API-KEY"] == "serper-secret"
    assert raw == {"balance": 2499, "rateLimit": 50}


def test_recursive_cleaning_preserves_shape_and_other_values() -> None:
    value = {
        3: (True, 2, 1.5, None, object()),
        "accountID": "account-1",
        "APIKey": "credential-1",
        "USER_ID": "user-1",
        "workspace-id": "workspace-1",
        "token": "anything",
    }
    cleaned = clean_provider_value(value, ())
    assert isinstance(cleaned, dict)
    assert cleaned["accountID"] == REDACTED
    assert cleaned["APIKey"] == REDACTED
    assert cleaned["USER_ID"] == REDACTED
    assert cleaned["workspace-id"] == REDACTED
    assert cleaned["token"] == REDACTED
    sequence = cast(list[object], cleaned["3"])
    assert sequence[:4] == [True, 2, 1.5, None]
    assert cast(str, sequence[4]).startswith("<object object at ")


async def test_request_helper_wraps_non_dictionary_json(
    http_client: httpx.AsyncClient,
) -> None:
    with respx.mock:
        respx.get("https://usage.example/value").mock(
            return_value=httpx.Response(200, json=[1, "two"])
        )
        raw = await request_usage_json(
            http_client,
            ProviderSecrets(),
            "GET",
            "https://usage.example/value",
        )
    assert raw == {"value": [1, "two"]}


async def test_http_error_retains_status_and_non_json_body(
    http_client: httpx.AsyncClient,
) -> None:
    with respx.mock:
        respx.get("https://usage.example/error").mock(
            return_value=httpx.Response(429, text="quota exhausted")
        )
        with pytest.raises(UsageResponseError) as captured:
            await request_usage_json(
                http_client,
                ProviderSecrets(),
                "GET",
                "https://usage.example/error",
            )
    assert str(captured.value) == "usage endpoint returned HTTP 429"
    assert captured.value.status_code == 429
    assert captured.value.raw == {"body": "quota exhausted"}


async def test_non_json_body_with_unknown_charset_falls_back_to_utf8(
    monkeypatch: pytest.MonkeyPatch,
    http_client: httpx.AsyncClient,
) -> None:
    response = httpx.Response(500, content=b"upstream failed")
    monkeypatch.setattr(
        httpx.Response,
        "encoding",
        property(lambda _response: "definitely-real"),
    )
    with respx.mock:
        respx.get("https://usage.example/unknown-charset").mock(
            return_value=response
        )
        with pytest.raises(UsageResponseError) as captured:
            await request_usage_json(
                http_client,
                ProviderSecrets(),
                "GET",
                "https://usage.example/unknown-charset",
            )
    assert captured.value.raw == {"body": "upstream failed"}


async def test_response_larger_than_one_mebibyte_is_rejected(
    http_client: httpx.AsyncClient,
) -> None:
    with respx.mock:
        respx.get("https://usage.example/large").mock(
            return_value=httpx.Response(
                200, content=b"x" * (MAX_RESPONSE_BYTES + 1)
            )
        )
        with pytest.raises(ValueError, match="exceeded the 1 MiB limit"):
            await request_usage_json(
                http_client,
                ProviderSecrets(),
                "GET",
                "https://usage.example/large",
            )


async def test_snapshot_enumerates_every_registered_provider_when_unconfigured(
    http_client: httpx.AsyncClient,
) -> None:
    usage = build_usage_runtime(http_client)
    snapshot = await usage.get_snapshot()
    search = cast(dict[str, dict[str, object]], snapshot["search"])
    fetch = cast(dict[str, dict[str, object]], snapshot["fetch"])

    assert list(search) == list(usage.search_requirements)
    assert list(fetch) == list(usage.fetch_requirements)
    assert search["tavily"] == {
        "configured": False,
        "status": "unconfigured",
        "supported": True,
    }
    assert search["brave"] == {
        "configured": False,
        "status": "not_implemented",
        "supported": False,
    }
    assert fetch["tavily"] == search["tavily"]
    assert search["firecrawl"] == {
        "configured": False,
        "status": "unconfigured",
        "supported": True,
    }
    assert fetch["firecrawl"] == search["firecrawl"]
    assert fetch["github"] == {
        "configured": False,
        "status": "unconfigured",
        "supported": True,
    }
    assert search["serpapi"] == {
        "configured": False,
        "status": "unconfigured",
        "supported": True,
    }
    assert fetch["serpapi"] == search["serpapi"]
    assert search["serper"] == {
        "configured": False,
        "status": "unconfigured",
        "supported": True,
    }
    assert "serper" not in fetch


async def test_configured_tavily_is_collected_once_for_both_families(
    http_client: httpx.AsyncClient,
) -> None:
    with respx.mock:
        route = respx.get("https://api.tavily.com/usage").mock(
            return_value=httpx.Response(
                200,
                json={"account": {"plan_usage": 4}, "key": {"usage": 2}},
            )
        )
        usage = build_usage_runtime(
            http_client, secrets={"TAVILY_API_KEY": "tvly-test"}
        )
        snapshot = await usage.get_snapshot()

    expected = {
        "configured": True,
        "status": "ok",
        "supported": True,
        "raw": {"account": {"plan_usage": 4}, "key": {"usage": 2}},
    }
    assert cast(dict[str, object], snapshot["search"])["tavily"] == expected
    assert cast(dict[str, object], snapshot["fetch"])["tavily"] == expected
    assert route.call_count == 1


async def test_unconfigured_tavily_makes_no_http_call(
    http_client: httpx.AsyncClient,
) -> None:
    with respx.mock:
        route = respx.get("https://api.tavily.com/usage").mock(
            return_value=httpx.Response(500)
        )
        await build_usage_runtime(http_client).get_snapshot()
    assert route.call_count == 0


async def test_provider_http_error_is_isolated_with_cleaned_raw_response(
    http_client: httpx.AsyncClient,
) -> None:
    secret = "tvly-test"
    with respx.mock:
        respx.get("https://api.tavily.com/usage").mock(
            return_value=httpx.Response(
                401,
                json={"api_key": secret, "detail": f"bad {secret}"},
            )
        )
        snapshot = await build_usage_runtime(
            http_client, secrets={"TAVILY_API_KEY": secret}
        ).get_snapshot()

    record = cast(dict[str, Any], snapshot["search"])["tavily"]
    assert record == {
        "configured": True,
        "status": "error",
        "supported": True,
        "raw": {"api_key": REDACTED, "detail": f"bad {REDACTED}"},
        "error": {"type": "http_error", "status_code": 401},
    }


async def test_firecrawl_http_error_is_shared_and_redacted(
    caplog: pytest.LogCaptureFixture,
    http_client: httpx.AsyncClient,
) -> None:
    secret = "fc-test"
    with respx.mock:
        route = respx.get(
            "https://api.firecrawl.dev/v2/team/credit-usage"
        ).mock(
            return_value=httpx.Response(
                404,
                json={
                    "success": False,
                    "error": f"credential {secret} was rejected",
                    "apiKey": secret,
                },
            )
        )
        with caplog.at_level(logging.WARNING, logger="jasa.usage"):
            snapshot = await build_usage_runtime(
                http_client, secrets={"FIRECRAWL_API_KEY": secret}
            ).get_snapshot()

    expected = {
        "configured": True,
        "status": "error",
        "supported": True,
        "raw": {
            "success": False,
            "error": f"credential {REDACTED} was rejected",
            "apiKey": REDACTED,
        },
        "error": {"type": "http_error", "status_code": 404},
    }
    assert cast(dict[str, Any], snapshot["search"])["firecrawl"] == expected
    assert cast(dict[str, Any], snapshot["fetch"])["firecrawl"] == expected
    assert route.call_count == 1
    assert "Usage probe firecrawl returned HTTP 404" in caplog.messages


async def test_github_http_error_is_fetch_only_and_redacted(
    caplog: pytest.LogCaptureFixture,
    http_client: httpx.AsyncClient,
) -> None:
    secret = "github-test"
    with respx.mock:
        route = respx.get("https://api.github.com/rate_limit").mock(
            return_value=httpx.Response(
                401,
                json={
                    "message": f"Bad credentials: {secret}",
                    "token": secret,
                },
            )
        )
        with caplog.at_level(logging.WARNING, logger="jasa.usage"):
            snapshot = await build_usage_runtime(
                http_client, secrets={"GITHUB_API_KEY": secret}
            ).get_snapshot()

    expected = {
        "configured": True,
        "status": "error",
        "supported": True,
        "raw": {
            "message": f"Bad credentials: {REDACTED}",
            "token": REDACTED,
        },
        "error": {"type": "http_error", "status_code": 401},
    }
    assert "github" not in cast(dict[str, Any], snapshot["search"])
    assert cast(dict[str, Any], snapshot["fetch"])["github"] == expected
    assert route.call_count == 1
    assert "Usage probe github returned HTTP 401" in caplog.messages


async def test_serpapi_http_error_is_shared_and_redacted(
    caplog: pytest.LogCaptureFixture,
    http_client: httpx.AsyncClient,
) -> None:
    secret = "serpapi-test"
    with respx.mock:
        route = respx.get("https://serpapi.com/account.json").mock(
            return_value=httpx.Response(
                401,
                json={
                    "error": f"Invalid API key: {secret}",
                    "api_key": secret,
                },
            )
        )
        with caplog.at_level(logging.WARNING, logger="jasa.usage"):
            snapshot = await build_usage_runtime(
                http_client, secrets={"SERPAPI_API_KEY": secret}
            ).get_snapshot()

    expected = {
        "configured": True,
        "status": "error",
        "supported": True,
        "raw": {
            "error": f"Invalid API key: {REDACTED}",
            "api_key": REDACTED,
        },
        "error": {"type": "http_error", "status_code": 401},
    }
    assert cast(dict[str, Any], snapshot["search"])["serpapi"] == expected
    assert cast(dict[str, Any], snapshot["fetch"])["serpapi"] == expected
    assert route.call_count == 1
    assert "Usage probe serpapi returned HTTP 401" in caplog.messages


async def test_serper_http_error_is_search_only_and_redacted(
    caplog: pytest.LogCaptureFixture,
    http_client: httpx.AsyncClient,
) -> None:
    secret = "serper-test"
    with respx.mock:
        route = respx.get("https://google.serper.dev/account").mock(
            return_value=httpx.Response(
                403,
                json={
                    "message": f"Unauthorized API key: {secret}",
                    "apiKey": secret,
                    "statusCode": 403,
                },
            )
        )
        with caplog.at_level(logging.WARNING, logger="jasa.usage"):
            snapshot = await build_usage_runtime(
                http_client, secrets={"SERPER_API_KEY": secret}
            ).get_snapshot()

    expected = {
        "configured": True,
        "status": "error",
        "supported": True,
        "raw": {
            "message": f"Unauthorized API key: {REDACTED}",
            "apiKey": REDACTED,
            "statusCode": 403,
        },
        "error": {"type": "http_error", "status_code": 403},
    }
    assert cast(dict[str, Any], snapshot["search"])["serper"] == expected
    assert "serper" not in cast(dict[str, Any], snapshot["fetch"])
    assert route.call_count == 1
    assert "Usage probe serper returned HTTP 403" in caplog.messages


async def test_unexpected_probe_error_is_isolated_and_redacted(
    monkeypatch: pytest.MonkeyPatch,
    http_client: httpx.AsyncClient,
) -> None:
    secret = "tvly-test"

    async def fail(
        _client: httpx.AsyncClient, _secrets: ProviderSecrets
    ) -> dict[str, Any]:
        raise RuntimeError(f"credential {secret} failed")

    monkeypatch.setattr(
        runtime_module,
        "PROVIDER_USAGE_PROBES",
        {"tavily": UsageProbe(("TAVILY_API_KEY",), fail)},
    )
    snapshot = await build_usage_runtime(
        http_client, secrets={"TAVILY_API_KEY": secret}
    ).get_snapshot()
    record = cast(dict[str, Any], snapshot["search"])["tavily"]
    assert record["status"] == "error"
    assert record["error"] == {
        "type": "RuntimeError",
        "message": f"credential {REDACTED} failed",
    }


async def test_background_trigger_is_nonblocking_and_skips_fresh_or_closed(
    monkeypatch: pytest.MonkeyPatch,
    http_client: httpx.AsyncClient,
) -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    async def fetch(
        _client: httpx.AsyncClient, _secrets: ProviderSecrets
    ) -> dict[str, Any]:
        started.set()
        await release.wait()
        return {"remaining": 1}

    monkeypatch.setattr(
        runtime_module,
        "PROVIDER_USAGE_PROBES",
        {"tavily": UsageProbe(("TAVILY_API_KEY",), fetch)},
    )
    usage = build_usage_runtime(
        http_client, secrets={"TAVILY_API_KEY": "secret"}
    )
    usage.trigger_refresh()
    assert not started.is_set()
    async with asyncio.timeout(1):
        await started.wait()
    release.set()
    await usage.get_snapshot()
    assert usage._refresh_task is None
    deepcopy = MagicMock(side_effect=AssertionError("unexpected deep copy"))
    monkeypatch.setattr(copy, "deepcopy", deepcopy)
    usage.trigger_refresh()
    deepcopy.assert_not_called()
    assert usage._refresh_task is None
    await usage.close()
    usage.trigger_refresh()
    assert usage._refresh_task is None


async def test_refresh_task_exception_is_observed_and_logged(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    http_client: httpx.AsyncClient,
) -> None:
    async def fail(_usage: UsageRuntime) -> dict[str, Any]:
        raise RuntimeError("refresh failed")

    monkeypatch.setattr(UsageRuntime, "_refresh_if_missing", fail)
    usage = build_usage_runtime(http_client)
    with caplog.at_level(logging.WARNING, logger="jasa.usage"):
        usage.trigger_refresh()
        async with asyncio.timeout(1):
            while usage._refresh_task is not None:
                await asyncio.sleep(0)
    assert "Usage refresh failed (RuntimeError)" in caplog.messages


async def test_close_cancels_and_observes_active_refresh(
    monkeypatch: pytest.MonkeyPatch,
    http_client: httpx.AsyncClient,
) -> None:
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def fetch(
        _client: httpx.AsyncClient, _secrets: ProviderSecrets
    ) -> dict[str, Any]:
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()
        return {}

    monkeypatch.setattr(
        runtime_module,
        "PROVIDER_USAGE_PROBES",
        {"tavily": UsageProbe(("TAVILY_API_KEY",), fetch)},
    )
    usage = build_usage_runtime(
        http_client, secrets={"TAVILY_API_KEY": "secret"}
    )
    usage.trigger_refresh()
    async with asyncio.timeout(1):
        await started.wait()
    await usage.close()

    assert cancelled.is_set()
    assert usage._closed is True
    with pytest.raises(RuntimeError, match="usage runtime is closed"):
        await usage.get_snapshot()
    await usage.close()


async def test_close_ignores_an_already_completed_refresh_task(
    http_client: httpx.AsyncClient,
) -> None:
    usage = build_usage_runtime(http_client)
    task = asyncio.create_task(usage._collect_snapshot())
    await task
    usage._refresh_finished(task)
    usage._refresh_task = task
    await usage.close()
    assert task.done()


@pytest.mark.parametrize(
    ("tool_name", "expected_calls"),
    [("web_search", 1), ("web_fetch", 1), ("say_hello", 0), (None, 0)],
)
async def test_middleware_triggers_only_public_search_and_fetch_tools(
    tool_name: str | None,
    expected_calls: int,
    monkeypatch: pytest.MonkeyPatch,
    http_client: httpx.AsyncClient,
) -> None:
    trigger = MagicMock()
    monkeypatch.setattr(UsageRuntime, "trigger_refresh", trigger)
    usage = build_usage_runtime(http_client)
    middleware = UsageRefreshMiddleware(usage)

    async def call_next(_context: object) -> str:
        return "next"

    context = SimpleNamespace(message=SimpleNamespace(name=tool_name))
    result = await middleware.on_call_tool(
        cast(Any, context), cast(Any, call_next)
    )
    assert result == "next"
    assert trigger.call_count == expected_calls


def test_usage_route_uses_shared_auth_and_returns_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JASA_API_KEY", "rest-secret")
    snapshot = AsyncMock(
        return_value={"schema_version": 1, "search": {}, "fetch": {}}
    )
    monkeypatch.setattr(UsageRuntime, "get_snapshot", snapshot)
    composition = build_composition(load_config())
    with TestClient(composition.server.http_app()) as client:
        unauthorized = client.get("/usage")
        authorized = client.get(
            "/usage", headers={"Authorization": "Bearer rest-secret"}
        )

    assert unauthorized.status_code == 401
    assert authorized.status_code == 200
    assert authorized.json() == {
        "schema_version": 1,
        "search": {},
        "fetch": {},
    }
    snapshot.assert_awaited_once()


def test_usage_route_timeout_returns_504(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def blocked(_usage: UsageRuntime) -> dict[str, Any]:
        await asyncio.Event().wait()
        return {}

    monkeypatch.setattr(UsageRuntime, "get_snapshot", blocked)
    monkeypatch.setattr("jasa.rest._USAGE_TIMEOUT_SECONDS", 0.001)
    composition = build_composition(load_config())
    with TestClient(composition.server.http_app()) as client:
        response = client.get("/usage")

    assert response.status_code == 504
    assert response.json() == {"error": "usage timed out"}


def test_usage_route_closed_runtime_returns_503(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def closed(_usage: UsageRuntime) -> dict[str, Any]:
        raise RuntimeError("usage runtime is closed")

    monkeypatch.setattr(UsageRuntime, "get_snapshot", closed)
    composition = build_composition(load_config())
    with TestClient(composition.server.http_app()) as client:
        response = client.get("/usage")

    assert response.status_code == 503
    assert response.json() == {"error": "usage unavailable"}


def test_rest_execution_routes_trigger_background_refresh_after_auth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JASA_API_KEY", "rest-secret")
    trigger = MagicMock()
    monkeypatch.setattr(UsageRuntime, "trigger_refresh", trigger)
    composition = build_composition(load_config())
    with TestClient(composition.server.http_app()) as client:
        assert client.post("/search", json={"query": "test"}).status_code == 401
        assert client.post("/fetch", json={}).status_code == 401
        assert client.get("/researcher").status_code == 401
        assert trigger.call_count == 0
        headers = {"Authorization": "Bearer rest-secret"}
        assert (
            client.post(
                "/search", json={"query": "test"}, headers=headers
            ).status_code
            == 503
        )
        assert (
            client.post("/fetch", json={}, headers=headers).status_code == 400
        )
        assert client.get("/researcher", headers=headers).status_code == 400
    assert trigger.call_count == 3


async def test_parent_middleware_observes_search_and_mounted_fetch_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trigger = MagicMock()
    monkeypatch.setattr(UsageRuntime, "trigger_refresh", trigger)
    composition = await build_composition_async(load_config())
    async with Client(composition.server) as client:
        with pytest.raises(ToolError):
            await client.call_tool("web_search", {"query": "test"})
        with pytest.raises(ToolError):
            await client.call_tool("web_fetch", {"url": "https://example.com"})
    assert trigger.call_count == 2
