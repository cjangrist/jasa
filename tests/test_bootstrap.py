"""CLI parsing, uvloop selection, server run, and the bootstrap sequence."""

from __future__ import annotations

import sys
from typing import Any

import pytest

from jasa.__main__ import (
    collect_overrides,
    install_uvloop,
    main,
    parse_args,
    run_server,
    validate_startup,
)
from jasa.config import load_config


def test_parse_args_defaults() -> None:
    args = parse_args([])
    assert args.transport is None
    assert args.host is None
    assert args.port is None
    assert args.log_level is None


def test_parse_args_values() -> None:
    args = parse_args(
        ["--transport", "http", "--port", "9001", "--host", "1.2.3.4"]
    )
    assert args.transport == "http"
    assert args.port == 9001
    assert args.host == "1.2.3.4"


def test_collect_overrides_drops_unset_flags() -> None:
    overrides = collect_overrides(parse_args(["--host", "0.0.0.0"]))
    assert overrides == {"host": "0.0.0.0"}


def test_validate_startup_default_ok() -> None:
    validate_startup(load_config())


def test_validate_startup_redis_requires_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JASA_CACHE_BACKEND", "redis")
    with pytest.raises(SystemExit, match="JASA_REDIS_URL is required"):
        validate_startup(load_config())


def test_validate_startup_redis_rejects_whitespace_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JASA_CACHE_BACKEND", "redis")
    monkeypatch.setenv("JASA_REDIS_URL", "   ")
    with pytest.raises(SystemExit, match="JASA_REDIS_URL is required"):
        validate_startup(load_config())


def test_validate_startup_redis_with_url_ok(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JASA_CACHE_BACKEND", "redis")
    monkeypatch.setenv("JASA_REDIS_URL", "redis://127.0.0.1:6379/0")
    validate_startup(load_config())


def test_validate_startup_grounding_on_requires_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JASA_GROUNDING_MODE", "on")
    with pytest.raises(SystemExit):
        validate_startup(load_config())


def test_validate_startup_grounding_on_with_key_ok(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JASA_GROUNDING_MODE", "on")
    monkeypatch.setenv("CEREBRAS_API_KEY", "sk-test")
    validate_startup(load_config())


def test_install_uvloop_off_is_default_loop() -> None:
    assert install_uvloop("off") is False


def test_install_uvloop_on_installs_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("uvloop.install", lambda: None)
    assert install_uvloop("auto") is True


def test_install_uvloop_missing_uses_default_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "uvloop", None)
    assert install_uvloop("auto") is False


def test_run_server_stdio(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class FakeServer:
        def run(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    monkeypatch.setattr("jasa.server.build_server", lambda config: FakeServer())
    run_server(load_config(transport="stdio"))
    assert captured == {"transport": "stdio"}


def test_run_server_http(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class FakeServer:
        def run(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    monkeypatch.setattr("jasa.server.build_server", lambda config: FakeServer())
    run_server(load_config(transport="http", host="0.0.0.0", port=1234))
    assert captured == {
        "transport": "http",
        "host": "0.0.0.0",
        "port": 1234,
    }


def test_main_runs_full_bootstrap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("uvloop.install", lambda: None)
    monkeypatch.setattr("jasa.__main__.run_server", lambda config: None)
    main([])
