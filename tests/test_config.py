"""Configuration loading and startup validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jasa.config import load_config


def test_defaults_match_contract() -> None:
    config = load_config()
    assert config.server.transport == "stdio"
    assert config.server.host == "127.0.0.1"
    assert config.server.port == 8000
    assert config.server.log_level == "INFO"
    assert config.server.uvloop == "auto"
    assert config.cache.backend == "memory"
    assert config.cache.disk_path == ".cache/jasa"
    assert config.cache.redis_url == ""
    assert config.grounding.mode == "auto"
    assert config.telemetry.otel_service_name == "jasa"


def test_cli_overrides_take_precedence() -> None:
    config = load_config(
        transport="http", host="0.0.0.0", port=9000, log_level="DEBUG"
    )
    assert config.server.transport == "http"
    assert config.server.host == "0.0.0.0"
    assert config.server.port == 9000
    assert config.server.log_level == "DEBUG"


def test_env_overrides_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JASA_PORT", "7000")
    monkeypatch.setenv("JASA_CACHE_BACKEND", "disk")
    monkeypatch.setenv("JASA_GROUNDING_MODE", "off")
    config = load_config()
    assert config.server.port == 7000
    assert config.cache.backend == "disk"
    assert config.grounding.mode == "off"


def test_invalid_port_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JASA_PORT", "99999")
    with pytest.raises(ValidationError):
        load_config()


def test_settings_groups_are_frozen() -> None:
    config = load_config()
    with pytest.raises(ValidationError):
        config.server.transport = "http"
