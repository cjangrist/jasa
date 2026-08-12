"""Configuration loading and startup validation."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from jasa.config import (
    CacheSettings,
    CompositionSettings,
    GroundingSettings,
    load_config,
    ServerSettings,
    TelemetrySettings,
)
from jasa.search.providers import KNOWN_SEARCH_SECRET_ENVS
from jasa.server import _omnifetch_child_config
from omnifetch.fetch.providers.registry import import_all_providers
from omnifetch.fetch.shared.config import ProviderSecrets


def _settings_environment_names() -> set[str]:
    settings_classes = (
        ServerSettings,
        CacheSettings,
        GroundingSettings,
        CompositionSettings,
        TelemetrySettings,
    )
    return {
        str(field.validation_alias)
        for settings_class in settings_classes
        for field in settings_class.model_fields.values()
    }


def _example_environment_names() -> set[str]:
    example = Path(__file__).resolve().parents[1] / ".env.example"
    return {
        line.split("=", 1)[0]
        for line in example.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    }


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
    assert not hasattr(config.composition, "compat_fetch_tool")
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


def test_env_example_exactly_covers_supported_runtime_environment() -> None:
    fetch_secret_names = {
        secret_name
        for provider_class in import_all_providers().values()
        for secret_name in provider_class.required_secrets
    }
    expected_names = (
        _settings_environment_names()
        | set(KNOWN_SEARCH_SECRET_ENVS)
        | fetch_secret_names
        | {
            "BRIGHT_DATA_ZONE",
            "CEREBRAS_API_KEY",
            "JASA_API_KEY",
            "JASA_DOCKER_HOST",
            "JASA_DOCKER_PORT",
            "OMNISEARCH_API_KEY",
            "OPENWEBUI_API_KEY",
        }
    )
    assert _example_environment_names() == expected_names


def test_composed_child_ignores_omnifetch_runtime_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OMNIFETCH_TRANSPORT", "sse")
    monkeypatch.setenv("OMNIFETCH_HOST", "192.0.2.1")
    monkeypatch.setenv("OMNIFETCH_PORT", "9000")
    monkeypatch.setenv("OMNIFETCH_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("OMNIFETCH_CACHE_BACKEND", "redis")
    monkeypatch.setenv("OMNIFETCH_REDIS_URL", "redis://192.0.2.2")
    monkeypatch.setenv("OMNIFETCH_DISK_CACHE_PATH", "/tmp/omnifetch")
    monkeypatch.setenv("OMNIFETCH_HTTP_LIMIT_PER_HOST", "1")
    monkeypatch.setenv("OMNIFETCH_HTTP_TRANSIENT_RETRIES", "5")
    monkeypatch.setenv("OMNIFETCH_UVLOOP", "off")
    monkeypatch.setenv("OMNIFETCH_REST_WEB_FETCH", "true")

    config = _omnifetch_child_config(ProviderSecrets.from_env())

    assert config.server.transport == "stdio"
    assert config.server.host == "127.0.0.1"
    assert config.server.port == 8000
    assert config.server.log_level == "INFO"
    assert config.server.cache_backend == "memory"
    assert config.server.redis_url == ""
    assert config.server.disk_cache_path == ".cache/omnifetch"
    assert config.server.http_limit_per_host == 20
    assert config.server.http_transient_retries == 0
    assert config.server.uvloop == "auto"
    assert config.server.rest_web_fetch is False
