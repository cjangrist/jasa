"""Configuration loading and startup validation."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from jasa.auth import _KEY_ALIASES
from jasa.config import (
    CacheSettings,
    CompositionSettings,
    GroundingSettings,
    load_config,
    SearchSettings,
    ServerSettings,
    TelemetrySettings,
)
from jasa.search.providers import (
    KNOWN_SEARCH_SECRET_ENVS,
    KNOWN_SEARCH_SETTING_ENVS,
)
from jasa.server import _omnifetch_child_config
from omnifetch.fetch.providers.registry import import_all_providers
from omnifetch.fetch.shared.config import ProviderSecrets


def _settings_environment_names() -> set[str]:
    settings_classes = (
        ServerSettings,
        CacheSettings,
        SearchSettings,
        GroundingSettings,
        CompositionSettings,
        TelemetrySettings,
    )
    return {
        str(field.validation_alias)
        for settings_class in settings_classes
        for field in settings_class.model_fields.values()
    }


def _parse_environment_assignment(line: str) -> tuple[str, str]:
    name, separator, value = line.partition("=")
    assert separator, f"environment assignment is missing '=': {line!r}"
    return name, value


def _example_environment_entries() -> list[tuple[str, str]]:
    example = Path(__file__).resolve().parents[1] / ".env.example"
    return [
        _parse_environment_assignment(line)
        for line in example.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ]


def _example_environment_names() -> set[str]:
    return {name for name, _value in _example_environment_entries()}


def _compose_environment_names() -> set[str]:
    compose = Path(__file__).resolve().parents[1] / "docker-compose.yml"
    return set(
        re.findall(
            r"\$\{([A-Z][A-Z0-9_]*)",
            compose.read_text(encoding="utf-8"),
        )
    )


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
    assert config.cache.max_entries == 10_000
    assert config.cache.search_ttl_seconds == 129_600
    assert config.cache.fetch_ttl_seconds == 864_000
    assert config.cache.volatile_fetch_ttl_seconds == 300
    assert config.cache.grounding_ttl_seconds == 86_400
    assert config.cache.usage_ttl_seconds == 600
    assert config.search.timeout_ms == 50_000
    assert config.search.fanout_timeout_ms == 25_000
    assert config.grounding.mode == "auto"
    assert config.grounding.per_url_deadline_ms == 30_000
    assert config.grounding.concurrency == config.grounding.top_n == 20
    assert config.grounding.llm_timeout_ms == 25_000
    assert config.grounding.max_content_chars == 48_000
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
    monkeypatch.setenv("JASA_CACHE_MAX_ENTRIES", "42")
    monkeypatch.setenv("JASA_SEARCH_CACHE_TTL_SECONDS", "101")
    monkeypatch.setenv("JASA_FETCH_CACHE_TTL_SECONDS", "102")
    monkeypatch.setenv("JASA_GROUNDING_CACHE_TTL_SECONDS", "103")
    monkeypatch.setenv("JASA_USAGE_CACHE_TTL_SECONDS", "104")
    monkeypatch.setenv("JASA_GROUNDING_MODE", "off")
    monkeypatch.setenv("JASA_GROUNDING_PER_URL_DEADLINE_MS", "105")
    config = load_config()
    assert config.server.port == 7000
    assert config.cache.backend == "disk"
    assert config.cache.max_entries == 42
    assert config.cache.search_ttl_seconds == 101
    assert config.cache.fetch_ttl_seconds == 102
    assert config.cache.grounding_ttl_seconds == 103
    assert config.cache.usage_ttl_seconds == 104
    assert config.grounding.mode == "off"
    assert config.grounding.per_url_deadline_ms == 105


def test_invalid_port_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JASA_PORT", "99999")
    with pytest.raises(ValidationError):
        load_config()


@pytest.mark.parametrize(
    "name",
    [
        "JASA_CACHE_MAX_ENTRIES",
        "JASA_SEARCH_CACHE_TTL_SECONDS",
        "JASA_FETCH_CACHE_TTL_SECONDS",
        "JASA_VOLATILE_FETCH_CACHE_TTL_SECONDS",
        "JASA_GROUNDING_CACHE_TTL_SECONDS",
        "JASA_USAGE_CACHE_TTL_SECONDS",
    ],
)
def test_nonpositive_cache_setting_rejected(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    monkeypatch.setenv(name, "0")
    with pytest.raises(ValidationError):
        load_config()


def test_settings_groups_are_frozen() -> None:
    config = load_config()
    with pytest.raises(ValidationError):
        config.server.transport = "http"


def test_redis_url_is_hidden_from_configuration_repr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    redis_url = "redis://user:secret@cache.example:6379/0"
    monkeypatch.setenv("JASA_REDIS_URL", redis_url)
    assert redis_url not in repr(load_config())


def test_env_example_exactly_covers_documented_runtime_contract() -> None:
    fetch_secret_names = {
        secret_name
        for provider_class in import_all_providers().values()
        for secret_name in provider_class.required_secrets
    }
    expected_names = (
        _settings_environment_names()
        | set(KNOWN_SEARCH_SECRET_ENVS)
        | set(KNOWN_SEARCH_SETTING_ENVS)
        | fetch_secret_names
        | _compose_environment_names()
        | set(_KEY_ALIASES)
        | {"BRIGHT_DATA_ZONE", "CEREBRAS_API_KEY"}
    )
    assert _example_environment_names() == expected_names


def test_env_example_documents_every_search_setting_default() -> None:
    configured_values = dict(_example_environment_entries())
    assert all(configured_values[name] for name in KNOWN_SEARCH_SETTING_ENVS)


def test_environment_assignment_parser_reports_malformed_line() -> None:
    with pytest.raises(
        AssertionError,
        match="environment assignment is missing '=': 'MALFORMED_LINE'",
    ):
        _parse_environment_assignment("MALFORMED_LINE")


def test_env_example_contains_no_populated_secret_values() -> None:
    configured_entries = _example_environment_entries()
    configured_values = dict(configured_entries)
    fetch_secret_names = {
        secret_name
        for provider_class in import_all_providers().values()
        for secret_name in provider_class.required_secrets
    }
    secret_names = (
        set(KNOWN_SEARCH_SECRET_ENVS)
        | fetch_secret_names
        | set(_KEY_ALIASES)
        | {"CEREBRAS_API_KEY", "JASA_REDIS_URL"}
    )
    assert len(configured_values) == len(configured_entries)
    assert all(configured_values[name] == "" for name in secret_names)


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
    monkeypatch.setenv("OMNIFETCH_CACHE_MAX_ENTRIES", "1")
    monkeypatch.setenv("OMNIFETCH_FETCH_CACHE_TTL_SECONDS", "2")
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
    assert config.server.cache_max_entries == 10_000
    assert config.server.fetch_cache_ttl_seconds == 864_000
    assert config.server.volatile_fetch_cache_ttl_seconds == 300
    assert config.server.http_limit_per_host == 20
    assert config.server.http_transient_retries == 0
    assert config.server.uvloop == "auto"
    assert config.server.rest_web_fetch is False


def test_child_config_carries_both_configured_fetch_ttls() -> None:
    config = _omnifetch_child_config(
        ProviderSecrets.from_env(),
        fetch_cache_ttl_seconds=4321,
        volatile_fetch_cache_ttl_seconds=21,
    )

    assert config.server.fetch_cache_ttl_seconds == 4321
    assert config.server.volatile_fetch_cache_ttl_seconds == 21
