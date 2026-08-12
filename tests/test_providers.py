"""Provider-registry invariants and env-gated loading."""

from __future__ import annotations

import httpx

from jasa.search.providers import (
    CANONICAL_PROVIDER_ORDER,
    KNOWN_SEARCH_SECRET_ENVS,
    load_search_providers,
    PROVIDER_CLASSES,
)
from jasa.search.providers.base import SearchProvider
from omnifetch.fetch.shared.config import ProviderSecrets

_DUMMY_CLIENT = httpx.AsyncClient()


def test_canonical_order_matches_registered_classes() -> None:
    assert (
        tuple(cls.name for cls in PROVIDER_CLASSES) == CANONICAL_PROVIDER_ORDER
    )


def test_all_providers_subclass_base_and_declare_attrs() -> None:
    for provider_cls in PROVIDER_CLASSES:
        assert issubclass(provider_cls, SearchProvider)
        assert provider_cls.name
        assert provider_cls.secret_env
        assert provider_cls.base_url
        assert provider_cls.default_timeout_s > 0


def test_provider_secrets_covered_by_isolation_fixture() -> None:
    for provider_cls in PROVIDER_CLASSES:
        assert provider_cls.secret_env in KNOWN_SEARCH_SECRET_ENVS


def test_load_only_configured_providers() -> None:
    active = load_search_providers(
        ProviderSecrets.from_env({"TAVILY_API_KEY": "k"}), _DUMMY_CLIENT
    )
    assert list(active.keys()) == ["tavily"]
    ordered = load_search_providers(
        ProviderSecrets.from_env(
            {"TAVILY_API_KEY": "a", "KAGI_API_KEY": "c", "BRAVE_API_KEY": "b"}
        ),
        _DUMMY_CLIENT,
    )
    assert list(ordered.keys()) == ["tavily", "brave", "kagi"]
    empty = load_search_providers(ProviderSecrets.from_env({}), _DUMMY_CLIENT)
    assert empty == {}


def test_empty_provider_secret_needs_no_redaction() -> None:
    provider = PROVIDER_CLASSES[0]("", _DUMMY_CLIENT)
    assert provider._redact_secret("request failed") == "request failed"
