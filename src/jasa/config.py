"""Typed, immutable application configuration for the jasa server.

Each setting declares its exact environment variable via ``validation_alias``.
``load_config`` reads settings once and returns a frozen ``AppConfig`` that is
passed explicitly through the application. Provider secrets keep provider-native
names with no ``JASA_`` prefix by design: five of them (TAVILY, FIRECRAWL,
LINKUP, YOU, SERPAPI) enable a provider in both the jasa search family and the
mounted omnifetch fetch family.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

TransportName = Literal["stdio", "http", "sse"]
CacheBackendName = Literal["memory", "disk", "redis"]
UvloopModeName = Literal["auto", "on", "off"]
GroundingModeName = Literal["auto", "on", "off"]
OtelExporterName = Literal["", "none", "console", "otlp"]
OtelProtocolName = Literal["grpc", "http/protobuf"]

DEFAULT_CACHE_MAX_ENTRIES = 10_000
DEFAULT_SEARCH_TIMEOUT_MS = 50_000
DEFAULT_FANOUT_TIMEOUT_MS = 25_000
DEFAULT_SEARCH_CACHE_TTL_SECONDS = 129_600
DEFAULT_FETCH_CACHE_TTL_SECONDS = 864_000
DEFAULT_VOLATILE_FETCH_CACHE_TTL_SECONDS = 300
DEFAULT_GROUNDING_CACHE_TTL_SECONDS = 86_400
DEFAULT_USAGE_CACHE_TTL_SECONDS = 600

_SETTINGS_MODEL_CONFIG = SettingsConfigDict(
    case_sensitive=True,
    extra="ignore",
    frozen=True,
    populate_by_name=True,
)


class ServerSettings(BaseSettings):
    """Runtime transport and process settings, read from ``JASA_`` variables.

    ``public_url`` is the externally reachable base this server answers on. It
    is only ever advertised, never bound: the bind address is ``host``/``port``,
    which behind a reverse proxy says nothing about how a client reaches the
    server. Leaving it empty is a working configuration; it only changes the
    icon from an inlined image to a set of links.

    It is kept out of the representation because startup logs the whole config
    at DEBUG before anything validates this field. A URL is a shape that can
    carry a credential, and the value is rejected for carrying one only after
    that line has already been written.
    """

    model_config = _SETTINGS_MODEL_CONFIG

    transport: TransportName = Field(
        default="stdio", validation_alias="JASA_TRANSPORT"
    )
    host: str = Field(default="127.0.0.1", validation_alias="JASA_HOST")
    port: int = Field(
        default=8000, ge=1, le=65535, validation_alias="JASA_PORT"
    )
    log_level: str = Field(default="INFO", validation_alias="JASA_LOG_LEVEL")
    uvloop: UvloopModeName = Field(
        default="auto", validation_alias="JASA_UVLOOP"
    )
    public_url: str = Field(
        default="", repr=False, validation_alias="JASA_PUBLIC_URL"
    )


class CacheSettings(BaseSettings):
    """Shared search, fetch, grounding, and usage cache settings.

    ``memory`` is the application default. Select ``disk`` to survive restarts
    or ``redis`` to share entries across replicas.
    """

    model_config = _SETTINGS_MODEL_CONFIG

    backend: CacheBackendName = Field(
        default="memory", validation_alias="JASA_CACHE_BACKEND"
    )
    disk_path: str = Field(
        default=".cache/jasa", validation_alias="JASA_DISK_CACHE_PATH"
    )
    redis_url: str = Field(
        default="", repr=False, validation_alias="JASA_REDIS_URL"
    )
    max_entries: int = Field(
        default=DEFAULT_CACHE_MAX_ENTRIES,
        ge=1,
        validation_alias="JASA_CACHE_MAX_ENTRIES",
    )
    search_ttl_seconds: int = Field(
        default=DEFAULT_SEARCH_CACHE_TTL_SECONDS,
        ge=1,
        validation_alias="JASA_SEARCH_CACHE_TTL_SECONDS",
    )
    fetch_ttl_seconds: int = Field(
        default=DEFAULT_FETCH_CACHE_TTL_SECONDS,
        ge=1,
        validation_alias="JASA_FETCH_CACHE_TTL_SECONDS",
    )
    volatile_fetch_ttl_seconds: int = Field(
        default=DEFAULT_VOLATILE_FETCH_CACHE_TTL_SECONDS,
        ge=1,
        validation_alias="JASA_VOLATILE_FETCH_CACHE_TTL_SECONDS",
    )
    grounding_ttl_seconds: int = Field(
        default=DEFAULT_GROUNDING_CACHE_TTL_SECONDS,
        ge=1,
        validation_alias="JASA_GROUNDING_CACHE_TTL_SECONDS",
    )
    usage_ttl_seconds: int = Field(
        default=DEFAULT_USAGE_CACHE_TTL_SECONDS,
        ge=1,
        validation_alias="JASA_USAGE_CACHE_TTL_SECONDS",
    )


class SearchSettings(BaseSettings):
    """The request budget and the share of it the provider fan-out may spend.

    ``timeout_ms`` is the whole-request budget applied when a caller names no
    deadline of its own. ``fanout_timeout_ms`` bounds the provider fan-out
    inside that budget so the stages after it -- ranking and, above all,
    grounding -- inherit time rather than scraps. A fan-out given the entire
    budget starves grounding of the seconds its fetch and LLM call need, which
    wastes an LLM call that was already paid for.

    The default budget sits below the 60-second request timeout MCP clients
    commonly ship with, because that timeout is the real ceiling: a client that
    gives up mid-request abandons every fetch and completion the server has
    already paid for, which is worse than returning whatever finished in time.
    Raise this only alongside the client's own timeout.
    """

    model_config = _SETTINGS_MODEL_CONFIG

    timeout_ms: int = Field(
        default=DEFAULT_SEARCH_TIMEOUT_MS,
        ge=1,
        validation_alias="JASA_SEARCH_TIMEOUT_MS",
    )
    fanout_timeout_ms: int = Field(
        default=DEFAULT_FANOUT_TIMEOUT_MS,
        ge=1,
        validation_alias="JASA_SEARCH_FANOUT_TIMEOUT_MS",
    )


class GroundingSettings(BaseSettings):
    """Grounding tuning (concurrency, deadlines, model, generation params).

    The ``llm_*`` values configure the waterfall tiers that omit them, and
    ``waterfall_path`` replaces the packaged chain with an operator's own file.

    The deadlines are deliberately generous. One grounded result costs a page
    fetch plus at least one LLM completion, both already billed by the time any
    deadline can fire, so abandoning that work to save a few seconds spends
    money for nothing. ``per_url_deadline_ms`` covers a cold fetch through the
    omnifetch provider waterfall *and* every LLM tier behind it, so it must be
    a multiple of a single tier's ``llm_timeout_ms`` rather than a peer of it.
    It is also the bound on a single pathological page: the stage waits for its
    slowest worker, so a very large value lets one bad URL hold the finished
    nineteen.

    ``concurrency`` defaults to ``top_n`` so the whole page set is resolved in
    one wave. A lower value splits it into waves, and the later waves start so
    close to the deadline that they time out -- paying for fetches that arrive
    too late to use. Matching the two costs no extra LLM calls.
    """

    model_config = _SETTINGS_MODEL_CONFIG

    mode: GroundingModeName = Field(
        default="auto", validation_alias="JASA_GROUNDING_MODE"
    )
    concurrency: int = Field(
        default=20, ge=1, validation_alias="JASA_GROUNDING_CONCURRENCY"
    )
    per_url_deadline_ms: int = Field(
        default=30000,
        ge=100,
        validation_alias="JASA_GROUNDING_PER_URL_DEADLINE_MS",
    )
    top_n: int = Field(
        default=20, ge=1, validation_alias="JASA_GROUNDING_TOP_N"
    )
    llm_base_url: str = Field(
        default="https://api.cerebras.ai/v1",
        validation_alias="JASA_GROUNDING_LLM_BASE_URL",
    )
    llm_model: str = Field(
        default="gpt-oss-120b", validation_alias="JASA_GROUNDING_LLM_MODEL"
    )
    llm_timeout_ms: int = Field(
        default=25000,
        ge=1000,
        validation_alias="JASA_GROUNDING_LLM_TIMEOUT_MS",
    )
    waterfall_path: str = Field(
        default="", validation_alias="JASA_GROUNDING_WATERFALL_PATH"
    )
    max_content_chars: int = Field(
        default=48000,
        ge=100,
        validation_alias="JASA_GROUNDING_MAX_CONTENT_CHARS",
    )


class CompositionSettings(BaseSettings):
    """Composition toggles."""

    model_config = _SETTINGS_MODEL_CONFIG

    expose_hello: bool = Field(
        default=False, validation_alias="JASA_EXPOSE_HELLO"
    )


class TelemetrySettings(BaseSettings):
    """OpenTelemetry settings read from the standard ``OTEL_`` variables.

    An empty ``otel_traces_exporter`` keeps tracing a no-op; set it to
    ``console`` or ``otlp`` to activate the OpenTelemetry SDK.
    """

    model_config = _SETTINGS_MODEL_CONFIG

    otel_sdk_disabled: bool = Field(
        default=False, validation_alias="OTEL_SDK_DISABLED"
    )
    otel_service_name: str = Field(
        default="jasa", validation_alias="OTEL_SERVICE_NAME"
    )
    otel_traces_exporter: OtelExporterName = Field(
        default="", validation_alias="OTEL_TRACES_EXPORTER"
    )
    otel_exporter_otlp_endpoint: str = Field(
        default="", validation_alias="OTEL_EXPORTER_OTLP_ENDPOINT"
    )
    otel_exporter_otlp_protocol: OtelProtocolName = Field(
        default="http/protobuf", validation_alias="OTEL_EXPORTER_OTLP_PROTOCOL"
    )


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Frozen aggregate of all settings, passed explicitly through the app."""

    server: ServerSettings
    cache: CacheSettings
    search: SearchSettings
    grounding: GroundingSettings
    composition: CompositionSettings
    telemetry: TelemetrySettings


def load_config(**server_overrides: Any) -> AppConfig:
    """Read configuration from the environment into a frozen ``AppConfig``.

    ``server_overrides`` (parsed CLI flags) take precedence over the environment
    for server settings only.
    """
    return AppConfig(
        server=ServerSettings(**server_overrides),
        cache=CacheSettings(),
        search=SearchSettings(),
        grounding=GroundingSettings(),
        composition=CompositionSettings(),
        telemetry=TelemetrySettings(),
    )
