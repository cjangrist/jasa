"""MCP tool input/output models.

Pydantic models generate JSON schemas that forbid additional properties (the
``extra="forbid"`` config), satisfying the §13.3 schema meta-test.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

_QUERY_DESCRIPTION = "The search query, 1 to 2000 characters."
_TIMEOUT_DESCRIPTION = (
    "DO NOT SET unless latency is critical -- omitting uses the server's own"
    " budget, which leaves grounding the time it needs. The server applies its"
    " own fan-out cap either way, so the slowest providers may still be"
    " omitted. A short value here starves grounding and returns ungrounded"
    " snippets, or fails the call outright if the budget runs out before"
    " grounding begins."
)
_INCLUDE_SNIPPETS_DESCRIPTION = (
    "Include result snippets in the output (default true)."
)
_GROUNDED_DESCRIPTION = (
    "Regenerate snippets from fetched page content. Unset grounds when a key is"
    " configured; false disables; true requires a key."
)


class WebSearchInput(BaseModel):
    """Input model for the ``web_search`` MCP tool (port plan §4.1)."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        min_length=1, max_length=2000, description=_QUERY_DESCRIPTION
    )
    timeout_ms: int | None = Field(
        default=None, gt=0, description=_TIMEOUT_DESCRIPTION
    )
    include_snippets: bool = Field(
        default=True, description=_INCLUDE_SNIPPETS_DESCRIPTION
    )
    grounded_snippets: bool | None = Field(
        default=None, description=_GROUNDED_DESCRIPTION
    )


class _WebSearchOutputModel(BaseModel):
    """Immutable strict base for the public ``web_search`` response."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class WebSearchProviderSuccess(_WebSearchOutputModel):
    """One search provider that completed successfully."""

    provider: str
    duration_ms: int = Field(ge=0)


class WebSearchProviderFailure(_WebSearchOutputModel):
    """One isolated search-provider failure."""

    provider: str
    error: str
    duration_ms: int = Field(ge=0)


class WebSearchGrounding(_WebSearchOutputModel):
    """Grounding-stage request, completion, and outcome counts."""

    requested: bool
    attempted: int = Field(ge=0)
    grounded: int = Field(ge=0)
    outcomes: dict[str, int]


class WebSearchTruncation(_WebSearchOutputModel):
    """Counts for result truncation and tail rescue."""

    total_before: int = Field(ge=0)
    kept: int = Field(ge=0)
    rescued: int = Field(ge=0)


class WebSearchResult(_WebSearchOutputModel):
    """One fused, attributed, and optionally grounded search result."""

    title: str
    url: str
    source_providers: list[str]
    score: float
    snippet_source: Literal["aggregated", "grounded", "fallback"]
    snippets: list[str] | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )


class WebSearchResponse(_WebSearchOutputModel):
    """Complete machine-readable output contract for ``web_search``."""

    query: str
    total_duration_ms: int = Field(ge=0)
    providers_succeeded: list[WebSearchProviderSuccess]
    providers_failed: list[WebSearchProviderFailure]
    grounding: WebSearchGrounding
    truncation: WebSearchTruncation
    web_results: list[WebSearchResult]
