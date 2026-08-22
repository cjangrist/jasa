"""MCP tool input/output models.

Pydantic models generate JSON schemas that forbid additional properties (the
``extra="forbid"`` config), satisfying the §13.3 schema meta-test.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

_QUERY_DESCRIPTION = "The search query, 1 to 2000 characters."
_TIMEOUT_DESCRIPTION = (
    "DO NOT SET unless latency is critical -- omitting uses the server's own"
    " generous budget, which waits for every provider and leaves grounding the"
    " time it needs. A short value here starves grounding and returns"
    " ungrounded snippets."
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
