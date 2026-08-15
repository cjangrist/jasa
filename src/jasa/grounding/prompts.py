"""Grounding prompts: the verbatim system prompt + user-message builder.

The system prompt is frozen from the pinned omnisearch commit
(``grounded_prompts.ts:30-110``) and shipped as package data
(``system_prompt.txt``). A test pins its SHA-256 so any accidental edit fails
the build. The user message format is also ported verbatim.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

_PROMPT_FILE = Path(__file__).resolve().parent / "system_prompt.txt"
SYSTEM_PROMPT = _PROMPT_FILE.read_text(encoding="utf-8")
SYSTEM_PROMPT_SHA256 = hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()

SNIPPET_MAX_CHARS = 2000
GROUNDING_MAX_TOKENS = 512
CONTENT_TRUNCATION_MARKER = "\n\n[content truncated]"
USER_MESSAGE_TEMPLATE = (
    "Query: {query}\n\nPage title: {title}\n\nPage content:\n{content}"
)


def grounding_prompt_semantics() -> dict[str, str]:
    """Return prompt constants that affect the effective LLM input."""
    return {
        "content_truncation_marker": CONTENT_TRUNCATION_MARKER,
        "system_prompt_sha256": SYSTEM_PROMPT_SHA256,
        "user_message_template": USER_MESSAGE_TEMPLATE,
    }


def build_grounded_user_message(
    query: str, title: str, content: str, max_chars: int
) -> str:
    """Build the user message for the snippet-writing LLM call."""
    truncated = (
        content[:max_chars] + CONTENT_TRUNCATION_MARKER
        if len(content) > max_chars
        else content
    )
    title_display = title or "(untitled)"
    return USER_MESSAGE_TEMPLATE.format(
        query=query,
        title=title_display,
        content=truncated,
    )
