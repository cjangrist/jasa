"""Grounding prompts: the verbatim system prompt + user-message builder.

The system prompt is frozen from the pinned omnisearch commit
(``grounded_prompts.ts:30-110``) and shipped as package data
(``system_prompt.txt``). A test pins its SHA-256 so any accidental edit fails
the build. The user message format is also ported verbatim.
"""

from __future__ import annotations

from pathlib import Path

_PROMPT_FILE = Path(__file__).resolve().parent / "system_prompt.txt"
SYSTEM_PROMPT = _PROMPT_FILE.read_text(encoding="utf-8")

SNIPPET_MAX_CHARS = 2000
_CONTENT_TRUNCATION_MARKER = "\n\n[content truncated]"


def build_grounded_user_message(
    query: str, title: str, content: str, max_chars: int
) -> str:
    """Build the user message for the snippet-writing LLM call."""
    truncated = (
        content[:max_chars] + _CONTENT_TRUNCATION_MARKER
        if len(content) > max_chars
        else content
    )
    title_display = title or "(untitled)"
    return (
        f"Query: {query}\n\n"
        f"Page title: {title_display}\n\n"
        f"Page content:\n{truncated}"
    )
