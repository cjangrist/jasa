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

# The prompt caps the snippet body at 2000 characters and then requires a
# closing Coverage line of up to 200 more, on its own line. Capping at 2000
# severed that closing line mid-sentence on the longest snippets -- the ones
# whose pages had the most to say. The separating newline counts too: without
# it a maximal body and a maximal Coverage line lose their final character,
# which reads as a cut generation and trims the whole Coverage line away,
# reproducing the very defect the larger cap exists to fix.
SNIPPET_BODY_MAX_CHARS = 2000
COVERAGE_LINE_MAX_CHARS = 200
COVERAGE_SEPARATOR_MAX_CHARS = 2
SNIPPET_MAX_CHARS = (
    SNIPPET_BODY_MAX_CHARS
    + COVERAGE_SEPARATOR_MAX_CHARS
    + COVERAGE_LINE_MAX_CHARS
)
# The system prompt permits 2000 characters plus a mandatory Coverage line of
# up to 200 more, and it requires the snippet to be written in the query's
# language. CJK text runs near one character per token, which is the worst
# ratio the contract has to survive, so the ceiling is sized for 2200
# characters at that rate. A 512-token cap could not reach the contract in any
# language: generations were cut mid-word and lost the Coverage line entirely.
# Raising it costs nothing for snippets that finish early, because billing
# follows the tokens actually generated rather than the cap.
WORST_CASE_CHARS_PER_TOKEN = 1
GROUNDING_MAX_TOKENS = 2400
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
