"""Grounding detectors: junk-content tiers, sentinel tiers, fence repair.

Ported verbatim from omnisearch ``grounded_prompts.ts``. Junk detection runs
BEFORE the LLM call; sentinel detection runs AFTER. The two-tier junk system
(tight always fires, ambiguous only on short bodies) prevents false positives on
long-form prose that legitimately mentions paywall phrases.
"""

from __future__ import annotations

import re

_JUNK_AMBIGUOUS_MAX_CONTENT_CHARS = 3000
_SENTINEL_SUBSTRING_MAX_CHARS = 200

_JUNK_TIGHT_PATTERNS = (
    "subscribe to continue reading",
    "subscribe to read",
    "create a free account to continue",
    "create an account to continue",
    "log in to read",
    "log in to continue",
    "sign in to continue",
    "sign up to continue",
    "sign up to read",
    "this content is for members only",
    "this content is for subscribers",
    "register to continue",
    "register to read",
    "unlock this article",
    "please enable javascript",
    "javascript is required",
    "javascript must be enabled",
    "this site requires javascript",
    "enable cookies to continue",
    "please enable cookies",
    "cf-browser-verification",
    "checking your browser",
    "unusual activity from your network",
    "verify you are not a robot",
    "verify you're not a robot",
    "verify you are a human",
    "verify you're a human",
    "press and hold to confirm",
    "press & hold to confirm",
    "recaptcha verification",
    "hcaptcha challenge",
)

_JUNK_AMBIGUOUS_PATTERNS = (
    "access denied",
    "before accessing",
    "security check",
    "browser security check",
    "human verification",
    "just a moment",
    "before you continue to",
    "are you a human",
    "become a member",
)

_SENTINELS = (
    "[no usable content]",
    "[navigation only]",
    "[page not found]",
    "[search results page]",
    "[login required]",
)

_SENTINEL_NORMALIZE = re.compile(r"""^[\s*_"'`]+|[\s*_"'`.,;:!?]+$""")
_FENCE_LINE = re.compile(r"^[ ]{0,3}```", re.MULTILINE)


def grounding_detector_semantics() -> dict[str, object]:
    """Return every detector constant that affects grounding output."""
    return {
        "fence_line_flags": _FENCE_LINE.flags,
        "fence_line_pattern": _FENCE_LINE.pattern,
        "junk_ambiguous_max_content_chars": (_JUNK_AMBIGUOUS_MAX_CONTENT_CHARS),
        "junk_ambiguous_patterns": _JUNK_AMBIGUOUS_PATTERNS,
        "junk_tight_patterns": _JUNK_TIGHT_PATTERNS,
        "sentinel_normalize_flags": _SENTINEL_NORMALIZE.flags,
        "sentinel_normalize_pattern": _SENTINEL_NORMALIZE.pattern,
        "sentinel_substring_max_chars": _SENTINEL_SUBSTRING_MAX_CHARS,
        "sentinels": _SENTINELS,
    }


def detect_grounded_junk(content: str) -> str | None:
    """Return a junk identifier if the content is a wall/shell, else None."""
    if not content:
        return "empty_body"
    lower = content.lower()
    for pattern in _JUNK_TIGHT_PATTERNS:
        if pattern in lower:
            return f"pattern:{pattern}"
    if len(content) <= _JUNK_AMBIGUOUS_MAX_CONTENT_CHARS:
        for pattern in _JUNK_AMBIGUOUS_PATTERNS:
            if pattern in lower:
                return f"pattern:{pattern}"
    return None


def detect_grounded_sentinel(snippet: str) -> str | None:
    """Return the sentinel string if the snippet is a sentinel, else None."""
    normalized = _SENTINEL_NORMALIZE.sub("", snippet.strip().lower()).strip()
    for sentinel in _SENTINELS:
        if normalized == sentinel:
            return sentinel
    if len(normalized) <= _SENTINEL_SUBSTRING_MAX_CHARS:
        for sentinel in _SENTINELS:
            if sentinel in normalized:
                return sentinel
    return None


def repair_unbalanced_fence(snippet: str) -> str:
    """Close an unbalanced triple-backtick fence."""
    fence_count = len(_FENCE_LINE.findall(snippet))
    if fence_count % 2 == 1:
        return snippet + "\n```"
    return snippet
