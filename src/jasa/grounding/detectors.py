"""Grounding detectors: junk-content tiers, sentinel tiers, fence repair.

Ported verbatim from omnisearch ``grounded_prompts.ts``. Junk detection runs
BEFORE the LLM call; sentinel detection runs AFTER. The two-tier junk system
(tight always fires, ambiguous only on short bodies) prevents false positives on
long-form prose that legitimately mentions paywall phrases.
"""

from __future__ import annotations

import re

from jasa.grounding.prompts import COVERAGE_LINE_MAX_CHARS

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
_COVERAGE_LINE = re.compile(
    r"Coverage: answers .+; does NOT cover .+[.!?。！？]$"  # noqa: RUF001
)
FENCE_REPAIR_SUFFIX = "\n```"

# The full-width stops are deliberate: a snippet written in the query's
# language ends in them, and their ASCII lookalikes would never match. Every
# closing mark that can follow a terminator is kept with it, so a trim cannot
# strand the opening half of a quotation.
_ASCII_TERMINATORS = r"[.!?]"
_WIDE_TERMINATORS = r"[。！？]"  # noqa: RUF001
_CLOSERS = r"""["'`)\]”’）】」』]"""  # noqa: RUF001
_SENTENCE_BOUNDARY = re.compile(
    rf"{_ASCII_TERMINATORS}{_CLOSERS}*(?=\s|$)"
    rf"|{_WIDE_TERMINATORS}{_CLOSERS}*"
)
TRUNCATION_TRIM_MAX_CHARS = 400


def grounding_detector_semantics() -> dict[str, object]:
    """Return every detector constant that affects grounding output."""
    return {
        "fence_line_flags": _FENCE_LINE.flags,
        "fence_line_pattern": _FENCE_LINE.pattern,
        "fence_repair_suffix": FENCE_REPAIR_SUFFIX,
        "coverage_line_max_chars": COVERAGE_LINE_MAX_CHARS,
        "coverage_line_pattern": _COVERAGE_LINE.pattern,
        "junk_ambiguous_max_content_chars": (_JUNK_AMBIGUOUS_MAX_CONTENT_CHARS),
        "junk_ambiguous_patterns": _JUNK_AMBIGUOUS_PATTERNS,
        "junk_tight_patterns": _JUNK_TIGHT_PATTERNS,
        "sentence_boundary_pattern": _SENTENCE_BOUNDARY.pattern,
        "sentinel_normalize_flags": _SENTINEL_NORMALIZE.flags,
        "sentinel_normalize_pattern": _SENTINEL_NORMALIZE.pattern,
        "sentinel_substring_max_chars": _SENTINEL_SUBSTRING_MAX_CHARS,
        "sentinels": _SENTINELS,
        "truncation_trim_max_chars": TRUNCATION_TRIM_MAX_CHARS,
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


def complete_coverage_line(snippet: str) -> str | None:
    """Return a valid final coverage line, or None when it is incomplete."""
    marker_index = snippet.rfind("Coverage:")
    if marker_index < 0:
        return None
    coverage_line = snippet[marker_index:].strip()
    if len(coverage_line) <= COVERAGE_LINE_MAX_CHARS and bool(
        _COVERAGE_LINE.fullmatch(coverage_line)
    ):
        return coverage_line
    return None


def has_complete_coverage_line(snippet: str) -> bool:
    """Return whether a normal snippet ends with the required coverage line."""
    return complete_coverage_line(snippet) is not None


def repair_unbalanced_fence(snippet: str) -> str:
    """Close an unbalanced triple-backtick fence."""
    fence_count = len(_FENCE_LINE.findall(snippet))
    if fence_count % 2 == 1:
        return snippet + FENCE_REPAIR_SUFFIX
    return snippet


def trim_truncated_snippet(snippet: str) -> str:
    """Cut a generation stopped at its token ceiling back to a clean end.

    A snippet the model never finished ends mid-word and, because the closing
    Coverage line is written last, without it. Nothing can recover the missing
    text -- a second tier would hit the same ceiling -- so the aim is only to
    avoid publishing a fragment that stops mid-thought.

    A snippet containing a fence is left alone: cutting at a sentence boundary
    inside a code block would corrupt the code, and ``repair_unbalanced_fence``
    already closes what was cut. The trim is also abandoned when it would
    discard more than ``TRUNCATION_TRIM_MAX_CHARS``, because losing a long tail
    of real evidence is worse than an unpolished ending.

    A boundary match always consumes its terminating character, so the slice
    always retains at least that character and cannot strip down to nothing.

    CJK terminators are matched without the trailing-whitespace lookahead the
    ASCII ones require, because CJK text does not space its sentences apart.
    The prompt asks for the snippet in the query's language, so a Japanese or
    Chinese answer that ends in a full-width stop must be trimmable too.
    """
    if "```" in snippet:
        return snippet
    boundaries = list(_SENTENCE_BOUNDARY.finditer(snippet))
    if not boundaries:
        return snippet
    trimmed = snippet[: boundaries[-1].end()].rstrip()
    if len(snippet.rstrip()) - len(trimmed) > TRUNCATION_TRIM_MAX_CHARS:
        return snippet
    return trimmed
