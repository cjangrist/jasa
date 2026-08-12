"""Snippet collapse, ported from omnisearch ``snippet_selector.ts``.

Given multiple provider snippets for one URL, select or merge into one optimal
snippet maximizing information density and query relevance. Validated against
``tests/fixtures/golden/snippets.json``.
"""

from __future__ import annotations

import dataclasses
import math
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jasa.search.ranking import RankedWebResult

MERGE_CHAR_BUDGET = 500
DIVERSITY_THRESHOLD = 0.3
MIN_WORDS_FOR_SCORE = 2
SENTENCE_NEAR_DUPLICATE_THRESHOLD = 0.7
MIN_CANDIDATES_FOR_MERGE = 2
_SENTENCE_MIN_LENGTH = 15
_LOG_LENGTH_BASE = 600

_ENTITY_FIXES = [
    ("&amp;", "&"),
    ("&lt;", "<"),
    ("&gt;", ">"),
    ("&quot;", '"'),
    ("&#39;", "'"),
]
_GENERAL_ENTITY = re.compile(r"&#?\w+;")
_WHITESPACE = re.compile(r"\s+")
_TRAILING_ELLIPSIS = re.compile(r"\.{3,}$")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])|[\n\r]+")
_WORD_SPLIT = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class _Sentence:
    text: str
    bigrams: frozenset[str]
    order: int


@dataclass(frozen=True, slots=True)
class _Candidate:
    original: str
    norm: str
    score: float


def normalize_snippet(snippet: str) -> str:
    """Apply the limited entity fix set, collapse whitespace, strip ellipsis."""
    fixed = snippet
    for entity, char in _ENTITY_FIXES:
        fixed = fixed.replace(entity, char)
    fixed = _GENERAL_ENTITY.sub("", fixed)
    fixed = _WHITESPACE.sub(" ", fixed)
    fixed = _TRAILING_ELLIPSIS.sub("", fixed)
    return fixed.strip()


def word_tokenize(text: str) -> list[str]:
    """Lowercase, split on whitespace, drop tokens of length 1 or less."""
    return [word for word in _WORD_SPLIT.split(text.lower()) if len(word) > 1]


def build_bigrams(words: list[str]) -> frozenset[str]:
    """Return the set of adjacent word bigrams."""
    return frozenset(
        f"{words[i]} {words[i + 1]}" for i in range(len(words) - 1)
    )


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    """Return the Jaccard similarity; NaN (matching JS) for an empty union."""
    intersection = len(a & b)
    union = len(a) + len(b) - intersection
    if union == 0:
        return float("nan")
    return intersection / union


def score_snippet(normalized: str, query_terms: list[str]) -> float:
    """Score a normalized snippet by density, relevance, and length."""
    words = word_tokenize(normalized)
    if len(words) < MIN_WORDS_FOR_SCORE:
        return 0.0
    bigrams = build_bigrams(words)
    trigrams = frozenset(
        f"{words[i]} {words[i + 1]} {words[i + 2]}"
        for i in range(len(words) - 2)
    )
    unique_ngrams = len(bigrams) + len(trigrams)
    density = unique_ngrams / len(normalized)
    snippet_lower = normalized.lower()
    query_hits = sum(1 for term in query_terms if term in snippet_lower)
    relevance = query_hits / len(query_terms) if query_terms else 0.0
    length_factor = min(
        1.0, math.log(len(normalized) + 1) / math.log(_LOG_LENGTH_BASE)
    )
    return density * (1 + 0.3 * relevance) * (0.7 + 0.3 * length_factor)


def split_sentences(text: str) -> list[str]:
    """Split on sentence boundaries or newlines, dropping short fragments."""
    return [
        segment.strip()
        for segment in _SENTENCE_SPLIT.split(text)
        if len(segment.strip()) > _SENTENCE_MIN_LENGTH
    ]


def _collect_sentences(snippets: list[str]) -> list[_Sentence]:
    sentences: list[_Sentence] = []
    order = 0
    for snippet in snippets:
        for sentence_text in split_sentences(snippet):
            bigrams = build_bigrams(word_tokenize(sentence_text))
            sentences.append(_Sentence(sentence_text, bigrams, order))
            order += 1
    return sentences


def _dedupe_sentences(sentences: list[_Sentence]) -> list[_Sentence]:
    deduped: list[_Sentence] = []
    for sentence in sentences:
        is_dupe = any(
            jaccard(existing.bigrams, sentence.bigrams)
            > SENTENCE_NEAR_DUPLICATE_THRESHOLD
            for existing in deduped
        )
        if not is_dupe:
            deduped.append(sentence)
    return deduped


def sentence_merge(snippets: list[str], budget: int) -> str:
    """Greedily set-cover complementary sentences within the char budget."""
    deduped = _dedupe_sentences(_collect_sentences(snippets))
    covered: set[str] = set()
    selected: list[_Sentence] = []
    remaining = budget

    while remaining > 0 and deduped:
        best_index = -1
        best_new_count = 0
        for index, candidate in enumerate(deduped):
            new_count = sum(
                1 for bigram in candidate.bigrams if bigram not in covered
            )
            if new_count > best_new_count:
                best_new_count = new_count
                best_index = index
        if best_index == -1 or best_new_count == 0:
            break

        best = deduped[best_index]
        if len(best.text) > remaining:
            deduped.pop(best_index)
            continue

        selected.append(best)
        covered.update(best.bigrams)
        remaining -= len(best.text)
        deduped.pop(best_index)

    selected.sort(key=lambda sentence: sentence.order)
    return " ".join(sentence.text for sentence in selected)


def select_best_snippet(snippets: list[str], query: str) -> str:
    """Select or merge the best single snippet from candidates."""
    query_terms = word_tokenize(query)
    candidates = sorted(
        (
            _Candidate(snippet, norm, score_snippet(norm, query_terms))
            for snippet, norm in (
                (snippet, normalize_snippet(snippet)) for snippet in snippets
            )
        ),
        key=lambda candidate: candidate.score,
        reverse=True,
    )
    primary = candidates[0]
    if len(candidates) < MIN_CANDIDATES_FOR_MERGE:
        return primary.original
    runner_up = candidates[1]
    if runner_up.score < primary.score * 0.3:
        return primary.original

    primary_bigrams = build_bigrams(word_tokenize(primary.norm))
    runner_up_bigrams = build_bigrams(word_tokenize(runner_up.norm))
    similarity = jaccard(primary_bigrams, runner_up_bigrams)
    if similarity < DIVERSITY_THRESHOLD:
        merged = sentence_merge(
            [primary.original, runner_up.original], MERGE_CHAR_BUDGET
        )
        return merged or primary.original
    return primary.original


def collapse_snippets(
    results: list[RankedWebResult], query: str
) -> list[RankedWebResult]:
    """Reduce each result's snippets to a single best snippet."""
    return [
        dataclasses.replace(
            result,
            snippets=(
                result.snippets
                if len(result.snippets) <= 1
                else [select_best_snippet(result.snippets, query)]
            ),
        )
        for result in results
    ]
