"""Deterministic, evidence-preserving Review Aspect baseline."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from yelp_agent.config import ReviewAspectConfig, ReviewAspectVocabulary
from yelp_agent.reviews.schema import (
    ASPECT_NAMES,
    AspectName,
    AspectSentiment,
    ReviewAspectRecord,
    ReviewDocument,
)

_ABBREVIATIONS = frozenset({"dr.", "e.g.", "i.e.", "mr.", "mrs.", "ms.", "st.", "vs."})
_CLAUSE_BOUNDARY = re.compile(
    r"\s*[,;:]?\s+\b(?:but|however|though|yet)\b\s*[,;:]?\s+",
    flags=re.IGNORECASE,
)
_TOKEN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
_NEGATORS = frozenset(
    {
        "not",
        "never",
        "hardly",
        "no",
        "isn't",
        "wasn't",
        "weren't",
        "don't",
        "doesn't",
        "didn't",
    }
)
_TRIM_CHARACTERS = " \t\r\n,;:.!?\"'()[]{}"


@dataclass(frozen=True, slots=True)
class _TextSpan:
    text: str
    start: int
    end: int


def _trim_span(text: str, start: int, end: int) -> _TextSpan | None:
    while start < end and text[start] in _TRIM_CHARACTERS:
        start += 1
    while end > start and text[end - 1] in _TRIM_CHARACTERS:
        end -= 1
    if start >= end:
        return None
    return _TextSpan(text=text[start:end], start=start, end=end)


def _token_before_period(text: str, period_index: int) -> str:
    start = period_index
    while start > 0 and (text[start - 1].isalpha() or text[start - 1] == "."):
        start -= 1
    return text[start : period_index + 1].casefold()


def _sentence_spans(text: str) -> tuple[_TextSpan, ...]:
    spans: list[_TextSpan] = []
    start = 0
    index = 0
    while index < len(text):
        character = text[index]
        if character in "\r\n":
            span = _trim_span(text, start, index)
            if span is not None:
                spans.append(span)
            while index < len(text) and text[index] in "\r\n":
                index += 1
            start = index
            continue
        if character not in ".!?":
            index += 1
            continue
        if character == "." and (
            _token_before_period(text, index) in _ABBREVIATIONS
            or (
                index > 0
                and index + 1 < len(text)
                and text[index - 1].isdigit()
                and text[index + 1].isdigit()
            )
        ):
            index += 1
            continue
        end = index + 1
        while end < len(text) and text[end] in ".!?\"')]}":
            end += 1
        if end == len(text) or text[end].isspace():
            span = _trim_span(text, start, end)
            if span is not None:
                spans.append(span)
            while end < len(text) and text[end].isspace():
                end += 1
            start = end
            index = end
            continue
        index += 1
    tail = _trim_span(text, start, len(text))
    if tail is not None:
        spans.append(tail)
    return tuple(spans)


def _clause_spans(text: str) -> tuple[_TextSpan, ...]:
    clauses: list[_TextSpan] = []
    for sentence in _sentence_spans(text):
        clause_start = sentence.start
        for match in _CLAUSE_BOUNDARY.finditer(sentence.text):
            boundary_start = sentence.start + match.start()
            span = _trim_span(text, clause_start, boundary_start)
            if span is not None:
                clauses.append(span)
            clause_start = sentence.start + match.end()
        span = _trim_span(text, clause_start, sentence.end)
        if span is not None:
            clauses.append(span)
    return tuple(clauses)


def segment_review_clauses(text: str) -> tuple[str, ...]:
    """Return deterministic evidence clauses without changing source text."""

    return tuple(span.text for span in _clause_spans(text))


def _term_pattern(terms: list[str]) -> str:
    escaped = sorted((re.escape(term) for term in terms), key=len, reverse=True)
    return rf"(?<!\w)(?:{'|'.join(escaped)})(?!\w)"


def _compile_vocabulary(
    vocabulary: ReviewAspectVocabulary,
    sentiment: str,
) -> re.Pattern[str]:
    alternatives = []
    for aspect in ASPECT_NAMES:
        terms = getattr(vocabulary.aspects[aspect], sentiment)
        alternatives.append(f"(?P<{aspect}>{_term_pattern(terms)})")
    return re.compile("|".join(alternatives), flags=re.IGNORECASE)


class RuleBasedAspectExtractor:
    """Extract exact, deterministic evidence through one small interface."""

    def __init__(
        self,
        config: ReviewAspectConfig,
        vocabulary: ReviewAspectVocabulary,
    ) -> None:
        self._config = config
        self._patterns = {
            "positive": _compile_vocabulary(vocabulary, "positive"),
            "negative": _compile_vocabulary(vocabulary, "negative"),
        }
        self._aspect_order = {
            aspect: index for index, aspect in enumerate(ASPECT_NAMES)
        }

    def _is_negated(self, clause: str, match_start: int) -> bool:
        preceding = [
            match.group(0).casefold() for match in _TOKEN.finditer(clause[:match_start])
        ]
        window = preceding[-self._config.negation_window_tokens :]
        return any(token in _NEGATORS for token in window)

    def extract(
        self,
        review: ReviewDocument,
    ) -> tuple[ReviewAspectRecord, ...]:
        source_hash = hashlib.sha256(review.text.encode("utf-8")).hexdigest()
        records: list[ReviewAspectRecord] = []
        for clause in _clause_spans(review.text):
            evidence: dict[AspectName, set[AspectSentiment]] = {}
            negated: set[AspectName] = set()
            for raw_sentiment, pattern in self._patterns.items():
                for match in pattern.finditer(clause.text):
                    aspect = match.lastgroup
                    if aspect is None:
                        continue
                    typed_aspect: AspectName = aspect  # type: ignore[assignment]
                    sentiment: AspectSentiment = raw_sentiment  # type: ignore[assignment]
                    if self._is_negated(clause.text, match.start()):
                        negated.add(typed_aspect)
                        sentiment = (
                            "negative" if raw_sentiment == "positive" else "positive"
                        )
                    evidence.setdefault(typed_aspect, set()).add(sentiment)

            for aspect in sorted(evidence, key=self._aspect_order.__getitem__):
                sentiments = evidence[aspect]
                if len(sentiments) == 1:
                    sentiment = next(iter(sentiments))
                    confidence = (
                        self._config.negated_confidence
                        if aspect in negated
                        else self._config.rule_confidence
                    )
                else:
                    sentiment = "mixed"
                    confidence = self._config.mixed_confidence
                records.append(
                    ReviewAspectRecord(
                        review_id=review.review_id,
                        business_id=review.business_id,
                        user_id=review.user_id,
                        review_time=review.review_time,
                        aspect=aspect,
                        sentiment=sentiment,
                        confidence=confidence,
                        evidence_span=clause.text,
                        evidence_start=clause.start,
                        evidence_end=clause.end,
                        source_text_sha256=source_hash,
                        extractor_name=self._config.extractor_name,
                        extractor_version=self._config.extractor_version,
                    )
                )
        return tuple(records)
