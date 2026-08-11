"""Deterministic paragraph/sentence-aware Review segmentation."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from .config import ReviewRAGConfig
from .schema import ReviewSegment


_BOUNDARY = re.compile(r"(?<=[.!?。！？])(?:\s+|$)|\n+")


@dataclass(frozen=True, slots=True)
class SegmentBuild:
    segments: tuple[ReviewSegment, ...]
    truncated: bool


def segment_review(row: dict[str, object], config: ReviewRAGConfig) -> SegmentBuild:
    text = str(row.get("text") or "")
    if not text.strip():
        return SegmentBuild(segments=(), truncated=False)
    spans = _sentence_spans(text, config.segment_max_chars)
    chunks = _combine_spans(text, spans, config.segment_max_chars)
    truncated = len(chunks) > config.max_segments_per_review
    chunks = chunks[: config.max_segments_per_review]
    review_hash = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
    review_id = str(row["review_id"])
    segments: list[ReviewSegment] = []
    for index, (start, end) in enumerate(chunks):
        raw = text[start:end]
        left = len(raw) - len(raw.lstrip())
        right = len(raw.rstrip())
        start += left
        end = start + max(0, right - left)
        value = text[start:end]
        if len(value) < config.segment_min_chars and segments:
            continue
        text_hash = hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()
        segment_id = hashlib.sha256(
            f"{review_id}\x1f{index}\x1f{start}\x1f{end}\x1f{text_hash}".encode()
        ).hexdigest()
        segments.append(
            ReviewSegment(
                segment_id=segment_id,
                review_id=review_id,
                business_id=str(row["business_id"]),
                user_id=str(row["user_id"]),
                review_time=row["date"],  # type: ignore[arg-type]
                stars=float(row["stars"]),
                useful=int(row["useful"]),
                segment_index=index,
                char_start=start,
                char_end=end,
                text=value,
                text_sha256=text_hash,
                review_text_sha256=review_hash,
            )
        )
    return SegmentBuild(segments=tuple(segments), truncated=truncated)


def _sentence_spans(text: str, max_chars: int) -> list[tuple[int, int]]:
    points = [0]
    points.extend(match.end() for match in _BOUNDARY.finditer(text))
    if points[-1] != len(text):
        points.append(len(text))
    result: list[tuple[int, int]] = []
    for start, end in zip(points, points[1:]):
        if end <= start:
            continue
        cursor = start
        while end - cursor > max_chars:
            split = text.rfind(" ", cursor, cursor + max_chars + 1)
            if split <= cursor:
                split = cursor + max_chars
            result.append((cursor, split))
            cursor = split
        if end > cursor:
            result.append((cursor, end))
    return result


def _combine_spans(
    text: str,
    spans: list[tuple[int, int]],
    max_chars: int,
) -> list[tuple[int, int]]:
    if not spans:
        return []
    result: list[tuple[int, int]] = []
    start, end = spans[0]
    for next_start, next_end in spans[1:]:
        if next_end - start <= max_chars:
            end = next_end
        else:
            if text[start:end].strip():
                result.append((start, end))
            start, end = next_start, next_end
    if text[start:end].strip():
        result.append((start, end))
    return result
