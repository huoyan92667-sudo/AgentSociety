"""按完整句子切评论，并让相邻片段共享少量上下文。"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from itertools import pairwise

from pydantic import Field

from yelp_agent.models import StrictModel
from yelp_agent.review_rag.schema import ReviewSegment

# 英文句号后通常有空格；中文标点后不一定有空格，所以两类边界分开写。
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])(?:\s+|$)|(?<=[。！？])|\n+")


class OverlapSegmentConfig(StrictModel):
    """评论切分规则；不设置每条评论的片段数量上限。"""

    max_chars: int = Field(default=900, ge=100, le=4000)
    overlap_sentences: int = Field(default=1, ge=0, le=3)


@dataclass(frozen=True, slots=True)
class OverlapSegmentBuild:
    """一条评论的全部片段及是否发生过超长单句拆分。"""

    segments: tuple[ReviewSegment, ...]
    split_long_sentence: bool


def segment_review_with_overlap(
    row: dict[str, object],
    config: OverlapSegmentConfig | None = None,
) -> OverlapSegmentBuild:
    """切分一条评论；保留完整句子、相邻上下文和评论尾部。"""

    config = config or OverlapSegmentConfig()
    text = str(row.get("text") or "")
    if not text.strip():
        return OverlapSegmentBuild(segments=(), split_long_sentence=False)

    sentence_spans, split_long_sentence = _sentence_spans(text, config.max_chars)
    chunks = _overlapping_chunks(
        text,
        sentence_spans,
        max_chars=config.max_chars,
        overlap_sentences=config.overlap_sentences,
    )
    review_hash = _sha256_text(text)
    review_id = str(row["review_id"])
    segments: list[ReviewSegment] = []
    for index, (raw_start, raw_end) in enumerate(chunks):
        start, end = _trim_span(text, raw_start, raw_end)
        if end <= start:
            continue
        value = text[start:end]
        text_hash = _sha256_text(value)
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
    return OverlapSegmentBuild(
        segments=tuple(segments),
        split_long_sentence=split_long_sentence,
    )


def _sentence_spans(text: str, max_chars: int) -> tuple[list[tuple[int, int]], bool]:
    """先找完整句子；只有单句过长时才在词语边界继续拆。"""

    points = [0]
    points.extend(match.end() for match in _SENTENCE_BOUNDARY.finditer(text))
    if points[-1] != len(text):
        points.append(len(text))
    spans: list[tuple[int, int]] = []
    split_long_sentence = False
    for start, end in pairwise(points):
        if end <= start or not text[start:end].strip():
            continue
        cursor = start
        while end - cursor > max_chars:
            split = _last_word_boundary(text, cursor, cursor + max_chars)
            spans.append((cursor, split))
            cursor = split
            split_long_sentence = True
        if end > cursor and text[cursor:end].strip():
            spans.append((cursor, end))
    return spans, split_long_sentence


def _overlapping_chunks(
    text: str,
    spans: list[tuple[int, int]],
    *,
    max_chars: int,
    overlap_sentences: int,
) -> list[tuple[int, int]]:
    """把相邻句子装进片段；下一片段重复上一片段末尾的一句。"""

    if not spans:
        return []
    result: list[tuple[int, int]] = []
    start_index = 0
    while start_index < len(spans):
        chunk_start = spans[start_index][0]
        end_index = start_index
        while end_index < len(spans):
            proposed_end = spans[end_index][1]
            if end_index > start_index and proposed_end - chunk_start > max_chars:
                break
            end_index += 1
        chunk_end = spans[end_index - 1][1]
        if text[chunk_start:chunk_end].strip():
            result.append((chunk_start, chunk_end))
        if end_index >= len(spans):
            break

        # 至少向前推进一句，避免一个接近上限的长句被无限重复。
        start_index = max(start_index + 1, end_index - overlap_sentences)
    return result


def _last_word_boundary(text: str, start: int, proposed_end: int) -> int:
    """超长单句优先在空白处切；没有空白时才按字符硬切。"""

    split = max(text.rfind(" ", start + 1, proposed_end + 1), text.rfind("\t", start + 1, proposed_end + 1))
    return split if split > start else proposed_end


def _trim_span(text: str, start: int, end: int) -> tuple[int, int]:
    raw = text[start:end]
    left = len(raw) - len(raw.lstrip())
    right = len(raw) - len(raw.rstrip())
    return start + left, end - right


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
