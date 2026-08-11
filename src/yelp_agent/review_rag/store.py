"""DuckDB-backed exact-cutoff Review passage store."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from pathlib import Path
from threading import Lock

import duckdb
import pyarrow.parquet as pq

from yelp_agent.reviews.schema import REVIEW_ASPECT_SCHEMA

from .config import ReviewRAGConfig
from .schema import REVIEW_SEGMENT_SCHEMA, ReviewSegment, SegmentAspectEvidence


class ReviewRAGStore:
    """Hide scope filtering, cutoff enforcement, and Aspect joins behind one seam."""

    def __init__(
        self,
        segments_path: str | Path,
        aspect_records_path: str | Path,
        config: ReviewRAGConfig,
    ) -> None:
        segments = Path(segments_path)
        aspects = Path(aspect_records_path)
        if not segments.is_file() or not aspects.is_file():
            raise FileNotFoundError("Review RAG segment or Aspect artifact is missing")
        if not pq.ParquetFile(segments).schema_arrow.equals(
            REVIEW_SEGMENT_SCHEMA, check_metadata=False
        ):
            raise ValueError("Review segment schema is incompatible")
        if not pq.ParquetFile(aspects).schema_arrow.equals(
            REVIEW_ASPECT_SCHEMA, check_metadata=False
        ):
            raise ValueError("Review Aspect schema is incompatible")
        self._config = config
        self._connection = duckdb.connect(database=":memory:")
        self._lock = Lock()
        segment_sql = str(segments.resolve()).replace("'", "''")
        aspect_sql = str(aspects.resolve()).replace("'", "''")
        self._connection.execute(
            f"CREATE VIEW review_segments AS SELECT * FROM read_parquet('{segment_sql}')"
        )
        self._connection.execute(
            f"CREATE VIEW review_aspects AS SELECT * FROM read_parquet('{aspect_sql}')"
        )

    def close(self) -> None:
        self._connection.close()

    def segments_before(
        self,
        business_id: str,
        cutoff_time: datetime,
    ) -> tuple[ReviewSegment, ...]:
        if not business_id:
            raise ValueError("business_id cannot be empty")
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT * FROM review_segments
                WHERE business_id = ? AND review_time < ?
                ORDER BY review_time DESC, review_id, segment_index
                LIMIT ?
                """,
                [
                    business_id,
                    cutoff_time,
                    self._config.max_store_segments_per_business,
                ],
            ).to_arrow_table().to_pylist()
        result = tuple(ReviewSegment.model_validate(row) for row in rows)
        if any(
            row.business_id != business_id or row.review_time >= cutoff_time
            for row in result
        ):
            raise RuntimeError("Review store violated business scope or cutoff")
        return result

    def aspect_evidence(
        self,
        segments: tuple[ReviewSegment, ...],
        *,
        aspects: list[str],
        cutoff_time: datetime,
    ) -> dict[str, tuple[SegmentAspectEvidence, ...]]:
        if not segments or not aspects:
            return {}
        business_ids = sorted({item.business_id for item in segments})
        review_ids = sorted({item.review_id for item in segments})
        business_marks = ",".join("?" for _ in business_ids)
        review_marks = ",".join("?" for _ in review_ids)
        aspect_marks = ",".join("?" for _ in aspects)
        with self._lock:
            rows = self._connection.execute(
                f"""
                SELECT review_id, business_id, review_time, aspect, sentiment,
                       confidence, evidence_span, evidence_start, evidence_end
                FROM review_aspects
                WHERE business_id IN ({business_marks})
                  AND review_id IN ({review_marks})
                  AND aspect IN ({aspect_marks})
                  AND review_time < ? AND confidence >= ?
                ORDER BY review_id, confidence DESC, aspect
                """,
                [
                    *business_ids,
                    *review_ids,
                    *aspects,
                    cutoff_time,
                    self._config.minimum_aspect_confidence,
                ],
            ).to_arrow_table().to_pylist()
        segments_by_review: defaultdict[str, list[ReviewSegment]] = defaultdict(list)
        for segment in segments:
            segments_by_review[segment.review_id].append(segment)
        result: defaultdict[str, list[SegmentAspectEvidence]] = defaultdict(list)
        for row in rows:
            candidates = segments_by_review[str(row["review_id"])]
            start = int(row["evidence_start"])
            end = int(row["evidence_end"])
            containing = next(
                (
                    segment
                    for segment in candidates
                    if segment.char_start < end and segment.char_end > start
                ),
                candidates[0] if candidates else None,
            )
            if containing is None:
                continue
            result[containing.segment_id].append(
                SegmentAspectEvidence(
                    aspect=str(row["aspect"]),  # type: ignore[arg-type]
                    sentiment=str(row["sentiment"]),  # type: ignore[arg-type]
                    confidence=float(row["confidence"]),
                    evidence_span=str(row["evidence_span"]),
                )
            )
        return {key: tuple(value) for key, value in result.items()}
