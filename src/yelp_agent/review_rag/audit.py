"""Offline artifact and hidden-label audit; never imported by Agent runtime."""

from __future__ import annotations

import hashlib
from pathlib import Path

import duckdb
import pyarrow.parquet as pq

from yelp_agent.agent_benchmark import (
    load_evidence_labels,
    load_visible_scenarios,
)

from .schema import REVIEW_SEGMENT_SCHEMA, ReviewRAGAuditReport


def audit_review_rag_artifacts(
    *,
    reviews_path: str | Path,
    segments_path: str | Path,
    visible_scenarios_path: str | Path,
    evidence_labels_path: str | Path,
) -> ReviewRAGAuditReport:
    reviews = Path(reviews_path)
    segments = Path(segments_path)
    if not pq.ParquetFile(segments).schema_arrow.equals(
        REVIEW_SEGMENT_SCHEMA, check_metadata=False
    ):
        raise ValueError("Review RAG segment schema is incompatible")
    connection = duckdb.connect(database=":memory:")
    try:
        source_count = int(
            connection.execute("SELECT count(*) FROM read_parquet(?)", [str(reviews)]).fetchone()[0]
        )
        row = connection.execute(
            """
            SELECT count(*), count(DISTINCT segment_id), count(DISTINCT review_id),
                   count(DISTINCT business_id),
                   sum(CASE WHEN char_end <= char_start THEN 1 ELSE 0 END)
            FROM read_parquet(?)
            """,
            [str(segments)],
        ).fetchone()
        segment_count = int(row[0])
        duplicate_ids = segment_count - int(row[1])
        distinct_reviews = int(row[2])
        distinct_businesses = int(row[3])
        invalid_offsets = int(row[4] or 0)
        source_without = int(
            connection.execute(
                """
                SELECT count(*) FROM read_parquet(?) r
                WHERE length(trim(r.text)) > 0 AND NOT EXISTS (
                    SELECT 1 FROM read_parquet(?) s WHERE s.review_id = r.review_id
                )
                """,
                [str(reviews), str(segments)],
            ).fetchone()[0]
        )
    finally:
        connection.close()
    labels = [
        item
        for item in load_evidence_labels(evidence_labels_path)
        if item.source_type == "review"
    ]
    scenarios = {
        item.scenario_id: item for item in load_visible_scenarios(visible_scenarios_path)
    }
    label_ids = {str(item.review_id) for item in labels}
    table = pq.read_table(
        reviews,
        filters=[("review_id", "in", sorted(label_ids))],
        columns=["review_id", "text"],
    ).to_pylist()
    source_hashes = {
        str(row["review_id"]): hashlib.sha256(
            str(row["text"]).encode("utf-8", errors="replace")
        ).hexdigest()
        for row in table
    }
    covered = sum(str(item.review_id) in source_hashes for item in labels)
    hash_matches = sum(
        source_hashes.get(str(item.review_id)) == item.source_text_sha256 for item in labels
    )
    cutoff_violations = sum(
        item.event_time is None
        or item.event_time >= scenarios[item.scenario_id].cutoff_time
        for item in labels
    )
    denominator = len(labels) or 1
    coverage = covered / denominator
    hash_rate = hash_matches / denominator
    passed = all(
        (
            duplicate_ids == 0,
            invalid_offsets == 0,
            source_without == 0,
            coverage == 1.0,
            hash_rate == 1.0,
            cutoff_violations == 0,
        )
    )
    return ReviewRAGAuditReport(
        source_review_count=source_count,
        segment_count=segment_count,
        distinct_review_count=distinct_reviews,
        distinct_business_count=distinct_businesses,
        duplicate_segment_ids=duplicate_ids,
        invalid_segment_offsets=invalid_offsets,
        source_reviews_without_segments=source_without,
        hidden_review_label_count=len(labels),
        hidden_review_id_coverage=coverage,
        hidden_hash_match_rate=hash_rate,
        cutoff_violation_count=cutoff_violations,
        passed=passed,
    )
