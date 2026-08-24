from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from yelp_agent.data.reviews import REVIEW_SCHEMA
from yelp_agent.recommendation_v2.review_features import (
    CANDIDATE_ASPECT_SCHEMA,
    CANDIDATE_REVIEW_SCHEMA,
    merge_review_candidates,
)


def _review_row(review_id: str, text: str) -> dict[str, object]:
    return {
        "review_id": review_id,
        "user_id": f"user-{review_id}",
        "business_id": "b1",
        "stars": 4.0,
        "useful": 1,
        "funny": 0,
        "cool": 0,
        "text": text,
        "date": datetime(2025, 1, 1, tzinfo=UTC),
    }


def _aspect_row(
    review_id: str,
    aspect: str,
    *,
    keyword: bool,
    semantic: bool,
    term: str | None = None,
    score: float | None = None,
) -> dict[str, object]:
    return {
        "review_id": review_id,
        "business_id": "b1",
        "aspect": aspect,
        "keyword_hit": keyword,
        "semantic_hit": semantic,
        "matched_terms": [] if term is None else [term],
        "semantic_score": score,
        "matched_anchor_ids": [] if score is None else [f"{aspect}:0"],
    }


def test_merges_routes_and_keeps_each_full_review_once(tmp_path: Path) -> None:
    reviews_path = tmp_path / "reviews.parquet"
    output_root = tmp_path / "features"
    output_root.mkdir()
    pq.write_table(
        pa.Table.from_pylist(
            [
                _review_row("r1", "The room was quiet."),
                _review_row("r2", "We drove around before finding a space."),
                _review_row("outside", "This review was not recalled."),
            ],
            schema=REVIEW_SCHEMA,
        ),
        reviews_path,
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                _aspect_row(
                    "r1",
                    "quiet_environment",
                    keyword=True,
                    semantic=False,
                    term="quiet",
                )
            ],
            schema=CANDIDATE_ASPECT_SCHEMA,
        ),
        output_root / "keyword_candidate_aspects.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                _aspect_row(
                    "r1",
                    "quiet_environment",
                    keyword=False,
                    semantic=True,
                    score=0.9,
                ),
                _aspect_row(
                    "r2",
                    "parking",
                    keyword=False,
                    semantic=True,
                    score=0.8,
                ),
            ],
            schema=CANDIDATE_ASPECT_SCHEMA,
        ),
        output_root / "semantic_candidate_aspects.parquet",
    )
    (output_root / "keyword_manifest.json").write_text("{}\n", encoding="utf-8")
    (output_root / "semantic_manifest.json").write_text("{}\n", encoding="utf-8")

    result = merge_review_candidates(reviews_path, output_root)
    reviews = pq.read_table(output_root / "candidate_reviews.parquet")
    aspects = pq.read_table(output_root / "candidate_aspects.parquet")
    rows = {(row["review_id"], row["aspect"]): row for row in aspects.to_pylist()}

    assert result.manifest.candidate_review_count == 2
    assert result.manifest.candidate_aspect_count == 2
    assert result.manifest.route_counts == {"both": 1, "semantic_only": 1}
    assert reviews.schema.equals(CANDIDATE_REVIEW_SCHEMA, check_metadata=False)
    assert aspects.schema.equals(CANDIDATE_ASPECT_SCHEMA, check_metadata=False)
    assert set(reviews.column("review_id").to_pylist()) == {"r1", "r2"}
    assert rows[("r1", "quiet_environment")]["keyword_hit"] is True
    assert rows[("r1", "quiet_environment")]["semantic_hit"] is True
    assert rows[("r1", "quiet_environment")]["matched_terms"] == ["quiet"]
    assert rows[("r2", "parking")]["keyword_hit"] is False
