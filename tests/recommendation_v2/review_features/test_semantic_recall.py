from __future__ import annotations

import hashlib
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from yelp_agent.recommendation_v2.review_features import (
    CANDIDATE_ASPECT_SCHEMA,
    build_semantic_review_candidates,
)
from yelp_agent.recommendation_v2.review_index import (
    REVIEW_INDEX_SEGMENT_SCHEMA,
    ReviewVectorIndexManifest,
)
from yelp_agent.semantic_embedding.encoder import EncodedBatch
from yelp_agent.semantic_embedding.schema import EmbeddingInputType

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class AnchorEncoder:
    provider = "fake"
    model = "fake-review-encoder"
    dimension = 3
    batch_size = 8

    def encode(
        self,
        texts: Sequence[str],
        *,
        input_type: EmbeddingInputType,
    ) -> EncodedBatch:
        assert input_type == "query"
        vectors: list[np.ndarray] = []
        for text in texts:
            value = text.casefold()
            if any(
                term in value
                for term in ("quiet", "loud", "noisy", "conversation", "hear")
            ):
                vector = [1.0, 0.0, 0.0]
            elif "park" in value:
                vector = [0.0, 1.0, 0.0]
            else:
                vector = [0.0, 0.0, 1.0]
            vectors.append(np.asarray(vector, dtype=np.float32))
        return EncodedBatch(
            vectors=tuple(vectors),
            model=self.model,
            input_tokens=len(texts),
            request_id="fake",
            latency_ms=0.0,
            per_text_input_tokens=tuple(1 for _ in texts),
        )


def _segment_row(
    row_index: int,
    review_id: str,
    text: str,
) -> dict[str, object]:
    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return {
        "row_index": row_index,
        "segment_id": f"{row_index + 1:064x}",
        "review_id": review_id,
        "business_id": "b1",
        "user_id": f"user-{review_id}",
        "review_time": datetime(2025, 1, 1, tzinfo=UTC),
        "stars": 4.0,
        "useful": 0,
        "segment_index": 0,
        "char_start": 0,
        "char_end": len(text),
        "text": text,
        "text_sha256": text_hash,
        "review_text_sha256": text_hash,
    }


def test_keeps_top_distinct_review_per_business_and_aspect(tmp_path: Path) -> None:
    index_root = tmp_path / "index"
    output_root = tmp_path / "output"
    index_root.mkdir()
    rows = [
        _segment_row(0, "quiet-review", "We could talk comfortably."),
        _segment_row(1, "parking-review", "We found a space immediately."),
        _segment_row(2, "other-review", "The meal was good."),
    ]
    pq.write_table(
        pa.Table.from_pylist(rows, schema=REVIEW_INDEX_SEGMENT_SCHEMA),
        index_root / "review_segments.parquet",
    )
    vectors = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float16,
    )
    np.save(index_root / "segment_embeddings.npy", vectors)
    manifest = ReviewVectorIndexManifest(
        source_paths={"review_segments": "source", "business_facts": "source"},
        source_sha256={"review_segments": "source", "business_facts": "source"},
        model=AnchorEncoder.model,
        dimension=AnchorEncoder.dimension,
        business_count=1,
        review_count=3,
        segment_count=3,
        output_sha256={"segments": "hash", "embeddings": "hash"},
    )
    (index_root / "manifest.json").write_text(
        manifest.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )

    result = build_semantic_review_candidates(
        AnchorEncoder(),
        index_root,
        PROJECT_ROOT / "configs",
        output_root,
        top_reviews_per_business_aspect=1,
    )
    table = pq.read_table(output_root / "semantic_candidate_aspects.parquet")
    pairs = dict(
        zip(
            table.column("aspect").to_pylist(),
            table.column("review_id").to_pylist(),
            strict=True,
        )
    )

    assert result.manifest.candidate_aspect_count == 14
    assert table.schema.equals(CANDIDATE_ASPECT_SCHEMA, check_metadata=False)
    assert pairs["quiet_environment"] == "quiet-review"
    assert pairs["parking"] == "parking-review"
    assert all(table.column("semantic_hit").to_pylist())
    assert not any(table.column("keyword_hit").to_pylist())
