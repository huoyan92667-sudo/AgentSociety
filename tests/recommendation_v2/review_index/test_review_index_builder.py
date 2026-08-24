from __future__ import annotations

import hashlib
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from yelp_agent.recommendation_v2.business_facts import BUSINESS_FACT_SCHEMA
from yelp_agent.recommendation_v2.review_index import (
    REVIEW_INDEX_SEGMENT_SCHEMA,
    build_review_vector_index,
)
from yelp_agent.review_rag.schema import REVIEW_SEGMENT_SCHEMA
from yelp_agent.semantic_embedding.encoder import EncodedBatch
from yelp_agent.semantic_embedding.schema import EmbeddingInputType


class FakeEncoder:
    provider = "fake"
    model = "fake-review-encoder"
    dimension = 4
    batch_size = 2

    def encode(
        self,
        texts: Sequence[str],
        *,
        input_type: EmbeddingInputType,
    ) -> EncodedBatch:
        assert input_type == "document"
        vectors = tuple(
            np.asarray(
                [len(text), text.count("a") + 1, text.count("e") + 1, 1],
                dtype=np.float32,
            )
            for text in texts
        )
        return EncodedBatch(
            vectors=vectors,
            model=self.model,
            input_tokens=len(texts),
            request_id="fake",
            latency_ms=0.0,
            per_text_input_tokens=tuple(1 for _ in texts),
        )


def _business_row(business_id: str) -> dict[str, object]:
    return {
        "business_id": business_id,
        "name": "Test Restaurant",
        "address": "1 Main St",
        "city": "Philadelphia",
        "state": "PA",
        "postal_code": "19107",
        "latitude": 39.95,
        "longitude": -75.16,
        "categories": ["Restaurants"],
        "price_level": None,
        "price_lower_usd": None,
        "price_upper_usd": None,
        "rating": 4.0,
        "review_count": 2,
        "accepts_reservations": None,
        "delivery": None,
        "takeout": None,
        "outdoor_seating": None,
        "good_for_kids": None,
        "good_for_groups": None,
        "wheelchair_accessible": None,
        "dogs_allowed": None,
        "parking_available": None,
        "parking_garage": None,
        "parking_street": None,
        "parking_validated": None,
        "parking_lot": None,
        "parking_valet": None,
    }


def _segment_row(
    segment_id: str,
    review_id: str,
    business_id: str,
    text: str,
) -> dict[str, object]:
    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return {
        "segment_id": segment_id * 64,
        "review_id": review_id,
        "business_id": business_id,
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


def test_builds_scoped_segments_and_aligned_float16_vectors(tmp_path: Path) -> None:
    segment_source = tmp_path / "segments.parquet"
    business_source = tmp_path / "businesses.parquet"
    output_root = tmp_path / "index"
    pq.write_table(
        pa.Table.from_pylist(
            [
                _segment_row("a", "r1", "b1", "quiet dining room"),
                _segment_row("b", "r2", "b1", "easy parking"),
                _segment_row("c", "r3", "outside", "outside scope"),
            ],
            schema=REVIEW_SEGMENT_SCHEMA,
        ),
        segment_source,
    )
    pq.write_table(
        pa.Table.from_pylist([_business_row("b1")], schema=BUSINESS_FACT_SCHEMA),
        business_source,
    )

    first = build_review_vector_index(
        FakeEncoder(),
        segment_source,
        business_source,
        output_root,
    )
    second = build_review_vector_index(
        FakeEncoder(),
        segment_source,
        business_source,
        output_root,
    )
    segment_table = pq.read_table(output_root / "review_segments.parquet")
    vectors = np.load(output_root / "segment_embeddings.npy", mmap_mode="r")

    assert first.manifest.business_count == 1
    assert first.manifest.review_count == 2
    assert first.manifest.segment_count == 2
    assert second.status == "skipped"
    assert segment_table.schema.equals(REVIEW_INDEX_SEGMENT_SCHEMA, check_metadata=False)
    assert segment_table.column("row_index").to_pylist() == [0, 1]
    assert segment_table.column("review_id").to_pylist() == ["r1", "r2"]
    assert vectors.shape == (2, 4)
    assert vectors.dtype == np.float16
    assert np.allclose(np.linalg.norm(vectors.astype(np.float32), axis=1), 1.0, atol=1e-3)
    # 生成时会按文字长度重排同批片段来减少无用计算，但写回后必须仍与原行号对应。
    expected = []
    for text in ("quiet dining room", "easy parking"):
        value = np.asarray(
            [len(text), text.count("a") + 1, text.count("e") + 1, 1],
            dtype=np.float32,
        )
        expected.append(value / np.linalg.norm(value))
    assert np.allclose(vectors.astype(np.float32), expected, atol=1e-3)
