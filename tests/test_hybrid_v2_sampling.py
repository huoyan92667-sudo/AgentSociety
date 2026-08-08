from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from yelp_agent.config import HybridV2Config
from yelp_agent.learning_to_rank.sampling import build_training_selection
from yelp_agent.retrieval.benchmark import CANDIDATE_SCHEMA


def _candidate(task_id: str, rank: int, business_id: str) -> dict[str, object]:
    return {
        "task_id": task_id,
        "rank": rank,
        "business_id": business_id,
        "fusion_score": 1.0 / rank,
        "route_count": 4,
        "quality_rank": rank,
        "quality_score": rank / 100.0,
        "category_rank": 31 - rank,
        "category_score": rank / 30.0,
        "text_rank": rank,
        "text_score": 1.0 / rank,
        "location_rank": rank,
        "location_score": 1.0 / rank,
        "distance_km": float(rank),
        "item_knn_rank": rank,
        "item_knn_positive_score": 1.0 / rank,
        "item_knn_negative_evidence": 0.0,
        "item_knn_positive_support_count": 1,
        "item_knn_negative_support_count": 0,
        "item_knn_positive_neighbor_count": 1,
        "item_knn_negative_neighbor_count": 0,
        "item_knn_missing": False,
    }


def _config() -> HybridV2Config:
    return HybridV2Config(
        schema_version=1,
        model_version="2.0.0-a",
        random_seed=42,
        hard_negative_count=10,
        profile_similar_negative_count=5,
        random_negative_count=5,
        regularization_c_candidates=[0.01, 0.1, 1.0, 10.0],
        blend_alphas=[0.0, 0.25, 0.5, 0.75, 1.0],
        primary_metric="avg_hr",
        feature_batch_size=1000,
    )


def test_training_selection_is_balanced_unique_and_deterministic(
    tmp_path: Path,
) -> None:
    candidates = tmp_path / "candidates.parquet"
    truth = tmp_path / "truth.parquet"
    first = tmp_path / "first.parquet"
    second = tmp_path / "second.parquet"
    rows = [_candidate("task-hit", rank, f"b{rank:02d}") for rank in range(1, 31)] + [
        _candidate("task-miss", rank, f"m{rank:02d}") for rank in range(1, 31)
    ]
    pq.write_table(pa.Table.from_pylist(rows, schema=CANDIDATE_SCHEMA), candidates)
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"task_id": "task-hit", "target_business_id": "b07"},
                {"task_id": "task-miss", "target_business_id": "outside"},
            ]
        ),
        truth,
    )

    result = build_training_selection(candidates, truth, first, _config())
    build_training_selection(candidates, truth, second, _config())

    selected = pq.read_table(first).to_pylist()
    assert result.source_tasks == 2
    assert result.selected_tasks == 1
    assert result.target_not_retrieved_tasks == 1
    assert len(selected) == 21
    assert selected[0]["business_id"] == "b07"
    assert selected[0]["label"] is True
    assert {row["negative_kind"] for row in selected[1:]} == {
        "hard",
        "profile_similar",
        "random",
    }
    assert sum(row["negative_kind"] == "hard" for row in selected) == 10
    assert sum(row["negative_kind"] == "profile_similar" for row in selected) == 5
    assert sum(row["negative_kind"] == "random" for row in selected) == 5
    assert len({row["business_id"] for row in selected}) == 21
    assert first.read_bytes() == second.read_bytes()
