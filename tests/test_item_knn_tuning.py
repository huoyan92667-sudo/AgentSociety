from __future__ import annotations

from pathlib import Path

import pandas as pd

from yelp_agent.collaborative.artifacts import build_item_knn_artifacts
from yelp_agent.collaborative.tuning import (
    ItemKNNTuningSources,
    tune_item_knn,
)
from yelp_agent.config import ItemKNNConfig, RetrievalConfig
from yelp_agent.evaluation.item_knn import compare_item_knn_retrieval


def test_validation_tuning_selects_the_half_life_with_better_fused_recall(
    tmp_path: Path,
) -> None:
    businesses = tmp_path / "businesses.parquet"
    reviews = tmp_path / "reviews.parquet"
    interactions = tmp_path / "interactions.parquet"
    contexts = tmp_path / "contexts.parquet"
    ground_truth = tmp_path / "ground_truth.parquet"
    base_provenance = tmp_path / "base_provenance.parquet"
    graph_source = tmp_path / "graph_source.parquet"
    business_ids = ("a", "b", "c", "z")
    pd.DataFrame(
        [
            {
                "business_id": business_id,
                "name": business_id,
                "address": "",
                "city": "Philadelphia",
                "state": "PA",
                "postal_code": "",
                "latitude": 39.95,
                "longitude": -75.16,
                "categories": ["Restaurants"],
                "attributes_json": "{}",
            }
            for business_id in business_ids
        ]
    ).to_parquet(businesses, index=False)
    availability = [
        {
            "review_id": f"available-{business_id}",
            "user_id": "availability-user",
            "business_id": business_id,
            "stars": 4.0,
            "text": "",
            "date": pd.Timestamp("2017-01-01"),
        }
        for business_id in business_ids
    ]
    target_rows = [
        {
            "review_id": "target-history",
            "user_id": "target-user",
            "business_id": "a",
            "stars": 5.0,
            "text": "",
            "date": pd.Timestamp("2020-12-15"),
        },
        {
            "review_id": "target-review",
            "user_id": "target-user",
            "business_id": "c",
            "stars": 5.0,
            "text": "",
            "date": pd.Timestamp("2021-01-01"),
        },
    ]
    pd.DataFrame([*availability, *target_rows]).to_parquet(
        reviews,
        index=False,
    )
    pd.DataFrame(target_rows).to_parquet(interactions, index=False)
    pd.DataFrame(
        [
            {
                "task_id": "validation:target-user",
                "split": "validation",
                "user_id": "target-user",
                "cutoff_time": pd.Timestamp("2021-01-01"),
                "history_count": 1,
            }
        ]
    ).to_parquet(contexts, index=False)
    pd.DataFrame(
        [
            {
                "task_id": "validation:target-user",
                "target_business_id": "c",
            }
        ]
    ).to_parquet(ground_truth, index=False)
    pd.DataFrame(
        [
            {
                "task_id": "validation:target-user",
                "route": "quality",
                "route_rank": 1,
                "business_id": "z",
                "route_score": 1.0,
            }
        ]
    ).to_parquet(base_provenance, index=False)

    graph_rows: list[dict[str, object]] = []
    for index in range(8):
        for business_id, day in (("a", "01"), ("b", "02")):
            graph_rows.append(
                {
                    "user_id": f"old-{index}",
                    "review_id": f"old-{index}-{business_id}",
                    "business_id": business_id,
                    "stars": 5.0,
                    "date": pd.Timestamp(f"2018-01-{day}"),
                }
            )
    for index in range(2):
        for business_id, day in (("a", "01"), ("c", "02")):
            graph_rows.append(
                {
                    "user_id": f"recent-{index}",
                    "review_id": f"recent-{index}-{business_id}",
                    "business_id": business_id,
                    "stars": 5.0,
                    "date": pd.Timestamp(f"2020-12-{day}"),
                }
            )
    pd.DataFrame(graph_rows).to_parquet(graph_source, index=False)
    base_config = ItemKNNConfig(
        shrinkage_beta=1.0,
        half_life_days=None,
        reserved_tail_interactions=0,
    )
    artifact = build_item_knn_artifacts(
        graph_source,
        tmp_path / "graph",
        base_config,
    )
    retrieval_config = RetrievalConfig(
        candidate_limit=1,
        per_route_limit=1,
        rrf_constant=60.0,
        exclude_history_businesses=True,
        bayesian_prior_count=20,
        location_scale_km=10.0,
        metric_cutoffs=[1],
        provenance_splits=["validation"],
    )

    result = tune_item_knn(
        ItemKNNTuningSources(
            businesses=businesses,
            reviews=reviews,
            interactions=interactions,
            contexts=contexts,
            ground_truth=ground_truth,
            base_route_provenance=base_provenance,
            positive_events=Path(artifact.positive_events_path),
            negative_events=Path(artifact.negative_events_path),
            neutral_events=Path(artifact.neutral_events_path),
        ),
        tmp_path / "tuning",
        base_config,
        retrieval_config,
        half_life_candidates=(None, 365),
    )

    assert result.selected_half_life_days == 365
    by_label = {candidate.label: candidate for candidate in result.candidates}
    assert by_label["none"].fused_recall_at["1"] == 0.0
    assert by_label["365"].fused_recall_at["1"] == 1.0
    assert Path(result.selected_config_path).is_file()


def test_item_knn_comparison_counts_novel_hits_damage_and_unique_signal(
    tmp_path: Path,
) -> None:
    contexts = tmp_path / "contexts.parquet"
    baseline = tmp_path / "baseline.parquet"
    item_knn = tmp_path / "item_knn.parquet"
    task_ids = ("task-1", "task-2", "task-3")
    pd.DataFrame(
        [
            {"task_id": task_id, "history_count": history_count}
            for task_id, history_count in zip(
                task_ids,
                (10, 20, 70),
                strict=True,
            )
        ]
    ).to_parquet(contexts, index=False)
    shared = {
        "catalog_eligible": True,
        "target_in_history": False,
    }
    pd.DataFrame(
        [
            {"task_id": "task-1", "target_rank": None, **shared},
            {"task_id": "task-2", "target_rank": 2, **shared},
            {"task_id": "task-3", "target_rank": 1, **shared},
        ]
    ).to_parquet(baseline, index=False)
    pd.DataFrame(
        [
            {
                "task_id": "task-1",
                "target_rank": 3,
                "item_knn_rank": 2,
                "category_rank": None,
                "text_rank": None,
                **shared,
            },
            {
                "task_id": "task-2",
                "target_rank": None,
                "item_knn_rank": None,
                "category_rank": 2,
                "text_rank": None,
                **shared,
            },
            {
                "task_id": "task-3",
                "target_rank": 1,
                "item_knn_rank": None,
                "category_rank": 1,
                "text_rank": 1,
                **shared,
            },
        ]
    ).to_parquet(item_knn, index=False)

    result = compare_item_knn_retrieval(
        baseline,
        item_knn,
        contexts,
        metric_cutoffs=[5],
    )

    assert result.novel_fused_hits_at == {"5": 1}
    assert result.damaged_fused_hits_at == {"5": 1}
    assert result.item_unique_vs_category_text_at == {"5": 1}
    assert result.recall_delta_at == {"5": 0.0}
    assert result.history_buckets["8-15"].novel_hits_at == {"5": 1}
