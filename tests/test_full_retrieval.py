from __future__ import annotations

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from yelp_agent.config import RetrievalConfig, load_config, load_retrieval_config
from yelp_agent.data.temporal_view import BusinessRecord, InteractionRecord
from yelp_agent.evaluation.retrieval import (
    RetrievalEvaluationError,
    evaluate_full_retrieval,
)
from yelp_agent.features.text import fit_tfidf_model
from yelp_agent.retrieval.benchmark import (
    RetrievalBenchmarkManifest,
    RetrievalSourcePaths,
    build_full_retrieval_benchmark,
)
from yelp_agent.retrieval.multi_route import (
    MultiRouteRetriever,
    RetrievalTaskContext,
)

PROJECT_CONFIG_DIR = Path(__file__).parents[1] / "configs"


def _config(**updates: object) -> RetrievalConfig:
    values: dict[str, object] = {
        "candidate_limit": 3,
        "per_route_limit": 3,
        "rrf_constant": 60.0,
        "exclude_history_businesses": True,
        "bayesian_prior_count": 20,
        "location_scale_km": 10.0,
        "metric_cutoffs": [1, 3],
        "provenance_splits": ["validation"],
    }
    values.update(updates)
    return RetrievalConfig.model_validate(values)


class _FakeView:
    def __init__(self) -> None:
        self.ids = ["a", "b", "c", "d", "future", "history"]
        self.history = (
            InteractionRecord(
                review_id="history-review",
                user_id="user-1",
                business_id="history",
                stars=5.0,
                text="coffee",
                date=datetime(2020, 1, 1),
            ),
        )

    def businesses(self):
        return tuple(
            BusinessRecord(
                business_id=business_id,
                name=business_id,
                address="",
                city="Philadelphia",
                state="PA",
                postal_code="",
                latitude=39.95,
                longitude=-75.16,
                categories=("Food", "Coffee & Tea"),
                attributes_json="{}",
            )
            for business_id in self.ids
        )

    def user_history(self, user_id, cutoff_time):
        return self.history if user_id == "user-1" else ()


class _FakeQuality:
    def score_catalog(self, cutoff_time):
        scores = {
            "a": 0.9,
            "b": 0.8,
            "c": 0.7,
            "d": 0.6,
            "future": 1.0,
            "history": 1.0,
        }
        business_ids = tuple(sorted(scores))
        return SimpleNamespace(
            business_ids=business_ids,
            review_counts=np.asarray(
                [0 if item == "future" else 10 for item in business_ids]
            ),
            quality_scores=np.asarray([scores[item] for item in business_ids]),
        )


class _FakeCategory:
    def score_candidates(self, request):
        scores = {"a": 0.1, "b": 1.0, "c": 0.8, "d": 0.0}
        return {item: scores[item] for item in request.candidate_business_ids}


class _FakeText:
    def score_candidates(self, request):
        scores = {"a": 0.2, "b": 0.3, "c": 1.0, "d": 0.9}
        return SimpleNamespace(
            business_ids=tuple(request.candidate_business_ids),
            positive_review_count=1,
            negative_review_count=0,
            text_scores=np.asarray(
                [scores[item] for item in request.candidate_business_ids]
            ),
        )


class _FakeLocation:
    def score_candidates(self, request):
        scores = {"a": (0.4, 9.0), "b": (0.5, 7.0), "c": (0.6, 5.0), "d": (1.0, 0.1)}
        return SimpleNamespace(
            business_ids=tuple(request.candidate_business_ids),
            location_scores=tuple(
                scores[item][0] for item in request.candidate_business_ids
            ),
            distances_km=tuple(
                scores[item][1] for item in request.candidate_business_ids
            ),
        )


def test_default_retrieval_configuration_describes_real_top_500() -> None:
    config = load_retrieval_config(PROJECT_CONFIG_DIR)

    assert config.candidate_limit == 500
    assert config.per_route_limit == 500
    assert config.metric_cutoffs == [50, 100, 500]
    assert config.provenance_splits == ["validation"]


def test_multi_route_retrieval_is_target_blind_deterministic_and_time_safe() -> None:
    view = _FakeView()
    retriever = MultiRouteRetriever(
        view,  # type: ignore[arg-type]
        category_store=_FakeCategory(),  # type: ignore[arg-type]
        text_store=_FakeText(),  # type: ignore[arg-type]
        quality_store=_FakeQuality(),  # type: ignore[arg-type]
        location_store=_FakeLocation(),  # type: ignore[arg-type]
        config=_config(),
    )
    task = RetrievalTaskContext(
        task_id="validation:user-1",
        split="validation",
        user_id="user-1",
        cutoff_time=datetime(2020, 2, 1),
        history_count=1,
    )

    first = retriever.retrieve(task)
    second = retriever.retrieve(task)

    assert first.candidates == second.candidates
    assert len(first.candidates) == 3
    assert len({item.business_id for item in first.candidates}) == 3
    assert "history" not in {item.business_id for item in first.candidates}
    assert "future" not in {item.business_id for item in first.candidates}
    assert first.catalog_size == 5
    assert first.excluded_history_businesses == 1
    assert first.candidates[0].business_id == "c"
    assert first.candidates[0].route_count == 4
    assert all(
        not hasattr(item, "target_business_id") for item in first.candidates
    )


def test_recall_metrics_average_binary_hits_across_tasks(tmp_path: Path) -> None:
    contexts = tmp_path / "contexts.parquet"
    truth = tmp_path / "truth.parquet"
    candidates = tmp_path / "candidates.parquet"
    audit = tmp_path / "audit.parquet"
    provenance = tmp_path / "provenance.parquet"
    reviews = tmp_path / "reviews.parquet"
    interactions = tmp_path / "interactions.parquet"
    cutoff = pd.Timestamp("2020-02-01")
    task_ids = [f"validation:user-{index}" for index in range(1, 6)]
    targets = [f"target-{index}" for index in range(1, 6)]
    pd.DataFrame(
        [
            {
                "task_id": task_id,
                "split": "validation",
                "user_id": f"user-{index}",
                "cutoff_time": cutoff,
                "history_count": 1,
            }
            for index, task_id in enumerate(task_ids, start=1)
        ]
    ).to_parquet(contexts, index=False)
    pd.DataFrame(
        [
            {"task_id": task_id, "target_business_id": target}
            for task_id, target in zip(task_ids, targets, strict=True)
        ]
    ).to_parquet(truth, index=False)

    target_ranks = [2, 40, 80, 230, None]
    candidate_rows: list[dict[str, object]] = []
    provenance_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    for task_id, target, target_rank in zip(
        task_ids,
        targets,
        target_ranks,
        strict=True,
    ):
        candidate_count = target_rank or 1
        for rank in range(1, candidate_count + 1):
            business_id = target if rank == target_rank else f"{task_id}-decoy-{rank}"
            candidate_rows.append(
                {"task_id": task_id, "rank": rank, "business_id": business_id}
            )
            provenance_rows.append(
                {
                    "task_id": task_id,
                    "route": "quality",
                    "route_rank": rank,
                    "business_id": business_id,
                    "route_score": 1.0 / rank,
                }
            )
        audit_rows.append(
            {
                "task_id": task_id,
                "split": "validation",
                "candidate_count": candidate_count,
                "catalog_size": 1000,
                "latency_ms": 10.0,
            }
        )
    pd.DataFrame(candidate_rows).to_parquet(candidates, index=False)
    pd.DataFrame(audit_rows).to_parquet(audit, index=False)
    pd.DataFrame(provenance_rows).to_parquet(provenance, index=False)
    candidate_business_ids = sorted(
        {str(row["business_id"]) for row in candidate_rows}.union(targets)
    )
    pd.DataFrame(
        [
            {
                "review_id": f"availability-{index}",
                "business_id": business_id,
                "date": pd.Timestamp("2020-01-01"),
            }
            for index, business_id in enumerate(candidate_business_ids)
        ]
    ).to_parquet(reviews, index=False)
    pd.DataFrame(
        [
            {
                "user_id": "unrelated-user",
                "business_id": "unrelated-business",
                "date": pd.Timestamp("2020-01-01"),
            }
        ]
    ).to_parquet(interactions, index=False)

    metrics = evaluate_full_retrieval(
        split="validation",
        contexts_path=contexts,
        ground_truth_path=truth,
        candidates_path=candidates,
        task_audit_path=audit,
        reviews_path=reviews,
        interactions_path=interactions,
        metric_cutoffs=[50, 100, 500],
        route_provenance_path=provenance,
    )

    assert metrics.recall_at == pytest.approx(
        {"50": 0.4, "100": 0.6, "500": 0.8}
    )
    assert metrics.all_task_recall_at == pytest.approx(metrics.recall_at)
    assert metrics.route_recall_at["quality"] == pytest.approx(
        metrics.recall_at
    )
    assert metrics.route_recall_at["category"] == {
        "50": 0.0,
        "100": 0.0,
        "500": 0.0,
    }
    assert metrics.target_not_retrieved_count == 1
    assert metrics.policy_reachable_task_count == 5
    assert set(pq.ParquetFile(candidates).schema_arrow.names).isdisjoint(
        {"target_business_id", "ground_truth"}
    )


@pytest.mark.parametrize(
    ("first_review", "history_business"),
    [
        ("2020-03-01", "unrelated"),
        ("2020-01-01", "candidate-business"),
    ],
)
def test_evaluator_rejects_future_or_previously_visited_candidates(
    tmp_path: Path,
    first_review: str,
    history_business: str,
) -> None:
    task_id = "validation:user-1"
    paths = {
        name: tmp_path / f"{name}.parquet"
        for name in (
            "contexts",
            "truth",
            "candidates",
            "audit",
            "reviews",
            "interactions",
        )
    }
    pd.DataFrame(
        [
            {
                "task_id": task_id,
                "split": "validation",
                "user_id": "user-1",
                "cutoff_time": pd.Timestamp("2020-02-01"),
                "history_count": 1,
            }
        ]
    ).to_parquet(paths["contexts"], index=False)
    pd.DataFrame(
        [{"task_id": task_id, "target_business_id": "candidate-business"}]
    ).to_parquet(paths["truth"], index=False)
    pd.DataFrame(
        [
            {
                "task_id": task_id,
                "rank": 1,
                "business_id": "candidate-business",
            }
        ]
    ).to_parquet(paths["candidates"], index=False)
    pd.DataFrame(
        [
            {
                "task_id": task_id,
                "split": "validation",
                "candidate_count": 1,
                "catalog_size": 1,
                "latency_ms": 1.0,
            }
        ]
    ).to_parquet(paths["audit"], index=False)
    pd.DataFrame(
        [
            {
                "review_id": "availability",
                "business_id": "candidate-business",
                "date": pd.Timestamp(first_review),
            }
        ]
    ).to_parquet(paths["reviews"], index=False)
    pd.DataFrame(
        [
            {
                "user_id": "user-1",
                "business_id": history_business,
                "date": pd.Timestamp("2020-01-01"),
            }
        ]
    ).to_parquet(paths["interactions"], index=False)

    with pytest.raises(RetrievalEvaluationError, match="inconsistent"):
        evaluate_full_retrieval(
            split="validation",
            contexts_path=paths["contexts"],
            ground_truth_path=paths["truth"],
            candidates_path=paths["candidates"],
            task_audit_path=paths["audit"],
            reviews_path=paths["reviews"],
            interactions_path=paths["interactions"],
            metric_cutoffs=[1],
        )


def _write_real_retrieval_fixture(root: Path, *, add_future: bool = False):
    root.mkdir(parents=True, exist_ok=True)
    businesses = root / "businesses.parquet"
    reviews = root / "reviews.parquet"
    interactions = root / "interactions.parquet"
    histories = root / "histories.parquet"
    contexts = root / "contexts.parquet"
    business_ids = [
        "history-coffee",
        "history-bakery",
        "target-coffee",
        *[f"other-{index}" for index in range(5)],
    ]
    pd.DataFrame(
        [
            {
                "business_id": business_id,
                "name": business_id.replace("-", " "),
                "address": "Market Street",
                "city": "Philadelphia",
                "state": "PA",
                "postal_code": "19103",
                "latitude": 39.95 + index * 0.001,
                "longitude": -75.16,
                "categories": (
                    ["Food", "Coffee & Tea"]
                    if "coffee" in business_id
                    else ["Food", "Bakeries"]
                ),
                "attributes_json": "{}",
            }
            for index, business_id in enumerate(business_ids)
        ]
    ).to_parquet(businesses, index=False)
    availability = [
        {
            "review_id": f"availability-{business_id}",
            "user_id": "availability-user",
            "business_id": business_id,
            "stars": 4.0,
            "text": "historically available",
            "date": pd.Timestamp("2019-01-01"),
        }
        for business_id in business_ids
    ]
    personal = [
        {
            "review_id": "history-1",
            "user_id": "user-1",
            "business_id": "history-coffee",
            "stars": 5.0,
            "text": "excellent coffee quiet friendly",
            "date": pd.Timestamp("2020-01-01"),
        },
        {
            "review_id": "history-2",
            "user_id": "user-1",
            "business_id": "history-bakery",
            "stars": 2.0,
            "text": "slow noisy bakery",
            "date": pd.Timestamp("2020-01-05"),
        },
        {
            "review_id": "target-review",
            "user_id": "user-1",
            "business_id": "target-coffee",
            "stars": 5.0,
            "text": "future target review",
            "date": pd.Timestamp("2020-02-01"),
        },
    ]
    review_rows = [*availability, *personal]
    if add_future:
        review_rows.extend(
            {
                "review_id": f"future-{index}",
                "user_id": "future-user",
                "business_id": "other-0",
                "stars": 1.0,
                "text": "future only",
                "date": pd.Timestamp("2030-01-01")
                + pd.Timedelta(index, unit="D"),
            }
            for index in range(20)
        )
    pd.DataFrame(review_rows).to_parquet(reviews, index=False)
    pd.DataFrame(personal).to_parquet(interactions, index=False)
    pd.DataFrame(
        [
            {
                "task_id": "validation:user-1",
                "position": 1,
                "review_id": "history-1",
            },
            {
                "task_id": "validation:user-1",
                "position": 2,
                "review_id": "history-2",
            },
        ]
    ).to_parquet(histories, index=False)
    pd.DataFrame(
        [
            {
                "task_id": "validation:user-1",
                "split": "validation",
                "user_id": "user-1",
                "cutoff_time": pd.Timestamp("2020-02-01"),
                "history_count": 2,
            }
        ]
    ).to_parquet(contexts, index=False)
    config = load_config(PROJECT_CONFIG_DIR)
    tfidf_config = config.tfidf.model_copy(
        update={"min_df": 1, "max_features": 1000}
    )
    artifact = root / "tfidf.joblib"
    manifest = root / "tfidf_manifest.json"
    fit_tfidf_model(
        businesses,
        interactions,
        histories,
        artifact,
        manifest,
        tfidf_config,
    )
    return (
        config,
        RetrievalSourcePaths(
            businesses=businesses,
            reviews=reviews,
            interactions=interactions,
            tfidf_artifact=artifact,
            tfidf_manifest=manifest,
        ),
        contexts,
    )


def test_benchmark_builder_never_reads_labels_and_future_reviews_do_not_change_candidates(
    tmp_path: Path,
) -> None:
    normal_config, normal_sources, normal_contexts = _write_real_retrieval_fixture(
        tmp_path / "normal"
    )
    future_config, future_sources, future_contexts = _write_real_retrieval_fixture(
        tmp_path / "future",
        add_future=True,
    )
    retrieval_config = _config()

    normal = build_full_retrieval_benchmark(
        normal_sources,
        {"validation": normal_contexts},
        tmp_path / "normal-output",
        normal_config,
        retrieval_config,
    )
    reused = build_full_retrieval_benchmark(
        normal_sources,
        {"validation": normal_contexts},
        tmp_path / "normal-output",
        normal_config,
        retrieval_config,
    )
    future = build_full_retrieval_benchmark(
        future_sources,
        {"validation": future_contexts},
        tmp_path / "future-output",
        future_config,
        retrieval_config,
    )

    normal_candidates = pd.read_parquet(
        Path(normal.output_root) / "validation_candidates.parquet"
    ).drop(columns=[])
    future_candidates = pd.read_parquet(
        Path(future.output_root) / "validation_candidates.parquet"
    ).drop(columns=[])
    manifest = RetrievalBenchmarkManifest.model_validate_json(
        Path(normal.manifest_path).read_text(encoding="utf-8")
    )

    assert normal.status == "written"
    assert reused.status == "skipped"
    assert manifest.target_conditioned is False
    assert manifest.ground_truth_files_read is False
    assert all("ground_truth" not in name for name in manifest.source_sha256)
    assert list(normal_candidates["rank"]) == [1, 2, 3]
    assert set(normal_candidates["business_id"]).isdisjoint(
        {"history-coffee", "history-bakery"}
    )
    pd.testing.assert_frame_equal(normal_candidates, future_candidates)
