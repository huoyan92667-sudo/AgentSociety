from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from synthetic_pipeline import run_synthetic_pipeline
from yelp_agent.config import load_config
from yelp_agent.models import RecommendationTask
from yelp_agent.ranking.assembly import (
    HybridSourcePaths,
    build_frozen_hybrid_runtime,
)
from yelp_agent.tuning.hybrid import HybridTuningError


def _sources(root: Path) -> HybridSourcePaths:
    return HybridSourcePaths(
        businesses=root / "processed" / "businesses.parquet",
        reviews=root / "processed" / "reviews.parquet",
        interactions=root / "processed" / "interactions.parquet",
        histories=(
            root / "task_dataset" / "tasks" / "temporal_histories.parquet"
        ),
        tfidf_artifact=root / "features" / "tfidf_vectorizer.joblib",
        tfidf_manifest=root / "features" / "tfidf_manifest.json",
    )


def _first_task(path: Path) -> RecommendationTask:
    first_line = path.read_text(encoding="utf-8").splitlines()[0]
    return RecommendationTask.model_validate_json(first_line)


def test_frozen_runtime_hides_assembly_and_preserves_ranking(
    tmp_path: Path,
) -> None:
    run = run_synthetic_pipeline(tmp_path / "pipeline")

    runtime = build_frozen_hybrid_runtime(
        load_config(),
        _sources(run.root),
        run.root / "hybrid" / "weights.json",
    )
    task = _first_task(
        run.root / "task_dataset" / "tasks" / "test_tasks.jsonl"
    )

    assert runtime.ranker.rank(task).ranking == run.hybrid_ranking
    assert set(runtime.assembly.feature_sources_sha256) == {
        "businesses",
        "reviews",
        "interactions",
        "histories",
        "tfidf_artifact",
        "tfidf_manifest",
    }


def test_frozen_runtime_rejects_a_different_fingerprint_contract(
    tmp_path: Path,
) -> None:
    run = run_synthetic_pipeline(tmp_path / "pipeline")
    sources = replace(_sources(run.root), config_dir=Path("configs"))

    with pytest.raises(HybridTuningError, match="feature sources"):
        build_frozen_hybrid_runtime(
            load_config(),
            sources,
            run.root / "hybrid" / "weights.json",
        )
