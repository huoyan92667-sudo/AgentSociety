"""Post-freeze Legacy Test comparison that cannot alter Hybrid V2-A."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from yelp_agent.config import (
    AppConfig,
    BusinessProfileConfig,
    HybridV2Config,
    ItemKNNConfig,
    RetrievalConfig,
)
from yelp_agent.experiments import write_json_artifact
from yelp_agent.learning_to_rank.artifacts import load_frozen_hybrid_v2
from yelp_agent.learning_to_rank.evaluation import (
    HybridV2RankingMetrics,
    HybridV2RankingWriteResult,
    evaluate_hybrid_v2_ranking,
    write_hybrid_v2_rankings,
)
from yelp_agent.learning_to_rank.features import (
    HybridV1Weights,
    HybridV2FeatureBuildResult,
    HybridV2FeatureSources,
    build_hybrid_v2_features,
)
from yelp_agent.models import StrictModel
from yelp_agent.retrieval.benchmark import (
    RetrievalBenchmarkBuildResult,
    RetrievalSourcePaths,
    build_full_retrieval_benchmark,
)


@dataclass(frozen=True, slots=True)
class HybridV2LegacyTestSources:
    contexts: Path
    ground_truth: Path
    businesses: Path
    reviews: Path
    interactions: Path
    tfidf_artifact: Path
    tfidf_manifest: Path
    item_knn_root: Path
    user_profile_root: Path
    business_profile_root: Path
    hybrid_v1_weights: Path
    frozen_model_root: Path


class HybridV2LegacyTestReport(StrictModel):
    experiment_name: Literal["Hybrid V2-A Legacy Test historical comparison"]
    evaluation_status: Literal["historical_comparison_only"]
    model_selected_before_test: Literal[True]
    test_used_for_training: Literal[False]
    test_used_for_selection: Literal[False]
    frozen_manifest_sha256: str
    retrieval: RetrievalBenchmarkBuildResult
    features: HybridV2FeatureBuildResult
    predictions: HybridV2RankingWriteResult
    hybrid_v1: HybridV2RankingMetrics
    hybrid_v2: HybridV2RankingMetrics
    avg_hr_delta: float
    mrr_delta: float


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_weights(path: Path) -> HybridV1Weights:
    payload = json.loads(path.read_text(encoding="utf-8"))
    selected = payload.get("selected_weights")
    if not isinstance(selected, dict):
        raise TypeError("Hybrid V1 artifact does not contain selected_weights")
    return HybridV1Weights.model_validate(selected)


def run_hybrid_v2_legacy_test(
    sources: HybridV2LegacyTestSources,
    *,
    retrieval_output_root: str | Path,
    feature_output_path: str | Path,
    prediction_output_path: str | Path,
    report_path: str | Path,
    app_config: AppConfig,
    retrieval_config: RetrievalConfig,
    item_knn_config: ItemKNNConfig,
    model_config: HybridV2Config,
    business_profile_config: BusinessProfileConfig,
) -> HybridV2LegacyTestReport:
    """Run Legacy Test only after a validation-frozen model already exists."""

    report = Path(report_path)
    if report.is_file():
        return HybridV2LegacyTestReport.model_validate_json(
            report.read_text(encoding="utf-8")
        )
    model, manifest = load_frozen_hybrid_v2(sources.frozen_model_root)
    if manifest.test_data_used_for_training or manifest.test_data_used_for_selection:
        raise ValueError("Frozen Hybrid V2 manifest violates test isolation")
    retrieval = build_full_retrieval_benchmark(
        RetrievalSourcePaths(
            businesses=sources.businesses,
            reviews=sources.reviews,
            interactions=sources.interactions,
            tfidf_artifact=sources.tfidf_artifact,
            tfidf_manifest=sources.tfidf_manifest,
            item_knn_positive_events=(
                sources.item_knn_root / "positive_events.parquet"
            ),
            item_knn_negative_events=(
                sources.item_knn_root / "negative_events.parquet"
            ),
            item_knn_neutral_events=(sources.item_knn_root / "neutral_events.parquet"),
            item_knn_manifest=sources.item_knn_root / "manifest.json",
        ),
        {"test": sources.contexts},
        retrieval_output_root,
        app_config,
        retrieval_config,
        item_knn_config=item_knn_config,
    )
    feature_path = Path(feature_output_path)
    if feature_path.exists():
        raise FileExistsError(f"Legacy Test features already exist: {feature_path}")
    features = build_hybrid_v2_features(
        HybridV2FeatureSources(
            candidates=Path(retrieval_output_root) / "test_candidates.parquet",
            user_profile_root=sources.user_profile_root,
            business_profile_root=sources.business_profile_root,
        ),
        feature_path,
        split="test",
        weights=_load_weights(sources.hybrid_v1_weights),
        broad_categories=set(app_config.data.broad_categories),
        business_profile_config=business_profile_config,
        batch_size=model_config.feature_batch_size,
    )
    v1 = evaluate_hybrid_v2_ranking(
        features_path=feature_path,
        contexts_path=sources.contexts,
        ground_truth_path=sources.ground_truth,
        reviews_path=sources.reviews,
        interactions_path=sources.interactions,
        split="test",
        model_name="hybrid_v1",
        model=None,
        blend_alpha=0.0,
    )
    v2 = evaluate_hybrid_v2_ranking(
        features_path=feature_path,
        contexts_path=sources.contexts,
        ground_truth_path=sources.ground_truth,
        reviews_path=sources.reviews,
        interactions_path=sources.interactions,
        split="test",
        model_name=f"hybrid_v2_{manifest.selected_feature_set}",
        model=model,
        blend_alpha=manifest.selected_blend_alpha,
    )
    predictions = write_hybrid_v2_rankings(
        features_path=feature_path,
        output_path=prediction_output_path,
        model=model,
        blend_alpha=manifest.selected_blend_alpha,
    )
    result = HybridV2LegacyTestReport(
        experiment_name="Hybrid V2-A Legacy Test historical comparison",
        evaluation_status="historical_comparison_only",
        model_selected_before_test=True,
        test_used_for_training=False,
        test_used_for_selection=False,
        frozen_manifest_sha256=_sha256(sources.frozen_model_root / "manifest.json"),
        retrieval=retrieval,
        features=features,
        predictions=predictions,
        hybrid_v1=v1,
        hybrid_v2=v2,
        avg_hr_delta=v2.avg_hr - v1.avg_hr,
        mrr_delta=v2.mrr - v1.mrr,
    )
    write_json_artifact(report, result)
    return result
