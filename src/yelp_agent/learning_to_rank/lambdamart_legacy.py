"""Frozen Hybrid V2-B evaluation on the previously observed Legacy Test."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from yelp_agent.learning_to_rank.evaluation import (
    HybridV2RankingMetrics,
    HybridV2RankingWriteResult,
)
from yelp_agent.learning_to_rank.lambdamart_artifacts import (
    load_frozen_lambdamart,
)
from yelp_agent.learning_to_rank.lambdamart_selection import selection_key
from yelp_agent.learning_to_rank.scored_ranking import (
    CandidateScoreWriteResult,
    evaluate_candidate_scores,
    write_candidate_scores,
    write_scored_rankings,
)
from yelp_agent.models import StrictModel


@dataclass(frozen=True, slots=True)
class LambdaMARTLegacyTestSources:
    test_features: Path
    test_contexts: Path
    test_ground_truth: Path
    reviews: Path
    interactions: Path
    frozen_model_root: Path
    hybrid_v2_a_legacy_report: Path


class LambdaMARTLegacyTestReport(StrictModel):
    experiment_name: Literal["Hybrid V2-B Legacy Test historical comparison"]
    evaluation_status: Literal["historical_comparison_only"]
    model_selected_before_test: Literal[True]
    test_used_for_training: Literal[False]
    test_used_for_selection: Literal[False]
    frozen_manifest_sha256: str
    scores: CandidateScoreWriteResult
    predictions: HybridV2RankingWriteResult
    hybrid_v1: HybridV2RankingMetrics
    hybrid_v2_a_logistic: HybridV2RankingMetrics
    hybrid_v2_b_lambdamart: HybridV2RankingMetrics
    absolute_delta_vs_logistic: dict[str, float]
    relative_delta_vs_logistic: dict[str, float]
    historical_test_winner: Literal["hybrid_v2_a_logistic", "hybrid_v2_b_lambdamart"]


def _required(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _delta(
    challenger: HybridV2RankingMetrics,
    baseline: HybridV2RankingMetrics,
) -> tuple[dict[str, float], dict[str, float]]:
    names = ("hr_at_1", "hr_at_3", "hr_at_5", "avg_hr", "mrr", "ndcg_at_5")
    absolute = {
        name: float(getattr(challenger, name) - getattr(baseline, name))
        for name in names
    }
    relative = {
        name: (
            0.0
            if float(getattr(baseline, name)) == 0.0
            else absolute[name] / float(getattr(baseline, name))
        )
        for name in names
    }
    return absolute, relative


def run_lambdamart_legacy_test(
    sources: LambdaMARTLegacyTestSources,
    *,
    score_output_path: str | Path,
    prediction_output_path: str | Path,
    report_output_path: str | Path,
    prediction_batch_size: int,
) -> LambdaMARTLegacyTestReport:
    """Score Test once with the already frozen Validation winner."""

    model, manifest = load_frozen_lambdamart(sources.frozen_model_root)
    if manifest.test_data_used_for_training or manifest.test_data_used_for_selection:
        raise ValueError("Frozen LambdaMART manifest violates Test isolation")
    test_features = _required(sources.test_features, "Test features")
    contexts = _required(sources.test_contexts, "Test contexts")
    truth = _required(sources.test_ground_truth, "Test ground truth")
    reviews = _required(sources.reviews, "Reviews")
    interactions = _required(sources.interactions, "Interactions")
    logistic_report_path = _required(
        sources.hybrid_v2_a_legacy_report, "Hybrid V2-A Legacy Test report"
    )
    scores = write_candidate_scores(
        features_path=test_features,
        output_path=score_output_path,
        scorer=model,
        batch_size=prediction_batch_size,
    )
    predictions = write_scored_rankings(
        scores_path=score_output_path,
        output_path=prediction_output_path,
        blend_alpha=manifest.selected_blend_alpha,
    )
    lambdamart = evaluate_candidate_scores(
        scores_path=score_output_path,
        contexts_path=contexts,
        ground_truth_path=truth,
        reviews_path=reviews,
        interactions_path=interactions,
        split="test",
        model_name=f"hybrid_v2_b_{manifest.selected_feature_set}",
        blend_alpha=manifest.selected_blend_alpha,
    )
    payload = json.loads(logistic_report_path.read_text(encoding="utf-8"))
    hybrid_v1 = HybridV2RankingMetrics.model_validate(payload["hybrid_v1"])
    logistic = HybridV2RankingMetrics.model_validate(payload["hybrid_v2"])
    absolute, relative = _delta(lambdamart, logistic)
    winner: Literal["hybrid_v2_a_logistic", "hybrid_v2_b_lambdamart"] = (
        "hybrid_v2_b_lambdamart"
        if selection_key(lambdamart) > selection_key(logistic)
        else "hybrid_v2_a_logistic"
    )
    report = LambdaMARTLegacyTestReport(
        experiment_name="Hybrid V2-B Legacy Test historical comparison",
        evaluation_status="historical_comparison_only",
        model_selected_before_test=True,
        test_used_for_training=False,
        test_used_for_selection=False,
        frozen_manifest_sha256=_sha256(
            Path(sources.frozen_model_root) / "manifest.json"
        ),
        scores=scores,
        predictions=predictions,
        hybrid_v1=hybrid_v1,
        hybrid_v2_a_logistic=logistic,
        hybrid_v2_b_lambdamart=lambdamart,
        absolute_delta_vs_logistic=absolute,
        relative_delta_vs_logistic=relative,
        historical_test_winner=winner,
    )
    output = Path(report_output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".partial")
    partial.write_text(
        json.dumps(
            report.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(partial, output)
    return report
