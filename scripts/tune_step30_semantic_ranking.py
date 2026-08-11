"""Select Step 30 semantic score and rank-protection policy on Development only."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

from yelp_agent.agent_benchmark import load_scenario_ground_truth
from yelp_agent.semantic_ranking import (
    SemanticRankingPolicy,
    SemanticRankingResult,
    apply_ranking_policy,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/semantic_ranking_v1/development_tuning.json"),
    )
    args = parser.parse_args()
    root = args.project_root.resolve()
    diagnostics = _load_diagnostics(_resolve(root, args.diagnostics))
    latest_by_context = {
        str(item.context_id): item
        for item in diagnostics
        if item.context_id is not None
    }
    truth = {
        item.scenario_id: item
        for item in load_scenario_ground_truth(
            root / "benchmarks" / "agent_scenarios_v1" / "hidden" / "ground_truth.jsonl"
        )
        if item.task_type in {"recommendation_request", "feedback_refinement"}
        and item.acceptable_business_ids
    }
    usable = [
        item
        for item in latest_by_context.values()
        if item.context_id in truth
        and item.candidate_scores
        and item.intent.rankable_condition_count > 0
    ]
    if not usable:
        raise ValueError("no Development semantic-ranking diagnostics are tunable")
    if any(truth[str(item.context_id)].scenario_id != item.context_id for item in usable):
        raise ValueError("diagnostics and benchmark labels do not align")

    weight_rows: list[dict[str, object]] = []
    for embedding_units in range(11):
        for cross_units in range(11 - embedding_units):
            structured_units = 10 - embedding_units - cross_units
            policy = _policy(
                scenario_count=len(usable),
                embedding_weight=embedding_units / 10,
                cross_encoder_weight=cross_units / 10,
                structured_weight=structured_units / 10,
            )
            metrics = _evaluate(usable, truth, policy, mode="aggressive")
            weight_rows.append({"policy": policy.model_dump(mode="json"), "metrics": metrics})
    best_weights = max(weight_rows, key=lambda row: _selection_key(row["metrics"]))
    selected_base = SemanticRankingPolicy.model_validate(best_weights["policy"])

    protected_rows: list[dict[str, object]] = []
    for (
        minimum_confidence,
        alpha,
        evidence_floor,
        moves,
        protected_top_k,
        margin,
    ) in itertools.product(
        (0.0, 0.5, 0.7),
        (0.3, 0.5, 0.7, 1.0),
        (0.25, 0.5, 0.75),
        ((3, 3), (5, 3), (10, 5)),
        (3, 5),
        (0.0, 0.1, 0.2),
    ):
        policy = selected_base.model_copy(
            update={
                "minimum_intent_confidence": minimum_confidence,
                "maximum_fusion_alpha": alpha,
                "evidence_floor": evidence_floor,
                "maximum_upward_move": moves[0],
                "maximum_downward_move": moves[1],
                "protected_top_k": protected_top_k,
                "displacement_margin": margin,
            }
        )
        metrics = _evaluate(usable, truth, policy, mode="protected")
        protected_rows.append(
            {"policy": policy.model_dump(mode="json"), "metrics": metrics}
        )
    selected = max(protected_rows, key=lambda row: _selection_key(row["metrics"]))
    selected_policy = SemanticRankingPolicy.model_validate(selected["policy"]).model_copy(
        update={"development_metrics": selected["metrics"]}
    )
    payload = {
        "schema_version": 1,
        "selection_split": "development",
        "validation_used_for_selection": False,
        "development_scenario_count": len(usable),
        "baseline_metrics": _baseline_metrics(usable, truth),
        "selected_aggressive": best_weights,
        "selected_protected": {
            "policy": selected_policy.model_dump(mode="json"),
            "metrics": selected["metrics"],
        },
        "weight_candidate_count": len(weight_rows),
        "protected_candidate_count": len(protected_rows),
        "weight_grid": weight_rows,
        "protected_grid": protected_rows,
    }
    output = _resolve(root, args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"output={output}")
    print(json.dumps(payload["selected_protected"], indent=2))


def _policy(
    *,
    scenario_count: int,
    embedding_weight: float,
    cross_encoder_weight: float,
    structured_weight: float,
) -> SemanticRankingPolicy:
    return SemanticRankingPolicy(
        candidate_limit=20,
        embedding_weight=embedding_weight,
        cross_encoder_weight=cross_encoder_weight,
        structured_weight=structured_weight,
        minimum_intent_confidence=0.5,
        maximum_fusion_alpha=0.5,
        evidence_floor=0.5,
        maximum_upward_move=5,
        maximum_downward_move=3,
        protected_top_k=5,
        displacement_margin=0.1,
        development_scenario_count=scenario_count,
        development_metrics={},
    )


def _rerank(
    result: SemanticRankingResult,
    policy: SemanticRankingPolicy,
    *,
    mode: str,
) -> list[str]:
    scores = []
    for item in result.candidate_scores:
        semantic = (
            policy.embedding_weight * item.embedding_score
            + policy.cross_encoder_weight * item.cross_encoder_score
            + policy.structured_weight * item.structured_score
        )
        scores.append(item.model_copy(update={"semantic_score": semantic}))
    ordered = sorted(scores, key=lambda item: (-item.semantic_score, item.business_id))
    rank_by_id = {item.business_id: rank for rank, item in enumerate(ordered, 1)}
    scores = [
        item.model_copy(update={"semantic_rank": rank_by_id[item.business_id]})
        for item in scores
    ]
    ranking, _, _, _ = apply_ranking_policy(
        base_ranking=result.base_ranking,
        rows=scores,
        intent=result.intent,
        mode=mode,  # type: ignore[arg-type]
        policy=policy,
    )
    return ranking


def _evaluate(
    results: list[SemanticRankingResult],
    truth: dict[str, object],
    policy: SemanticRankingPolicy,
    *,
    mode: str,
) -> dict[str, float]:
    rankings = [
        (
            item,
            _rerank(item, policy, mode=mode),
            set(getattr(truth[str(item.context_id)], "acceptable_business_ids")),
        )
        for item in results
    ]
    return _ranking_metrics(rankings)


def _baseline_metrics(
    results: list[SemanticRankingResult], truth: dict[str, object]
) -> dict[str, float]:
    return _ranking_metrics(
        [
            (
                item,
                item.base_ranking,
                set(getattr(truth[str(item.context_id)], "acceptable_business_ids")),
            )
            for item in results
        ]
    )


def _ranking_metrics(
    rows: list[tuple[SemanticRankingResult, list[str], set[str]]],
) -> dict[str, float]:
    ranks: list[int | None] = []
    base_ranks: list[int | None] = []
    movements: list[float] = []
    for result, ranking, acceptable in rows:
        ranks.append(_first_rank(ranking, acceptable))
        base_ranks.append(_first_rank(result.base_ranking, acceptable))
        positions = {business_id: rank for rank, business_id in enumerate(ranking, 1)}
        movements.extend(
            abs(item.base_rank - positions[item.business_id])
            for item in result.candidate_scores
        )
    denominator = len(rows)
    improved = sum(
        final is not None and (base is None or final < base)
        for final, base in zip(ranks, base_ranks, strict=True)
    )
    harmed = sum(
        base is not None and (final is None or final > base)
        for final, base in zip(ranks, base_ranks, strict=True)
    )
    return {
        "hr_at_1": sum(rank == 1 for rank in ranks) / denominator,
        "hr_at_3": sum(rank is not None and rank <= 3 for rank in ranks) / denominator,
        "hr_at_5": sum(rank is not None and rank <= 5 for rank in ranks) / denominator,
        "mrr": sum(0 if rank is None else 1 / rank for rank in ranks) / denominator,
        "ndcg_at_5": sum(
            0 if rank is None or rank > 5 else 1 / math.log2(rank + 1)
            for rank in ranks
        )
        / denominator,
        "improvement_rate": improved / denominator,
        "harm_rate": harmed / denominator,
        "mean_movement": sum(movements) / len(movements) if movements else 0.0,
    }


def _first_rank(ranking: list[str], acceptable: set[str]) -> int | None:
    return next(
        (rank for rank, business_id in enumerate(ranking, 1) if business_id in acceptable),
        None,
    )


def _selection_key(value: object) -> tuple[float, ...]:
    metrics = value if isinstance(value, dict) else {}
    return (
        float(metrics.get("hr_at_5", 0)),
        float(metrics.get("mrr", 0)),
        float(metrics.get("hr_at_1", 0)),
        -float(metrics.get("harm_rate", 1)),
        -float(metrics.get("mean_movement", 1e9)),
    )


def _load_diagnostics(path: Path) -> list[SemanticRankingResult]:
    if not path.is_file():
        raise FileNotFoundError(f"diagnostics do not exist: {path}")
    return [
        SemanticRankingResult.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _resolve(root: Path, value: Path) -> Path:
    return value if value.is_absolute() else root / value


if __name__ == "__main__":
    main()
