"""Prepare, tune, finalize, and evaluate the frozen Step 33 ranking pipeline."""

from __future__ import annotations

import argparse
import json
from contextlib import ExitStack
from pathlib import Path
from typing import Sequence, TypeVar

from pydantic import BaseModel

from yelp_agent.cross_encoder import LocalCrossEncoderEnvironment
from yelp_agent.query_aware_ranking import (
    PreparedQueryAwareCase,
    QueryAwareFinalizationRuntime,
    QueryAwarePreparationRuntime,
    QueryAwareRankingPolicy,
    QueryAwareRankingResult,
    QueryAwareRankingSources,
    apply_coarse_policy,
    build_benchmark_runs,
    evaluate_query_aware_ranking,
    load_query_aware_ranking_config,
    load_query_aware_ranking_policy,
    select_query_weight,
    write_query_aware_ranking_policy,
)
from yelp_agent.query_recommendation_benchmark import (
    QueryRecommendationGroundTruth,
    VisibleQueryRecommendationCase,
)
from yelp_agent.query.schema import RecommendationRequest
from yelp_agent.semantic_embedding import LocalEmbeddingEnvironment
from yelp_agent.semantic_ranking import SemanticRankingPolicy


ModelT = TypeVar("ModelT", bound=BaseModel)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--config-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=Path("benchmarks/query_recommendation_v1"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("runs/query_aware_ranking_v1"),
    )
    parser.add_argument(
        "--stage",
        choices=("prepare", "tune", "finalize", "evaluate", "all"),
        default="all",
    )
    parser.add_argument("--embedding-model-path", type=Path)
    parser.add_argument("--embedding-python", type=Path)
    parser.add_argument("--embedding-device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--cross-model-path", type=Path)
    parser.add_argument("--cross-python", type=Path)
    parser.add_argument("--cross-device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--limit", type=int)
    return parser


def _load_jsonl(path: Path, model: type[ModelT]) -> tuple[ModelT, ...]:
    return tuple(
        _validate_json(line, model)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def _validate_json(value: str, model: type[ModelT]) -> ModelT:
    """Read canonical artifacts plus the pre-fix request-computed-field format."""

    payload = json.loads(value)
    request = payload.get("request") if isinstance(payload, dict) else None
    if isinstance(request, dict):
        for field in RecommendationRequest.model_computed_fields:
            request.pop(field, None)
    return model.model_validate(payload)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    payload = (
        value.model_dump(mode="json", exclude_computed_fields=True)
        if hasattr(value, "model_dump")
        else value
    )
    partial.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    partial.replace(path)


def _write_jsonl(path: Path, values: Sequence[BaseModel]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(
        "".join(
            item.model_dump_json(exclude_computed_fields=True) + "\n"
            for item in values
        ),
        encoding="utf-8",
        newline="\n",
    )
    partial.replace(path)


def _initial_policy(project_root: Path, config_root: Path) -> QueryAwareRankingPolicy:
    config = load_query_aware_ranking_config(
        config_root / "configs" / "query_aware_ranking.yaml"
    )
    base_path = project_root / config.semantic_policy_source
    semantic = SemanticRankingPolicy.model_validate_json(
        base_path.read_text(encoding="utf-8")
    ).model_copy(update={"candidate_limit": config.semantic_candidate_limit})
    return QueryAwareRankingPolicy(
        selected_query_weight=config.query_weight_candidates[0],
        query_retrieval_signal_weight=config.query_retrieval_signal_weight,
        embedding_signal_weight=config.embedding_signal_weight,
        union_candidate_limit=config.union_candidate_limit,
        coarse_diagnostic_limit=config.coarse_diagnostic_limit,
        semantic_candidate_limit=config.semantic_candidate_limit,
        internal_result_limit=config.internal_result_limit,
        display_limit=config.display_limit,
        tie_breakers=["MRR", "HR@1", "NDCG@10", "lower_query_weight"],
        development_case_count=1,
        development_metrics={},
        semantic_policy=semantic,
    )


def _selected_visible(args: argparse.Namespace) -> tuple[VisibleQueryRecommendationCase, ...]:
    visible = _load_jsonl(
        args.benchmark_root / "visible" / "cases.jsonl",
        VisibleQueryRecommendationCase,
    )
    if args.limit is not None:
        if args.limit < 1:
            raise SystemExit("--limit must be positive")
        visible = visible[: args.limit]
    return visible


def _prepare(
    args: argparse.Namespace,
    visible: Sequence[VisibleQueryRecommendationCase],
    sources: QueryAwareRankingSources,
) -> tuple[PreparedQueryAwareCase, ...]:
    config = load_query_aware_ranking_config(
        args.config_root / "configs" / "query_aware_ranking.yaml"
    )
    cache_root = args.output_root / "prepared_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    prepared: list[PreparedQueryAwareCase] = []
    missing_ids = {
        case.case_id
        for case in visible
        if not (cache_root / f"{case.case_id}.json").is_file()
    }
    environment_kwargs = {
        "model_path": args.embedding_model_path,
        "device": args.embedding_device,
    }
    if args.embedding_python is not None:
        environment_kwargs["python_executable"] = args.embedding_python
    environment = LocalEmbeddingEnvironment.model_validate(environment_kwargs)
    if missing_ids and not environment.enabled:
        raise SystemExit("--embedding-model-path is required for uncached preparation")
    with ExitStack() as stack:
        runtime = (
            stack.enter_context(
                QueryAwarePreparationRuntime.from_sources(
                    sources,
                    config=config,
                    provisional_policy=_initial_policy(
                        args.project_root,
                        args.config_root,
                    ),
                    embedding_environment=environment,
                )
            )
            if missing_ids
            else None
        )
        for index, case in enumerate(visible, start=1):
            cache_path = cache_root / f"{case.case_id}.json"
            if cache_path.is_file():
                item = _validate_json(
                    cache_path.read_text(encoding="utf-8"),
                    PreparedQueryAwareCase,
                )
                _write_json(cache_path, item)
            else:
                assert runtime is not None
                item = runtime.prepare_case(case)
                _write_json(cache_path, item)
            prepared.append(item)
            if index == 1 or index % 5 == 0 or index == len(visible):
                print(
                    json.dumps(
                        {"stage": "prepare", "completed": index, "total": len(visible)},
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
    _write_jsonl(args.output_root / "prepared_cases.jsonl", prepared)
    return tuple(prepared)


def _tune(
    args: argparse.Namespace,
    prepared: Sequence[PreparedQueryAwareCase],
) -> QueryAwareRankingPolicy:
    config = load_query_aware_ranking_config(
        args.config_root / "configs" / "query_aware_ranking.yaml"
    )
    truth = _load_jsonl(
        args.benchmark_root / "hidden" / "ground_truth.jsonl",
        QueryRecommendationGroundTruth,
    )
    selected_ids = {item.case_id for item in prepared}
    target_by_case = {
        item.case_id: item.target_business_id
        for item in truth
        if item.case_id in selected_ids
    }
    semantic = SemanticRankingPolicy.model_validate_json(
        (args.project_root / config.semantic_policy_source).read_text(encoding="utf-8")
    )
    selection = select_query_weight(prepared, target_by_case, config, semantic)
    policy_path = args.project_root / config.policy_relative_path
    write_query_aware_ranking_policy(selection.policy, policy_path)
    _write_json(args.output_root / "development_selection.json", selection)
    print(
        json.dumps(
            {
                "stage": "tune",
                "selected_query_weight": selection.policy.selected_query_weight,
                "development_metrics": selection.policy.development_metrics,
                "validation_used_for_selection": False,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    return selection.policy


def _finalize(
    args: argparse.Namespace,
    prepared: Sequence[PreparedQueryAwareCase],
    sources: QueryAwareRankingSources,
    policy: QueryAwareRankingPolicy,
) -> tuple[QueryAwareRankingResult, ...]:
    cache_root = args.output_root / "final_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    results: list[QueryAwareRankingResult] = []
    missing_ids = {
        item.case_id
        for item in prepared
        if not (cache_root / f"{item.case_id}.json").is_file()
    }
    environment_kwargs = {
        "model_path": args.cross_model_path,
        "device": args.cross_device,
    }
    if args.cross_python is not None:
        environment_kwargs["python_executable"] = args.cross_python
    environment = LocalCrossEncoderEnvironment.model_validate(environment_kwargs)
    if missing_ids and not environment.enabled:
        raise SystemExit("--cross-model-path is required for uncached finalization")
    with ExitStack() as stack:
        runtime = (
            stack.enter_context(
                QueryAwareFinalizationRuntime.from_sources(
                    sources,
                    policy=policy,
                    cross_encoder_environment=environment,
                )
            )
            if missing_ids
            else None
        )
        for index, item in enumerate(prepared, start=1):
            cache_path = cache_root / f"{item.case_id}.json"
            if cache_path.is_file():
                result = _validate_json(
                    cache_path.read_text(encoding="utf-8"),
                    QueryAwareRankingResult,
                )
                _write_json(cache_path, result)
            else:
                assert runtime is not None
                result = runtime.finalize_case(item)
                _write_json(cache_path, result)
            results.append(result)
            if index == 1 or index % 5 == 0 or index == len(prepared):
                print(
                    json.dumps(
                        {"stage": "finalize", "completed": index, "total": len(prepared)},
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
    _write_jsonl(args.output_root / "final_results.jsonl", results)
    return tuple(results)


def _evaluate(
    args: argparse.Namespace,
    visible: Sequence[VisibleQueryRecommendationCase],
    prepared: Sequence[PreparedQueryAwareCase],
    final_results: Sequence[QueryAwareRankingResult],
    policy: QueryAwareRankingPolicy,
) -> object:
    coarse_rankings = {
        item.case_id: [row.business_id for row in apply_coarse_policy(item, policy)]
        for item in prepared
    }
    runs = build_benchmark_runs(prepared, final_results, coarse_rankings)
    _write_jsonl(args.output_root / "benchmark_runs.jsonl", runs)
    all_truth = _load_jsonl(
        args.benchmark_root / "hidden" / "ground_truth.jsonl",
        QueryRecommendationGroundTruth,
    )
    selected_ids = {item.case_id for item in visible}
    truth = tuple(item for item in all_truth if item.case_id in selected_ids)
    report = evaluate_query_aware_ranking(
        visible,
        truth,
        prepared,
        final_results,
        runs,
    )
    _write_json(args.output_root / "metrics.json", report)
    print(
        json.dumps(
            {
                "stage": "evaluate",
                "case_count": report.case_count,
                "protected_union_recall": report.protected_union_recall,
                "post_filter_union_recall": report.post_filter_union_recall,
                "metrics": {
                    method: {
                        "hr_at_1": metrics.hr_at_1,
                        "hr_at_3": metrics.hr_at_3,
                        "hr_at_5": metrics.hr_at_5,
                        "hr_at_10": metrics.hr_at_10,
                        "mrr": metrics.mrr,
                    }
                    for method, metrics in report.overall.items()
                },
                "external_model_calls": 0,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.source_root = args.source_root.resolve()
    args.project_root = args.project_root.resolve()
    args.config_root = args.config_root.resolve()
    args.benchmark_root = args.benchmark_root.resolve()
    args.output_root = args.output_root.resolve()
    visible = _selected_visible(args)
    sources = QueryAwareRankingSources(
        source_root=args.source_root,
        project_root=args.project_root,
        config_root=args.config_root,
    )

    prepared_path = args.output_root / "prepared_cases.jsonl"
    final_path = args.output_root / "final_results.jsonl"
    config = load_query_aware_ranking_config(
        args.config_root / "configs" / "query_aware_ranking.yaml"
    )
    policy_path = args.project_root / config.policy_relative_path

    if args.stage in {"prepare", "all"}:
        prepared = _prepare(args, visible, sources)
    else:
        prepared = _load_jsonl(prepared_path, PreparedQueryAwareCase)
    if args.stage == "prepare":
        return 0

    if args.stage in {"tune", "all"}:
        policy = _tune(args, prepared)
    else:
        policy = load_query_aware_ranking_policy(policy_path)
    if args.stage == "tune":
        return 0

    if args.stage in {"finalize", "all"}:
        final_results = _finalize(args, prepared, sources, policy)
    else:
        final_results = _load_jsonl(final_path, QueryAwareRankingResult)
    if args.stage == "finalize":
        return 0

    _evaluate(args, visible, prepared, final_results, policy)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
