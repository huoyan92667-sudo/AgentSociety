"""Run and evaluate the full Agent on Query Recommendation Benchmark V1.

The ``run`` stage opens only ``visible/cases.jsonl``.  The ``evaluate`` stage
first verifies the frozen prediction hash and only then opens hidden labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

from dotenv import load_dotenv

from yelp_agent.query_recommendation_agent import (
    BusinessConditionIndex,
    QueryRecommendationAgentPrediction,
    QueryRecommendationAgentRunner,
    evaluate_frozen_predictions,
    load_predictions,
    verify_visible_run,
    write_evaluation,
    write_visible_run,
)
from yelp_agent.query_recommendation_benchmark import (
    VisibleQueryRecommendationCase,
    load_query_recommendation_bundle,
)
from yelp_agent.rule_router import RuleAgentSourcePaths, build_real_rule_agent_runtime


MAX_EXPERIMENT_CASES = 500


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path.cwd())
    parser.add_argument("--code-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=Path("benchmarks/query_recommendation_v1"),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--stage", choices=("run", "evaluate", "all"), default="all")
    parser.add_argument(
        "--router-kind",
        choices=("constrained", "rule"),
        default="constrained",
    )
    parser.add_argument(
        "--split",
        choices=("development", "validation", "all"),
        default="all",
    )
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--embedding-model-path", type=Path, default=None)
    parser.add_argument("--cross-encoder-model-path", type=Path, default=None)
    parser.add_argument("--model-python", type=Path, default=None)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--env-file", type=Path, default=None)
    parser.add_argument(
        "--memory-config", type=Path, default=Path("configs/session_memory.yaml")
    )
    parser.add_argument(
        "--controlled-llm-config",
        type=Path,
        default=Path("configs/controlled_llm.yaml"),
    )
    parser.add_argument(
        "--semantic-ranking-config",
        type=Path,
        default=Path("configs/semantic_ranking.yaml"),
    )
    parser.add_argument(
        "--router-config",
        type=Path,
        default=Path("configs/constrained_llm_router.yaml"),
    )
    parser.add_argument(
        "--harness-config",
        type=Path,
        default=Path("configs/agent_harness_constrained_llm.yaml"),
    )
    parser.add_argument(
        "--query-aware-config",
        type=Path,
        default=Path("configs/query_aware_ranking.yaml"),
    )
    parser.add_argument(
        "--query-aware-policy",
        type=Path,
        default=Path("configs/query_aware_ranking_policy.json"),
    )
    args = parser.parse_args()
    if args.limit is None or not 1 <= args.limit <= MAX_EXPERIMENT_CASES:
        raise ValueError(
            "this cost-bounded experiment requires --limit between 1 and "
            f"{MAX_EXPERIMENT_CASES}"
        )

    code_root = args.code_root.resolve()
    data_root = args.data_root.resolve()
    benchmark_root = _resolve(code_root, args.benchmark_root)
    output_root = _resolve(code_root, args.output_root)
    if args.env_file is not None:
        env_path = _resolve(data_root, args.env_file)
        if not env_path.is_file():
            raise FileNotFoundError(f"environment file does not exist: {env_path}")
        load_dotenv(env_path, override=False)

    if args.stage in {"run", "all"}:
        _run_stage(args, data_root, code_root, benchmark_root, output_root)
    if args.stage in {"evaluate", "all"}:
        _evaluate_stage(data_root, benchmark_root, output_root)


def _run_stage(
    args: argparse.Namespace,
    data_root: Path,
    code_root: Path,
    benchmark_root: Path,
    output_root: Path,
) -> None:
    model_paths = (
        args.embedding_model_path,
        args.cross_encoder_model_path,
        args.model_python,
    )
    if any(value is None for value in model_paths):
        raise ValueError(
            "run stage requires --embedding-model-path, "
            "--cross-encoder-model-path, and --model-python"
        )
    visible_path = benchmark_root / "visible" / "cases.jsonl"
    visible = _select_visible(
        _load_visible_cases(visible_path),
        split=args.split,
        case_ids=set(args.case_id),
        limit=args.limit,
    )
    existing = _load_checkpoint(output_root / "checkpoint_predictions.jsonl")
    selected_ids = {item.case_id for item in visible}
    existing = {key: value for key, value in existing.items() if key in selected_ids}
    remaining = [item for item in visible if item.case_id not in existing]
    print(
        f"selected={len(visible)} resumed={len(existing)} remaining={len(remaining)}",
        flush=True,
    )
    sources = RuleAgentSourcePaths.from_project_root(data_root)
    embedding_environment = {
        "LOCAL_EMBEDDING_MODEL_PATH": str(args.embedding_model_path.resolve()),
        "LOCAL_EMBEDDING_PYTHON": str(args.model_python.resolve()),
        "LOCAL_EMBEDDING_DEVICE": args.device,
    }
    cross_environment = {
        "LOCAL_CROSS_ENCODER_MODEL_PATH": str(args.cross_encoder_model_path.resolve()),
        "LOCAL_CROSS_ENCODER_PYTHON": str(args.model_python.resolve()),
        "LOCAL_CROSS_ENCODER_DEVICE": args.device,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    if remaining:
        with build_real_rule_agent_runtime(
            sources,
            embedding_environment=embedding_environment,
            cross_encoder_environment=cross_environment,
            controlled_llm_config_path=(
                _resolve(code_root, args.controlled_llm_config)
                if args.router_kind == "constrained"
                else None
            ),
            session_memory_config_path=(
                _resolve(code_root, args.memory_config)
                if args.router_kind == "constrained"
                else None
            ),
            constrained_router_config_path=(
                _resolve(code_root, args.router_config)
                if args.router_kind == "constrained"
                else None
            ),
            agent_harness_config_path=_resolve(code_root, args.harness_config),
            query_retrieval_mode="history_only",
            query_aware_ranking_config_path=_resolve(
                code_root, args.query_aware_config
            ),
            query_aware_ranking_policy_path=_resolve(
                code_root, args.query_aware_policy
            ),
        ) as runtime:
            runner = QueryRecommendationAgentRunner(runtime.harness)
            for index, case in enumerate(remaining, start=1):
                prediction = runner.run((case,))[0]
                existing[case.case_id] = prediction
                _append_checkpoint(
                    output_root / "checkpoint_predictions.jsonl",
                    prediction,
                )
                print(
                    f"[{len(existing)}/{len(visible)}] case={case.case_id[:12]} "
                    f"status={prediction.status} response={prediction.response_kind} "
                    f"tokens={(prediction.input_tokens or 0) + (prediction.output_tokens or 0)}",
                    flush=True,
                )
            _write_llm_ledgers(output_root, runtime)
    predictions = tuple(existing[item.case_id] for item in visible)
    manifest = write_visible_run(
        predictions,
        output_root=output_root,
        visible_cases_path=visible_path,
    )
    print(f"predictions={output_root / 'predictions.jsonl'}", flush=True)
    print(f"prediction_sha256={manifest.predictions_sha256}", flush=True)


def _evaluate_stage(
    data_root: Path,
    benchmark_root: Path,
    output_root: Path,
) -> None:
    manifest = verify_visible_run(output_root)
    visible_path = benchmark_root / "visible" / "cases.jsonl"
    if _sha256(visible_path) != manifest.visible_cases_sha256:
        raise ValueError("visible benchmark changed after Agent predictions were frozen")
    predictions = load_predictions(output_root / "predictions.jsonl")
    prediction_ids = {item.case_id for item in predictions}
    # Hidden files are first opened here, after the prediction hash was verified.
    bundle = load_query_recommendation_bundle(benchmark_root)
    visible = tuple(item for item in bundle.visible_cases if item.case_id in prediction_ids)
    truth = tuple(item for item in bundle.ground_truth if item.case_id in prediction_ids)
    frames = tuple(item for item in bundle.frames if item.case_id in prediction_ids)
    facts = BusinessConditionIndex.from_parquet(
        businesses_path=data_root / "data" / "processed" / "businesses.parquet",
        aspect_events_path=(
            data_root
            / "data"
            / "features"
            / "business_profiles"
            / "v1"
            / "aspect_events.parquet"
        ),
    )
    evaluation = evaluate_frozen_predictions(
        visible_cases=visible,
        ground_truth=truth,
        frames=frames,
        predictions=predictions,
        facts=facts,
    )
    metrics_path, _, audits_path = write_evaluation(
        evaluation,
        output_root=output_root,
    )
    print(f"metrics={metrics_path}", flush=True)
    print(f"case_audits={audits_path}", flush=True)
    for name in ("Recall@500", "HR@1", "HR@5", "HR@10", "QueryCompliance@5"):
        metric = evaluation.report.metrics[name]
        print(f"{name}={metric.value} status={metric.status}", flush=True)


def _load_visible_cases(path: Path) -> tuple[VisibleQueryRecommendationCase, ...]:
    if not path.is_file():
        raise FileNotFoundError(f"visible Query benchmark does not exist: {path}")
    return tuple(
        VisibleQueryRecommendationCase.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def _select_visible(
    values: Sequence[VisibleQueryRecommendationCase],
    *,
    split: str,
    case_ids: set[str],
    limit: int | None,
) -> tuple[VisibleQueryRecommendationCase, ...]:
    selected = [
        item
        for item in values
        if (split == "all" or item.split == split)
        and (not case_ids or item.case_id in case_ids)
    ]
    if case_ids - {item.case_id for item in selected}:
        raise ValueError("one or more requested case IDs are outside the selected split")
    selected.sort(key=lambda item: item.case_id)
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        selected = selected[:limit]
    if not selected:
        raise ValueError("Query Agent selection is empty")
    return tuple(selected)


def _load_checkpoint(path: Path) -> dict[str, QueryRecommendationAgentPrediction]:
    if not path.is_file():
        return {}
    output: dict[str, QueryRecommendationAgentPrediction] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = QueryRecommendationAgentPrediction.model_validate_json(line)
        output[value.case_id] = value
    return output


def _append_checkpoint(path: Path, prediction: QueryRecommendationAgentPrediction) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(prediction.model_dump_json(exclude_computed_fields=True) + "\n")
        handle.flush()


def _write_llm_ledgers(output_root: Path, runtime: object) -> None:
    summaries: dict[str, object] = {}
    for name, component in (
        ("router", getattr(runtime, "constrained_router", None)),
        ("semantic_and_answer", getattr(runtime, "controlled_llm", None)),
        ("session_memory", getattr(runtime, "session_memory", None)),
    ):
        ledger = getattr(component, "ledger", None)
        if ledger is None:
            summaries[name] = {}
            continue
        ledger.write(output_root / f"{name}_llm")
        summaries[name] = ledger.summary()
    summaries["cumulative_known_tokens"] = sum(
        int(value.get("total_tokens", 0))
        for value in summaries.values()
        if isinstance(value, dict)
    )
    (output_root / "total_llm_usage.json").write_text(
        json.dumps(summaries, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def _resolve(root: Path, value: Path) -> Path:
    return value if value.is_absolute() else root / value


if __name__ == "__main__":
    main()
