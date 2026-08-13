"""Run the complete Step 35 Agent with a constrained DeepSeek Router."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from yelp_agent.agent_benchmark import load_visible_scenarios
from yelp_agent.agent_evaluation import load_agent_scenario_runs
from yelp_agent.constrained_llm_router import write_router_report
from yelp_agent.controlled_llm import augment_runtime_metrics
from yelp_agent.rule_router import (
    RuleAgentSourcePaths,
    build_real_rule_agent_runtime,
    run_rule_agent_benchmark,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--code-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--router-kind",
        choices=("constrained", "rule"),
        default="constrained",
        help="Change only the Router while keeping the complete runtime fixed.",
    )
    parser.add_argument(
        "--split", choices=("development", "validation", "all"), required=True
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--scenario-id", action="append", default=[])
    parser.add_argument("--embedding-model-path", type=Path, required=True)
    parser.add_argument("--cross-encoder-model-path", type=Path, required=True)
    parser.add_argument("--model-python", type=Path, required=True)
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
    parser.add_argument("--baseline-root", type=Path, default=None)
    args = parser.parse_args()

    if args.env_file is not None:
        if not args.env_file.is_file():
            raise FileNotFoundError(f"environment file does not exist: {args.env_file}")
        load_dotenv(args.env_file, override=False)
    data_root = args.project_root.resolve()
    code_root = args.code_root.resolve()
    sources = RuleAgentSourcePaths.from_project_root(data_root)
    selected = _selected_scenarios(sources, args.split, args.scenario_id, args.limit)
    embedding_environment = {
        "LOCAL_EMBEDDING_MODEL_PATH": str(args.embedding_model_path),
        "LOCAL_EMBEDDING_PYTHON": str(args.model_python),
        "LOCAL_EMBEDDING_DEVICE": args.device,
    }
    cross_environment = {
        "LOCAL_CROSS_ENCODER_MODEL_PATH": str(args.cross_encoder_model_path),
        "LOCAL_CROSS_ENCODER_PYTHON": str(args.model_python),
        "LOCAL_CROSS_ENCODER_DEVICE": args.device,
    }
    output = _resolve(code_root, args.output_root)
    with build_real_rule_agent_runtime(
        sources,
        embedding_config_path=data_root / "configs" / "embedding.yaml",
        embedding_environment=embedding_environment,
        cross_encoder_config_path=data_root / "configs" / "cross_encoder.yaml",
        cross_encoder_environment=cross_environment,
        review_rag_config_path=data_root / "configs" / "review_rag.yaml",
        review_embedding_environment=embedding_environment,
        evidence_aggregation_config_path=(
            data_root / "configs" / "evidence_aggregator.yaml"
        ),
        controlled_llm_config_path=_resolve(data_root, args.controlled_llm_config),
        semantic_ranking_config_path=_resolve(data_root, args.semantic_ranking_config),
        session_memory_config_path=_resolve(data_root, args.memory_config),
        constrained_router_config_path=(
            _resolve(code_root, args.router_config)
            if args.router_kind == "constrained"
            else None
        ),
        agent_harness_config_path=_resolve(code_root, args.harness_config),
    ) as runtime:
        result = run_rule_agent_benchmark(
            runtime.harness,
            benchmark_root=sources.benchmark_root,
            output_root=output,
            split=args.split,
            scenario_ids=selected or None,
            progress=_progress,
        )
        if runtime.controlled_llm is not None:
            runtime.controlled_llm.ledger.write(output / "answer_llm")
        if runtime.session_memory is not None:
            runtime.session_memory.ledger.write(output / "memory_llm")
        if args.router_kind == "constrained":
            if runtime.constrained_router is None:
                raise RuntimeError("constrained Router runtime was not assembled")
            runtime.constrained_router.ledger.write(output / "router_llm")
            runs = load_agent_scenario_runs(result.runs_path)
            router_metrics_path, router_summary_path = write_router_report(
                runs, output / "router"
            )
            router_usage = json.loads(
                router_metrics_path.read_text(encoding="utf-8")
            )
            _augment_router_metrics(result.runtime_metrics_path, router_usage)
            _write_total_usage(output, runtime, router_usage)
            print(f"router_metrics={router_metrics_path}")
            print(f"router_summary={router_summary_path}")
        else:
            _write_total_usage(output, runtime, {})
        if args.baseline_root is not None:
            _write_comparison(
                output,
                _resolve(code_root, args.baseline_root),
                result.metrics_path,
            )
    print(f"runs={result.runs_path}")
    print(f"metrics={result.metrics_path}")


def _selected_scenarios(
    sources: RuleAgentSourcePaths,
    split: str,
    requested: list[str],
    limit: int | None,
) -> set[str]:
    selected = set(requested)
    if limit is None:
        return selected
    if limit < 1:
        raise ValueError("limit must be positive")
    visible = [
        item
        for item in load_visible_scenarios(
            sources.benchmark_root / "visible" / "scenarios.jsonl"
        )
        if split == "all" or item.split == split
    ]
    eligible = [item for item in visible if not selected or item.scenario_id in selected]
    limited = {
        item.scenario_id
        for item in sorted(eligible, key=lambda item: item.scenario_id)[:limit]
    }
    return selected & limited if selected else limited


def _augment_router_metrics(path: Path, usage: dict[str, object]) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(
        {
            "router_decision_count": usage["decision_count"],
            "router_model_decision_count": usage["model_decision_count"],
            "router_single_choice_bypass_count": usage[
                "single_choice_bypass_count"
            ],
            "router_rule_fallback_count": usage["rule_fallback_count"],
            "router_task_correction_count": usage["task_correction_count"],
            "router_information_gap_correction_count": usage[
                "information_gap_correction_count"
            ],
            "router_provider_call_count": usage["provider_call_count"],
            "router_input_tokens": usage["input_tokens"],
            "router_output_tokens": usage["output_tokens"],
            "router_total_tokens": usage["total_tokens"],
            "router_mean_latency_ms": usage["mean_latency_ms"],
        }
    )
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_total_usage(output: Path, runtime: object, router: dict[str, object]) -> None:
    controlled = getattr(runtime, "controlled_llm", None)
    memory = getattr(runtime, "session_memory", None)
    answer_summary = (
        {} if controlled is None else controlled.ledger.summary()
    )
    memory_summary = {} if memory is None else memory.ledger.summary()
    payload = {
        "schema_version": 1,
        "router": router,
        "semantic_and_answer": answer_summary,
        "session_memory": memory_summary,
        "cumulative_known_tokens": int(router.get("total_tokens", 0))
        + int(answer_summary.get("total_tokens", 0))
        + int(memory_summary.get("total_tokens", 0)),
        "usage_unknown_count": int(router.get("usage_unknown_count", 0))
        + int(answer_summary.get("usage_unknown_count", 0))
        + int(memory_summary.get("usage_unknown_count", 0)),
    }
    (output / "total_llm_usage.json").write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_comparison(output: Path, baseline: Path, current_metrics: Path) -> None:
    old = json.loads((baseline / "metrics.json").read_text(encoding="utf-8"))
    new = json.loads(current_metrics.read_text(encoding="utf-8"))
    keys = (
        "action_accuracy",
        "tool_selection_accuracy",
        "invalid_action_rate",
        "direct_return_precision",
        "missing_field_detection_precision",
        "missing_field_detection_recall",
        "unnecessary_question_rate",
        "fallback_rate",
        "valid_candidate_rate",
        "business_scope_isolation_rate",
        "citation_correctness",
        "mean_latency_ms",
        "p95_latency_ms",
    )
    comparison = {}
    for key in keys:
        before = _metric_value(old, key)
        after = _metric_value(new, key)
        comparison[key] = {
            "rule_router": before,
            "constrained_llm_router": after,
            "delta": None if before is None or after is None else after - before,
        }
    (output / "router_comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _metric_value(payload: dict[str, object], key: str) -> float | None:
    metrics = payload.get("metrics")
    row = metrics.get(key) if isinstance(metrics, dict) else None
    value = row.get("value") if isinstance(row, dict) else None
    return float(value) if isinstance(value, (int, float)) else None


def _progress(index: int, total: int, _scenario: object) -> None:
    if index == 1 or index % 20 == 0 or index == total:
        print(f"[{index}/{total}] completed", flush=True)


def _resolve(root: Path, value: Path) -> Path:
    return value if value.is_absolute() else root / value


if __name__ == "__main__":
    main()
