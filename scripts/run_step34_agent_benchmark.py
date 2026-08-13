"""Run the complete pre-Step35 Agent with canonical Step34 session memory."""

from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from yelp_agent.agent_benchmark import load_visible_scenarios
from yelp_agent.controlled_llm import augment_runtime_metrics
from yelp_agent.rule_router import (
    RuleAgentSourcePaths,
    build_real_rule_agent_runtime,
    run_rule_agent_benchmark,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-root", type=Path, required=True)
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
        "--memory-config",
        type=Path,
        default=Path("configs/session_memory.yaml"),
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
    args = parser.parse_args()
    if args.env_file is not None:
        if not args.env_file.is_file():
            raise FileNotFoundError(f"environment file does not exist: {args.env_file}")
        load_dotenv(args.env_file, override=False)
    root = args.project_root.resolve()
    sources = RuleAgentSourcePaths.from_project_root(root)
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
    output = _resolve(root, args.output_root)
    with build_real_rule_agent_runtime(
        sources,
        embedding_config_path=root / "configs" / "embedding.yaml",
        embedding_environment=embedding_environment,
        cross_encoder_config_path=root / "configs" / "cross_encoder.yaml",
        cross_encoder_environment=cross_environment,
        review_rag_config_path=root / "configs" / "review_rag.yaml",
        review_embedding_environment=embedding_environment,
        evidence_aggregation_config_path=root / "configs" / "evidence_aggregator.yaml",
        controlled_llm_config_path=_resolve(root, args.controlled_llm_config),
        semantic_ranking_config_path=_resolve(root, args.semantic_ranking_config),
        session_memory_config_path=_resolve(root, args.memory_config),
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
            traces, usage = runtime.session_memory.ledger.write(output / "memory_llm")
            augment_runtime_metrics(
                result.runtime_metrics_path,
                runtime.session_memory.ledger.summary(),
            )
            print(f"memory_calls={traces}")
            print(f"memory_usage={usage}")
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


def _progress(index: int, total: int, _scenario: object) -> None:
    if index == 1 or index % 20 == 0 or index == total:
        print(f"[{index}/{total}] completed", flush=True)


def _resolve(root: Path, value: Path) -> Path:
    return value if value.is_absolute() else root / value


if __name__ == "__main__":
    main()
