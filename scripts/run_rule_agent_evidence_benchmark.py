"""Run the Step 28 Agent with local retrieval, reranking, RAG, and aggregation."""

from __future__ import annotations

import argparse
from pathlib import Path

from yelp_agent.agent_benchmark import load_scenario_ground_truth
from yelp_agent.rule_router import (
    RuleAgentSourcePaths,
    build_real_rule_agent_runtime,
    run_rule_agent_benchmark,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--split", choices=("development", "validation", "all"), default="all")
    parser.add_argument("--embedding-model-path", type=Path, required=True)
    parser.add_argument("--cross-encoder-model-path", type=Path, required=True)
    parser.add_argument("--model-python", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--affected-only", action="store_true")
    parser.add_argument("--scenario-id", action="append", default=[])
    args = parser.parse_args()
    root = args.project_root.resolve()
    sources = RuleAgentSourcePaths.from_project_root(root)
    output = args.output_root or root / "runs" / "rule_agent_evidence_v1" / args.split
    scenario_ids = set(args.scenario_id)
    if args.affected_only:
        affected = {
            item.scenario_id
            for item in load_scenario_ground_truth(
                sources.benchmark_root / "hidden" / "ground_truth.jsonl"
            )
            if "retrieve_business_reviews" in item.required_actions
        }
        scenario_ids = scenario_ids & affected if scenario_ids else affected
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

    def progress(index: int, total: int, scenario: object) -> None:
        if index == 1 or index % 20 == 0 or index == total:
            print(f"[{index}/{total}] completed", flush=True)

    with build_real_rule_agent_runtime(
        sources,
        embedding_config_path=root / "configs" / "embedding.yaml",
        embedding_environment=embedding_environment,
        cross_encoder_config_path=root / "configs" / "cross_encoder.yaml",
        cross_encoder_environment=cross_environment,
        review_rag_config_path=root / "configs" / "review_rag.yaml",
        review_embedding_environment=embedding_environment,
        evidence_aggregation_config_path=root / "configs" / "evidence_aggregator.yaml",
    ) as runtime:
        result = run_rule_agent_benchmark(
            runtime.harness,
            benchmark_root=sources.benchmark_root,
            output_root=output,
            split=args.split,
            progress=progress,
            scenario_ids=scenario_ids or None,
        )
    print(f"runs={result.runs_path}")
    print(f"metrics={result.metrics_path}")
    print(f"summary={result.summary_path}")


if __name__ == "__main__":
    main()
