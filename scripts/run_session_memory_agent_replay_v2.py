"""Run complete local retrieval/ranking Agent with Rule or DeepSeek memory."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from dotenv import load_dotenv

from yelp_agent.rule_router import RuleAgentSourcePaths, build_real_rule_agent_runtime
from yelp_agent.session_memory_benchmark.agent_replay import run_agent_replay_v2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-root", type=Path, default=Path.cwd())
    parser.add_argument("--source-project-root", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("rule", "api", "cache"), required=True)
    parser.add_argument("--embedding-model-path", type=Path, required=True)
    parser.add_argument("--cross-encoder-model-path", type=Path, required=True)
    parser.add_argument("--model-python", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.env_file is not None:
        load_dotenv(args.env_file, override=False)
    code = args.code_root.resolve()
    original = RuleAgentSourcePaths.from_project_root(args.source_project_root)
    sources = replace(
        original,
        # Keep immutable data paths on the source project, while all mutable
        # SQLite caches and diagnostics stay inside this worktree.
        project_root=code,
        rule_router_config=code / "configs" / "rule_router_memory_v2.yaml",
        agent_harness_config=code / "configs" / "agent_harness_memory_v2.yaml",
        agent_tools_config=code / "configs" / "agent_tools.yaml",
    )
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
    memory_config = code / "configs" / {
        "api": "session_memory.yaml",
        "cache": "session_memory_cache_only.yaml",
        "rule": "session_memory_rule.yaml",
    }[args.mode]
    output = _resolve(code, args.output_root)
    with build_real_rule_agent_runtime(
        sources,
        embedding_config_path=code / "configs" / "embedding.yaml",
        embedding_environment=embedding_environment,
        cross_encoder_config_path=code / "configs" / "cross_encoder.yaml",
        cross_encoder_environment=cross_environment,
        semantic_ranking_config_path=code / "configs" / "semantic_ranking.yaml",
        session_memory_config_path=memory_config,
    ) as runtime:
        result = run_agent_replay_v2(
            runtime.harness,
            benchmark_root=_resolve(code, args.benchmark_root),
            output_root=output,
            progress=_progress,
            limit=args.limit,
        )
        if runtime.session_memory is not None:
            runtime.session_memory.ledger.write(output / "memory_llm")
    print(f"runs={result.runs_path}")
    print(f"metrics={result.metrics_path}")


def _progress(current: int, total: int) -> None:
    if current == 1 or current % 10 == 0 or current == total:
        print(f"[{current}/{total}] sessions", flush=True)


def _resolve(root: Path, value: Path) -> Path:
    return value if value.is_absolute() else root / value


if __name__ == "__main__":
    main()
