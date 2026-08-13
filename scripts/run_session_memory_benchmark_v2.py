"""Run the frozen Benchmark V2 against Rule or DeepSeek session memory."""

from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from yelp_agent.session_memory.config import load_session_memory_config
from yelp_agent.session_memory.manager import SessionMemoryManager
from yelp_agent.session_memory.runtime import build_session_memory_runtime
from yelp_agent.session_memory_benchmark.evaluation import run_memory_benchmark_v2
from yelp_agent.session_memory_benchmark.sources import (
    BenchmarkV2Sources,
    load_planning_contexts,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--mode", choices=("rule", "api"), required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--source-project-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    benchmark = _resolve(root, args.benchmark_root)
    output = _resolve(root, args.output_root)
    if args.env_file is not None:
        load_dotenv(args.env_file, override=False)
    config = load_session_memory_config(root / "configs" / "session_memory.yaml")
    candidate_contexts = {
        item.initial_session.source_scenario_id: item.candidate_businesses
        for item in load_planning_contexts(
            BenchmarkV2Sources.from_project_root(
                args.source_project_root.resolve(),
                pipeline_version="step30-llm-protected-full",
            )
        )
    }
    if args.mode == "rule":
        result = run_memory_benchmark_v2(
            SessionMemoryManager(config=config.model_copy(update={"enabled": False})),
            benchmark_root=benchmark,
            output_root=output,
            progress=_progress,
            candidate_contexts=candidate_contexts,
        )
    else:
        runtime = build_session_memory_runtime(project_root=root, config=config)
        try:
            result = run_memory_benchmark_v2(
                runtime.manager,
                benchmark_root=benchmark,
                output_root=output,
                progress=lambda current, total: _api_progress(
                    current, total, runtime, output
                ),
                candidate_contexts=candidate_contexts,
            )
        finally:
            runtime.ledger.write(output / "llm")
            runtime.close()
    print(f"metrics={result.metrics_path}")
    print(f"summary={result.summary_path}")


def _progress(current: int, total: int) -> None:
    if current == 1 or current % 20 == 0 or current == total:
        print(f"[{current}/{total}] sessions", flush=True)


def _api_progress(current: int, total: int, runtime: object, output: Path) -> None:
    _progress(current, total)
    if current % 10 == 0 or current == total:
        runtime.ledger.write(output / "llm")


def _resolve(root: Path, value: Path) -> Path:
    return value if value.is_absolute() else root / value


if __name__ == "__main__":
    main()
