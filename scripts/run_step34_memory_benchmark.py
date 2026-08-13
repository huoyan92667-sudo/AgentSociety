from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from yelp_agent.session_memory.config import load_session_memory_config
from yelp_agent.session_memory.evaluation import run_session_memory_benchmark
from yelp_agent.session_memory.manager import SessionMemoryManager
from yelp_agent.session_memory.runtime import build_session_memory_runtime


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--mode",
        choices=("api", "rule"),
        default="rule",
        help="rule is the no-key reproducible baseline; api uses configured DeepSeek",
    )
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--env-file", type=Path, default=None)
    args = parser.parse_args()
    if args.env_file is not None:
        if not args.env_file.is_file():
            raise FileNotFoundError(f"environment file does not exist: {args.env_file}")
        load_dotenv(args.env_file, override=False)
    root = args.project_root.resolve()
    config = load_session_memory_config(root / "configs" / "session_memory.yaml")
    output = args.output_root or root / "runs" / "session_memory_v1" / args.mode
    if args.mode == "api":
        with build_session_memory_runtime(
            project_root=root,
            config=config,
        ) as runtime:
            result = run_session_memory_benchmark(
                runtime.manager,
                benchmark_root=root / "benchmarks" / "agent_scenarios_v1",
                output_root=output,
            )
            runtime.ledger.write(output)
    else:
        result = run_session_memory_benchmark(
            SessionMemoryManager(config=config.model_copy(update={"enabled": False})),
            benchmark_root=root / "benchmarks" / "agent_scenarios_v1",
            output_root=output,
        )
    print(f"cases={result.cases_path}")
    print(f"metrics={result.metrics_path}")
    print(f"summary={result.summary_path}")


if __name__ == "__main__":
    main()
