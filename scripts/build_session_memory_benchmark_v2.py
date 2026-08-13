"""Build the 500-turn context-grounded Step 34.5 benchmark with DeepSeek."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from dotenv import load_dotenv

from yelp_agent.agent.llm import OpenAICompatibleLLM
from yelp_agent.config import AgentConfig
from yelp_agent.session_memory_benchmark.artifacts import freeze_benchmark_v2
from yelp_agent.session_memory_benchmark.config import (
    load_session_memory_benchmark_v2_config,
)
from yelp_agent.session_memory_benchmark.generation import generate_and_review_turns
from yelp_agent.session_memory_benchmark.planner import BenchmarkV2Planner
from yelp_agent.session_memory_benchmark.sources import (
    BenchmarkV2Sources,
    load_planning_contexts,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-root", type=Path, default=Path.cwd())
    parser.add_argument("--source-project-root", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    load_dotenv(args.env_file, override=False)
    code_root = args.code_root.resolve()
    config = load_session_memory_benchmark_v2_config(
        code_root / "configs" / "session_memory_benchmark_v2.yaml"
    )
    sources = BenchmarkV2Sources.from_project_root(
        args.source_project_root.resolve(), pipeline_version=config.pipeline_version
    )
    contexts = load_planning_contexts(sources)
    plan = BenchmarkV2Planner(config).plan(contexts)
    llm = OpenAICompatibleLLM.from_environment(
        AgentConfig(
            enabled=True,
            temperature=0.0,
            timeout_seconds=config.generation_timeout_seconds,
            max_retries=config.generation_max_retries,
            max_tokens=config.generation_max_output_tokens,
            response_format_json=True,
            thinking="disabled",
        )
    )
    generated = generate_and_review_turns(
        plan.turn_specs,
        generator=llm,
        reviewer=llm,
        config=config,
        progress=_progress,
    )
    run_root = _resolve(code_root, args.run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "generation").mkdir(parents=True, exist_ok=True)
    (run_root / "generation" / "accepted_turns.jsonl").write_text(
        "".join(item.model_dump_json() + "\n" for item in generated.turns),
        encoding="utf-8",
    )
    (run_root / "generation" / "ground_truth.jsonl").write_text(
        "".join(item.model_dump_json() + "\n" for item in generated.ground_truth),
        encoding="utf-8",
    )
    (run_root / "generation" / "provider_calls.jsonl").write_text(
        "".join(json.dumps(asdict(item), ensure_ascii=False, sort_keys=True) + "\n" for item in generated.calls),
        encoding="utf-8",
    )
    (run_root / "generation" / "usage.json").write_text(
        json.dumps(
            {
                "provider_call_count": generated.provider_call_count,
                "generation_input_tokens": generated.generation_input_tokens,
                "generation_output_tokens": generated.generation_output_tokens,
                "review_input_tokens": generated.review_input_tokens,
                "review_output_tokens": generated.review_output_tokens,
                "rejected_generation_count": generated.rejected_generation_count,
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    (run_root / "generation_plan.json").write_text(
        json.dumps(plan.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    result = freeze_benchmark_v2(
        plan=plan,
        generated=generated,
        config=config,
        output_root=_resolve(code_root, args.benchmark_root),
        run_output_root=run_root,
    )
    print(f"benchmark={result.benchmark_root}")
    print(f"turns={result.manifest.turn_count}")
    print(f"provider_calls={result.manifest.provider_call_count}")
    print(
        "tokens="
        f"{result.manifest.generation_input_tokens + result.manifest.review_input_tokens}/"
        f"{result.manifest.generation_output_tokens + result.manifest.review_output_tokens}"
    )


def _progress(accepted: int, total: int, round_index: int) -> None:
    print(f"[generation round {round_index}] accepted={accepted}/{total}", flush=True)


def _resolve(root: Path, value: Path) -> Path:
    return value if value.is_absolute() else root / value


if __name__ == "__main__":
    main()
