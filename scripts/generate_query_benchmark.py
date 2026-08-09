"""Generate the 500-query benchmark with a configured OpenAI-compatible API."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from yelp_agent.agent.llm import LLMCallResult, LLMMessage, OpenAICompatibleLLM
from yelp_agent.config import AgentConfig, load_llm_environment
from yelp_agent.experiments import write_json_artifact, write_text_artifact
from yelp_agent.query.generation import (
    QueryBenchmarkGenerationSummary,
    benchmark_payload,
    build_generation_messages_payload,
    build_query_frame_specs,
    expand_generated_batch,
    parse_generated_batch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/query_aware_v2/queries_500.jsonl"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("benchmarks/query_aware_v2/manifest.json"),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("runs/query_aware_v2/generation_cache"),
    )
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument(
        "--allow-real-api",
        action="store_true",
        help="required safety flag before sending requests to the configured provider",
    )
    return parser.parse_args()


def _sum_optional(values: list[int | None]) -> int | None:
    known = [value for value in values if value is not None]
    return sum(known) if len(known) == len(values) else None


def main() -> None:
    args = parse_args()
    if not args.allow_real_api:
        raise SystemExit("Refusing real API calls without --allow-real-api")
    if args.batch_size < 1 or 100 % args.batch_size:
        raise SystemExit("--batch-size must be a positive divisor of 100")

    environment = load_llm_environment()
    if not environment.llm_enabled or environment.model is None:
        raise SystemExit("OPENAI_API_KEY and OPENAI_MODEL must be configured")
    llm = OpenAICompatibleLLM(
        AgentConfig(
            enabled=True,
            temperature=0.0,
            timeout_seconds=90,
            max_retries=1,
            max_tokens=12000,
            response_format_json=True,
            thinking="disabled",
        ),
        environment,
    )
    frames = build_query_frame_specs()
    all_cases = []
    results: list[LLMCallResult] = []
    api_batch_count = len(frames) // args.batch_size
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    for offset in range(0, len(frames), args.batch_size):
        batch_index = offset // args.batch_size + 1
        batch_frames = frames[offset : offset + args.batch_size]
        system_prompt, user_prompt, prompt_hash = build_generation_messages_payload(
            batch_frames
        )
        cache_path = args.cache_dir / f"batch-{batch_index:02d}-{prompt_hash[:12]}.json"
        if cache_path.is_file():
            cached_result = LLMCallResult.model_validate_json(
                cache_path.read_text(encoding="utf-8")
            )
        else:
            cached_result = None
        if cached_result is not None and cached_result.status == "success":
            result = cached_result
            source = "cache"
        else:
            result = llm.generate(
                (
                    LLMMessage(role="system", content=system_prompt),
                    LLMMessage(role="user", content=user_prompt),
                )
            )
            write_json_artifact(cache_path, result)
            source = "api"
        if result.status != "success" or result.content is None:
            raise RuntimeError(
                f"batch {batch_index} failed: {result.failure_reason or result.status}"
            )
        try:
            generated = parse_generated_batch(result.content, batch_frames)
        except Exception as exc:
            raise ValueError(f"batch {batch_index} returned invalid benchmark JSON") from exc
        batch_cases = expand_generated_batch(
            generated,
            batch_frames,
            model=environment.model,
            prompt_sha256=prompt_hash,
        )
        all_cases.extend(batch_cases)
        results.append(result)
        print(
            json.dumps(
                {
                    "batch": batch_index,
                    "of": api_batch_count,
                    "source": source,
                    "cases": len(batch_cases),
                    "latency_ms": round(result.latency_ms, 1),
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                    "total_tokens": result.total_tokens,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    if len(all_cases) != 500:
        raise AssertionError(f"expected 500 generated cases, received {len(all_cases)}")
    normalized_texts = [case.query_text.strip().casefold() for case in all_cases]
    if len(set(normalized_texts)) != 500:
        raise ValueError("generated query texts must be globally unique")
    payload = benchmark_payload(all_cases)
    write_text_artifact(args.output, payload.decode("utf-8"))

    split_counts: dict[str, int] = {}
    language_counts: dict[str, int] = {}
    difficulty_by_frame = {frame.frame_id: frame.difficulty for frame in frames}
    difficulty_counts: dict[str, int] = {}
    for case in all_cases:
        split_counts[case.split] = split_counts.get(case.split, 0) + 1
        language_counts[case.language] = language_counts.get(case.language, 0) + 1
        difficulty = difficulty_by_frame[case.frame_family]
        difficulty_counts[difficulty] = difficulty_counts.get(difficulty, 0) + 1
    summary = QueryBenchmarkGenerationSummary(
        generator_model=environment.model,
        case_count=len(all_cases),
        semantic_frame_count=len(frames),
        split_counts=dict(sorted(split_counts.items())),
        language_counts=dict(sorted(language_counts.items())),
        difficulty_counts=dict(sorted(difficulty_counts.items())),
        api_batch_count=api_batch_count,
        api_attempt_count=sum(result.attempt_count for result in results),
        input_tokens=_sum_optional([result.input_tokens for result in results]),
        output_tokens=_sum_optional([result.output_tokens for result in results]),
        total_tokens=_sum_optional([result.total_tokens for result in results]),
        total_latency_ms=sum(result.latency_ms for result in results),
        dataset_sha256=hashlib.sha256(payload).hexdigest(),
    )
    write_json_artifact(args.manifest, summary)
    print(summary.model_dump_json(indent=2), flush=True)


if __name__ == "__main__":
    main()
