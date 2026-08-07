"""Run a cached, anonymous DeepSeek audit over stratified rule outputs."""

from __future__ import annotations

import argparse
from itertools import zip_longest
from pathlib import Path

from yelp_agent.agent.llm import OpenAICompatibleLLM
from yelp_agent.config import (
    AgentConfig,
    LLMEnvironment,
    load_llm_environment,
    load_review_aspect_settings,
)
from yelp_agent.experiments import write_json_artifact, write_jsonl_artifact
from yelp_agent.reviews.audit import (
    ReviewAspectAuditor,
    run_review_aspect_audit,
    sample_aspect_audit_items,
    sample_unmatched_aspect_discovery_items,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--records",
        type=Path,
        default=Path(
            "data/features/review_aspects/development/aspect_records.parquet"
        ),
    )
    parser.add_argument(
        "--reviews",
        type=Path,
        default=Path("data/processed/interactions.parquet"),
    )
    parser.add_argument("--config-dir", type=Path, default=Path("configs"))
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("runs/review_aspect_v1/api_cache"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("runs/review_aspect_v1/audit_summary.json"),
    )
    parser.add_argument(
        "--traces",
        type=Path,
        default=Path("runs/review_aspect_v1/audit_traces.jsonl"),
    )
    parser.add_argument(
        "--batches",
        type=Path,
        default=Path("runs/review_aspect_v1/audit_batches.jsonl"),
    )
    parser.add_argument("--sample-size", type=int)
    parser.add_argument(
        "--allow-real-api",
        action="store_true",
        help="allow the configured OpenAI-compatible provider to be called",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config, _ = load_review_aspect_settings(args.config_dir)
    sample_size = args.sample_size or config.audit.sample_size
    matched_size = (sample_size + 1) // 2
    unmatched_size = sample_size - matched_size
    matched = sample_aspect_audit_items(
        args.records,
        sample_size=matched_size,
        seed=config.audit.seed,
    )
    unmatched = sample_unmatched_aspect_discovery_items(
        args.reviews,
        args.records,
        sample_size=unmatched_size,
        seed=config.audit.seed,
    )
    items = tuple(
        item
        for pair in zip_longest(matched, unmatched)
        for item in pair
        if item is not None
    )
    environment = load_llm_environment()
    if not args.allow_real_api:
        environment = LLMEnvironment(
            api_key=None,
            base_url=environment.base_url,
            model=environment.model,
        )
    llm = OpenAICompatibleLLM(
        AgentConfig(
            enabled=config.audit.enabled,
            temperature=config.audit.temperature,
            timeout_seconds=config.audit.timeout_seconds,
            max_retries=config.audit.max_retries,
            max_tokens=config.audit.max_tokens,
            response_format_json=config.audit.response_format_json,
            thinking=config.audit.thinking,
        ),
        environment,
    )
    auditor = ReviewAspectAuditor(
        llm,
        model_name=environment.model,
        cache_dir=args.cache_dir,
        request_profile=config.audit.model_dump_json(),
    )
    execution = run_review_aspect_audit(
        items,
        auditor,
        batch_size=config.audit.batch_size,
    )
    write_json_artifact(args.report, execution.summary)
    write_jsonl_artifact(args.traces, execution.traces)
    write_jsonl_artifact(args.batches, execution.batches)
    print(execution.summary.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
