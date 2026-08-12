"""Build the 500-case behavior-anchored Query Recommendation Benchmark V1."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Sequence

from dotenv import load_dotenv

from yelp_agent.agent.llm import LLMCallResult, LLMMessage, OpenAICompatibleLLM
from yelp_agent.business_profiles import BusinessKnowledgeStore
from yelp_agent.config import (
    AgentConfig,
    load_business_profile_config,
    load_config,
    load_llm_environment,
)
from yelp_agent.data.temporal_view import TemporalDataView
from yelp_agent.experiments import write_json_artifact
from yelp_agent.query_recommendation_benchmark import (
    QueryGenerationReport,
    QueryRecommendationSources,
    YelpAnchorKnowledgeReader,
    assemble_query_recommendation_bundle,
    audit_query_recommendation_benchmark,
    build_query_audit_prompt,
    build_query_rewrite_prompt,
    combine_generation_and_audit,
    load_behavior_anchor_candidates,
    load_query_recommendation_build_config,
    parse_fidelity_audits,
    parse_generated_queries,
    plan_query_recommendation_frames,
    publish_query_recommendation_bundle,
    select_behavior_anchors,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path.cwd(),
        help="project root containing the frozen Yelp data artifacts",
    )
    parser.add_argument(
        "--config-root",
        type=Path,
        default=Path.cwd(),
        help="project root containing query benchmark construction config",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("benchmarks/query_recommendation_v1"),
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("runs/query_recommendation_v1/generation_cache"),
    )
    parser.add_argument(
        "--allow-real-api",
        action="store_true",
        help="required before Query rendering and fidelity-audit provider calls",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="validate the 500 real anchors and frames without calling a provider",
    )
    return parser


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_cached_call(path: Path) -> LLMCallResult | None:
    if not path.is_file():
        return None
    result = LLMCallResult.model_validate_json(path.read_text(encoding="utf-8"))
    return result if result.status == "success" and result.content else None


def _call(
    llm: OpenAICompatibleLLM,
    *,
    system: str,
    user: str,
    prompt_hash: str,
    cache_root: Path,
    label: str,
) -> tuple[LLMCallResult, str]:
    cache_path = cache_root / f"{label}-{prompt_hash[:16]}.json"
    cached = _load_cached_call(cache_path)
    if cached is not None:
        return cached, "cache"
    result = llm.generate(
        (
            LLMMessage(role="system", content=system),
            LLMMessage(role="user", content=user),
        )
    )
    write_json_artifact(cache_path, result)
    if result.status != "success" or result.content is None:
        raise RuntimeError(
            f"{label} failed: {result.failure_reason or result.status}"
        )
    return result, "api"


def _sum_optional(values: list[int | None]) -> int | None:
    return sum(value for value in values if value is not None) if all(
        value is not None for value in values
    ) else None


def _cached_call_results(cache_root: Path) -> list[LLMCallResult]:
    return [
        LLMCallResult.model_validate_json(path.read_text(encoding="utf-8"))
        for path in sorted(cache_root.glob("*.json"))
    ]


def _duplicate_rewrite_case_ids(rewrites) -> list[str]:
    seen: dict[str, str] = {}
    duplicates = []
    for item in rewrites:
        normalized = " ".join(item.query_text.casefold().split())
        if normalized in seen:
            duplicates.append(item.case_id)
        else:
            seen[normalized] = item.case_id
    return duplicates


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source_root = args.source_root.resolve()
    config_root = args.config_root.resolve()
    config_path = config_root / "configs" / "query_recommendation_benchmark.yaml"
    build_config = load_query_recommendation_build_config(config_path)
    sources = QueryRecommendationSources.from_project_root(source_root)
    candidates = load_behavior_anchor_candidates(sources)
    anchors = select_behavior_anchors(candidates, build_config.selection)
    data_view = TemporalDataView(
        source_root / "data" / "processed" / "businesses.parquet",
        source_root / "data" / "processed" / "reviews.parquet",
        source_root / "data" / "processed" / "interactions.parquet",
    )
    business_knowledge = BusinessKnowledgeStore.from_artifacts(
        source_root / "data" / "features" / "business_profiles" / "v1",
        config=load_business_profile_config(source_root / "configs"),
    )
    reader = YelpAnchorKnowledgeReader(
        data_view,
        business_knowledge,
        broad_categories=set(load_config(source_root / "configs").data.broad_categories),
    )
    drafts = plan_query_recommendation_frames(
        anchors,
        reader,
        build_config.frames,
    )
    plan_summary = {
        "candidate_count": len(candidates),
        "selected_anchor_count": len(anchors),
        "case_count": len(drafts),
        "split_counts": dict(Counter(item.anchor.benchmark_split for item in drafts)),
        "family_counts": dict(Counter(item.frame.frame_family for item in drafts)),
        "language_counts": dict(Counter(item.language for item in drafts)),
        "aspect_source_scope": business_knowledge.source_scope,
        "target_review_text_loaded": False,
    }
    print(json.dumps({"stage": "plan", **plan_summary}, ensure_ascii=False), flush=True)
    if args.plan_only:
        return 0
    if not args.allow_real_api:
        raise SystemExit("Refusing real provider calls without --allow-real-api")

    load_dotenv(source_root / ".env", override=False)
    environment = load_llm_environment()
    if not environment.llm_enabled or environment.model is None:
        raise SystemExit("OPENAI_API_KEY and OPENAI_MODEL are not configured")
    generation_config = build_config.generation
    llm = OpenAICompatibleLLM(
        AgentConfig(
            enabled=True,
            temperature=generation_config.temperature,
            timeout_seconds=generation_config.timeout_seconds,
            max_retries=generation_config.max_retries,
            max_tokens=generation_config.max_output_tokens,
            response_format_json=True,
            thinking=generation_config.thinking,
        ),
        environment,
    )
    args.cache_root.mkdir(parents=True, exist_ok=True)
    rewrites = []
    call_results: list[LLMCallResult] = []
    generation_hashes: list[str] = []
    audit_hashes: list[str] = []
    provider_calls_this_run = 0
    batches = [
        drafts[offset : offset + generation_config.batch_size]
        for offset in range(0, len(drafts), generation_config.batch_size)
    ]
    for batch_index, batch in enumerate(batches, start=1):
        failures: dict[str, list[str]] = {}
        for generation_attempt in range(
            1, generation_config.maximum_generation_attempts + 1
        ):
            system, user, generation_hash = build_query_rewrite_prompt(batch)
            if failures:
                user += "\nRegenerate these cases after prior audit failures: " + json.dumps(
                    failures,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                generation_hash = hashlib.sha256(
                    (system + "\n" + user).encode()
                ).hexdigest()
            for draft in batch:
                forbidden = (
                    draft.anchor.target_business_id,
                    draft.anchor.target_review_id,
                    draft.anchor.user_id,
                )
                if any(value in system + user for value in forbidden):
                    raise AssertionError("provider prompt contains hidden target identity")
            generation_result, generation_source = _call(
                llm,
                system=system,
                user=user,
                prompt_hash=generation_hash,
                cache_root=args.cache_root,
                label=f"batch-{batch_index:02d}-generation-{generation_attempt}",
            )
            provider_calls_this_run += int(generation_source == "api")
            call_results.append(generation_result)
            generation_hashes.append(generation_hash)
            try:
                generated = parse_generated_queries(
                    generation_result.content or "", batch
                )
            except (ValueError, TypeError):
                retry_user = (
                    user
                    + "\nSTRICT_SCHEMA_RETRY: Return valid JSON with exactly the "
                    + "top-level key queries; every item must contain only case_id "
                    + "and query_text."
                )
                retry_hash = hashlib.sha256(
                    (system + "\n" + retry_user).encode()
                ).hexdigest()
                generation_result, generation_source = _call(
                    llm,
                    system=system,
                    user=retry_user,
                    prompt_hash=retry_hash,
                    cache_root=args.cache_root,
                    label=(
                        f"batch-{batch_index:02d}-generation-"
                        f"{generation_attempt}-schema-retry"
                    ),
                )
                provider_calls_this_run += int(generation_source == "api")
                call_results.append(generation_result)
                generation_hashes.append(retry_hash)
                generated = parse_generated_queries(
                    generation_result.content or "", batch
                )
            audit_system, audit_user, audit_hash = build_query_audit_prompt(
                batch, generated
            )
            audit_result, audit_source = _call(
                llm,
                system=audit_system,
                user=audit_user,
                prompt_hash=audit_hash,
                cache_root=args.cache_root,
                label=f"batch-{batch_index:02d}-audit-{generation_attempt}",
            )
            provider_calls_this_run += int(audit_source == "api")
            call_results.append(audit_result)
            audit_hashes.append(audit_hash)
            try:
                audited = parse_fidelity_audits(audit_result.content or "", batch)
            except (ValueError, TypeError):
                retry_audit_user = (
                    audit_user
                    + "\nSTRICT_SCHEMA_RETRY: Return valid JSON with exactly the "
                    + "top-level key audits; every item must contain only case_id, "
                    + "passed, and reason_codes. Spell reason_codes exactly."
                )
                retry_audit_hash = hashlib.sha256(
                    (audit_system + "\n" + retry_audit_user).encode()
                ).hexdigest()
                audit_result, audit_source = _call(
                    llm,
                    system=audit_system,
                    user=retry_audit_user,
                    prompt_hash=retry_audit_hash,
                    cache_root=args.cache_root,
                    label=(
                        f"batch-{batch_index:02d}-audit-"
                        f"{generation_attempt}-schema-retry"
                    ),
                )
                provider_calls_this_run += int(audit_source == "api")
                call_results.append(audit_result)
                audit_hashes.append(retry_audit_hash)
                audited = parse_fidelity_audits(
                    audit_result.content or "", batch
                )
            combined = combine_generation_and_audit(batch, generated, audited)
            failures = {
                item.case_id: item.audit_reason_codes
                for item in combined
                if not item.condition_fidelity_passed
            }
            print(
                json.dumps(
                    {
                        "stage": "generation",
                        "batch": batch_index,
                        "of": len(batches),
                        "attempt": generation_attempt,
                        "generation_source": generation_source,
                        "audit_source": audit_source,
                        "failed_audits": len(failures),
                        "generation_latency_ms": generation_result.latency_ms,
                        "audit_latency_ms": audit_result.latency_ms,
                        "generation_tokens": generation_result.total_tokens,
                        "audit_tokens": audit_result.total_tokens,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if not failures:
                rewrites.extend(combined)
                break
        else:
            raise RuntimeError(
                f"batch {batch_index} still failed fidelity audit: {failures}"
            )
    duplicate_ids = _duplicate_rewrite_case_ids(rewrites)
    for uniqueness_attempt in range(1, 4):
        if not duplicate_ids:
            break
        duplicate_set = set(duplicate_ids)
        duplicate_drafts = [
            draft for draft in drafts if draft.frame.case_id in duplicate_set
        ]
        uniqueness_system, uniqueness_user, _ = build_query_rewrite_prompt(
            duplicate_drafts
        )
        uniqueness_user += (
            "\nUNIQUENESS_REPAIR: Preserve every canonical condition exactly, but "
            "rewrite each case with a noticeably different natural sentence "
            "structure. Vary openings and word order across this repair batch. "
            "Do not add any new requirement."
        )
        uniqueness_hash = hashlib.sha256(
            (uniqueness_system + "\n" + uniqueness_user).encode()
        ).hexdigest()
        uniqueness_result, uniqueness_source = _call(
            llm,
            system=uniqueness_system,
            user=uniqueness_user,
            prompt_hash=uniqueness_hash,
            cache_root=args.cache_root,
            label=f"uniqueness-repair-{uniqueness_attempt}",
        )
        provider_calls_this_run += int(uniqueness_source == "api")
        call_results.append(uniqueness_result)
        generation_hashes.append(uniqueness_hash)
        unique_generated = parse_generated_queries(
            uniqueness_result.content or "", duplicate_drafts
        )
        audit_system, audit_user, audit_hash = build_query_audit_prompt(
            duplicate_drafts,
            unique_generated,
        )
        audit_result, audit_source = _call(
            llm,
            system=audit_system,
            user=audit_user,
            prompt_hash=audit_hash,
            cache_root=args.cache_root,
            label=f"uniqueness-repair-audit-{uniqueness_attempt}",
        )
        provider_calls_this_run += int(audit_source == "api")
        call_results.append(audit_result)
        audit_hashes.append(audit_hash)
        unique_audits = parse_fidelity_audits(
            audit_result.content or "", duplicate_drafts
        )
        replacements = combine_generation_and_audit(
            duplicate_drafts,
            unique_generated,
            unique_audits,
        )
        failed = [
            item.case_id
            for item in replacements
            if not item.condition_fidelity_passed
        ]
        if failed:
            raise RuntimeError(
                f"uniqueness repair failed condition audit: {failed}"
            )
        replacements_by_id = {item.case_id: item for item in replacements}
        rewrites = [
            replacements_by_id.get(item.case_id, item) for item in rewrites
        ]
        duplicate_ids = _duplicate_rewrite_case_ids(rewrites)
        print(
            json.dumps(
                {
                    "stage": "uniqueness_repair",
                    "attempt": uniqueness_attempt,
                    "rewritten_case_count": len(replacements),
                    "remaining_duplicate_count": len(duplicate_ids),
                    "generation_source": uniqueness_source,
                    "audit_source": audit_source,
                    "generation_tokens": uniqueness_result.total_tokens,
                    "audit_tokens": audit_result.total_tokens,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    if duplicate_ids:
        raise RuntimeError(
            f"global Query uniqueness repair failed: {duplicate_ids}"
        )
    recipe_hash = hashlib.sha256(
        "\n".join(generation_hashes + audit_hashes).encode()
    ).hexdigest()
    bundle = assemble_query_recommendation_bundle(
        drafts,
        tuple(rewrites),
        generator_model=environment.model,
        generator_prompt_sha256=recipe_hash,
    )
    audit = audit_query_recommendation_benchmark(bundle, drafts, reader)
    if not audit.passed:
        raise RuntimeError(f"benchmark leakage audit failed: {audit.violations[:5]}")
    cumulative_calls = _cached_call_results(args.cache_root)
    generation_report = QueryGenerationReport(
        generator_model=environment.model,
        case_count=len(bundle.visible_cases),
        batch_count=len(batches),
        provider_call_count=len(call_results),
        attempt_count=sum(item.attempt_count for item in call_results),
        input_tokens=_sum_optional([item.input_tokens for item in call_results]),
        output_tokens=_sum_optional([item.output_tokens for item in call_results]),
        total_tokens=_sum_optional([item.total_tokens for item in call_results]),
        total_latency_ms=sum(item.latency_ms for item in call_results),
        cumulative_cache_call_count=len(cumulative_calls),
        cumulative_cache_total_tokens=_sum_optional(
            [item.total_tokens for item in cumulative_calls]
        ),
        cumulative_cache_latency_ms=sum(item.latency_ms for item in cumulative_calls),
        generation_prompt_sha256=generation_hashes,
        audit_prompt_sha256=audit_hashes,
    )
    source_hashes = {
        f"source_{index:02d}": _sha256_file(path)
        for index, path in enumerate(sources.paths(), start=1)
    }
    source_hashes["businesses"] = _sha256_file(
        source_root / "data" / "processed" / "businesses.parquet"
    )
    source_hashes["business_profile_manifest"] = _sha256_file(
        source_root / "data" / "features" / "business_profiles" / "v1" / "manifest.json"
    )
    result = publish_query_recommendation_bundle(
        bundle,
        args.output_root,
        audit=audit,
        generation_report=generation_report,
        source_sha256=source_hashes,
        configuration_sha256=_sha256_file(config_path),
    )
    print(
        json.dumps(
            {
                "stage": "complete",
                **plan_summary,
                "generator_model": environment.model,
                "provider_calls_this_run": provider_calls_this_run,
                "input_tokens": generation_report.input_tokens,
                "output_tokens": generation_report.output_tokens,
                "total_tokens": generation_report.total_tokens,
                "cumulative_cache_total_tokens": (
                    generation_report.cumulative_cache_total_tokens
                ),
                "output_root": str(result.root.resolve()),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
