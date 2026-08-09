import hashlib
import json
from collections import Counter
from pathlib import Path

from yelp_agent.query.benchmark import load_query_benchmark
from yelp_agent.query.generation import (
    QueryBenchmarkGenerationSummary,
    build_generation_messages_payload,
    build_query_frame_specs,
    expand_generated_batch,
    parse_generated_batch,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_frame_plan_has_100_frames_and_expected_distribution() -> None:
    frames = build_query_frame_specs()

    assert len(frames) == 100
    assert len({frame.frame_id for frame in frames}) == 100
    signatures = {
        (
            tuple(
                (
                    condition.field,
                    condition.operator,
                    str(condition.value),
                    condition.importance,
                    condition.enforcement,
                )
                for condition in frame.expected_conditions
            ),
            frame.expected_party_size,
        )
        for frame in frames
    }
    assert len(signatures) == 100
    assert sum(frame.split == "development" for frame in frames) == 80
    assert sum(frame.split == "validation" for frame in frames) == 20
    assert sum(frame.difficulty == "simple" for frame in frames) == 20
    assert sum(frame.difficulty == "multi_constraint" for frame in frames) == 40
    assert sum(frame.difficulty == "negation_priority" for frame in frames) == 20
    assert sum(frame.difficulty == "clarification" for frame in frames) == 20


def test_provider_paraphrases_expand_to_cases_with_local_gold_labels() -> None:
    frames = build_query_frame_specs()[:1]
    system, user, prompt_hash = build_generation_messages_payload(frames)
    payload = {
        "frames": [
            {
                "frame_id": frames[0].frame_id,
                "paraphrases": [
                    {"language": "zh-CN", "style": "direct_zh", "text": "我想吃牛排。"},
                    {
                        "language": "zh-CN",
                        "style": "colloquial_zh",
                        "text": "今天整点牛排吧。",
                    },
                    {
                        "language": "zh-CN",
                        "style": "implicit_zh",
                        "text": "今晚想找个做牛排的地方。",
                    },
                    {
                        "language": "en-US",
                        "style": "natural_en",
                        "text": "I would like a steakhouse.",
                    },
                    {
                        "language": "en-US",
                        "style": "conversational_en",
                        "text": "How about somewhere for steak tonight?",
                    },
                ],
            }
        ]
    }

    generated = parse_generated_batch(json.dumps(payload), frames)
    cases = expand_generated_batch(
        generated,
        frames,
        model="test-model",
        prompt_sha256=prompt_hash,
    )

    assert system
    assert user
    assert '"enforcement"' not in user
    assert '"missing_fields"' not in user
    assert '"scenario"' not in user
    assert len(cases) == 5
    assert sum(case.language == "zh-CN" for case in cases) == 3
    assert sum(case.language == "en-US" for case in cases) == 2
    assert all(case.expected_conditions == frames[0].expected_conditions for case in cases)
    assert all(case.generator_model == "test-model" for case in cases)


def test_frozen_v2_benchmark_contains_500_auditable_queries() -> None:
    benchmark_path = PROJECT_ROOT / "benchmarks/query_aware_v2/queries_500.jsonl"
    manifest_path = PROJECT_ROOT / "benchmarks/query_aware_v2/manifest.json"

    cases = load_query_benchmark(benchmark_path)
    manifest = QueryBenchmarkGenerationSummary.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    frame_counts = Counter(case.frame_family for case in cases)

    assert len(cases) == 500
    assert len(frame_counts) == 100
    assert set(frame_counts.values()) == {5}
    assert Counter(case.split for case in cases) == {
        "development": 400,
        "validation": 100,
    }
    assert Counter(case.language for case in cases) == {
        "zh-CN": 300,
        "en-US": 200,
    }
    assert all(case.generator_kind == "openai_compatible" for case in cases)
    assert all(case.generator_model == "deepseek-v4-flash" for case in cases)
    assert manifest.case_count == 500
    assert manifest.dataset_sha256 == hashlib.sha256(
        benchmark_path.read_bytes()
    ).hexdigest()
