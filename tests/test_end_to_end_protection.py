from __future__ import annotations

import json

import pandas as pd

from synthetic_pipeline import run_synthetic_pipeline


def test_synthetic_data_reaches_hybrid_agent_and_evaluation(tmp_path) -> None:
    run = run_synthetic_pipeline(tmp_path / "run")

    assert run.validation_task_count == 1
    assert run.test_task_count == 1
    assert len(run.hybrid_ranking) == 20
    assert len(set(run.hybrid_ranking)) == 20
    assert len(run.agent_ranking) == 20
    assert set(run.agent_ranking) == set(run.hybrid_ranking)
    assert run.agent_ranking[:8] == list(reversed(run.hybrid_ranking[:8]))
    assert run.agent_ranking[8:] == run.hybrid_ranking[8:]
    assert run.agent_fallback is False
    assert run.agent_tool_calls == 4
    assert run.agent_evaluation_task_count == 1
    assert run.agent_valid_output_rate == 1.0
    assert run.fake_llm_call_count == 1


def test_same_input_repeats_frozen_artifacts_byte_for_byte(tmp_path) -> None:
    first = run_synthetic_pipeline(tmp_path / "first")
    second = run_synthetic_pipeline(tmp_path / "second")

    stable_artifacts = (
        "task_dataset/tasks/validation_tasks.jsonl",
        "task_dataset/tasks/test_tasks.jsonl",
        "features/tfidf_manifest.json",
        "features/tfidf_vectorizer.joblib",
        "hybrid/weights.json",
    )
    for relative_path in stable_artifacts:
        assert (first.root / relative_path).read_bytes() == (
            second.root / relative_path
        ).read_bytes()
    assert first.hybrid_ranking == second.hybrid_ranking
    assert first.agent_ranking == second.agent_ranking


def test_review_jsonl_line_order_does_not_change_tasks_or_rankings(tmp_path) -> None:
    original = run_synthetic_pipeline(tmp_path / "original")
    shuffled = run_synthetic_pipeline(
        tmp_path / "shuffled",
        reverse_review_rows=True,
    )

    for relative_path in (
        "task_dataset/tasks/validation_tasks.jsonl",
        "task_dataset/tasks/test_tasks.jsonl",
    ):
        assert (original.root / relative_path).read_bytes() == (
            shuffled.root / relative_path
        ).read_bytes()
    assert original.hybrid_ranking == shuffled.hybrid_ranking
    assert original.agent_ranking == shuffled.agent_ranking


def test_future_reviews_cannot_change_point_in_time_features_or_rankings(
    tmp_path,
) -> None:
    original = run_synthetic_pipeline(tmp_path / "original")
    with_future = run_synthetic_pipeline(
        tmp_path / "with-future",
        include_future_reviews=True,
    )

    for relative_path in (
        "task_dataset/tasks/validation_tasks.jsonl",
        "task_dataset/tasks/test_tasks.jsonl",
        "features/tfidf_manifest.json",
        "features/tfidf_vectorizer.joblib",
    ):
        assert (original.root / relative_path).read_bytes() == (
            with_future.root / relative_path
        ).read_bytes()
    assert original.hybrid_score_snapshot == with_future.hybrid_score_snapshot
    assert original.hybrid_ranking == with_future.hybrid_ranking
    assert original.agent_ranking == with_future.agent_ranking


def test_fake_llm_failures_always_fall_back_to_valid_hybrid(tmp_path) -> None:
    expected_reasons = {
        "non_json": "non_json",
        "duplicate": "duplicate_id",
        "missing": "missing_id",
        "unknown": "unknown_id",
        "empty": "empty_response",
        "exception": "llm_error",
        "timeout": "timeout",
        "disabled": "llm_disabled",
    }
    run = run_synthetic_pipeline(
        tmp_path / "failures",
        llm_behaviors=tuple(expected_reasons),
    )

    for behavior, expected_reason in expected_reasons.items():
        outcome = run.agent_outcomes[behavior]
        assert outcome.ranking == run.hybrid_ranking
        assert outcome.fallback is True
        assert outcome.fallback_reason == expected_reason
        assert outcome.valid_output_rate == 1.0
        assert outcome.task_count == 1
    assert run.agent_outcomes["disabled"].llm_failure_rate == 0.0
    assert run.agent_outcomes["timeout"].llm_failure_rate == 1.0


def test_agent_context_has_no_truth_label_secret_or_real_api_access(
    tmp_path,
    monkeypatch,
) -> None:
    secret_canary = "must-not-appear-in-agent-artifacts"
    monkeypatch.setenv("OPENAI_API_KEY", secret_canary)

    def reject_real_client(*args, **kwargs):
        raise AssertionError("a real OpenAI-compatible client was constructed")

    monkeypatch.setattr("yelp_agent.agent.llm.OpenAI", reject_real_client)
    run = run_synthetic_pipeline(tmp_path / "isolated")

    prompt_payload = run.agent_prompt_payloads["reverse"]
    serialized_prompt = json.dumps(prompt_payload, sort_keys=True).lower()
    traces = (run.root / "agent" / "traces.jsonl").read_text(encoding="utf-8")
    public_tasks = (
        run.root / "task_dataset" / "tasks" / "test_tasks.jsonl"
    ).read_text(encoding="utf-8")
    truth = pd.read_parquet(
        run.root
        / "task_dataset"
        / "ground_truth"
        / "candidate_ground_truth.parquet"
    )
    target_business_id = str(truth.iloc[0]["target_business_id"])
    prompt_candidate_ids = {
        candidate["business_id"] for candidate in prompt_payload["candidates"]
    }

    assert target_business_id in prompt_candidate_ids
    assert "ground_truth" not in serialized_prompt
    assert "target_business_id" not in serialized_prompt
    assert "target_business_id" not in public_tasks
    assert secret_canary not in serialized_prompt
    assert secret_canary not in traces
