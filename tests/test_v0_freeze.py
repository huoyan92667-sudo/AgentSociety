from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
FREEZE_DIR = ROOT / "docs" / "experiments" / "v0"


def _load_manifest() -> dict[str, object]:
    return json.loads((FREEZE_DIR / "manifest.json").read_text(encoding="utf-8"))


def test_v0_freeze_has_required_protocol_and_no_task_ids() -> None:
    manifest = _load_manifest()
    policy = manifest["evaluation_policy"]
    selection = manifest["task_selection"]

    assert policy["legacy_test_status"] == "previously_observed"
    assert policy["strict_final_blind_holdout"] is False
    assert policy["development_policy"] == "validation_and_user_level_cv"
    assert policy["test_usage"] == "historical_comparison_only"
    assert selection["selected_task_count"] == 20
    assert selection["unique_task_count"] == 20
    assert selection["matches_prediction_order"] is True
    assert selection["task_ids_stored_in_manifest"] is False

    serialized = json.dumps(manifest, ensure_ascii=False).lower()
    assert "test:" not in serialized
    assert ".env" not in serialized
    assert "bearer " not in serialized
    assert "authorization" not in serialized

    privacy = manifest["privacy"]
    assert not any(privacy.values())
    assert manifest["agent"]["secret_values_recorded"] is False
    assert manifest["agent"]["authentication_headers_recorded"] is False
    assert manifest["agent"]["environment_variable_names"] == [
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_MODEL",
    ]


def test_v0_metrics_are_internally_consistent() -> None:
    metrics = _load_manifest()["metrics"]
    hybrid = metrics["hybrid"]
    agent = metrics["hybrid_agent"]
    delta = metrics["agent_minus_hybrid"]
    changes = metrics["rank_changes"]
    runtime = metrics["runtime"]

    for name in ("hr_at_1", "hr_at_3", "hr_at_5", "avg_hr", "mrr", "ndcg_at_5"):
        assert delta[name] == pytest.approx(agent[name] - hybrid[name])
    assert agent["avg_hr"] == pytest.approx(
        (agent["hr_at_1"] + agent["hr_at_3"] + agent["hr_at_5"]) / 3
    )
    assert hybrid["avg_hr"] == pytest.approx(
        (hybrid["hr_at_1"] + hybrid["hr_at_3"] + hybrid["hr_at_5"]) / 3
    )
    assert (
        changes["improved_count"]
        + changes["unchanged_count"]
        + changes["worsened_count"]
        == agent["task_count"]
    )
    assert runtime["observed_input_tokens"] + runtime["observed_output_tokens"] == runtime["observed_total_tokens"]
    assert agent["mean_llm_tokens"] == pytest.approx(
        runtime["observed_total_tokens"] / agent["task_count"]
    )


def test_v0_fingerprints_and_environment_snapshot_are_valid() -> None:
    manifest = _load_manifest()
    fingerprints = manifest["fingerprints"]
    sha256_pattern = re.compile(r"[0-9a-f]{64}")

    for group_name in (
        "source_and_configuration",
        "feature_artifacts",
        "experiment_artifacts",
        "tracked_freeze_documents",
    ):
        for digest in fingerprints[group_name].values():
            assert sha256_pattern.fullmatch(digest)

    artifact_paths = [
        *fingerprints["feature_artifacts"],
        *fingerprints["experiment_artifacts"],
    ]
    assert not any("ground_truth" in path for path in artifact_paths)

    environment_path = FREEZE_DIR / "environment.txt"
    environment_bytes = environment_path.read_bytes()
    expected_file_hash = fingerprints["tracked_freeze_documents"][
        "docs/experiments/v0/environment.txt"
    ]
    assert hashlib.sha256(environment_bytes).hexdigest() == expected_file_hash

    environment_text = environment_bytes.decode("utf-8")
    package_lines = environment_text.split("[packages]\n", maxsplit=1)[1].splitlines()
    canonical_packages = "".join(f"{line}\n" for line in package_lines if line)
    expected_packages_hash = re.search(
        r"dependency_snapshot_sha256=([0-9a-f]{64})", environment_text
    )
    assert expected_packages_hash is not None
    assert hashlib.sha256(canonical_packages.encode("utf-8")).hexdigest() == expected_packages_hash.group(1)


def test_v0_summary_exposes_negative_result_and_provenance_limit() -> None:
    summary = (FREEZE_DIR / "summary.md").read_text(encoding="utf-8")

    assert "Agent - Hybrid" in summary
    assert "-0.116667" in summary
    assert "352,849" in summary
    assert "Legacy Test V0" in summary
    assert "还不是能够自主观察" in summary
    assert "无法证明实验执行时的精确源码提交" in summary
