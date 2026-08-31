from __future__ import annotations

import json
from pathlib import Path

import pytest

from yelp_agent.recommendation_v2.rag_benchmark.schema import ClaudeWorkerTrace
from yelp_agent.recommendation_v2.review_evidence.teacher_labeling.audit import (
    build_teacher_audit,
)
from yelp_agent.recommendation_v2.review_evidence.teacher_labeling.candidate_sampler import (
    _fill_bucket_by_semantic_distance,
    _select_bucket,
)
from yelp_agent.recommendation_v2.review_evidence.teacher_labeling.contracts import (
    TeacherLabel,
)
from yelp_agent.recommendation_v2.review_evidence.teacher_labeling.teacher_runner import (
    TeacherRunConfig,
    run_teacher_labeling,
)
from yelp_agent.recommendation_v2.review_evidence.teacher_labeling.reuse_labels import (
    reuse_existing_labels,
)
from yelp_agent.recommendation_v2.schema import ASPECT_FIELDS


class _FakeTeacherWorker:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None]] = []

    def generate(
        self,
        prompt: str,
        output_model: type[TeacherLabel],
        *,
        system_prompt: str | None = None,
    ) -> tuple[TeacherLabel, ClaudeWorkerTrace, dict[str, object]]:
        self.calls.append((prompt, system_prompt))
        label = output_model(relevance=3, strength=4)
        trace = ClaudeWorkerTrace(
            model="fake-glm",
            duration_ms=10,
            input_tokens=100,
            output_tokens=5,
        )
        wrapper: dict[str, object] = {
            "result": label.model_dump(mode="json"),
            "usage": {"input_tokens": 100, "output_tokens": 5},
        }
        return label, trace, wrapper


def test_teacher_label_rejects_invalid_null_relationship() -> None:
    with pytest.raises(ValueError):
        TeacherLabel(relevance=0, strength=2)
    with pytest.raises(ValueError):
        TeacherLabel(relevance=2, strength=None)


def test_candidate_bucket_second_pass_never_reuses_a_review() -> None:
    rows = [
        {
            "review_id": f"review-{index}",
            "business_id": "same-business",
            "selection_strength": 4,
            "semantic_score": 0.9 - index / 100,
            "useful": 0,
            "review_text": f"review {index}",
        }
        for index in range(5)
    ]

    selected = _select_bucket(
        rows,
        aspect="food_quality",
        bucket=4,
        count=5,
        used=set(),
    )

    assert len(selected) == 5
    assert len({item["review_id"] for item in selected}) == 5


def test_semantic_fallback_second_pass_never_reuses_a_review() -> None:
    rows = [
        {
            "review_id": f"review-{index}",
            "business_id": "same-business",
            "selection_strength": 2,
            "semantic_score": 0.5,
            "useful": 0,
            "review_text": f"review {index}",
        }
        for index in range(5)
    ]

    selected = _fill_bucket_by_semantic_distance(
        rows,
        bucket=3,
        count=5,
        used=set(),
    )

    assert len(selected) == 5
    assert len({item["review_id"] for item in selected}) == 5


def test_runner_sends_only_model_input_and_resumes(tmp_path: Path) -> None:
    dataset_root, template_path = _write_dataset(tmp_path)
    worker = _FakeTeacherWorker()
    config = TeacherRunConfig(
        dataset_root=dataset_root,
        template_path=template_path,
        run_id="test_run",
        model="fake-glm",
        max_workers=1,
        max_attempts=1,
        limit_per_aspect=1,
    )

    first = run_teacher_labeling(config, worker=worker)

    assert first["success_this_invocation"] == 14
    assert len(worker.calls) == 14
    for prompt, system_prompt in worker.calls:
        payload = json.loads(prompt)
        assert set(payload) == {
            "aspect_id",
            "definition",
            "relevance_scale",
            "strength_scale",
            "special_rules",
            "review_text",
        }
        assert "sample_id" not in payload
        assert "selection_strength" not in payload
        assert system_prompt == "固定教师要求"

    second = run_teacher_labeling(config, worker=worker)

    assert second["pending_at_invocation_start"] == 0
    assert second["success_this_invocation"] == 0
    assert len(worker.calls) == 14
    labeled_lines = sum(
        len(path.read_text(encoding="utf-8").splitlines())
        for path in (dataset_root / "labeled").glob("*.jsonl")
    )
    assert labeled_lines == 14


def test_audit_reports_selection_teacher_disagreements(tmp_path: Path) -> None:
    dataset_root, template_path = _write_dataset(tmp_path)
    run_teacher_labeling(
        TeacherRunConfig(
            dataset_root=dataset_root,
            template_path=template_path,
            run_id="test_run",
            model="fake-glm",
            max_workers=1,
            max_attempts=1,
        ),
        worker=_FakeTeacherWorker(),
    )

    report = build_teacher_audit(dataset_root)

    assert report["candidate_count"] == 14
    assert report["labeled_count"] == 14
    assert report["missing_label_count"] == 0
    assert report["disagreement_count"] == 14
    disagreements = (
        dataset_root / "audit" / "disagreements.jsonl"
    ).read_text(encoding="utf-8").splitlines()
    assert len(disagreements) == 14


def test_reuse_labels_matches_exact_model_input_and_rewrites_sample_id(
    tmp_path: Path,
) -> None:
    source_root, template_path = _write_dataset(tmp_path / "source")
    run_teacher_labeling(
        TeacherRunConfig(
            dataset_root=source_root,
            template_path=template_path,
            run_id="source_run",
            model="fake-glm",
            max_workers=1,
            max_attempts=1,
        ),
        worker=_FakeTeacherWorker(),
    )
    target_root, _ = _write_dataset(tmp_path / "target")
    first_path = target_root / "candidates" / f"{ASPECT_FIELDS[0]}.jsonl"
    first = json.loads(first_path.read_text(encoding="utf-8"))
    first["sample_id"] = f"{ASPECT_FIELDS[0]}_000002"
    first_path.write_text(json.dumps(first, ensure_ascii=False) + "\n", encoding="utf-8")

    result = reuse_existing_labels(
        source_dataset_root=source_root,
        target_dataset_root=target_root,
    )

    assert result["reused_label_count"] == 14
    reused = json.loads(
        (target_root / "labeled" / f"{ASPECT_FIELDS[0]}.jsonl").read_text(
            encoding="utf-8"
        )
    )
    assert reused["sample_id"] == f"{ASPECT_FIELDS[0]}_000002"
    assert reused["model_output"] == {"relevance": 3, "strength": 4}


def _write_dataset(tmp_path: Path) -> tuple[Path, Path]:
    dataset_root = tmp_path / "teacher_dataset" / "v1"
    candidate_root = dataset_root / "candidates"
    candidate_root.mkdir(parents=True)
    template_path = tmp_path / "teacher_input_templates.v1.json"
    template_path.write_text(
        json.dumps({"system_prompt": "固定教师要求"}, ensure_ascii=False),
        encoding="utf-8",
    )
    aspects: dict[str, object] = {}
    for aspect in ASPECT_FIELDS:
        sample_id = f"{aspect}_000001"
        model_input = {
            "aspect_id": aspect,
            "definition": f"判断{aspect}",
            "relevance_scale": {
                "0": "无关",
                "1": "间接",
                "2": "明确但有限",
                "3": "明确充分",
            },
            "strength_scale": {
                "0": "最低",
                "1": "较低",
                "2": "中间",
                "3": "较高",
                "4": "最高",
            },
            "special_rules": ["不得猜测"],
            "review_text": f"真实评论 {aspect}",
        }
        row = {
            "sample_id": sample_id,
            "review_id": f"review-{aspect}",
            "business_id": f"business-{aspect}",
            "selection_relevance": 3,
            "selection_strength": 3,
            "model_input": model_input,
        }
        relative = f"candidates/{aspect}.jsonl"
        (dataset_root / relative).write_text(
            json.dumps(row, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        aspects[aspect] = {"candidate_count": 1, "output": relative}
    (dataset_root / "candidate_manifest.json").write_text(
        json.dumps({"schema_version": "1.0", "aspects": aspects}),
        encoding="utf-8",
    )
    return dataset_root, template_path
