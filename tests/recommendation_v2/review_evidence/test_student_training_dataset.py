from __future__ import annotations

import json
from pathlib import Path

from yelp_agent.recommendation_v2.review_evidence.student_training.dataset_builder import (
    BuildConfig,
    build_student_dataset,
    validate_student_dataset,
)


def test_build_student_dataset_keeps_review_groups_and_hides_trace_fields(tmp_path: Path) -> None:
    teacher_root = tmp_path / "teacher"
    output_root = tmp_path / "student"
    template_path = tmp_path / "template.json"
    (teacher_root / "candidates").mkdir(parents=True)
    (teacher_root / "labeled").mkdir(parents=True)
    template_path.write_text(json.dumps({"system_prompt": "固定教师要求"}), encoding="utf-8")

    for aspect_index, aspect in enumerate(("food_quality", "service")):
        candidates = []
        labels = []
        for sample_index in range(20):
            sample_id = f"{aspect}_{sample_index:06d}"
            # 前五条在两个特征中共享review_id，用来验证跨特征分组。
            review_id = f"shared-{sample_index}" if sample_index < 5 else f"{aspect}-{sample_index}"
            model_input = {
                "aspect_id": aspect,
                "definition": f"判断{aspect}",
                "relevance_scale": {str(level): f"相关{level}" for level in range(4)},
                "strength_scale": {str(level): f"强度{level}" for level in range(5)},
                "special_rules": ["只看评论"],
                "review_text": f"review text {aspect_index}-{sample_index}",
            }
            candidates.append(
                {
                    "sample_id": sample_id,
                    "review_id": review_id,
                    "business_id": "business",
                    "selection_relevance": 3,
                    "selection_strength": sample_index % 5,
                    "model_input": model_input,
                }
            )
            labels.append(
                {
                    "sample_id": sample_id,
                    "model_input": model_input,
                    "model_output": {"relevance": 3, "strength": sample_index % 5},
                }
            )
        _write_jsonl(teacher_root / "candidates" / f"{aspect}.jsonl", candidates)
        _write_jsonl(teacher_root / "labeled" / f"{aspect}.jsonl", labels)

    manifest = build_student_dataset(
        BuildConfig(
            teacher_dataset_root=teacher_root,
            teacher_template_path=template_path,
            output_root=output_root,
            split_search_trials=16,
        )
    )

    assert manifest["sample_count"] == 40
    assert manifest["review_leakage_count"] == 0
    assert validate_student_dataset(output_root)["valid"] is True

    review_splits: dict[str, set[str]] = {}
    for row in _read_jsonl(output_root / "split_index.jsonl"):
        review_splits.setdefault(row["review_id"], set()).add(row["split"])
    assert all(len(splits) == 1 for splits in review_splits.values())

    first_record = _read_jsonl(output_root / "train.jsonl")[0]
    assert set(first_record) == {"messages"}
    serialized_record = json.dumps(first_record, ensure_ascii=False)
    assert "sample_id" not in serialized_record
    assert "review_id" not in serialized_record
    assert "selection_strength" not in serialized_record
    assert json.loads(first_record["messages"][2]["content"]).keys() == {
        "relevance",
        "strength",
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
