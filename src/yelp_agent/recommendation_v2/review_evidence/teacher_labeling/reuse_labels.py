"""把旧批次中输入完全相同的教师标签复用到新批次。"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from yelp_agent.recommendation_v2.schema import ASPECT_FIELDS

from .contracts import LabeledTeacherSample, TeacherCandidate


def reuse_existing_labels(
    *,
    source_dataset_root: str | Path,
    target_dataset_root: str | Path,
) -> dict[str, object]:
    """按模型实际输入复用旧标签，绝不按旧挑选桶猜测答案。"""

    source_root = Path(source_dataset_root).resolve()
    target_root = Path(target_dataset_root).resolve()
    source_labels = _load_labels(source_root / "labeled")
    target_candidates = _load_candidates(target_root / "candidates")

    labels_by_input: dict[str, LabeledTeacherSample] = {}
    for sample in source_labels:
        key = _input_key(sample.model_input.model_dump(mode="json"))
        previous = labels_by_input.get(key)
        if previous and previous.model_output != sample.model_output:
            raise ValueError(
                "source dataset contains conflicting labels for identical model input"
            )
        labels_by_input[key] = sample

    reusable: dict[str, list[LabeledTeacherSample]] = defaultdict(list)
    reused_source_ids: set[str] = set()
    for candidate in target_candidates:
        key = _input_key(candidate.model_input.model_dump(mode="json"))
        source = labels_by_input.get(key)
        if source is None:
            continue
        reusable[candidate.model_input.aspect_id].append(
            LabeledTeacherSample(
                sample_id=candidate.sample_id,
                model_input=candidate.model_input,
                model_output=source.model_output,
            )
        )
        reused_source_ids.add(source.sample_id)

    labeled_root = target_root / "labeled"
    labeled_root.mkdir(parents=True, exist_ok=True)
    for aspect in ASPECT_FIELDS:
        rows = reusable.get(aspect, [])
        path = labeled_root / f"{aspect}.jsonl"
        path.write_text(
            "".join(
                json.dumps(row.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":"))
                + "\n"
                for row in rows
            ),
            encoding="utf-8",
        )

    result: dict[str, object] = {
        "source_label_count": len(source_labels),
        "target_candidate_count": len(target_candidates),
        "reused_label_count": sum(len(rows) for rows in reusable.values()),
        "unmatched_source_label_count": len(source_labels) - len(reused_source_ids),
        "remaining_for_teacher": len(target_candidates)
        - sum(len(rows) for rows in reusable.values()),
        "reused_by_aspect": {
            aspect: len(reusable.get(aspect, [])) for aspect in ASPECT_FIELDS
        },
    }
    (target_root / "reused_labels_manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def _input_key(value: dict[str, object]) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_labels(root: Path) -> list[LabeledTeacherSample]:
    values: list[LabeledTeacherSample] = []
    seen: set[str] = set()
    for path in sorted(root.glob("*.jsonl")):
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                item = LabeledTeacherSample.model_validate_json(line)
            except Exception as exc:
                raise ValueError(f"invalid source label at {path}:{line_number}") from exc
            if item.sample_id in seen:
                raise ValueError(f"duplicate source label: {item.sample_id}")
            seen.add(item.sample_id)
            values.append(item)
    return values


def _load_candidates(root: Path) -> list[TeacherCandidate]:
    values: list[TeacherCandidate] = []
    seen: set[str] = set()
    for aspect in ASPECT_FIELDS:
        path = root / f"{aspect}.jsonl"
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                item = TeacherCandidate.model_validate_json(line)
            except Exception as exc:
                raise ValueError(f"invalid target candidate at {path}:{line_number}") from exc
            if item.sample_id in seen:
                raise ValueError(f"duplicate target candidate: {item.sample_id}")
            seen.add(item.sample_id)
            values.append(item)
    return values


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="复用输入完全相同的已有教师标签")
    parser.add_argument("--source-dataset-root", type=Path, required=True)
    parser.add_argument("--target-dataset-root", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = reuse_existing_labels(
        source_dataset_root=args.source_dataset_root,
        target_dataset_root=args.target_dataset_root,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
