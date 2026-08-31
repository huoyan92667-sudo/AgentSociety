"""比较候选挑选桶和教师标签，生成分布及不一致清单。"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from yelp_agent.recommendation_v2.schema import ASPECT_FIELDS

from .contracts import LabeledTeacherSample, TeacherCandidate


def build_teacher_audit(dataset_root: str | Path) -> dict[str, object]:
    """读取全部候选和正式标签，并覆盖生成可人工检查的审计结果。"""

    root = Path(dataset_root).resolve()
    candidates = _load_candidates(root / "candidates")
    labeled = _load_labeled(root / "labeled")
    missing = sorted(set(candidates) - set(labeled))
    unknown = sorted(set(labeled) - set(candidates))
    if unknown:
        raise ValueError(f"labeled data contains unknown samples: {unknown[:5]}")

    audit_root = root / "audit"
    audit_root.mkdir(parents=True, exist_ok=True)
    disagreements: list[dict[str, object]] = []
    by_aspect: dict[str, object] = {}

    for aspect in ASPECT_FIELDS:
        aspect_samples = [
            sample
            for sample in labeled.values()
            if sample.model_input.aspect_id == aspect
        ]
        relevance_counts = Counter(
            sample.model_output.relevance for sample in aspect_samples
        )
        strength_counts = Counter(
            sample.model_output.strength for sample in aspect_samples
        )
        aspect_disagreements = 0
        for sample in aspect_samples:
            candidate = candidates[sample.sample_id]
            reasons: list[str] = []
            if candidate.selection_relevance != sample.model_output.relevance:
                reasons.append("relevance")
            # 挑选时只有明确相关评论的程度桶才具有比较意义；模糊和无关
            # 评论的selection_strength只是为了占位，不能当成参考答案。
            if (
                candidate.selection_relevance >= 2
                and sample.model_output.relevance > 0
                and candidate.selection_strength != sample.model_output.strength
            ):
                reasons.append("strength")
            if not reasons:
                continue
            aspect_disagreements += 1
            candidate_payload = candidate.model_dump(mode="json")
            disagreements.append(
                {
                    "sample_id": sample.sample_id,
                    "aspect_id": aspect,
                    "reasons": reasons,
                    "selection_relevance": candidate.selection_relevance,
                    "teacher_relevance": sample.model_output.relevance,
                    "selection_strength": candidate.selection_strength,
                    "teacher_strength": sample.model_output.strength,
                    "selection_fallback": bool(
                        candidate_payload.get("selection_fallback", False)
                    ),
                    "review_id": candidate.review_id,
                    "business_id": candidate.business_id,
                    "review_text": candidate.model_input.review_text,
                }
            )
        by_aspect[aspect] = {
            "labeled_count": len(aspect_samples),
            "relevance_counts": {
                str(level): relevance_counts[level] for level in range(4)
            },
            "strength_counts": {
                **{str(level): strength_counts[level] for level in range(5)},
                "null": strength_counts[None],
            },
            "disagreement_count": aspect_disagreements,
        }

    _write_jsonl(audit_root / "disagreements.jsonl", disagreements)
    report: dict[str, object] = {
        "candidate_count": len(candidates),
        "labeled_count": len(labeled),
        "missing_label_count": len(missing),
        "missing_sample_ids": missing,
        "disagreement_count": len(disagreements),
        "aspects": by_aspect,
    }
    (audit_root / "label_distribution.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def _load_candidates(root: Path) -> dict[str, TeacherCandidate]:
    values: dict[str, TeacherCandidate] = {}
    for path in sorted(root.glob("*.jsonl")):
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not line.strip():
                continue
            try:
                candidate = TeacherCandidate.model_validate_json(line)
            except Exception as exc:
                raise ValueError(f"invalid candidate at {path}:{line_number}") from exc
            if candidate.sample_id in values:
                raise ValueError(f"duplicate candidate: {candidate.sample_id}")
            values[candidate.sample_id] = candidate
    return values


def _load_labeled(root: Path) -> dict[str, LabeledTeacherSample]:
    values: dict[str, LabeledTeacherSample] = {}
    for path in sorted(root.glob("*.jsonl")):
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not line.strip():
                continue
            try:
                sample = LabeledTeacherSample.model_validate_json(line)
            except Exception as exc:
                raise ValueError(f"invalid label at {path}:{line_number}") from exc
            if sample.sample_id in values:
                raise ValueError(f"duplicate label: {sample.sample_id}")
            values[sample.sample_id] = sample
    return values


def _write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"
            for value in values
        ),
        encoding="utf-8",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="生成教师标签分布和不一致清单")
    parser.add_argument("--dataset-root", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    report = build_teacher_audit(args.dataset_root)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
