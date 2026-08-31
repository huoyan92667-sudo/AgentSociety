"""从teacher dataset v2生成LLaMA-Factory训练、验证和测试数据。

这个模块刻意把“模型实际看到的数据”和“用于追溯的数据”分开：

* train/validation/test.jsonl 只包含system、user、assistant三条消息；
* split_index.jsonl 单独保存sample_id和review_id，不会交给学生模型；
* 同一个review_id始终只进入一个集合，避免验证成绩因为数据泄漏而虚高。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from yelp_agent.recommendation_v2.review_evidence.teacher_labeling.contracts import (
    LabeledTeacherSample,
    TeacherCandidate,
)

SplitName = Literal["train", "validation", "test"]
SPLITS: tuple[SplitName, ...] = ("train", "validation", "test")
DATASET_NAMES: dict[SplitName, str] = {
    "train": "teacher_v2_train",
    "validation": "teacher_v2_validation",
    "test": "teacher_v2_test",
}
DEFAULT_RATIOS: dict[SplitName, float] = {
    "train": 0.8,
    "validation": 0.1,
    "test": 0.1,
}


@dataclass(frozen=True, slots=True)
class BuildConfig:
    """构建训练数据所需的输入、输出和可重复切分参数。"""

    teacher_dataset_root: Path
    teacher_template_path: Path
    output_root: Path
    seed: int = 20260831
    split_search_trials: int = 1024

    def __post_init__(self) -> None:
        if self.split_search_trials < 1:
            raise ValueError("split_search_trials must be positive")


class _ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)


class _TrainingRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    messages: list[_ChatMessage] = Field(min_length=3, max_length=3)

    @model_validator(mode="after")
    def validate_roles(self) -> "_TrainingRecord":
        if [message.role for message in self.messages] != ["system", "user", "assistant"]:
            raise ValueError("messages must be ordered as system, user, assistant")
        return self


@dataclass(frozen=True, slots=True)
class _JoinedSample:
    candidate: TeacherCandidate
    labeled: LabeledTeacherSample

    @property
    def review_id(self) -> str:
        return self.candidate.review_id

    @property
    def aspect_id(self) -> str:
        return str(self.labeled.model_input.aspect_id)

    @property
    def relevance(self) -> int:
        return self.labeled.model_output.relevance

    @property
    def strength_bucket(self) -> str:
        value = self.labeled.model_output.strength
        return "null" if value is None else str(value)


def build_student_dataset(config: BuildConfig) -> dict[str, Any]:
    """生成三份模型数据、独立追溯索引、LLaMA-Factory登记和审计报告。"""

    teacher_root = config.teacher_dataset_root.resolve()
    output_root = config.output_root.resolve()
    system_prompt = _load_system_prompt(config.teacher_template_path.resolve(), teacher_root)
    samples = _load_and_join_samples(teacher_root)
    review_assignment, chosen_trial = _choose_grouped_split(
        samples,
        seed=config.seed,
        trials=config.split_search_trials,
    )

    output_root.mkdir(parents=True, exist_ok=True)
    records_by_split: dict[SplitName, list[tuple[_JoinedSample, _TrainingRecord]]] = {
        split: [] for split in SPLITS
    }
    for sample in sorted(samples, key=lambda item: item.candidate.sample_id):
        split = review_assignment[sample.review_id]
        records_by_split[split].append((sample, _make_training_record(sample, system_prompt)))

    index_rows: list[dict[str, object]] = []
    for split in SPLITS:
        output_path = output_root / f"{split}.jsonl"
        rows = records_by_split[split]
        _write_jsonl(output_path, (record.model_dump(mode="json") for _, record in rows))
        for line_number, (sample, _) in enumerate(rows, 1):
            index_rows.append(
                {
                    "split": split,
                    "line_number": line_number,
                    "sample_id": sample.candidate.sample_id,
                    "review_id": sample.review_id,
                    "business_id": sample.candidate.business_id,
                    "aspect_id": sample.aspect_id,
                }
            )

    _write_jsonl(output_root / "split_index.jsonl", index_rows)
    _write_json(output_root / "dataset_info.json", _dataset_info())
    distribution = _build_distribution(records_by_split)
    _write_json(output_root / "distribution.json", distribution)
    manifest = _build_manifest(
        config=config,
        teacher_root=teacher_root,
        output_root=output_root,
        samples=samples,
        records_by_split=records_by_split,
        chosen_trial=chosen_trial,
        system_prompt=system_prompt,
    )
    _write_json(output_root / "manifest.json", manifest)
    validate_student_dataset(output_root)
    return manifest


def validate_student_dataset(output_root: Path) -> dict[str, object]:
    """检查消息格式、答案关系、追溯索引以及三份数据之间是否泄漏。"""

    output_root = output_root.resolve()
    index_rows = _read_jsonl(output_root / "split_index.jsonl")
    index_by_location: dict[tuple[str, int], dict[str, Any]] = {}
    review_splits: dict[str, set[str]] = defaultdict(set)
    for row in index_rows:
        split = str(row.get("split"))
        line_number = int(row.get("line_number") or 0)
        if split not in SPLITS or line_number < 1:
            raise ValueError(f"invalid split index row: {row}")
        location = (split, line_number)
        if location in index_by_location:
            raise ValueError(f"duplicate split index location: {location}")
        index_by_location[location] = row
        review_splits[str(row.get("review_id"))].add(split)

    leaked = {review_id: splits for review_id, splits in review_splits.items() if len(splits) > 1}
    if leaked:
        example = next(iter(leaked.items()))
        raise ValueError(f"review leakage across splits: {example}")

    counts: dict[str, int] = {}
    for split in SPLITS:
        rows = _read_jsonl(output_root / f"{split}.jsonl")
        counts[split] = len(rows)
        for line_number, row in enumerate(rows, 1):
            record = _TrainingRecord.model_validate(row)
            user_payload = json.loads(record.messages[1].content)
            answer = json.loads(record.messages[2].content)
            if set(user_payload) != {
                "aspect_id",
                "definition",
                "relevance_scale",
                "strength_scale",
                "special_rules",
                "review_text",
            }:
                raise ValueError(f"unexpected model input at {split}:{line_number}")
            if set(answer) != {"relevance", "strength"}:
                raise ValueError(f"unexpected model output at {split}:{line_number}")
            relevance = answer["relevance"]
            strength = answer["strength"]
            if relevance not in {0, 1, 2, 3}:
                raise ValueError(f"invalid relevance at {split}:{line_number}")
            if (relevance == 0) != (strength is None):
                raise ValueError(f"invalid relevance/strength relation at {split}:{line_number}")
            if strength is not None and strength not in {0, 1, 2, 3, 4}:
                raise ValueError(f"invalid strength at {split}:{line_number}")
            if (split, line_number) not in index_by_location:
                raise ValueError(f"missing trace index for {split}:{line_number}")

    if sum(counts.values()) != len(index_rows):
        raise ValueError("training rows and trace-index rows have different counts")
    return {
        "valid": True,
        "sample_counts": counts,
        "unique_review_count": len(review_splits),
        "review_leakage_count": 0,
    }


def _load_system_prompt(template_path: Path, teacher_root: Path) -> str:
    template = json.loads(template_path.read_text(encoding="utf-8"))
    system_prompt = template.get("system_prompt")
    if not isinstance(system_prompt, str) or not system_prompt.strip():
        raise ValueError("teacher template has no system_prompt")

    # 教师运行记录保存了当时真正使用的提示词；两者必须相同，避免学生
    # 在不知情的情况下学习另一套任务说明。
    run_manifests = sorted((teacher_root / "teacher_runs").glob("*/run_manifest.json"))
    if run_manifests:
        expected_prompts = {
            json.loads(path.read_text(encoding="utf-8")).get("system_prompt")
            for path in run_manifests
        }
        expected_prompts.discard(None)
        if expected_prompts and expected_prompts != {system_prompt}:
            raise ValueError("teacher template and teacher-run system prompts do not match")
    return system_prompt


def _load_and_join_samples(teacher_root: Path) -> list[_JoinedSample]:
    candidates: dict[str, TeacherCandidate] = {}
    for path in sorted((teacher_root / "candidates").glob("*.jsonl")):
        for line_number, row in enumerate(_read_jsonl(path), 1):
            candidate = TeacherCandidate.model_validate(row)
            if candidate.sample_id in candidates:
                raise ValueError(f"duplicate candidate: {candidate.sample_id}")
            candidates[candidate.sample_id] = candidate

    labeled_samples: dict[str, LabeledTeacherSample] = {}
    for path in sorted((teacher_root / "labeled").glob("*.jsonl")):
        for line_number, row in enumerate(_read_jsonl(path), 1):
            labeled = LabeledTeacherSample.model_validate(row)
            if labeled.sample_id in labeled_samples:
                raise ValueError(f"duplicate label: {labeled.sample_id}")
            labeled_samples[labeled.sample_id] = labeled

    missing_candidates = sorted(set(labeled_samples) - set(candidates))
    missing_labels = sorted(set(candidates) - set(labeled_samples))
    if missing_candidates or missing_labels:
        raise ValueError(
            "candidate/label mismatch: "
            f"missing_candidates={missing_candidates[:3]}, missing_labels={missing_labels[:3]}"
        )

    joined: list[_JoinedSample] = []
    for sample_id in sorted(labeled_samples):
        candidate = candidates[sample_id]
        labeled = labeled_samples[sample_id]
        if candidate.model_input.model_dump(mode="json") != labeled.model_input.model_dump(mode="json"):
            raise ValueError(f"teacher input changed between candidate and label: {sample_id}")
        joined.append(_JoinedSample(candidate=candidate, labeled=labeled))
    if not joined:
        raise ValueError("teacher dataset is empty")
    return joined


def _choose_grouped_split(
    samples: list[_JoinedSample],
    *,
    seed: int,
    trials: int,
) -> tuple[dict[str, SplitName], int]:
    """在多次可重复随机切分中选择整体和逐类分布最接近目标的一次。"""

    review_ids = sorted({sample.review_id for sample in samples})
    best_score: float | None = None
    best_assignment: dict[str, SplitName] | None = None
    best_trial = 0
    for trial in range(trials):
        shuffled = list(review_ids)
        random.Random(seed + trial).shuffle(shuffled)
        assignment = _assign_review_groups(shuffled)
        score = _split_score(samples, assignment)
        if best_score is None or score < best_score:
            best_score = score
            best_assignment = assignment
            best_trial = trial
    if best_assignment is None:
        raise RuntimeError("failed to choose a dataset split")
    return best_assignment, best_trial


def _assign_review_groups(review_ids: list[str]) -> dict[str, SplitName]:
    total = len(review_ids)
    train_end = round(total * DEFAULT_RATIOS["train"])
    validation_end = train_end + round(total * DEFAULT_RATIOS["validation"])
    assignment: dict[str, SplitName] = {}
    for index, review_id in enumerate(review_ids):
        if index < train_end:
            assignment[review_id] = "train"
        elif index < validation_end:
            assignment[review_id] = "validation"
        else:
            assignment[review_id] = "test"
    return assignment


def _split_score(samples: list[_JoinedSample], assignment: dict[str, SplitName]) -> float:
    """同时衡量总量、14种特征、相关程度和五档强度的偏差。"""

    global_strata = Counter(_stratum(sample) for sample in samples)
    split_strata: dict[SplitName, Counter[tuple[str, ...]]] = {
        split: Counter() for split in SPLITS
    }
    for sample in samples:
        split = assignment[sample.review_id]
        split_strata[split][_stratum(sample)] += 1

    score = 0.0
    for split in SPLITS:
        ratio = DEFAULT_RATIOS[split]
        actual_total = sum(split_strata[split].values())
        expected_total = len(samples) * ratio
        score += 8.0 * ((actual_total - expected_total) / max(1.0, expected_total)) ** 2
        for stratum, global_count in global_strata.items():
            expected = global_count * ratio
            actual = split_strata[split][stratum]
            # 稀有组合不能拥有与大类同样的绝对误差权重，按期望数量平滑。
            score += ((actual - expected) ** 2) / max(1.0, expected)
    return score


def _stratum(sample: _JoinedSample) -> tuple[str, ...]:
    return (
        sample.aspect_id,
        f"relevance={sample.relevance}",
        f"strength={sample.strength_bucket}",
    )


def _make_training_record(sample: _JoinedSample, system_prompt: str) -> _TrainingRecord:
    user_content = _compact_json(sample.labeled.model_input.model_dump(mode="json"))
    assistant_content = _compact_json(sample.labeled.model_output.model_dump(mode="json"))
    return _TrainingRecord(
        messages=[
            _ChatMessage(role="system", content=system_prompt),
            _ChatMessage(role="user", content=user_content),
            _ChatMessage(role="assistant", content=assistant_content),
        ]
    )


def _dataset_info() -> dict[str, object]:
    common = {
        "formatting": "sharegpt",
        "columns": {"messages": "messages"},
        "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant",
            "system_tag": "system",
        },
    }
    return {
        DATASET_NAMES[split]: {"file_name": f"{split}.jsonl", **common}
        for split in SPLITS
    }


def _build_distribution(
    records_by_split: dict[SplitName, list[tuple[_JoinedSample, _TrainingRecord]]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for split in SPLITS:
        samples = [sample for sample, _ in records_by_split[split]]
        aspects = Counter(sample.aspect_id for sample in samples)
        relevance = Counter(str(sample.relevance) for sample in samples)
        strength = Counter(sample.strength_bucket for sample in samples)
        aspect_labels: dict[str, dict[str, dict[str, int]]] = {}
        for aspect in sorted(aspects):
            selected = [sample for sample in samples if sample.aspect_id == aspect]
            aspect_labels[aspect] = {
                "relevance": {
                    str(level): sum(sample.relevance == level for sample in selected)
                    for level in range(4)
                },
                "strength": {
                    **{
                        str(level): sum(sample.strength_bucket == str(level) for sample in selected)
                        for level in range(5)
                    },
                    "null": sum(sample.strength_bucket == "null" for sample in selected),
                },
            }
        result[split] = {
            "sample_count": len(samples),
            "unique_review_count": len({sample.review_id for sample in samples}),
            "aspect_counts": dict(sorted(aspects.items())),
            "relevance_counts": {str(level): relevance[str(level)] for level in range(4)},
            "strength_counts": {
                **{str(level): strength[str(level)] for level in range(5)},
                "null": strength["null"],
            },
            "by_aspect": aspect_labels,
        }
    return result


def _build_manifest(
    *,
    config: BuildConfig,
    teacher_root: Path,
    output_root: Path,
    samples: list[_JoinedSample],
    records_by_split: dict[SplitName, list[tuple[_JoinedSample, _TrainingRecord]]],
    chosen_trial: int,
    system_prompt: str,
) -> dict[str, Any]:
    input_lengths = [
        len(_compact_json(sample.labeled.model_input.model_dump(mode="json")))
        for sample in samples
    ]
    review_lengths = [len(sample.labeled.model_input.review_text) for sample in samples]
    return {
        "schema_version": "1.0",
        # 使用可移植的逻辑来源名；不能把某台电脑的绝对路径写进要上传的产物。
        "source_dataset": "teacher_dataset/v2",
        "sample_count": len(samples),
        "unique_review_count": len({sample.review_id for sample in samples}),
        "split_rule": "grouped_by_review_id",
        "split_ratios": DEFAULT_RATIOS,
        "seed": config.seed,
        "split_search_trials": config.split_search_trials,
        "chosen_trial": chosen_trial,
        "sample_counts": {
            split: len(records_by_split[split]) for split in SPLITS
        },
        "unique_review_counts": {
            split: len({sample.review_id for sample, _ in records_by_split[split]})
            for split in SPLITS
        },
        "review_leakage_count": 0,
        "system_prompt_sha256": _sha256_text(system_prompt),
        "character_lengths": {
            "model_input": _length_summary(input_lengths),
            "review_text": _length_summary(review_lengths),
            "note": "字符数不是模型词元数；上传模型后仍需用真实tokenizer检查截断。",
        },
        "test_set_status": "teacher_labeled_not_manually_verified",
        "model_visible_fields": {
            "user": [
                "aspect_id",
                "definition",
                "relevance_scale",
                "strength_scale",
                "special_rules",
                "review_text",
            ],
            "assistant": ["relevance", "strength"],
        },
        "trace_only_fields": [
            "sample_id",
            "review_id",
            "business_id",
            "aspect_id",
            "split",
            "line_number",
        ],
        "files": {
            path.name: {
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
            for path in [
                *(output_root / f"{split}.jsonl" for split in SPLITS),
                output_root / "split_index.jsonl",
                output_root / "dataset_info.json",
                output_root / "distribution.json",
            ]
        },
    }


def _length_summary(values: list[int]) -> dict[str, float | int]:
    ordered = sorted(values)

    def percentile(fraction: float) -> int:
        index = min(len(ordered) - 1, round((len(ordered) - 1) * fraction))
        return ordered[index]

    return {
        "minimum": ordered[0],
        "average": round(sum(ordered) / len(ordered), 2),
        "p50": percentile(0.5),
        "p90": percentile(0.9),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "maximum": ordered[-1],
    }


def _compact_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"JSONL row must be an object at {path}:{line_number}")
        rows.append(value)
    return rows


def _write_jsonl(path: Path, rows: Iterable[object]) -> None:
    path.write_text(
        "".join(_compact_json(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _default_project_paths() -> tuple[Path, Path, Path]:
    recommendation_root = Path(__file__).resolve().parents[2]
    teacher_root = recommendation_root / "data" / "teacher_dataset" / "v2"
    template_path = (
        recommendation_root
        / "review_evidence"
        / "training_data"
        / "teacher_input_templates.v1.json"
    )
    output_root = (
        recommendation_root
        / "data"
        / "student_training"
        / "qwen3_4b_teacher_v2"
        / "v1"
    )
    return teacher_root, template_path, output_root


def main(argv: list[str] | None = None) -> None:
    teacher_root, template_path, output_root = _default_project_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build", help="从teacher v2重新生成学生训练数据")
    build_parser.add_argument("--teacher-root", type=Path, default=teacher_root)
    build_parser.add_argument("--template", type=Path, default=template_path)
    build_parser.add_argument("--output-root", type=Path, default=output_root)
    build_parser.add_argument("--seed", type=int, default=20260831)
    build_parser.add_argument("--split-search-trials", type=int, default=1024)
    validate_parser = subparsers.add_parser("validate", help="检查已经生成的数据")
    validate_parser.add_argument("--output-root", type=Path, default=output_root)
    args = parser.parse_args(argv)

    if args.command == "build":
        result = build_student_dataset(
            BuildConfig(
                teacher_dataset_root=args.teacher_root,
                teacher_template_path=args.template,
                output_root=args.output_root,
                seed=args.seed,
                split_search_trials=args.split_search_trials,
            )
        )
    else:
        result = validate_student_dataset(args.output_root)
    print(json.dumps(result, ensure_ascii=False, indent=2))
