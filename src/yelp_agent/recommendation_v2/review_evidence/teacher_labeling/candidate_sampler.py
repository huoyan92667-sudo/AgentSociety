"""从已有评论候选池中挑选教师标注种子数据。

这里的等级只是“挑选目标”，不是教师模型答案。模型输入只包含软偏好
定义和评论正文；商家编号、评论编号、星级、点赞数等信息只写入候选文件，
用于回溯和检查，绝不会传给教师模型。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


ASPECT_ORDER = (
    "food_quality",
    "service",
    "price_value",
    "quiet_environment",
    "crowded",
    "queue_time",
    "portion_size",
    "parking",
    "pet_friendly",
    "family_friendly",
    "date_suitable",
    "group_suitable",
    "spiciness",
    "cleanliness",
)

# 这三个特征在旧词表中“正面词”代表低端，而新的客观刻度仍然要求高分
# 表示数值更高：拥挤、等位时间、辣度。
LOW_FIRST_ASPECTS = {"crowded", "queue_time", "spiciness"}


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _load_vocab(path: Path) -> dict[str, dict[str, list[str]]]:
    import yaml

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    aspects = payload.get("aspects") if isinstance(payload, dict) else None
    if not isinstance(aspects, dict):
        raise ValueError("review vocabulary must contain an aspects mapping")
    result: dict[str, dict[str, list[str]]] = {}
    for aspect in ASPECT_ORDER:
        item = aspects.get(aspect, {})
        result[aspect] = {
            "positive": [str(value).casefold() for value in item.get("positive", [])],
            "negative": [str(value).casefold() for value in item.get("negative", [])],
        }
    return result


def _term_pattern(terms: list[str]) -> re.Pattern[str]:
    values = sorted((re.escape(value) for value in terms if value), key=len, reverse=True)
    return re.compile(rf"(?<!\w)(?:{'|'.join(values)})(?!\w)", re.IGNORECASE) if values else re.compile(r"$^")


def _term_count(text: str, pattern: re.Pattern[str]) -> int:
    return sum(1 for _ in pattern.finditer(text))


def _stable_rank(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:12], 16)


def _fetch_segments(
    segment_path: Path,
    review_ids: set[str] | list[str],
) -> dict[str, list[dict[str, Any]]]:
    """从片段 Parquet 中挑出候选评论对应的片段。

    当前环境没有预装 DuckDB，因此这里直接用 PyArrow 分批读取。虽然仍会
    顺序扫一遍片段文件，但不会把 60 多万条评论正文一次性载入内存。
    """

    if not review_ids:
        return {}
    wanted = {str(value) for value in review_ids}
    columns = [
        "review_id",
        "segment_id",
        "segment_index",
        "text",
        "user_id",
        "review_time",
        "stars",
        "useful",
        "business_id",
    ]
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    parquet_file = pq.ParquetFile(segment_path)
    for batch in parquet_file.iter_batches(batch_size=8192, columns=columns):
        for item in batch.to_pylist():
            review_id = str(item.get("review_id"))
            if review_id in wanted:
                result[review_id].append(item)
    for rows in result.values():
        rows.sort(key=lambda item: int(item.get("segment_index") or 0))
    return dict(result)


def _choose_segment(
    segments: list[dict[str, Any]],
    positive_pattern: re.Pattern[str],
    negative_pattern: re.Pattern[str],
) -> dict[str, Any] | None:
    """优先选包含该特征词的片段；没有词时保留评论第一段。"""

    if not segments:
        return None
    ranked = sorted(
        segments,
        key=lambda item: (
            -(_term_count(str(item["text"]), positive_pattern) + _term_count(str(item["text"]), negative_pattern)),
            int(item["segment_index"]),
        ),
    )
    return ranked[0]


def _candidate_quality(row: dict[str, Any]) -> tuple[float, int, int, str]:
    """候选排序：优先两路都命中、语义分高、且评论信息更充分的样本。"""

    return (
        float(row.get("semantic_score") or 0.0)
        + (0.04 if bool(row.get("keyword_hit")) else 0.0)
        + (0.02 if bool(row.get("semantic_hit")) else 0.0),
        int(row.get("useful") or 0),
        len(str(row.get("review_text") or "")),
        str(row["review_id"]),
    )


def _objective_bucket(
    *,
    aspect: str,
    high_count: int,
    low_count: int,
    semantic_score: float,
    anchor_ids: list[str],
) -> int:
    """把候选暂时放进一个“待教师确认”的程度桶，不冒充最终标签。"""

    if high_count and low_count:
        return 2
    if high_count:
        return 4 if high_count >= 2 or semantic_score >= 0.70 else 3
    if low_count:
        return 0 if low_count >= 2 or semantic_score >= 0.70 else 1

    # 语义路线没有旧词时，使用原有四端语义锚点判断它靠近哪一端；
    # 中间等级仍然优先从没有明显端点词的评论里挑选。
    sides = {item.rsplit(":", 1)[-1] for item in anchor_ids}
    if sides & {"0", "1"} and not sides & {"2", "3"}:
        return 1 if aspect in LOW_FIRST_ASPECTS else 3
    if sides & {"2", "3"} and not sides & {"0", "1"}:
        return 3 if aspect in LOW_FIRST_ASPECTS else 1
    return 2


def _load_all_aspect_rows(candidate_aspects_path: Path) -> dict[str, list[dict[str, Any]]]:
    """一次读取候选关系，并按14种特征分组，避免每个特征重复扫文件。"""

    columns = [
        "review_id",
        "business_id",
        "aspect",
        "keyword_hit",
        "semantic_hit",
        "semantic_score",
        "matched_terms",
        "matched_anchor_ids",
    ]
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    parquet_file = pq.ParquetFile(candidate_aspects_path)
    for batch in parquet_file.iter_batches(batch_size=16384, columns=columns):
        for item in batch.to_pylist():
            aspect = str(item.get("aspect") or "")
            if aspect in ASPECT_ORDER:
                result[aspect].append(item)
    return dict(result)


def _load_unrelated_rows(
    candidate_reviews_path: Path,
    excluded_ids: set[str],
    limit: int,
) -> list[dict[str, Any]]:
    """从其它真实评论中找不含当前特征词的无关候选。"""

    columns = [
        "review_id",
        "business_id",
        "user_id",
        "review_time",
        "stars",
        "useful",
        "review_text",
    ]
    pool: list[dict[str, Any]] = []
    parquet_file = pq.ParquetFile(candidate_reviews_path)
    for batch in parquet_file.iter_batches(batch_size=16384, columns=columns):
        for item in batch.to_pylist():
            if str(item.get("review_id")) not in excluded_ids:
                pool.append(item)
    pool.sort(key=lambda item: _stable_rank(str(item.get("review_id"))))
    return pool[: max(limit * 30, 500)]


def _make_input(
    *,
    template: dict[str, Any],
    aspect: dict[str, Any],
    review_text: str,
) -> dict[str, Any]:
    """生成真正发送给教师和学生的精简输入。"""

    return {
        "aspect_id": aspect["id"],
        "definition": aspect["definition"],
        "relevance_scale": template["common_relevance_scale"],
        "strength_scale": aspect["strength_scale"],
        "special_rules": aspect["special_rules"],
        "review_text": review_text,
    }


def _select_bucket(
    rows: list[dict[str, Any]],
    *,
    aspect: str,
    bucket: int,
    count: int,
    used: set[str],
) -> list[dict[str, Any]]:
    """按目标等级挑选不同商家的真实评论，缺少时才允许同商家补位。"""

    pool = [row for row in rows if row["review_id"] not in used and row["selection_strength"] == bucket]
    pool.sort(key=_candidate_quality, reverse=True)
    selected: list[dict[str, Any]] = []
    businesses: set[str] = set()
    for distinct_business in (True, False):
        for row in pool:
            if len(selected) >= count:
                break
            review_id = str(row["review_id"])
            # 第二轮会再次遍历同一个候选池；必须跳过第一轮已加入的评论，
            # 否则同一条评论会占据两个训练样本位置。
            if review_id in used:
                continue
            if distinct_business and row["business_id"] in businesses:
                continue
            selected.append(dict(row))
            used.add(review_id)
            businesses.add(str(row["business_id"]))
        if len(selected) >= count:
            break
    return selected


def _fill_bucket_by_semantic_distance(
    rows: list[dict[str, Any]],
    *,
    bucket: int,
    count: int,
    used: set[str],
) -> list[dict[str, Any]]:
    """某个程度桶不足时，用语义分最接近的真实评论补位。

    补位只为让教师看到完整的五档样本，绝不是把程序的猜测当成答案；
    `selection_fallback` 会明确记录这条样本是补位得到的，最终等级仍以
    教师模型输出为准。
    """

    target = bucket / 4.0
    pool = [row for row in rows if str(row["review_id"]) not in used]
    pool.sort(
        key=lambda row: (
            abs(float(row.get("semantic_score") or 0.0) - target),
            -int(row.get("useful") or 0),
            str(row["review_id"]),
        )
    )
    selected: list[dict[str, Any]] = []
    businesses: set[str] = set()
    # 先尽量保证每档来自不同商家，避免某一商家的评论占满一档。
    for distinct_business in (True, False):
        for row in pool:
            if len(selected) >= count:
                break
            review_id = str(row["review_id"])
            # 和精确等级挑选一样，补位的第二轮也不能重复选择第一轮的评论。
            if review_id in used:
                continue
            if distinct_business and str(row["business_id"]) in businesses:
                continue
            item = dict(row)
            item["selection_original_bucket"] = item.get("selection_strength")
            item["selection_strength"] = bucket
            item["selection_fallback"] = True
            selected.append(item)
            used.add(review_id)
            businesses.add(str(item["business_id"]))
        if len(selected) >= count:
            break
    return selected


def build_teacher_candidates(
    *,
    template_path: str | Path,
    vocabulary_path: str | Path,
    candidate_aspects_path: str | Path,
    candidate_reviews_path: str | Path,
    segments_path: str | Path,
    output_root: str | Path,
    per_bucket: int = 5,
) -> dict[str, Any]:
    """按指定规模生成教师候选，并写成按特征分片的JSONL。"""

    if per_bucket < 1:
        raise ValueError("per_bucket must be positive")
    template = _read_json(Path(template_path))
    aspects = template.get("aspects")
    if not isinstance(aspects, list) or {item.get("id") for item in aspects} != set(ASPECT_ORDER):
        raise ValueError("teacher template does not contain exactly the 14 expected aspects")
    aspect_by_id = {str(item["id"]): item for item in aspects}
    vocabulary = _load_vocab(Path(vocabulary_path))
    destination = Path(output_root)
    candidates_dir = destination / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"schema_version": "1.0", "per_bucket": per_bucket, "aspects": {}}

    relation_rows_by_aspect = _load_all_aspect_rows(Path(candidate_aspects_path))
    for aspect in ASPECT_ORDER:
        vocab = vocabulary[aspect]
        positive_pattern = _term_pattern(vocab["positive"])
        negative_pattern = _term_pattern(vocab["negative"])
        relation_rows = relation_rows_by_aspect.get(aspect, [])
        by_review: dict[str, dict[str, Any]] = {}
        for raw in relation_rows:
            review_id = str(raw["review_id"])
            previous = by_review.get(review_id)
            if previous is None or _candidate_quality(raw) > _candidate_quality(previous):
                by_review[review_id] = raw

        # 每个特征只扫描一次片段文件，并且只保留本特征的评论，控制内存。
        segment_map = _fetch_segments(Path(segments_path), set(by_review))
        prepared: list[dict[str, Any]] = []
        for raw in by_review.values():
            segment = _choose_segment(segment_map.get(str(raw["review_id"]), []), positive_pattern, negative_pattern)
            if segment is None:
                continue
            text = str(segment["text"])
            positive_count = _term_count(text, positive_pattern)
            negative_count = _term_count(text, negative_pattern)
            if aspect in LOW_FIRST_ASPECTS:
                high_count, low_count = negative_count, positive_count
            else:
                high_count, low_count = positive_count, negative_count
            level = _objective_bucket(
                aspect=aspect,
                high_count=high_count,
                low_count=low_count,
                semantic_score=float(raw.get("semantic_score") or 0.0),
                anchor_ids=[str(value) for value in (raw.get("matched_anchor_ids") or [])],
            )
            prepared.append(
                {
                    "review_id": str(raw["review_id"]),
                    "business_id": str(raw["business_id"]),
                    "segment_id": str(segment["segment_id"]),
                    "user_id": str(segment["user_id"]),
                    "review_time": segment["review_time"].isoformat() if hasattr(segment["review_time"], "isoformat") else str(segment["review_time"]),
                    "stars": float(segment["stars"]),
                    "useful": int(segment["useful"]),
                    "review_text": text,
                    "keyword_hit": bool(raw.get("keyword_hit")),
                    "semantic_hit": bool(raw.get("semantic_hit")),
                    "semantic_score": float(raw.get("semantic_score") or 0.0),
                    "matched_terms": [str(value) for value in (raw.get("matched_terms") or [])],
                    "matched_anchor_ids": [str(value) for value in (raw.get("matched_anchor_ids") or [])],
                    "positive_term_count": positive_count,
                    "negative_term_count": negative_count,
                    "selection_strength": level,
                    "selection_relevance": 3 if (high_count or low_count) else 2,
                }
            )

        used: set[str] = set()
        selected: list[dict[str, Any]] = []
        for level in range(5):
            level_rows = _select_bucket(prepared, aspect=aspect, bucket=level, count=per_bucket, used=used)
            selected.extend(level_rows)
            if len(level_rows) < per_bucket:
                selected.extend(
                    _fill_bucket_by_semantic_distance(
                        prepared,
                        bucket=level,
                        count=per_bucket - len(level_rows),
                        used=used,
                    )
                )

        # 有关但证据不足的样本：未被等级样本使用、没有明显词，或同时出现两端描述。
        unclear_pool = [
            row for row in prepared
            if row["review_id"] not in used
            and (row["positive_term_count"] + row["negative_term_count"] == 0
                 or (row["positive_term_count"] and row["negative_term_count"]))
        ]
        unclear_pool.sort(key=_candidate_quality, reverse=True)
        unclear_needed = per_bucket * 3
        unclear_added = 0
        for row in unclear_pool:
            if unclear_added >= unclear_needed:
                break
            row = dict(row)
            row["selection_relevance"] = 1 if row["positive_term_count"] + row["negative_term_count"] == 0 else 2
            row["selection_strength"] = 2
            selected.append(row)
            used.add(row["review_id"])
            unclear_added += 1

        # 完全无关样本直接使用真实评论正文的前900字符，不再额外扫描片段文件。
        unrelated = _load_unrelated_rows(Path(candidate_reviews_path), set(by_review), per_bucket)
        unrelated_added = 0
        for row in unrelated:
            if positive_pattern.search(str(row["review_text"])) or negative_pattern.search(str(row["review_text"])):
                continue
            selected.append(
                {
                    "review_id": str(row["review_id"]),
                    "business_id": str(row["business_id"]),
                    "segment_id": None,
                    "user_id": str(row["user_id"]),
                    "review_time": row["review_time"].isoformat() if hasattr(row["review_time"], "isoformat") else str(row["review_time"]),
                    "stars": float(row["stars"]),
                    "useful": int(row["useful"]),
                    "review_text": str(row["review_text"])[:900],
                    "keyword_hit": False,
                    "semantic_hit": False,
                    "semantic_score": 0.0,
                    "matched_terms": [],
                    "matched_anchor_ids": [],
                    "positive_term_count": 0,
                    "negative_term_count": 0,
                    "selection_strength": 2,
                    "selection_relevance": 0,
                }
            )
            unrelated_added += 1
            if unrelated_added >= per_bucket:
                break

        # 生成模型真正要看的输入；外层追踪信息不会送给教师或学生模型。
        for index, row in enumerate(selected, 1):
            row["sample_id"] = f"{aspect}_{index:06d}"
            row["model_input"] = _make_input(
                template=template,
                aspect=aspect_by_id[aspect],
                review_text=row["review_text"],
            )
        selected.sort(key=lambda item: (int(item["selection_relevance"]), int(item["selection_strength"]), str(item["sample_id"])))
        output_path = candidates_dir / f"{aspect}.jsonl"
        output_path.write_text(
            "".join(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n" for item in selected),
            encoding="utf-8",
        )
        summary["aspects"][aspect] = {
            "candidate_count": len(selected),
            "strength_counts": {str(level): sum(item["selection_strength"] == level for item in selected) for level in range(5)},
            "relevance_counts": {str(level): sum(item["selection_relevance"] == level for item in selected) for level in range(4)},
            "source_relation_count": len(relation_rows),
            "unique_source_reviews": len(by_review),
            "unclear_added": unclear_added,
            "unrelated_added": unrelated_added,
            "fallback_bucket_count": sum(bool(item.get("selection_fallback")) for item in selected),
            # 清单随项目目录一起移动，因此这里只保存相对于本批数据根目录的路径。
            "output": output_path.relative_to(destination).as_posix(),
        }

    manifest_path = destination / "candidate_manifest.json"
    manifest_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="挑选14种软偏好的教师标注候选评论")
    parser.add_argument("--template-path", type=Path, required=True)
    parser.add_argument("--vocabulary-path", type=Path, required=True)
    parser.add_argument("--candidate-aspects-path", type=Path, required=True)
    parser.add_argument("--candidate-reviews-path", type=Path, required=True)
    parser.add_argument("--segments-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--per-bucket", type=int, default=5)
    return parser


def main() -> None:
    args = _parser().parse_args()
    summary = build_teacher_candidates(
        template_path=args.template_path,
        vocabulary_path=args.vocabulary_path,
        candidate_aspects_path=args.candidate_aspects_path,
        candidate_reviews_path=args.candidate_reviews_path,
        segments_path=args.segments_path,
        output_root=args.output_root,
        per_bucket=args.per_bucket,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
