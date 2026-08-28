"""在同一批评论片段上并排比较三种证据方向判断方式。"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from time import perf_counter

from yelp_agent.recommendation_v2.rag_benchmark.schema import LabeledReview
from yelp_agent.recommendation_v2.review_evidence.qwen_five_class import (
    LocalQwenFiveClassClassifier,
    QwenEvidenceInput,
    QwenFiveClassEvidenceJudge,
)
from yelp_agent.recommendation_v2.review_evidence.schema import (
    PreferenceSearchDescription,
)


def run_qwen_five_class_experiment(
    *,
    source_cross_encoder_result: Path,
    source_labels: Path,
    output_root: Path,
    model_path: Path,
    python_executable: Path,
    batch_size: int = 4,
    max_sequence_length: int = 768,
) -> dict[str, object]:
    """复用前两版的90条候选，只新增第三版并记录冷、热两次耗时。"""

    source = json.loads(source_cross_encoder_result.read_text(encoding="utf-8"))
    details = source["details"]
    requirement = PreferenceSearchDescription.model_validate(source["requirement"])
    labels = _load_labels(source_labels)
    reference_by_review = {item.review.review_id: item for item in labels}
    candidates = [
        QwenEvidenceInput(
            review_id=str(item["review_id"]),
            business_id=str(item["business_id"]),
            segment_text=str(item["matched_segment_text"]),
        )
        for item in details
    ]
    # 三个版本都判断命中的片段，因此正确答案也必须确认原文证据真的落在片段里。
    segment_truth = {
        item.review_id: _segment_reference_label(
            reference_by_review[item.review_id], item.segment_text
        )
        for item in candidates
    }
    version_one = {
        str(item["review_id"]): str(item["old_label"]) for item in details
    }
    version_two = {
        str(item["review_id"]): str(item["new_label"]) for item in details
    }

    model_started = perf_counter()
    classifier = LocalQwenFiveClassClassifier.start(
        model_path=model_path,
        python_executable=python_executable,
        batch_size=batch_size,
        max_sequence_length=max_sequence_length,
    )
    model_startup_ms = (perf_counter() - model_started) * 1000
    try:
        judge = QwenFiveClassEvidenceJudge(classifier)
        # 第一次包含CUDA首次执行的额外开销；第二次表示模型常驻后的稳定请求。
        first = judge.judge(requirement, candidates)
        steady = judge.judge(requirement, candidates)
    finally:
        classifier.close()
    first_predictions = {item.review_id: item.label for item in first.judgments}
    steady_predictions = {item.review_id: item.label for item in steady.judgments}
    deterministic = first_predictions == steady_predictions

    retrieval_timing = source["timing"]
    retrieval_startup_ms = float(retrieval_timing["retrieval_startup_ms"])
    retrieval_wall_ms = float(retrieval_timing["retrieval_wall_ms"])
    comparisons = {
        "version_1_similarity_gap": _compare(segment_truth, version_one),
        "version_2_cross_encoder": _compare(segment_truth, version_two),
        "version_3_qwen_five_class": _compare(segment_truth, steady_predictions),
    }
    timing = {
        "shared_retrieval_startup_ms": retrieval_startup_ms,
        "shared_retrieval_wall_ms": retrieval_wall_ms,
        "version_1_judgment_ms": 0.0,
        "version_1_warm_total_ms": retrieval_wall_ms,
        "version_1_cold_total_ms": retrieval_startup_ms + retrieval_wall_ms,
        "version_2_model_startup_ms": float(
            retrieval_timing["cross_encoder_model_startup_ms"]
        ),
        "version_2_judgment_ms": float(
            retrieval_timing["cross_encoder_judgment_wall_ms"]
        ),
        "version_2_warm_total_ms": float(retrieval_timing["new_warm_request_ms"]),
        "version_2_cold_total_ms": float(retrieval_timing["new_cold_request_ms"]),
        "version_3_model_startup_ms": model_startup_ms,
        "version_3_first_judgment_ms": first.metrics.wall_latency_ms,
        "version_3_steady_judgment_ms": steady.metrics.wall_latency_ms,
        "version_3_warm_total_ms": retrieval_wall_ms + steady.metrics.wall_latency_ms,
        "version_3_cold_total_ms": (
            retrieval_startup_ms
            + retrieval_wall_ms
            + model_startup_ms
            + first.metrics.wall_latency_ms
        ),
    }
    rows = []
    for item in details:
        review_id = str(item["review_id"])
        rows.append(
            {
                "review_id": review_id,
                "business_id": item["business_id"],
                "business_name": item.get("business_name"),
                "segment_reference_label": segment_truth[review_id],
                "full_review_reference_label": item["reference_label"],
                "version_1_label": version_one[review_id],
                "version_2_label": version_two[review_id],
                "version_3_label": steady_predictions[review_id],
                "matched_segment_text": item["matched_segment_text"],
            }
        )
    result: dict[str, object] = {
        "experiment": "three_version_review_evidence_direction_comparison",
        "query_text": source["query_text"],
        "candidate_count": len(candidates),
        "reference_policy": (
            "以GLM保存的逐字证据位置为准；证据不在命中片段中时，片段标签记为irrelevant"
        ),
        "reference_counts": dict(sorted(Counter(segment_truth.values()).items())),
        "qwen_model": classifier.model_name,
        "qwen_output_policy": "只允许A-E五选一，正式结果只保存唯一标签",
        "qwen_first_and_steady_labels_identical": deterministic,
        "qwen_first_metrics": first.metrics.model_dump(mode="json"),
        "qwen_steady_metrics": steady.metrics.model_dump(mode="json"),
        "timing": timing,
        "comparisons": comparisons,
        "details": rows,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "three_version_comparison.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_root / "three_version_comparison.md").write_text(
        _summary_markdown(result), encoding="utf-8"
    )
    return result


def _segment_reference_label(item: LabeledReview, segment_text: str) -> str:
    """完整评论的证据没落进当前片段，就不能用来奖励片段分类器。"""

    label = item.label
    positive = any(span in segment_text for span in label.positive_spans)
    negative = any(span in segment_text for span in label.negative_spans)
    ambiguous = bool(
        label.ambiguous_span and label.ambiguous_span in segment_text
    )
    if positive and negative:
        return "mixed"
    if positive:
        return "positive"
    if negative:
        return "negative"
    if ambiguous:
        return "ambiguous"
    return "irrelevant"


def _compare(
    truth: dict[str, str], predictions: dict[str, str]
) -> dict[str, object]:
    if set(truth) != set(predictions):
        raise ValueError("comparison predictions do not cover the same reviews")
    labels = ("irrelevant", "positive", "negative", "mixed", "ambiguous")
    exact = sum(truth[key] == predictions[key] for key in truth)
    per_class = {}
    confusion = {}
    for label in labels:
        ids = [key for key, value in truth.items() if value == label]
        correct = sum(predictions[key] == label for key in ids)
        per_class[label] = {
            "reference_count": len(ids),
            "correct_count": correct,
            "recall": _ratio(correct, len(ids)),
        }
        confusion[label] = dict(
            sorted(Counter(predictions[key] for key in ids).items())
        )
    gold_relevant = {key for key, value in truth.items() if value != "irrelevant"}
    predicted_relevant = {
        key for key, value in predictions.items() if value != "irrelevant"
    }
    relevant_correct = len(gold_relevant & predicted_relevant)
    return {
        "predicted_counts": dict(sorted(Counter(predictions.values()).items())),
        "exact_count": exact,
        "exact_rate": _ratio(exact, len(truth)),
        "relevance_precision": _ratio(
            relevant_correct, len(predicted_relevant)
        ),
        "relevance_recall": _ratio(relevant_correct, len(gold_relevant)),
        "per_class": per_class,
        "confusion": confusion,
    }


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _percent(value: float | None) -> str:
    return "—" if value is None else f"{value:.2%}"


def _summary_markdown(result: dict[str, object]) -> str:
    comparisons = result["comparisons"]
    timing = result["timing"]
    names = (
        ("第一版：正反相似度差值", "version_1_similarity_gap"),
        ("第二版：交叉判断模型", "version_2_cross_encoder"),
        ("第三版：Qwen五分类", "version_3_qwen_five_class"),
    )
    rows = []
    for display, key in names:
        item = comparisons[key]
        rows.append(
            f"| {display} | {item['exact_count']}/{result['candidate_count']} "
            f"({_percent(item['exact_rate'])}) | {_percent(item['relevance_precision'])} "
            f"| {_percent(item['relevance_recall'])} |"
        )
    time_rows = [
        (
            "第一版：正反相似度差值",
            timing["version_1_judgment_ms"],
            timing["version_1_warm_total_ms"],
            timing["version_1_cold_total_ms"],
        ),
        (
            "第二版：交叉判断模型",
            timing["version_2_judgment_ms"],
            timing["version_2_warm_total_ms"],
            timing["version_2_cold_total_ms"],
        ),
        (
            "第三版：Qwen五分类",
            timing["version_3_steady_judgment_ms"],
            timing["version_3_warm_total_ms"],
            timing["version_3_cold_total_ms"],
        ),
    ]
    time_text = "\n".join(
        f"| {name} | {judge:.1f} | {warm:.1f} | {cold:.1f} |"
        for name, judge, warm, cold in time_rows
    )
    class_lines = []
    for display, key in names:
        classes = comparisons[key]["per_class"]
        class_lines.append(
            f"| {display} | "
            + " | ".join(
                f"{classes[label]['correct_count']}/{classes[label]['reference_count']}"
                for label in ("irrelevant", "positive", "negative", "mixed", "ambiguous")
            )
            + " |"
        )
    return f"""# 三种评论证据判断方法真实对比

问题：{result['query_text']}

三种方法使用完全相同的 {result['candidate_count']} 条评论片段。正确答案按证据是否真实出现在命中片段中重新计算：{result['reference_policy']}。

## 总体结果

| 方法 | 五类完全正确 | 相关证据查准率 | 相关证据查全率 |
|---|---:|---:|---:|
{chr(10).join(rows)}

## 各类判断正确数量

| 方法 | 无关 | 正面 | 反面 | 正反都有 | 无法判断 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(class_lines)}

## 耗时（毫秒）

| 方法 | 仅判断 | 模型常驻后的整轮 | 完全冷启动整轮 |
|---|---:|---:|---:|
{time_text}

共同的混合召回耗时为 {timing['shared_retrieval_wall_ms']:.1f} 毫秒。第三版模型首次加载耗时为 {timing['version_3_model_startup_ms']:.1f} 毫秒；第一次90条判断为 {timing['version_3_first_judgment_ms']:.1f} 毫秒，模型稳定后的第二次判断为 {timing['version_3_steady_judgment_ms']:.1f} 毫秒。两次标签完全一致：{result['qwen_first_and_steady_labels_identical']}。

第三版没有覆盖前两版，也没有接入正式排序。其正式判断结果只保存五类之一，不保存模型概率。
"""


def _load_labels(path: Path) -> list[LabeledReview]:
    return [
        LabeledReview.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-cross-result", type=Path, required=True)
    parser.add_argument("--source-labels", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path(r"D:\model\Qwen2.5-3B-Instruct"),
    )
    parser.add_argument(
        "--python-executable",
        type=Path,
        default=Path(r"D:\anaconda3\python.exe"),
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-sequence-length", type=int, default=768)
    args = parser.parse_args()
    result = run_qwen_five_class_experiment(
        source_cross_encoder_result=args.source_cross_result.resolve(),
        source_labels=args.source_labels.resolve(),
        output_root=args.output_root.resolve(),
        model_path=args.model_path.resolve(),
        python_executable=args.python_executable.resolve(),
        batch_size=args.batch_size,
        max_sequence_length=args.max_sequence_length,
    )
    # 逐评论明细已写入文件，控制台只打印便于查看的小摘要。
    print(
        json.dumps(
            {
                "candidate_count": result["candidate_count"],
                "reference_counts": result["reference_counts"],
                "timing": result["timing"],
                "comparisons": result["comparisons"],
                "output_root": str(args.output_root.resolve()),
            },
            ensure_ascii=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
