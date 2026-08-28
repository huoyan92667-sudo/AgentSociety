"""在同一批混合召回评论上，对比旧相似度差值和本地小模型两步判断。"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Literal

from yelp_agent.cross_encoder.config import (
    LocalCrossEncoderEnvironment,
    load_cross_encoder_config,
)
from yelp_agent.cross_encoder.local_reranker import LocalQwenCrossEncoder
from yelp_agent.recommendation_v2.review_evidence.cross_encoder_judge import (
    REVIEW_EVIDENCE_INSTRUCTION,
    CrossEncoderJudgmentConfig,
    CrossEncoderReviewEvidenceJudge,
)
from yelp_agent.recommendation_v2.review_evidence.schema import (
    PreferenceSearchDescription,
    ReviewSimilarityCandidate,
)

from .retrieval_experiment import (
    RetrievalExperimentConfig,
    _build_retriever,
    _load_labels,
)


type ComparisonLabel = Literal[
    "positive", "negative", "mixed", "ambiguous", "irrelevant"
]


@dataclass(frozen=True, slots=True)
class CrossEncoderExperimentConfig:
    """一次真实实验所需的数据、模型和结果位置。"""

    source_project_root: Path
    case_root: Path
    model_path: Path = Path(r"D:\model\Qwen3-Reranker-0.6B")
    python_executable: Path = Path(r"D:\anaconda3\python.exe")
    device: Literal["cuda", "cpu"] = "cuda"
    qdrant_url: str = "http://localhost:6333"

    @property
    def requirement_path(self) -> Path:
        return self.case_root / "audit/selected_production_description.json"

    @property
    def result_path(self) -> Path:
        return self.case_root / "audit/cross_encoder_judgment_experiment.json"

    @property
    def summary_path(self) -> Path:
        return self.case_root / "audit/cross_encoder_judgment_summary.md"


def run_cross_encoder_experiment(
    config: CrossEncoderExperimentConfig,
) -> dict[str, object]:
    """真实召回一次、判断一次，并把新旧准确性和耗时写入同一报告。"""

    report = json.loads((config.case_root / "report.json").read_text("utf-8"))
    business_ids = [
        str(item["business_id"]) for item in report["hard_filtered_businesses"]
    ]
    business_names = {
        str(item["business_id"]): str(item["name"])
        for item in report["hard_filtered_businesses"]
    }
    requirement = PreferenceSearchDescription.model_validate_json(
        config.requirement_path.read_text("utf-8")
    )
    labels = _load_labels(config.case_root / "hidden/all_review_labels.jsonl")
    truth = {item.review.review_id: item.label for item in labels}

    # 召回仍走当前正式配置，保证候选集合与旧判断完全相同。
    retrieval_config = RetrievalExperimentConfig(
        source_project_root=config.source_project_root,
        case_root=config.case_root,
        qdrant_url=config.qdrant_url,
        repeats=1,
        requirement_override_path=config.requirement_path,
    )
    retrieval_build_started = perf_counter()
    retriever = _build_retriever(retrieval_config, enable_bm25=True)
    retrieval_startup_ms = (perf_counter() - retrieval_build_started) * 1000
    try:
        retrieval_started = perf_counter()
        retrieved = retriever.retrieve_many([requirement], business_ids)
        retrieval_wall_ms = (perf_counter() - retrieval_started) * 1000
    finally:
        retriever.close()
    by_business = retrieved.by_requirement[requirement.requirement_id]
    candidates = _flatten_candidates(by_business, business_ids)

    # 复用旧系统已经稳定运行的本地进程接口，只替换成评论证据专用指令。
    base_config = load_cross_encoder_config(
        Path(__file__).resolve().parents[4] / "configs/cross_encoder.yaml"
    )
    model_config = base_config.model_copy(
        update={"instruction": REVIEW_EVIDENCE_INSTRUCTION}
    )
    model_started = perf_counter()
    scorer = LocalQwenCrossEncoder.from_environment(
        model_config,
        LocalCrossEncoderEnvironment(
            model_path=config.model_path,
            python_executable=config.python_executable,
            device=config.device,
        ),
    )
    model_startup_ms = (perf_counter() - model_started) * 1000
    try:
        judge = CrossEncoderReviewEvidenceJudge(
            scorer,
            config=CrossEncoderJudgmentConfig(
                relevance_threshold=0.50,
                support_threshold=0.50,
                direction_margin=0.10,
            ),
        )
        judged = judge.judge(requirement, candidates)
    finally:
        scorer.close()

    old_predictions = {item.review_id: item.direction for item in candidates}
    new_predictions = {item.review_id: item.label for item in judged.judgments}
    old_comparison = _compare_candidate_labels(truth, old_predictions)
    new_comparison = _compare_candidate_labels(truth, new_predictions)
    timing = {
        "retrieval_startup_ms": retrieval_startup_ms,
        "retrieval_wall_ms": retrieval_wall_ms,
        "cross_encoder_model_startup_ms": model_startup_ms,
        "cross_encoder_judgment_wall_ms": judged.metrics.wall_latency_ms,
        "old_warm_request_ms": retrieval_wall_ms,
        "new_warm_request_ms": retrieval_wall_ms + judged.metrics.wall_latency_ms,
        "old_cold_request_ms": retrieval_startup_ms + retrieval_wall_ms,
        "new_cold_request_ms": (
            retrieval_startup_ms
            + retrieval_wall_ms
            + model_startup_ms
            + judged.metrics.wall_latency_ms
        ),
        "new_warm_added_ms": judged.metrics.wall_latency_ms,
        "new_warm_added_ratio": _ratio(
            judged.metrics.wall_latency_ms, retrieval_wall_ms
        ),
    }

    details = []
    judgment_by_id = {item.review_id: item for item in judged.judgments}
    for candidate in candidates:
        reference = truth[candidate.review_id]
        judgment = judgment_by_id[candidate.review_id]
        details.append(
            {
                "review_id": candidate.review_id,
                "business_id": candidate.business_id,
                "business_name": business_names.get(candidate.business_id),
                "reference_label": reference.label,
                "reference_grade": reference.evidence_grade,
                "old_label": candidate.direction,
                "new_label": judgment.label,
                "relevance_score": judgment.relevance_score,
                "positive_support_score": judgment.positive_support_score,
                "negative_support_score": judgment.negative_support_score,
                "old_positive_similarity": candidate.positive_similarity,
                "old_negative_similarity": candidate.negative_similarity,
                "matched_segment_text": candidate.matched_segment_text,
            }
        )

    result: dict[str, object] = {
        "case_id": report["case_id"],
        "query_text": report["question"]["query_text"],
        "requirement": requirement.model_dump(mode="json"),
        "candidate_count": len(candidates),
        "thresholds": judge.config.model_dump(mode="json"),
        "timing": timing,
        "retrieval_metrics": retrieved.metrics.model_dump(mode="json"),
        "judgment_metrics": judged.metrics.model_dump(mode="json"),
        "old_method": old_comparison,
        "cross_encoder_method": new_comparison,
        "details": details,
    }
    config.result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    config.summary_path.write_text(_markdown_summary(result), encoding="utf-8")
    return result


def _flatten_candidates(
    by_business: dict[str, list[ReviewSimilarityCandidate]],
    business_ids: list[str],
) -> list[ReviewSimilarityCandidate]:
    """评论编号是去重依据；不把大段正文拿来做字符串去重。"""

    result: list[ReviewSimilarityCandidate] = []
    seen: set[str] = set()
    for business_id in business_ids:
        for candidate in by_business.get(business_id, []):
            if candidate.review_id in seen:
                continue
            seen.add(candidate.review_id)
            result.append(candidate)
    return result


def _compare_candidate_labels(
    truth: dict[str, object],
    predictions: dict[str, ComparisonLabel],
) -> dict[str, object]:
    """只比较两种方法共同面对的候选评论，不把召回率混入方向判断。"""

    gold = {review_id: truth[review_id].label for review_id in predictions}
    exact = sum(predictions[review_id] == label for review_id, label in gold.items())
    gold_relevant = {key for key, value in gold.items() if value != "irrelevant"}
    predicted_relevant = {
        key for key, value in predictions.items() if value != "irrelevant"
    }
    true_positive = len(gold_relevant & predicted_relevant)
    false_positive = len(predicted_relevant - gold_relevant)
    false_negative = len(gold_relevant - predicted_relevant)
    precision = _ratio(true_positive, true_positive + false_positive)
    recall = _ratio(true_positive, true_positive + false_negative)
    f1 = (
        None
        if precision is None or recall is None or precision + recall == 0
        else 2 * precision * recall / (precision + recall)
    )
    directional_ids = {
        key for key, value in gold.items() if value in {"positive", "negative", "mixed"}
    }
    directional_exact = sum(
        predictions[key] == gold[key] for key in directional_ids
    )
    confusion: dict[str, Counter[str]] = {}
    for review_id, gold_label in gold.items():
        confusion.setdefault(gold_label, Counter())[predictions[review_id]] += 1
    return {
        "candidate_count": len(predictions),
        "reference_counts": dict(sorted(Counter(gold.values()).items())),
        "predicted_counts": dict(sorted(Counter(predictions.values()).items())),
        "exact_match_count": exact,
        "exact_match_rate": _ratio(exact, len(predictions)),
        "relevance_precision": precision,
        "relevance_recall": recall,
        "relevance_f1": f1,
        "directional_reference_count": len(directional_ids),
        "directional_exact_count": directional_exact,
        "directional_exact_rate": _ratio(directional_exact, len(directional_ids)),
        "confusion": {
            key: dict(sorted(value.items())) for key, value in sorted(confusion.items())
        },
    }


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _markdown_summary(result: dict[str, object]) -> str:
    timing = result["timing"]
    old = result["old_method"]
    new = result["cross_encoder_method"]
    usage = result["judgment_metrics"]
    return f"""# 本地小模型评论方向实验

问题：{result['query_text']}

候选评论：{result['candidate_count']} 条（新旧方法使用完全相同的候选）。

## 判断效果

| 指标 | 旧相似度差值 | 两步小模型 |
|---|---:|---:|
| 五类完全一致 | {old['exact_match_count']} / {old['candidate_count']} ({old['exact_match_rate']:.2%}) | {new['exact_match_count']} / {new['candidate_count']} ({new['exact_match_rate']:.2%}) |
| 正反/混合方向完全一致 | {old['directional_exact_count']} / {old['directional_reference_count']} ({old['directional_exact_rate']:.2%}) | {new['directional_exact_count']} / {new['directional_reference_count']} ({new['directional_exact_rate']:.2%}) |
| 相关评论查准率 | {old['relevance_precision']:.2%} | {new['relevance_precision']:.2%} |
| 相关评论查全率 | {old['relevance_recall']:.2%} | {new['relevance_recall']:.2%} |

## 耗时

| 阶段 | 毫秒 |
|---|---:|
| 当前混合召回 | {timing['retrieval_wall_ms']:.1f} |
| 本地小模型首次加载（常驻后不重复） | {timing['cross_encoder_model_startup_ms']:.1f} |
| 两步判断 | {timing['cross_encoder_judgment_wall_ms']:.1f} |
| 旧方法热运行合计 | {timing['old_warm_request_ms']:.1f} |
| 新方法热运行合计 | {timing['new_warm_request_ms']:.1f} |
| 旧方法冷启动合计 | {timing['old_cold_request_ms']:.1f} |
| 新方法冷启动合计 | {timing['new_cold_request_ms']:.1f} |

本地小模型实际判断 {usage['scored_pair_count']} 对文本，分 {usage['scorer_call_count']} 批，输入 {usage['input_tokens']} 个模型词元；截断 {usage['truncated_pair_count']} 对。

注意：正确答案来自此前 GLM-5.1 对 893 条完整评论的结构化标注，不是人工金标准。本次 0.50/0.10 门槛没有用正确答案反向调参，因此结果属于一次真实试跑，不是最终结论。
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-project-root",
        type=Path,
        default=Path(r"C:\Users\29072\PycharmProjects\AgentSociety"),
    )
    parser.add_argument("--case-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=Path(r"D:\model\Qwen3-Reranker-0.6B"))
    parser.add_argument("--python-executable", type=Path, default=Path(r"D:\anaconda3\python.exe"))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    args = parser.parse_args()
    result = run_cross_encoder_experiment(
        CrossEncoderExperimentConfig(
            source_project_root=args.source_project_root.resolve(),
            case_root=args.case_root.resolve(),
            model_path=args.model_path.resolve(),
            python_executable=args.python_executable.resolve(),
            device=args.device,
            qdrant_url=args.qdrant_url,
        )
    )
    # 控制台只返回便于人查看的小摘要；90条逐条分数已经写入结果文件。
    print(
        json.dumps(
            {
                "candidate_count": result["candidate_count"],
                "timing": result["timing"],
                "old_method": result["old_method"],
                "cross_encoder_method": result["cross_encoder_method"],
                "result_path": str(args.case_root / "audit/cross_encoder_judgment_experiment.json"),
            },
            ensure_ascii=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
