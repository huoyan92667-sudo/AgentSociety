"""Atomic persistence for target-blind Query Recommendation Agent outputs."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import TypeVar

from yelp_agent.models import StrictModel

from .schema import (
    QueryRecommendationAgentPrediction,
    QueryRecommendationAgentRunManifest,
)
from .evaluation import QueryRecommendationAgentEvaluation


ModelT = TypeVar("ModelT", bound=StrictModel)


def load_predictions(
    path: str | Path,
) -> tuple[QueryRecommendationAgentPrediction, ...]:
    return _load_jsonl(Path(path), QueryRecommendationAgentPrediction)


def write_visible_run(
    predictions: tuple[QueryRecommendationAgentPrediction, ...],
    *,
    output_root: str | Path,
    visible_cases_path: str | Path,
) -> QueryRecommendationAgentRunManifest:
    if not predictions:
        raise ValueError("cannot write an empty Query Agent run")
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    predictions_path = output / "predictions.jsonl"
    content = "".join(
        item.model_dump_json(exclude_computed_fields=True) + "\n"
        for item in sorted(predictions, key=lambda value: value.case_id)
    )
    _atomic_write(predictions_path, content)
    manifest = QueryRecommendationAgentRunManifest(
        case_count=len(predictions),
        split_counts=dict(sorted(Counter(item.split for item in predictions).items())),
        agent_versions=sorted({item.agent_version for item in predictions}),
        visible_cases_sha256=_sha256(Path(visible_cases_path)),
        predictions_sha256=_sha256(predictions_path),
    )
    _atomic_write(output / "run_manifest.json", manifest.model_dump_json(indent=2) + "\n")
    return manifest


def verify_visible_run(output_root: str | Path) -> QueryRecommendationAgentRunManifest:
    output = Path(output_root)
    manifest = QueryRecommendationAgentRunManifest.model_validate_json(
        (output / "run_manifest.json").read_text(encoding="utf-8")
    )
    predictions = load_predictions(output / "predictions.jsonl")
    if len(predictions) != manifest.case_count:
        raise ValueError("prediction count does not match the run manifest")
    if _sha256(output / "predictions.jsonl") != manifest.predictions_sha256:
        raise ValueError("frozen prediction hash does not match the run manifest")
    return manifest


def write_evaluation(
    evaluation: QueryRecommendationAgentEvaluation,
    *,
    output_root: str | Path,
) -> tuple[Path, Path, Path]:
    """Publish hidden-label metrics separately from target-blind predictions."""

    output = Path(output_root)
    evaluation_root = output / "evaluation"
    evaluation_root.mkdir(parents=True, exist_ok=True)
    cases_path = evaluation_root / "end_to_end_cases.jsonl"
    audits_path = evaluation_root / "case_audits.jsonl"
    metrics_path = evaluation_root / "metrics.json"
    _atomic_write(
        cases_path,
        "".join(
            item.model_dump_json(exclude_computed_fields=True) + "\n"
            for item in evaluation.end_to_end_cases
        ),
    )
    _atomic_write(
        audits_path,
        "".join(
            item.model_dump_json(exclude_computed_fields=True) + "\n"
            for item in evaluation.case_audits
        ),
    )
    _atomic_write(metrics_path, evaluation.report.model_dump_json(indent=2) + "\n")
    _atomic_write(evaluation_root / "report.md", _markdown_report(evaluation))
    return metrics_path, cases_path, audits_path


def _load_jsonl(path: Path, model: type[ModelT]) -> tuple[ModelT, ...]:
    if not path.is_file():
        raise FileNotFoundError(f"JSONL artifact does not exist: {path}")
    return tuple(
        model.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _markdown_report(evaluation: QueryRecommendationAgentEvaluation) -> str:
    metrics = evaluation.report.metrics
    lines = [
        "# Query Recommendation Agent V1：完整 Agent 评测",
        "",
        f"- 样本数：{evaluation.report.case_count}",
        "- 运行边界：Agent 运行阶段只读取 visible cases；预测冻结后才加载 Ground Truth 与结构化条件。",
        "- HR/Recall 含义：唯一真实正例来自用户在 cutoff 后真实给出 4–5 星的商家。",
        "- 证据判断限制：当前 Query Recommendation V1 没有人工证据正确性标签，因此相关指标会明确显示 unavailable，而不会用别的 Benchmark 冒充。",
        "",
        "## 全部指标",
        "",
        "| 指标 | 状态 | 数值 | 分子 / 分母 |",
        "|---|---:|---:|---:|",
    ]
    for name, metric in metrics.items():
        value = "—" if metric.value is None else f"{metric.value:.6f}"
        lines.append(
            f"| {name} | {metric.status} | {value} | "
            f"{metric.numerator:g} / {metric.denominator:g} |"
        )
        if metric.reason:
            lines.append(f"| ↳ 原因 |  | {metric.reason} |  |")

    successes = sorted(
        (
            item
            for item in evaluation.case_audits
            if item.target_final_rank is not None and item.target_final_rank <= 5
        ),
        key=lambda item: (item.target_final_rank or 999, item.case_id),
    )[:3]
    failures = sorted(
        (
            item
            for item in evaluation.case_audits
            if item.target_final_rank is None or item.target_final_rank > 10
        ),
        key=lambda item: (
            item.target_retrieval_rank is None,
            item.target_retrieval_rank or 9999,
            item.case_id,
        ),
    )[:5]
    lines.extend(["", "## 真实成功案例", ""])
    lines.extend(_case_markdown(item) for item in successes)
    lines.extend(["", "## 真实失败案例", ""])
    lines.extend(_case_markdown(item) for item in failures)
    return "\n".join(lines) + "\n"


def _case_markdown(item: object) -> str:
    audit = item
    displayed = ", ".join(
        f"#{row['rank']} {row.get('business_name') or row['business_id']}"
        for row in audit.displayed_businesses
    ) or "未返回推荐"
    return (
        f"- `{audit.case_id}`：{audit.query_text}\n"
        f"  - 真实商家：{audit.target_business_name or audit.target_business_id}；"
        f"召回排名={audit.target_retrieval_rank or '未召回'}；"
        f"最终排名={audit.target_final_rank or '未进入最终排序'}。\n"
        f"  - Agent 状态：{audit.status}/{audit.response_kind}；"
        f"工具链：{' → '.join(audit.tool_sequence) or '无工具调用'}。\n"
        f"  - Top 结果：{displayed}；证据卡={audit.evidence_card_count}；"
        f"回退原因={audit.fallback_reason or '无'}。"
    )
