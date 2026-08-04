"""Validation-only diagnosis for the frozen Hybrid V1 ranker."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import os
from pathlib import Path
from typing import Literal, Protocol, Sequence

import duckdb
import numpy as np
from pydantic import Field, ValidationError

from yelp_agent.config import EvaluationDataUsageConfig
from yelp_agent.evaluation.data_usage import (
    DataUsageViolation,
    assign_development_task_folds,
    deterministic_bootstrap_users,
)
from yelp_agent.features.hybrid import (
    HybridComponentFeatures,
    HybridWeights,
)
from yelp_agent.features.location import LocationTaskFeatures
from yelp_agent.features.quality import BusinessQuality
from yelp_agent.models import (
    RecommendationTask,
    ScoreBreakdown,
    StrictModel,
)
from yelp_agent.rankers.hybrid_ranker import HybridRanker


ComponentName = Literal["category", "text", "quality", "location", "none"]
SUPPORTED_SEGMENT_DIMENSIONS = {
    "history_size",
    "target_category_seen",
    "candidate_refill",
    "hybrid_margin",
    "feature_disagreement",
    "missing_location",
    "candidate_popularity",
    "negative_sampling_frequency",
}


class HybridDiagnosisError(RuntimeError):
    """Raised when a safe validation-only diagnosis cannot be produced."""


class DiagnosticMetrics(StrictModel):
    hr_at_1: float = Field(ge=0, le=1)
    hr_at_3: float = Field(ge=0, le=1)
    hr_at_5: float = Field(ge=0, le=1)
    avg_hr: float = Field(ge=0, le=1)
    mrr: float = Field(ge=0, le=1)
    ndcg_at_5: float = Field(ge=0, le=1)
    mean_target_rank: float = Field(ge=1, le=20)


class DiagnosticDispersion(StrictModel):
    hr_at_1: float = Field(ge=0, le=1)
    hr_at_3: float = Field(ge=0, le=1)
    hr_at_5: float = Field(ge=0, le=1)
    avg_hr: float = Field(ge=0, le=1)
    mrr: float = Field(ge=0, le=1)
    ndcg_at_5: float = Field(ge=0, le=1)
    mean_target_rank: float = Field(ge=0, le=20)


class BootstrapInterval(StrictModel):
    lower: float
    upper: float


class SegmentDiagnostic(StrictModel):
    task_count: int = Field(gt=0)
    metrics: DiagnosticMetrics


class HybridTaskDiagnostic(StrictModel):
    task_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    fold: int = Field(ge=1)
    target_rank: int = Field(ge=1, le=20)
    history_count: int = Field(ge=0)
    target_category_seen: bool
    candidate_refill: bool
    hybrid_margin: float = Field(ge=0, le=1)
    feature_disagreement_span: int = Field(ge=0, le=19)
    missing_location: bool
    target_popularity: float = Field(ge=0, le=1)
    target_review_count: int = Field(ge=0)
    max_negative_sampling_frequency: int = Field(ge=0)
    target_scores: ScoreBreakdown
    top_scores: ScoreBreakdown
    target_component_ranks: dict[str, int]
    primary_disadvantage: ComponentName
    segment_values: dict[str, str]


class HybridDiagnosisSummary(StrictModel):
    format_version: int = Field(ge=1)
    diagnosis_name: str
    development_split: Literal["validation"]
    legacy_test_loaded: Literal[False]
    strict_blind_holdout: Literal[False]
    task_count: int = Field(gt=0)
    user_count: int = Field(gt=0)
    hybrid_weights: HybridWeights
    zero_weight_components: list[str]
    overall: DiagnosticMetrics
    target_rank_counts: dict[str, int]
    fold_task_counts: dict[str, int]
    fold_metrics: dict[str, DiagnosticMetrics]
    fold_mean: DiagnosticMetrics
    fold_standard_deviation: DiagnosticDispersion
    bootstrap_samples: int = Field(ge=100)
    bootstrap_confidence_level: float = Field(gt=0, lt=1)
    bootstrap_intervals: dict[str, BootstrapInterval]
    segment_metrics: dict[str, dict[str, SegmentDiagnostic]]
    failed_top_5_count: int = Field(ge=0)
    failed_top_5_primary_disadvantage: dict[str, int]
    task_diagnostics_include_identifiers: Literal[True]
    published_summary_includes_identifiers: Literal[False]
    source_sha256: dict[str, str]
    limitations: list[str]


class _ComponentProvider(Protocol):
    def features_for(
        self,
        task: RecommendationTask,
    ) -> HybridComponentFeatures: ...


class _QualityProvider(Protocol):
    def score_businesses(
        self,
        business_ids: list[str],
        cutoff_time: object,
    ) -> dict[str, BusinessQuality]: ...


class _LocationProvider(Protocol):
    def features_for(
        self,
        task: RecommendationTask,
    ) -> LocationTaskFeatures: ...


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_validation_tasks(path: Path) -> list[RecommendationTask]:
    if not path.is_file():
        raise FileNotFoundError(f"Validation task JSONL does not exist: {path}")
    tasks: list[RecommendationTask] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                task = RecommendationTask.model_validate_json(line)
            except ValidationError as exc:
                raise HybridDiagnosisError(
                    f"Invalid validation task at line {line_number}"
                ) from exc
            if not task.task_id.startswith("validation:"):
                raise DataUsageViolation(
                    "Hybrid V1 diagnosis accepts validation tasks only"
                )
            if task.task_id in seen:
                raise HybridDiagnosisError(
                    f"Duplicate validation task_id: {task.task_id!r}"
                )
            seen.add(task.task_id)
            tasks.append(task)
    if not tasks:
        raise HybridDiagnosisError("Validation task JSONL contains no tasks")
    return tasks


def _load_validation_truth(
    path: Path,
    tasks: Sequence[RecommendationTask],
) -> tuple[dict[str, str], str]:
    if not path.is_file():
        raise FileNotFoundError(f"Ground truth Parquet does not exist: {path}")
    expected = {task.task_id: task for task in tasks}
    try:
        with duckdb.connect() as connection:
            rows = connection.execute(
                """
                SELECT task_id, target_business_id
                FROM read_parquet(?)
                WHERE starts_with(task_id, 'validation:')
                ORDER BY task_id
                """,
                [str(path)],
            ).fetchall()
    except duckdb.Error as exc:
        raise HybridDiagnosisError(
            f"Could not read validation ground truth: {path}"
        ) from exc

    truth: dict[str, str] = {}
    for raw_task_id, raw_target in rows:
        task_id = str(raw_task_id)
        if task_id not in expected:
            raise HybridDiagnosisError(
                f"Unexpected validation ground truth task: {task_id!r}"
            )
        target = str(raw_target)
        if not target or target not in expected[task_id].candidate_business_ids:
            raise HybridDiagnosisError(
                f"Invalid validation target for task {task_id!r}"
            )
        if task_id in truth:
            raise HybridDiagnosisError(
                f"Duplicate validation ground truth for {task_id!r}"
            )
        truth[task_id] = target
    missing = sorted(set(expected).difference(truth))
    if missing:
        raise HybridDiagnosisError(
            f"Validation ground truth is missing tasks: {missing[:3]}"
        )

    digest = hashlib.sha256()
    for task_id, target in sorted(truth.items()):
        digest.update(task_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(target.encode("utf-8"))
        digest.update(b"\n")
    return truth, digest.hexdigest()


def _load_validation_provenance(
    path: Path,
    tasks: Sequence[RecommendationTask],
    truth: dict[str, str],
) -> tuple[dict[str, bool], dict[str, int], str]:
    if not path.is_file():
        raise FileNotFoundError(f"Candidate provenance does not exist: {path}")
    expected = {task.task_id: task for task in tasks}
    try:
        with duckdb.connect() as connection:
            rows = connection.execute(
                """
                SELECT task_id, business_id, source_bucket, final_position
                FROM read_parquet(?)
                WHERE starts_with(task_id, 'validation:')
                ORDER BY task_id, final_position
                """,
                [str(path)],
            ).fetchall()
    except duckdb.Error as exc:
        raise HybridDiagnosisError(
            f"Could not read validation candidate provenance: {path}"
        ) from exc

    grouped: defaultdict[str, list[tuple[str, str, int]]] = defaultdict(list)
    negative_frequency: Counter[str] = Counter()
    digest = hashlib.sha256()
    for raw_task_id, raw_business_id, raw_bucket, raw_position in rows:
        task_id = str(raw_task_id)
        if task_id not in expected:
            raise HybridDiagnosisError(
                f"Unexpected validation provenance task: {task_id!r}"
            )
        business_id = str(raw_business_id)
        bucket = str(raw_bucket)
        position = int(raw_position)
        grouped[task_id].append((business_id, bucket, position))
        if business_id != truth[task_id]:
            negative_frequency[business_id] += 1
        digest.update(task_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(business_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bucket.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(position).encode("ascii"))
        digest.update(b"\n")

    refill: dict[str, bool] = {}
    for task_id, task in expected.items():
        task_rows = grouped.get(task_id, [])
        business_ids = [row[0] for row in task_rows]
        positions = [row[2] for row in task_rows]
        target_rows = [row for row in task_rows if row[1] == "target"]
        if (
            len(task_rows) != 20
            or set(business_ids) != set(task.candidate_business_ids)
            or positions != list(range(1, 21))
            or len(target_rows) != 1
            or target_rows[0][0] != truth[task_id]
        ):
            raise HybridDiagnosisError(
                f"Candidate provenance is inconsistent for {task_id!r}"
            )
        refill[task_id] = any(
            bucket.startswith("refill_") for _, bucket, _ in task_rows
        )
    return refill, dict(negative_frequency), digest.hexdigest()


def _component_value(breakdown: ScoreBreakdown, name: str) -> float:
    return float(getattr(breakdown, f"{name}_score"))


def _component_ranks(
    task: RecommendationTask,
    target: str,
    scores: dict[str, ScoreBreakdown],
) -> dict[str, int]:
    ranks: dict[str, int] = {}
    for component in ("category", "text", "quality", "location"):
        ranking = sorted(
            task.candidate_business_ids,
            key=lambda business_id: (
                -_component_value(scores[business_id], component),
                business_id,
            ),
        )
        ranks[component] = ranking.index(target) + 1
    return ranks


def _history_segment(count: int) -> str:
    if count <= 10:
        return "small_10_or_less"
    if count <= 30:
        return "medium_11_to_30"
    return "rich_over_30"


def _margin_segment(margin: float) -> str:
    if margin <= 0.01:
        return "tight_0_to_0.01"
    if margin <= 0.05:
        return "moderate_over_0.01_to_0.05"
    return "clear_over_0.05"


def _disagreement_segment(span: int) -> str:
    if span <= 3:
        return "aligned_span_0_to_3"
    if span <= 9:
        return "mixed_span_4_to_9"
    return "conflicted_span_10_to_19"


def _popularity_segment(popularity: float) -> str:
    if popularity < 0.5:
        return "low_below_0.5"
    if popularity < 0.8:
        return "medium_0.5_to_below_0.8"
    return "high_0.8_to_1.0"


def _frequency_segment(frequency: int) -> str:
    if frequency <= 50:
        return "low_50_or_less"
    if frequency <= 200:
        return "medium_51_to_200"
    return "high_over_200"


def _primary_disadvantage(
    target: ScoreBreakdown,
    top: ScoreBreakdown,
    weights: HybridWeights,
) -> ComponentName:
    gaps = {
        component: weights.as_dict[component]
        * (
            _component_value(top, component)
            - _component_value(target, component)
        )
        for component in ("category", "text", "quality", "location")
    }
    component, gap = max(gaps.items(), key=lambda item: (item[1], item[0]))
    return component if gap > 1e-12 else "none"


def _metrics(rows: Sequence[HybridTaskDiagnostic]) -> DiagnosticMetrics:
    if not rows:
        raise ValueError("At least one task diagnostic is required")
    ranks = np.asarray([row.target_rank for row in rows], dtype=np.float64)
    hr_at_1 = float(np.mean(ranks <= 1))
    hr_at_3 = float(np.mean(ranks <= 3))
    hr_at_5 = float(np.mean(ranks <= 5))
    return DiagnosticMetrics(
        hr_at_1=hr_at_1,
        hr_at_3=hr_at_3,
        hr_at_5=hr_at_5,
        avg_hr=(hr_at_1 + hr_at_3 + hr_at_5) / 3.0,
        mrr=float(np.mean(1.0 / ranks)),
        ndcg_at_5=float(
            np.mean(np.where(ranks <= 5, 1.0 / np.log2(ranks + 1), 0.0))
        ),
        mean_target_rank=float(np.mean(ranks)),
    )


def _mean_metric_models(
    metrics: Sequence[DiagnosticMetrics],
) -> DiagnosticMetrics:
    if not metrics:
        raise ValueError("At least one metric model is required")
    payload: dict[str, float] = {}
    for field_name in DiagnosticMetrics.model_fields:
        values = np.asarray(
            [float(getattr(metric, field_name)) for metric in metrics]
        )
        payload[field_name] = float(np.mean(values))
    return DiagnosticMetrics.model_validate(payload)


def _standard_deviation_metric_models(
    metrics: Sequence[DiagnosticMetrics],
) -> DiagnosticDispersion:
    if not metrics:
        raise ValueError("At least one metric model is required")
    return DiagnosticDispersion.model_validate(
        {
            field_name: float(
                np.std(
                    [
                        float(getattr(metric, field_name))
                        for metric in metrics
                    ]
                )
            )
            for field_name in DiagnosticMetrics.model_fields
        }
    )


def _bootstrap_intervals(
    rows: Sequence[HybridTaskDiagnostic],
    policy: EvaluationDataUsageConfig,
) -> dict[str, BootstrapInterval]:
    grouped: defaultdict[str, list[HybridTaskDiagnostic]] = defaultdict(list)
    for row in rows:
        grouped[row.user_id].append(row)
    users = sorted(grouped)
    user_counts = np.asarray([len(grouped[user]) for user in users])
    user_sums = np.asarray(
        [
            [
                sum(item.target_rank <= 1 for item in grouped[user]),
                sum(item.target_rank <= 3 for item in grouped[user]),
                sum(item.target_rank <= 5 for item in grouped[user]),
                sum(1.0 / item.target_rank for item in grouped[user]),
                sum(
                    1.0 / np.log2(item.target_rank + 1)
                    if item.target_rank <= 5
                    else 0.0
                    for item in grouped[user]
                ),
                sum(item.target_rank for item in grouped[user]),
            ]
            for user in users
        ],
        dtype=np.float64,
    )
    user_index = {user: index for index, user in enumerate(users)}
    samples = np.empty((policy.bootstrap.samples, 7), dtype=np.float64)
    for replicate in range(policy.bootstrap.samples):
        sampled_users = deterministic_bootstrap_users(
            users,
            replicate_index=replicate,
            policy=policy,
        )
        indices = np.fromiter(
            (user_index[user] for user in sampled_users),
            dtype=np.int64,
            count=len(sampled_users),
        )
        task_count = int(np.sum(user_counts[indices]))
        totals = np.sum(user_sums[indices], axis=0)
        hr_at_1, hr_at_3, hr_at_5, mrr, ndcg, mean_rank = (
            totals / task_count
        )
        samples[replicate] = (
            hr_at_1,
            hr_at_3,
            hr_at_5,
            (hr_at_1 + hr_at_3 + hr_at_5) / 3.0,
            mrr,
            ndcg,
            mean_rank,
        )
    metric_names = list(DiagnosticMetrics.model_fields)
    alpha = (1.0 - policy.bootstrap.confidence_level) / 2.0
    lower_percentile = 100.0 * alpha
    upper_percentile = 100.0 * (1.0 - alpha)
    return {
        name: BootstrapInterval(
            lower=float(np.percentile(samples[:, index], lower_percentile)),
            upper=float(np.percentile(samples[:, index], upper_percentile)),
        )
        for index, name in enumerate(metric_names)
    }


def _task_diagnostics_payload(rows: Sequence[HybridTaskDiagnostic]) -> str:
    return "".join(row.model_dump_json() + "\n" for row in rows)


def diagnose_hybrid_v1_validation(
    *,
    validation_tasks_path: str | Path,
    ground_truth_path: str | Path,
    provenance_path: str | Path,
    feature_store: _ComponentProvider,
    quality_store: _QualityProvider,
    location_store: _LocationProvider,
    weights: HybridWeights,
    policy: EvaluationDataUsageConfig,
) -> tuple[list[HybridTaskDiagnostic], HybridDiagnosisSummary]:
    """Diagnose frozen Hybrid V1 without loading any Legacy Test task."""

    tasks_path = Path(validation_tasks_path)
    truth_path = Path(ground_truth_path)
    candidate_provenance_path = Path(provenance_path)
    tasks = _load_validation_tasks(tasks_path)
    task_folds = assign_development_task_folds(tasks, policy)
    unsupported_dimensions = set(policy.segment_dimensions).difference(
        SUPPORTED_SEGMENT_DIMENSIONS
    )
    if unsupported_dimensions:
        raise HybridDiagnosisError(
            "Unsupported diagnosis segment dimensions: "
            f"{sorted(unsupported_dimensions)}"
        )
    truth, truth_digest = _load_validation_truth(truth_path, tasks)
    refill, negative_frequency, provenance_digest = (
        _load_validation_provenance(
            candidate_provenance_path,
            tasks,
            truth,
        )
    )
    ranker = HybridRanker(feature_store, weights)
    diagnostics: list[HybridTaskDiagnostic] = []
    for task in tasks:
        target = truth[task.task_id]
        scored = ranker.score(task)
        scores = scored.score_breakdowns
        if set(scores) != set(task.candidate_business_ids):
            raise HybridDiagnosisError(
                f"Hybrid scores do not match candidates for {task.task_id!r}"
            )
        ranking = sorted(
            task.candidate_business_ids,
            key=lambda business_id: (
                -scores[business_id].hybrid_score,
                business_id,
            ),
        )
        target_rank = ranking.index(target) + 1
        top_business = ranking[0]
        margin = (
            scores[ranking[0]].hybrid_score
            - scores[ranking[1]].hybrid_score
        )
        component_ranks = _component_ranks(task, target, scores)
        quality = quality_store.score_businesses(
            task.candidate_business_ids,
            task.cutoff_time,
        )[target]
        location = location_store.features_for(task)
        target_location = location.business_scores[target]
        max_frequency = max(
            negative_frequency.get(business_id, 0)
            for business_id in task.candidate_business_ids
            if business_id != target
        )
        segment_values = {
            "history_size": _history_segment(scored.profile.history_count),
            "target_category_seen": (
                "seen" if scores[target].category_score > 0 else "unseen"
            ),
            "candidate_refill": (
                "refill_used" if refill[task.task_id] else "no_refill"
            ),
            "hybrid_margin": _margin_segment(margin),
            "feature_disagreement": _disagreement_segment(
                max(component_ranks.values()) - min(component_ranks.values())
            ),
            "missing_location": (
                "missing"
                if location.location_center is None
                or target_location.distance_km is None
                else "complete"
            ),
            "candidate_popularity": _popularity_segment(
                quality.normalized_popularity
            ),
            "negative_sampling_frequency": _frequency_segment(max_frequency),
        }
        diagnostics.append(
            HybridTaskDiagnostic(
                task_id=task.task_id,
                user_id=task.user_id,
                fold=task_folds[task.task_id],
                target_rank=target_rank,
                history_count=scored.profile.history_count,
                target_category_seen=scores[target].category_score > 0,
                candidate_refill=refill[task.task_id],
                hybrid_margin=margin,
                feature_disagreement_span=(
                    max(component_ranks.values())
                    - min(component_ranks.values())
                ),
                missing_location=(
                    location.location_center is None
                    or target_location.distance_km is None
                ),
                target_popularity=quality.normalized_popularity,
                target_review_count=quality.review_count,
                max_negative_sampling_frequency=max_frequency,
                target_scores=scores[target],
                top_scores=scores[top_business],
                target_component_ranks=component_ranks,
                primary_disadvantage=_primary_disadvantage(
                    scores[target],
                    scores[top_business],
                    weights,
                ),
                segment_values=segment_values,
            )
        )

    fold_metrics: dict[str, DiagnosticMetrics] = {}
    for fold in range(1, policy.cross_validation.folds + 1):
        fold_rows = [row for row in diagnostics if row.fold == fold]
        if not fold_rows:
            raise HybridDiagnosisError(
                f"Validation user fold {fold} contains no tasks"
            )
        fold_metrics[str(fold)] = _metrics(fold_rows)
    segment_metrics: dict[str, dict[str, SegmentDiagnostic]] = {}
    for dimension in policy.segment_dimensions:
        grouped_rows: defaultdict[str, list[HybridTaskDiagnostic]] = defaultdict(
            list
        )
        for row in diagnostics:
            grouped_rows[row.segment_values[dimension]].append(row)
        segment_metrics[dimension] = {
            label: SegmentDiagnostic(
                task_count=len(group),
                metrics=_metrics(group),
            )
            for label, group in sorted(grouped_rows.items())
        }
    failures = [row for row in diagnostics if row.target_rank > 5]
    task_payload = _task_diagnostics_payload(diagnostics)
    summary = HybridDiagnosisSummary(
        format_version=1,
        diagnosis_name="Frozen Hybrid V1 Validation Diagnosis",
        development_split="validation",
        legacy_test_loaded=False,
        strict_blind_holdout=policy.strict_blind_holdout,
        task_count=len(diagnostics),
        user_count=len({row.user_id for row in diagnostics}),
        hybrid_weights=weights,
        zero_weight_components=[
            name for name, value in weights.as_dict.items() if value == 0
        ],
        overall=_metrics(diagnostics),
        target_rank_counts={
            str(rank): sum(row.target_rank == rank for row in diagnostics)
            for rank in range(1, 21)
        },
        fold_task_counts={
            str(fold): sum(row.fold == fold for row in diagnostics)
            for fold in range(1, policy.cross_validation.folds + 1)
        },
        fold_metrics=fold_metrics,
        fold_mean=_mean_metric_models(list(fold_metrics.values())),
        fold_standard_deviation=_standard_deviation_metric_models(
            list(fold_metrics.values())
        ),
        bootstrap_samples=policy.bootstrap.samples,
        bootstrap_confidence_level=policy.bootstrap.confidence_level,
        bootstrap_intervals=_bootstrap_intervals(diagnostics, policy),
        segment_metrics=segment_metrics,
        failed_top_5_count=len(failures),
        failed_top_5_primary_disadvantage=dict(
            sorted(Counter(row.primary_disadvantage for row in failures).items())
        ),
        task_diagnostics_include_identifiers=True,
        published_summary_includes_identifiers=False,
        source_sha256={
            "validation_tasks": _sha256_file(tasks_path),
            "selected_validation_truth": truth_digest,
            "selected_validation_provenance": provenance_digest,
            "task_diagnostics": hashlib.sha256(
                task_payload.encode("utf-8")
            ).hexdigest(),
        },
        limitations=[
            (
                "Hybrid V1 的权重曾在整个 validation 上选过；五组结果只是"
                "稳定性切片，不是严格的折外模型选择成绩。"
            ),
            (
                "这个实验只排序根据目标商家构造出的 20 个候选，不能代表"
                "从全量商家中召回目标的能力。"
            ),
            (
                "主要分数差距只是帮助排查问题的描述性线索，不是因果证明。"
            ),
            "本次体检没有加载或使用 Legacy Test V0。",
        ],
    )
    return diagnostics, summary


def _atomic_write_text(path: Path, payload: str) -> None:
    if path.is_file() and path.read_text(encoding="utf-8") == payload:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.unlink(missing_ok=True)
    try:
        partial.write_text(payload, encoding="utf-8", newline="\n")
        os.replace(partial, path)
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def render_hybrid_diagnosis_markdown(
    summary: HybridDiagnosisSummary,
) -> str:
    """Render a beginner-friendly, identifier-free Markdown report."""

    overall = summary.overall
    lines = [
        "# Hybrid V1 验证集体检报告",
        "",
        "## 一句话结论",
        "",
        (
            f"Hybrid V1 在 {summary.task_count} 个 validation 任务上的 "
            f"AvgHR 为 {overall.avg_hr:.4f}，正确商家的平均排名为 "
            f"{overall.mean_target_rank:.2f}。这是一份问题定位报告，不是新的盲测成绩。"
        ),
        "",
        "## 当前模型实际在用什么",
        "",
        "| 特征 | 权重 |",
        "|---|---:|",
    ]
    component_labels = {
        "category": "类别",
        "text": "文本",
        "quality": "质量与热度",
        "location": "位置",
    }
    for name, value in summary.hybrid_weights.as_dict.items():
        lines.append(f"| {component_labels[name]} | {value:.1f} |")
    lines.extend(["", "权重为 0 的特征不会影响最终名次："])
    if summary.zero_weight_components:
        lines.append(
            "、".join(component_labels[name] for name in summary.zero_weight_components)
            + "。"
        )
    else:
        lines.append("没有。")
    lines.extend(
        [
            "",
            "## 总体成绩",
            "",
            "| 指标 | 数值 | 95% 用户 Bootstrap 区间 |",
            "|---|---:|---:|",
        ]
    )
    for name, label in (
        ("hr_at_1", "HR@1"),
        ("hr_at_3", "HR@3"),
        ("hr_at_5", "HR@5"),
        ("avg_hr", "AvgHR"),
        ("mrr", "MRR"),
        ("ndcg_at_5", "NDCG@5"),
        ("mean_target_rank", "正确商家平均排名"),
    ):
        interval = summary.bootstrap_intervals[name]
        lines.append(
            f"| {label} | {getattr(overall, name):.4f} | "
            f"[{interval.lower:.4f}, {interval.upper:.4f}] |"
        )
    lines.extend(
        [
            "",
            "## 五组稳定性检查",
            "",
            "这里是把同一个冻结模型分别放到五组用户上，不是在每一组重新调权重。",
            "",
            "| 用户组 | 任务数 | AvgHR | MRR | HR@1 | 平均排名 |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for fold, metrics in summary.fold_metrics.items():
        lines.append(
            f"| {fold} | {summary.fold_task_counts[fold]} | "
            f"{metrics.avg_hr:.4f} | {metrics.mrr:.4f} | "
            f"{metrics.hr_at_1:.4f} | {metrics.mean_target_rank:.2f} |"
        )
    lines.extend(
        [
            "",
            (
                f"五组 AvgHR 的标准差为 "
                f"{summary.fold_standard_deviation.avg_hr:.4f}。"
            ),
            "",
            "## 没进前五时，哪项分数差距最明显",
            "",
            (
                f"共有 {summary.failed_top_5_count} 个任务的正确商家没有进入前五。"
                "下面的标签表示第一名相对正确商家最大的加权分数优势，只是排查线索，"
                "不能直接当成因果结论。"
            ),
            "",
            "| 主要分数差距 | 任务数 |",
            "|---|---:|",
        ]
    )
    for name, count in summary.failed_top_5_primary_disadvantage.items():
        lines.append(f"| {component_labels.get(name, '无明显差距')} | {count} |")
    lines.extend(["", "## 最值得继续验证的线索", ""])
    if "category" in summary.zero_weight_components:
        lines.append(
            "- Category 已经被计算，但冻结权重为 0，所以现在完全不会改变最终名次。"
        )
    location_failures = summary.failed_top_5_primary_disadvantage.get(
        "location",
        0,
    )
    if summary.failed_top_5_count:
        location_rate = location_failures / summary.failed_top_5_count
        lines.append(
            f"- 没进前五的任务中，有 {location_failures} 个（{location_rate:.1%}）"
            "的最大加权差距来自位置分；这说明位置权重值得优先做消融实验。"
        )
    popularity_groups = summary.segment_metrics["candidate_popularity"]
    low_popularity = popularity_groups.get("low_below_0.5")
    high_popularity = popularity_groups.get("high_0.8_to_1.0")
    if low_popularity is not None and high_popularity is not None:
        lines.append(
            f"- 低热度目标的 AvgHR 为 {low_popularity.metrics.avg_hr:.4f}，"
            f"高热度目标为 {high_popularity.metrics.avg_hr:.4f}；当前方案对冷门商家"
            "明显更困难。"
        )
    history_groups = summary.segment_metrics["history_size"]
    small_history = history_groups.get("small_10_or_less")
    rich_history = history_groups.get("rich_over_30")
    if small_history is not None and rich_history is not None:
        lines.append(
            f"- 历史较少用户的 AvgHR 为 {small_history.metrics.avg_hr:.4f}，"
            f"历史丰富用户反而只有 {rich_history.metrics.avg_hr:.4f}；简单平均全部"
            "历史可能没有利用好兴趣变化和多种兴趣。"
        )
    category_groups = summary.segment_metrics["target_category_seen"]
    category_seen = category_groups.get("seen")
    category_unseen = category_groups.get("unseen")
    if category_seen is not None and category_unseen is not None:
        lines.append(
            f"- 见过目标类别时 AvgHR 为 {category_seen.metrics.avg_hr:.4f}，"
            f"没见过时为 {category_unseen.metrics.avg_hr:.4f}；冷启动类别仍然更难。"
        )
    lines.extend(
        [
            "",
            "## 分情况看成绩",
            "",
            "| 观察维度 | 分组 | 任务数 | AvgHR | MRR | HR@5 | 平均排名 |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    dimension_labels = {
        "history_size": "用户历史多少",
        "target_category_seen": "是否见过目标类别",
        "candidate_refill": "候选是否回填",
        "hybrid_margin": "前两名分差",
        "feature_disagreement": "四项特征是否冲突",
        "missing_location": "位置是否缺失",
        "candidate_popularity": "目标商家历史热度",
        "negative_sampling_frequency": "重复负样本频率",
    }
    for dimension, groups in summary.segment_metrics.items():
        for label, segment in groups.items():
            metrics = segment.metrics
            lines.append(
                f"| {dimension_labels[dimension]} | `{label}` | "
                f"{segment.task_count} | {metrics.avg_hr:.4f} | "
                f"{metrics.mrr:.4f} | {metrics.hr_at_5:.4f} | "
                f"{metrics.mean_target_rank:.2f} |"
            )
    lines.extend(
        [
            "",
            "## 这些数字现在能说明什么",
            "",
            "- 可以发现哪类用户或候选任务更难。",
            "- 可以发现当前固定权重忽略了哪些已经计算的特征。",
            "- 可以为 Hybrid V2 选择改进方向，但不能证明某个特征一定导致失败。",
            "- 逐任务明细只保存在被 Git 忽略的 runs 目录，不发布用户或商家 ID。",
            "",
            "## 对 Hybrid V2 的建议顺序",
            "",
            "1. 先做位置特征消融：比较原位置权重、降低权重和完全移除位置。",
            "2. 为长历史加入时间衰减或多兴趣画像，避免把很久以前的兴趣全部平均。",
            "3. 单独检查 Category 信号；不要强迫它有权重，要验证改良后是否真有增益。",
            "4. 加入协同过滤信号，帮助没有见过目标类别和低热度目标的任务。",
            "5. 所有方案都在五折用户划分上比较，并用配对 Bootstrap 判断差值。",
            "",
            "## 必须记住的限制",
            "",
        ]
    )
    lines.extend(f"- {limitation}" for limitation in summary.limitations)
    return "\n".join(lines) + "\n"


def write_hybrid_diagnosis(
    diagnostics: Sequence[HybridTaskDiagnostic],
    summary: HybridDiagnosisSummary,
    *,
    task_diagnostics_path: str | Path,
    summary_path: str | Path,
    markdown_path: str | Path,
) -> None:
    """Write local task details plus anonymous, publishable summaries."""

    _atomic_write_text(
        Path(task_diagnostics_path),
        _task_diagnostics_payload(diagnostics),
    )
    _atomic_write_text(
        Path(summary_path),
        summary.model_dump_json(indent=2) + "\n",
    )
    _atomic_write_text(
        Path(markdown_path),
        render_hybrid_diagnosis_markdown(summary),
    )
