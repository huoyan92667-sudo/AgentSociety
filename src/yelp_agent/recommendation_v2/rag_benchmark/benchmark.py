"""一个入口完成单问题生成、真实硬筛、全量标注和当前召回对比。"""

from __future__ import annotations

import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import duckdb
from dotenv import load_dotenv

from yelp_agent.recommendation_v2.business_facts import BusinessFactCatalog
from yelp_agent.recommendation_v2.category_catalog import FixedCategoryCatalog
from yelp_agent.recommendation_v2.preference_fusion import (
    PreferenceFusionAttempt,
    PreferenceFusionRequest,
    build_preference_fusion,
)
from yelp_agent.recommendation_v2.review_evidence.full_reviews import FullReviewStore
from yelp_agent.recommendation_v2.review_evidence.qdrant_store import (
    QdrantReviewSegmentStore,
)
from yelp_agent.recommendation_v2.review_evidence.retrieval import (
    ReviewEvidenceRetriever,
)
from yelp_agent.recommendation_v2.review_evidence.runtime import (
    _local_embedding_environment,
)
from yelp_agent.recommendation_v2.review_evidence.segment_vectors import (
    ReviewSegmentVectorStore,
)
from yelp_agent.recommendation_v2.review_evidence.schema import (
    PreferenceSearchDescription,
)
from yelp_agent.recommendation_v2.schema import (
    GeoPoint,
    HardConstraint,
    OpenRequirement,
    RequirementBasis,
    SearchCenter,
    UnifiedRecommendationState,
    merchant_feature_for,
    requirement_unit_for,
)
from yelp_agent.recommendation_v2.tools import (
    GeographicDistanceTool,
    StructuredHardFilterTool,
)
from yelp_agent.review_rag.config import load_review_rag_config
from yelp_agent.semantic_embedding import LocalQwenEmbeddingEncoder

from .claude_worker import (
    ClaudeCodeWorker,
    ClaudeStructuredOutputError,
    load_prompt,
)
from .schema import (
    ClaudeWorkerTrace,
    LabeledReview,
    QuestionProposal,
    RecallComparison,
    ReviewAnnotationBatch,
    ReviewAnnotationRecord,
    ReviewEvidenceLabel,
    SingleCaseBenchmarkReport,
)

_MODULE_ROOT = Path(__file__).resolve().parent
_CODE_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_QUESTION_PROMPT = _MODULE_ROOT / "prompts" / "question_generation.txt"
_ANNOTATION_PROMPT = _MODULE_ROOT / "prompts" / "review_annotation.txt"


@dataclass(frozen=True, slots=True)
class SingleCaseBenchmarkConfig:
    """运行一次真实试验需要知道的数据位置和分批上限。"""

    source_project_root: Path
    output_root: Path | None = None
    case_id: str = "rag_authentic_szechuan_0001"
    qdrant_url: str = "http://localhost:6333"
    max_reviews_per_batch: int = 50
    max_chars_per_batch: int = 60_000
    annotation_concurrency: int = 4
    request_time: datetime = datetime(2026, 8, 27, 12, 0, tzinfo=UTC)

    @property
    def business_facts_path(self) -> Path:
        return (
            self.source_project_root
            / "src/yelp_agent/recommendation_v2/data/business_facts/v1/business_facts.parquet"
        )

    @property
    def price_bands_path(self) -> Path:
        return (
            self.source_project_root
            / "src/yelp_agent/recommendation_v2/data/business_facts/v1/price_bands.json"
        )

    @property
    def category_catalog_path(self) -> Path:
        return (
            self.source_project_root
            / "src/yelp_agent/recommendation_v2/data/category_catalog/v1/catalog.json"
        )

    @property
    def reviews_path(self) -> Path:
        return self.source_project_root / "data/processed/reviews.parquet"

    @property
    def segment_embeddings_path(self) -> Path:
        return (
            self.source_project_root
            / "src/yelp_agent/recommendation_v2/data/review_evidence/v1/index/segment_embeddings.npy"
        )

    @property
    def environment_file(self) -> Path:
        return self.source_project_root / ".env"

    @property
    def case_output_root(self) -> Path:
        base = self.output_root or (
            _CODE_PROJECT_ROOT
            / "src/yelp_agent/recommendation_v2/data/rag_benchmark/v1/cases"
        )
        return Path(base) / self.case_id


def run_single_case_benchmark(
    config: SingleCaseBenchmarkConfig,
    *,
    worker: ClaudeCodeWorker | None = None,
) -> SingleCaseBenchmarkReport:
    """运行一个问题；输入商家范围内的每条评论都必须获得一个标签。"""

    _validate_config(config)
    worker = worker or ClaudeCodeWorker()
    output = config.case_output_root
    inputs_root = output / "inputs"
    raw_root = output / "raw_outputs"
    hidden_root = output / "hidden"
    audit_root = output / "audit"
    for path in (inputs_root, raw_root, hidden_root, audit_root):
        path.mkdir(parents=True, exist_ok=True)

    question_wrapper_path = raw_root / "question_wrapper.json"
    if question_wrapper_path.is_file():
        question_wrapper = json.loads(
            question_wrapper_path.read_text(encoding="utf-8")
        )
        question, question_trace, _ = worker.parse_wrapper(
            question_wrapper,
            QuestionProposal,
        )
        print("question_resumed=true", flush=True)
    else:
        question, question_trace, question_wrapper = _generate_question(worker)
        _write_json(question_wrapper_path, question_wrapper)
    _validate_generated_question(question)
    _write_json(inputs_root / "question.json", question.model_dump(mode="json"))

    load_dotenv(config.environment_file, override=False)
    catalog = BusinessFactCatalog.from_files(
        config.business_facts_path,
        config.price_bands_path,
    )
    category_catalog = FixedCategoryCatalog.from_file(config.category_catalog_path)
    fusion_attempt, state, state_source = _fuse_question(
        config,
        question,
        catalog,
        saved_attempt_path=raw_root / "fusion_attempt.json",
    )
    hard_filter = _hard_filter(state, catalog, category_catalog)
    _validate_hard_scope(state, hard_filter.candidate_count)
    requirement = _select_authenticity_requirement(
        fusion_attempt,
        question,
    )

    reviews = _load_all_reviews(
        config.reviews_path,
        hard_filter.candidate_business_ids,
    )
    batches = _split_reviews(
        reviews,
        max_reviews=config.max_reviews_per_batch,
        max_chars=config.max_chars_per_batch,
    )
    businesses = [
        {
            "business_id": item.business.business_id,
            "name": item.business.name,
            "categories": item.business.categories,
            "address": item.business.address,
        }
        for item in hard_filter.candidates
    ]
    batch_jobs: list[tuple[str, list[ReviewAnnotationRecord], dict[str, object]]] = []
    for index, batch_reviews in enumerate(batches, start=1):
        batch_id = f"{config.case_id}.batch_{index:04d}"
        payload = {
            "batch_id": batch_id,
            "case_id": config.case_id,
            "user_query": question.query_text,
            "evidence_requirement": {
                "text": question.evidence_requirement,
                "positive_definition": question.positive_definition,
                "negative_definition": question.negative_definition,
            },
            "hard_filtered_businesses": businesses,
            "review_batch": [item.model_dump(mode="json") for item in batch_reviews],
        }
        _write_json(inputs_root / f"{batch_id}.json", payload)
        batch_jobs.append((batch_id, batch_reviews, payload))

    def annotate(job):
        batch_id, batch_reviews, payload = job
        traces: list[ClaudeWorkerTrace] = []
        last_error: Exception | None = None
        existing_paths = sorted(raw_root.glob(f"{batch_id}.attempt_*.json"))
        for existing_path in existing_paths:
            wrapper = json.loads(existing_path.read_text(encoding="utf-8"))
            try:
                result, trace, _ = worker.parse_wrapper(
                    wrapper,
                    ReviewAnnotationBatch,
                )
                _validate_annotation_batch(
                    result,
                    expected_batch_id=batch_id,
                    expected_case_id=config.case_id,
                    reviews=batch_reviews,
                )
            except (ClaudeStructuredOutputError, TypeError, ValueError) as exc:
                last_error = exc
                continue
            traces.append(trace)
            joined = [
                LabeledReview(review=review, label=label, batch_id=batch_id)
                for review, label in _join_reviews_and_labels(
                    batch_reviews,
                    result.labels,
                )
            ]
            print(
                f"annotation_batch_resumed={batch_id} reviews={len(batch_reviews)}",
                flush=True,
            )
            return joined, traces

        if len(existing_paths) >= 2 and len(batch_reviews) > 12:
            return annotate_split(batch_id, batch_reviews, payload, traces)

        first_new_attempt = len(existing_paths) + 1
        for attempt_index in range(first_new_attempt, first_new_attempt + 2):
            correction = ""
            if last_error is not None:
                correction = (
                    "\n\n上一次返回未通过程序检查："
                    + str(last_error)
                    + "。请重新逐字核对本批每个 review_id，不能改动任何字符。"
                )
            try:
                result, trace, wrapper = worker.generate(
                    _annotation_prompt(payload) + correction,
                    ReviewAnnotationBatch,
                )
            except ClaudeStructuredOutputError as exc:
                traces.append(exc.trace)
                _write_json(
                    raw_root / f"{batch_id}.attempt_{attempt_index}.json",
                    exc.wrapper,
                )
                last_error = exc
                continue
            else:
                traces.append(trace)
                _write_json(
                    raw_root / f"{batch_id}.attempt_{attempt_index}.json",
                    wrapper,
                )
            try:
                _validate_annotation_batch(
                    result,
                    expected_batch_id=batch_id,
                    expected_case_id=config.case_id,
                    reviews=batch_reviews,
                )
            except (TypeError, ValueError) as exc:
                last_error = exc
                continue
            joined = [
                LabeledReview(review=review, label=label, batch_id=batch_id)
                for review, label in _join_reviews_and_labels(
                    batch_reviews,
                    result.labels,
                )
            ]
            print(
                f"annotation_batch_completed={batch_id} reviews={len(batch_reviews)} attempts={attempt_index}",
                flush=True,
            )
            return joined, traces
        if len(batch_reviews) > 12:
            return annotate_split(batch_id, batch_reviews, payload, traces)
        raise RuntimeError(f"annotation batch failed twice: {batch_id}: {last_error}")

    def annotate_split(
        parent_batch_id: str,
        parent_reviews: list[ReviewAnnotationRecord],
        parent_payload: dict[str, object],
        parent_traces: list[ClaudeWorkerTrace],
    ):
        """连续失败的批次只按容量对半拆，不挑选或丢弃评论。"""

        midpoint = len(parent_reviews) // 2
        parts = [parent_reviews[:midpoint], parent_reviews[midpoint:]]
        all_labels: list[LabeledReview] = []
        all_traces = list(parent_traces)
        print(
            f"annotation_batch_split={parent_batch_id} reviews={len(parent_reviews)} parts={len(parts)}",
            flush=True,
        )
        for part_index, part_reviews in enumerate(parts, start=1):
            child_id = f"{parent_batch_id}.part_{part_index:02d}"
            child_payload = {
                **parent_payload,
                "batch_id": child_id,
                "review_batch": [
                    item.model_dump(mode="json") for item in part_reviews
                ],
            }
            _write_json(inputs_root / f"{child_id}.json", child_payload)
            child_labels, child_traces = annotate(
                (child_id, part_reviews, child_payload)
            )
            all_labels.extend(child_labels)
            all_traces.extend(child_traces)
        return all_labels, all_traces

    worker_count = min(max(config.annotation_concurrency, 1), len(batch_jobs) or 1)
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        completed_batches = list(executor.map(annotate, batch_jobs))
    labels = [
        item
        for batch_labels, _ in completed_batches
        for item in batch_labels
    ]
    annotation_traces = [
        trace
        for _, traces in completed_batches
        for trace in traces
    ]

    _validate_complete_labels(reviews, labels)
    _write_jsonl(
        hidden_root / "all_review_labels.jsonl",
        [item.model_dump(mode="json") for item in labels],
    )

    retrieval_started = perf_counter()
    retrieved = _run_current_retrieval(
        config,
        requirement=requirement,
        business_ids=hard_filter.candidate_business_ids,
    )
    retrieval_latency_ms = (perf_counter() - retrieval_started) * 1000
    _write_json(
        audit_root / "current_retrieval.json",
        {
            business_id: [item.model_dump(mode="json") for item in candidates]
            for business_id, candidates in retrieved.items()
        },
    )
    comparison = _compare(labels, retrieved)
    missed = {
        item.review.review_id: item
        for item in labels
        if item.review.review_id in set(comparison.missed_review_ids)
    }
    _write_jsonl(
        audit_root / "missed_evidence.jsonl",
        [missed[key].model_dump(mode="json") for key in comparison.missed_review_ids],
    )

    all_annotation_traces = _load_all_annotation_traces(
        raw_root,
        worker,
        config.case_id,
    )
    report = SingleCaseBenchmarkReport(
        case_id=config.case_id,
        question=question,
        fusion_status=fusion_attempt.status,
        fusion_model=fusion_attempt.model,
        fusion_failure_reason=fusion_attempt.failure_reason,
        state_source=state_source,
        search_center=(
            None
            if state.search_center is None
            else state.search_center.model_dump(mode="json")
        ),
        hard_constraints=[item.model_dump(mode="json") for item in state.hard_constraints],
        default_constraints=[
            item.model_dump(mode="json") for item in state.default_constraints
        ],
        review_search_descriptions=[requirement.model_dump(mode="json")],
        hard_filtered_business_count=hard_filter.candidate_count,
        hard_filtered_businesses=businesses,
        exhaustive_review_count=len(reviews),
        annotation_batch_count=len({item.batch_id for item in labels}),
        annotation_label_counts=dict(
            sorted(Counter(item.label.label for item in labels).items())
        ),
        question_trace=question_trace,
        fusion_latency_ms=fusion_attempt.latency_ms,
        fusion_input_tokens=fusion_attempt.input_tokens,
        fusion_output_tokens=fusion_attempt.output_tokens,
        annotation_model_call_count=len(all_annotation_traces),
        annotation_total_input_tokens=sum(
            item.input_tokens or 0 for item in all_annotation_traces
        ),
        annotation_total_output_tokens=sum(
            item.output_tokens or 0 for item in all_annotation_traces
        ),
        annotation_total_thinking_tokens=sum(
            item.thinking_tokens or 0 for item in all_annotation_traces
        ),
        annotation_api_time_ms_sum=sum(
            item.duration_ms for item in all_annotation_traces
        ),
        annotation_reported_cost_usd=sum(
            item.reported_cost_usd or 0.0 for item in all_annotation_traces
        ),
        annotation_traces=all_annotation_traces,
        retrieval_latency_ms=retrieval_latency_ms,
        comparison=comparison,
        output_root=str(output.resolve()),
    )
    _write_json(output / "report.json", report.model_dump(mode="json"))
    (output / "summary.md").write_text(
        _summary_markdown(report),
        encoding="utf-8",
        newline="\n",
    )
    return report


def _generate_question(
    worker: ClaudeCodeWorker,
) -> tuple[QuestionProposal, ClaudeWorkerTrace, dict[str, object]]:
    prompt = load_prompt(_QUESTION_PROMPT)
    return worker.generate(prompt, QuestionProposal)


def _fuse_question(
    config: SingleCaseBenchmarkConfig,
    question: QuestionProposal,
    catalog: BusinessFactCatalog,
    *,
    saved_attempt_path: Path,
):
    if saved_attempt_path.is_file():
        attempt = PreferenceFusionAttempt.model_validate_json(
            saved_attempt_path.read_text(encoding="utf-8")
        )
        print("fusion_attempt_resumed=true", flush=True)
    else:
        fusion = build_preference_fusion(catalog)
        attempt = fusion.fuse(
            PreferenceFusionRequest(
                user_id="rag-benchmark-user",
                session_id=config.case_id,
                turn_index=1,
                query_text=question.query_text,
                request_time=config.request_time,
                previous_state=None,
                conversation_history=[],
                profile_preferences=None,
                user_location=None,
            )
        )
        _write_json(saved_attempt_path, attempt.model_dump(mode="json"))
    if attempt.status == "success" and attempt.state is not None:
        return attempt, attempt.state, "four_source_fusion"
    # 这次评测只验证评论召回。地点解析失败不能被静默忽略，但也不能让
    # 一个与评论召回无关的坐标格式问题阻断试验。这里使用题目生成阶段
    # 已固定并在真实商家表中核对过的唐人街中心与300米范围。
    return attempt, _fixed_benchmark_state(config, question), "benchmark_fixed_scope_fallback"


def _hard_filter(
    state: UnifiedRecommendationState,
    catalog: BusinessFactCatalog,
    category_catalog: FixedCategoryCatalog,
):
    geography = (
        None
        if state.search_center is None
        else GeographicDistanceTool(catalog).execute(state.search_center)
    )
    return StructuredHardFilterTool(catalog, category_catalog).execute(
        state,
        geography=geography,
    )


def _load_all_reviews(
    reviews_path: Path,
    business_ids: list[str],
) -> list[ReviewAnnotationRecord]:
    if not business_ids:
        return []
    marks = ",".join("?" for _ in business_ids)
    escaped = str(reviews_path.resolve()).replace("'", "''")
    query = f"""
        SELECT review_id, business_id, date AS review_time, text AS review_text
        FROM read_parquet('{escaped}')
        WHERE business_id IN ({marks})
        ORDER BY business_id, review_time, review_id
    """
    with duckdb.connect(database=":memory:") as connection:
        rows = connection.execute(query, business_ids).fetch_arrow_table().to_pylist()
    reviews = [ReviewAnnotationRecord.model_validate(row) for row in rows]
    if len({item.review_id for item in reviews}) != len(reviews):
        raise ValueError("source review IDs are not unique")
    return reviews


def _split_reviews(
    reviews: list[ReviewAnnotationRecord],
    *,
    max_reviews: int,
    max_chars: int,
) -> list[list[ReviewAnnotationRecord]]:
    """只按模型容量分批；所有评论仍然恰好出现一次。"""

    if max_reviews < 1 or max_chars < 5000:
        raise ValueError("review batch limits are too small")
    batches: list[list[ReviewAnnotationRecord]] = []
    current: list[ReviewAnnotationRecord] = []
    current_chars = 0
    for review in reviews:
        review_chars = len(review.review_text)
        if current and (
            len(current) >= max_reviews
            or current_chars + review_chars > max_chars
        ):
            batches.append(current)
            current = []
            current_chars = 0
        current.append(review)
        current_chars += review_chars
    if current:
        batches.append(current)
    flattened = [item.review_id for batch in batches for item in batch]
    if flattened != [item.review_id for item in reviews]:
        raise RuntimeError("review batching lost or reordered reviews")
    return batches


def _annotation_prompt(payload: dict[str, object]) -> str:
    return (
        load_prompt(_ANNOTATION_PROMPT)
        + "\n\n下面是本批输入 JSON：\n"
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )


def _validate_annotation_batch(
    result: ReviewAnnotationBatch,
    *,
    expected_batch_id: str,
    expected_case_id: str,
    reviews: list[ReviewAnnotationRecord],
) -> None:
    if result.batch_id != expected_batch_id or result.case_id != expected_case_id:
        raise ValueError("annotation returned a different batch or case ID")
    expected = {item.review_id: item for item in reviews}
    actual_ids = [item.review_id for item in result.labels]
    if len(actual_ids) != len(set(actual_ids)):
        raise ValueError("annotation contains duplicate review IDs")
    if set(actual_ids) != set(expected):
        missing = set(expected) - set(actual_ids)
        unexpected = set(actual_ids) - set(expected)
        raise ValueError(
            f"annotation coverage mismatch: missing={len(missing)} unexpected={len(unexpected)}"
        )
    for label in result.labels:
        source = expected[label.review_id]
        if label.business_id != source.business_id:
            raise ValueError(f"business ID changed for review {label.review_id}")
        spans = [
            *label.positive_spans,
            *label.negative_spans,
            *([] if label.ambiguous_span is None else [label.ambiguous_span]),
        ]
        for span in spans:
            if span not in source.review_text:
                raise ValueError(
                    f"evidence span is not an exact substring of review {label.review_id}"
                )


def _join_reviews_and_labels(
    reviews: list[ReviewAnnotationRecord],
    labels: list[ReviewEvidenceLabel],
) -> list[tuple[ReviewAnnotationRecord, ReviewEvidenceLabel]]:
    by_id = {item.review_id: item for item in labels}
    return [(review, by_id[review.review_id]) for review in reviews]


def _validate_complete_labels(
    reviews: list[ReviewAnnotationRecord],
    labels: list[LabeledReview],
) -> None:
    review_ids = [item.review_id for item in reviews]
    label_ids = [item.review.review_id for item in labels]
    if label_ids != review_ids:
        raise ValueError("merged labels do not cover every source review exactly once")


def _select_authenticity_requirement(
    fusion_attempt,
    question: QuestionProposal,
) -> PreferenceSearchDescription:
    requirements = fusion_attempt.review_search_descriptions
    matching = [
        item
        for item in requirements
        if item.kind == "long_tail"
        and any(token in item.requirement_text for token in ("地道", "正宗"))
    ]
    if len(matching) == 1:
        return matching[0]
    # 即使完整统一状态因为地点字段校验失败，原始模型输出中通常仍然已经
    # 生成了本轮长尾评论检索说法。只读取这部分，不接受模型改写硬范围。
    positive, negative = _raw_authenticity_descriptions(fusion_attempt.raw_json)
    return PreferenceSearchDescription(
        requirement_id="open.authentic_szechuan.benchmark",
        requirement_text=question.evidence_requirement,
        kind="long_tail",
        priority=1,
        preference_strength=100,
        positive_descriptions=positive,
        negative_descriptions=negative,
    )


def _raw_authenticity_descriptions(raw_json: str | None) -> tuple[list[str], list[str]]:
    if raw_json:
        try:
            payload = json.loads(raw_json)
            plans = payload.get("review_search_plans", [])
            for plan in plans:
                if plan.get("kind") != "long_tail":
                    continue
                positive = plan.get("positive_descriptions")
                negative = plan.get("negative_descriptions")
                if (
                    isinstance(positive, list)
                    and 2 <= len(positive) <= 5
                    and all(isinstance(item, str) and item for item in positive)
                    and isinstance(negative, list)
                    and 2 <= len(negative) <= 5
                    and all(isinstance(item, str) and item for item in negative)
                ):
                    return positive, negative
        except (TypeError, ValueError, json.JSONDecodeError):
            pass
    return (
        [
            "authentic Szechuan cuisine",
            "traditional Szechuan recipes",
            "true Szechuan taste",
        ],
        [
            "not authentic Szechuan",
            "westernized Szechuan food",
            "lacks Szechuan authenticity",
        ],
    )


def _fixed_benchmark_state(
    config: SingleCaseBenchmarkConfig,
    question: QuestionProposal,
) -> UnifiedRecommendationState:
    basis = RequirementBasis(
        source="current_query",
        text=question.query_text,
        turn_index=1,
    )
    preference_basis = basis.model_copy(update={"preference_strength": 100})
    return UnifiedRecommendationState(
        user_id="rag-benchmark-user",
        session_id=config.case_id,
        revision=1,
        turn_index=1,
        latest_query_text=question.query_text,
        search_center=SearchCenter(
            kind="named_place",
            label="费城唐人街中心",
            location=GeoPoint(latitude=39.9534, longitude=-75.1579),
        ),
        hard_constraints=[
            HardConstraint(
                key="query.category.required.szechuan",
                field="category",
                operator="any_of",
                value=["Szechuan"],
                unit=requirement_unit_for("category"),
                merchant_feature=merchant_feature_for("category"),
                controlling_source="current_query",
                sources=[basis],
            ),
            HardConstraint(
                key="query.distance.max.0_3km",
                field="distance_km",
                operator="less_than_or_equal",
                value=0.3,
                unit=requirement_unit_for("distance_km"),
                merchant_feature=merchant_feature_for("distance_km"),
                controlling_source="current_query",
                sources=[basis.model_copy(deep=True)],
            ),
        ],
        open_requirements=[
            OpenRequirement(
                key="open.authentic_szechuan.benchmark",
                text=question.evidence_requirement,
                behavior="prefer",
                priority=1,
                controlling_source="current_query",
                sources=[preference_basis],
            )
        ],
    )


def _run_current_retrieval(
    config: SingleCaseBenchmarkConfig,
    *,
    requirement,
    business_ids: list[str],
):
    rag_config = load_review_rag_config(_CODE_PROJECT_ROOT / "configs/review_rag.yaml")
    encoder = LocalQwenEmbeddingEncoder.from_environment(
        rag_config.semantic_config().model_copy(update={"batch_size": 16}),
        _local_embedding_environment(),
    )
    retriever = ReviewEvidenceRetriever(
        store=QdrantReviewSegmentStore.from_url(config.qdrant_url),
        encoder=encoder,
        segment_vectors=ReviewSegmentVectorStore(config.segment_embeddings_path),
        full_reviews=FullReviewStore(config.reviews_path),
        recall_threshold=0.55,
        acceptance_threshold=0.60,
        direction_margin=0.05,
        recall_each_side=15,
        initial_segment_group_size=15,
        middle_segment_group_size=30,
        final_segment_group_size=60,
        minimum_clear_evidence=5,
        search_concurrency=4,
        enable_bm25=True,
        rrf_k=60,
    )
    try:
        return retriever.retrieve(
            requirement,
            business_ids,
            cutoff_time=config.request_time,
        )
    finally:
        retriever.close()


def _compare(labels: list[LabeledReview], retrieved) -> RecallComparison:
    truth = {item.review.review_id: item.label for item in labels}
    direct = {
        review_id
        for review_id, label in truth.items()
        if label.label in {"positive", "negative"}
        and label.evidence_grade == "direct"
    }
    positive_direct = {
        review_id for review_id in direct if truth[review_id].label == "positive"
    }
    negative_direct = direct - positive_direct
    strict = {
        review_id
        for review_id, label in truth.items()
        if label.label in {"positive", "negative"}
        and label.evidence_grade in {"direct", "supporting"}
    }
    positive = {review_id for review_id in strict if truth[review_id].label == "positive"}
    negative = {review_id for review_id in strict if truth[review_id].label == "negative"}
    retrieved_items = {
        item.review_id: item
        for candidates in retrieved.values()
        for item in candidates
    }
    retrieved_ids = set(retrieved_items)
    retrieved_strict = strict & retrieved_ids
    wrong_direction = sorted(
        review_id
        for review_id in retrieved_strict
        if retrieved_items[review_id].direction != truth[review_id].label
    )
    direction_correct = sum(
        retrieved_items[review_id].direction == truth[review_id].label
        for review_id in retrieved_strict
    )
    false_directional = sorted(
        review_id
        for review_id, item in retrieved_items.items()
        if item.direction in {"positive", "negative"} and review_id not in strict
    )
    return RecallComparison(
        total_review_count=len(labels),
        direct_relevant_count=len(direct),
        positive_direct_count=len(positive_direct),
        negative_direct_count=len(negative_direct),
        retrieved_direct_count=len(direct & retrieved_ids),
        direct_recall=_ratio(len(direct & retrieved_ids), len(direct)),
        positive_direct_recall=_ratio(
            len(positive_direct & retrieved_ids),
            len(positive_direct),
        ),
        negative_direct_recall=_ratio(
            len(negative_direct & retrieved_ids),
            len(negative_direct),
        ),
        strict_relevant_count=len(strict),
        positive_relevant_count=len(positive),
        negative_relevant_count=len(negative),
        retrieved_review_count=len(retrieved_ids),
        retrieved_strict_relevant_count=len(retrieved_strict),
        strict_recall=_ratio(len(retrieved_strict), len(strict)),
        positive_recall=_ratio(len(positive & retrieved_ids), len(positive)),
        negative_recall=_ratio(len(negative & retrieved_ids), len(negative)),
        direction_correct_count=direction_correct,
        wrong_direction_count=len(wrong_direction),
        false_directional_evidence_count=len(false_directional),
        missed_review_ids=sorted(strict - retrieved_ids),
        wrong_direction_review_ids=wrong_direction,
        false_directional_review_ids=false_directional,
    )


def _validate_generated_question(question: QuestionProposal) -> None:
    text = question.query_text
    if not any(token in text for token in ("300米", "三百米")):
        raise ValueError("generated question does not contain the 300-meter hard limit")
    for options, message in (
        (("费城唐人街", "Philadelphia Chinatown"), "search center"),
        (("川菜", "四川菜"), "Szechuan category"),
        (("地道", "正宗"), "authenticity requirement"),
    ):
        if not any(token in text for token in options):
            raise ValueError(f"generated question is missing {message}")


def _validate_hard_scope(state: UnifiedRecommendationState, candidate_count: int) -> None:
    fields = {item.field for item in state.hard_constraints}
    if not {"category", "distance_km"}.issubset(fields):
        raise ValueError("fusion did not produce category and distance hard constraints")
    distance_values = [
        float(item.value)
        for item in state.hard_constraints
        if item.field == "distance_km"
    ]
    if len(distance_values) != 1 or abs(distance_values[0] - 0.3) > 0.02:
        raise ValueError(f"fusion changed the 300-meter limit: {distance_values}")
    category_values = [
        item.value for item in state.hard_constraints if item.field == "category"
    ]
    if not category_values or not any(
        "Szechuan" in value
        for values in category_values
        for value in (values if isinstance(values, list) else [values])
    ):
        raise ValueError("fusion did not choose the real Szechuan category")
    if not 1 <= candidate_count <= 20:
        raise ValueError(f"hard-filtered business count is unsuitable: {candidate_count}")


def _validate_config(config: SingleCaseBenchmarkConfig) -> None:
    for path in (
        config.business_facts_path,
        config.price_bands_path,
        config.category_catalog_path,
        config.reviews_path,
        config.segment_embeddings_path,
        config.environment_file,
        _QUESTION_PROMPT,
        _ANNOTATION_PROMPT,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)


def _load_all_annotation_traces(
    raw_root: Path,
    worker: ClaudeCodeWorker,
    case_id: str,
) -> list[ClaudeWorkerTrace]:
    """失败、重试和拆分调用都计入成本，不能只统计最终成功批次。"""

    traces: list[ClaudeWorkerTrace] = []
    for path in sorted(raw_root.glob(f"{case_id}.batch_*.json")):
        wrapper = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(wrapper.get("usage"), dict):
            continue
        traces.append(worker.trace_from_wrapper(wrapper))
    return traces


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )


def _summary_markdown(report: SingleCaseBenchmarkReport) -> str:
    comparison = report.comparison
    return f"""# 单问题全量评论召回试验

- 问题：{report.question.query_text}
- 硬筛商家：{report.hard_filtered_business_count} 家
- 全量评论：{report.exhaustive_review_count} 条
- GLM 分批：{report.annotation_batch_count} 批
- 评论标注模型调用（含失败与重试）：{report.annotation_model_call_count} 次
- 评论标注输入词元：{report.annotation_total_input_tokens}
- 评论标注输出词元：{report.annotation_total_output_tokens}
- Claude Code 返回的费用估算：${report.annotation_reported_cost_usd:.4f}
- 直接正反证据：{comparison.direct_relevant_count} 条
- 直接证据找回比例：{_format_ratio(comparison.direct_recall)}
- 直接正面证据找回比例：{_format_ratio(comparison.positive_direct_recall)}
- 直接反面证据找回比例：{_format_ratio(comparison.negative_direct_recall)}
- 严格正反证据：{comparison.strict_relevant_count} 条
- 当前召回评论：{comparison.retrieved_review_count} 条
- 命中严格证据：{comparison.retrieved_strict_relevant_count} 条
- 总证据找回比例：{_format_ratio(comparison.strict_recall)}
- 正面证据找回比例：{_format_ratio(comparison.positive_recall)}
- 反面证据找回比例：{_format_ratio(comparison.negative_recall)}
- 漏掉的严格证据：{len(comparison.missed_review_ids)} 条
- 方向判断错误：{comparison.wrong_direction_count} 条
- 把非严格证据判成明确方向：{comparison.false_directional_evidence_count} 条
"""


def _format_ratio(value: float | None) -> str:
    return "无可计算证据" if value is None else f"{value:.2%}"
