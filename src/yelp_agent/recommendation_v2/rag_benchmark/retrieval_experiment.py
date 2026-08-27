"""复用单问题全量标签，对比查找说法数量以及纯向量/混合召回。"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

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
from yelp_agent.recommendation_v2.review_evidence.schema import (
    PreferenceSearchDescription,
)
from yelp_agent.recommendation_v2.review_evidence.segment_vectors import (
    ReviewSegmentVectorStore,
)
from yelp_agent.review_rag.config import load_review_rag_config
from yelp_agent.semantic_embedding import LocalQwenEmbeddingEncoder

from .benchmark import _compare
from .schema import LabeledReview


@dataclass(frozen=True, slots=True)
class RetrievalExperimentConfig:
    """一个已完成全量标注的案例和真实数据位置。"""

    source_project_root: Path
    case_root: Path
    qdrant_url: str = "http://localhost:6333"
    repeats: int = 3
    requirement_override_path: Path | None = None
    request_time: datetime = datetime(2026, 8, 27, 12, 0, tzinfo=UTC)

    @property
    def embeddings_path(self) -> Path:
        return (
            self.source_project_root
            / "src/yelp_agent/recommendation_v2/data/review_evidence/v1/index/segment_embeddings.npy"
        )

    @property
    def reviews_path(self) -> Path:
        return self.source_project_root / "data/processed/reviews.parquet"

    @property
    def rag_config_path(self) -> Path:
        return self.source_project_root / "configs/review_rag.yaml"


def run_retrieval_experiment(config: RetrievalExperimentConfig) -> dict[str, object]:
    """同一批正确答案依次比较1、2、3……条说法和两种检索方式。"""

    if config.repeats < 1:
        raise ValueError("experiment repeats must be positive")
    report = json.loads((config.case_root / "report.json").read_text(encoding="utf-8"))
    labels = _load_labels(config.case_root / "hidden/all_review_labels.jsonl")
    business_ids = [
        str(item["business_id"]) for item in report["hard_filtered_businesses"]
    ]
    source_requirement = _load_requirement(config, report)
    maximum = min(
        len(source_requirement.positive_descriptions),
        len(source_requirement.negative_descriptions),
    )
    if maximum < 1:
        raise ValueError("benchmark requirement has no paired search descriptions")

    variants: list[dict[str, object]] = []
    mode_startups: dict[str, dict[str, float]] = {}
    for mode in ("dense", "hybrid"):
        build_started = perf_counter()
        retriever = _build_retriever(config, enable_bm25=mode == "hybrid")
        startup_ms = (perf_counter() - build_started) * 1000
        try:
            # 最大说法数量先跑一次，只用于模型和Qdrant缓存预热，不进入结果。
            warm_requirement = _slice_requirement(source_requirement, maximum)
            warm_started = perf_counter()
            retriever.retrieve_many(
                [warm_requirement],
                business_ids,
                cutoff_time=config.request_time,
            )
            warmup_ms = (perf_counter() - warm_started) * 1000
            mode_startups[mode] = {
                "startup_ms": startup_ms,
                "warmup_ms": warmup_ms,
            }

            for count in range(1, maximum + 1):
                requirement = _slice_requirement(source_requirement, count)
                runs: list[dict[str, object]] = []
                latest_retrieval = None
                for _ in range(config.repeats):
                    started = perf_counter()
                    latest_retrieval = retriever.retrieve_many(
                        [requirement],
                        business_ids,
                        cutoff_time=config.request_time,
                    )
                    total_ms = (perf_counter() - started) * 1000
                    metrics = latest_retrieval.metrics
                    runs.append(
                        {
                            "total_ms": total_ms,
                            **metrics.model_dump(mode="json"),
                        }
                    )
                assert latest_retrieval is not None
                retrieved = latest_retrieval.by_requirement[requirement.requirement_id]
                comparison = _compare(labels, retrieved)
                variants.append(
                    {
                        "mode": mode,
                        "descriptions_each_side": count,
                        "positive_descriptions": requirement.positive_descriptions,
                        "negative_descriptions": requirement.negative_descriptions,
                        "median_total_ms": statistics.median(
                            float(item["total_ms"]) for item in runs
                        ),
                        "min_total_ms": min(float(item["total_ms"]) for item in runs),
                        "max_total_ms": max(float(item["total_ms"]) for item in runs),
                        "runs": runs,
                        "comparison": comparison.model_dump(mode="json"),
                    }
                )
        finally:
            retriever.close()

    result = {
        "case_id": report["case_id"],
        "query_text": report["question"]["query_text"],
        "business_ids": business_ids,
        "label_count": len(labels),
        "repeats": config.repeats,
        "mode_startups": mode_startups,
        "variants": variants,
    }
    destination = config.case_root / "audit/retrieval_ablation.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def _build_retriever(
    config: RetrievalExperimentConfig,
    *,
    enable_bm25: bool,
) -> ReviewEvidenceRetriever:
    rag_config = load_review_rag_config(config.rag_config_path)
    encoder = LocalQwenEmbeddingEncoder.from_environment(
        rag_config.semantic_config().model_copy(update={"batch_size": 16}),
        _local_embedding_environment(),
    )
    return ReviewEvidenceRetriever(
        store=QdrantReviewSegmentStore.from_url(config.qdrant_url),
        encoder=encoder,
        segment_vectors=ReviewSegmentVectorStore(config.embeddings_path),
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
        enable_bm25=enable_bm25,
        rrf_k=60,
    )


def _slice_requirement(
    source: PreferenceSearchDescription,
    count: int,
) -> PreferenceSearchDescription:
    if count < 1:
        raise ValueError("description count must be positive")
    # 数据结构要求正式状态至少两条；实验中的单条版本只用于测量底层召回，
    # 因而先复制成两条再在检索入口处使用去重后的单条列表并不合适。
    # 这里通过model_construct保留同一个公开结构，同时不改变生产校验规则。
    if count == 1:
        return PreferenceSearchDescription.model_construct(
            **{
                **source.model_dump(),
                "positive_descriptions": source.positive_descriptions[:1],
                "negative_descriptions": source.negative_descriptions[:1],
            }
        )
    return source.model_copy(
        update={
            "positive_descriptions": source.positive_descriptions[:count],
            "negative_descriptions": source.negative_descriptions[:count],
        },
        deep=True,
    )


def _load_labels(path: Path) -> list[LabeledReview]:
    return [
        LabeledReview.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _load_requirement(
    config: RetrievalExperimentConfig,
    report: dict[str, object],
) -> PreferenceSearchDescription:
    """默认复用原评测说法，也可换成本次真实模型新生成的说法。"""

    if config.requirement_override_path is not None:
        payload = json.loads(
            config.requirement_override_path.read_text(encoding="utf-8")
        )
        return PreferenceSearchDescription.model_validate(payload)
    descriptions = report.get("review_search_descriptions")
    if not isinstance(descriptions, list) or not descriptions:
        raise ValueError("benchmark report has no review search descriptions")
    return PreferenceSearchDescription.model_validate(descriptions[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-project-root",
        type=Path,
        default=Path(r"C:\Users\29072\PycharmProjects\AgentSociety"),
    )
    parser.add_argument("--case-root", type=Path, required=True)
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--requirement-override", type=Path)
    args = parser.parse_args()
    result = run_retrieval_experiment(
        RetrievalExperimentConfig(
            source_project_root=args.source_project_root.resolve(),
            case_root=args.case_root.resolve(),
            qdrant_url=args.qdrant_url,
            repeats=args.repeats,
            requirement_override_path=(
                args.requirement_override.resolve()
                if args.requirement_override is not None
                else None
            ),
        )
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
