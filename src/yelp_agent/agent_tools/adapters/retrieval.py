"""Target-blind multi-route candidate expansion adapter."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from typing import Protocol, Sequence

from yelp_agent.retrieval import RetrievalResult, RetrievalTaskContext
from yelp_agent.query import RecommendationRequest
from yelp_agent.query_retrieval import (
    DualChannelFusion,
    QueryCandidateRetriever,
    QueryRetrievalCandidate,
    QueryRetrievalTask,
)

from ..registry import ToolDefinition
from ..schema import ToolExecutionContext, ToolObservation
from ..tool_schemas import CandidateRetrievalOutput, EmptyToolInput


class HistoryReader(Protocol):
    def user_history(self, user_id: str, cutoff_time: datetime) -> Sequence[object]: ...


class CandidateRetriever(Protocol):
    def retrieve(
        self,
        task: RetrievalTaskContext,
        *,
        include_route_provenance: bool = False,
    ) -> RetrievalResult: ...


class ExpandCandidatesTool:
    definition = ToolDefinition(
        name="EXPAND_CANDIDATES",
        version="2.0.0",
        kind="deterministic",
        allowed_actions=("retrieve_candidates",),
        input_model=EmptyToolInput,
        output_model=CandidateRetrievalOutput,
        public_summary=(
            "Retrieve a target-blind candidate set using history and current Query."
        ),
        preconditions=("user has at least one pre-cutoff interaction",),
        resolves_uncertainties=("candidate_set_missing",),
    )

    def __init__(
        self,
        retriever: CandidateRetriever,
        history_reader: HistoryReader,
        *,
        query_retriever: QueryCandidateRetriever | None = None,
        dual_fusion: DualChannelFusion | None = None,
    ) -> None:
        if (query_retriever is None) != (dual_fusion is None):
            raise ValueError(
                "query_retriever and dual_fusion must be configured together"
            )
        self._retriever = retriever
        self._history_reader = history_reader
        self._query_retriever = query_retriever
        self._dual_fusion = dual_fusion

    def run(
        self,
        arguments: EmptyToolInput,
        context: ToolExecutionContext,
    ) -> ToolObservation:
        del arguments
        history = self._history_reader.user_history(
            context.user_id,
            context.cutoff_time,
        )
        split = str(context.state_snapshot.get("split") or "development")
        task = RetrievalTaskContext(
            task_id=context.request_id,
            split=split,
            user_id=context.user_id,
            cutoff_time=context.cutoff_time,
            history_count=len(history),
        )
        result = self._retriever.retrieve(task, include_route_provenance=True)
        if self._query_retriever is None or self._dual_fusion is None:
            return self._history_observation(result)
        try:
            request = self._request(context)
            query = self._query_retriever.retrieve(
                QueryRetrievalTask(
                    request=request,
                    usage_scope=f"agent:{context.request_id}",
                )
            )
            dual = self._dual_fusion.fuse(result, query)
        except Exception as exc:
            return self._history_observation(
                result,
                retrieval_mode="history_fallback",
                warnings=[f"QUERY_RETRIEVAL_FALLBACK:{type(exc).__name__}"],
            )
        history_by_id = {item.business_id: item for item in result.candidates}
        query_by_id = {item.business_id: item for item in query.candidates}
        rows = []
        for item in dual.candidates:
            history_item = history_by_id.get(item.business_id)
            query_item = query_by_id.get(item.business_id)
            row = (
                asdict(history_item)
                if history_item is not None
                else self._empty_history_row(item.business_id)
            )
            row.update(
                {
                    "rank": item.rank,
                    "fusion_score": item.fusion_score,
                    "route_count": (
                        (0 if history_item is None else history_item.route_count)
                        + (0 if query_item is None else query_item.route_count)
                    ),
                    "history_rank": item.history_rank,
                    "history_fusion_score": item.history_score,
                    "query_rank": item.query_rank,
                    "query_fusion_score": item.query_score,
                    **self._query_fields(query_item),
                    "source_channels": [
                        channel
                        for channel, present in (
                            ("history", history_item is not None),
                            ("query", query_item is not None),
                        )
                        if present
                    ],
                }
            )
            rows.append(row)
        data = {
            "candidate_business_ids": [row["business_id"] for row in rows],
            "candidates": rows,
            "catalog_size": max(result.catalog_size, query.catalog_size),
            "eligible_candidate_count": max(
                result.eligible_candidate_count,
                query.eligible_business_count,
            ),
            "excluded_history_businesses": result.excluded_history_businesses,
            "route_result_counts": {
                **result.route_result_counts,
                **query.route_result_counts,
            },
            "retrieval_mode": "dual_channel",
            "history_candidate_count": dual.history_candidate_count,
            "query_candidate_count": dual.query_candidate_count,
            "overlap_count": dual.overlap_count,
            "query_pre_cutoff_business_count": query.pre_cutoff_business_count,
            "query_eligible_business_count": query.eligible_business_count,
            "query_warnings": query.warnings,
        }
        if not rows:
            return ToolObservation(
                tool_name=self.definition.name,
                status="no_result",
                data=data,
                confidence=1.0,
                warnings=["no eligible businesses were retrieved"],
                latency_ms=dual.latency_ms,
            )
        return ToolObservation(
            tool_name=self.definition.name,
            status="success",
            data=data,
            confidence=1.0,
            warnings=query.warnings,
            input_tokens=query.usage.embedding_input_tokens,
            output_tokens=0,
            cache_hit=(
                query.usage.cache_misses == 0
                and query.usage.embedding_logical_tokens > 0
            ),
            latency_ms=dual.latency_ms,
        )

    def _history_observation(
        self,
        result: RetrievalResult,
        *,
        retrieval_mode: str = "history_only",
        warnings: list[str] | None = None,
    ) -> ToolObservation:
        rows = []
        for candidate in result.candidates:
            row = asdict(candidate)
            row.update(
                {
                    "history_rank": candidate.rank,
                    "history_fusion_score": candidate.fusion_score,
                    "source_channels": ["history"],
                }
            )
            rows.append(row)
        data = {
            "candidate_business_ids": [row["business_id"] for row in rows],
            "candidates": rows,
            "catalog_size": result.catalog_size,
            "eligible_candidate_count": result.eligible_candidate_count,
            "excluded_history_businesses": result.excluded_history_businesses,
            "route_result_counts": result.route_result_counts,
            "retrieval_mode": retrieval_mode,
            "history_candidate_count": len(rows),
            "query_warnings": warnings or [],
        }
        if not rows:
            return ToolObservation(
                tool_name=self.definition.name,
                status="no_result",
                data=data,
                confidence=1.0,
                warnings=[*(warnings or []), "no eligible businesses were retrieved"],
                latency_ms=result.latency_ms,
            )
        return ToolObservation(
            tool_name=self.definition.name,
            status="success",
            data=data,
            confidence=1.0,
            warnings=warnings or [],
            latency_ms=result.latency_ms,
        )

    @staticmethod
    def _request(context: ToolExecutionContext) -> RecommendationRequest:
        raw = context.state_snapshot.get("request")
        if not isinstance(raw, dict):
            raise ValueError("current structured request is absent from Agent state")
        return RecommendationRequest.model_validate(
            {
                key: value
                for key, value in raw.items()
                if key in RecommendationRequest.model_fields
            }
        )

    @staticmethod
    def _query_fields(
        item: QueryRetrievalCandidate | None,
    ) -> dict[str, object]:
        return {
            "query_category_rank": None if item is None else item.category_rank,
            "query_category_score": None if item is None else item.category_score,
            "query_embedding_rank": None if item is None else item.embedding_rank,
            "query_embedding_score": None if item is None else item.embedding_score,
            "query_aspect_rank": None if item is None else item.aspect_rank,
            "query_aspect_score": None if item is None else item.aspect_score,
            "query_location_rank": None if item is None else item.location_rank,
            "query_location_score": None if item is None else item.location_score,
            "query_distance_km": None if item is None else item.distance_km,
        }

    @staticmethod
    def _empty_history_row(business_id: str) -> dict[str, object]:
        return {
            "business_id": business_id,
            "rank": 1,
            "fusion_score": 1.0,
            "route_count": 1,
            "quality_rank": None,
            "quality_score": None,
            "category_rank": None,
            "category_score": None,
            "text_rank": None,
            "text_score": None,
            "location_rank": None,
            "location_score": None,
            "distance_km": None,
            "item_knn_rank": None,
            "item_knn_positive_score": 0.0,
            "item_knn_negative_evidence": 0.0,
            "item_knn_positive_support_count": 0,
            "item_knn_negative_support_count": 0,
            "item_knn_positive_neighbor_count": 0,
            "item_knn_negative_neighbor_count": 0,
            "item_knn_missing": True,
        }
