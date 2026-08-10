"""Target-blind multi-route candidate expansion adapter."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from typing import Protocol, Sequence

from yelp_agent.retrieval import RetrievalResult, RetrievalTaskContext

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
        version="1.0.0",
        kind="deterministic",
        allowed_actions=("retrieve_candidates",),
        input_model=EmptyToolInput,
        output_model=CandidateRetrievalOutput,
        public_summary=(
            "Retrieve a target-blind full-catalog candidate set using five routes."
        ),
        preconditions=("user has at least one pre-cutoff interaction",),
        resolves_uncertainties=("candidate_set_missing",),
    )

    def __init__(
        self,
        retriever: CandidateRetriever,
        history_reader: HistoryReader,
    ) -> None:
        self._retriever = retriever
        self._history_reader = history_reader

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
        rows = [asdict(candidate) for candidate in result.candidates]
        data = {
            "candidate_business_ids": [row["business_id"] for row in rows],
            "candidates": rows,
            "catalog_size": result.catalog_size,
            "eligible_candidate_count": result.eligible_candidate_count,
            "excluded_history_businesses": result.excluded_history_businesses,
            "route_result_counts": result.route_result_counts,
        }
        if not rows:
            return ToolObservation(
                tool_name=self.definition.name,
                status="no_result",
                data=data,
                confidence=1.0,
                warnings=["no eligible businesses were retrieved"],
                latency_ms=result.latency_ms,
            )
        return ToolObservation(
            tool_name=self.definition.name,
            status="success",
            data=data,
            confidence=1.0,
            latency_ms=result.latency_ms,
        )
