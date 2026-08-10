"""Online Hybrid V2 ranking tool using retrieval observations as input."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Protocol

from ..registry import ToolDefinition
from ..schema import ToolExecutionContext, ToolObservation
from ..tool_schemas import CandidateBusinessIdsInput, HybridRankingOutput


class HybridRankingService(Protocol):
    def rank(
        self,
        *,
        request_id: str,
        user_id: str,
        cutoff_time: datetime,
        candidates: Sequence[Mapping[str, object]],
    ) -> Sequence[Mapping[str, object]]: ...


class GetHybridRankingTool:
    definition = ToolDefinition(
        name="GET_HYBRID_RANKING",
        version="2.0.0",
        kind="deterministic",
        allowed_actions=("rank_candidates",),
        input_model=CandidateBusinessIdsInput,
        output_model=HybridRankingOutput,
        public_summary=(
            "Rank the scoped retrieval set with the frozen online Hybrid V2 model."
        ),
        preconditions=(
            "EXPAND_CANDIDATES observation exists",
            "business IDs are inside current scope",
        ),
        resolves_uncertainties=("candidate_order_missing",),
    )

    def __init__(self, service: HybridRankingService) -> None:
        self._service = service

    def run(
        self,
        arguments: CandidateBusinessIdsInput,
        context: ToolExecutionContext,
    ) -> ToolObservation:
        candidates = self._retrieval_candidates(context)
        by_id = {str(row.get("business_id")): row for row in candidates}
        if any(business_id not in by_id for business_id in arguments.business_ids):
            return ToolObservation.error(
                tool_name=self.definition.name,
                status="permanent_error",
                error_code="RETRIEVAL_FEATURES_MISSING",
                warning="one or more scoped businesses have no retrieval features",
            )
        selected = [by_id[business_id] for business_id in arguments.business_ids]
        ranked = list(
            self._service.rank(
                request_id=context.request_id,
                user_id=context.user_id,
                cutoff_time=context.cutoff_time,
                candidates=selected,
            )
        )
        ranking = [str(row.get("business_id")) for row in ranked]
        if set(ranking) != set(arguments.business_ids) or len(ranking) != len(
            arguments.business_ids
        ):
            return ToolObservation.error(
                tool_name=self.definition.name,
                status="permanent_error",
                error_code="INCOMPLETE_RANKING",
                warning="Hybrid V2 did not return a complete candidate permutation",
            )
        return ToolObservation.success(
            tool_name=self.definition.name,
            data={"ranking": ranking, "scored_candidates": ranked},
            confidence=1.0,
        )

    @staticmethod
    def _retrieval_candidates(
        context: ToolExecutionContext,
    ) -> list[dict[str, object]]:
        observations = context.state_snapshot.get("observations")
        if not isinstance(observations, list):
            return []
        for observation in reversed(observations):
            if not isinstance(observation, dict):
                continue
            payload = observation.get("payload")
            if not isinstance(payload, dict):
                continue
            if payload.get("tool_name") != "EXPAND_CANDIDATES":
                continue
            data = payload.get("data")
            rows = data.get("candidates") if isinstance(data, dict) else None
            if isinstance(rows, list):
                return [dict(row) for row in rows if isinstance(row, dict)]
        return []
