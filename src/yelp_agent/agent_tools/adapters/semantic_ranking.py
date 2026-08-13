"""Agent tool Adapter for Step 30 structured semantic ranking."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from yelp_agent.cross_encoder import fuse_ranking_and_cross_encoder
from yelp_agent.query.schema import RecommendationRequest
from yelp_agent.semantic_embedding import fuse_hybrid_and_semantic
from yelp_agent.semantic_ranking import SemanticRankingResult

from ..registry import ToolDefinition
from ..request_context import request_from_tool_context
from ..schema import ToolExecutionContext, ToolObservation
from ..tool_schemas import CandidateBusinessIdsInput, SemanticRankingOutput


class SemanticRankingService(Protocol):
    def rank(
        self,
        *,
        request: RecommendationRequest,
        base_ranking: list[str],
        cutoff_time: datetime,
        usage_scope: str | None = None,
    ) -> SemanticRankingResult: ...


class ApplySemanticRankingTool:
    definition = ToolDefinition(
        name="APPLY_SEMANTIC_RANKING",
        version="1.0.0",
        kind="semantic",
        allowed_actions=("rank_candidates",),
        input_model=CandidateBusinessIdsInput,
        output_model=SemanticRankingOutput,
        public_summary=(
            "Apply accepted structured request semantics through aggressive or "
            "rank-protected deterministic fusion."
        ),
        preconditions=(
            "GET_HYBRID_RANKING observation exists",
            "COMPUTE_EMBEDDING_MATCH observation exists",
            "COMPUTE_CROSS_ENCODER_MATCH observation exists",
            "business IDs equal the current Step-26 ranking",
        ),
        resolves_uncertainties=("structured_request_ranking_effect",),
        cache_scope="none",
        max_attempts=1,
        timeout_ms=90_000,
        fallback_policy="safe_step26_ranking",
    )

    def __init__(
        self,
        service: SemanticRankingService,
        *,
        embedding_alpha: float,
        cross_encoder_beta: float,
    ) -> None:
        self._service = service
        self._embedding_alpha = embedding_alpha
        self._cross_encoder_beta = cross_encoder_beta

    def run(
        self,
        arguments: CandidateBusinessIdsInput,
        context: ToolExecutionContext,
    ) -> ToolObservation:
        base = self._step26_ranking(context)
        if arguments.business_ids != base:
            return ToolObservation.error(
                tool_name=self.definition.name,
                status="permanent_error",
                error_code="STEP26_RANKING_REQUIRED",
                warning="semantic ranking requires the complete current Step-26 order",
            )
        try:
            request = request_from_tool_context(context)
        except (TypeError, ValueError):
            return ToolObservation.error(
                tool_name=self.definition.name,
                status="permanent_error",
                error_code="REQUEST_MISSING",
                warning="canonical request is missing from visible state",
            )
        result = self._service.rank(
            request=request,
            base_ranking=base,
            cutoff_time=context.cutoff_time,
            usage_scope=str(
                context.state_snapshot.get("scenario_id") or context.request_id
            ),
        )
        warnings = []
        if result.fallback:
            warnings.append(
                "semantic ranking failed safely and preserved the Step-26 ranking"
            )
        if result.no_op_reason is not None:
            warnings.append(result.no_op_reason)
        return ToolObservation(
            tool_name=self.definition.name,
            status="partial" if result.fallback else "success",
            data=result.model_dump(mode="json"),
            confidence=result.intent.mean_confidence,
            warnings=warnings,
            input_tokens=result.usage.actual_input_tokens,
            output_tokens=0,
            cache_hit=result.usage.cache_misses == 0,
        )

    def _step26_ranking(self, context: ToolExecutionContext) -> list[str]:
        observations = context.state_snapshot.get("observations")
        if not isinstance(observations, list):
            return []
        hybrid: list[str] = []
        embedding: dict[str, int] = {}
        cross: dict[str, int] = {}
        for observation in observations:
            if not isinstance(observation, dict):
                continue
            payload = observation.get("payload")
            if not isinstance(payload, dict):
                continue
            data = payload.get("data")
            if not isinstance(data, dict):
                continue
            tool_name = payload.get("tool_name")
            if tool_name == "GET_HYBRID_RANKING":
                values = data.get("ranking")
                if isinstance(values, list):
                    hybrid = [value for value in values if isinstance(value, str)]
            elif tool_name == "COMPUTE_EMBEDDING_MATCH":
                embedding = self._ranks(data.get("matches"), "semantic_rank")
            elif tool_name == "COMPUTE_CROSS_ENCODER_MATCH":
                cross = self._ranks(data.get("matches"), "cross_encoder_rank")
        step25 = fuse_hybrid_and_semantic(
            hybrid,
            embedding,
            alpha=self._embedding_alpha,
        )
        return fuse_ranking_and_cross_encoder(
            step25,
            cross,
            beta=self._cross_encoder_beta,
        )

    @staticmethod
    def _ranks(value: object, key: str) -> dict[str, int]:
        if not isinstance(value, list):
            return {}
        return {
            str(row["business_id"]): int(row[key])
            for row in value
            if isinstance(row, dict)
            and isinstance(row.get("business_id"), str)
            and isinstance(row.get(key), int)
        }
