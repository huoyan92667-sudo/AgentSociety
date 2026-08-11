"""Safe Hybrid V2 fallback for controlled Agent execution failures."""

from __future__ import annotations

from yelp_agent.agent_harness.schema import ActionOutcome, AgentSession
from yelp_agent.cross_encoder import fuse_ranking_and_cross_encoder
from yelp_agent.semantic_embedding import fuse_hybrid_and_semantic

from .adapters.ranking import HybridRankingService


class HybridV2FallbackHandler:
    """Reuse known scope, preferring an existing or recomputed Hybrid V2 order."""

    def __init__(self, ranking_service: HybridRankingService) -> None:
        self._ranking_service = ranking_service

    def fallback(self, state: AgentSession, reason: str) -> ActionOutcome:
        del reason
        existing = self._latest_ranking(state)
        scope = set(state.business_scope) if state.business_scope_known else None
        if existing:
            ranking = [value for value in existing if scope is None or value in scope]
            return self._outcome(ranking)

        candidates = self._retrieval_candidates(state)
        if scope is not None:
            candidates = [
                row for row in candidates if str(row.get("business_id")) in scope
            ]
        if not candidates:
            return self._outcome([])
        expected = [str(row.get("business_id")) for row in candidates]
        try:
            ranked = self._ranking_service.rank(
                request_id=state.request.request_id,
                user_id=state.user_id,
                cutoff_time=state.cutoff_time,
                candidates=candidates,
            )
            ranking = [str(row.get("business_id")) for row in ranked]
            if len(ranking) != len(expected) or set(ranking) != set(expected):
                raise ValueError("fallback ranking is not a complete permutation")
        except Exception:
            ranking = [
                str(row.get("business_id"))
                for row in sorted(
                    candidates,
                    key=lambda row: (
                        int(row.get("rank") or 10**9),
                        str(row.get("business_id")),
                    ),
                )
            ]
        return self._outcome(ranking)

    @staticmethod
    def _outcome(ranking: list[str]) -> ActionOutcome:
        return ActionOutcome(
            status="completed",
            response_kind="fallback",
            candidate_ranking=ranking,
            recommended_business_ids=ranking[:1],
        )

    @staticmethod
    def _latest_ranking(state: AgentSession) -> list[str]:
        for observation in reversed(state.observations):
            payload = observation.payload
            if payload.get("tool_name") != "GET_HYBRID_RANKING":
                continue
            data = payload.get("data")
            ranking = data.get("ranking") if isinstance(data, dict) else None
            if isinstance(ranking, list):
                return [str(value) for value in ranking]
        return []

    @staticmethod
    def _retrieval_candidates(state: AgentSession) -> list[dict[str, object]]:
        for observation in reversed(state.observations):
            payload = observation.payload
            if payload.get("tool_name") != "EXPAND_CANDIDATES":
                continue
            data = payload.get("data")
            rows = data.get("candidates") if isinstance(data, dict) else None
            if isinstance(rows, list):
                return [dict(row) for row in rows if isinstance(row, dict)]
        return []


class RankingCascadeFallbackHandler(HybridV2FallbackHandler):
    """Fall back Cross-Encoder -> Step 25 -> Hybrid without losing a batch."""

    def __init__(
        self,
        ranking_service: HybridRankingService,
        *,
        display_limit: int,
        embedding_alpha: float,
        cross_encoder_beta: float,
    ) -> None:
        super().__init__(ranking_service)
        self._display_limit = display_limit
        self._embedding_alpha = embedding_alpha
        self._cross_encoder_beta = cross_encoder_beta

    def fallback(self, state: AgentSession, reason: str) -> ActionOutcome:
        baseline = super().fallback(state, reason)
        ranking = list(baseline.candidate_ranking)
        if ranking:
            semantic = self._ranks(state, "COMPUTE_EMBEDDING_MATCH", "semantic_rank")
            cross = self._ranks(
                state, "COMPUTE_CROSS_ENCODER_MATCH", "cross_encoder_rank"
            )
            try:
                ranking = fuse_hybrid_and_semantic(
                    ranking, semantic, alpha=self._embedding_alpha
                )
                ranking = fuse_ranking_and_cross_encoder(
                    ranking, cross, beta=self._cross_encoder_beta
                )
            except ValueError:
                # Malformed optional evidence must never destroy the valid Hybrid order.
                ranking = list(baseline.candidate_ranking)
        return ActionOutcome(
            status="completed",
            response_kind="fallback",
            candidate_ranking=ranking,
            recommended_business_ids=ranking[: self._display_limit],
        )

    @staticmethod
    def _ranks(
        state: AgentSession, tool_name: str, rank_field: str
    ) -> dict[str, int]:
        for observation in reversed(state.observations):
            payload = observation.payload
            if payload.get("tool_name") != tool_name:
                continue
            data = payload.get("data")
            rows = data.get("matches") if isinstance(data, dict) else None
            if not isinstance(rows, list):
                return {}
            return {
                str(row["business_id"]): int(row[rank_field])
                for row in rows
                if isinstance(row, dict)
                and isinstance(row.get("business_id"), str)
                and isinstance(row.get(rank_field), int)
            }
        return {}
