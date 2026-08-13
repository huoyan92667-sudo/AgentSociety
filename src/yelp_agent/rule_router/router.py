"""Deterministic next-action selection for Agent V1."""

from __future__ import annotations

from yelp_agent.agent_harness.schema import AgentDecision, AgentState

from .clarification import GAP_REASON_CODES, highest_priority_gap
from .state_view import RouteFacts


class RuleRouter:
    """Choose exactly one structured action from visible state facts."""

    def __init__(
        self,
        *,
        display_limit: int = 3,
        semantic_enabled: bool = False,
        semantic_candidate_limit: int = 30,
        fusion_alpha: float = 0.0,
        cross_encoder_enabled: bool = False,
        cross_encoder_candidate_limit: int = 20,
        cross_encoder_beta: float = 0.0,
        review_rag_enabled: bool = False,
        evidence_aggregation_enabled: bool = False,
        semantic_ranking_enabled: bool = False,
    ) -> None:
        if not 1 <= display_limit <= 100:
            raise ValueError("display_limit must be between 1 and 100")
        self._display_limit = display_limit
        if not 1 <= semantic_candidate_limit <= 500:
            raise ValueError("semantic candidate limit must be between 1 and 500")
        if not 0 <= fusion_alpha <= 1:
            raise ValueError("fusion alpha must be between zero and one")
        self._semantic_enabled = semantic_enabled
        self._semantic_candidate_limit = semantic_candidate_limit
        self._fusion_alpha = fusion_alpha
        if not 1 <= cross_encoder_candidate_limit <= 100:
            raise ValueError("Cross-Encoder candidate limit must be between 1 and 100")
        if not 0 <= cross_encoder_beta <= 1:
            raise ValueError("Cross-Encoder beta must be between zero and one")
        self._cross_encoder_enabled = cross_encoder_enabled
        self._cross_encoder_candidate_limit = cross_encoder_candidate_limit
        self._cross_encoder_beta = cross_encoder_beta
        self._review_rag_enabled = review_rag_enabled
        self._evidence_aggregation_enabled = evidence_aggregation_enabled
        self._semantic_ranking_enabled = semantic_ranking_enabled

    def choose_action(self, state: AgentState) -> AgentDecision:
        facts = RouteFacts.from_state(state)
        if facts.information_gaps:
            gap = highest_priority_gap(facts.information_gaps)
            return AgentDecision(
                action="ask_clarification",
                arguments={"information_gaps": [gap]},
                reason_code=GAP_REASON_CODES[gap],
            )
        if facts.task_type in {"recommendation_request", "feedback_refinement"}:
            if facts.task_type == "feedback_refinement" and not facts.feedback_applied:
                rejected = [
                    business_id
                    for business_id in facts.rejected_business_ids
                    if not facts.business_scope_known
                    or business_id in facts.candidate_ids
                ]
                if not rejected and facts.reject_previous_recommendation:
                    rejected = facts.previous_recommended_business_ids[:1]
                return AgentDecision(
                    action="apply_feedback",
                    arguments={"rejected_business_ids": rejected},
                    reason_code="FEEDBACK_CONTEXT_REQUIRED",
                )
            return self._recommendation_decision(facts)
        if facts.task_type == "business_detail_question":
            return self._business_detail_decision(facts)
        if facts.task_type == "review_experience_question":
            return self._review_experience_decision(facts)
        if facts.task_type == "official_policy_question":
            return self._official_policy_decision(facts)
        if facts.task_type == "candidate_comparison":
            return self._comparison_decision(facts)
        return AgentDecision(
            action="safe_fallback",
            reason_code="UNSUPPORTED_TASK",
        )

    def _recommendation_decision(self, facts: RouteFacts) -> AgentDecision:
        if facts.candidate_retrieval is None or not facts.business_scope_known:
            return AgentDecision(
                action="retrieve_candidates",
                reason_code="CANDIDATES_REQUIRED",
                tool_name="EXPAND_CANDIDATES",
                tool_kind="deterministic",
            )
        if not facts.candidate_ids:
            return AgentDecision(
                action="return_uncertain_answer",
                arguments={"reason": "no_eligible_candidates"},
                reason_code="NO_ELIGIBLE_CANDIDATES",
            )
        if facts.hard_constraints_required and facts.constraint_filter is None:
            return AgentDecision(
                action="apply_hard_constraints",
                arguments={"business_ids": facts.candidate_ids},
                reason_code="HARD_CONSTRAINT_PRESENT",
                tool_name="APPLY_CONSTRAINTS",
                tool_kind="deterministic",
            )
        if facts.hybrid_ranking is None:
            return AgentDecision(
                action="rank_candidates",
                arguments={"business_ids": facts.candidate_ids},
                reason_code="RANKING_REQUIRED",
                tool_name="GET_HYBRID_RANKING",
                tool_kind="deterministic",
            )
        if (
            self._semantic_enabled
            and facts.semantic_match is None
            and facts.remaining.semantic_calls > 0
        ):
            return AgentDecision(
                action="rank_candidates",
                arguments={
                    "business_ids": facts.ranked_business_ids[
                        : self._semantic_candidate_limit
                    ]
                },
                reason_code="SEMANTIC_MATCH_REQUIRED",
                tool_name="COMPUTE_EMBEDDING_MATCH",
                tool_kind="semantic",
            )
        if (
            self._cross_encoder_enabled
            and facts.cross_encoder_match is None
            and facts.remaining.semantic_calls > 0
        ):
            step25_ranking = facts.embedding_ranking(
                fusion_alpha=self._fusion_alpha
            )
            return AgentDecision(
                action="rank_candidates",
                arguments={
                    "business_ids": step25_ranking[
                        : self._cross_encoder_candidate_limit
                    ]
                },
                reason_code="CROSS_ENCODER_MATCH_REQUIRED",
                tool_name="COMPUTE_CROSS_ENCODER_MATCH",
                tool_kind="semantic",
            )
        if (
            self._semantic_ranking_enabled
            and facts.semantic_ranking is None
            and facts.remaining.semantic_calls > 0
        ):
            step26_ranking = facts.final_ranking(
                fusion_alpha=self._fusion_alpha,
                cross_encoder_beta=self._cross_encoder_beta,
            )
            return AgentDecision(
                action="rank_candidates",
                arguments={"business_ids": step26_ranking},
                reason_code="STRUCTURED_SEMANTIC_RANKING_REQUIRED",
                tool_name="APPLY_SEMANTIC_RANKING",
                tool_kind="semantic",
            )
        final_ranking = facts.final_ranking(
            fusion_alpha=self._fusion_alpha,
            cross_encoder_beta=self._cross_encoder_beta,
        )
        display_ids = final_ranking[: self._display_limit]
        missing_details = [
            business_id
            for business_id in display_ids
            if business_id not in facts.detailed_business_ids
        ]
        if missing_details:
            return AgentDecision(
                action="get_business_details",
                arguments={"business_ids": missing_details},
                reason_code="BUSINESS_DETAILS_REQUIRED",
                tool_name="GET_BUSINESS_DETAILS",
                tool_kind="deterministic",
            )
        return AgentDecision(
            action="return_recommendation",
            arguments={"business_ids": display_ids},
            reason_code="READY_TO_FINALIZE",
        )

    @staticmethod
    def _business_detail_decision(facts: RouteFacts) -> AgentDecision:
        missing = [
            business_id
            for business_id in facts.referenced_business_ids
            if business_id not in facts.detailed_business_ids
        ]
        if missing:
            return AgentDecision(
                action="get_business_details",
                arguments={"business_ids": missing},
                reason_code="BUSINESS_DETAILS_REQUIRED",
                tool_name="GET_BUSINESS_DETAILS",
                tool_kind="deterministic",
            )
        return AgentDecision(
            action="return_grounded_answer",
            arguments={"business_ids": facts.referenced_business_ids},
            reason_code="STRUCTURED_EVIDENCE_SUFFICIENT",
        )

    def _review_experience_decision(self, facts: RouteFacts) -> AgentDecision:
        if self._review_rag_enabled and facts.review_search is None:
            return AgentDecision(
                action="retrieve_business_reviews",
                arguments={"business_ids": facts.referenced_business_ids, "top_k": 5},
                reason_code="UNSTRUCTURED_EVIDENCE_REQUIRED",
                tool_name="SEARCH_BUSINESS_REVIEWS",
                tool_kind="review_rag",
            )
        if self._review_rag_enabled:
            if facts.review_evidence_count == 0:
                return AgentDecision(
                    action="return_uncertain_answer",
                    arguments={"reason": "no_review_evidence_before_cutoff"},
                    reason_code="NO_REVIEW_EVIDENCE",
                )
            if (
                self._evidence_aggregation_enabled
                and facts.evidence_aggregation is None
            ):
                return AgentDecision(
                    action="retrieve_business_reviews",
                    arguments={"business_ids": facts.referenced_business_ids},
                    reason_code="REVIEW_EVIDENCE_AGGREGATION_REQUIRED",
                    tool_name="AGGREGATE_REVIEW_EVIDENCE",
                    tool_kind="review_rag",
                )
            if self._evidence_aggregation_enabled:
                if not facts.evidence_aggregation_has_atoms:
                    return AgentDecision(
                        action="return_uncertain_answer",
                        arguments={"reason": "no_aspect_evidence_before_cutoff"},
                        reason_code="NO_REVIEW_EVIDENCE",
                    )
                if (
                    facts.evidence_aggregation_conflict
                    or not facts.evidence_aggregation_sufficient
                ):
                    return AgentDecision(
                        action="return_uncertain_answer",
                        arguments={"reason": "aggregated_review_evidence_uncertain"},
                        reason_code=(
                            "CONFLICTING_REVIEW_EVIDENCE"
                            if facts.evidence_aggregation_conflict
                            else "SPARSE_REVIEW_EVIDENCE"
                        ),
                    )
                return AgentDecision(
                    action="return_grounded_answer",
                    arguments={"business_ids": facts.referenced_business_ids},
                    reason_code="AGGREGATED_REVIEW_EVIDENCE_SUFFICIENT",
                )
            if facts.review_evidence_conflict or facts.explicit_uncertainty_request:
                return AgentDecision(
                    action="return_uncertain_answer",
                    arguments={"reason": "review_evidence_uncertain"},
                    reason_code="CONFLICTING_REVIEW_EVIDENCE",
                )
            return AgentDecision(
                action="return_grounded_answer",
                arguments={"business_ids": facts.referenced_business_ids},
                reason_code="REVIEW_EVIDENCE_SUFFICIENT",
            )
        missing = [
            business_id
            for business_id in facts.referenced_business_ids
            if business_id not in facts.profiled_business_ids
        ]
        if missing:
            return AgentDecision(
                action="get_business_details",
                arguments={"business_ids": missing},
                reason_code="BUSINESS_PROFILE_REQUIRED",
                tool_name="GET_BUSINESS_PROFILE",
                tool_kind="deterministic",
            )
        if facts.structured_evidence_sufficient:
            return AgentDecision(
                action="return_grounded_answer",
                arguments={"business_ids": facts.referenced_business_ids},
                reason_code="STRUCTURED_EVIDENCE_SUFFICIENT",
            )
        return AgentDecision(
            action="return_uncertain_answer",
            arguments={"reason": "review_rag_unavailable"},
            reason_code="UNSTRUCTURED_EVIDENCE_REQUIRED",
        )

    @staticmethod
    def _official_policy_decision(facts: RouteFacts) -> AgentDecision:
        missing = [
            business_id
            for business_id in facts.referenced_business_ids
            if business_id not in facts.detailed_business_ids
        ]
        if missing:
            return AgentDecision(
                action="get_business_details",
                arguments={"business_ids": missing},
                reason_code="BUSINESS_DETAILS_REQUIRED",
                tool_name="GET_BUSINESS_DETAILS",
                tool_kind="deterministic",
            )
        return AgentDecision(
            action="return_uncertain_answer",
            arguments={
                "reason": "official_verification_required",
                "recommended_official_verification": True,
            },
            reason_code="OFFICIAL_VERIFICATION_REQUIRED",
        )

    def _comparison_decision(self, facts: RouteFacts) -> AgentDecision:
        missing = [
            business_id
            for business_id in facts.referenced_business_ids
            if business_id not in facts.profiled_business_ids
        ]
        if missing:
            return AgentDecision(
                action="get_business_details",
                arguments={"business_ids": missing},
                reason_code="BUSINESS_PROFILE_REQUIRED",
                tool_name="GET_BUSINESS_PROFILE",
                tool_kind="deterministic",
            )
        if (
            self._review_rag_enabled
            and facts.requested_aspects
            and facts.review_search is None
        ):
            return AgentDecision(
                action="retrieve_business_reviews",
                arguments={"business_ids": facts.referenced_business_ids, "top_k": 5},
                reason_code="COMPARISON_REVIEW_EVIDENCE_REQUIRED",
                tool_name="SEARCH_BUSINESS_REVIEWS",
                tool_kind="review_rag",
            )
        if (
            self._evidence_aggregation_enabled
            and facts.review_search is not None
            and facts.review_evidence_count > 0
            and facts.evidence_aggregation is None
        ):
            return AgentDecision(
                action="retrieve_business_reviews",
                arguments={"business_ids": facts.referenced_business_ids},
                reason_code="COMPARISON_EVIDENCE_AGGREGATION_REQUIRED",
                tool_name="AGGREGATE_REVIEW_EVIDENCE",
                tool_kind="review_rag",
            )
        if facts.comparison is None:
            return AgentDecision(
                action="compare_candidates",
                arguments={"business_ids": facts.referenced_business_ids},
                reason_code="COMPARISON_REQUIRED",
                tool_name="COMPARE_BUSINESSES",
                tool_kind="deterministic",
            )
        if facts.comparison_ranking:
            return AgentDecision(
                action="return_grounded_answer",
                arguments={"business_ids": facts.comparison_ranking},
                reason_code="STRUCTURED_EVIDENCE_SUFFICIENT",
            )
        return AgentDecision(
            action="return_uncertain_answer",
            arguments={"reason": "comparison_has_no_eligible_business"},
            reason_code="NO_ELIGIBLE_CANDIDATES",
        )
