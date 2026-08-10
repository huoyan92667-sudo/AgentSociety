"""Deterministic next-action selection for Agent V1."""

from __future__ import annotations

from yelp_agent.agent_harness.schema import AgentDecision, AgentState

from .clarification import GAP_REASON_CODES, highest_priority_gap
from .state_view import RouteFacts


class RuleRouter:
    """Choose exactly one structured action from visible state facts."""

    def __init__(self, *, display_limit: int = 3) -> None:
        if not 1 <= display_limit <= 100:
            raise ValueError("display_limit must be between 1 and 100")
        self._display_limit = display_limit

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
                rejected = (
                    facts.previous_recommended_business_ids[:1]
                    if facts.reject_previous_recommendation
                    else []
                )
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
        display_ids = facts.ranked_business_ids[: self._display_limit]
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

    @staticmethod
    def _review_experience_decision(facts: RouteFacts) -> AgentDecision:
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

    @staticmethod
    def _comparison_decision(facts: RouteFacts) -> AgentDecision:
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
