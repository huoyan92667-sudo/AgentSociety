"""Code-enforced dynamic action policy for Rule and model Routers."""

from __future__ import annotations

from yelp_agent.agent_benchmark.schema import AgentAction
from yelp_agent.agent_harness.schema import AgentState

from .state_view import RouteFacts


class RuleBasedActionPolicy:
    """Return only actions that are safe and useful in the current state."""

    def __init__(
        self,
        *,
        display_limit: int = 3,
        semantic_enabled: bool = False,
        fusion_alpha: float = 0.0,
    ) -> None:
        if not 1 <= display_limit <= 100:
            raise ValueError("display_limit must be between 1 and 100")
        self._display_limit = display_limit
        self._semantic_enabled = semantic_enabled
        if not 0 <= fusion_alpha <= 1:
            raise ValueError("fusion alpha must be between zero and one")
        self._fusion_alpha = fusion_alpha

    def allowed_actions(self, state: AgentState) -> tuple[AgentAction, ...]:
        facts = RouteFacts.from_state(state)
        if facts.information_gaps:
            return ("ask_clarification", "safe_fallback")
        if facts.task_type == "feedback_refinement" and not facts.feedback_applied:
            return ("apply_feedback", "safe_fallback")
        if facts.task_type in {"recommendation_request", "feedback_refinement"}:
            return self._recommendation_actions(facts)
        if facts.task_type == "business_detail_question":
            if not set(facts.referenced_business_ids).issubset(
                facts.detailed_business_ids
            ):
                return ("get_business_details", "safe_fallback")
            return ("return_grounded_answer", "safe_fallback")
        if facts.task_type == "review_experience_question":
            if not set(facts.referenced_business_ids).issubset(
                facts.profiled_business_ids
            ):
                return ("get_business_details", "safe_fallback")
            if facts.structured_evidence_sufficient:
                return ("return_grounded_answer", "safe_fallback")
            return ("return_uncertain_answer", "safe_fallback")
        if facts.task_type == "official_policy_question":
            if not set(facts.referenced_business_ids).issubset(
                facts.detailed_business_ids
            ):
                return ("get_business_details", "safe_fallback")
            return ("return_uncertain_answer", "safe_fallback")
        if facts.task_type == "candidate_comparison":
            if not set(facts.referenced_business_ids).issubset(
                facts.profiled_business_ids
            ):
                return ("get_business_details", "safe_fallback")
            if facts.comparison is None:
                return ("compare_candidates", "safe_fallback")
            if facts.comparison_ranking:
                return ("return_grounded_answer", "safe_fallback")
            return ("return_uncertain_answer", "safe_fallback")
        return ("safe_fallback",)

    def _recommendation_actions(
        self,
        facts: RouteFacts,
    ) -> tuple[AgentAction, ...]:
        if facts.candidate_retrieval is None or not facts.business_scope_known:
            return ("retrieve_candidates", "safe_fallback")
        if not facts.candidate_ids:
            return ("return_uncertain_answer", "safe_fallback")
        if facts.hard_constraints_required and facts.constraint_filter is None:
            return ("apply_hard_constraints", "safe_fallback")
        if facts.hybrid_ranking is None:
            return ("rank_candidates", "safe_fallback")
        if (
            self._semantic_enabled
            and facts.semantic_match is None
            and facts.remaining.semantic_calls > 0
        ):
            return ("rank_candidates", "safe_fallback")
        display_ids = facts.final_ranking(
            fusion_alpha=self._fusion_alpha
        )[: self._display_limit]
        if not set(display_ids).issubset(facts.detailed_business_ids):
            return ("get_business_details", "safe_fallback")
        return ("return_recommendation", "safe_fallback")
