"""Build complete safe decisions before the model is invoked."""

from __future__ import annotations

import hashlib
import json

from yelp_agent.agent_harness.schema import AgentDecision, AgentState
from yelp_agent.agent_harness.validation import decision_rejection
from yelp_agent.decision_readiness.schema import TaskType
from yelp_agent.rule_router.clarification import GAP_REASON_CODES, highest_priority_gap
from yelp_agent.rule_router.router import RuleRouter

from .schema import RouterChoice

_TASKS: tuple[TaskType, ...] = (
    "recommendation_request",
    "feedback_refinement",
    "business_detail_question",
    "candidate_comparison",
    "review_experience_question",
    "official_policy_question",
)

_SUMMARIES = {
    "ask_clarification": "Ask only for a blocking piece of missing information.",
    "apply_feedback": "Apply the user's visible rejection or refinement to the current scope.",
    "retrieve_candidates": "Retrieve a fresh query-and-history-aware candidate pool.",
    "apply_hard_constraints": "Remove candidates that violate explicit mandatory conditions.",
    "rank_candidates": "Run the named ranking stage over the code-selected candidate scope.",
    "get_business_details": "Read cutoff-safe details or profiles for explicitly scoped businesses.",
    "retrieve_business_reviews": "Retrieve or aggregate cutoff-safe review evidence for locked businesses.",
    "compare_candidates": "Compare explicitly referenced candidates using the final request.",
    "return_recommendation": "Return the verified prefix of the current final ranking.",
    "return_grounded_answer": "Answer only from evidence already present in state.",
    "return_uncertain_answer": "Abstain or answer conservatively because evidence is insufficient.",
    "safe_fallback": "Use the deterministic safety fallback.",
}


class ConstrainedDecisionBuilder:
    """Expose safe alternative decisions while hiding all argument construction."""

    def __init__(self, rule_router: RuleRouter, *, maximum_choices: int = 10) -> None:
        self._rule_router = rule_router
        self._maximum_choices = maximum_choices

    def build(self, state: AgentState) -> tuple[tuple[RouterChoice, AgentDecision], ...]:
        decisions: list[AgentDecision] = []
        routed_state = state
        if state.readiness.information_gaps:
            gap = highest_priority_gap(list(state.readiness.information_gaps))
            decisions.append(
                AgentDecision(
                    action="ask_clarification",
                    arguments={"information_gaps": [gap]},
                    reason_code=GAP_REASON_CODES[gap],
                    routed_task_type=state.readiness.task_type,
                    routed_information_gaps=list(
                        state.readiness.information_gaps
                    ),
                )
            )
            if not self._can_reconsider_ambiguous_reference(state):
                decision = decisions[0]
                return ((self._choice(decision), decision),)
            routed_state = state.model_copy(
                update={
                    "readiness": state.readiness.model_copy(
                        update={"information_gaps": []}
                    )
                }
            )
        elif self._should_offer_conflict_clarification(state):
            decisions.append(
                AgentDecision(
                    action="ask_clarification",
                    arguments={"information_gaps": ["constraint_conflict"]},
                    reason_code=GAP_REASON_CODES["constraint_conflict"],
                    routed_task_type=state.readiness.task_type,
                    routed_information_gaps=["constraint_conflict"],
                )
            )

        ordered_tasks = [state.readiness.task_type]
        # The model resolves semantic ambiguity once at the start of a user
        # turn.  After one action has been accepted, the corrected task is
        # locked and deterministic prerequisite steps no longer pay for the
        # same task-classification decision repeatedly.
        if state.step_count == 0:
            ordered_tasks.extend(task for task in _TASKS if task not in ordered_tasks)
        for task_type in ordered_tasks:
            if not self._task_plausible(routed_state, task_type):
                continue
            routed = routed_state.model_copy(
                update={
                    "readiness": routed_state.readiness.model_copy(
                        update={"task_type": task_type}
                    )
                }
            )
            try:
                decision = self._rule_router.choose_action(routed)
            except (TypeError, ValueError):
                continue
            if self._decision_executable(state, decision):
                decisions.append(
                    decision.model_copy(
                        update={
                            "routed_task_type": task_type,
                            "routed_information_gaps": (
                                []
                                if routed_state is not state
                                else list(state.readiness.information_gaps)
                            ),
                        }
                    )
                )

        unique: dict[str, AgentDecision] = {}
        for decision in decisions:
            unique.setdefault(_decision_key(decision), decision)
        if not unique:
            fallback = AgentDecision(
                action="safe_fallback",
                reason_code="NO_SAFE_ROUTER_CHOICE",
                routed_task_type=state.readiness.task_type,
            )
            unique[_decision_key(fallback)] = fallback
        rows = [
            (self._choice(decision), decision)
            for decision in unique.values()
        ]
        return tuple(rows[: self._maximum_choices])

    @staticmethod
    def _can_reconsider_ambiguous_reference(state: AgentState) -> bool:
        """Allow the model to reject one known false-positive gap safely.

        Relative feedback such as "make it cheaper" refers to the previous
        recommendation set, not to one mandatory business.  Continuing is
        offered only when that previous set is present; all other gaps remain
        hard code-enforced clarification gates.
        """

        if set(state.readiness.information_gaps) != {"ambiguous_reference"}:
            return False
        prior = [] if not state.turns else state.turns[-1].recommended_business_ids
        memory_prior = (
            [] if state.memory is None else state.memory.last_presented_business_ids
        )
        return state.readiness.task_type == "feedback_refinement" and bool(
            prior or memory_prior
        )

    @staticmethod
    def _should_offer_conflict_clarification(state: AgentState) -> bool:
        """Expose, but do not automatically choose, a missed-conflict option."""

        if state.step_count != 0:
            return False
        text = (
            state.memory.recent_turns[-1].query_text
            if state.memory is not None and state.memory.recent_turns
            else state.request.query_text
        ).casefold()
        explicit_words = ("conflict", "contradict", "冲突", "矛盾")
        if any(word in text for word in explicit_words):
            return True
        return ("must" in text and "exclude" in text) or (
            "必须" in text and ("排除" in text or "不要" in text)
        )

    @staticmethod
    def _task_plausible(state: AgentState, task_type: TaskType) -> bool:
        references = state.request.referenced_business_ids
        prior_recommendations = (
            [] if not state.turns else state.turns[-1].recommended_business_ids
        )
        if task_type == "recommendation_request":
            return True
        if task_type == "feedback_refinement":
            return bool(references or prior_recommendations or state.memory)
        if task_type == "candidate_comparison":
            return len(references) >= 2
        return bool(references)

    @staticmethod
    def _decision_executable(
        state: AgentState,
        decision: AgentDecision,
    ) -> bool:
        ids = decision.arguments.get("business_ids")
        if decision.action in {
            "get_business_details",
            "retrieve_business_reviews",
            "return_recommendation",
            "return_grounded_answer",
        } and (not isinstance(ids, list) or not ids):
            return False
        if decision.action == "compare_candidates" and (
            not isinstance(ids, list) or len(ids) < 2
        ):
            return False
        if decision.action == "apply_hard_constraints" and not state.business_scope:
            return False
        return decision_rejection(state, decision, [decision.action]) is None

    @staticmethod
    def _choice(decision: AgentDecision) -> RouterChoice:
        digest = hashlib.sha256(_decision_key(decision).encode("utf-8")).hexdigest()[:12]
        tool = f" using {decision.tool_name}" if decision.tool_name else ""
        return RouterChoice(
            choice_id=f"choice_{digest}",
            action=decision.action,
            task_type=(decision.routed_task_type or "unknown"),
            reason_code=decision.reason_code,
            tool_name=decision.tool_name,
            tool_kind=decision.tool_kind,
            public_summary=f"{_SUMMARIES[decision.action]}{tool}",
        )


def _decision_key(decision: AgentDecision) -> str:
    return json.dumps(
        {
            "action": decision.action,
            "arguments": decision.arguments,
            "reason_code": decision.reason_code,
            "tool_name": decision.tool_name,
            "tool_kind": decision.tool_kind,
            "routed_task_type": decision.routed_task_type,
            "routed_information_gaps": decision.routed_information_gaps,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
