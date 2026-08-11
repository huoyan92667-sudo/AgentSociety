"""Deterministic, evidence-bounded terminal responses for Agent V1."""

from __future__ import annotations

from yelp_agent.agent_evaluation.schema import (
    ClarificationQuestionTrace,
    EvidenceReference,
    ResponseClaimTrace,
)
from yelp_agent.agent_harness.schema import ActionOutcome, AgentDecision, AgentState
from yelp_agent.agent_tools.schema import ToolObservation
from pydantic import TypeAdapter, ValidationError

from yelp_agent.decision_readiness.schema import InformationGap

from .state_view import RouteFacts


_QUESTIONS: dict[str, dict[InformationGap, str]] = {
    "zh-CN": {
        "missing_location": "请告诉我你现在的大概位置，也可以提供附近地标。",
        "missing_budget": "你希望每人大约控制在多少钱以内？",
        "missing_party_size": "这次大约有几个人一起用餐？",
        "constraint_conflict": "你的要求之间存在冲突，想优先保留哪一个条件？",
        "ambiguous_reference": "你指的是哪一家商家？请提供名称或商家编号。",
    },
    "en-US": {
        "missing_location": "What location or nearby landmark should I search around?",
        "missing_budget": "What is your approximate budget per person?",
        "missing_party_size": "How many people are in your party?",
        "constraint_conflict": (
            "Some requirements conflict. Which one should take priority?"
        ),
        "ambiguous_reference": (
            "Which business do you mean? Please provide its name or ID."
        ),
    },
}
_GAP_LIST_ADAPTER = TypeAdapter(list[InformationGap])


class TerminalActionExecutor:
    """Execute terminal actions without inventing businesses or evidence."""

    def __init__(
        self,
        *,
        fusion_alpha: float = 0.0,
        cross_encoder_beta: float = 0.0,
    ) -> None:
        if not 0 <= fusion_alpha <= 1:
            raise ValueError("fusion alpha must be between zero and one")
        self._fusion_alpha = fusion_alpha
        if not 0 <= cross_encoder_beta <= 1:
            raise ValueError("Cross-Encoder beta must be between zero and one")
        self._cross_encoder_beta = cross_encoder_beta

    def execute(
        self,
        state: AgentState,
        decision: AgentDecision,
    ) -> ActionOutcome:
        if decision.action == "ask_clarification":
            return self._clarification(state, decision)
        if decision.action == "apply_feedback":
            return self._apply_feedback(state, decision)
        if decision.action == "return_recommendation":
            return self._recommendation(state, decision)
        if decision.action == "return_grounded_answer":
            return self._grounded_answer(state, decision)
        if decision.action == "return_uncertain_answer":
            return self._uncertain_answer(state, decision)
        return ActionOutcome(
            status="failed",
            failure_reason=f"unsupported_terminal_action:{decision.action}",
        )

    @staticmethod
    def _apply_feedback(
        state: AgentState,
        decision: AgentDecision,
    ) -> ActionOutcome:
        raw_ids = decision.arguments.get("rejected_business_ids", [])
        if not isinstance(raw_ids, list) or any(
            not isinstance(value, str) for value in raw_ids
        ):
            return ActionOutcome(
                status="failed",
                failure_reason="feedback_rejected_business_ids_invalid",
            )
        rejected = set(raw_ids)
        if state.business_scope_known and not rejected.issubset(state.business_scope):
            return ActionOutcome(
                status="failed",
                failure_reason="feedback_business_out_of_scope",
            )
        scope = (
            [value for value in state.business_scope if value not in rejected]
            if state.business_scope_known
            else None
        )
        return ActionOutcome(
            status="completed",
            observation={
                "feedback_applied": True,
                "rejected_business_ids": sorted(rejected),
            },
            business_scope=scope,
        )

    @staticmethod
    def _clarification(
        state: AgentState,
        decision: AgentDecision,
    ) -> ActionOutcome:
        raw_gaps = decision.arguments.get("information_gaps")
        if not isinstance(raw_gaps, list) or not raw_gaps:
            return ActionOutcome(
                status="failed",
                failure_reason="clarification_gap_missing",
            )
        try:
            gaps = _GAP_LIST_ADAPTER.validate_python(raw_gaps)
        except ValidationError:
            return ActionOutcome(
                status="failed",
                failure_reason="clarification_gap_invalid",
            )
        language = state.language if state.language in _QUESTIONS else "en-US"
        questions = _QUESTIONS[language]
        text = " ".join(questions[gap] for gap in gaps)
        return ActionOutcome(
            status="completed",
            response_kind="clarification",
            clarification_question=ClarificationQuestionTrace(
                question_text=text,
                requested_information_gaps=gaps,
            ),
            reported_conflict="constraint_conflict" in gaps,
        )

    def _recommendation(
        self,
        state: AgentState,
        decision: AgentDecision,
    ) -> ActionOutcome:
        facts = RouteFacts.from_state(state)
        raw_ids = decision.arguments.get("business_ids")
        if (
            not isinstance(raw_ids, list)
            or not raw_ids
            or any(not isinstance(value, str) for value in raw_ids)
        ):
            return ActionOutcome(
                status="failed",
                failure_reason="recommendation_business_ids_invalid",
            )
        business_ids = list(raw_ids)
        ranking = facts.final_ranking(
            fusion_alpha=self._fusion_alpha,
            cross_encoder_beta=self._cross_encoder_beta,
        )
        if not ranking or business_ids != ranking[: len(business_ids)]:
            return ActionOutcome(
                status="failed",
                failure_reason="recommendation_not_ranking_prefix",
            )
        return ActionOutcome(
            status="completed",
            response_kind="recommendation",
            candidate_ranking=ranking,
            recommended_business_ids=business_ids,
        )

    def _grounded_answer(
        self,
        state: AgentState,
        decision: AgentDecision,
    ) -> ActionOutcome:
        if state.readiness.task_type == "business_detail_question":
            claims = self._detail_claims(state, decision)
        else:
            claims = self._profile_or_comparison_claims(state, decision)
        if not claims:
            return ActionOutcome(
                status="failed",
                failure_reason="grounded_answer_has_no_supported_claim",
            )
        facts = RouteFacts.from_state(state)
        return ActionOutcome(
            status="completed",
            response_kind="grounded_answer",
            claims=claims,
            reported_conflict=facts.structured_evidence_conflict,
            reported_evidence_recency=any(
                "latest_evidence_time" in claim.text for claim in claims
            ),
        )

    @staticmethod
    def _uncertain_answer(
        state: AgentState,
        decision: AgentDecision,
    ) -> ActionOutcome:
        del state
        return ActionOutcome(
            status="completed",
            response_kind="uncertain_answer",
            reported_conflict=(decision.reason_code == "CONSTRAINT_CONFLICT"),
            recommended_official_verification=(
                decision.arguments.get("recommended_official_verification") is True
            ),
        )

    @staticmethod
    def _detail_claims(
        state: AgentState,
        decision: AgentDecision,
    ) -> list[ResponseClaimTrace]:
        data = _latest_tool_data(state, "GET_BUSINESS_DETAILS")
        rows = data.get("businesses")
        if not isinstance(rows, list):
            return []
        requested = decision.arguments.get("business_ids")
        requested_ids = (
            [value for value in requested if isinstance(value, str)]
            if isinstance(requested, list)
            else state.request.referenced_business_ids
        )
        by_id = {
            str(row.get("business_id")): row
            for row in rows
            if isinstance(row, dict) and isinstance(row.get("business_id"), str)
        }
        claims: list[ResponseClaimTrace] = []
        for business_id in requested_ids:
            row = by_id.get(business_id)
            if row is None:
                continue
            categories = row.get("categories")
            if not isinstance(categories, list) or not categories:
                continue
            name = str(row.get("name") or business_id)
            text = f"{name}: categories={', '.join(map(str, categories))}"
            claims.append(
                ResponseClaimTrace(
                    claim_id=f"detail:{business_id}:categories",
                    text=text,
                    business_id=business_id,
                    evidence_refs=[
                        EvidenceReference(
                            business_id=business_id,
                            source_type="business_attribute",
                            source_field="categories",
                        )
                    ],
                )
            )
        return claims

    @staticmethod
    def _profile_or_comparison_claims(
        state: AgentState,
        decision: AgentDecision,
    ) -> list[ResponseClaimTrace]:
        if state.readiness.task_type == "candidate_comparison":
            return _comparison_claims(state, decision)
        data = _latest_tool_data(state, "GET_BUSINESS_PROFILE")
        rows = data.get("profiles")
        if not isinstance(rows, list):
            return []
        by_id = {
            str(row.get("business_id")): row
            for row in rows
            if isinstance(row, dict) and isinstance(row.get("business_id"), str)
        }
        facts = RouteFacts.from_state(state)
        raw_ids = decision.arguments.get("business_ids")
        business_ids = (
            [value for value in raw_ids if isinstance(value, str)]
            if isinstance(raw_ids, list)
            else facts.referenced_business_ids
        )
        claims: list[ResponseClaimTrace] = []
        for business_id in business_ids:
            profile = by_id.get(business_id)
            summaries = None if profile is None else profile.get("aspect_summaries")
            if not isinstance(summaries, dict):
                continue
            for aspect in facts.requested_aspects:
                summary = summaries.get(aspect)
                if not isinstance(summary, dict) or summary.get("status") != "known":
                    continue
                positive = summary.get("weighted_positive_ratio")
                direction = (
                    "positive"
                    if isinstance(positive, (int, float)) and positive >= 0.5
                    else "negative"
                )
                confidence = summary.get("confidence")
                latest = summary.get("latest_evidence_time")
                name = str(profile.get("name") or business_id)
                text = (
                    f"{name}: aggregated {aspect} evidence is {direction}; "
                    f"confidence={confidence}; latest_evidence_time={latest}"
                )
                claims.append(
                    ResponseClaimTrace(
                        claim_id=f"profile:{business_id}:{aspect}",
                        text=text,
                        business_id=business_id,
                        evidence_refs=[
                            EvidenceReference(
                                business_id=business_id,
                                source_type="business_attribute",
                                source_field=f"aspect_summaries.{aspect}",
                            )
                        ],
                    )
                )
        return claims


def _latest_tool_data(state: AgentState, tool_name: str) -> dict[str, object]:
    for observation in reversed(state.observations):
        try:
            payload = ToolObservation.model_validate(observation.payload)
        except ValueError:
            continue
        if payload.tool_name == tool_name and payload.status in {"success", "partial"}:
            return dict(payload.data)
    return {}


def _comparison_claims(
    state: AgentState,
    decision: AgentDecision,
) -> list[ResponseClaimTrace]:
    comparison = _latest_tool_data(state, "COMPARE_BUSINESSES")
    ranking = comparison.get("ranking")
    if not isinstance(ranking, list) or not ranking:
        return []
    raw_ids = decision.arguments.get("business_ids")
    business_ids = (
        [value for value in raw_ids if isinstance(value, str)]
        if isinstance(raw_ids, list)
        else [value for value in ranking if isinstance(value, str)]
    )
    top_id = business_ids[0] if business_ids else None
    if top_id is None or top_id not in ranking:
        return []
    return [
        ResponseClaimTrace(
            claim_id=f"comparison:{top_id}:rank",
            text=f"{top_id} ranks first in the structured candidate comparison.",
            business_id=top_id,
            evidence_refs=[
                EvidenceReference(
                    business_id=top_id,
                    source_type="business_attribute",
                    source_field="comparison.query_score",
                )
            ],
        )
    ]
