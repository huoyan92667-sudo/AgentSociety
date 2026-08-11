"""Deterministic, evidence-bounded terminal responses for Agent V1."""

from __future__ import annotations

from pydantic import TypeAdapter, ValidationError

from yelp_agent.agent_evaluation.schema import (
    ClarificationQuestionTrace,
    EvidenceReference,
    ResponseClaimTrace,
)
from yelp_agent.agent_harness.schema import (
    ActionOutcome,
    AgentDecision,
    AgentState,
    ModelResultMetadata,
)
from yelp_agent.agent_tools.schema import ToolObservation
from yelp_agent.controlled_llm import (
    AnswerCompositionInput,
    AnswerEvidenceItem,
    GroundedAnswerComposer,
)
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
        answer_composer: GroundedAnswerComposer | None = None,
        answer_evidence_limit: int = 12,
    ) -> None:
        if not 0 <= fusion_alpha <= 1:
            raise ValueError("fusion alpha must be between zero and one")
        self._fusion_alpha = fusion_alpha
        if not 0 <= cross_encoder_beta <= 1:
            raise ValueError("Cross-Encoder beta must be between zero and one")
        self._cross_encoder_beta = cross_encoder_beta
        self._answer_composer = answer_composer
        self._answer_evidence_limit = answer_evidence_limit

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
            return self._compose_answer(
                state,
                self._grounded_answer(state, decision),
            )
        if decision.action == "return_uncertain_answer":
            return self._compose_answer(
                state,
                self._uncertain_answer(state, decision),
            )
        return ActionOutcome(
            status="failed",
            failure_reason=f"unsupported_terminal_action:{decision.action}",
        )

    def _compose_answer(
        self,
        state: AgentState,
        baseline: ActionOutcome,
    ) -> ActionOutcome:
        if (
            self._answer_composer is None
            or baseline.status != "completed"
            or baseline.response_kind not in {"grounded_answer", "uncertain_answer"}
            or not baseline.claims
        ):
            return baseline
        allowed = list(
            dict.fromkeys(
                [
                    *state.business_scope,
                    *state.request.referenced_business_ids,
                    *[
                        claim.business_id
                        for claim in baseline.claims
                        if claim.business_id is not None
                    ],
                ]
            )
        )
        evidence = [
            AnswerEvidenceItem(evidence_code=f"E{index}", claim=claim)
            for index, claim in enumerate(
                baseline.claims[: self._answer_evidence_limit],
                start=1,
            )
        ]
        composed = self._answer_composer.compose(
            AnswerCompositionInput(
                context_id=state.scenario_id,
                turn_index=state.current_turn,
                query_text=state.request.query_text,
                language=state.language,
                task_type=state.readiness.task_type,
                response_kind=baseline.response_kind,  # type: ignore[arg-type]
                allowed_business_ids=allowed,
                evidence=evidence,
                reported_conflict=baseline.reported_conflict,
                reported_evidence_recency=baseline.reported_evidence_recency,
                recommended_official_verification=(
                    baseline.recommended_official_verification
                ),
            )
        )
        trace = composed.trace
        return baseline.model_copy(
            update={
                "claims": composed.claims,
                "model_result": ModelResultMetadata(
                    capability="answer_composition",
                    status=composed.status,
                    provider_called=trace.provider_called,
                    input_tokens=(
                        trace.input_tokens if trace.provider_called else None
                    ),
                    output_tokens=(
                        trace.output_tokens if trace.provider_called else None
                    ),
                    cost_usd=None,
                    cache_hit=trace.cache_hit,
                ),
            }
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
        if state.readiness.task_type == "review_experience_question":
            claims = self._aggregation_claims(state, decision)
            if not claims:
                claims = self._review_claims(state, decision)
            if not claims:
                claims = self._profile_or_comparison_claims(state, decision)
        elif state.readiness.task_type == "candidate_comparison" and _latest_tool_data(
            state, "SEARCH_BUSINESS_REVIEWS"
        ).get("hits"):
            claims = self._aggregation_claims(state, decision)
            if not claims:
                claims = self._review_claims(state, decision)
        elif state.readiness.task_type == "business_detail_question":
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
            reported_conflict=(
                facts.structured_evidence_conflict
                or facts.evidence_aggregation_conflict
                or facts.review_evidence_conflict
            ),
            reported_evidence_recency=(
                bool(
                    _latest_tool_data(state, "AGGREGATE_REVIEW_EVIDENCE").get(
                        "businesses"
                    )
                )
                or bool(_latest_tool_data(state, "SEARCH_BUSINESS_REVIEWS").get("hits"))
                or any("latest_evidence_time" in claim.text for claim in claims)
            ),
        )

    @staticmethod
    def _uncertain_answer(
        state: AgentState,
        decision: AgentDecision,
    ) -> ActionOutcome:
        facts = RouteFacts.from_state(state)
        claims = _aggregation_claims_from_data(state, decision)
        aggregation = _latest_tool_data(state, "AGGREGATE_REVIEW_EVIDENCE")
        return ActionOutcome(
            status="completed",
            response_kind="uncertain_answer",
            claims=claims,
            reported_conflict=(
                decision.reason_code
                in {"CONSTRAINT_CONFLICT", "CONFLICTING_REVIEW_EVIDENCE"}
                or facts.evidence_aggregation_conflict
            ),
            reported_evidence_recency=any(
                isinstance(item, dict) and item.get("latest_evidence_time") is not None
                for item in aggregation.get("businesses", [])
            ) if isinstance(aggregation.get("businesses"), list) else False,
            recommended_official_verification=(
                decision.arguments.get("recommended_official_verification") is True
                or aggregation.get("recommend_official_verification") is True
            ),
        )

    @staticmethod
    def _aggregation_claims(
        state: AgentState,
        decision: AgentDecision,
    ) -> list[ResponseClaimTrace]:
        return _aggregation_claims_from_data(state, decision)

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
    def _review_claims(
        state: AgentState,
        decision: AgentDecision,
    ) -> list[ResponseClaimTrace]:
        data = _latest_tool_data(state, "SEARCH_BUSINESS_REVIEWS")
        rows = data.get("hits")
        if not isinstance(rows, list):
            return []
        requested = decision.arguments.get("business_ids")
        business_ids = (
            [value for value in requested if isinstance(value, str)]
            if isinstance(requested, list)
            else state.request.referenced_business_ids
        )
        claims: list[ResponseClaimTrace] = []
        for business_id in business_ids:
            candidates = [
                row
                for row in rows
                if isinstance(row, dict)
                and row.get("business_id") == business_id
                and isinstance(row.get("review_id"), str)
            ]
            selected = [
                row for row in candidates if row.get("matched_aspects")
            ][:2] or candidates[:1]
            if not selected:
                continue
            snippets = [
                " ".join(str(row.get("text") or "").split())[:240]
                for row in selected
            ]
            dates = [str(row.get("review_time") or "")[:10] for row in selected]
            claims.append(
                ResponseClaimTrace(
                    claim_id=f"review:{business_id}:{selected[0]['review_id']}",
                    text=(
                        f"{business_id}: cutoff-safe review evidence ({', '.join(dates)}): "
                        + " | ".join(snippets)
                    ),
                    business_id=business_id,
                    evidence_refs=[
                        EvidenceReference(
                            business_id=business_id,
                            source_type="review",
                            review_id=str(row["review_id"]),
                        )
                        for row in selected
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


def _aggregation_claims_from_data(
    state: AgentState,
    decision: AgentDecision,
) -> list[ResponseClaimTrace]:
    data = _latest_tool_data(state, "AGGREGATE_REVIEW_EVIDENCE")
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
        business = by_id.get(business_id)
        aspects = None if business is None else business.get("aspects")
        if not isinstance(aspects, list):
            continue
        for row in aspects:
            if not isinstance(row, dict):
                continue
            review_ids = row.get("citation_review_ids")
            citations = (
                [value for value in review_ids if isinstance(value, str)]
                if isinstance(review_ids, list)
                else []
            )
            if not citations:
                continue
            condition_rows = row.get("condition_groups")
            condition_tags = [
                str(item["condition_tag"])
                for item in condition_rows or []
                if isinstance(item, dict) and isinstance(item.get("condition_tag"), str)
            ]
            latest = str(row.get("latest_evidence_time") or "unknown")[:10]
            aspect = str(row.get("aspect") or "unknown")
            text = (
                f"{business_id}: aggregated_review_evidence aspect={aspect}; "
                f"consensus={row.get('consensus')}; "
                f"confidence={row.get('confidence_level')} "
                f"({float(row.get('confidence_score') or 0.0):.4f}); "
                f"support={int(row.get('support_count') or 0)}, "
                f"contradict={int(row.get('contradiction_count') or 0)}, "
                f"neutral={int(row.get('neutral_count') or 0)}, "
                f"unique_users={int(row.get('unique_user_count') or 0)}; "
                f"latest_evidence_time={latest}"
            )
            if condition_tags:
                text += f"; explicit_conditions={','.join(condition_tags)}"
            if row.get("requires_caveat") is True:
                text += "; conclusion_requires_caveat=true"
            claims.append(
                ResponseClaimTrace(
                    claim_id=f"aggregate:{business_id}:{aspect}",
                    text=text,
                    business_id=business_id,
                    evidence_refs=[
                        EvidenceReference(
                            business_id=business_id,
                            source_type="review",
                            review_id=review_id,
                        )
                        for review_id in citations
                    ],
                )
            )
    return claims


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
