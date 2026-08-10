"""Deterministic construction of Step 21 traces and runtime observations."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from yelp_agent.agent_evaluation.schema import (
    AgentActionTrace,
    AgentTurnTrace,
    ClarificationQuestionTrace,
    ResponseClaimTrace,
    ToolCallTrace,
)

from .schema import ActionOutcome, AgentDecision, AgentObservation, AgentSession


def arguments_sha256(arguments: dict[str, Any]) -> str:
    payload = json.dumps(
        arguments,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def call_signature(state: AgentSession, decision: AgentDecision) -> str | None:
    """Identify a tool request within one interpreted request."""

    if decision.tool_name is None:
        return None
    payload = (
        f"{state.request.request_id}:{decision.tool_name}:"
        f"{arguments_sha256(decision.arguments)}"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_observation(
    *,
    state: AgentSession,
    action_step_index: int,
    decision: AgentDecision,
    payload: dict[str, Any],
) -> AgentObservation:
    serialized = json.dumps(
        {
            "scenario_id": state.scenario_id,
            "turn_index": state.current_turn,
            "step_index": action_step_index,
            "action": decision.action,
            "arguments": decision.arguments,
            "payload": payload,
        },
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return AgentObservation(
        observation_id=hashlib.sha256(serialized).hexdigest(),
        turn_index=state.current_turn,
        step_index=action_step_index,
        action=decision.action,
        payload=payload,
    )


def build_tool_trace(
    *,
    state: AgentSession,
    decision: AgentDecision,
    outcome: ActionOutcome,
    action_step_index: int,
    latency_ms: float,
) -> ToolCallTrace:
    assert decision.tool_name is not None
    assert decision.tool_kind is not None
    metadata = outcome.tool_result
    arguments_hash = arguments_sha256(decision.arguments)
    call_id_payload = (
        f"{state.scenario_id}:{state.current_turn}:{action_step_index}:"
        f"{decision.tool_name}:{arguments_hash}"
    ).encode("utf-8")
    return ToolCallTrace(
        call_id=hashlib.sha256(call_id_payload).hexdigest(),
        action_step_index=action_step_index,
        action=decision.action,
        tool_name=decision.tool_name,
        tool_kind=decision.tool_kind,
        arguments_sha256=arguments_hash,
        status="completed" if outcome.status == "completed" else "failed",
        latency_ms=latency_ms,
        input_tokens=None if metadata is None else metadata.input_tokens,
        output_tokens=None if metadata is None else metadata.output_tokens,
        cost_usd=None if metadata is None else metadata.cost_usd,
        cache_hit=False if metadata is None else metadata.cache_hit,
        retrieved_evidence=[] if metadata is None else metadata.retrieved_evidence,
    )


@dataclass(slots=True)
class TurnTraceRecorder:
    """Accumulate one turn without exposing trace bookkeeping to the engine."""

    actions: list[AgentActionTrace] = field(default_factory=list)
    tool_calls: list[ToolCallTrace] = field(default_factory=list)
    clarification_questions: list[ClarificationQuestionTrace] = field(
        default_factory=list
    )
    candidate_ranking: list[str] = field(default_factory=list)
    recommended_business_ids: list[str] = field(default_factory=list)
    claims: list[ResponseClaimTrace] = field(default_factory=list)
    response_kind: str = "none"
    reported_conflict: bool = False
    reported_evidence_recency: bool = False
    recommended_official_verification: bool = False

    @property
    def next_step_index(self) -> int:
        return len(self.actions) + 1

    def add_action(
        self,
        decision: AgentDecision,
        *,
        status: str,
        reason_code: str | None = None,
    ) -> int:
        step_index = self.next_step_index
        self.actions.append(
            AgentActionTrace(
                step_index=step_index,
                action=decision.action,
                status=status,
                reason_code=reason_code or decision.reason_code,
            )
        )
        return step_index

    def add_tool_call(
        self,
        *,
        state: AgentSession,
        decision: AgentDecision,
        outcome: ActionOutcome,
        action_step_index: int,
        latency_ms: float,
    ) -> None:
        self.tool_calls.append(
            build_tool_trace(
                state=state,
                decision=decision,
                outcome=outcome,
                action_step_index=action_step_index,
                latency_ms=latency_ms,
            )
        )

    def absorb(self, outcome: ActionOutcome) -> None:
        self.candidate_ranking = outcome.candidate_ranking
        self.recommended_business_ids = outcome.recommended_business_ids
        self.claims = outcome.claims
        self.response_kind = outcome.response_kind
        self.reported_conflict = outcome.reported_conflict
        self.reported_evidence_recency = outcome.reported_evidence_recency
        self.recommended_official_verification = (
            outcome.recommended_official_verification
        )
        if outcome.clarification_question is not None:
            self.clarification_questions = [outcome.clarification_question]

    def build_turn(self, state: AgentSession) -> AgentTurnTrace:
        return AgentTurnTrace(
            turn_index=state.current_turn,
            predicted_task_type=state.readiness.task_type,
            detected_information_gaps=state.readiness.information_gaps,
            actions=self.actions,
            tool_calls=self.tool_calls,
            clarification_questions=self.clarification_questions,
            candidate_ranking=self.candidate_ranking,
            recommended_business_ids=self.recommended_business_ids,
            claims=self.claims,
            response_kind=self.response_kind,
            reported_conflict=self.reported_conflict,
            reported_evidence_recency=self.reported_evidence_recency,
            recommended_official_verification=(
                self.recommended_official_verification
            ),
        )
