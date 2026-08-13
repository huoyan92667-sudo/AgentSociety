"""Adapter from Step 22 decisions to the Step 23 tool registry."""

from __future__ import annotations

from typing import Any

from yelp_agent.agent_evaluation.schema import RetrievedEvidenceTrace
from yelp_agent.agent_harness.interfaces import ActionExecutor
from yelp_agent.agent_harness.schema import (
    ActionOutcome,
    AgentDecision,
    AgentSession,
    ToolResultMetadata,
)

from .registry import AgentToolRegistry
from .schema import ToolExecutionContext, ToolObservation
from yelp_agent.session_memory.context import compact_memory
from yelp_agent.session_memory.effective_request import compile_effective_request


class RegistryActionExecutor:
    """Execute tool decisions and delegate non-tool actions through one seam."""

    def __init__(
        self,
        registry: AgentToolRegistry,
        *,
        fallback: ActionExecutor | None = None,
    ) -> None:
        self._registry = registry
        self._fallback = fallback

    def execute(
        self,
        state: AgentSession,
        decision: AgentDecision,
    ) -> ActionOutcome:
        if decision.tool_name is None:
            if self._fallback is None:
                return ActionOutcome(
                    status="failed",
                    failure_reason="non_tool_action_has_no_executor",
                )
            return self._fallback.execute(state, decision)
        context = self._tool_context(state, decision)
        try:
            descriptor = self._registry.describe(decision.tool_name)
        except KeyError:
            observation = self._registry.execute(
                decision.tool_name,
                decision.arguments,
                context,
            )
            return self._to_outcome(decision, observation)
        if decision.tool_kind != descriptor.kind:
            return ActionOutcome(
                status="failed",
                failure_reason="tool_kind_mismatch",
            )
        observation = self._registry.execute(
            decision.tool_name,
            decision.arguments,
            context,
        )
        return self._to_outcome(decision, observation)

    @classmethod
    def _tool_context(
        cls,
        state: AgentSession,
        decision: AgentDecision,
    ) -> ToolExecutionContext:
        snapshot = cls._visible_snapshot(state)
        effective = snapshot.get("effective_request")
        effective_id = (
            effective.get("effective_request_id")
            if isinstance(effective, dict)
            else None
        )
        return ToolExecutionContext(
            request_id=(
                str(effective_id)
                if isinstance(effective_id, str)
                else state.request.request_id
            ),
            user_id=state.user_id,
            cutoff_time=state.cutoff_time,
            action=decision.action,
            business_scope=tuple(state.business_scope),
            business_scope_known=state.business_scope_known,
            state_snapshot=snapshot,
        )

    @staticmethod
    def _visible_snapshot(state: AgentSession) -> dict[str, Any]:
        snapshot = {
            "scenario_id": state.scenario_id,
            "session_id": state.session_id,
            "split": state.split,
            "turn_index": state.current_turn,
            "request": state.request.model_dump(
                mode="json", exclude_computed_fields=True
            ),
            "readiness": state.readiness.model_dump(mode="json"),
            "observations": [item.model_dump(mode="json") for item in state.observations],
        }
        if state.memory is not None:
            snapshot["memory_context"] = compact_memory(state.memory).model_dump(
                mode="json"
            )
            snapshot["effective_request"] = compile_effective_request(
                state.memory
            ).model_dump(mode="json", exclude_computed_fields=True)
        return snapshot

    @staticmethod
    def _to_outcome(
        decision: AgentDecision,
        observation: ToolObservation,
    ) -> ActionOutcome:
        successful = observation.status in {"success", "no_result", "partial"}
        metadata = ToolResultMetadata(
            input_tokens=observation.input_tokens,
            output_tokens=observation.output_tokens,
            cost_usd=observation.cost_usd,
            cache_hit=observation.cache_hit,
            retrieved_evidence=[
                RetrievedEvidenceTrace(rank=rank, evidence=evidence)
                for rank, evidence in enumerate(observation.evidence, start=1)
            ],
        )
        if not successful:
            return ActionOutcome(
                status="failed",
                observation=observation.model_dump(mode="json"),
                failure_reason=(
                    f"tool:{observation.tool_name}:"
                    f"{observation.error_code or 'UNKNOWN_ERROR'}"
                ),
                tool_result=metadata,
            )
        payload = observation.model_dump(mode="json")
        candidate_ids = observation.data.get("candidate_business_ids")
        ranking = observation.data.get("ranking")
        business_scope = None
        if decision.action in {"retrieve_candidates", "apply_hard_constraints"}:
            if isinstance(candidate_ids, list):
                business_scope = [str(value) for value in candidate_ids]
        return ActionOutcome(
            status="completed",
            observation=payload,
            candidate_ranking=(
                [str(value) for value in ranking] if isinstance(ranking, list) else []
            ),
            business_scope=business_scope,
            tool_result=metadata,
        )
