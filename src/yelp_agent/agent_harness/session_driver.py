"""Evaluation-only release of hidden scripted user turns."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from pydantic import Field

from yelp_agent.agent_benchmark import ScriptedUserTurn, VisibleAgentScenario
from yelp_agent.models import StrictModel

from .engine import AgentHarness
from .schema import HarnessResult, UserTurnInput


class BenchmarkSessionResult(StrictModel):
    """Outcome of driving one visible scenario with hidden scripted replies."""

    result: HarnessResult
    released_turn_indices: list[int] = Field(default_factory=list)
    stop_reason: Literal[
        "completed",
        "scripts_exhausted",
        "trigger_not_observed",
        "agent_not_awaiting_user",
    ]


class BenchmarkSessionDriver:
    """Keep hidden replies outside the Harness and release them on trigger only."""

    def drive(
        self,
        harness: AgentHarness,
        scenario: VisibleAgentScenario,
        scripted_turns: Sequence[ScriptedUserTurn],
    ) -> BenchmarkSessionResult:
        result = harness.start(scenario)
        released: list[int] = []
        ordered = sorted(scripted_turns, key=lambda item: item.turn_index)
        for scripted_turn in ordered:
            if result.session.status == "completed" or result.session.status == "fallback":
                return BenchmarkSessionResult(
                    result=result,
                    released_turn_indices=released,
                    stop_reason="agent_not_awaiting_user",
                )
            if result.session.status != "awaiting_user":
                return BenchmarkSessionResult(
                    result=result,
                    released_turn_indices=released,
                    stop_reason="agent_not_awaiting_user",
                )
            previous_turn = result.session.turns[-1]
            triggered = any(
                action.action == scripted_turn.trigger_action
                and action.status == "completed"
                for action in previous_turn.actions
            )
            if not triggered:
                return BenchmarkSessionResult(
                    result=result,
                    released_turn_indices=released,
                    stop_reason="trigger_not_observed",
                )
            latitude = scripted_turn.state_updates.get("user_latitude")
            longitude = scripted_turn.state_updates.get("user_longitude")
            result = harness.resume(
                result.session,
                UserTurnInput(
                    query_text=scripted_turn.query_text,
                    user_latitude=(
                        float(latitude) if isinstance(latitude, (int, float)) else None
                    ),
                    user_longitude=(
                        float(longitude)
                        if isinstance(longitude, (int, float))
                        else None
                    ),
                ),
            )
            released.append(scripted_turn.turn_index)
        stop_reason = (
            "completed"
            if result.session.status in {"completed", "fallback"}
            else "scripts_exhausted"
        )
        return BenchmarkSessionResult(
            result=result,
            released_turn_indices=released,
            stop_reason=stop_reason,
        )
