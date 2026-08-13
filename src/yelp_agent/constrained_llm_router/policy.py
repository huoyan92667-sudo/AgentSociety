"""Code-enforced action policy derived from complete Router choices."""

from __future__ import annotations

from yelp_agent.agent_benchmark.schema import AgentAction
from yelp_agent.agent_harness.schema import AgentState

from .choices import ConstrainedDecisionBuilder


class ConstrainedActionPolicy:
    def __init__(self, builder: ConstrainedDecisionBuilder) -> None:
        self._builder = builder

    def allowed_actions(self, state: AgentState) -> tuple[AgentAction, ...]:
        return tuple(
            dict.fromkeys(
                decision.action for _, decision in self._builder.build(state)
            )
        )
