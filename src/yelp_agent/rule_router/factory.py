"""One assembly seam for the complete deterministic Rule Agent."""

from __future__ import annotations

from yelp_agent.agent_harness import (
    AgentHarness,
    HarnessBudget,
    RuleBasedRequestInterpreter,
)
from yelp_agent.agent_harness.interfaces import Clock, FallbackHandler
from yelp_agent.agent_tools import AgentToolRegistry, RegistryActionExecutor

from .policy import RuleBasedActionPolicy
from .router import RuleRouter
from .terminal_executor import TerminalActionExecutor


def build_rule_agent(
    *,
    registry: AgentToolRegistry,
    fallback_handler: FallbackHandler | None = None,
    budget: HarnessBudget | None = None,
    display_limit: int = 3,
    agent_version: str = "step24-rule-agent-v1",
    clock: Clock | None = None,
) -> AgentHarness:
    """Connect the Step 18/22/23/24 modules behind one runner interface."""

    return AgentHarness(
        agent_version=agent_version,
        interpreter=RuleBasedRequestInterpreter(),
        action_policy=RuleBasedActionPolicy(display_limit=display_limit),
        router=RuleRouter(display_limit=display_limit),
        executor=RegistryActionExecutor(
            registry,
            fallback=TerminalActionExecutor(),
        ),
        fallback_handler=fallback_handler,
        budget=budget,
        clock=clock,
    )
