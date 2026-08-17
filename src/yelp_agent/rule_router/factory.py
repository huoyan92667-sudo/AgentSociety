"""One assembly seam for the complete deterministic Rule Agent."""

from __future__ import annotations

from yelp_agent.agent_harness import (
    AgentHarness,
    HarnessBudget,
    RuleBasedRequestInterpreter,
)
from yelp_agent.agent_harness.interfaces import (
    ActionRouter,
    AllowedActionPolicy,
    Clock,
    FallbackHandler,
)
from yelp_agent.agent_tools import AgentToolRegistry, RegistryActionExecutor
from yelp_agent.controlled_llm import (
    ControlledRequestInterpreter,
    ControlledSemanticEnhancer,
    GroundedAnswerComposer,
)
from yelp_agent.session_memory.integration import MemoryAwareRequestInterpreter
from yelp_agent.session_memory.manager import SessionMemoryManager

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
    semantic_enabled: bool = False,
    semantic_candidate_limit: int = 30,
    fusion_alpha: float = 0.0,
    cross_encoder_enabled: bool = False,
    cross_encoder_candidate_limit: int = 20,
    cross_encoder_beta: float = 0.0,
    review_rag_enabled: bool = False,
    evidence_aggregation_enabled: bool = False,
    semantic_ranking_enabled: bool = False,
    query_aware_enabled: bool = False,
    semantic_enhancer: ControlledSemanticEnhancer | None = None,
    session_memory_manager: SessionMemoryManager | None = None,
    answer_composer: GroundedAnswerComposer | None = None,
    answer_evidence_limit: int = 12,
    action_policy: AllowedActionPolicy | None = None,
    router: ActionRouter | None = None,
    clock: Clock | None = None,
) -> AgentHarness:
    """Connect the Step 18/22/23/24 modules behind one runner interface."""

    return AgentHarness(
        agent_version=agent_version,
        interpreter=(
            MemoryAwareRequestInterpreter(session_memory_manager)
            if session_memory_manager is not None
            else ControlledRequestInterpreter(semantic_enhancer)
            if semantic_enhancer is not None
            else RuleBasedRequestInterpreter()
        ),
        action_policy=action_policy or RuleBasedActionPolicy(
            display_limit=display_limit,
            semantic_enabled=semantic_enabled,
            fusion_alpha=fusion_alpha,
            cross_encoder_enabled=cross_encoder_enabled,
            cross_encoder_beta=cross_encoder_beta,
            review_rag_enabled=review_rag_enabled,
            evidence_aggregation_enabled=evidence_aggregation_enabled,
            semantic_ranking_enabled=semantic_ranking_enabled,
            query_aware_enabled=query_aware_enabled,
        ),
        router=router or RuleRouter(
            display_limit=display_limit,
            semantic_enabled=semantic_enabled,
            semantic_candidate_limit=semantic_candidate_limit,
            fusion_alpha=fusion_alpha,
            cross_encoder_enabled=cross_encoder_enabled,
            cross_encoder_candidate_limit=cross_encoder_candidate_limit,
            cross_encoder_beta=cross_encoder_beta,
            review_rag_enabled=review_rag_enabled,
            evidence_aggregation_enabled=evidence_aggregation_enabled,
            semantic_ranking_enabled=semantic_ranking_enabled,
            query_aware_enabled=query_aware_enabled,
        ),
        executor=RegistryActionExecutor(
            registry,
            fallback=TerminalActionExecutor(
                fusion_alpha=fusion_alpha,
                cross_encoder_beta=cross_encoder_beta,
                answer_composer=answer_composer,
                answer_evidence_limit=answer_evidence_limit,
            ),
        ),
        fallback_handler=fallback_handler,
        budget=budget,
        clock=clock,
    )
