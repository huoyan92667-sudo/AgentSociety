"""Assembly and lifetime ownership for the constrained Router."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Self

from yelp_agent.agent.llm import OpenAICompatibleLLM
from yelp_agent.config import AgentConfig, load_llm_environment
from yelp_agent.controlled_llm.cache import SqliteControlledLLMCache
from yelp_agent.controlled_llm.gateway import ControlledJSONCaller
from yelp_agent.controlled_llm.ledger import ControlledLLMUsageLedger
from yelp_agent.rule_router.router import RuleRouter

from .choices import ConstrainedDecisionBuilder
from .config import ConstrainedLLMRouterConfig
from .policy import ConstrainedActionPolicy
from .router import ConstrainedLLMRouter


@dataclass(slots=True)
class ConstrainedLLMRouterRuntime:
    router: ConstrainedLLMRouter
    action_policy: ConstrainedActionPolicy
    ledger: ControlledLLMUsageLedger
    cache: SqliteControlledLLMCache

    def close(self) -> None:
        self.cache.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def build_constrained_llm_router_runtime(
    *,
    project_root: str | Path,
    config: ConstrainedLLMRouterConfig,
    fallback_router: RuleRouter,
    environment: Mapping[str, str] | None = None,
) -> ConstrainedLLMRouterRuntime:
    env = load_llm_environment(environment)
    cache = SqliteControlledLLMCache(Path(project_root) / config.cache_relative_path)
    ledger = ControlledLLMUsageLedger()
    generator = OpenAICompatibleLLM(
        AgentConfig(
            enabled=config.enabled,
            temperature=config.temperature,
            timeout_seconds=config.timeout_seconds,
            max_retries=config.max_retries,
            max_tokens=config.max_output_tokens,
            response_format_json=config.response_format_json,
            thinking=config.thinking,
        ),
        env,
    )
    caller = ControlledJSONCaller(
        generator=generator,
        model_name=env.model,
        cache=cache,
        ledger=ledger,
    )
    builder = ConstrainedDecisionBuilder(
        fallback_router,
        maximum_choices=config.maximum_choices,
    )
    return ConstrainedLLMRouterRuntime(
        router=ConstrainedLLMRouter(
            config=config,
            caller=caller,
            builder=builder,
            fallback_router=fallback_router,
        ),
        action_policy=ConstrainedActionPolicy(builder),
        ledger=ledger,
        cache=cache,
    )
