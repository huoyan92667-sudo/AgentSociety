"""组装可实际调用大模型的问题编译器。"""

from __future__ import annotations

from yelp_agent.agent.llm import OpenAICompatibleLLM
from yelp_agent.config import AgentConfig
from yelp_agent.recommendation_v2.query_compiler.compiler import QueryCompiler


def build_query_compiler() -> QueryCompiler:
    """读取项目现有模型配置，建立只输出 JSON 的问题编译器。"""

    generator = OpenAICompatibleLLM.from_environment(
        AgentConfig(
            enabled=True,
            temperature=0.0,
            timeout_seconds=90,
            max_retries=2,
            max_tokens=3000,
            response_format_json=True,
            thinking="disabled",
        )
    )
    return QueryCompiler(generator)
