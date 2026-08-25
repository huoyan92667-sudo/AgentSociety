"""建立读取完整历史的一次调用偏好融合器。"""

from __future__ import annotations

from yelp_agent.agent.llm import OpenAICompatibleLLM
from yelp_agent.config import AgentConfig
from yelp_agent.recommendation_v2.business_facts import BusinessFactCatalog
from yelp_agent.recommendation_v2.preference_fusion.fusion import PreferenceFusion
from yelp_agent.recommendation_v2.tools import BusinessFactsTool


def build_preference_fusion(
    business_catalog: BusinessFactCatalog | None = None,
) -> PreferenceFusion:
    """复用项目现有模型配置，固定使用严格 JSON 和关闭随机性。"""

    generator = OpenAICompatibleLLM.from_environment(
        AgentConfig(
            enabled=True,
            temperature=0.0,
            timeout_seconds=90,
            max_retries=2,
            max_tokens=8000,
            response_format_json=True,
            thinking="disabled",
        )
    )
    return PreferenceFusion(
        generator,
        business_tool=(
            None if business_catalog is None else BusinessFactsTool(business_catalog)
        ),
    )
