"""用于检查问题编译器的六个场景中文问题。"""

from __future__ import annotations

from yelp_agent.models import StrictModel
from yelp_agent.recommendation_v2.query_compiler.compiler import (
    QueryCompiler,
    QueryCompilerAttempt,
    QueryCompilerRequest,
)
from yelp_agent.recommendation_v2.schema import BaselineSceneKind


class SceneQueryExample(StrictModel):
    """一条场景问题及其预期场景，只用于人工检查和测试。"""

    scene: BaselineSceneKind
    scene_label: str
    query_text: str


class SceneQueryCompilation(StrictModel):
    """一条场景问题的模型原始 JSON 和最终编译结果。"""

    expected_scene: BaselineSceneKind
    scene_label: str
    query_text: str
    attempt: QueryCompilerAttempt


SIX_SCENE_QUERIES: tuple[SceneQueryExample, ...] = (
    SceneQueryExample(
        scene="casual",
        scene_label="随便吃",
        query_text=(
            "今天就随便吃点，想吃川菜，距离3公里以内，价格二档以内，"
            "食物质量第一，性价比第二，距离近第三。"
        ),
    ),
    SceneQueryExample(
        scene="date",
        scene_label="约会",
        query_text=(
            "今晚和对象约会，想吃日料，距离5公里以内，价格三档以内，"
            "安静最重要，适合约会其次，服务第三。"
        ),
    ),
    SceneQueryExample(
        scene="business",
        scene_label="商务",
        query_text=(
            "明天和客户商务午餐，想吃中餐，距离6公里以内，价格三档以内，"
            "安静第一，服务第二，适合多人第三。"
        ),
    ),
    SceneQueryExample(
        scene="friends",
        scene_label="朋友聚餐",
        query_text=(
            "周末和朋友聚餐，想吃火锅，距离8公里以内，价格二档以内，"
            "适合多人第一，食物质量第二，性价比第三。"
        ),
    ),
    SceneQueryExample(
        scene="family",
        scene_label="家庭聚餐",
        query_text=(
            "周日一家人聚餐，想吃粤菜，距离5公里以内，价格二档以内，"
            "适合家庭第一，干净第二，停车方便第三。"
        ),
    ),
    SceneQueryExample(
        scene="solo",
        scene_label="一个人吃",
        query_text=(
            "今晚我一个人简单吃点，想吃面，距离3公里以内，价格二档以内，"
            "距离近第一，少排队第二，性价比第三。"
        ),
    ),
)


def compile_six_scene_queries(
    compiler: QueryCompiler,
) -> tuple[SceneQueryCompilation, ...]:
    """依次调用大模型编译六个场景问题，保留每次真实返回。"""

    results: list[SceneQueryCompilation] = []
    for example in SIX_SCENE_QUERIES:
        attempt = compiler.compile(QueryCompilerRequest(query_text=example.query_text))
        results.append(
            SceneQueryCompilation(
                expected_scene=example.scene,
                scene_label=example.scene_label,
                query_text=example.query_text,
                attempt=attempt,
            )
        )
    return tuple(results)
