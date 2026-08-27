"""主循环依赖的模型接口。"""

from __future__ import annotations

from typing import Protocol

from ..runtime.schema import ModelRequest, ModelResponse


class LanguageModel(Protocol):
    """任何真实或测试模型只需完成一次结构化动作生成。"""

    async def generate(self, request: ModelRequest) -> ModelResponse: ...
