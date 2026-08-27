"""不访问网络的脚本模型，用于完整测试 Agent 循环。"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence

from ..runtime.schema import ModelRequest, ModelResponse


class ScriptedLanguageModel:
    """按照预设顺序返回响应，并保存每次实际收到的请求。"""

    def __init__(self, responses: Sequence[ModelResponse]) -> None:
        self._responses = deque(item.model_copy(deep=True) for item in responses)
        self.requests: list[ModelRequest] = []

    async def generate(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request.model_copy(deep=True))
        if not self._responses:
            raise AssertionError("scripted model has no response left")
        return self._responses.popleft().model_copy(deep=True)

    @property
    def remaining_response_count(self) -> int:
        return len(self._responses)
