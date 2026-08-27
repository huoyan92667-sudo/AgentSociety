"""支持 DeepSeek 等 OpenAI 兼容接口的真实 Agent 模型。"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from time import perf_counter
from typing import Any, Protocol

from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import Field, SecretStr, field_validator

from yelp_agent.models import StrictModel

from ..runtime.schema import (
    FinalAnswerAction,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TokenUsage,
    ToolCall,
    ToolCallsAction,
)


class AgentModelSettings(StrictModel):
    """真实主模型配置；密钥不会出现在日志或模型上下文中。"""

    api_key: SecretStr
    model: str = Field(min_length=1)
    base_url: str | None = None
    temperature: float = Field(default=0.0, ge=0, le=2)
    max_tokens: int | None = Field(default=3000, ge=1)
    timeout_seconds: float = Field(default=120.0, gt=0)
    max_retries: int = Field(default=2, ge=0, le=10)

    @field_validator("model")
    @classmethod
    def strip_model(cls, value: str) -> str:
        return value.strip()

    @classmethod
    def from_environment(
        cls,
        environment: Mapping[str, str] | None = None,
    ) -> AgentModelSettings:
        """沿用项目现有 OPENAI_* 配置，因此无需再保存一份 DeepSeek 密钥。"""

        if environment is None:
            load_dotenv()
            environment = os.environ

        def required(name: str) -> str:
            value = environment.get(name, "").strip()
            if not value:
                raise ValueError(f"{name} is required")
            return value

        base_url = environment.get("OPENAI_BASE_URL", "").strip() or None
        return cls(
            api_key=SecretStr(required("OPENAI_API_KEY")),
            model=required("OPENAI_MODEL"),
            base_url=base_url,
        )


class _AsyncOpenAIClient(Protocol):
    chat: Any


class OpenAICompatibleAgentModel:
    """把真实模型的自然回答和原生工具调用转换成框架统一动作。"""

    def __init__(
        self,
        settings: AgentModelSettings,
        *,
        client: _AsyncOpenAIClient | None = None,
    ) -> None:
        self._settings = settings
        if client is None:
            options: dict[str, Any] = {
                "api_key": settings.api_key.get_secret_value(),
                "timeout": settings.timeout_seconds,
                "max_retries": settings.max_retries,
            }
            if settings.base_url is not None:
                options["base_url"] = settings.base_url
            client = AsyncOpenAI(**options)
        self._client = client

    async def generate(self, request: ModelRequest) -> ModelResponse:
        """调用一次真实模型；是否使用工具由模型根据当前问题自行决定。"""

        payload: dict[str, Any] = {
            "model": self._settings.model,
            "messages": self._messages(request),
            "temperature": self._settings.temperature,
        }
        if self._settings.max_tokens is not None:
            payload["max_tokens"] = self._settings.max_tokens
        if request.tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.input_schema,
                    },
                }
                for tool in request.tools
            ]
            payload["tool_choice"] = "auto"

        started_at = perf_counter()
        response = await self._client.chat.completions.create(**payload)
        latency_ms = (perf_counter() - started_at) * 1000.0
        if not response.choices:
            raise RuntimeError("model returned no choices")
        message = response.choices[0].message
        raw_tool_calls = list(getattr(message, "tool_calls", None) or [])
        if raw_tool_calls:
            calls = [self._tool_call(item) for item in raw_tool_calls]
            action = ToolCallsAction(calls=calls)
        else:
            content = getattr(message, "content", None)
            if not isinstance(content, str) or not content.strip():
                raise RuntimeError("model returned neither text nor tool calls")
            action = FinalAnswerAction(answer=content.strip())

        usage = getattr(response, "usage", None)
        return ModelResponse(
            action=action,
            model=str(getattr(response, "model", None) or self._settings.model),
            provider=self._provider_name(),
            usage=TokenUsage(
                input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
                output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            ),
            latency_ms=latency_ms,
            provider_request_id=self._request_id(response),
        )

    @staticmethod
    def _tool_call(raw: Any) -> ToolCall:
        """严格解析模型返回的工具参数，坏 JSON 不会进入工具执行层。"""

        function = getattr(raw, "function", None)
        call_id = getattr(raw, "id", None)
        name = getattr(function, "name", None)
        arguments_text = getattr(function, "arguments", None)
        if not isinstance(call_id, str) or not call_id:
            raise ValueError("model tool call is missing an ID")
        if not isinstance(name, str) or not name:
            raise ValueError("model tool call is missing a name")
        if not isinstance(arguments_text, str):
            raise TypeError("model tool arguments must be JSON text")
        arguments = json.loads(arguments_text)
        if not isinstance(arguments, dict):
            raise TypeError("model tool arguments must be a JSON object")
        return ToolCall(call_id=call_id, tool_name=name, arguments=arguments)

    @classmethod
    def _messages(cls, request: ModelRequest) -> list[dict[str, Any]]:
        """把框架历史还原成供应商接受的 system/user/assistant/tool 消息。"""

        values: list[dict[str, Any]] = [
            {"role": "system", "content": request.system_prompt}
        ]
        values.extend(cls._message(item) for item in request.messages)
        return values

    @staticmethod
    def _message(message: ModelMessage) -> dict[str, Any]:
        if message.role == "user":
            return {"role": "user", "content": message.content}
        if message.role == "tool":
            return {
                "role": "tool",
                "tool_call_id": message.tool_call_id,
                "content": message.content,
            }
        if message.tool_calls:
            return {
                "role": "assistant",
                "content": message.content,
                "tool_calls": [
                    {
                        "id": call.call_id,
                        "type": "function",
                        "function": {
                            "name": call.tool_name,
                            "arguments": json.dumps(
                                call.arguments,
                                ensure_ascii=False,
                                separators=(",", ":"),
                            ),
                        },
                    }
                    for call in message.tool_calls
                ],
            }
        return {"role": "assistant", "content": message.content}

    def _provider_name(self) -> str:
        base_url = (self._settings.base_url or "").lower()
        return "deepseek" if "deepseek" in base_url else "openai_compatible"

    @staticmethod
    def _request_id(response: Any) -> str | None:
        for name in ("_request_id", "request_id", "id"):
            value = getattr(response, name, None)
            if isinstance(value, str) and value:
                return value
        return None
