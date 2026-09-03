from __future__ import annotations

from types import SimpleNamespace

from yelp_agent.agent_platform import AgentModelSettings, OpenAICompatibleAgentModel
from yelp_agent.agent_platform.runtime.schema import (
    FinalAnswerAction,
    ModelMessage,
    ModelRequest,
    ToolCallsAction,
    ToolSchema,
)


class RecordingCompletions:
    def __init__(self, responses: list[object]) -> None:
        self.responses = list(responses)
        self.requests: list[dict[str, object]] = []

    async def create(self, **kwargs: object) -> object:
        self.requests.append(kwargs)
        return self.responses.pop(0)


class FakeClient:
    def __init__(self, responses: list[object]) -> None:
        self.chat = SimpleNamespace(completions=RecordingCompletions(responses))


def _request(messages: list[ModelMessage]) -> ModelRequest:
    return ModelRequest(
        session_id="session-1",
        turn_id="turn-1",
        step_index=1,
        system_prompt="根据问题自行选择工具。",
        messages=messages,
        tools=[
            ToolSchema(
                name="lookup_business_facts",
                description="读取商家事实。",
                input_schema={
                    "type": "object",
                    "properties": {
                        "business_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                        }
                    },
                    "required": ["business_ids"],
                },
            )
        ],
    )


def test_real_model_converts_native_tool_call_and_usage() -> None:
    async def scenario() -> None:
        response = SimpleNamespace(
            id="provider-request-1",
            model="deepseek-chat",
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=None,
                        tool_calls=[
                            SimpleNamespace(
                                id="call-1",
                                function=SimpleNamespace(
                                    name="lookup_business_facts",
                                    arguments='{"business_ids":["business-1"]}',
                                ),
                            )
                        ],
                    )
                )
            ],
            usage=SimpleNamespace(prompt_tokens=120, completion_tokens=25),
        )
        client = FakeClient([response])
        model = OpenAICompatibleAgentModel(
            AgentModelSettings(
                api_key="secret-key",
                base_url="https://api.deepseek.com",
                model="deepseek-chat",
            ),
            client=client,
        )

        result = await model.generate(
            _request([ModelMessage(role="user", content="第二家能停车吗？")])
        )

        assert isinstance(result.action, ToolCallsAction)
        assert result.action.calls[0].tool_name == "lookup_business_facts"
        assert result.action.calls[0].arguments == {"business_ids": ["business-1"]}
        assert result.provider == "deepseek"
        assert result.usage.input_tokens == 120
        assert result.usage.output_tokens == 25
        sent = client.chat.completions.requests[0]
        assert sent["tool_choice"] == "auto"
        assert sent["tools"][0]["function"]["name"] == "lookup_business_facts"

    import asyncio

    asyncio.run(scenario())


def test_real_model_sends_tool_history_back_and_accepts_final_text() -> None:
    async def scenario() -> None:
        response = SimpleNamespace(
            id="provider-request-2",
            model="deepseek-chat",
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="第二家记录显示可以停车。",
                        tool_calls=None,
                    )
                )
            ],
            usage=SimpleNamespace(prompt_tokens=80, completion_tokens=12),
        )
        client = FakeClient([response])
        model = OpenAICompatibleAgentModel(
            AgentModelSettings(api_key="secret-key", model="deepseek-chat"),
            client=client,
        )
        request = _request(
            [
                ModelMessage(role="user", content="第二家能停车吗？"),
                ModelMessage(
                    role="assistant",
                    tool_calls=[
                        {
                            "call_id": "call-1",
                            "tool_name": "lookup_business_facts",
                            "arguments": {"business_ids": ["business-1"]},
                        }
                    ],
                ),
                ModelMessage(
                    role="tool",
                    content='{"parking_available":true}',
                    tool_call_id="call-1",
                    tool_name="lookup_business_facts",
                ),
            ]
        )

        result = await model.generate(request)

        assert isinstance(result.action, FinalAnswerAction)
        assert result.action.answer == "第二家记录显示可以停车。"
        sent_messages = client.chat.completions.requests[0]["messages"]
        assert sent_messages[-2]["tool_calls"][0]["id"] == "call-1"
        assert sent_messages[-1] == {
            "role": "tool",
            "tool_call_id": "call-1",
            "content": '{"parking_available":true}',
        }

    import asyncio

    asyncio.run(scenario())


def test_settings_hide_api_key() -> None:
    settings = AgentModelSettings.from_environment(
        {
            "OPENAI_API_KEY": "secret-value",
            "OPENAI_BASE_URL": "https://api.deepseek.com",
            "OPENAI_MODEL": "deepseek-chat",
        }
    )

    assert "secret-value" not in repr(settings)
    assert settings.api_key.get_secret_value() == "secret-value"


def test_malformed_tool_arguments_are_returned_to_pipeline_for_correction() -> None:
    async def scenario() -> None:
        response = SimpleNamespace(
            id="provider-request-invalid",
            model="deepseek-chat",
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=None,
                        tool_calls=[
                            SimpleNamespace(
                                id="call-invalid",
                                function=SimpleNamespace(
                                    name="lookup_business_facts",
                                    arguments='{"business_ids":["business-1"',
                                ),
                            )
                        ],
                    )
                )
            ],
            usage=SimpleNamespace(prompt_tokens=50, completion_tokens=10),
        )
        model = OpenAICompatibleAgentModel(
            AgentModelSettings(api_key="secret-key", model="deepseek-chat"),
            client=FakeClient([response]),
        )

        result = await model.generate(
            _request([ModelMessage(role="user", content="查一下商家")])
        )

        assert isinstance(result.action, ToolCallsAction)
        assert "__invalid_arguments_json__" in result.action.calls[0].arguments

    import asyncio

    asyncio.run(scenario())
