"""模型适配接口及测试实现。"""

from .adapter import LanguageModel
from .fake import ScriptedLanguageModel
from .openai_compatible import AgentModelSettings, OpenAICompatibleAgentModel

__all__ = [
    "AgentModelSettings",
    "LanguageModel",
    "OpenAICompatibleAgentModel",
    "ScriptedLanguageModel",
]
