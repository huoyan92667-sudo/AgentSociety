"""工具注册表：控制模型能看到什么，以及调用名称对应哪个实现。"""

from __future__ import annotations

from collections.abc import Iterable

from ..runtime.schema import ToolSchema
from .definition import ToolDefinition


class ToolRegistry:
    """保持注册顺序并拒绝重复名称。"""

    def __init__(self, tools: Iterable[ToolDefinition] = ()) -> None:
        self._definitions: dict[str, ToolDefinition] = {}
        for definition in tools:
            self.register(definition)

    def register(self, definition: ToolDefinition) -> None:
        if definition.name in self._definitions:
            raise ValueError(f"duplicate tool name: {definition.name}")
        self._definitions[definition.name] = definition

    def get(
        self,
        name: str,
        *,
        allowed_names: set[str] | None = None,
    ) -> ToolDefinition | None:
        if allowed_names is not None and name not in allowed_names:
            return None
        return self._definitions.get(name)

    def schemas(
        self,
        *,
        allowed_names: set[str] | None = None,
    ) -> list[ToolSchema]:
        return [
            definition.model_schema()
            for name, definition in self._definitions.items()
            if allowed_names is None or name in allowed_names
        ]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._definitions)
