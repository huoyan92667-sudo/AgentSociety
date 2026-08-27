"""把过大的工具结果移出会话事件的执行后处理。"""

from __future__ import annotations

import json
from datetime import UTC, datetime

from ..tools.definition import ToolExecution
from ..tools.result import ToolResult
from .schema import ResultArtifactDraft
from .store import ResultStore


class PersistLargeToolResultHook:
    """保存完整大结果，只把结果编号和摘要留给会话及模型。"""

    def __init__(
        self,
        store: ResultStore,
        *,
        threshold_bytes: int = 64 * 1024,
    ) -> None:
        if threshold_bytes < 1024:
            raise ValueError("tool result threshold must be at least 1024 bytes")
        self._store = store
        self._threshold_bytes = threshold_bytes

    async def __call__(
        self,
        execution: ToolExecution,
        result: ToolResult,
    ) -> ToolResult:
        if result.status != "success" or result.artifact_id is not None:
            return result
        raw = json.dumps(
            result.value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        if len(raw) <= self._threshold_bytes:
            return result

        summary = {
            "tool_name": result.tool_name,
            "status": result.status,
            "warning_count": len(result.warnings),
        }
        artifact = await self._store.save_result(
            ResultArtifactDraft(
                session_id=execution.context.session_id,
                turn_id=execution.context.turn_id,
                kind=f"tool_result/{result.tool_name}",
                summary=summary,
                content=result.value,
            ),
            now=datetime.now(UTC),
        )
        reference = {
            "result_id": artifact.result_id,
            "kind": artifact.kind,
            "summary": artifact.summary,
            "size_bytes": artifact.size_bytes,
            "sha256": artifact.sha256,
        }
        model_content = self._add_result_id(result.model_content, artifact.result_id)
        return result.model_copy(
            update={
                "value": reference,
                "model_content": model_content,
                "artifact_id": artifact.result_id,
                "warnings": [*result.warnings, "full_result_stored_separately"],
            }
        )

    @staticmethod
    def _add_result_id(model_content: str, result_id: str) -> str:
        """把大结果编号放进给后续模型看的小摘要，不重新塞回完整正文。"""

        try:
            value = json.loads(model_content)
        except (json.JSONDecodeError, TypeError):
            return f"{model_content}\n完整结果编号：{result_id}"
        if not isinstance(value, dict):
            return f"{model_content}\n完整结果编号：{result_id}"
        value["result_id"] = result_id
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
