"""通过 Claude Code 调用已经配置好的 GLM，并强制结构化返回。"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Sequence
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from .schema import ClaudeWorkerTrace

T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True, slots=True)
class ClaudeStructuredResult:
    """一次模型调用解析后的内容和用量记录。"""

    value: BaseModel
    trace: ClaudeWorkerTrace
    raw_wrapper: dict[str, object]


class ClaudeStructuredOutputError(ValueError):
    """模型调用成功，但其结构化内容没有通过固定数据结构。"""

    def __init__(
        self,
        message: str,
        *,
        wrapper: dict[str, object],
        trace: ClaudeWorkerTrace,
    ) -> None:
        super().__init__(message)
        self.wrapper = wrapper
        self.trace = trace


class ClaudeCodeWorker:
    """只暴露一个结构化生成入口，隐藏命令行和返回包装细节。"""

    def __init__(
        self,
        *,
        model: str = "glm-5.1",
        timeout_seconds: int = 900,
        executable: str | Sequence[str] = "claude",
        legacy_cli: bool = False,
    ) -> None:
        self.model = model
        self.timeout_seconds = timeout_seconds
        # Windows下的旧版Claude Code需要用“node + cli.js”启动，因此这里允许
        # 调用入口由多个参数组成，而不是把整条命令拼成一个不安全的字符串。
        self.command_prefix = (
            (executable,) if isinstance(executable, str) else tuple(executable)
        )
        self.legacy_cli = legacy_cli

    def generate(
        self,
        prompt: str,
        output_model: type[T],
        *,
        system_prompt: str | None = None,
    ) -> tuple[T, ClaudeWorkerTrace, dict[str, object]]:
        """把输入走标准输入交给Claude Code，并允许调用方固定系统要求。"""

        schema = json.dumps(
            output_model.model_json_schema(),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        command = [
            *self.command_prefix,
            "-p",
            "--model",
            self.model,
            "--output-format",
            "json",
        ]
        if self.legacy_cli:
            # Z.AI验证过的Claude Code 2.0.14尚不支持--json-schema等新参数。
            # 用最小系统说明约束其只完成结构化生成，再由Pydantic做同样严格的校验。
            command.extend(
                [
                    "--permission-mode",
                    "plan",
                    "--system-prompt",
                    system_prompt
                    or "你只完成用户给出的结构化生成任务，不调用工具，不解释，只返回JSON。",
                ]
            )
        else:
            command.extend(
                [
                    "--tools",
                    "",
                    "--no-session-persistence",
                    "--safe-mode",
                    "--prompt-suggestions",
                    "false",
                    "--json-schema",
                    schema,
                ]
            )
            if system_prompt is not None:
                # --system-prompt会替换Claude Code默认系统说明，避免项目上下文
                # 和工具说明混进教师模型的判断输入。
                command.extend(["--system-prompt", system_prompt])
        environment = os.environ.copy()
        environment["MAX_THINKING_TOKENS"] = "0"
        environment["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"
        completed = subprocess.run(
            command,
            input=prompt,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=self.timeout_seconds,
            check=False,
            env=environment,
        )
        if completed.returncode != 0:
            excerpt = _safe_excerpt(completed.stderr or completed.stdout)
            raise RuntimeError(
                f"Claude Code failed with exit code {completed.returncode}: {excerpt}"
            )
        wrapper = json.loads(completed.stdout)
        return self.parse_wrapper(
            wrapper,
            output_model,
            stderr=completed.stderr,
        )

    def parse_wrapper(
        self,
        wrapper: dict[str, object],
        output_model: type[T],
        *,
        stderr: str = "",
    ) -> tuple[T, ClaudeWorkerTrace, dict[str, object]]:
        """解析已保存的原始包装，支持失败后断点续跑而不重复调用。"""

        trace = self.trace_from_wrapper(wrapper, stderr=stderr)
        try:
            raw_result = wrapper.get("result")
            if isinstance(raw_result, str):
                payload = json.loads(_strip_code_fence(raw_result))
            elif isinstance(raw_result, dict):
                payload = raw_result
            else:
                raise ValueError("Claude Code returned no structured result")
            value = output_model.model_validate(payload)
        except (json.JSONDecodeError, TypeError, ValueError, ValidationError) as exc:
            raise ClaudeStructuredOutputError(
                str(exc),
                wrapper=wrapper,
                trace=trace,
            ) from exc
        return value, trace, wrapper

    def trace_from_wrapper(
        self,
        wrapper: dict[str, object],
        *,
        stderr: str = "",
    ) -> ClaudeWorkerTrace:
        """即使模型内容不合格，也能从原始包装中统计真实调用用量。"""

        usage = wrapper.get("usage") if isinstance(wrapper.get("usage"), dict) else {}
        output_details = (
            usage.get("output_tokens_details")
            if isinstance(usage.get("output_tokens_details"), dict)
            else {}
        )
        trace = ClaudeWorkerTrace(
            model=self.model,
            duration_ms=int(wrapper.get("duration_api_ms") or wrapper.get("duration_ms") or 0),
            input_tokens=_optional_int(usage.get("input_tokens")),
            output_tokens=_optional_int(usage.get("output_tokens")),
            thinking_tokens=_optional_int(output_details.get("thinking_tokens")),
            reported_cost_usd=_optional_float(wrapper.get("total_cost_usd")),
            stderr_excerpt=_safe_excerpt(stderr) or None,
        )
        return trace


def load_prompt(path: str | Path) -> str:
    """提示词单独存文件，便于用户直接检查和修改。"""

    return Path(path).read_text(encoding="utf-8")


def _strip_code_fence(value: str) -> str:
    text = value.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return text


def _safe_excerpt(value: str, limit: int = 1000) -> str:
    return " ".join(value.strip().split())[:limit]


def _optional_int(value: object) -> int | None:
    return None if value is None else int(value)


def _optional_float(value: object) -> float | None:
    return None if value is None else float(value)
