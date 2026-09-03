"""从命令行向真实第三批 Agent 发送一条消息。"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

from .application import build_restaurant_agent_application
from .runtime.schema import AgentTurnInput


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行真实餐饮 Agent")
    parser.add_argument("query", help="用户当前原话")
    parser.add_argument("--user-id", required=True, help="真实画像中的用户编号")
    parser.add_argument("--session-id", required=True, help="持久化会话编号")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="真实数据所在的项目根目录；默认使用当前代码所属项目",
    )
    parser.add_argument(
        "--follow-up",
        action="append",
        default=[],
        help="在同一进程和会话中继续发送的问题；可以重复提供",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="使用真实流式入口，并记录第一段文字到达时间",
    )
    return parser.parse_args()


async def _run() -> None:
    # Windows 终端默认代码页可能把中文输出成乱码，演示入口统一写 UTF-8。
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = _arguments()
    project_root = (
        args.project_root.resolve()
        if args.project_root is not None
        else Path(__file__).resolve().parents[3]
    )
    startup_started = perf_counter()
    application = build_restaurant_agent_application(project_root)
    startup_ms = (perf_counter() - startup_started) * 1000
    turns: list[dict[str, object]] = []
    async with application:
        for query in [args.query, *args.follow_up]:
            turn_started = perf_counter()
            turn_input = AgentTurnInput(
                user_id=args.user_id,
                session_id=args.session_id,
                message=query,
                request_time=datetime.now(UTC),
            )
            first_delta_ms: float | None = None
            delta_count = 0
            streamed_text: list[str] = []
            if args.stream:
                result = None
                async for event in application.handle_stream(turn_input):
                    if event.type == "answer_delta":
                        if first_delta_ms is None:
                            first_delta_ms = event.elapsed_ms
                        delta_count += 1
                        streamed_text.append(event.delta or "")
                    else:
                        result = event.result
                if result is None:
                    raise RuntimeError("stream ended without a final turn result")
            else:
                result = await application.handle(turn_input)
            turns.append(
                {
                    "query": query,
                    "wall_latency_ms": (perf_counter() - turn_started) * 1000,
                    "first_answer_delta_ms": first_delta_ms,
                    "answer_delta_count": delta_count,
                    "streamed_text_matches_final_answer": (
                        None
                        if not args.stream
                        else "".join(streamed_text) == (result.answer or "")
                    ),
                    "result": result.model_dump(mode="json"),
                }
            )
    payload: dict[str, object]
    if args.follow_up or args.stream:
        payload = {
            "application_startup_ms": startup_ms,
            "process_reused_for_all_turns": True,
            "turns": turns,
        }
    else:
        payload = turns[0]["result"]  # 保持原有单轮命令输出格式。
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(_run())
