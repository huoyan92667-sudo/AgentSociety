# Agent PostgreSQL 持久化

## 保存内容

PostgreSQL 是对话和运行状态的真实数据来源，包含八张表：

| 表 | 保存内容 |
|---|---|
| `agent_sessions` | 会话归属、状态、当前运行轮次和事件序号 |
| `agent_turns` | 每轮用户问题、最终回答、状态、工具和总消耗 |
| `agent_events` | 按顺序只追加的完整运行事实 |
| `agent_domain_state_versions` | 餐饮、旅游等领域的完整状态版本 |
| `agent_result_artifacts` | 推荐审计、评论证据等大结果的编号和索引 |
| `agent_llm_calls` | 每次模型调用的模型、用途、词元和耗时 |
| `agent_working_memories` | 每个会话最新的关注对象、最近结果顺序、业务状态版本和摘要水位 |
| `agent_conversation_episodes` | 已结束旧对话的短摘要、来源轮次、真实对象和未解决问题 |

Qdrant 继续保存评论片段向量，不保存严格有序的会话状态。

## 数据库地址

本地开发可以先启动仓库自带的 PostgreSQL：

```powershell
docker compose `
  -f src/yelp_agent/agent_platform/persistence/docker-compose.yml `
  up -d
```

默认开发端口是 `5433`，避免占用机器上可能已有的 `5432`。

正式连接地址通过环境变量提供：

```powershell
$env:AGENT_DATABASE_URL = "postgresql+asyncpg://agent:agent_dev_password@localhost:5433/agent"
```

默认密码只供本地开发，部署时必须通过环境变量替换。

程序中读取：

```python
from pathlib import Path

from yelp_agent.agent_platform import (
    AgentDatabase,
    DatabaseSettings,
    PostgresAgentPersistence,
)
from yelp_agent.agent_platform.results import LocalJsonContentStore

database = AgentDatabase(DatabaseSettings.from_environment())
content_store = LocalJsonContentStore(Path("runs/agent_artifacts"))
persistence = PostgresAgentPersistence(
    database.sessions,
    content_store=content_store,
)
```

连接地址使用 `SecretStr` 保存，不会被普通模型输出或日志自动打印。

## 执行数据库升级

生产环境不能调用测试用的 `create_schema_for_tests()`，必须执行 Alembic：

```powershell
.\.venv\Scripts\python.exe -m alembic `
  -c src/yelp_agent/agent_platform/persistence/alembic.ini `
  upgrade head
```

当前最新升级号为：

```text
0003_turn_index
```

第二次升级增加记忆表；第三次升级给每个会话的轮次补上严格递增编号。
两次升级都不会删除原始轮次、事件、领域状态或已有结果。

## 长对话怎样保存

完整事实仍保存在轮次和事件表，方便中断恢复和审计。交给模型的上下文则是一个有限视图：

```text
最近六轮用户原话和最终回答
+ 当前轮完整工具调用链
+ 一份最新工作记忆
+ 最近两段旧话题摘要
```

达到门槛后，最老的已结束轮次会生成 `agent_conversation_episodes` 记录，
工作记忆中的“已总结到第几轮”随之向前推进。这里使用严格递增的轮次编号，
不会因两轮结束时间恰好相同而漏掉原文。模型生成的是摘要文字；
来源轮次、商家或其他对象编号、结果编号和时间由程序从真实记录补入。

用户追问旧事时，主模型可调用 `search_conversation_memory`。默认读取摘要；只有摘要不足时
才取回对应的用户原话和最终回答。旧工具调用链和内部候选不会再次塞回模型。

## 接入 AgentRuntime

```python
runtime = AgentRuntime(
    model=model,
    session_store=persistence,
    result_store=persistence,
    tools=tools,
)

# 应用启动时执行一次，关闭上次异常退出遗留的运行中轮次。
recovery = await runtime.recover_interrupted()

result = await runtime.handle(turn_input)
```

一轮开始时，会在同一个事务中：

```text
创建轮次
更新会话状态
写入 turn/start
写入 user/message
```

一轮结束时，也会在同一个事务中：

```text
更新轮次摘要
清除当前运行轮次
更新会话状态
写入 turn/end
```

主模型调用记录和对应的 `assistant/message` 事件同样原子写入。

## 领域状态版本

餐饮状态不会原地覆盖：

```python
state = await persistence.save_domain_state(
    DomainStateWrite(
        session_id="session-1",
        domain="restaurant",
        state=unified_state.model_dump(mode="json"),
        expected_previous_version=2,
    ),
    now=current_time,
)
```

如果数据库最新版本已经不是2，保存会被拒绝，避免两个并发请求互相覆盖。

## 大结果

小结果直接保存在 PostgreSQL 的 JSON 字段中。超过默认256KB时：

```text
完整JSON
→ gzip压缩文件
→ PostgreSQL保存结果编号、摘要、相对地址、大小和SHA-256
```

工具结果超过默认64KB时，执行后处理会提前保存完整内容，并把会话事件中的
巨大正文替换为：

```json
{
  "result_id": "...",
  "kind": "tool_result/search_review_evidence",
  "summary": {
    "tool_name": "search_review_evidence",
    "status": "success"
  },
  "size_bytes": 120000,
  "sha256": "..."
}
```

后续用户追问时，通过结果编号按需读取完整内容。读取时会校验用户归属和
SHA-256，不能读取其他用户的结果。
