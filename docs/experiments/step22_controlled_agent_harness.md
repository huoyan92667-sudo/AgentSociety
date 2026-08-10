# 第22步：通用受控 Agent Harness

## 1. 本步完成了什么

第22步把此前已经存在的请求解析、任务判断和评测输出格式连接成了一个统一的 Agent 运行框架。

```text
用户问题
  → 第18步 RecommendationRequest
  → 第19步 DecisionReadiness
  → 建立 AgentState
  → 计算当前允许动作
  → Router 每次选择一个动作
  → 安全校验
  → Executor 执行动作
  → 观察结果写回状态
  → 继续 / 追问暂停 / 完成 / 安全回退
  → 第21步 AgentScenarioRun
```

公开入口只有两个：

```python
result = harness.start(visible_scenario)
result = harness.resume(result.session, user_turn)
```

`start()` 和 `resume()` 都只接收用户可见数据。Harness 的构造函数、AgentState、Router 和 Executor 均不接收 Ground Truth、正确商家、隐藏意图或脚本用户的未来回复。

## 2. AgentState 里保存什么

代码中的持久化类名为 `AgentSession`，同时公开 `AgentState` 作为架构名称。它保存：

- 当前结构化请求和任务判断；
- 当前轮次、状态和允许动作；
- 已执行动作、工具观察和候选商家作用域；
- 已执行工具签名，用于阻止原地重复调用；
- 总步骤、工具、语义模型和 Review RAG 调用次数；
- Token、成本和耗时；
- 已完成的第21步 Turn Trace；
- 回退状态和原因。

暂停状态可用 `session.to_json()` 保存，用 `AgentSession.from_json()` 重新加载，之后再传给 `resume()`。因此暂停和恢复不是只在同一个 Python 进程内有效。

## 3. 状态和动作循环

每一步只允许 Router 返回一个 `AgentDecision`：

```python
AgentDecision(
    action="retrieve_business_reviews",
    arguments={"business_id": "business-1", "aspect": "quiet_environment"},
    reason_code="REVIEW_EVIDENCE_REQUIRED",
    tool_name="REVIEW_SEARCH",
    tool_kind="review_rag",
)
```

`reason_code` 是可审计的简短原因，不保存大模型的思维过程。非终止动作必须产生一个非空观察；观察写回状态后，Router 才能决定下一步。只有追问、最终回答或回退会终止当前循环。

## 4. 暂停和恢复

当动作是 `ask_clarification` 且返回合法问题时：

```text
running → awaiting_user
```

Harness 此时不生成伪造的最终结果，而是返回可持久化会话。用户补充信息后：

```text
awaiting_user → 合并旧请求和新用户输入 → running
```

最终 `AgentScenarioRun` 会同时包含追问前和追问后的两个 Turn。

评测专用的 `BenchmarkSessionDriver` 单独持有第20步隐藏脚本。它只有在上一轮真实完成指定 `trigger_action` 后才会把下一条用户回复传给 `resume()`。隐藏脚本不存入会话，也不会传给 Router。

## 5. 代码强制执行的安全规则

| 风险 | Harness 的处理 |
|---|---|
| Router 选择未允许动作 | 记录 `rejected`，不调用 Executor，进入一次安全回退 |
| 同一请求重复相同工具和参数 | 根据 `request_id + tool_name + arguments_sha256` 拦截 |
| 超过最大步骤 | `max_steps_exceeded` |
| 超过工具、语义模型或 RAG 次数 | 在调用前拒绝 |
| 超过总 Token 或运行时间 | 记录已发生的调用，然后安全回退 |
| 工具异常、超时或非法结构 | 转成失败轨迹，整批任务继续 |
| 非终止动作没有新观察 | `no_state_progress`，防止空转 |
| Review RAG 未锁定商家 | 调用前拒绝 |
| Review 证据来自其他商家 | `evidence_business_mismatch` |
| 排名返回候选作用域外商家 | `business_id_out_of_scope` |
| 硬条件已经过滤掉的商家被重新加入 | 作用域校验拒绝 |
| 回退处理器自己失败或输出非法 | 使用最小静态回退，保证合法终止 |

系统回退只执行一次。当前默认回退是空的合法回退；第23步接入真实工具后，再提供读取 Hybrid V2 安全排名的 Adapter。本步没有伪装成“已经回退到 Hybrid V2”。

## 6. 轨迹和成本记录

每次运行自动产生第21步规定的：

- `AgentActionTrace`；
- `ToolCallTrace`；
- `ClarificationQuestionTrace`；
- `EvidenceReference`；
- `ResponseClaimTrace`；
- `AgentTurnTrace`；
- `AgentScenarioRun`。

工具轨迹记录工具名、种类、参数 SHA256、状态、延迟、Token、成本、缓存命中和检索证据。API Key、认证头和原始 Provider 对象没有进入任何状态或轨迹。

运行时任务判断允许如实记录 `unknown`。这只扩展了 Agent 的公开预测字段；第20步隐藏标准答案仍然限制为六种正式任务类型。

## 7. 文件职责

```text
src/yelp_agent/agent_harness/
├── schema.py             # AgentState、Decision、Outcome、Budget 等稳定类型
├── interfaces.py         # Interpreter、Policy、Router、Executor、Fallback 接口
├── interpreter.py        # 连接第18步和第19步
├── engine.py             # start/resume 与状态—动作循环
├── validation.py         # 预算、动作、证据、商家作用域安全规则
├── state_transition.py   # 集中更新不可变会话状态
├── trace_recorder.py     # 生成确定性的第21步轨迹
├── fallback.py           # 最小静态安全回退
├── session_driver.py     # 评测侧隐藏多轮回复释放器
├── fakes.py              # Fake Clock、Scripted Router/Executor/Fallback
├── config.py             # 严格 YAML 配置加载
└── __init__.py           # 公开接口

configs/agent_harness.yaml
tests/test_agent_harness.py
```

旧的 `src/yelp_agent/agent/` 和 `src/yelp_agent/rankers/agent_ranker.py` 属于 Agent V0 历史实验，本步没有删除或倒改。

## 8. 测试覆盖

本步的合成测试不读取 Yelp 大数据，也不调用外部 API，覆盖：

- 单步和多步骤正常完成；
- 追问、暂停、JSON 落盘、恢复；
- 非法动作、空动作集合和 Router/Executor 失败；
- 重复工具、最大步数、无状态进展；
- 工具超时、Token、成本和调用次数；
- Review 商家锁定和证据归属；
- 候选作用域与硬条件不可覆盖；
- 回退只执行一次；
- 隐藏回复按触发动作释放；
- AgentState 不含隐藏答案；
- 固定输入重复运行得到字节一致的 `AgentScenarioRun`。

复现命令：

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_agent_harness.py -q
```

## 9. 本步明确没有做什么

第22步只建立运行框架，没有实现：

- 第23步真实 Yelp 工具注册表；
- 第24步正式 Rule Router；
- Hybrid V2 真实回退 Adapter；
- Review RAG、Embedding 或 Cross-Encoder；
- 第二层 LLM 语义解析器；
- DeepSeek 调用或 LLM Router；
- 500 场景上的正式 Agent 效果对比。

因此，本步证明的是 Agent 可以被安全、确定、可恢复地运行和评测，不代表推荐效果已经提高。第23步负责给它真实工具，第24步才加入第一版真正的决策大脑。
