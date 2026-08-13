# 第 35 步：Constrained LLM Router V1

## 1. 本步解决什么问题

第 34 步已经能够把完整 Session 中仍然有效的条件编译成一份
`EffectiveSessionRequest`，但 Agent 的下一步动作仍由 `RuleRouter` 决定。
当规则把“换一家便宜点”识别成商家详情问题时，即使 Session 条件已经正确，
Agent 仍可能走错工具或根本不返回推荐。

本步在现有 Agent Harness 的 Router seam 上新增 `ConstrainedLLMRouter`：

```text
EffectiveSessionRequest + 当前可见状态
                ↓
代码生成完整、合法的候选决策
                ↓
DeepSeek 只选择一个 choice_id
                ↓
代码取回已经绑定好的动作、工具和参数
                ↓
Harness 再次检查动作、预算、范围和重复调用
                ↓
执行工具；失败时回退 Rule Router / 安全 Fallback
```

这里的 LLM 不是自由调用工具。它看不到可编辑的 `business_ids`、cutoff、
候选全集或工具参数，只能从代码提供的 `choice_id` 中选择。

## 2. 为什么没有直接让 LLM 输出工具参数

如果让模型输出：

```json
{
  "tool": "GET_BUSINESS_DETAILS",
  "business_ids": ["模型自己写的ID"]
}
```

模型可能编造商家、漏掉候选、把已拒绝商家放回来，或者绕过当前 business scope。
现在模型只允许输出：

```json
{
  "choice_id": "choice_fcb7bbe32635",
  "confidence": 0.90,
  "reason_code": "NEW_RECOMMENDATION_REQUIRED"
}
```

`choice_id` 背后的 `AgentDecision` 在调用模型以前已经由代码创建完成。

## 3. 模型能够真正决定什么

旧 `AllowedActionPolicy` 在多数状态只提供“唯一动作 + safe_fallback”。
如果继续使用这个集合，调用 LLM 只是昂贵地复读规则，没有 Agent 决策意义。

新 `ConstrainedDecisionBuilder` 会针对当前可见状态，分别检查六种受支持任务路径：

- recommendation_request；
- feedback_refinement；
- business_detail_question；
- candidate_comparison；
- review_experience_question；
- official_policy_question。

每条路径仍然由成熟的 `RuleRouter` 生成完整决策，然后由代码删除：

- 当前预算不允许的决策；
- 重复工具和相同参数；
- 没有锁定商家的 Review RAG；
- 商家 ID 为空或超出当前 scope；
- 比较对象不足两个；
- 没有候选却要求过滤或排序的决策。

DeepSeek 最终看到的是经过上述检查的多个安全选项。因此它可以把错误的
“详情问题”改成“继续推荐”，但不能发明第七种任务或不存在的工具。

## 4. 任务类型修正如何真正影响后续流程

只让模型选择另一种工具还不够。Terminal Executor、Review RAG 和后续 Router
都会读取当前任务类型。因此本步为 `AgentDecision` 增加代码受控的
`routed_task_type`。

决定通过 Harness 校验后，`apply_routed_task_type` 会同步更新：

- 当前 `DecisionReadiness.task_type`；
- 当前 Session Memory 的 `current_task_type`；
- 后续工具可见状态；
- Turn Trace 与 Case Explorer 中的最终任务类型。

这样“模型理解正确，但后面的执行器仍按旧任务回答”的上下文错位不会继续发生。

## 5. 模型输入与隐藏数据隔离

`RouterDecisionContext` 只包含：

- 最新用户消息；
- 完整 `EffectiveSessionRequest`；
- 当前任务类型和 information gaps；
- 候选、排序、详情和评论证据数量；
- 本轮已经执行的工具摘要；
- 剩余步骤、工具、RAG、语义调用和 Token 预算；
- 代码生成的安全候选选择。

模型输入不包含：

- Ground Truth；
- Benchmark acceptable/excluded 标签；
- 评测器正确动作；
- API Key 或认证 Header；
- 允许模型编辑的商家 ID 列表；
- 全量用户评论或全量候选详情。

## 6. 调用、校验和回退

配置冻结在 `configs/constrained_llm_router.yaml`：

- temperature = 0；
- thinking = disabled；
- JSON response format；
- 单次最大输出 300 Token；
- 超时 90 秒；
- 置信度下限 0.60；
- 非 JSON、Schema 错误或 choice 越界时最多修复一次；
- API 超时/连接失败不进行格式修复；
- 单一安全选项时直接执行，不调用模型；
- 失败后回退原 Rule Router。

模型输出依次接受以下验证：

1. 合法 JSON；
2. 严格 Pydantic Schema；
3. `choice_id` 必须来自当前候选；
4. 置信度达到门槛；
5. 候选决策未触发 Harness 重复调用、预算、Review 锁定或 scope 检查；
6. 工具执行结果仍需通过原 Harness outcome validator。

## 7. Trace 与 Token 记录

每个 `AgentActionTrace` 新增 `router_decision`：

- 输入任务类型和选择后的任务类型；
- 允许的 choice IDs；
- 模型提出和最终执行的 choice ID；
- 模型、置信度、状态和回退原因；
- Provider 是否真正调用；
- cache hit；
- 延迟和尝试次数；
- 输入、输出、总 Token；
- Sanitized call IDs。

`scripts/run_step35_constrained_llm_router.py` 会额外生成：

```text
router/router_metrics.json
router/router_summary.md
router_llm/llm_calls.jsonl
router_llm/llm_usage.json
memory_llm/...
answer_llm/...
total_llm_usage.json
```

`total_llm_usage.json` 分别记录 Router、Session Memory、语义/回答模型的消耗，
并给出本次实验已知的累计 Token。

## 8. Fake LLM 验证

Fake 测试覆盖：

- 合法模型选择；
- 单一选项免调用；
- 非 JSON；
- 外部 choice ID；
- 一次格式修复成功；
- 修复后仍非法；
- 低置信度；
- Provider 超时；
- no-LLM 模式；
- 代码参数不可被模型覆盖；
- business scope 隔离；
- 重复调用和预算过滤；
- Router Token 写入 Agent Session；
- 任务类型修正真正写回运行状态；
- Router 输入不包含 Ground Truth。

相关测试及全项目可运行回归均已通过。

## 9. 已完成的真实验证

### 9.1 合成状态 DeepSeek 冒烟

合成用户消息：

```text
第一家有点贵。请继续给我推荐，但要更便宜一些。
```

修复后的模型面对五个代码授权选项，选择：

```text
action      = retrieve_candidates
tool        = EXPAND_CANDIDATES
confidence  = 0.90
model       = deepseek-v4-flash
```

真实调用记录：

| 项目 | 结果 |
|---|---:|
| Provider 调用 | 1 |
| 输入 Token | 982 |
| 输出 Token | 29 |
| 总 Token | 1011 |
| Provider 延迟 | 2297.36 ms |
| Rule Router 回退 | 否 |

该冒烟使用合成 ID，没有发送 Benchmark Ground Truth 或真实 Yelp 用户上下文。
开发过程中共进行了两次合成 Router 冒烟，累计输入 / 输出 / 总 Token 为
1842 / 58 / 1900；第二次是加入任务类型写回后的当前实现结果。

### 9.2 无 API 的真实 Yelp 端到端装配测试

在不加载 `.env` 的情况下运行了一个真实 development 场景，验证 Yelp Parquet、
本地 Qwen Embedding、Qwen Reranker、Review RAG、Evidence Aggregator、Harness、
Trace 和报告写入可以完整运行。

结果：

- 场景数：1；
- Agent 动作数：3；
- 工具调用数：2；
- Router Provider 调用：0；
- API Token：0；
- Router 因 LLM disabled 回退 Rule Router：3 次；
- Agent Harness 安全 Fallback：0；
- Action Accuracy：100%（单例只证明装配正确，不代表总体效果）。

产物保存在 `runs/step35_no_api_smoke/`，运行产物默认被 Git 忽略。

## 10. 20 / 500 场景真实 API 实验状态

20 场景运行命令已经实现，但本次尝试被系统数据外发安全审查拦截。
原因是完整实验会把真实 Yelp Benchmark 中的用户查询、Session 最终请求和
候选状态摘要发送给外部 DeepSeek。合成冒烟已经获准并完成，真实 Yelp 内容
尚未发送。

继续运行需要用户对以下内容给出专项授权：

> 同意把 Yelp Benchmark 的用户查询、EffectiveSessionRequest 和候选状态摘要
> 发送给 DeepSeek，用于 20 个 development 场景和正式 500 场景 Router 实验。

获得该授权后执行顺序固定为：

1. development 20 场景；
2. 检查非法动作、任务修正、Fallback、Token 和延迟；
3. 不修改 validation 标签或评分规则；
4. 固定配置运行 development + validation 共 500 场景；
5. 与相同工具和排序模块下的 Rule Router 基线比较；
6. 把真实指标和错误案例追加到本文档。

## 11. 本步仍未解决的能力

Constrained LLM Router 解决的是“下一步做什么”和“当前属于什么任务”，
并不等于所有理解结果已经转化为候选分数。

仍需后续完成：

- 查询参考商家的真实价格、距离、安静程度；
- 把“比第一家更便宜/更近/更安静”变成候选级强特征或过滤条件；
- 在完整 500 场景实验后分析哪些错误来自 Router，哪些来自召回和排序；
- Cost-aware Router：简单状态优先规则，只有真正歧义时才调用 LLM；
- 后续 Learned Router / Agentic RL 只能在稳定 Trace 和可靠 Benchmark 上进行。
