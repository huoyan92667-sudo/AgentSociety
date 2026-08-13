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

## 10. 真实 API development 实验与 V1.1 修复

用户已明确授权把 Benchmark 用户查询、`EffectiveSessionRequest` 和候选状态摘要
发送给 DeepSeek。Router 输入仍不包含 Ground Truth、隐藏标签、完整评论正文、
API Key 或认证 Header。

第一版 20 场景实验暴露了两个工程问题：

1. 规则解析器误报 `ambiguous_reference` 后，代码只提供“追问”一个选项，LLM
   实际被绕过；
2. 规则漏掉显式条件冲突时，候选动作中没有“询问冲突”，LLM 即使理解冲突也
   无法选择正确动作。

V1.1 增加了代码受控的信息缺口修正，并把 Router 调用收紧为“每个用户回合开始
时至多调用一次”。模型确定本轮任务和是否需要追问后，召回、过滤、Embedding、
Cross-Encoder、语义融合、详情读取和最终返回继续由代码执行。这样不是取消 LLM
决策，而是避免在已经确定的工具链中重复付费。

同一组 20 个 development 场景修复前后如下：

| 指标 | V1 | V1.1 | 变化 |
|---|---:|---:|---:|
| 失败场景数 | 4 | 0 | -4 |
| Tool Selection Accuracy | 79.31% | 80.77% | +1.46 pp |
| Direct Return Precision | 52.38% | 65.52% | +13.14 pp |
| Missing-field Precision | 25.00% | 100.00% | +75.00 pp |
| Missing-field Recall | 33.33% | 66.67% | +33.34 pp |
| Unnecessary Question Rate | 75.00% | 0.00% | -75.00 pp |
| 平均场景延迟 | 15.28 s | 4.65 s | -10.63 s |
| Router Provider 调用 | 151 | 16 | -135 |
| Router Token | 171,449 | 18,271 | -153,178（-89.34%） |
| Router Rule Fallback | 4 | 0 | -4 |

V1.1 development 中共有 30 个逻辑模型决策，其中 14 个命中缓存、16 个真实调用；
其余 166 个确定性后续动作由单选项直接执行。

## 11. 正式 500 场景实验

### 11.1 实验范围、对照组和因果限制

正式实验覆盖 500 个场景，其中 development 400、validation 100；脚本最多还会
释放 250 个后续用户回合。Constrained 组和 Rule 组使用相同的：

- Query/History 多路召回；
- Hybrid V2 与 LightGBM 排序；
- 本地 Qwen3 Embedding 与 Qwen3 Reranker；
- Review RAG、Evidence Aggregator 和语义融合；
- Top-5 展示、Session Memory 和 Harness 预算。

目标上的核心差异是 Router。Rule 对照在 API Key 和 Base URL 被显式清空的情况下运行，
Memory/回答只允许读取已经存在的 DeepSeek 缓存，未命中时按规则安全回退；因此
对照组新增外部 Provider 调用和 Token 均为 0。这个对照避免再次外发未授权的新
Memory/回答上下文，但应准确称为 **Rule Router + cache-only Memory/Answer**。

它是当前授权边界内最接近同配置的可复现对照，但不是严格的单变量因果 A/B：不同
Router 会走出不同轨迹，进而形成不同的 Memory/回答缓存键。Constrained 组在这些新
轨迹上发生了 31 次 Memory 和 107 次回答 Provider 调用；Rule 组缓存未命中时不能
在线补齐。因此下表可以说明完整系统的实际差异，但不能把每一点变化全部归因给
Router 本身。严格单变量实验需要额外授权两组都在线调用，或预先冻结一套共享的
Memory/Answer 输出。

### 11.2 总体结果

| 指标 | Rule | Constrained LLM | 差值 |
|---|---:|---:|---:|
| Action Accuracy | 69.11% | 69.56% | +0.44 pp |
| Tool Selection Accuracy | 78.68% | 78.80% | +0.12 pp |
| Invalid Action Rate | 20.66% | 17.83% | -2.83 pp |
| Direct Return Precision | 52.66% | 65.06% | +12.40 pp |
| Missing-field Precision | 33.33% | 73.13% | +39.80 pp |
| Missing-field Recall | 48.75% | 61.25% | +12.50 pp |
| Unnecessary Question Rate | 66.67% | 26.87% | -39.80 pp |
| Question Answerability Rate | 33.33% | 73.13% | +39.80 pp |
| Fallback Rate | 0.20% | 0.00% | -0.20 pp |
| Business Scope Isolation | 92.63% | 96.09% | +3.46 pp |
| Citation Correctness | 76.16% | 78.23% | +2.07 pp |
| 平均场景延迟 | 2.69 s | 5.66 s | +2.97 s |
| P95 场景延迟 | 7.88 s | 14.77 s | +6.90 s |

Validation 100 场景没有用于修改提示词或阈值。其关键结果同样成立：Action Accuracy
从 70.00% 到 70.56%，Direct Return Precision 从 50.49% 到 61.94%，不必要追问率
从 63.64% 降到 16.67%，Missing-field Precision 从 36.36% 到 83.33%；平均延迟
从 2.90 秒增加到 5.84 秒。

### 11.3 多轮执行结果

500 个首轮场景之外共有 250 个隐藏后续回合。Rule 只释放 125 个，释放率 50.0%；
Constrained LLM 释放 231 个，释放率 92.4%。失败场景从 105 降到 35：

| 失败类别 | Rule | Constrained LLM |
|---|---:|---:|
| Multi-turn feedback | 60 | 0 |
| Information gap | 34 | 24 |
| Hard constraint | 6 | 6 |
| Candidate comparison | 5 | 5 |
| 合计 | 105 | 35 |

这说明 Constrained Router 的主要价值不是提高静态排序准确率，而是让 Agent 在多轮
反馈中继续执行正确流程，不再因为规则误判而提前追问或停住。

### 11.4 Router 调用与 Token

正式 500 场景实际产生 731 个用户回合和 4,033 个 Agent 动作：

- 模型逻辑决策 674 次；
- 真实 Provider 调用 644 次，缓存命中 30 次；
- 其余 3,359 个确定性动作单选项直接执行；
- 任务类型修正 26 次，information gap 修正 166 次；
- Router 非法输出、低置信度回退和 Rule Router 回退均为 0；
- Router 输入 / 输出 / 总 Token：759,960 / 20,065 / 780,025；
- Router 平均 Provider 延迟：1,020.53 ms；
- 连同 Memory 和回答生成，正式实验已知总 Token 为 931,287；另有 1 次回答调用
  无法取得 usage，因此真实值略高于该数字。

从 Step 35 开始至本次正式实验，所有成功写出台账的真实 Yelp 运行和两次合成冒烟
至少记录了 1,246,658 Token。一次中途异常的 4 场景运行只写入缓存、没有保存
Token 台账，因此项目总消耗只能报告“至少”，不能伪造精确值。

### 11.5 指标口径注意事项

`repeated_tool_call_rate` 在 Constrained 组显示为 13.79%，但旧评测器跨用户回合只用
“工具名 + 参数哈希”判断重复，没有包含本轮 request ID。多轮反馈合法地重新召回或
重新排序时会被误报。Harness 的真实防循环签名包含 request ID，因此没有执行同一
请求下的原地重复工具调用。该指标本轮不作为 Router 优劣结论。

`valid_candidate_rate` 约 1% 也不能解释为只有 1% 的推荐合法。旧指标把当前全目录
Query 召回结果与早期冻结的 20 个 `business_scope` 求交；两者不是同一个候选宇宙。
真正的运行时越界应看 `business_scope_isolation_rate` 和 Harness scope validator。

### 11.6 真实修复案例与剩余错误

场景 `025a729b...` 的用户明确要求“必须是酒吧，同时排除所有酒吧，并先澄清冲突”。
V1 直接召回并推荐；V1.1 选择 `ask_clarification(CONSTRAINT_CONFLICT)`，用户回答
“保留 Bars、去掉排除条件”后恢复召回和推荐。

场景 `01cf800f...`、`05d0378e...` 和 `0a7ccc3d...` 包含“更便宜、再近一点、不要
连锁店”等连续反馈。V1 会在中途把相对反馈误判成模糊指代；V1.1 均走完 4 轮，
每一轮先由模型确定任务，后续工具链由代码执行。

剩余 35 个失败中，典型问题是规则解析器先写入 `missing_party_size`，而 V1.1 只允许
模型安全覆盖“有上一轮推荐时的 ambiguous_reference”和显式冲突，其他缺口仍是
代码硬门。另有少数冻结标签本身可疑，例如用户已经说“每人不能超过 30 美元”，
Ground Truth 仍要求 `missing_budget`。这些案例必须在后续 Benchmark V3 审核中
单独修正，不能通过迎合错误标签来调 Router。

## 12. 本步仍未解决的能力

Constrained LLM Router 解决的是“下一步做什么”和“当前属于什么任务”，
并不等于所有理解结果已经转化为候选分数。

仍需后续完成：

- 查询参考商家的真实价格、距离、安静程度；
- 把“比第一家更便宜/更近/更安静”变成候选级强特征或过滤条件；
- Agent Benchmark 的动作正确性仍不能代替 Query Recommendation Benchmark 的
  最终商家相关性、HR@1/3/5/10 和 query-conditioned compliance；
- 需要把 Router 错误、语义解析错误、召回失败和排序失败进一步分层归因；
- Cost-aware Router：简单状态优先规则，只有真正歧义时才调用 LLM；
- 后续 Learned Router / Agentic RL 只能在稳定 Trace 和可靠 Benchmark 上进行。
