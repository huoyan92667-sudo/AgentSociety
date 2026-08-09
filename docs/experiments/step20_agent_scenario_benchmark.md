# 第 20 步：Agent Scenario Benchmark V1

## 1. 本步结果

第 20 步已经冻结一套 500 场景的 Agent 行为 Benchmark。它不是新的推荐模型，也不是 Agent 本体，而是后续 Router、工具、Review RAG、多轮状态和安全回退共同使用的固定“考卷 + 隐藏答案”。

```text
Agent 可见题目
    ↓
未来 AgentState / Router / Tools
    ↓
评测器读取隐藏动作、作用域、证据和不确定性政策
```

公开的深模块接口是：

```python
build_agent_benchmark(sources, config, output_root=...)
```

调用者不需要了解七类构造器、实体切分、证据选择、输出哈希和泄漏审计的内部实现。

## 2. 数据规模

| 场景类型 | 数量 |
|---|---:|
| A. 硬约束 `hard_constraint` | 100 |
| B. 当前请求与画像冲突 `profile_conflict` | 70 |
| C. 信息缺失与澄清 `information_gap` | 70 |
| D. 商家细节追问 `business_detail` | 80 |
| E. 候选比较 `candidate_comparison` | 60 |
| F. 多轮反馈 `multi_turn_feedback` | 60 |
| G. 证据不足与冲突 `evidence_uncertainty` | 60 |
| 合计 | 500 |

固定切分：

| Split | 场景数 |
|---|---:|
| Development | 400 |
| Validation | 100 |

语言分布为中文 300、英文 200。多轮脚本共有 250 个后续用户 Turn：70 个澄清回答，加上 60 个场景各 3 轮反馈。

## 3. 与第 18 步 500 Query 的关系

旧数据测试“自然语言能否解析成 `RecommendationRequest`”。新数据测试“听懂之后是否采取正确行动”。

本批 500 场景中有 100 个复用了旧 Query 的可见表达：

- 60 个硬约束场景；
- 40 个信息缺失场景。

这些记录保留旧 `case_id`、DeepSeek 模型名和 Prompt SHA256。其余 400 个场景由确定性模板和真实 Yelp 实体构造。旧 Query 的标准答案没有被当作商家相关性标签。

## 4. 可见题目与隐藏答案

Agent 只能加载：

```text
benchmarks/agent_scenarios_v1/visible/scenarios.jsonl
```

其中只有：用户、会话、cutoff、可见问句、可选位置和明确引用的商家 ID。

评测器单独加载：

```text
benchmarks/agent_scenarios_v1/hidden/ground_truth.jsonl
benchmarks/agent_scenarios_v1/hidden/evidence_labels.parquet
```

隐藏答案包含：

- 六种第 19 步任务类型；
- 预期信息缺口；
- 允许、必须和禁止的高层动作；
- 商家作用域和满足硬约束的商家；
- 脚本化用户回复及每轮状态更新；
- Review-ID 级证据相关性和支持/反驳方向；
- `proceed / clarify / caveat / conflict / official verification / abstain` 政策。

Agent 构造函数没有 Ground Truth 路径参数。

## 5. 七类场景如何落地

### A. 硬约束

保存结构化 `filter` 条件、20 个时点已存在商家的作用域，以及其中真正满足条件的商家 ID。当前请求先过滤，画像只能参与合格商家之间的排序。

### B. 画像冲突

从用户画像读取最强类别偏好，再选择不同类别作为当前强制请求。`current_request_overrides_profile=true` 是隐藏标准，不把长期画像当成本轮硬要求。

### C. 澄清

覆盖位置、预算、人数、指代和互相冲突的条件。初始允许动作只有追问/安全回退；脚本回答只有在 `ask_clarification` 后才允许释放。

### D. 商家追问

静态类别问题用 `business_attribute` 证据；体验问题用当前商家的 Review ID；“现在官方允许带宠物吗”要求外部官方验证，不能由历史 Yelp 评论直接断言。

### E. 候选比较

每题绑定两个商家和同一 Aspect，要求在相同维度检索两家的评论证据。相关证据必须位于两家作用域内。

### F. 多轮反馈

隐藏脚本固定为“太贵 → 更近 → 不要连锁店”，记录新增条件、拒绝过的商家和状态更新，供后续 Session Memory 评测。

### G. 不确定性

从真实 Aspect 事件中选择正负冲突、稀疏证据，另加必须官方确认的当前政策问题。标准行为是报告冲突、带警告回答或要求官方核验，不强行给确定答案。

“牛排偏肥还是偏瘦”没有重新加入通用 Aspect；此类细粒度问题仍应由后续 Business-scoped Review RAG 检索，证据不足时谨慎回答。

## 6. 证据标签

共冻结 545 条标签：

| 标签 | 数量 |
|---|---:|
| 相关 Review 证据 | 385 |
| 作用域外 Review 干扰项 | 130 |
| 静态商家字段证据 | 30 |

Review 标签只保存 `review_id / business_id / aspect / stance / time / hash`，不在 Git 中复制 Yelp 长评论。后续 RAG 可以用本地 `reviews.parquet` 按 Review ID 解析文本。

## 7. 泄漏与确定性审计

`audit_report.json` 已证明：

- 500 个 Scenario ID 和可见问句唯一；
- Visible 与 Hidden ID 一一对应；
- Development/Validation 用户不重叠；
- Development/Validation 核心商家不重叠；
- Development/Validation 模板族不重叠；
- 所有 Review 证据严格早于 cutoff；
- 相关证据在商家作用域内，干扰项在作用域外；
- 没有 Test 来源任务；
- 可见文件不含隐藏字段；
- 输入、配置和输出均保存 SHA256；
- 重复执行会验证并复用同一冻结产物。

LLM 不决定 Hidden Intent、动作、商家范围、证据标签或评分。当前构建没有发起真实 API 请求。

## 8. 语言改写接口

模块提供三个 Adapter：

- `DeterministicScenarioRewriter`：冻结默认；
- `FakeScenarioRewriter`：无网络测试；
- `OpenAICompatibleScenarioRewriter`：显式注入的预留接口。

Provider Adapter 的接口只接收 `language + visible_query`，不接收 Scenario Ground Truth、证据或用户标签。默认 CLI 遇到 `openai_compatible` 配置会拒绝隐式调用，真实 DeepSeek 仍需单独授权和显式注入。

## 9. 第 19 步基线体检

当前规则解析器和 `DecisionReadinessAnalyzer` 在 500 场景上的结果：

| 指标 | 结果 |
|---|---:|
| Task Type Accuracy | 73.40% |
| Information Gap Exact Match | 90.80% |
| Information Gap Precision | 76.79% |
| Information Gap Recall | 53.75% |
| Information Gap F1 | 63.24% |
| 正确拒绝伪造 Query-aware 置信度 | 74.14% |

类别任务识别：

| 类别 | Accuracy |
|---|---:|
| 商家细节 | 100% |
| 候选比较 | 11.67% |
| 证据不确定性 | 100% |
| 硬约束 | 87.00% |
| 信息缺口 | 82.86% |
| 多轮初始请求 | 100% |
| 画像冲突 | 21.43% |

这不是最终 Agent 分数。它清楚暴露了现有规则对自然表达的候选比较和画像冲突识别较弱，也解释了为什么后续需要语义解析器与受控 Router。正式指标在第 21 步冻结。

## 10. 复现

```powershell
.\.venv\Scripts\python.exe scripts\build_agent_benchmark.py
.\.venv\Scripts\python.exe scripts\evaluate_decision_readiness_benchmark.py
```

核心代码：

```text
src/yelp_agent/agent_benchmark/schema.py
src/yelp_agent/agent_benchmark/sources.py
src/yelp_agent/agent_benchmark/builder.py
src/yelp_agent/agent_benchmark/audit.py
src/yelp_agent/agent_benchmark/artifacts.py
src/yelp_agent/agent_benchmark/rewriting.py
src/yelp_agent/agent_benchmark/baseline.py
```

## 11. 本步边界

- 没有实现正式 Agent Harness；
- 没有实现 Router 或工具注册表；
- 没有实现 Review RAG；
- 没有把 DeepSeek 当标签员；
- 没有新增 Test 或 Final Holdout；
- 没有因为基线失败而修改 Validation 标准答案；
- 第 21 步才冻结正式评测指标。
