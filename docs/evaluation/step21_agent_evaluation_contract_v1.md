# 第 21 步：Agent 完整评测契约 V1

## 1. 本步结论

第 21 步没有训练模型、没有实现 Agent Harness，也没有调用 DeepSeek。它先把后续所有 Rule Router、语义模型、Review RAG 和 LLM Router 共同使用的“答题格式 + 阅卷规则”冻结下来。

```text
第 20 步：500 道可见题目 + 隔离的隐藏答案
                     ↓
未来 Agent：只写标准 AgentScenarioRun
                     ↓
第 21 步评测器：匹配隐藏答案、判断适用指标、计算并分桶汇总
```

评测契约版本为 `1.0.0`，共固定 46 项指标：43 项由 Agent Scenario Benchmark 计算，3 项全库召回指标必须来自第 11 步 Full Retrieval Benchmark。

冻结产物：

```text
benchmarks/agent_scenarios_v1/evaluation_contract.json
```

它保存了：

- 指标定义 SHA256；
- 配置 SHA256；
- 第 20 步 visible、ground truth、evidence 和 manifest 的 SHA256；
- Agent 场景指标与 Full Retrieval 指标的来源边界；
- `hidden_labels_visible_to_agent=false`；
- `full_retrieval_inferred_from_agent_scenarios=false`。

## 2. 深模块接口

未来 Agent 只需要学习一个输出接口：

```python
AgentScenarioRun(
    scenario_id,
    agent_version,
    turns,
    fallback,
    fallback_reason,
    latency_ms,
    input_tokens,
    output_tokens,
    cost_usd,
)
```

每个 `AgentTurnTrace` 记录：

- 预测的任务类型；
- 发现的信息缺口；
- 按顺序选择的高层动作；
- 结构化工具调用；
- 澄清问题及其要解决的字段；
- 完整候选排序 `candidate_ranking`；
- 真正展示给用户的 `recommended_business_ids`；
- 检索到的 Review/商家字段证据；
- 用户可见事实声明及明确引用；
- 是否报告评论日期、证据冲突和官方核验提醒。

`recommended_business_ids` 必须是 `candidate_ranking` 的子集。运行记录不接受隐藏任务类型、标准答案、允许动作或证据标签等额外字段。

统一评测接口为：

```python
evaluate_agent_scenario_runs(
    runs,
    visible_scenarios=...,
    ground_truth=...,
    evidence_labels=...,
)
```

调用方不需要自己判断哪个场景算哪些指标，也不能自行改变分母。评测器内部负责总体、Development/Validation 和七类场景分桶。

## 3. 为什么排序与最终推荐要分开

```text
candidate_ranking
    = Agent 对合法候选的完整排序

recommended_business_ids
    = 最终真正展示给用户的少量商家
```

HR、MRR、NDCG 使用候选排序；硬约束满足率与空结果率使用最终展示列表。这样不会出现“候选池中有一个不满足条件的商家，就误判 Agent 最终推荐违规”的问题。

## 4. 指标适用范围

| 场景能力 | 主要指标 |
|---|---|
| 请求理解 | Task Type Accuracy、Missing-field Precision/Recall |
| 推荐排序 | HR@1/3/5、MRR、NDCG@5 |
| 全库召回 | Recall@50/100/500，来源固定为第 11 步 |
| 硬约束 | Hard Constraint Satisfaction、Valid Candidate Rate、Empty Result Rate |
| 澄清 | 不必要问题率、问题可回答率、澄清后效用增益、结束前平均问题数 |
| 路由 | Action Accuracy、工具选择、无效动作、重复调用、直接返回、回退 |
| Review RAG | 商家作用域隔离、Review Recall@K、Evidence Precision@K |
| 回答可信度 | Grounded Answer、Unsupported Claim、Citation Correctness、日期/冲突/官方提醒 |
| 成本 | Mean/P50/P95 延迟、Mean/P95 Token、成功推荐成本、缓存命中率 |

不适合某个场景的指标保存为 `not_applicable`，不会记成 0。例如“这家能带宠物吗”没有推荐排名，因此不计算 HR。

理论上需要计算、但当前数据源不具备的指标保存为 `unavailable`。两种状态不能混用。

## 5. 排序与全库召回边界

第 20 步场景保存的是：

```text
business_scope
acceptable_business_ids
```

`acceptable_business_ids` 表示满足该场景已标注条件的商家，不是 Yelp 用户真实下一次访问的 target。因此：

- HR/MRR/NDCG 在 Agent 场景中表示“合格商家是否被排到前面”；
- 不能将它解释成真实 next-business 预测准确率；
- Recall@50/100/500 必须读取第 11 步 Full Retrieval Benchmark 的 held-out target；
- 纯 Agent 场景报告中的这三项固定返回 `unavailable`，原因是 `requires_step11_full_retrieval_benchmark`。

这继续遵守 Controlled Reranking、Full Retrieval 和 Agent Scenario 三类评测必须分开报告的原则。

## 6. 主要公式与聚合方法

### 排序

- `HR@K`：前 K 名是否至少包含一个 acceptable business；按适用场景宏平均。
- `MRR`：第一个 acceptable business 的倒数排名；按场景宏平均。
- `NDCG@5`：acceptable businesses 作为二元相关标签；按场景宏平均。

### 信息缺口

跨全部已观察用户 Turn 做微平均：

```text
Precision = 正确发现的缺口 / Agent 发现的全部缺口
Recall    = 正确发现的缺口 / 隐藏答案中的全部缺口
```

### 动作与工具

- `Action Accuracy`：完成的隐藏 required actions / required actions；
- `Invalid Action Rate`：不在 allowed actions 或命中 forbidden actions 的选择 / 全部选择；
- `Repeated Tool Call Rate`：重复 `tool_name + arguments_sha256` / 全部工具调用；
- `Direct-return Precision`：没有未解决缺口且已完成前置动作的直接返回 / 全部直接返回。

### 澄清后效用

```text
Post-clarification Utility Gain
= 澄清后 NDCG@5 - 澄清前 NDCG@5
```

只有同时具有澄清行为、前后排序和 acceptable labels 的场景才能计算。当前第 20 步信息缺口场景没有 Query × 商家相关性标签，因此正式报告会如实显示 `not_applicable`，而不会用 Agent 自报分数代替。

### Review 证据

- 检索必须保持 `business_id` 作用域；
- Review Recall@K 使用隐藏的 relevant Review ID；
- Evidence Precision@K 同时支持 Review 和静态商家字段；
- 每条用户可见事实声明必须显式携带 evidence refs；
- 引用了 out-of-scope 或 irrelevant 标签不算正确引用；
- 没有相关证据却给出事实结论，计入 Unsupported Claim。

Review 标签来自第 13 步高置信度规则抽取，属于可追溯银标，不宣称为人工金标。

### 成本

- P50/P95 使用线性插值；
- Token 为 input + output；
- 成本信息缺失时不默认为 0，而是 `unavailable`；
- 成功推荐要求结果非空、在合法商家作用域内，并满足适用的 acceptable/hard-constraint 标准。

## 7. 多轮安全规则

隐藏的 Scripted User Turn 只有在上一轮完成指定 `trigger_action` 后才能释放。评测器会记录：

```text
unexpected_user_turn
scripted_turn_released_without_trigger
missing_scripted_turn_after_trigger
```

这防止未来 Agent Harness 偷看后续用户答案，或在没有追问时直接获得澄清信息。

## 8. 文件、复现与当前边界

核心代码：

```text
src/yelp_agent/agent_evaluation/schema.py
src/yelp_agent/agent_evaluation/definitions.py
src/yelp_agent/agent_evaluation/evaluator.py
src/yelp_agent/agent_evaluation/artifacts.py
src/yelp_agent/agent_evaluation/contract.py
configs/agent_evaluation.yaml
```

冻结或验证契约：

```powershell
.\.venv\Scripts\python.exe scripts\freeze_agent_evaluation_contract.py
```

第 22 步 Agent Harness 生成 `runs.jsonl` 后，统一评测命令为：

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_agent_scenario_runs.py `
  --runs runs/agent_scenarios_v1/runs.jsonl
```

当前明确没有：

- 真实 Agent Scenario 运行结果；
- Rule Router；
- Review RAG；
- 第二层 LLM 语义解析器；
- DeepSeek API 调用；
- 用 Validation 结果修改评分公式。

下一步是第 22 步：实现通用受控 Agent Harness，并让 Fake/Rule Agent 首先写出本契约要求的 `AgentScenarioRun`。
