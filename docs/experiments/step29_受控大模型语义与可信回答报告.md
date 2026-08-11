# 第 29 步：受控大模型语义理解与可信回答

## 1. 完成范围

第 29 步在第 28 步 Rule Agent、评论 RAG 和 Evidence Aggregator 之上加入两个受控的大模型能力：

```text
能力 A：规则解析结果
→ 门控判断是否值得调用 LLM
→ LLM 提议语义条件和任务类型
→ 代码验证原文证据、字段、运算符和置信度
→ 规则优先合并

能力 B：Evidence Aggregator 的结构化 Claim
→ 为每条 Claim 分配 E1/E2…证据编号
→ LLM 只负责自然语言组织
→ 代码检查商家作用域、证据编号、引用和不确定性
→ 合法则返回；非法则保留确定性回答
```

本步骤没有让 LLM 自由选择工具，也没有让 LLM 直接生成商家 ID 或修改候选排名。工具决策仍由第 24 步 Rule Router 控制；语义证据如何在 Rank Protection 下改变排序属于第 30 步；受约束 LLM Router 属于后续步骤。

## 2. 为什么是两个独立模块

### 2.1 `ControlledSemanticEnhancer`

输入是规则解析器已经得到的 `RecommendationRequest` 和 `DecisionReadiness`，输出仍是相同的稳定类型。它只在以下情况调用 DeepSeek：

- 规则无法可靠识别任务；
- Query 含复杂比较、转折或反馈表达；
- 任务属于候选比较、评论体验或反馈修改；
- 规则没有提取到任何可用条件；
- 存在指代不清等高风险语义。

普通、明确的问题直接跳过。冻结 Query Benchmark 中，Development 触发 106/400，Validation 触发 21/100。

### 2.2 `GroundedAnswerComposer`

输入只包含第 28 步已经生成的 Claim、允许的商家 ID 和引用。LLM 输出的每个句子必须绑定至少一个 `E*` 证据编号。代码禁止：

- 引入候选外商家；
- 引入不存在的 Review 或结构化字段；
- 把评论说成官方政策；
- 在冲突或低置信情况下删除谨慎措辞；
- 使用“保证、肯定、绝对”等无证据强断言；
- 为非比较任务混用多家商家的证据。

输出不合法、网络失败、JSON 错误或无 API 配置时，完整保留第 28 步的确定性 Claim。

## 3. Rule-first 合并安全规则

LLM 返回的是“建议”，不是可直接执行的请求。只有满足以下条件才会进入正式 Request：

1. `evidence_span` 必须逐字出现在用户原句中；
2. 置信度必须达到 Development 冻结阈值 0.78；
3. 字段和运算符必须属于白名单；
4. 规则已经识别的类别、距离、预算和价格字段优先；
5. 没有明确硬性词时，LLM 不能把普通偏好升级为 mandatory；
6. 模型单独给出的 `missing_fields` 和 ambiguity 没有原文跨度，不能制造阻塞性追问；
7. 距离等已接受条件仍可由确定性代码推导 `missing_location`。

第 6 条来自 Development 中发现的真实失败：第一版直接采用 LLM 的 missing-field 标签，不必要追问率从规则基线的 31.11% 升到 74.62%。收紧后降到 43.10%，同时 Development Action Accuracy 升到 79.03%。该修正完成后生成冻结策略文件，才第一次运行 Validation。

冻结文件：`configs/controlled_llm_policy.json`。

冻结 SHA256：

```text
0bcf3743b13ab36c62d9cca2fe0b46ceecc00de966755eaccac75ab276575a98
```

## 4. 严格输出、缓存和调用台账

两个能力都通过同一个 `ControlledJSONCaller`：

- OpenAI-compatible DeepSeek 接口；
- `temperature=0`；
- `thinking=disabled`；
- 语义输出上限 900 Token，回答输出上限 1200 Token；
- Pydantic 严格 JSON Schema；
- Prompt 版本和输入 SHA256；
- SQLite 成功结果缓存；
- 每次逻辑调用记录 capability、状态、缓存命中、延迟、输入/输出 Token 和 provider request ID；
- Trace 不保存 API Key、认证 Header 或完整 Prompt。

缓存键包含模型、能力、Prompt 版本和 Prompt SHA256。只有通过结构 Schema 的 Provider 输出会缓存；每次重放仍执行本地安全策略校验。

## 5. Query 语义解析 Benchmark

### 5.1 Development 400

阈值只在 Development 比较 0.72、0.78、0.82、0.86 和 0.90。0.72 与 0.78 的主要指标相同，按照更保守阈值的并列规则选择 0.78。

| 指标 | Rule Parser | Step 29 | 差值 |
|---|---:|---:|---:|
| Condition F1 | 46.47% | 51.24% | +4.76pp |
| Exact Match | 14.00% | 23.50% | +9.50pp |
| Precision | 63.26% | 61.36% | -1.90pp |
| Recall | 36.73% | 43.98% | +7.26pp |
| Missing-fields Exact | 60.50% | 74.50% | +14.00pp |

### 5.2 冻结 Validation 100

| 指标 | Rule Parser | Step 29 | 差值 |
|---|---:|---:|---:|
| Condition F1 | 42.06% | 46.09% | +4.03pp |
| Exact Match | 11.00% | 15.00% | +4.00pp |
| Precision | 55.68% | 55.02% | -0.66pp |
| Recall | 33.79% | 39.66% | +5.86pp |
| Missing-fields Exact | 66.00% | 76.00% | +10.00pp |
| Party-size Accuracy | 88.00% | 89.00% | +1.00pp |

Validation 只报告一次，没有用于改规则。21/100 个 Query 触发 Provider，消耗 17,501 Token。

## 6. 正式 500 Agent 场景结果

完整结果由冻结 Development 400 和一次性 Validation 100 合并得到。

| 指标 | Step 28 Rule Agent | Step 29 Full | 差值 |
|---|---:|---:|---:|
| Action Accuracy | 70.89% | 79.00% | +8.11pp |
| Tool Selection Accuracy | 77.48% | 78.10% | +0.62pp |
| Direct Return Precision | 63.21% | 75.81% | +12.60pp |
| Missing-field Precision | 71.67% | 64.47% | -7.19pp |
| Missing-field Recall | 53.75% | 61.25% | +7.50pp |
| Unnecessary Question Rate | 27.78% | 38.57% | +10.79pp |
| Grounded Answer Rate | 75.50% | 91.41% | +15.91pp |
| Citation Correctness | 64.91% | 73.68% | +8.77pp |
| Unsupported Claim Rate | 27.17% | 10.10% | -17.08pp |
| Conflict Detection Accuracy | 93.33% | 88.33% | -5.00pp |
| Official-policy Caution | 100.00% | 100.00% | 0.00pp |
| HR@1 | 5.22% | 5.65% | +0.43pp |
| HR@3 | 8.70% | 10.87% | +2.17pp |
| HR@5 | 10.00% | 14.78% | +4.78pp |
| MRR | 8.31% | 10.27% | +1.96pp |
| Fallback Rate | 0.20% | 0.40% | +0.20pp |
| Mean Latency | 2185.08 ms | 3076.36 ms | +891.28 ms |
| P95 Latency | 6714.41 ms | 9236.43 ms | +2522.02 ms |

Recommendation 指标只在 230 个推荐型场景上计算。Step 29 没有实现新的排序器；HR 提升主要来自更正确的任务理解、少走错误分支和更多有效任务能够完成，不应表述为“LLM 直接重排提升”。

## 7. 冻结 Validation Agent 结果

Validation 100 场景相对 Step 28：

- Action Accuracy：70.00% → 78.89%；
- Direct Return Precision：61.19% → 75.19%；
- Missing-field Recall：57.14% → 71.43%；
- HR@5：8.70% → 17.39%，分母为 46 个推荐任务；
- Grounded Answer Rate：72.50% → 87.50%；
- Unsupported Claim Rate：29.73% → 16.39%；
- Unnecessary Question Rate：11.11% → 16.67%；
- Conflict Detection Accuracy：94.44% → 91.67%；
- Mean Latency：2325.64 ms → 4767.81 ms。

正向结果通过了冻结 Validation，但延迟、追问和冲突指标的退化同样保留，没有用 Validation 回调规则。

## 8. 四组消融实验

| 变体 | Action Accuracy | Grounded Answer | Citation | Unsupported Claim | HR@5 | Mean Latency |
|---|---:|---:|---:|---:|---:|---:|
| Step 28 Rule | 70.89% | 75.50% | 64.91% | 27.17% | 10.00% | 2185.1 ms |
| Semantic-only | 79.00% | 89.90% | 74.56% | 10.70% | 14.78% | 2174.3 ms |
| Answer-only | 70.89% | 76.00% | 66.97% | 23.11% | 10.00% | 2315.8 ms |
| Full Step 29 | 79.00% | 91.41% | 73.68% | 10.10% | 14.78% | 3076.4 ms |

结论：

1. Action 和 HR 的提升来自 Semantic Enhancer，而不是回答润色；
2. Answer-only 不改变动作或排名，但 Citation +2.07pp、Unsupported Claim -4.07pp；
3. Full 相比 Semantic-only 的 Grounded Answer +1.51pp、Unsupported Claim -0.60pp；
4. Full 的 Citation 比 Semantic-only 低 0.88pp，说明自然语言合并后，隐藏银标签对引用的覆盖并不总是更好；
5. 不能宣称“加入回答 LLM 后所有指标都提升”。

Answer-only 使用终止阶段重放：沿用 Step 28 已冻结动作、工具、候选、排名和 Claim，只替换 160 个 grounded/uncertain 终止回答。隐藏脚本仅由实验 Driver 用来重建已经按触发器释放给 Agent 的当轮文本，不进入 Router、工具或运行状态。

## 9. 一个可读案例

用户问题：

```text
Maggiano's Little Italy 和 Comfort & Floyd 哪个在服务方面评价更好？
```

Step 28 只执行商家详情和确定性比较。Step 29 仍由受控 Router 决策，但语义模块使它能够执行：

```text
GET_BUSINESS_DETAILS
→ SEARCH_BUSINESS_REVIEWS
→ AGGREGATE_REVIEW_EVIDENCE
→ COMPARE_BUSINESSES
→ RETURN_GROUNDED_ANSWER
```

最终回答由三条带 Review 引用的 Claim 组成：一条说明 Maggiano's 的晚餐服务正面证据，一条说明 Comfort & Floyd 也有正面但较少且较旧的证据，最后明确提示两边证据置信度和时间不同，不能过度确定地比较。

## 10. Token、失败和延迟

最终 Full 500 产物中：

- 逻辑模型步骤：876；
- 语义逻辑步骤：721，包含门控跳过；
- 回答逻辑步骤：155；
- 真实 Provider Calls：221；
- Cache Hits：475；
- 已知 Token：198,698；
- 结构/策略失败：8 个逻辑步骤；
- Agent Fallback：2/500；
- 成本金额：未配置模型单价，因此不伪造人民币或美元成本。

本步骤从小样本调试到全部正式/消融实验累计持久化记录：

- Provider Calls：713；
- 已知 Token：581,125；
- 另有 1 次早期超时调用没有返回 usage；
- 因此 581,125 是已知下界；
- 合并 Development/Validation 的视图文件不重复计入累计 Token。

最终 Full 的 Token 是该次正式输出真实发给 Provider 的用量。由于 Development 部分命中过调试阶段缓存，它不是“全冷缓存 500 场景成本”；累计值更接近本步骤实际消耗，但其中包含开发试错和消融。

## 11. 当前限制

- 不必要追问率仍比 Rule Agent 高 10.79pp；
- Missing-field Precision 下降 7.19pp；
- Conflict Detection Accuracy 下降 5pp；
- P95 延迟增加约 2.52 秒；
- 语义模型提高 Recall，但 Precision 略有下降；
- Answer Composer 对 Citation 并非稳定增益；
- 当前调用的是外部 DeepSeek，不是本地 LLM；
- LLM 仍不拥有工具选择权，也没有成本感知动态规划；
- 第 30 步之前，查询语义不能绕过 Rank Protection 自由改动推荐顺序。

## 12. 测试覆盖

新增测试覆盖：

- 逐字证据跨度；
- 规则字段优先；
- 普通偏好不能升级为硬条件；
- 无证据 missing field 不能制造追问；
- 非 JSON 和严格 Schema 失败回退；
- 未知商家和未知证据编号拒绝；
- 强断言拒绝；
- 候选比较允许引用两家作用域内商家；
- 缓存重放不调用 Provider；
- 无 API Key 时保持规则结果且 Token 为 0；
- Terminal 模型用量与 Tool Call 分开记账；
- 原有 Harness、Router、RAG 和评测测试不回归。

## 13. 复现命令

```powershell
# Query Development / Validation
.\.venv\Scripts\python.exe scripts\evaluate_step29_query_parser.py `
  --split development --output-root runs\controlled_llm_v1\query\development_gap_safe

.\.venv\Scripts\python.exe scripts\evaluate_step29_query_parser.py `
  --split validation --output-root runs\controlled_llm_v1\query\validation

# Full Agent split runs
.\.venv\Scripts\python.exe scripts\run_step29_agent_benchmark.py `
  --split development `
  --output-root runs\controlled_llm_v1\agent\development_gap_safe `
  --embedding-model-path D:\models\Qwen3-Embedding-0.6B `
  --cross-encoder-model-path D:\model\Qwen3-Reranker-0.6B `
  --model-python D:\anaconda3\python.exe --device cuda

.\.venv\Scripts\python.exe scripts\run_step29_agent_benchmark.py `
  --split validation `
  --output-root runs\controlled_llm_v1\agent\validation `
  --embedding-model-path D:\models\Qwen3-Embedding-0.6B `
  --cross-encoder-model-path D:\model\Qwen3-Reranker-0.6B `
  --model-python D:\anaconda3\python.exe --device cuda

# 冻结、合并和用量
.\.venv\Scripts\python.exe scripts\freeze_step29_policy.py `
  --development-agent-root runs\controlled_llm_v1\agent\development_gap_safe

.\.venv\Scripts\python.exe scripts\merge_step29_agent_benchmark.py `
  --development-root runs\controlled_llm_v1\agent\development_gap_safe `
  --validation-root runs\controlled_llm_v1\agent\validation `
  --output-root runs\controlled_llm_v1\agent\full

.\.venv\Scripts\python.exe scripts\summarize_step29_usage.py
```

## 14. 当前准确定位

第 29 步完成后，Agent 第一次具备了有真实评测收益的受控 LLM 语义能力：它能在复杂比较、评论体验、硬条件和反馈场景中改善任务理解，并把第 28 步证据写成更自然、可引用、失败可回退的回答。

它仍不是 LLM 自主 Router。当前最诚实的定位是：

```text
Rule Router + 受控语义副驾驶 + 证据约束回答器
```

下一步应在第 30 步实现保守融合与 Rank Protection，明确哪些 Query/Aspect 语义分数只能小范围调整 Top-N，哪些硬约束绝不允许被 LLM 覆盖。
