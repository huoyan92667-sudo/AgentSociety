# 第 36 步：Query-aware 排序工具接入 Agent 与 500 条在线实验

## 1. 本步解决的问题

第 33 步已经在 Query Recommendation Benchmark V1 上完成 Query-aware 排序链路，并在 500 条样本上取得 16.80% 的 HR@5。该结果证明，当前 Query 必须在候选保护、粗排和本地精排阶段持续参与计算。然而，旧版完整 Agent 并未调用第 33 步的最终排序入口，而是继续按照多次召回、过滤、Embedding、Cross-Encoder 和详情读取的旧工具链逐步执行。相同 500 条样本进入旧 Agent 后，HR@5 仅为 5.80%，平均工具调用达到 8.164 次，平均延迟为 12.108 秒。

因此，静态排序结果与 Agent 结果之间的差距并不首先说明 Hybrid V2 或 Query-aware Ranker 失效，而是说明第 33 步的能力没有成为 Agent 可调用的完整工具。第 36 步围绕这一接口缺口进行改进，将冻结的 Query-aware 排序流程封装为 `GET_QUERY_AWARE_RANKING`，接入 Rule Router、Constrained LLM Router、Agent Harness 和独立的 Query Recommendation Agent 评测器，并通过 1、10、50、500 条分级实验验证在线调用、排序效果、延迟和 Token 口径。

本步的研究问题可以概括为：在不修改第 33 步冻结排序策略的前提下，完整 Agent 能否直接复用静态 Query-aware 排序能力，并在保留安全路由、会话状态和工具审计的同时，使最终推荐指标恢复到静态管线附近。

## 2. 总体改进方案

改进后的首轮推荐流程如下：

```text
当前 Query + EffectiveSessionRequest
                ↓
Constrained LLM Router 在代码授权选项中选择
                ↓
GET_QUERY_AWARE_RANKING
                ↓
History Top-500 + Query Top-500 受保护并集
                ↓
确定性硬约束与 Session rejected_business_ids 排除
                ↓
冻结 Hybrid V2-B LambdaMART 分数
        + Query 召回与 Embedding 分数
                ↓
Query-aware 粗排
                ↓
Top-30 本地 Qwen3 Reranker 精排
                ↓
内部 Top-10 / 展示 Top-5
                ↓
读取 Top-5 商家详情并返回推荐
```

该设计没有让 LLM 直接生成商家 ID、排序分数或工具参数。DeepSeek 仍然只能在代码生成的合法 `choice_id` 中选择；具体的候选保护、硬约束、排序、Top-30 精排和 Top-5 展示全部由冻结代码执行。这样既保留了 Agent 对任务类型和下一步动作的理解能力，又避免模型改写排序结果、越过 business scope 或恢复已被用户拒绝的商家。

## 3. 将第 33 步封装为完整 Agent 工具

### 3.1 在线排序运行时

原第 33 步运行时以 Benchmark case 为入口，负责离线准备和正式评测。为了接入真实 Agent，本步增加 `OnlineQueryAwareRankingRuntime`，将准备阶段与最终精排阶段组合为一个长期持有本地模型和数据资源的运行时。该运行时直接接收已经编译完成的 `RecommendationRequest`，而不是重新从 Benchmark 标签或 target 构造输入。

在线入口继续使用第 33 步冻结的配置与策略，依次执行历史召回、Query 召回、受保护并集、硬约束过滤、LambdaMART 与 Query 信号融合以及本地 Cross-Encoder 精排。新增的 `rejected_business_ids` 参数在候选池阶段生成 `SESSION_REJECTED` 排除原因，从而保证用户明确否定过的商家不能被后续语义排序重新放回。

### 3.2 `GET_QUERY_AWARE_RANKING` 工具适配器

本步在 Agent Tool Registry 中新增 `GET_QUERY_AWARE_RANKING`。工具适配器负责把 Agent State 中的有效请求、场景 ID、数据 split 和 Session 拒绝列表转换为在线排序运行时输入，并把以下结果写回工具观察：

- 完整候选排列和内部 Top-10；
- 展示 Top-5；
- Query 权重和语义分数；
- 硬约束排除记录；
- Embedding 与 Cross-Encoder Token；
- 缓存命中、缓存未命中和本地模型调用信息；
- 延迟、回退状态和失败原因。

工具输出仍需通过 Harness schema 和 scope validator。Agent 不能向工具传入 target，模型也不能编辑候选 ID。

### 3.3 Router 与运行时装配

Rule Router 和 Constrained LLM Router 现在都能识别 Query-aware 工具模式。推荐任务尚未产生完整排序时，首步决策为：

```text
action      = retrieve_candidates
tool        = GET_QUERY_AWARE_RANKING
reason_code = QUERY_AWARE_RANKING_REQUIRED
```

排序完成后，如果 Top-5 商家详情尚未读取，Router 执行 `GET_BUSINESS_DETAILS`；详情齐备后执行 `return_recommendation`。后两步只有一个合法业务动作，因此直接使用 `single_choice_bypass`，不再重复调用 DeepSeek。预算同步放宽到能够容纳完整三步流程，但没有取消最大步骤数、最大工具调用数和最大语义调用数限制。

正式 500 条实验中的动作分布验证了该装配逻辑：495 条推荐场景分别执行一次 `retrieve_candidates`、一次 `get_business_details` 和一次 `return_recommendation`；5 条信息缺口场景执行 `ask_clarification`。推荐场景平均工具调用由旧 Agent 的 8.164 次降至 2 次左右。

## 4. 会话请求与跨平台复现修复

### 4.1 保留当前 Query 的原始语义

Agent 接入后曾出现排序指标下降。排查发现，`EffectiveSessionRequest` 虽然保存了结构化条件，但单轮请求也会被重写为模板式摘要，导致菜系别名、氛围、场景和自然语言软偏好不能完整进入 Query Embedding 与 Cross-Encoder。

本步调整 `compile_effective_request`：单轮首版请求直接保留最新用户 Query；多轮请求则把最新原话放在重建摘要第一行，并将语义文档限制在 2000 个字符以内。与此同时，`latest_user_query` 被加入 `effective_request_id` 的哈希输入，避免不同 Query 因结构化条件相同而错误复用 Session Memory 或排序缓存。

### 4.2 Session 拒绝商家进入硬约束

此前 Session 中保存的 `rejected_business_ids` 没有进入第 33 步候选池。本步将它们作为附加排除集合传给 `ProtectedCandidatePool`，并写入 `SESSION_REJECTED` 原因码。排除发生在粗排和精排之前，因此本地模型不能重新引入已拒绝商家。

### 4.3 Benchmark 哈希跨平台一致性

Query Recommendation Benchmark 的不可变文件以 LF 发布，但 Windows Git 工作区可能物化为 CRLF，导致相同文本在运行时无法通过 SHA256 校验。本步一方面在 `.gitattributes` 中冻结 Query Benchmark 的 JSON 和 JSONL 为 LF，另一方面让 Benchmark artifact 校验对 CRLF 进行规范化后再计算哈希。该修复只统一换行表示，不修改样本内容、标签或 split。

## 5. 独立 Query Recommendation Agent 评测链

本步新增 `query_recommendation_agent` 模块和 `scripts/run_query_recommendation_agent.py`，将运行与评测拆分为两个有数据隔离边界的阶段：

1. `run` 阶段只读取 `visible/cases.jsonl`，逐条运行 Agent，并持续写入 checkpoint；
2. 预测完成后冻结 `predictions.jsonl` 和 SHA256；
3. `evaluate` 阶段首先验证预测哈希，再读取 hidden ground truth、frames 和时点商家事实；
4. 评测器输出排序、约束满足、Query 满足、Agent 完成率、工具调用、延迟和 Token 指标。

运行脚本默认保持 50 条成本保护，只有显式传入 `--limit 500` 才允许运行完整数据集。checkpoint 以 `case_id` 保存，外部中断后可以跳过已经完成的样本。运行产物位于 `runs/`，按项目规则不进入 Git。

## 6. 在线调用有效性修正

第一次 50 条 Constrained Router 实验虽然完成了最终排序，但调用台账中的 50 次 Provider 尝试全部为 `api_connection`，随后进入 Rule Router fallback。其延迟仅为几十毫秒，`provider_request_id` 和 Token 均为空，因此该结果只能证明 fallback 外壳不会改变排序，不能证明在线 LLM Router 成功。

最小连通性复现表明，同一 Python 进程在受限沙箱中访问 Provider 时产生 `WinError 10013`，而在允许联网的执行环境中能够建立 HTTPS 连接。由此确认，失败发生在实验运行权限层，而不是 DeepSeek 返回非法 JSON、模型名错误或置信度不足。

获得用户明确授权后，实验按 1、10、50 条顺序重新执行，并在每一级检查 `status=success`、`rule_fallback=0` 和 Token 可观测性。三个阶段利用输入哈希缓存避免重复调用，合计覆盖前 50 个不同 Query 的 50 次真实 Provider 调用。随后脚本显式扩展到完整 500 条，前 50 条复用已验证缓存，其余样本继续在线运行。

## 7. 分级实验结果

表 1 汇总了正确 Query Recommendation Benchmark 上的 1、10、50 条实验。单样本指标只用于冒烟验证，正式效果结论不以 1 条或 10 条结果为依据。

| 阶段 | 逻辑 Router 调用 | 新 Provider 调用 | 缓存命中 | 成功 | 失败 | 新 Token | Provider 平均延迟 | HR@5 | HR@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 条 | 1 | 1 | 0 | 1 | 0 | 977 | 1268.19 ms | 100% | 100% |
| 10 条 | 10 | 9 | 1 | 10 | 0 | 9058 | 983.73 ms | 10% | 10% |
| 50 条 | 50 | 40 | 10 | 50 | 0 | 41,350 | 855.69 ms | 16% | 18% |

前 50 条实验中，完整排序与确定性 Rule Router 结果 50/50 一致，展示 Top-5 也 50/50 一致。该结果说明在线 Router 已经真实参与首步工具选择，但在这些单一推荐请求上没有改变冻结排序器的商家排列。

## 8. 完整 500 条在线实验

### 8.1 数据范围与运行状态

正式实验覆盖 Query Recommendation Benchmark V1 的全部 500 条样本，其中 development 400 条、validation 100 条。500 条均完成运行，没有 Case 级异常或 Agent fallback；495 条返回推荐，5 条进入澄清。共记录 1490 个 Agent 动作，非法动作数为 0。

需要模型选择的 495 个状态全部由 DeepSeek 成功返回，另外 995 个确定性后续动作直接单选旁路。500 条本轮运行中，前 50 个 Router 决策命中已验证缓存，剩余 445 次为新 Provider 调用。结合分级阶段，495 个不同的模型 Router 决策都曾真实调用 Provider 并成功：

- 成功率为 100%，失败和 Rule Router fallback 均为 0；
- 输入、输出和总 Token 分别为 496,236、16,335 和 512,571；
- 平均每次 Router 调用 1035.5 Token，P95 为 1132 Token；
- Provider 平均延迟为 865.75 ms，P95 为 1198.81 ms。

5 条澄清场景没有调用模型，因为代码只授权 `ask_clarification`。其中 2 条普通 pizza 或 ramen 推荐被上游理解为 `feedback_refinement + ambiguous_reference`；另外 3 条“适合多人或聚会”的推荐被判定缺少 `party_size`。这些问题发生在 Router 候选构造之前，不能通过要求 DeepSeek 从单一选项中重新选择来修复。

### 8.2 推荐与约束指标

表 2 给出完整 500 条实验的主要指标。QueryCompliance 的分母为实际返回推荐的 495 条；硬约束指标的分母为 140 条适用样本。

| 指标 | 结果 |
|---|---:|
| Recall@50 | 38.40% |
| Recall@100 | 53.60% |
| Recall@500 | 86.00% |
| HR@1 | 7.20% |
| HR@3 | 12.00% |
| HR@5 | 16.40% |
| HR@10 | 22.40% |
| MRR | 12.35% |
| NDCG@10 | 13.80% |
| Recommendation Completion Rate | 99.00% |
| Clarification Rate | 1.00% |
| HardConstraintSatisfaction@1 | 100%（140/140） |
| HardConstraintSatisfaction@5 | 98.71%（138.2/140） |
| QueryCompliance@1 | 60.61%（300/495） |
| QueryCompliance@5 | 52.77%（261.2/495） |
| Invalid Action Rate | 0%（0/1490） |
| Mean Tool Calls | 1.98 |
| 平均延迟 | 5.119 s |
| P95 延迟 | 9.028 s |

### 8.3 与旧 Agent 和静态 Query-aware 的对照

为了判断工具接入是否解决静态能力丢失问题，表 3 同时列出旧完整 Agent、第 33 步静态 Query-aware 和本步在线 Agent 的 500 条结果。

| 方法 | HR@1 | HR@3 | HR@5 | HR@10 | MRR | NDCG@10 | Recall@500 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 旧完整 Agent | 2.20% | 4.40% | 5.80% | 6.40% | 4.17% | 4.22% | 88.20% |
| 静态 Query-aware | 7.40% | 12.20% | 16.80% | 22.80% | 12.62% | 14.08% | 87.00% |
| 在线 Constrained Router Agent | 7.20% | 12.00% | 16.40% | 22.40% | 12.35% | 13.80% | 86.00% |

相对旧 Agent，本步将 HR@5 从 5.80% 提高到 16.40%，提高 10.60 个百分点；HR@10 从 6.40% 提高到 22.40%，提高 16.00 个百分点；QueryCompliance@5 从 28.67% 提高到 52.77%。平均延迟从 12.108 秒降至 5.119 秒，降低约 57.7%；平均工具调用从 8.164 次降至 1.98 次，降低约 75.7%。这些结果支持“完整 Query-aware 工具接入是主要改进来源”，而不是“增加一次 LLM 调用直接提高了商家排序”。

在线 Agent 相对静态 Query-aware 仍低 0.40 个百分点 HR@5。五条澄清产生空排序，其指标分子差异与静态结果吻合：Recall@500 少 5 个命中，HR@5 少 2 个命中，HR@10 少 2 个命中。因此，当前剩余差距主要位于 Task Type 与 Decision Readiness 的过度澄清，而不是 Query-aware Ranker 内部。

## 9. Token 与成本口径

Agent 评测器记录的平均 Token 为 11,292.91，P95 为 15,327，但该指标同时包含远程 Router Token 与本地模型输入量，不能全部按照 DeepSeek API Token 计费。本次 500 条运行的实际拆分为：

| 来源 | 实际 Token | 说明 |
|---|---:|---|
| DeepSeek Router | 461,186 | 本轮 445 次新调用；前 50 次命中缓存 |
| 本地 Qwen3 Reranker | 5,128,802 | Cross-Encoder 精排输入 |
| 本地 Qwen3 Embedding | 0 | 本轮命中已有向量缓存 |
| Session Memory LLM | 0 | 本轮没有外部调用 |
| Semantic / Answer LLM | 0 | 本轮没有外部调用 |

因此，运行过程中观察到的单条 7000 至 16,000 Token 长尾主要来自本地 Top-30 Reranker，而不是远程 Router。若后续目标是降低 API 费用，应优先用本地 SFT 模型复现约 1000 Token 的 Router 选择；若目标是降低本地推理时间和显存占用，则需要缩短 Cross-Encoder 输入、减少 Top-30 精排候选或增加更严格的粗排门槛。这两个优化目标不能使用同一 Token 数值混为一谈。

## 10. 工程实现与验证文件

本步涉及的主要模块为：

- `src/yelp_agent/query_aware_ranking/`：在线准备、附加拒绝过滤和完整冻结排序运行时；
- `src/yelp_agent/agent_tools/adapters/query_aware_ranking.py`：Query-aware Agent 工具适配；
- `src/yelp_agent/agent_tools/`：工具 schema、注册表和导出；
- `src/yelp_agent/rule_router/`：Query-aware 状态识别、动作策略和运行时装配；
- `src/yelp_agent/session_memory/effective_request.py`：保留最新 Query 并更新缓存身份；
- `src/yelp_agent/query_recommendation_agent/`：预测合同、运行、冻结产物和独立评测；
- `src/yelp_agent/evaluation/unified_end_to_end.py`：统一端到端指标出口；
- `scripts/run_query_recommendation_agent.py`：1 至 500 条成本受控运行脚本；
- `tests/test_query_aware_agent_tool.py` 与 `tests/test_query_recommendation_agent.py`：工具链和评测边界测试；
- `tests/test_effective_session_request.py`、`tests/test_rule_router.py`、`tests/test_query_aware_ranking_v2.py` 与 `tests/test_query_recommendation_benchmark.py`：会话、路由、排序和跨平台哈希回归。

本步完成后运行了全项目 Pytest 回归，517 项测试全部通过。测试过程仅出现来自 Joblib 与 NumPy 2.5 兼容性的既有弃用警告，没有测试失败。

正式 500 条产物位于：

```text
runs/query_recommendation_agent_v2/querybench_constrained_online_500/
├── predictions.jsonl
├── run_manifest.json
├── checkpoint_predictions.jsonl
├── router_llm/llm_calls.jsonl
├── total_llm_usage.json
├── evaluation/metrics.json
├── evaluation/case_audits.jsonl
└── full_500_online_report.md
```

`runs/`、`.env`、本地模型和处理后数据均为本地运行资产，不属于源代码提交内容。

## 11. 当前边界与下一步

本步已经证明，完整 Query-aware 排序能力可以作为受约束工具接入 Agent，并在 500 条 Query Recommendation Benchmark 上将 HR@5 从旧 Agent 的 5.80% 恢复到 16.40%。该结论仍受当前 Benchmark 设计限制：每条样本只有一个已知正例，Query 为基于 cutoff 前事实构造的半合成问句，未访问商家不能直接视为负例，因此结果不能替代真实线上搜索、点击、预订和满意度数据。

下一步不应继续无差别增加 Router 调用，而应按照错误来源分别处理：

1. 修复 2 条新推荐被误判为 `feedback_refinement + ambiguous_reference` 的任务类型错误；
2. 调整“适合多人或聚会”场景对 `party_size` 的追问必要性，避免 3 条不必要澄清；
3. 为本地 Router SFT 构建监督样本，输入使用当前受约束状态摘要，标签使用已经验证的 `choice_id`，目标是复现 DeepSeek 的稳定选择并降低外部费用；
4. 在 Router SFT 稳定后，再把工具选择正确率、无效动作、回退和成本作为奖励信号研究强化学习；
5. 单独优化 Top-30 Qwen3 Reranker 的输入长度和候选数量，因为本地 Cross-Encoder 才是当前 Agent Token 统计的主要来源；
6. 扩展多正例、真实日志和人工相关性标签，避免把单一后续到访商家当作唯一标准答案。

在当前单轮 Query Benchmark 中，DeepSeek 的 495 次模型决策全部选择 `GET_QUERY_AWARE_RANKING`。这说明该数据集足以验证在线 Router 稳定性，但对 Router 策略学习而言过于单一。SFT 可以先用于低成本复现；若要证明 Agentic RL 的价值，还需要包含详情查询、候选比较、评论证据、约束冲突和多轮反馈的混合任务 Benchmark。
