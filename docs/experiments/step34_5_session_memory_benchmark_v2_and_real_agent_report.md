# 第 34.5 步：Session Memory Benchmark V2 与真实 Agent 评测报告

## 1. 为什么重做 Benchmark

旧评测把“更便宜”强行标成 `price_level <= 2`，把“更近”强行标成 `distance_km <= 3`。用户没有说出这些数字，却会因为没有预测同一个数字而被扣分。V2 把相对要求标成 `price=lower`、`distance=closer`、`noise=quieter`；只有用户明确说出数字时，才评测数字。

标准答案由代码根据冻结的首轮 Top-5、候选价格、距离和噪声属性动态计算。DeepSeek 负责把已知意图改写成自然问句，并由另一次独立调用审核自然度和标签一致性；DeepSeek 不决定 Ground Truth。

## 2. 数据规模与泄漏审计

- 会话：200；后续轮次：500。
- Development / validation：400 / 100。
- 中文 / 英文：300 / 200。
- 场景：明确条件 200、相对偏好 110、拒绝/指代 70、组合修改 50、不应改变状态 30、澄清回答 20、冲突解决 20。
- 审计通过：`True`；完全重复 0；跨 split 重复 0；近重复 0；商家 ID 泄漏 0；相对请求数字泄漏 0；非法会话序列 0。

可见问题、冻结展示列表与隐藏标签分文件保存。Memory Manager、Router 和 Agent 运行时无法读取隐藏标签，只有 Benchmark Evaluator 可以读取。

## 3. 三层评测口径

1. **语义层**：本轮 `MemoryProposal` 是否正确理解任务、增删改条件、相对偏好、澄清答案和指代。
2. **Memory 层**：Reducer 执行后，正确内容是否真正进入 canonical state，而不只是模型输出里出现。
3. **行为层**：把更新后的状态应用到冻结候选，或交给完整 Agent 工具链，检查 Top-1/Top-5 是否满足新要求、拒绝商家是否被排除。

## 4. Rule 与 DeepSeek Memory 离线对比（500 轮）

| 指标 | Rule | DeepSeek | 差值 |
|---|---:|---:|---:|
| 任务类型准确率 | 17.80% | 60.60% | +42.80 pp |
| 任务目标准确率 | 50.00% | 90.60% | +40.60 pp |
| 条件字段+操作召回率 | 0.00% | 82.96% | +82.96 pp |
| 条件核心语义召回率 | 0.00% | 75.19% | +75.19 pp |
| 相对偏好 Precision | 0.00% | 82.47% | +82.47 pp |
| 相对偏好 Recall | 0.00% | 100.00% | +100.00 pp |
| 条件真正写入 Memory 的召回率 | 6.30% | 18.15% | +11.85 pp |
| 相对偏好真正写入 Memory 的召回率 | 0.00% | 100.00% | +100.00 pp |
| 信息缺口完全匹配率 | 95.60% | 69.40% | -26.20 pp |
| 指代解析召回率 | 74.35% | 72.17% | -2.17 pp |
| 拒绝商家召回率 | 18.57% | 12.86% | -5.71 pp |
| 不应修改状态准确率 | 93.33% | 86.67% | -6.67 pp |
| 冻结候选行为 Compliance@1 | 68.60% | 96.34% | +27.74 pp |
| 冻结候选行为 Compliance@5 | 82.01% | 96.95% | +14.94 pp |
| 数字幻觉率 | 0.00% | 0.00% | +0.00 pp |

DeepSeek 明显理解了规则难以覆盖的自然语言：相对偏好 Recall 从 0% 到 100%，任务目标准确率从 50.0% 到 90.6%，冻结候选 Compliance@1 从 68.6% 到 96.3%。

但这不是全胜。信息缺口完全匹配率从 95.6% 降到 69.4%，拒绝商家召回率也低于 Rule；说明模型会过度解释，且“不要第 N 家”的指代—拒绝链路仍不稳定。严格条件五元组 Recall 只有 18.1%，而条件核心语义 Recall 为 75.2%，说明多数时候意思理解对了，但 operator/importance 等标准化字段没有对齐。

### 冻结 validation（100 轮）

| 指标 | Rule | DeepSeek | 差值 |
|---|---:|---:|---:|
| 任务类型准确率 | 20.00% | 62.00% | +42.00 pp |
| 任务目标准确率 | 52.00% | 93.00% | +41.00 pp |
| 条件字段+操作召回率 | 0.00% | 83.33% | +83.33 pp |
| 条件核心语义召回率 | 0.00% | 75.93% | +75.93 pp |
| 相对偏好 Precision | 0.00% | 78.05% | +78.05 pp |
| 相对偏好 Recall | 0.00% | 100.00% | +100.00 pp |
| 条件真正写入 Memory 的召回率 | 1.85% | 16.67% | +14.81 pp |
| 相对偏好真正写入 Memory 的召回率 | 0.00% | 100.00% | +100.00 pp |
| 信息缺口完全匹配率 | 94.00% | 73.00% | -21.00 pp |
| 指代解析召回率 | 76.09% | 73.91% | -2.17 pp |
| 拒绝商家召回率 | 21.43% | 14.29% | -7.14 pp |
| 不应修改状态准确率 | 83.33% | 66.67% | -16.67 pp |
| 冻结候选行为 Compliance@1 | 71.64% | 91.04% | +19.40 pp |
| 冻结候选行为 Compliance@5 | 82.09% | 92.54% | +10.45 pp |
| 数字幻觉率 | 0.00% | 0.00% | +0.00 pp |

Validation 只用于最终报告，没有再用它调整规则或阈值。

## 5. 完整 Agent Harness 端到端对比（200 会话 / 500 轮）

完整链路包含 Query 召回、硬约束、LightGBM/Hybrid V2、本地 Qwen Embedding、本地 Qwen Reranker、语义排序、Top-5 展示和 Session Memory。

| 指标 | Rule | DeepSeek | 差值 |
|---|---:|---:|---:|
| Fallback 会话率 | 45.00% | 16.50% | -28.50 pp |
| 脚本轮次释放率 | 83.00% | 92.60% | +9.60 pp |
| 触发动作准确率 | 62.40% | 67.20% | +4.80 pp |
| 有效推荐率 | 100.00% | 100.00% | +0.00 pp |
| 拒绝商家排除率 | 76.09% | 89.09% | +13.00 pp |
| 真实 Agent Compliance@1 | 18.09% | 15.87% | -2.22 pp |
| 真实 Agent Compliance@5 | 26.32% | 23.17% | -3.14 pp |

平均会话延迟：Rule 5180 ms；DeepSeek cache-only 6903 ms。

端到端 DeepSeek 组使用了 349 条已经审核的 Memory Proposal，另有 114 条 cache miss 按 Rule 安全回退；本次回放新增 provider call 为 0、新增 API Token 为 0。因此这一列准确名称是 **DeepSeek Memory cache-only**，不是伪装成 500 轮全量在线调用。

结论必须如实报告：DeepSeek 把 Fallback 从 45.0% 降到 16.5%，轮次释放率从 83.0% 提高到 92.6%，拒绝商家排除率从 76.1% 提高到 89.1%；但真实 Agent Compliance@1 从 18.1% 降到 15.9%，Compliance@5 从 26.3% 降到 23.2%。它让 Agent 更能继续执行和保持会话，却没有让最终商家排序更准。这证明 Memory 的语义指标不能代替端到端推荐指标。

## 6. 本步修复的运行时问题

- 多轮重新召回曾被 Harness 误判为 `unauthorized_business_scope_change`。现在只有 `retrieve_candidates` 可以合法替换范围；硬约束仍只能缩小范围，其他工具仍不能换范围。
- 本地 Embedding/Reranker 工作量曾与外部 LLM Token 共用 12k 上限，造成大量 `max_total_tokens_exceeded`。V2 保持 Top-30/Top-20 和序列长度限制，但将本地工作预算提高到 100k；外部 API 仍由独立 LLM 配置限制。
- 发给 DeepSeek 的 Memory Context 已移除真实用户/商家 ID、候选列表和自由文本语义摘要。只发送结构化条件、信息缺口、相对偏好、结果序号和数量；显式 ID 使用一次性别名后在本地恢复。
- Replay Runner 支持 awaiting-user 快照和每 10 个会话 checkpoint，长跑失败不再丢掉全部结果。

## 7. API Token 账单

- Benchmark 生成和独立审核：50 次调用，输入 185,240，输出 75,468，共 260,708 tokens。
- 首次 500 条 Memory 全量实验：共 1,310,084 tokens。
- 两轮非法输出规范化复跑：新增 643,939 tokens。
- 完整 Agent cache-only 回放：0 次新 provider call，0 个新 API Token。
- 本步可核对的累计 API 消耗：至少 **2,214,731 tokens**。

最早一轮因问题重复而废弃的 500 条生成/审核，旧脚本没有在写入结果前保存 Token 台账。其精确消耗无法恢复，因此没有猜测或混入累计值；“至少”正是因为这一笔未知。Agent 指标中的 `input_tokens` 是本地模型工作量估计，不是外部 API Token。

## 8. 下一步改进

1. 统一标签、Proposal 和 Reducer 的 operator/importance canonicalization，同时保留严格匹配和核心语义匹配。
2. 为“不要第 N 家”建立统一的 reference + rejection contract，供 Parser、Resolver、Reducer、Router 共用。
3. 把相对偏好、拒绝列表和结构化 Delta 直接变成 Query 召回、硬约束与 LightGBM/语义排序特征，而不是只存在 Memory 中。
4. 将 Agent 的 provider Token、本地模型 work 和工具调用预算拆成三个独立字段。
5. 再比较 Rule、DeepSeek、Constrained LLM 和 Cost-aware LLM；同时报告 HR@1/3/5/10 与 Query-conditioned Compliance@1/5，不能只看动作是否正确。

## 9. 复现入口

- `scripts/build_session_memory_benchmark_v2.py`：生成、审核并冻结 Benchmark。
- `scripts/run_session_memory_benchmark_v2.py`：运行 Rule/DeepSeek Memory 三层离线评测。
- `scripts/run_session_memory_agent_replay_v2.py`：运行真实 Agent 多轮回放。
- `configs/rule_router_memory_v2.yaml`：本步专用 Top-5 Router 配置，不修改 Step 24 冻结的 Top-3 基线。

大体积 trace 位于 `runs/session_memory_v2/`，按仓库规则不提交；冻结 Benchmark、审计、Manifest、代码、测试和本报告提交 Git。

### 当前最终运行台账

- DeepSeek 离线最终运行：logical calls=500，cache hits=373，provider calls=127，本轮新增 tokens=331248。
- Agent DeepSeek cache-only：logical calls=463，cache hits=349，provider calls=0。
