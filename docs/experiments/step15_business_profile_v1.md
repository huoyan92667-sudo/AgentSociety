# 第 15 步：商家画像 V1 与 Business Knowledge Store

## 1. 本步骤解决什么问题

用户画像回答“这个人过去喜欢什么”，商家画像回答：

> 这家商店在某个时间点以前表现怎样，有哪些得到历史评论支持的特点？

商家画像不是某个用户的记忆，而是所有推荐任务共享的商家知识。每次读取都
必须提供 `cutoff_time`，评分、热度和 Review Aspect 只能来自该时间以前。

## 2. 第 13、14、15 步的关系

第 13 步产生的每条 `ReviewAspectRecord` 同时包含 `user_id` 和
`business_id`，它只是评论级原子证据，还不是任何一方的聚合画像。

```text
ReviewAspectRecord
├─ 按 user_id + cutoff 聚合     → 第 14 步用户画像
└─ 按 business_id + aspect + cutoff 聚合 → 第 15 步商家画像
```

## 3. 输入数据和正文隔离

构建过程读取：

- `businesses.parquet`：3,973 家有效商家的静态字段。
- `reviews.parquet`：617,718 条历史评分事件。
- `aspect_records.parquet`：第 13 步的 99,073 条 Aspect 证据。

对 `reviews.parquet` 只投影以下五列：

```text
review_id / business_id / user_id / stars / date
```

评论 `text` 列不会加载，也不会写入新产物。Aspect 索引只保存评论 ID、来源
Hash、方面、情感、置信度和时间，不保存 `evidence_span`。本步骤没有重新处理
61 万条长评论，也没有调用 LLM。

## 4. 商家画像结构

```python
BusinessProfileV1(
    business_id,
    cutoff_time,
    categories,
    structured_attributes,
    quality,
    aspect_summaries,
    profile_reliability,
    evidence_summary,
    source_scope,
    profile_version,
)
```

每个 `BusinessAspectSummary` 保存：

```text
positive/negative/neutral/mixed_count
evidence_count / unique_users / effective_evidence
positive_ratio / negative_ratio
weighted_positive_ratio / weighted_negative_ratio
latest_evidence_time / confidence / conflict / status
```

固定的 14 个 Aspect 都会返回。证据不足时 `status=unknown`，方向比例为
`None`，不会用零伪装成负面结论。

## 5. 时间安全聚合

所有动态记录严格满足：

```text
event_time < cutoff_time
```

Aspect 使用 365 天半衰期：

```text
time_weight = 0.5 ** (距离 cutoff 的天数 / 365)
effective_support = time_weight × extractor_confidence
```

原始比例用于说明证据条数，时间加权比例用于表示近期共识。

## 6. known、unknown 和冲突

门槛按“商家 + 具体 Aspect”分别判断，不能拿 food_quality 的大量证据替
parking 凑数。

一个 Aspect 成为 `known` 必须同时满足：

```text
至少 3 条明确 positive/negative 证据
至少来自 2 名不同用户
有效方向权重大于 0
```

`neutral` 和 `mixed` 会保留用于审计，但不能把方向结论凑成 known。

如果时间加权后的正面比例和负面比例都不低于 0.3，则：

```text
conflict = true
```

这表示评论存在明显分歧，而不是强行选择一边。

## 7. 时点评分和热度

禁止使用 business 文件中的快照 `stars/review_count`。质量只基于 cutoff 前
评分，沿用 Hybrid V1 的公式：

```text
BayesianRating = (评分总和 + 20 × 时点全局平均分) / (评论数 + 20)
Quality = 0.8 × 归一化贝叶斯评分 + 0.2 × 归一化热度
```

热度使用 `log1p(review_count)`，并按该 cutoff 下全商家 P95 截断。

## 8. 商家画像可靠度

整体可靠度组合三部分：

```text
0.5 × 评分证据量饱和度
+ 0.3 × Aspect 证据量饱和度
+ 0.2 × known Aspect 覆盖率
```

每个 Aspect 还有自己的独立置信度，综合考虑近期有效证据量、不同用户数量和
正负方向的一致性。整体画像可靠度高，不代表每个 Aspect 都可靠，调用方必须
继续检查具体 Aspect 的 `status/confidence`。

## 9. 为什么不生成 1.1 亿份画像

项目共有 27,920 个任务和 3,973 家商家。直接做笛卡尔积会产生
110,926,160 份重复画像。

本步骤改为：

```text
离线紧凑时间索引
→ BusinessKnowledgeStore.get(business_ids, cutoff_time)
→ 只计算请求的候选
→ 按 business_id + cutoff_time 做 LRU 缓存
```

缓存最多保存 10,000 份画像。第 16 步只会为训练正例、采样负例和 validation
候选生成数值特征，不会把所有任务和全商家相乘。

## 10. 产物和真实构建结果

产物位于 `data/features/business_profiles/v1/`：

- `businesses.parquet`：自包含的静态商家索引。
- `rating_events.parquet`：不含正文的评分时间索引。
- `aspect_events.parquet`：不含证据文本的 Aspect 时间索引。
- `business_coverage.parquet`：商家—Aspect 原始覆盖审计。
- `manifest.json`：输入、配置和输出 Hash 及血缘审计。

真实结果：

| 项目 | 数量 |
|---|---:|
| 商家 | 3,973 |
| 精简评分事件 | 617,718 |
| Aspect 事件 | 99,073 |
| 有 Aspect 的商家 | 3,519 |
| 原始商家—Aspect 组合 | 14,344 |
| 达到方向证据门槛的组合 | 6,252 |
| 至少有一个 known Aspect 的商家 | 2,392 |
| 重复评分 review_id | 0 |
| Aspect 来源不匹配 | 0 |

“达到门槛”是基于完整开发源的覆盖审计；较早 cutoff 能看到的证据更少，因此
当时的 known 数量可能更低。

生产读取烟雾测试中，Zahav 在 2023 cutoff 前有 3,173 条评分，10 个 Aspect
达到 known 门槛，整体画像可靠度约为 0.943。这个例子只用于验证读取链路，
不是推荐效果结论。

## 11. Agent 接口

新增任务约束工具：

```text
GET_BUSINESS_PROFILE
```

代码接口为：

```python
get_business_profiles(business_ids, cutoff_time)
```

它只能查询当前任务候选集内的商家，且 cutoff 必须和任务完全一致。当前
Agent V0 不会自动调用该工具，因此本步骤不会改变旧 Agent 排名；后续受控
Agent Harness 才决定什么时候使用它。

## 12. 当前限制和第 16 步用途

- Aspect 只来自选定 5000 名用户的评论，热门商家证据明显更多。
- 454 家商家没有任何 Aspect；具体方面证据不足时必须保持 unknown。
- 商家结构化属性仍是 Yelp 静态快照，需要在报告中保留时效性限制。
- Review RAG 尚未实现，本步骤不能回答“哪条评论具体这样说”。

第 16 步将使用：

```text
用户 Aspect 偏好 × 商家 Aspect 共识
用户负向偏好 × 商家负向证据
价格匹配 / 类别匹配 / 位置距离
时点质量 / 用户画像可靠度 / 商家画像可靠度
各项 unknown 缺失标记
```

这些会成为 Hybrid V2 的数值特征，而不是直接让 LLM决定排名。
