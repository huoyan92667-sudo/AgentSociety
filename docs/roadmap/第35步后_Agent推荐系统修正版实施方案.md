# 第 35 步后：Agent 推荐系统修正版实施方案

> 状态：待审核、尚未实施  
> 适用范围：第 35.1～35.10 步  
> 核心目标：让 Agent 不仅会选择动作，还能给出合理的 Top-5 推荐、清晰的证据、明确的不确定项，并把错误沉淀为后续训练数据。

## 1. 为什么需要修正

当前项目已经具备以下能力：

- Query + History 双路召回；
- LightGBM 粗排；
- 本地 Embedding；
- 本地 Qwen Reranker；
- Review RAG；
- Evidence Aggregator；
- Session Memory；
- Constrained LLM Router；
- Agent Harness；
- 多套离线 Benchmark。

当前主要问题不是模块缺失，而是这些模块还没有围绕一个完整的用户目标连接起来：

> 用户提出真实需求后，Agent 应理解完整上下文，决定需要查询什么信息，给出合适的 Top-5 商家，并明确展示每家商家为什么合适、有哪些证据、哪些地方仍然无法确认。

现有系统主要存在以下问题：

1. 固定规则过早限制 Agent，只给 LLM 一个可选动作；
2. 追问信息后，不一定真正改变召回、排序或证据检索；
3. Review RAG 主要用于详情问答和候选比较，没有接入普通推荐的 Top-5；
4. 官方政策问题直接保守回答，没有先查询静态属性与历史评论；
5. 第 35 步主要评估动作流程，没有完成可信的端到端推荐准确率实验；
6. 普通推荐只返回商家 ID，缺少可人工审核的推荐理由和证据；
7. 错误记录没有说明已有证据、缺失证据和根因，无法形成数据飞轮；
8. 多套 Benchmark 的用途混淆，错误地使用 Agent Scenario HR 衡量新的 Query 推荐结果。

---

## 2. 修正后的最终目标

Agent 的输入应当是：

```text
用户历史画像
+ 当前自然语言需求
+ 当前会话上下文
+ 已拒绝商家
+ 已展示候选
+ 用户补充信息
```

Agent 的输出必须包含：

```text
Top-5 推荐商家
+ 每家商家的条件匹配情况
+ 排名分数拆解
+ 支持证据
+ 反对证据
+ 信息缺失
+ 不确定性
+ 可点击证据入口
+ Agent 完整执行轨迹
```

目标流程如下：

```text
完整会话
    ↓
生成当前有效需求 Effective Request
    ↓
区分硬条件、软偏好和开放语义目标
    ↓
生成需要验证的 Evidence Needs
    ↓
LLM 判断当前是否真正需要追问
    ↓
LLM 从合法工具中选择下一步
    ↓
召回、过滤、排序、证据查询
    ↓
判断证据是否足够
    ↓
必要时继续查证、换工具或追问
    ↓
生成 Top-5 推荐和 Evidence Card
    ↓
统一端到端评测与错误归因
```

---

## 3. 重新设计 Effective Request

修正后的 Effective Request 同时保存结构化条件和开放语义，不再让固定 Aspect 成为 Agent 的能力边界。

### 3.1 硬条件

硬条件必须结构化，因为需要由代码确定性过滤：

```json
{
  "hard_constraints": [
    {
      "field": "category",
      "operator": "includes",
      "value": "Steakhouses"
    },
    {
      "field": "distance_km",
      "operator": "less_than_or_equal",
      "value": 5
    },
    {
      "field": "category",
      "operator": "excludes",
      "value": "Fast Food"
    }
  ]
}
```

硬条件由代码执行，LLM 不允许偷偷忽略或覆盖。

### 3.2 软偏好

软偏好用于排序，不应该因为信息不够精确就阻止推荐：

```json
{
  "soft_preferences": [
    {
      "field": "quiet_environment",
      "value": true,
      "importance": "strong"
    },
    {
      "field": "group_suitable",
      "value": true,
      "importance": "preferred"
    }
  ]
}
```

例如“最好适合聚餐”通常属于软偏好。人数未知时可以追问，但不一定必须停止推荐。

### 3.3 开放语义目标

开放语义用于保存固定 Aspect 没有覆盖的新需求：

```json
{
  "semantic_goal": "寻找一家适合8人求婚后聚餐的牛排馆，希望有比较私密的空间、服务稳定，并且不会太拥挤",
  "open_requirements": [
    "8个人能否坐在一起",
    "是否有大桌或包间",
    "是否适合求婚等特殊场合",
    "晚餐时段是否拥挤",
    "是否需要提前预约"
  ]
}
```

即使系统没有 `proposal_suitable`、`large_table` 或 `private_room` 等标签，也可以把自然语言目标交给 Review RAG 和开放证据判断模块。

### 3.4 证据需求

Agent 应明确记录自己想证明什么：

```json
{
  "evidence_needs": [
    {
      "claim_id": "claim-1",
      "claim": "这家店适合8人一起用餐",
      "source_preference": [
        "business_attribute",
        "review",
        "official_source"
      ],
      "freshness": "historical_experience_acceptable",
      "importance": "strong"
    },
    {
      "claim_id": "claim-2",
      "claim": "目前允许预订8人座位",
      "source_preference": ["official_source"],
      "freshness": "current_required",
      "importance": "preferred"
    }
  ]
}
```

这样 Agent 调用工具时有明确目标，而不是机械执行固定流程。

---

## 4. 重新定义信息缺口与追问策略

当前接近于：

```text
information_gaps 不为空
→ 只能 ask_clarification
```

修正后把信息缺口分成三类。

### 4.1 阻塞性缺口

没有这些信息就无法执行硬条件或锁定查询对象：

- 要求 3 公里以内，但没有用户位置；
- 用户说“第一家”，但当前没有候选列表；
- 同时要求必须是酒吧和必须排除酒吧；
- 用户要求比较商家，但没有提供明确商家。

这类缺口必须追问。

### 4.2 非阻塞、但可提升质量的缺口

例如：

- 适合聚餐，但没有具体人数；
- 希望便宜，但没有准确预算；
- 希望浪漫，但没有指定室内或室外；
- 希望停车方便，但没有说明是否接受路边停车。

这种情况下，LLM 可以选择：

```text
直接按一般需求推荐
先追问以提高质量
先查询证据，再判断是否需要追问
```

代码不应强制只允许追问。

### 4.3 开放偏好

例如“最好适合聚餐”。如果一般的 `group_suitable` 证据足够，可以先推荐并提示：

> 当前推荐基于一般聚餐体验；如果人数超过 8 人，我可以继续帮你核对大桌和预约信息。

### 4.4 追问价值评测

追问是否正确不能只看动作标签，而应比较：

```text
追问前 Top-5
→ 用户回答
→ 追问后 Top-5
```

需要检查：

- 新信息是否进入 Effective Request；
- 是否触发新的检索 Query；
- 是否改变候选过滤；
- 是否改变排序；
- 是否增加相关证据；
- 最终 Compliance 是否提高；
- 如果结果完全没有变化，追问是否属于低效追问。

---

## 5. 让 Party Size 真正影响推荐

当前 `party_size` 主要进入语义排序文本，对最终结果影响有限。

修正后的流程：

```text
party_size=8
    ↓
生成证据目标
    ↓
查询 party of eight、large group、large table、private room
    ↓
查询 reservation、group reservation、crowded
    ↓
为 Top 候选形成容量与聚餐证据
    ↓
重新排序
```

证据解释分为三类。

### 5.1 可以支持的历史体验

如果评论明确说：

> Our party of ten was seated together.

可以作为“适合大型聚餐”的支持证据。

### 5.2 只能弱支持的证据

如果评论只说：

> Great for groups.

只能支持“一般适合聚餐”，不能断言一定能容纳 8 人。

### 5.3 必须查询当前来源的信息

以下内容不能仅靠历史评论确定：

- 今天晚上是否有 8 人桌；
- 是否必须提前预约；
- 当前包间是否开放；
- 是否收取大团体服务费。

最终回答需要区分历史经验和当前政策：

> 历史评论中有两次 8 人以上聚餐记录，因此它大概率适合大型聚餐；但没有当前桌位和包间信息，建议预订前确认。

---

## 6. 将 Review RAG 升级为开放式 Claim RAG

### 6.1 保留现有能力

以下能力继续复用：

- 明确锁定商家范围；
- 严格限制 cutoff；
- Aspect 召回；
- BM25；
- 本地 Qwen Embedding；
- RRF 融合；
- Top-5 Review；
- Review ID；
- 商家范围隔离；
- 本地向量缓存。

### 6.2 新增开放 Claim 输入

在现有 `ReviewSearchRequest` 基础上增加：

```json
{
  "claim_to_verify": "这家店是否适合8个人一起用餐",
  "search_queries": [
    "party of eight",
    "large group",
    "large table",
    "private dining room",
    "group reservation",
    "seated together"
  ]
}
```

`aspects` 变成可选的召回加速信息，不再是能力边界。

### 6.3 没有标签时的召回

```text
Aspect 路线没有结果
    ↓
BM25 使用原始 Query 和 LLM 生成的查询表达
    ↓
Embedding 查找语义相近评论
    ↓
本地 Reranker 重排评论段
    ↓
Top 评论交给开放证据判断器
```

这样无需为每一个新问题修改长期词表。

### 6.4 开放证据判断器

每条检索结果需要判断为：

```text
supports
contradicts
partially_supports
irrelevant
insufficient
```

示例：

```json
{
  "claim": "适合8人聚餐",
  "stance": "supports",
  "review_id": "R123",
  "evidence_span": "Our group of ten was seated together...",
  "confidence": 0.86,
  "reason_code": "EXPLICIT_LARGE_GROUP_EXPERIENCE"
}
```

LLM 可以判断语义，但代码必须校验：

- Review ID 真实存在；
- Review 属于当前商家；
- Review 早于 cutoff；
- 引用原文确实出现在该 Review；
- 不允许凭空生成证据。

---

## 7. RAG 延迟控制

当前 Review RAG 正式实验的平均延迟约为 305 ms，P95 约为 1.05 秒，使用本地模型且不产生外部 API 费用。

它不应对数百家候选逐一运行，而应采用三层漏斗。

### 7.1 第一层：基础召回与排序

```text
Query 召回
+ History 召回
+ 硬约束
+ LightGBM
+ Embedding
+ Cross-Encoder
```

得到 Top-20 或 Top-30。

### 7.2 第二层：结构化商家画像

读取 Top-20 的：

- Aspect 聚合；
- 质量；
- 价格等级；
- 距离；
- 类别；
- Yelp 静态属性。

用低成本证据缩小到 Top-5 或 Top-8。

### 7.3 第三层：按需 Review RAG

推荐策略：

| 场景 | Review RAG 范围 |
|---|---:|
| 普通推荐 | Top-3 |
| 强体验需求 | Top-5 |
| 商家比较 | 明确比较的 2～3 家 |
| 官方政策 | 明确商家 |

### 7.4 批量与缓存

一次请求批量处理多个商家：

```text
一个 Query 向量
→ 多个商家评论候选
→ 批量 Embedding
→ 批量 Rerank
```

继续复用：

- 评论向量缓存；
- Query Hash；
- 商家 + Claim + cutoff 缓存；
- Evidence Assessment 缓存。

建议性能目标：

| 阶段 | 目标延迟 |
|---|---:|
| 基础召回排序 | 1～3 秒 |
| Top-5 Review RAG | 1～2 秒 |
| LLM 证据总结 | 1～3 秒 |
| 完整推荐平均延迟 | 5～8 秒 |
| 完整推荐 P95 | 不超过 15 秒 |

---

## 8. 统一事实验证：VERIFY_BUSINESS_CLAIM

不再把 `check_official_source` 理解为“只有官网，找不到就拒绝回答”。

新增统一能力：

```text
VERIFY_BUSINESS_CLAIM
```

### 8.1 第一阶段：静态属性

例如用户询问是否允许带狗，先检查可用的商家属性：

```text
DogsAllowed
OutdoorSeating
BusinessParking
RestaurantsReservations
```

需要记录属性值、来源和数据快照边界。

### 8.2 第二阶段：历史评论

调用 Review RAG 查询：

```text
dogs allowed
dog friendly
brought my dog
no dogs allowed
pet on patio
service animal
```

输出支持证据、反对证据、日期和置信度。

### 8.3 第三阶段：官方来源

未来接入：

```text
OFFICIAL_SOURCE_SEARCH
```

查询：

- 商家官方网站；
- 官方 FAQ；
- 官方订位页面；
- 官方社交账号；
- 当前公开商家页面。

### 8.4 证据状态

最终状态不再只有 True/False：

```text
officially_confirmed
historically_supported
historically_contradicted
conflicting_evidence
insufficient_evidence
current_verification_required
```

如果找不到当前官方政策，仍然必须展示已经找到的历史证据，而不是只让用户自行确认。

---

## 9. 重新设计 LLM Router

### 9.1 当前限制

当前逻辑中，如果规则先产生某些 `information_gaps`，LLM 可能只得到一个 `ask_clarification` 选项。这并不是真正的动态决策。

### 9.2 LLM 输入

```json
{
  "effective_request": {},
  "current_evidence": {},
  "missing_evidence": [],
  "candidate_summary": {},
  "tool_catalog": [],
  "remaining_budget": {},
  "previous_failures": []
}
```

### 9.3 LLM 输出

```json
{
  "action": "SEARCH_BUSINESS_REVIEWS",
  "objective": "确认Top-5商家是否适合8人聚餐",
  "arguments": {
    "business_ids": ["B1", "B2", "B3", "B4", "B5"],
    "claim_to_verify": "是否适合8人共同用餐",
    "query_text": "party of eight, large group, large table, private room"
  },
  "expected_evidence": [
    "明确的大型聚餐记录",
    "座位或包间描述",
    "预约要求"
  ],
  "stop_condition": "每家得到两条相关证据，或确认无证据",
  "reason_code": "GROUP_CAPACITY_EVIDENCE_REQUIRED"
}
```

### 9.4 代码负责的安全边界

LLM 不能绕过：

- 商家 Scope；
- cutoff；
- 最大步骤数；
- 最大工具调用数；
- 最大 Token；
- 硬约束；
- Pydantic Schema；
- 引用真实性；
- Review 与商家归属；
- 工具权限。

应该放开的是决策空间，不是安全边界。

### 9.5 工具没有结果时

第一次没有结果后，LLM 可以选择：

```text
更换检索表达
更换工具
修改证据目标
使用已有证据保守回答
追问用户
结束并说明缺失项
```

每次重试必须至少改变以下一项：

- Query；
- 工具；
- 商家 Scope；
- 证据目标。

不允许相同工具、相同参数在没有状态变化时原地循环。

---

## 10. Top-5 Recommendation Evidence Card

普通推荐不能再只有商家 ID。每一家 Top-5 都必须生成证据卡：

```json
{
  "rank": 1,
  "business_id": "B1",
  "business_name": "Example Steakhouse",
  "final_score": 0.83,
  "matched_requirements": [
    {
      "requirement": "牛排馆",
      "status": "satisfied",
      "source": "business_category"
    },
    {
      "requirement": "5公里以内",
      "status": "satisfied",
      "value": "3.2km",
      "source": "coordinates"
    },
    {
      "requirement": "适合8人聚餐",
      "status": "supported",
      "confidence": 0.78,
      "source": "reviews"
    }
  ],
  "supporting_evidence": [],
  "contradicting_evidence": [],
  "missing_evidence": [],
  "score_breakdown": {},
  "why_ranked_here": "满足类别和距离要求，并有大型聚餐评论证据",
  "warnings": []
}
```

### 10.1 分数拆解

至少展示：

- History 个性化分数；
- Query 召回分数；
- 类别分数；
- 距离分数；
- 价格匹配；
- LightGBM 分数；
- Embedding 分数；
- Cross-Encoder 分数；
- Aspect 分数；
- Review Evidence 分数；
- 最终融合分数。

### 10.2 证据引用

评论证据包含：

- Review ID；
- Business ID；
- 评论日期；
- 命中原文；
- 支持或反对；
- 相关度；
- 置信度。

### 10.3 可点击证据

Yelp Open Dataset 没有稳定公开 Review URL，因此第一版提供本地 Case Explorer：

```text
/evidence/business/{business_id}/review/{review_id}
```

页面显示：

- 商家名称；
- Review 原文；
- 命中片段高亮；
- 日期；
- 星级；
- Aspect；
- Agent 引用原因。

官方网页证据则保存真实 URL、访问时间和原文片段。

---

## 11. 详细错误记录

每一个失败都生成 `AgentFailureRecord`：

```json
{
  "scenario_id": "...",
  "user_goal": "寻找适合8人聚餐的餐厅",
  "stage": "evidence_retrieval",
  "attempted_action": "SEARCH_BUSINESS_REVIEWS",
  "attempted_query": "party of eight, large group",
  "available_evidence": [],
  "missing_evidence": [
    "没有找到8人聚餐记录",
    "没有当前包间信息"
  ],
  "failure_type": "INSUFFICIENT_EVIDENCE",
  "root_cause_layer": "source_data",
  "impact": "无法证明Top-1适合8人",
  "recovery_attempted": true,
  "next_recommended_action": "OFFICIAL_SOURCE_SEARCH",
  "user_visible_message": "...",
  "human_review_status": "pending"
}
```

统一错误层级：

```text
QUERY_UNDERSTANDING
MEMORY
REFERENCE_RESOLUTION
TOOL_SELECTION
RETRIEVAL
HARD_CONSTRAINT
RANKING
EVIDENCE_RETRIEVAL
EVIDENCE_JUDGMENT
ANSWER_COMPOSITION
BENCHMARK_LABEL
EVALUATOR
SOURCE_DATA
```

这样可以明确错误属于哪个模块，而不是统一归结为“Agent 错误”。

---

## 12. 重构 Benchmark 定位

### 12.1 Query Recommendation Benchmark V1

作为最终推荐准确率主 Benchmark。

包含：

- 500 条半合成 Query 推荐任务；
- Development 400 条；
- Validation 100 条；
- 用户历史；
- 当前自然语言需求；
- 一个后来真实访问并给出至少 4 星的已知正例；
- cutoff 前可证明条件。

用于评估：

```text
Recall@50/100/500
HR@1/3/5/10
MRR
NDCG@10
```

### 12.2 Agent Scenario Benchmark V1

保留为 Router、工具和安全测试集，只评估：

- 是否追问；
- 是否选择合理工具；
- 多轮是否能够恢复；
- 是否发生非法动作；
- 是否正确调用 Review RAG；
- 是否正确处理证据不足；
- 是否安全 Fallback。

不再使用该 Benchmark 的 HR 宣传最终推荐准确率。

需要修复：

- 明确预算却标记 `missing_budget`；
- 没有上一轮候选却回答“上一轮第一家”；
- 把所有 `group_suitable` 都强制标记为必须追问人数；
- 要求当前系统不存在的 `check_official_source` 工具。

### 12.3 Session Memory Benchmark V2

继续评估：

- 新增条件；
- 替换条件；
- 删除条件；
- 拒绝商家；
- 指代解析；
- 多轮恢复；
- 上下文压缩。

不能代替最终推荐准确率。

### 12.4 Benchmark 对照表

| 想知道什么 | 对应 Benchmark |
|---|---|
| 只用历史能否召回下一家 | Full Retrieval Benchmark |
| Query 能否找到并排好真实正例 | Query Recommendation Benchmark V1 |
| Agent 会不会追问、选工具和恢复 | Agent Scenario Benchmark V1 |
| 多轮条件有没有被保留和应用 | Session Memory Benchmark V2 |
| 推荐理由是否有可靠证据 | 新增 Evidence Benchmark / 人工审核集 |

---

## 13. 统一端到端评测指标

### 13.1 召回指标

```text
Recall@50
Recall@100
Recall@500
```

### 13.2 排序指标

```text
HR@1
HR@3
HR@5
HR@10
MRR
NDCG@10
```

这里的正确答案来自 Query Recommendation Benchmark V1：用户后来真实访问并给出至少 4 星的商家。

### 13.3 条件满足指标

```text
Hard Constraint Satisfaction@1/@5
Query Compliance@1/@5
Rejected Business Exclusion Rate
Post-Clarification Utility Gain
```

### 13.4 证据指标

```text
Evidence Coverage@1/@5
Evidence Support Precision
Citation Correctness
Contradiction Detection
Insufficient Evidence Detection
Unsupported Claim Rate
Source Freshness Reporting
```

`Evidence Coverage@5` 表示：Top-5 商家中，有多少家为关键推荐理由提供了可人工检查的证据。

### 13.5 Agent 指标

```text
Tool Selection Accuracy
Clarification Utility
Recovery Success Rate
Invalid Action Rate
Mean Tool Calls
Mean Latency
P95 Latency
Token Usage
```

---

## 14. 人工审核集

单一真实访问商家只是已知正例，不能把所有未访问商家当作错误答案。

建议从 Query Recommendation Benchmark 抽取：

- Development：80 条；
- Validation：20 条；
- 合计：100 条。

每条人工审核 Top-5：

```text
Query 理解是否正确
硬条件是否满足
软偏好是否满足
Top-1 是否合理
Top-5 是否合理
推荐理由是否支持排序
证据是否真的支持结论
是否存在反对证据
是否应该追问
追问是否改善结果
哪个商家更合理
错误来自哪个模块
```

可以使用 DeepSeek 辅助审核或第二模型交叉审核，但模型不能在没有人工检查的情况下同时生成题目和决定最终 Gold Label。分歧样本必须进入人工复核。

---

## 15. 数据飞轮

每次实验后自动生成四类数据。

### 15.1 成功轨迹

```text
正确理解
→ 正确选工具
→ 正确召回
→ 正确排序
→ 证据充分
```

用于 Router 行为克隆和后续策略训练。

### 15.2 理解错误

```text
原始 Query
正确 Effective Request
错误 Effective Request
差异字段
人工修正
```

用于训练语义解析器。

### 15.3 工具与证据错误

```text
LLM 选择的工具
实际应该选择的工具
检索 Query
已有证据
遗漏证据
```

用于训练 Tool Router 和 Claim RAG。

### 15.4 排序错误

```text
Top 候选
正确商家位置
完整特征
Query
用户画像
证据卡
人工偏好标签
```

用于：

- LightGBM 增量训练；
- Cross-Encoder 微调；
- Pairwise Preference Training；
- 后续 GRPO / Agentic RL。

---

## 16. 分步骤实施计划

### 第 35.1 步：清理评测口径

实施内容：

- 修正 Agent Scenario 错误标签；
- 删除不成立的上一轮引用脚本；
- 移除“所有聚餐都必须追问人数”的硬标签；
- 将第 35 步候选空间错位的 HR 标记为不可用；
- 建立统一 Benchmark Registry；
- 每个指标注明数据来源、分母、正确答案含义和适用范围。

验收标准：

- 不再混淆 Agent HR、Query HR 和 Memory Compliance；
- 每一个指标都能追溯到明确 Benchmark 和 Ground Truth。

### 第 35.2 步：开放 Effective Request

新增：

- `semantic_goal`；
- `open_requirements`；
- `evidence_needs`；
- `blocking_gaps`；
- `optional_gaps`。

验收标准：

- “适合8人求婚聚餐”不会因为没有固定 Aspect 而丢失；
- 人数、求婚、私密空间和预约需求都能保存在当前会话需求中。

### 第 35.3 步：动态澄清策略

实施内容：

- `missing_party_size` 不再默认阻塞；
- LLM 可在追问、先检索和直接推荐之间选择；
- 追问答案必须更新 Effective Request；
- 记录追问前后的候选、排名和证据变化。

验收标准：

- 询问人数后至少有检索 Query、证据或排序发生变化；
- 无实际收益的追问被记录为低效追问。

### 第 35.4 步：开放 Claim Review RAG

新增：

- `claim_to_verify`；
- `search_queries`；
- 开放式 Review Evidence Judge；
- 无固定 Aspect 的 Query 检索；
- 支持、反对、部分支持、无关和证据不足判断。

验收标准：

- 没有 `private_room` 标签也能查询和引用包间评论；
- 没有 `large_table` 标签也能查询大型聚餐评论；
- 所有引用均能通过 Review ID 和原文校验。

### 第 35.5 步：Top-5 推荐证据补全

普通推荐流程升级为：

```text
Top-20 结构化画像
→ Top-5 Review RAG
→ Evidence Aggregation
→ Evidence-aware Final Ranking
→ Recommendation Evidence Cards
```

验收标准：

- 每家 Top-5 都有匹配条件和分数拆解；
- 关键体验需求必须有证据或明确标记为未知；
- 完整推荐不再允许 `claims=[]`。

### 第 35.6 步：统一事实验证

实现 `VERIFY_BUSINESS_CLAIM`，先接入：

- Business Attributes；
- Business Profile；
- Review RAG。

验收标准：

- 宠物政策问题不再只返回“请自行确认”；
- 必须展示历史支持、反对证据和当前政策缺口。

### 第 35.7 步：官方来源工具

实现 `OFFICIAL_SOURCE_SEARCH`，保存：

- URL；
- 来源类型；
- 访问时间；
- 原文证据；
- 当前性；
- 支持或反对结论。

验收标准：

- 找不到官方信息时仍然返回历史 Review 证据；
- 历史评论不能冒充当前官方政策。

### 第 35.8 步：错误记录与 Case Explorer

实现：

- `AgentFailureRecord`；
- `RecommendationEvidenceCard`；
- 本地 Review 证据详情页；
- Query → 状态 → 动作 → 候选 → 排名 → 证据 → 输出全链路展示。

验收标准：

- 每条失败都能看到根因层级；
- Review ID 可以点击查看原文和命中高亮；
- 人工审核结果能直接保存为后续训练数据。

### 第 35.9 步：Query Recommendation 端到端实验

在 Query Recommendation Benchmark V1 的 500 条任务上比较：

```text
Query-aware Ranker
Rule Agent
DeepSeek Memory + Rule Router
DeepSeek Memory + Constrained LLM Router
Evidence-aware Constrained LLM Agent
```

必须报告：

- Recall@50/100/500；
- HR@1/3/5/10；
- MRR、NDCG@10；
- Compliance@1/5；
- Evidence Coverage；
- Citation Correctness；
- 平均/P95 延迟；
- Token；
- 失败案例和根因。

### 第 35.10 步：人工审核与数据飞轮

抽取 100 条任务人工审核 Top-5 和证据，输出：

```text
parser_corrections.jsonl
router_preferences.jsonl
retrieval_failures.jsonl
ranking_preferences.jsonl
evidence_judgments.jsonl
```

这些数据作为未来语义解析微调、Cross-Encoder 微调、Router 学习和 GRPO / Agentic RL 的基础。

---

## 17. 建议修改的代码位置

为了复用现有工程，不建议重新创建一套完全平行的 Agent。

建议修改或扩展：

```text
src/yelp_agent/session_memory/
  schema.py                 # 开放语义目标、证据需求、缺口等级
  effective_request.py      # 编译完整当前需求
  reducer.py                # 合并多轮开放需求

src/yelp_agent/constrained_llm_router/
  choices.py                # 不再让非阻塞缺口只产生一个动作
  context.py                # 加入已有证据、缺失证据和工具能力
  router.py                 # 选择目标、工具和恢复动作

src/yelp_agent/review_rag/
  schema.py                 # claim_to_verify、search_queries
  retriever.py              # 开放 Query 召回与批量检索

src/yelp_agent/evidence_aggregation/
  schema.py                 # 开放 Claim 判断结果
  aggregator.py             # 支持固定 Aspect 与开放 Claim

src/yelp_agent/agent_tools/
  catalog.py                # 注册统一事实验证与官方来源工具
  adapters/                 # Review、属性、官方来源 Adapter

src/yelp_agent/rule_router/
  terminal_executor.py      # 生成 Evidence Card 和可审计回答

src/yelp_agent/agent_evaluation/
  evaluator.py              # 修正 Benchmark 口径和多轮指标

新增建议：

src/yelp_agent/recommendation_evidence/
  schema.py
  card_builder.py
  failure_recorder.py
  claim_judge.py

src/yelp_agent/official_sources/
  schema.py
  search.py
  validator.py

src/yelp_agent/case_explorer/
  app.py
  views.py
```

旧版本实验和冻结结果继续保留，不倒改历史结果。

---

## 18. 最终冻结原则

1. LLM 负责理解目标和选择工具，代码负责安全、范围和真实性。
2. 固定 Aspect 是加速器，不是能力边界。
3. 追问不是天然正确，必须证明追问改善了后续结果。
4. RAG 应替用户阅读评论，而不是让用户自己寻找证据。
5. 历史评论可以证明历史体验，不能冒充当前官方政策。
6. 推荐结果必须有理由、有证据、有反对证据、有未知项。
7. Router 指标不能代替推荐准确率。
8. Query Recommendation Benchmark V1 是推荐准确率主评测集。
9. Agent Scenario Benchmark V1 只作为流程与安全测试集。
10. 每次失败必须沉淀为可训练、可复核的数据。

本方案最终追求的不是“Agent 按规定走完流程”，而是：

> Agent 能为真实需求主动收集信息、完成推荐、展示证据，并在失败时准确说明失败发生在哪里，为下一轮系统改进留下可以直接使用的数据。
