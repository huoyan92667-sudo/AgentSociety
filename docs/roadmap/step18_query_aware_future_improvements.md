# 第 18 步后续提升路线：语义解析、Query 数据与相关性评测

## 1. 当前基线不能解决什么

当前规则解析器是低成本、可解释的下限，但不具备真正开放域语义理解能力。21 条人工种子只用于固定接口和防止明显回归；现已增加 500 条 DeepSeek 改写 Benchmark，当前规则在其上的 Condition F1 为 45.56%。该数据仍是合成数据，不能代表完整的真实用户表达分布。

当前项目没有真实的：

```text
用户自然语言 Query
→ 当时展示的候选
→ 点击、收藏、到店或满意度反馈
```

因此当前真实 Yelp 示例只能证明工程链路可运行，不能证明 Hybrid+Query 在真实用户效用上优于 Hybrid V2 或 Query-only。

## 2. 优先级 P0：扩充 Synthetic Query Benchmark V2（首批已完成）

### 2.1 先定义语义框架，再生成文本

不能让模型直接自由生成“1000 个问题”。应先人工定义不包含具体商家的结构化 frame：

```json
{
  "desired_categories": ["Steakhouses"],
  "conditions": [
    {"field": "distance_km", "operator": "<=", "value": 5, "importance": "mandatory"},
    {"field": "quiet_environment", "importance": "preferred"},
    {"field": "date_suitable", "importance": "strong"}
  ],
  "party_size": 2
}
```

随后让 Codex 或 DeepSeek 只负责把同一个 frame 改写成：

- 正式表达；
- 口语表达；
- 省略主语；
- 错别字和语序变化；
- 否定、双重否定；
- 多条件优先级；
- 备选类别；
- 模糊、缺失和冲突请求；
- 中英文与中英混合。

标准答案来自 frame，不来自模型生成后的反向猜测。

### 2.2 严禁提供给生成模型的数据

生成 Query 时不能提供：

- target business_id 或商家名称；
- 能唯一识别商家的地址、菜名或描述；
- target Review；
- cutoff 后评论；
- next-business ground truth。

生成 Prompt、模型名称、温度、Prompt SHA256、Token、延迟和失败都要记录。API Key、认证头和完整响应头不能落盘。

### 2.3 生成后的三层审核

1. 代码检查数字、否定、类别、人数和禁止词；
2. 第二个模型对照 frame 检查是否增加、删除或反转条件；
3. 对模型不一致、复杂否定和冲突样本进行人工抽样复核。

不能让同一个模型同时成为唯一生成器和唯一裁判。

### 2.4 当前规模与切分

已经完成：

```text
100 个独立 semantic frames
每个 frame 生成 5 种说法
总计 500 条 Query
```

必须按 frame family 切分，而不是随机按句子切分：

```text
Development：规则和 Prompt 开发
Validation：语义 Adapter、阈值和融合选择
Frozen Evaluation：全部策略冻结后一次运行
```

## 3. 优先级 P1：实现第二层语义解析 Adapter

当前 seam：

```python
RequestSignalExtractor.extract(QueryParseInput)
```

建议依次比较三种 Adapter。

### 3.1 本地轻量分类/槽位模型

用途：

- intent 分类；
- field/slot 提取；
- mandatory/strong/preferred 分类；
- 否定范围与条件冲突识别。

可选模型应支持中英文，并记录模型版本、Tokenizer 版本和权重 Hash。优势是低成本、可离线、延迟稳定；缺点是需要标注数据，对开放表达覆盖有限。

建议代码位置：

```text
src/yelp_agent/query/extractors/local_classifier.py
```

### 3.2 Embedding 辅助语义映射

用途不是直接决定硬约束，而是把未知表达映射到已有字段，例如：

```text
“能坐下来好好聊” → quiet_environment
“有点仪式感” → date_suitable
“不想为了吃饭开很久” → distance preference
```

Embedding 返回候选字段和相似度，确定性政策再决定 `rank/evidence/clarify`。低于阈值必须返回 unknown，不能强行分类。

建议代码位置：

```text
src/yelp_agent/query/extractors/embedding.py
```

### 3.3 DeepSeek/OpenAI-compatible 结构化解析

DeepSeek 只输出严格 Schema：

```text
intent
extracted conditions
importance suggestion
evidence span
confidence
ambiguities
```

它不能输出最终 `enforcement=filter` 决策。最终权限仍属于确定性 policy。

建议代码位置：

```text
src/yelp_agent/query/extractors/openai_compatible.py
```

接入时复用现有安全 LLM 客户端，要求：

- thinking 保持项目已确认的 `disabled`；
- 最大输出 Token 可设为 12000，但生成任务应尽量批处理；
- 温度 0；
- JSON Schema；
- 超时、重试和 Token 记录；
- 请求缓存；
- Fake LLM 覆盖非 JSON、遗漏、冲突、超时和异常；
- 失败返回规则基线并记录 `parse_warnings`。

2026-08-09 用户已明确确认本次 500 条 Benchmark 生成，并已完成真实 API 调用。以后重新生成或扩大数据集仍需再次明确确认。

## 4. 优先级 P2：解析器融合与置信度校准

规则与语义模型不是简单投票。建议政策：

```text
数字、明确否定、显式类别 → 高精度规则优先
隐含 Aspect、复杂意图 → 语义模型补充
规则与模型冲突 → 标记 ambiguous_requirement
低置信度 → clarify，不进入 filter
```

在 Validation 上校准不同字段的置信度阈值，分别报告：

- Slot Precision/Recall/F1；
- Hard Constraint Precision/Recall；
- False Hard Filter Rate；
- Missing-field Detection Precision/Recall；
- Conflict Detection Accuracy；
- Expected Calibration Error。

其中 False Hard Filter 的代价最高，应单独加权。

## 5. 优先级 P3：Query 与商家的语义匹配

当前匹配使用类别和 BusinessProfile Aspect。后续按计划第 25–26 步比较：

1. TF-IDF Query—Business Profile；
2. 多语言 Sentence Embedding cosine；
3. Cross-Encoder 对 Top-K 精排；
4. Review RAG 对具体体验属性取证。

语义模型只能在满足硬约束的候选中调整排名。Embedding/Cross-Encoder 的模型版本、候选文档 Hash、cutoff 和缓存键必须可追溯。

## 6. 优先级 P4：建立真正的 Query 相关性标签

要证明“用户历史 + 当前需求”优于单独使用任一信息，需要独立标签：

```text
request_id × business_id
relevance_grade：0–3
hard_constraint_status
evidence refs
label source
```

当前代码已经提供 `QueryRelevanceJudgment` 和三方法评测器，但不会自己制造标签。

推荐标签来源优先级：

1. 真实 Query/曝光/点击/到店或用户研究；
2. 人工标注 Query—候选相关性；
3. 多模型辅助标注加人工抽查；
4. 规则弱标签只用于训练预热，不能作为唯一最终评测。

最终比较：

- Hybrid V2；
- Query-only；
- Hybrid V2 + Query；
- NDCG@5、MRR；
- Hard Constraint Satisfaction@5；
- Unknown Rate；
- 分场景和分语言结果。

## 7. 优先级 P5：更可靠的约束数据

Yelp 当前无法可靠提供：

- 精确人均消费；
- 实时营业和排队；
- 订桌与多人容量；
- 过敏原安全保证；
- 当前宠物政策；
- 实时停车情况。

未来需要商家官方字段、地图、营业时间、预约或电话等外部工具。Review 只能作为历史体验证据，不能冒充官方政策。

## 8. 优先级 P6：多轮 RequestPatch

第 18 步处理第一轮完整请求。第 33 步 Session Memory 应增加：

```python
RequestPatch(
    add_conditions,
    replace_conditions,
    remove_conditions,
    rejected_business_ids,
    referenced_business_ids,
)
```

支持：

```text
太贵了。
换个近一点的。
第一家能带狗吗？
不要刚才推荐过的。
```

临时 Query 和 RequestPatch 不自动写入长期用户画像。

## 9. 推荐执行顺序

```text
1. 已冻结首批 100 个 semantic frames
2. 已经确认并用 DeepSeek 生成 500 条改写
3. 下一步：第二模型语义审核 + 人工困难样本抽样
4. 实现本地语义 Adapter
5. 实现 DeepSeek challenger
6. 在相同 Benchmark 上比较 Rule / Local / DeepSeek
7. 建立 Query—Business 外部相关性标签
8. 再调 Query 与 Hybrid 的融合策略
9. 全部冻结后进入第 19 步置信度与 Agent Router
```

在第 7 项完成前，不能宣传“Query-aware 推荐准确率已经提升”；可以宣传的是：系统已经具备严格请求 Schema、确定性约束政策、可替换语义 Adapter、时点安全商家证据、失败回退和独立评测接口。
