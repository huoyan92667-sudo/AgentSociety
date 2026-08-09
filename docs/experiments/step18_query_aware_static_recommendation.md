# 第 18 步：RecommendationRequest 与 Query-aware 静态推荐

## 1. 本步骤解决的问题

第 17 步 LambdaMART 主要回答“根据用户历史，他下一次可能喜欢什么”。第 18 步增加当前请求，例如：

```text
想吃牛排，最好安静一点，适合约会。
```

本步骤将自然语言转换为稳定的 `RecommendationRequest`，执行可验证的硬约束，再对同一批 Hybrid V2-B 候选进行 Query-only 与 Hybrid+Query 静态排序。该结果可以直接作为后续 `AgentState.request`，但当前阶段本身不是 Agent，也没有让 LLM 控制流程。

## 2. 核心数据流

```text
QueryParseInput
    ↓
RecommendationRequestParser
    ├─ RuleBasedRequestSignalExtractor（当前可靠下限）
    └─ RequestSignalExtractor seam（未来语义模型，可选且失败安全）
    ↓
确定性约束政策
    ├─ filter：可以可靠验证的 mandatory 条件
    ├─ rank：可用于静态排序的 strong/preferred 条件
    ├─ evidence：用户很在意但只能用画像/评论证据判断
    └─ clarify：数据或阈值不足，需要追问
    ↓
BusinessProfileV1（严格 cutoff 前）
    ↓
Hybrid V2 / Query-only / Hybrid+Query
```

公开的 Agent-ready 接口只有：

```python
QueryAwareRecommender.recommend(parse_input, candidates)
```

调用者不需要理解词表、否定识别、Haversine、Aspect 证据收缩或 RRF 的内部实现。

## 3. 为什么不能只有 hard/soft 一个布尔值

请求条件使用两个独立维度：

1. 用户强度：`mandatory / strong / preferred`；
2. 系统执行方式：`filter / rank / evidence / clarify`。

只有“用户明确要求且当前数据能够可靠验证”的条件才能进入 `filter`。

| 用户表达 | 用户强度 | 执行方式 | 原因 |
|---|---|---|---|
| 只想吃牛排 | mandatory | filter | Yelp 类别可验证 |
| 不要酒吧 | mandatory | filter | Yelp 类别可验证 |
| 必须在 5 km 内，且已有坐标 | mandatory | filter | 距离可计算 |
| 必须在 5 km 内，但没有位置 | mandatory | clarify | 缺少请求位置 |
| 人均不超过 200 | mandatory | clarify | Yelp 没有可靠的人均金额 |
| 最好安静 | preferred | rank | 商家 Aspect 是概率性证据 |
| 必须适合求婚 | mandatory | evidence | 重要但主观，不能假装为确定字段 |
| 对花生过敏 | mandatory | clarify/evidence | 当前 Yelp 数据不能提供安全保证 |

不确定条件不会自动升级为硬约束。硬约束遇到未知商家数据时采用显式 `unknown_policy`，不能把 unknown 偷偷当成 true。

多个允许类别按 OR 处理，例如“只想吃牛排或者日料”允许任一类别，不要求商家同时属于两个类别；明确排除类别仍按 AND 排除。

## 4. 当前请求契约

每个 `RequestCondition` 保存：

```text
field / operator / value
importance / enforcement
explicit / confidence
evidence_span / evidence_start / evidence_end
source / unknown_policy
```

`RecommendationRequest` 保存：

```text
request_id / user_id / session_id / cutoff_time / query_text
intent / conditions / party_size / location_center
missing_fields / referenced_business_ids / parse_warnings
parser_version
```

并自动提供：

```text
hard_constraints
soft_preferences
evidence_requirements
clarification_requirements
desired_categories
excluded_categories
```

`request_id` 只由用户、Session、cutoff、原始 Query 和 Parser 版本生成，不读取候选、target、ground truth 或环境密钥。

## 5. 当前规则解析器的能力和边界

当前 `rule-based-v1.0.0` 是可审计的高精度基线，支持中英文的：

- 餐饮类别、明确包含与排除；
- 必须、只要、不要、最好、尽量、希望等强度词；
- 人数、公里范围和人均预算；
- 安静、约会、停车、宠物、亲子、聚会、拥挤、排队、辣度、卫生、服务、食物质量和性价比等通用 Aspect。

它不能可靠解决：

- 长距离否定和复杂语法；
- “不是非要安静”一类双重否定；
- 隐喻、反讽和非常口语化的需求；
- 多轮中的“第一家”“和上次一样”；
- 未收录的新类别、新属性和跨语言表达；
- 复杂冲突条件的自动协商。

因此当前词表只是可靠下限，不是最终语义理解方案。

## 6. 为语义模型预留的 seam

语义模型只需要实现：

```python
class RequestSignalExtractor(Protocol):
    version: str
    def extract(QueryParseInput) -> Sequence[ExtractedRequestSignal]: ...
```

接入方式：

```python
RecommendationRequestParser(
    primary_extractor=RuleBasedRequestSignalExtractor(),
    semantic_extractor=future_semantic_adapter,
)
```

规则信号优先于完全相同的模型信号。语义模型只能提交字段、值、强度、证据片段和置信度，最终 `filter/rank/evidence/clarify` 仍由确定性政策决定。语义模型超时或异常时返回规则结果，并记录：

```text
SEMANTIC_EXTRACTOR_FAILED:<version>:<error_type>
```

当前 `configs/query_aware.yaml` 将在线 `semantic_runtime` 固定为 `disabled`，所以推荐运行时不会调用 DeepSeek。后续经过用户明确确认，项目使用 DeepSeek 离线生成了 Query Benchmark V2；离线数据生成和在线请求解析是两条独立链路。

## 7. 静态排序

候选来自第 17 步已经冻结的 Hybrid V2-B，召回机制不变。

### Hybrid V2

完全保持第 17 步顺序，只使用历史行为和画像。

### Query-only

只根据当前请求计算：

- 类别是否匹配；
- cutoff 前商家 Aspect 正负比例；
- Aspect 置信度、未知和冲突；
- 可执行硬约束。

Aspect 分数会按照置信度向 0.5 收缩，数据不足返回 unknown，不把少量评论当成事实。

### Hybrid+Query

不直接相加 LambdaMART 原始分数和 Query 分数，而是对两个排名使用 Reciprocal Rank Fusion：

```text
1 / (k + hybrid_rank)
+ query_weight / (k + query_rank)
```

当前 `k=60`、`query_weight=1` 是未调优的透明基线。没有 Query 排序证据时，输出严格保持 Hybrid 顺序。

## 8. 受控 Query 数据协议

最初冻结了 21 条人工种子请求：

```text
Development：12
Validation：9
```

保存位置：

```text
benchmarks/query_aware_v1/seed_queries.jsonl
```

每条数据保存结构化标准答案、语言、frame family、生成来源和防泄漏声明。Schema 不接受 `target_business_id` 等额外字段，并要求：

```text
uses_specific_business = false
uses_future_review = false
```

同一 frame family 不能同时进入 Development 和 Validation，避免把同一模板的改写随机拆开造成虚高。

这 21 条现在只作为快速冒烟测试。正式的 V2 Benchmark 使用 100 个结构化语义场景，每个场景由 DeepSeek 改写为 3 条中文和 2 条英文，共 500 条：

```text
Development：400
Validation：100
```

保存位置：

```text
benchmarks/query_aware_v2/queries_500.jsonl
benchmarks/query_aware_v2/manifest.json
```

规则解析器在 21 条种子上的结构化指标均为 1.0，但在 V2 上 Exact Match 只有 13.4%、Condition F1 为 45.56%。这正好证明少量定制句式的 100% 不能代表真实 Query 泛化能力。生成方式、Token、质量检查和分项结果见 `docs/experiments/step18_query_benchmark_v2_500.md`。

## 9. 真实 Yelp 数据链路验证

脚本在 Validation 中自动选择一个包含至少 5 家牛排店的任务：

```text
task_id = validation:T4Uk_zyBFvIUsBVninUqRg
cutoff_time = 2022-01-19 05:18:11
candidate_count = 500
```

输入：

```text
想吃牛排，最好安静一点，适合约会。
```

真实结果示例：

| 方法 | Top-1 | 说明 |
|---|---|---|
| Hybrid V2-B | Talula's Garden | 历史排序保持不变 |
| Query-only | Barclay Prime | 当前 Query 匹配最高 |
| Hybrid+Query | Fogo de Chao | 同时参考 Hybrid rank=7 与 Query rank=3 |

该运行读取 500 个冻结候选和 cutoff 时点的 BusinessProfile，没有加载 ground truth，也没有加载 Legacy Test。它证明真实数据链路可运行，但由于不存在真实 Query—点击/满意度标签，不能由此声称 Query-aware 排序准确率提高。

## 10. 外部相关性评测接口

代码定义了独立的 `QueryRelevanceJudgment`：

```text
request_id
business_id
relevance_grade：0–3
hard_constraint_status：satisfied / violated / unknown
label_source：human / llm_assisted
evidence_refs
```

只有完整候选获得外部标签后，评测器才比较：

- Hybrid V2；
- Query-only；
- Hybrid+Query；
- NDCG@5；
- MRR；
- Hard Constraint Satisfaction@5；
- Hard Constraint Unknown Rate@5。

评测器不接受 next-business ground truth 字段，避免把“下一条 Yelp 评论商家”误当成自然语言 Query 的唯一相关答案。

## 11. 主要工程文件

```text
src/yelp_agent/query/schema.py
src/yelp_agent/query/parser.py
src/yelp_agent/query/ranking.py
src/yelp_agent/query/adapters.py
src/yelp_agent/query/engine.py
src/yelp_agent/query/benchmark.py
src/yelp_agent/query/generation.py
src/yelp_agent/query/evaluation.py
configs/query_aware.yaml
scripts/evaluate_query_request_parser.py
scripts/generate_query_benchmark.py
scripts/run_query_aware_demo.py
```

运行命令：

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_query_request_parser.py
.\.venv\Scripts\python.exe scripts\run_query_aware_demo.py
```

未来提升和 LLM 生成 Query 的严格方案见：

```text
docs/roadmap/step18_query_aware_future_improvements.md
```
