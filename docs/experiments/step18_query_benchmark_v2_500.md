# 第 18 步补充：500 条自然语言 Query Benchmark V2

## 1. 这批数据是什么

本批数据用于测试“自然语言请求 → `RecommendationRequest`”的解析能力，不是 Yelp 的真实用户搜索日志，也不是 Query-aware 商家排序的点击标签。

生成方法不是让模型同时出题和猜标准答案，而是：

```text
代码定义 100 个结构化语义场景
    ↓
DeepSeek 只负责给每个场景写 5 种自然表达
    ↓
代码重新附上原始结构化标准答案
    ↓
Pydantic、去重和防泄漏检查
```

每个场景包含：

```text
中文直接表达
中文口语表达
中文隐含/调序表达
英文自然表达
英文口语表达
```

## 2. 数据规模

| 项目 | 数量 |
|---|---:|
| 独立语义场景 | 100 |
| 每个场景的问法 | 5 |
| Query 总数 | 500 |
| Development | 400 |
| Validation | 100 |
| 中文 | 300 |
| 英文 | 200 |

难度分布：

| 难度 | 数量 |
|---|---:|
| 单类别或单偏好 | 100 |
| 多条件组合 | 200 |
| 否定与强弱优先级 | 100 |
| 缺少类别、位置或可验证预算 | 100 |

数据文件：

```text
benchmarks/query_aware_v2/queries_500.jsonl
benchmarks/query_aware_v2/manifest.json
```

原先的 21 条人工数据保留在 `benchmarks/query_aware_v1/seed_queries.jsonl`，只用于快速冒烟测试。

## 3. API 生成记录

本次用户明确允许调用项目已经配置的 DeepSeek OpenAI-compatible API。最终有效版本使用：

```text
model = deepseek-v4-flash
temperature = 0
thinking = disabled
max_tokens = 12000
timeout = 90 秒
API 批次 = 5
```

最终有效版本用量：

| 项目 | 结果 |
|---|---:|
| 输入 Token | 9,167 |
| 输出 Token | 22,621 |
| 总 Token | 31,788 |
| Provider 总延迟 | 111,861.6 ms |
| API 尝试 | 5 次，均一次成功 |

完整可公开元数据写入 `manifest.json`。API Key、认证头和完整响应头没有写入 Benchmark。

## 4. 数据质量保护

生成器做了以下检查：

- 必须正好得到 100 个不重复的结构化含义；
- 每个含义必须正好生成 3 条中文和 2 条英文；
- 500 个 `case_id` 和 500 条 Query 文本必须全局唯一；
- 同一个 `frame_family` 不能跨 Development 和 Validation；
- 模型只看到用户语义，不看到内部 `filter/rank/evidence/clarify` 政策；
- 模型不接收 `missing_fields`，避免生成“请提供位置”之类反向提问；
- Schema 不接受 target 商家、未来评论或 next-business 标签；
- 每条记录保存模型名称和对应批次 Prompt SHA256；
- Dataset SHA256 为 `a0c14ddf80157bc30b63fc8dc96dce4276795b0551b12a4e10339ae779351d92`。

本轮已经完成结构、数量、重复文本和明显异常措辞检查。按照既定路线，第二模型逐条语义审核和人工困难样本复核仍属于后续质量提升，不能把当前数据宣传成真人金标准。

## 5. 当前规则解析器结果

`rule-based-v1.0.0` 在这 500 条上的结果：

| 指标 | 整体 | Development | Validation |
|---|---:|---:|---:|
| Exact Match | 13.4% | 14.0% | 11.0% |
| Condition F1 | 45.56% | 46.5% | 42.1% |
| Party Size Accuracy | 85.0% | 84.3% | 88.0% |
| Missing Fields Exact Match | 61.6% | 60.5% | 66.0% |

21 条人工种子曾得到 100%，而自然改写后的 500 条明显下降。这说明原先的 100% 主要代表代码适配了少量固定句式，不能代表真实语言泛化。新 Benchmark 暴露的主要问题包括：

- 新餐饮类别不在旧词表中；
- 中文口语、英文同义词和隐含表达无法命中；
- “必须/想要/最好”的作用范围判断不稳定；
- 否定条件和多个条件组合容易漏掉；
- 口语化人数表达覆盖不足。

因此这批数据的价值不是让当前分数好看，而是给后续本地语义模型、Embedding Adapter 和 DeepSeek Parser Challenger 提供固定、可复现的比较基准。

## 6. 复现命令

重新生成会产生真实 API 调用，必须显式允许：

```powershell
.\.venv\Scripts\python.exe scripts\generate_query_benchmark.py --allow-real-api --batch-size 20
```

评测当前规则解析器：

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_query_request_parser.py --output runs\query_aware_v2\parser_benchmark.json
```
