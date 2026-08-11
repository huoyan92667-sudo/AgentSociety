# 第 27 步：商家评论 Review RAG 完整实现与 500 场景报告

## 1. 本步完成了什么

第 27 步把“评论体验问题需要原始评论证据”从一个不可用占位工具，升级为真实、可离线评测的 Agent 能力：

```text
用户问题 + 明确 business_id + cutoff
    ↓
只读取该商家 cutoff 之前的评论段
    ↓
Aspect 路线 + BM25 路线
    ↓
候选前 30 段使用本地 Qwen3-Embedding-0.6B
    ↓
RRF 融合、按 review_id 去重
    ↓
返回 Top-5 Review ID、原文片段和完整工具轨迹
```

Review RAG 的公开深模块接口只有：

```python
ReviewRetriever.search(ReviewSearchRequest) -> ReviewSearchResult
```

分段、DuckDB 查询、Aspect join、BM25、Embedding 缓存、RRF、去重和 token 统计都隐藏在模块内部。Router 只决定什么时候调用工具，不直接操作 Parquet 或计算检索分数。

## 2. 27.1～27.3：数据、分段和泄漏审计

输入是 `data/processed/reviews.parquet` 中费城有效商家范围内的全量评论，不再只使用 5000 名用户的评论。

| 数据 | 数量 |
|---|---:|
| 原始 Review | 617,718 |
| Review 段 | 765,283 |
| 唯一 Review ID | 617,718 |
| 商家 | 3,973 |
| 评论段 Parquet 大小 | 264,076,388 bytes |

审计结果：

| 审计项 | 结果 |
|---|---:|
| 重复 `segment_id` | 0 |
| 非法字符范围 | 0 |
| 非空 Review 未生成段 | 0 |
| 隐藏 Review 标签 ID 覆盖率 | 100% |
| 隐藏标签全文 SHA256 匹配率 | 100% |
| cutoff 违规 | 0 |

运行时 Store 的查询条件由代码强制为：

```sql
business_id = locked_business_id
AND review_time < cutoff_time
```

Agent、Router 和 Review Store 都没有 Ground Truth 路径参数。只有离线审计器和第 21 步 Evaluator 可以读取隐藏标签。

## 3. 27.4～27.6：三路检索和 Top-5

### Aspect 路线

使用第 13 步高置信度 `ReviewAspectRecord`。只接受置信度不低于 0.85、商家匹配且早于 cutoff 的记录。Aspect 证据必须落在当前评论段的字符范围内。

### BM25 路线

对锁定商家在 cutoff 前的评论段现场计算 BM25。查询词同时包括用户原话和可见 Query 推断出的 Aspect 扩展词，因此中文问题也能检索英文 Yelp 评论。

### 本地 Embedding 路线

Embedding 不对 765,283 个评论段进行全量预计算。它只处理 Aspect/BM25 候选并补齐到每家最多 30 段，使用：

```text
D:\models\Qwen3-Embedding-0.6B
D:\anaconda3\python.exe
CUDA
```

向量保存在独立 SQLite 缓存中，不保存原始 Query 文本。整个第 27 步累计：

| 本地 Embedding 统计 | 数量 |
|---|---:|
| 缓存向量 | 5,161 |
| 实际编码 token | 665,214 |
| 逻辑请求 token | 2,901,196 |
| 缓存节省 token | 2,235,982 |
| Encoder 调用 | 471 |
| 外部 API 调用 | 0 |
| 计费 token | 0 |
| 费用 | 0 元 |

### RRF 与去重

三路排名通过 Reciprocal Rank Fusion 融合。最终结果按 `review_id` 去重，而不是只按 `segment_id` 去重，防止一篇长评论占满 Top-5。两个商家比较时采用商家间轮转合并，让双方都能获得证据位置。

## 4. 27.7～27.8：Agent 工具和 Router

真实工具为：

```text
SEARCH_BUSINESS_REVIEWS
```

工具输入只允许显式锁定的 `business_ids` 和 `top_k=5`。Registry、Harness Validator 和 Review Store 三层都会检查商家范围。

评论体验问题的正常轨迹变成：

```text
retrieve_business_reviews
→ SEARCH_BUSINESS_REVIEWS
→ return_grounded_answer / return_uncertain_answer
```

回答中的事实声明显式携带 `EvidenceReference(review_id, business_id)`。如果没有 cutoff 前证据、用户明确询问“能确定吗”，或者检索到正负冲突，Agent 会保守回答，而不是伪造确定事实。官网实时政策问题仍禁止用历史 Review 代替官方来源。

## 5. 27.9：Development 调参与冻结

调参只使用 104 道 Development Review 场景，Validation 标签没有进入策略选择。

| 策略 | Recall@1 | Recall@3 | Recall@5 | AvgRecall | Precision@5 |
|---|---:|---:|---:|---:|---:|
| BM25 | 10.16% | 28.67% | 36.54% | 25.12% | 13.46% |
| Aspect + BM25 | 49.69% | 75.75% | 84.99% | 70.14% | 40.38% |
| 三路均衡 | 49.91% | 75.24% | 81.80% | 68.98% | 38.65% |
| Embedding 偏重 | 49.91% | 74.92% | 82.52% | 69.12% | 38.85% |
| Aspect 偏重 | **50.15%** | **77.79%** | **86.76%** | **71.57%** | **41.73%** |
| 三路均衡，RRF K=20 | 47.02% | 73.58% | 80.98% | 67.19% | 37.88% |

冻结权重为：

```text
Aspect = 1.5
BM25 = 0.75
Embedding = 1.0
RRF K = 60
Top K = 5
```

这里必须说明：隐藏相关 Review 来自同一套第 13 步 Aspect 银标签，因此 Aspect 路线天然更贴近当前答案。该结果适合比较工程版本，但不能替代人工 Gold Test Set。

## 6. 27.10：Validation 与正式 500 场景

### 直接检索器与完整 Agent

直接检索器假设已经知道“现在应该调用 Review RAG”；完整 Agent 还必须先正确理解任务并选择工具。

| 运行方式 | Recall@1 | Recall@3 | Recall@5 |
|---|---:|---:|---:|
| Development 直接检索器，104 题 | 50.15% | 77.79% | 86.76% |
| Validation 直接检索器，26 题 | 49.20% | 74.08% | 82.81% |
| Development 完整 Agent，400 场景 | 41.52% | 57.59% | 63.67% |
| Validation 完整 Agent，100 场景 | 41.79% | 57.44% | 62.37% |
| 合并 500 场景 | 41.57% | 57.56% | 63.41% |

完整 Benchmark 中有 130 道题要求调用 Review RAG，Agent 实际调用 93 道，覆盖率 71.54%。漏掉的 37 道全部属于 `candidate_comparison`：现有 Rule 语义解析器没有把这些自然表达稳定识别为候选比较。这是语义理解/路由上限，不是 Review Retriever 上限。

### Step 26 与 Step 27

| 指标 | Step 26 | Step 27 | 差值 |
|---|---:|---:|---:|
| Review Recall@1 | 0.00% | 41.57% | +41.57pp |
| Review Recall@3 | 0.00% | 57.56% | +57.56pp |
| Review Recall@5 | 0.00% | 63.41% | +63.41pp |
| Evidence Precision@1 | 0.00% | 48.75% | +48.75pp |
| Evidence Precision@5 | 0.00% | 20.87% | +20.87pp |
| Grounded Answer Rate | 65.00% | 80.00% | +15.00pp |
| Citation Correctness | 30.00% | 68.64% | +38.64pp |
| Fallback Rate | 0.20% | 0.20% | 0.00pp |
| 平均 Agent 延迟 | 2,122.96 ms | 2,195.61 ms | +72.65 ms |
| P95 Agent 延迟 | 6,386.89 ms | 6,714.41 ms | +327.52 ms |

官方 `business_scope_isolation_rate` 为 92.63%，因为 Evaluator 会把“应该调用但没有返回证据”的场景记为 0。对实际返回的 465 条证据单独检查，465 条全部位于目标商家作用域内，实际检索商家隔离率为 100%。

正式 500 场景中 Review RAG 工具调用 93 次，缓存命中率 79.57%，实际新增本地 token 为 88,865，工具平均/P95 延迟为 304.62/1,053.64 ms。外部 API、计费 token 和费用均为 0。

## 7. 当前边界和下一步

第 27 步准确完成的是：

> 针对明确商家和当前问题，在严格时间边界内返回 Top-5 原始 Review 证据，并把 Review ID 带入 Agent 轨迹和回答引用。

它还不等于“能够完美总结全部评论”。后续建议：

1. 第 28 步实现 Evidence Aggregator，汇总支持、反驳、证据数量、冲突和时间新旧；
2. 后续 LLM 语义解析器解决 37 道候选比较漏路由，而不是把 Ground Truth 规则写进 Router；
3. 抽取 Development/Validation 的小规模人工 Gold Review 集，检查银标签漏标导致的 Precision 低估；
4. Benchmark 模式可预计算固定 Query 向量，在线模式继续保留懒加载接口；
5. 未来 Review RAG 可复用同一 Store 和 Embedding 缓存回答更细粒度的临时问题，而不必把“肉偏肥偏瘦”等非通用特征写进长期 Aspect 表。

## 8. 复现命令

```powershell
.\.venv\Scripts\python.exe scripts\build_review_rag.py
.\.venv\Scripts\python.exe scripts\audit_review_rag.py

.\.venv\Scripts\python.exe scripts\tune_review_rag.py `
  --model-path D:\models\Qwen3-Embedding-0.6B `
  --model-python D:\anaconda3\python.exe `
  --device cuda

.\.venv\Scripts\python.exe scripts\run_rule_agent_review_rag_benchmark.py `
  --split development `
  --embedding-model-path D:\models\Qwen3-Embedding-0.6B `
  --cross-encoder-model-path D:\model\Qwen3-Reranker-0.6B `
  --model-python D:\anaconda3\python.exe

.\.venv\Scripts\python.exe scripts\run_rule_agent_review_rag_benchmark.py `
  --split validation `
  --embedding-model-path D:\models\Qwen3-Embedding-0.6B `
  --cross-encoder-model-path D:\model\Qwen3-Reranker-0.6B `
  --model-python D:\anaconda3\python.exe

.\.venv\Scripts\python.exe scripts\compare_review_rag_agent.py
```
