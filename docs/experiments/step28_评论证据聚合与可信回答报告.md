# 第 28 步：评论证据聚合与可信回答

## 1. 完成范围

第 28 步已在第 27 步 Business-scoped Review RAG 之后加入确定性 Evidence Aggregator：

```text
用户问题
→ 锁定 business_id 和 cutoff_time
→ SEARCH_BUSINESS_REVIEWS Top-5
→ Review 命中转换为 EvidenceAtom
→ AGGREGATE_REVIEW_EVIDENCE
→ 支持/反对/冲突/稀疏判断
→ RETURN_GROUNDED_ANSWER 或 RETURN_UNCERTAIN_ANSWER
```

本步骤不调用 LLM，不改变召回、Hybrid、Embedding 或 Cross-Encoder 排名。它只解释第 27 步已经找到的 Review 证据。

## 2. 模块 Interface

聚合模块对调用方只暴露：

```python
EvidenceAggregator.aggregate(
    EvidenceAggregationRequest
) -> EvidenceAssessment
```

Interface 接收可见 Query、任务类型、请求 Aspect、期望方向和第 27 步的 `ReviewSearchResult`。实现内部完成去重、时间衰减、证据立场、条件标签、冲突、置信度和回答策略。Agent 工具不重新读取评论库，因此不能扩大第 27 步已经锁定的商家作用域。

## 3. 证据原子

第 27 步的 `ReviewEvidenceHit` 增加：

- `user_id`：计算不同证据来源数量；
- `aspect_evidence`：保存 Aspect、sentiment、提取置信度和原始证据片段。

一个 Review 的同一 Aspect 只保留权重最高的原子，避免长评论切成多个 segment 后重复计票。证据立场分为：

```text
supports
contradicts
neutral
```

立场来自 Review Aspect sentiment，并相对于 Query 需要的正/负方向解释，不使用 Review 星级替代文本证据。

## 4. 权重与置信度

单条证据权重：

```text
weight = relevance × extraction_confidence × recency
recency = 2 ^ (-age_days / half_life_days)
```

整体置信度：

```text
confidence =
    mean_relevance
    × mean_recency
    × consistency
    × source_diversity
    × sample_strength
    × mean_extraction_confidence
```

其中：

- `consistency`：占优势方向的证据权重比例；
- `source_diversity`：不同用户数量，相同用户不会重复增加来源多样性；
- `sample_strength = n / (n + prior)`：防止一条评论产生高置信结论。

明确条件标签第一版支持 indoor、outdoor、weekday、weekend、lunch 和 dinner。只有评论明确出现对应词语时才记录，不推断未出现的条件。

## 5. 冲突与回答政策

当支持和反对两边都至少有一条方向性证据，并且较弱一方的权重占比达到冻结阈值时，标记：

```text
consensus = mixed
has_conflict = true
response_mode = uncertain
```

无方向性 Aspect 证据、用户明确询问证据是否足够且只有低置信证据时，同样返回不确定回答。普通体验问题可以使用一条低置信证据回答，但必须显示 `conclusion_requires_caveat=true`。

Review 永远保存：

```text
is_official_information = false
```

政策型问题继续由 Rule Router 保守返回，并建议用户通过官方渠道确认。Yelp Open Dataset 没有可靠电话字段，因此不会生成或伪造联系方式。

## 6. Development 策略选择

只使用 104 道 Development RAG 场景比较七套策略：默认、敏感冲突、宽松冲突、短期时间衰减、长期历史、高精度原子和多用户才允许回答。

选择目标：

```text
0.4 × 回答策略准确率
+ 0.3 × 冲突检测准确率
+ 0.2 × 证据立场准确率
+ 0.1 × 引用 Precision
```

最终冻结：

```text
policy_version = dev-sensitive-conflict-v1
recency_half_life_days = 730
conflict_minority_mass_share = 0.15
consensus_mass_share = 0.67
minimum_extraction_confidence = 0.80
maximum_citations_per_aspect = 3
```

Development 指标：

| 指标 | 结果 |
|---|---:|
| Selection Objective | 81.05% |
| Response Policy Accuracy | 73.08% |
| Conflict Accuracy | 87.50% |
| Evidence Stance Accuracy | 85.90% |
| Citation Precision | 83.89% |
| Citation Recall | 80.19% |

Validation 没有用于选阈值。

## 7. 冻结 Validation 聚合评测

26 道 Validation RAG 场景：

| 指标 | 结果 |
|---|---:|
| Response Policy Accuracy | 76.92% |
| Conflict Accuracy | 92.31% |
| Evidence Stance Accuracy | 100.00% |
| Citation Precision | 83.33% |
| Citation Recall | 77.36% |
| Selection Objective | 86.79% |

Validation 中 12 道题应该谨慎回答，其中 41.67% 被当前聚合规则判为可以基于证据回答。这是现有确定性策略的重要限制，结果已冻结报告，没有使用 Validation 再调整阈值。

## 8. 正式 500 场景结果

第 28 步只可能影响冻结契约中要求 `retrieve_business_reviews` 的 130 道场景。因此正式实验重新运行这 130 道，并复用第 27 步其余 370 道确定性不受影响的轨迹，然后重新统一评测完整 500 道场景。

| 指标 | Step 27 | Step 28 | 差值 |
|---|---:|---:|---:|
| Review Recall@1 | 41.57% | 41.57% | 0.00pp |
| Review Recall@3 | 57.56% | 57.56% | 0.00pp |
| Review Recall@5 | 63.41% | 63.41% | 0.00pp |
| Grounded Answer Rate | 80.00% | 75.50% | -4.50pp |
| Unsupported Claim Rate | 28.87% | 27.17% | -1.70pp |
| Citation Correctness | 68.64% | 64.91% | -3.73pp |
| Evidence Recency Reporting | 40.00% | 71.54% | +31.54pp |
| Conflict Detection Accuracy | 83.33% | 93.33% | +10.00pp |
| Official-policy Caution | 100.00% | 100.00% | 0.00pp |
| Fallback Rate | 0.20% | 0.20% | 0.00pp |
| Mean Latency | 2195.61 ms | 2185.08 ms | -10.53 ms |
| P95 Latency | 6714.41 ms | 6714.41 ms | 0.00 ms |

Review Recall 不变化是正确结果：Aggregator 不重新检索评论。主要收益是更准确地发现评论冲突、报告评论日期，并小幅降低无证据 Claim。Grounded Answer Rate 降低来自更保守的结束策略。

Citation Correctness 降低的主要原因是 Aggregator 为支持和反对两边最多保留三条引用，而隐藏银标签只覆盖部分 Review；198 个聚合引用中有一些实际相关但不在银标签中。由于 Validation 已运行，不能回头使用该结果修改引用数量。

## 9. Agent 覆盖和成本

- 应使用 Review 的场景：130；
- 实际搜索并聚合：93；
- Aggregator 覆盖率：71.54%；
- 未聚合：37，全部为 `candidate_comparison`；
- 原因：第 18/19 步规则语义解析器没有识别这些比较请求，不是 RAG 或 Aggregator 失败；
- 聚合 Tool Calls：93；
- 结构化聚合 Claim：117；
- Review 引用：198；
- 聚合工具平均/P95 延迟：7.39/13.26 ms；
- 外部 API：0；
- 计费 Token：0；
- 成本：0 元。

## 10. 测试边界

已覆盖：

- 多用户一致证据；
- 支持/反对冲突；
- 单条稀疏证据；
- Query 负向要求反转立场；
- 条件标签；
- 政策信息永远不是官方事实；
- Aggregator 不能扩大 Review 搜索作用域；
- cutoff 后证据不能进入结果；
- 相同输入字节确定性；
- Router 按 Search → Aggregate → Answer 执行；
- 冲突场景返回不确定回答；
- 全量旧测试不回归。

## 11. 复现命令

```powershell
.\.venv\Scripts\python.exe scripts\tune_evidence_aggregator.py `
  --model-path D:\models\Qwen3-Embedding-0.6B `
  --model-python D:\anaconda3\python.exe --device cuda

.\.venv\Scripts\python.exe scripts\evaluate_evidence_aggregator.py `
  --split validation `
  --model-path D:\models\Qwen3-Embedding-0.6B `
  --model-python D:\anaconda3\python.exe --device cuda

.\.venv\Scripts\python.exe scripts\run_rule_agent_evidence_benchmark.py `
  --split all --affected-only `
  --embedding-model-path D:\models\Qwen3-Embedding-0.6B `
  --cross-encoder-model-path D:\model\Qwen3-Reranker-0.6B `
  --model-python D:\anaconda3\python.exe --device cuda
```

## 12. 当前准确定位

第 28 步完成后，Agent 已经能够把同一商家的多条 Review 证据转换成可追溯的共识、冲突、稀疏度、时间和置信度结论，并按照体验型/政策型规则安全回答。

它仍然不能宣称已经拥有完美的自然语言理解或自然回答能力。第 29 步将加入受控 LLM 语义与回答工具；第 30 步才决定语义证据是否允许在 Rank Protection 下小范围改变排名。
