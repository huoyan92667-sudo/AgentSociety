# 第 26 步：本地 Cross-Encoder 精排与正式 500 场景实验

## 1. 本步完成了什么

第 26 步把本地 `Qwen3-Reranker-0.6B` 接入受控 Agent。最终推荐链路固定为：

```text
五路召回
  → Hybrid V2 对全部候选排序
  → Qwen3 Embedding 重排前 30
  → Qwen3 Cross-Encoder 重排前 20
  → 获取前 5 家商家详情
  → 返回 Top-5
```

Cross-Encoder 的作用是同时阅读“当前用户问题”和“一个商家静态文档”，直接判断两者是否匹配。它比 Embedding 的双塔余弦相似度更细，但计算更贵，因此只处理 Step 25 排名的前缀。

本步使用的模型路径为 `D:\model\Qwen3-Reranker-0.6B`，使用 `D:\anaconda3\python.exe` 中的 PyTorch/Transformers 和 CUDA。没有调用任何外部 API，也没有产生计费 token。

## 2. 模块边界

Cross-Encoder 被封装为独立 Module：

- `PairScorer` 是模型推理 Interface；
- `LocalQwenCrossEncoder` 是本地 Qwen3 Implementation/Adapter；
- `CachedCrossEncoderReranker` 负责静态商家文档、缓存、排序和用量统计；
- `ComputeCrossEncoderMatchTool` 是 Agent 工具 Adapter；
- Router 只决定何时调用工具，不直接加载模型或计算分数。

这个 seam 使以后替换 ONNX、量化模型、vLLM 或其他 reranker 时，不需要修改 Agent Harness、工具协议和评测器。

## 3. 输入数据与时间安全

模型输入只包含：

- 当前可见用户问句；
- Step 25 最终排名前 20 家；
- 商家名称、完整类别和静态 Yelp attributes。

模型不读取：

- ground truth；
- validation 标签；
- 未来评论；
- 动态评分、热度和用户历史文本；
- 第 27 步才会加入的 Review RAG 证据。

评分、热度、用户画像和协同信号仍由时间安全的 Hybrid V2 处理。这样避免同一家商家因为 cutoff 不同反复生成静态语义文档。

## 4. 本地推理、缓存和 token

本地 worker 使用官方 Qwen3 Reranker 的 yes/no logit 评分方式：

```text
P(match) = softmax(logit_no, logit_yes)[yes]
```

固定设置：

- batch size：8；
- 最大长度：512 token；
- temperature：不适用，模型不生成文本；
- 外部 API 调用：0；
- 外部费用：0 元。

缓存键包含模型、指令哈希、文档版本、问题哈希、商家文档哈希和最大长度。SQLite 中不保存原始问题或原始商家文档。正式完成时缓存有 9,691 个配对分数，文件约 4.96 MB。

## 5. 参数如何冻结

只使用 400 条 development 场景调参，100 条 validation 在参数冻结后才评测。开发集网格为：

- Cross-Encoder 前缀：5、10、20；
- 融合权重 β：0.0 到 1.0，步长 0.1；
- 主指标：HR@5；
- 并列顺序：MRR、HR@1、更小前缀、更小 β。

178 个 development 场景拥有可评测候选排名。最终冻结：

```json
{
  "candidate_limit": 20,
  "fusion_beta": 1.0,
  "display_limit": 5
}
```

同一开发口径下，Step 25 的 HR@5 为 7.30%，冻结设置为 10.67%。完整网格保存在 `runs/cross_encoder_v1/development_tuning.json`。

β=1.0 表示在前 20 内完全采用 Cross-Encoder 顺序；第 21 名以后仍严格保留 Step 25 顺序，因此 Cross-Encoder 不会改变召回集合。

## 6. 正式 500 场景结果

| 指标 | Step 24 Hybrid | Step 25 Embedding | Step 26 Cross-Encoder | Step26 - Step25 |
|---|---:|---:|---:|---:|
| HR@1 | 0.43% | 2.61% | 5.22% | +2.61pp |
| HR@3 | 3.48% | 5.65% | 8.70% | +3.04pp |
| HR@5 | 4.78% | 7.83% | 10.00% | +2.17pp |
| MRR | 3.68% | 5.69% | 8.31% | +2.62pp |
| NDCG@5 | 1.13% | 2.06% | 2.93% | +0.87pp |
| Fallback Rate | 0.20% | 0.20% | 0.20% | 0.00pp |

分割结果：

- development：HR@1 5.98%，HR@3 9.24%，HR@5 10.33%，MRR 8.82%；
- validation：HR@1 2.17%，HR@3 6.52%，HR@5 8.70%，MRR 6.27%。

在 224 个可直接比较排名的场景中：

- 223 个场景的 Top-20 顺序发生变化；
- 正确商家排名提高 22 个、下降 11 个、不变 160 个；
- 仍有 31 个场景的正确商家不在候选集合中。

正式结果保存在：

- `runs/rule_agent_cross_encoder_v1/scenario_runs.jsonl`；
- `runs/rule_agent_cross_encoder_v1/metrics.json`；
- `runs/rule_agent_cross_encoder_v1/reranking_comparison.json`；
- `runs/rule_agent_cross_encoder_v1/reranking_comparison.md`。

## 7. 回退和预算修复

失败顺序固定为：

```text
Cross-Encoder 失败 → 使用 Step 25 Embedding 排名
Embedding 失败     → 使用 Hybrid V2 排名
Hybrid 已有结果    → 返回其前 5 家
```

正式实验首次发现 Harness 把“会话累计 token”错误地当成“当前轮 token”检查，导致 60 个正常多轮场景被误回退。修复后：

- `input_tokens/output_tokens` 继续保存会话累计值，用于报告；
- `turn_input_tokens/turn_output_tokens` 每轮清零，只用于本轮预算；
- 只针对受影响的 60 个场景重跑，再验证 ID 子集并重新评测完整 500 场景；
- 最终只剩 1 个原有的 `router_requested_fallback`，Fallback Rate 为 0.20%。

## 8. 当前结论和限制

第 26 步证明 Cross-Encoder 对“已召回候选的前排排序”有明显价值，尤其 HR@1 从 Step 25 的 2.61% 提升到 5.22%。

但 `valid_candidate_rate` 基本不变，这说明 Cross-Encoder 无法找回根本没有进入候选集的正确商家。当前主要上限仍是召回覆盖率。第 27 步 Review RAG 应主要解决“为什么推荐、安静/服务/场合等细节有没有评论证据”，不能被描述为召回问题已经解决。

另一个工程瓶颈是每次正式运行都要重新构建全量时间索引。后续可把静态索引持久化，或把完整 Agent Runtime 做成长驻服务；这比继续压缩 Top-5 展示数量更合理。
