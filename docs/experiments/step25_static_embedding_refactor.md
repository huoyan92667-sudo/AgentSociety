# 第 25 步修订：静态商家 Embedding 与 Top-30 语义精排

## 为什么要修订

第 25 步第一版把截止时间前的评分、热度和 Aspect 汇总拼入商家文本，并对 Hybrid Top-100 进行远程 Embedding。真实联调证明功能链路可以运行，但这种设计会让同一家商户因为任务截止时间不同而反复生成向量，并且单个推荐任务的输入文本过多。

第一版正式 Benchmark 已停止。旧缓存保留在 `data/features/semantic_embeddings/v1`，仅作为问题复盘证据，不进入 V2 实验，也不会被删除或覆盖。

## 25.1 静态商家文档

V2 的 `build_business_document` 只接受 `BusinessRecord`，不能接收包含动态证据的 `BusinessProfileV1`。文档只包含：

- 商家名称；
- 完整类别；
- 城市和州；
- Yelp 静态 attributes。

明确不包含：

- 截止时间、评分、评论数量、质量分和热度；
- Aspect 正负比例、证据数量、置信度和冲突状态；
- 用户画像、用户历史和用户当前位置；
- 原始评论文本。

静态文档的 `source_id` 是 `business_id`，版本冻结为 `business-static-semantic-v2.0.0`。同一家商户在不同任务截止时间下会得到相同文本 Hash，因此只需生成一次向量。

## 25.2 动态信息继续走结构化排序

动态信息没有被删除。它仍由已有的点时安全模块计算，并进入 Hybrid V2/LambdaMART：

- `quality_score`、时点评分证据和热度；
- Item-KNN 正负协同信号；
- 用户类别和 Aspect 偏好；
- 商家 Aspect 证据覆盖、可靠性和冲突；
- 距离与硬约束过滤；
- 多路召回分数。

因此新的职责分工是：

```text
静态文字语义                     动态、数值和个性化信息
Business Static Embedding        Hybrid V2 / LambdaMART
             \                    /
              \                  /
               Top-30 保守融合排序
```

## 25.3 Top-30 语义精排

Agent 先执行五路召回、硬约束和 Hybrid V2，再只把 Hybrid 前 30 家交给 Embedding Matcher。Embedding 只允许重排这 30 家，未打语义分的第 31 名以后保持原 Hybrid 相对顺序不变。

生产配置同时恢复 12,000 Token 的单轮安全上限，不再沿用第一次联调临时提高的 128,000 上限。

## V2 版本与缓存隔离

```text
embedding_version: 2.0.0
business_document_version: business-static-semantic-v2.0.0
candidate_limit: 30
cache_relative_path: data/features/semantic_embeddings/v2
```

新旧缓存目录隔离，避免把动态 V1 商家向量误当作静态 V2 向量命中。

## 验证边界

本次修订只调整静态文档、动态特征职责和语义候选数量，不调用 DashScope，也不部署本地模型。后续本地 Encoder 可以放到现有 `EmbeddingEncoder` seam 后面；Matcher、缓存、Agent 工具和融合策略无需因模型来源改变而重写。
