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

25.1 至 25.3 先只调整静态文档、动态特征职责和语义候选数量，没有调用 DashScope。25.4 再把本地 Encoder 放到现有 `EmbeddingEncoder` seam 后面；Matcher、缓存、Agent 工具和融合策略没有因模型来源改变而重写。

## 25.4 本地 Qwen3 Encoder

当前实现已经把 `D:\models\Qwen3-Embedding-0.6B` 接到同一个 `EmbeddingEncoder` 接口后面。项目主进程继续使用 `.venv`；模型推理由一个持久化子进程调用已经具备 PyTorch、Transformers 和 CUDA 的 Python 环境。子进程只从本地目录加载模型，设置 `local_files_only=True`，不会下载模型，也不会调用 DashScope。

运行时通过以下环境变量指定机器相关路径，路径不会写死到代码或提交到 Git：

```powershell
$env:LOCAL_EMBEDDING_MODEL_PATH='D:\models\Qwen3-Embedding-0.6B'
$env:LOCAL_EMBEDDING_PYTHON='D:\anaconda3\python.exe'
$env:LOCAL_EMBEDDING_DEVICE='cuda'
```

Query 会使用 Qwen3 推荐的检索指令前缀，商家文档不加指令。模型输出最后一个有效 Token 的隐藏状态，经过 L2 归一化后得到 1024 维向量。Yelp 原始文本中的孤立旧编码字符会在 Encoder 边界被替换，避免 tokenizer 因损坏字符中断整个批次。

## 25.5 预计算、缓存与 Token 账本

先统计 Token、再生成向量：

```powershell
.\.venv\Scripts\python.exe scripts\dry_run_local_embeddings.py --model-path D:\models\Qwen3-Embedding-0.6B --python D:\anaconda3\python.exe
.\.venv\Scripts\python.exe scripts\precompute_local_business_embeddings.py --model-path D:\models\Qwen3-Embedding-0.6B --python D:\anaconda3\python.exe
.\.venv\Scripts\python.exe scripts\show_embedding_usage.py
```

本次实测共有 3973 个商家文档、752,688 个逻辑 Token，平均 189.45，最大 419，没有超过 512 Token 的文档。因为两组商家产生了完全相同的静态文本，最终缓存 3971 个唯一商家向量。首次预计算实际编码 752,528 Token；第二次相同运行实际编码 0 Token，全部从缓存读取。

`embedding_cache.sqlite3` 除向量外，还保存每次任务的用量事件：作用域、Provider、实际编码 Token、逻辑 Token、缓存节省 Token、Encoder 调用次数、外部 API 调用次数、截断数和推理延迟。原始 query 文本仍不落盘，只保存 Hash。因此可以区分“模型真正计算了多少”和“如果没有缓存原本需要计算多少”。

查询脚本还会只读统计旧 V1 缓存。V1 的 3785 条缓存记录归因了 1,432,216 个远程输入 Token；该数字不包含未成功写入缓存的失败请求，因此标记为历史缓存估计值，不能冒充供应商账单的精确总量。

## 25.6 当前真实联调结果

单场景 Agent 实测轨迹为：

```text
EXPAND_CANDIDATES
→ GET_HYBRID_RANKING
→ COMPUTE_EMBEDDING_MATCH（仅 Hybrid Top-30）
→ GET_BUSINESS_DETAILS
```

该场景正常完成，Embedding 仅新增 55 个 query Token，商家文档全部命中缓存；`api_calls=0`，费用为 0。模型进程由 `RuleAgentRuntime` 统一持有并在运行结束时关闭。只要显式传入 Embedding 配置但本地模型环境缺失，程序会立即报错，不会悄悄退化成第 24 步基线。

## 25.7 正式 500 场景结果

在冻结代码和配置后，按 `400 development + 100 validation` 完成了正式实验。Step 24 和 Step 25 使用同一批可见场景，隐藏标签只在离线评测器和对比脚本中读取。

| 指标 | Step 24 Hybrid | Step 25 Embedding | 变化 |
|---|---:|---:|---:|
| HR@1 | 0.43% | 2.61% | +2.17 个百分点 |
| HR@3 | 3.48% | 5.65% | +2.17 个百分点 |
| HR@5 | 4.78% | 7.83% | +3.04 个百分点 |
| MRR | 0.0368 | 0.0569 | +0.0201 |
| NDCG@5 | 0.0113 | 0.0206 | +0.0093 |
| valid_candidate_rate | 0.915% | 0.916% | 基本不变 |
| 平均延迟 | 2009.1 ms | 2239.5 ms | +230.4 ms |
| P95 延迟 | 6202.0 ms | 7179.0 ms | +977.1 ms |

在 286 个可以同时比较正确候选排名的推荐场景中，Embedding 让正确候选排名提高 25 次、下降 9 次、不变 159 次，另有 93 次正确候选没有进入当前候选排名。Top-30 内部顺序变化率为 99.30%，语义工具 466 次调用全部成功，没有发生 Embedding 回退。

这说明第 25 步确实改善了“已经召回的候选如何排序”：Top-1/3/5、MRR 和 NDCG@5 都提升。但 `valid_candidate_rate` 几乎不变，说明 Embedding 不能解决正确商家根本没有被召回的问题；召回瓶颈仍要在后续专门改进。与此同时，延迟增加约 230 ms，后续需要用更小候选、批量 query、量化或更快的本地模型继续优化。

完整产物：

- `runs/rule_agent_embedding_v2/development/`
- `runs/rule_agent_embedding_v2/validation/`
- `runs/rule_agent_embedding_v2/metrics.json`
- `runs/rule_agent_embedding_v2/reranking_comparison.json`
- `runs/rule_agent_embedding_v2/reranking_comparison.md`

可重复运行对比：

```powershell
.\.venv\Scripts\python.exe scripts\compare_rule_agent_reranking.py
```
