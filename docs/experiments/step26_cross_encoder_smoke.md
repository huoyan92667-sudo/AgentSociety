# 第 26.1 步：本地 Qwen3 Reranker 冒烟测试

## 目标

本步只验证 `D:\model\Qwen3-Reranker-0.6B` 能否在当前机器本地运行，以及它是否适合接在 Step25 Embedding 后面。本步不修改 Agent、Router、排序融合或正式500场景结果。

## 环境

- GPU：NVIDIA GeForce RTX 5070 Laptop，8151 MB；
- PyTorch：2.10.0+cu128；
- Transformers：5.3.0；
- 推理类型：BF16；
- 最大输入长度：512；
- 外部 API 调用：0。

模型采用官方 `yes/no` logits 计算方法。`yes` token ID 为 9693，`no` token ID 为 2152，最终匹配分是二者 softmax 后的 `yes` 概率。

## 语义检查

同一个“安静、浪漫、可预订的牛排馆”请求得到：

| 文档 | 匹配分 |
|---|---:|
| 明显匹配的浪漫安静牛排馆 | 0.9991 |
| 明显不匹配的嘈杂体育酒吧 | 0.0025 |
| 信息不足且类别不符的咖啡馆 | 0.0001 |

分数方向正确，可以进入下一阶段。

## 延迟和显存

| Batch | 延迟 | 吞吐 | 峰值显存 |
|---:|---:|---:|---:|
| 1 | 102.0 ms | 9.8 对/秒 | 1172 MB |
| 4 | 55.3 ms | 72.3 对/秒 | 1224 MB |
| 8 | 62.9 ms | 127.2 对/秒 | 1298 MB |
| 20 | 157.2 ms | 127.3 对/秒 | 1525 MB |

首次三对推理包含 CUDA 预热，用时约386.2 ms；预热后的20对推理约157.2 ms。

Reranker 单独驻留约1144 MB。再同时加载 `Qwen3-Embedding-0.6B` 后，两模型合计约2281 MB，显卡仍剩约4667 MB。因此当前硬件支持两个0.6B模型同时驻留，不需要每个场景重复加载模型。

## 结论

第26步第一版可以采用：

```text
Hybrid V2 全候选排序
→ Embedding 重排 Top-30
→ Qwen3 Reranker 精排 Top-20
→ 返回 Top-3
```

建议本地 Worker 默认 `batch_size=8`，处理 Top-20 时分成 `8+8+4` 三批。虽然一次处理20对也能运行，但较小 batch 给真实长商家文档保留更多显存余量。

## 复现

```powershell
D:\anaconda3\python.exe scripts\smoke_local_cross_encoder.py `
  --model-path D:\model\Qwen3-Reranker-0.6B `
  --embedding-model-path D:\models\Qwen3-Embedding-0.6B
```

结构化结果写入 `runs/cross_encoder_v1/smoke.json`。
