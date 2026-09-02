# 500家餐饮商户：4090服务器运行说明

把整个 `restaurants_diverse_500_v1` 目录上传到：

```text
/root/autodl-tmp/restaurants_diverse_500_v1
```

不要只上传 `input_shards`。服务器程序还需要根目录中的商家清单、14项定义、
模型合同和准备清单。

## 运行命令

```bash
python /root/autodl-tmp/restaurants_diverse_500_v1/run_server_judge.py \
  --input-dir /root/autodl-tmp/restaurants_diverse_500_v1 \
  --base-model /root/autodl-tmp/hf_cache/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554 \
  --adapter /root/autodl-tmp/qwen3_training/outputs/baseline \
  --output-dir /root/autodl-tmp/restaurants_diverse_500_v1/server_output \
  --batch-size 32 \
  --cutoff-len 1024 \
  --max-new-tokens 32
```

这与10家试跑命令的参数相同，只替换了输入和输出目录。

## 分片不会重复加载模型

根目录中共有5个输入分片，每个分片约100家商户。程序只加载一次基础模型和
微调权重，然后依次处理5个分片。分片的作用是限制内存并支持断点续跑，不会
把模型重新加载5次。

每完成一个模型批次，结果都会立即写入：

```text
server_output/predictions.partial.jsonl
```

服务器中断后，重新执行同一条命令即可。已经得到合法 `relevance` 和
`strength` 的样本会跳过；格式不合法的样本会再次尝试。

## 模型输入与输出

服务器程序根据 `model_contract.v1.json` 和每条记录中的 `model_review_text`
还原训练时使用的模型消息。微调模型仍然只输出：

```json
{"relevance": 3, "strength": 4}
```

评论编号、商家编号、用户编号、时间和 useful 不交给模型判断，只在模型输出后
由程序接回原记录。

## 最终产出

```text
server_output/
├── predictions.partial.jsonl
├── model_judgments.jsonl
├── invalid_outputs.json
├── profile_shards/
│   ├── shard_0001.json
│   └── ...
├── business_aspect_profiles.json
└── judge_manifest.json
```

每家商户的每项软偏好会包含：

- 程度和通俗档位；
- 证据充分程度；
- 争议程度；
- 强相关评论数量和不同用户数量；
- `usable_for_ranking`；
- 不能参与排序时的具体原因；
- 高程度、低程度和中间程度代表证据。

只有同时满足下面条件时，`usable_for_ranking` 才为 `true`：

1. 程度不是空值；
2. 至少3条评论的 `relevance` 为2或3；
3. 强相关评论至少来自3个不同用户；
4. 证据充分程度大于0.3。

未通过时原始程度和评论仍然保存，但 `ranking_degree` 为 `null`，后续排序必须按
未知处理。
