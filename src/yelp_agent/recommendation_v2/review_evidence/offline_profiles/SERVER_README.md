# 4090 服务器评论判断说明

本目录由本地准备程序完整生成。上传时请复制整个目录，不要只复制
`model_inputs.jsonl`，因为运行程序还会校验输入文件，并读取商家清单、14项定义
和聚合所需的统计信息。

## 输入是什么

- `model_inputs.jsonl`：每一行是“某家商店的一条评论 × 某个软偏好”。其中
  `messages` 就是微调模型真实收到的输入，与训练数据格式相同。
- `selection.json`：固定选出的10家牛排类别商家。
- `model_contract.v1.json`：14项软偏好的定义、0—4级含义和输出限制。
- `prepare_manifest.json`：本地召回数量、耗时、文件校验值和每家每项是否已经
  取满高低端各15条。

星级、点赞数、时间、商家编号不会塞进模型消息。模型只看特征定义和评论文字；
这些事实字段只在模型判断完成后参与证据聚合。

## 服务器环境

建议使用与微调、评测时相同的 Python 环境。至少需要：

```bash
pip install torch transformers peft accelerate bitsandbytes
```

## 运行

假设整个数据目录上传到：

```text
/root/autodl-tmp/steakhouses_top10_v1
```

基础模型与微调权重分别位于：

```text
/root/autodl-tmp/Qwen3-4B-Instruct-2507
/root/autodl-tmp/qwen3_training/outputs/baseline
```

运行：

```bash
cd /root/autodl-tmp/steakhouses_top10_v1
python run_server_judge.py \
  --input-dir . \
  --base-model /root/autodl-tmp/Qwen3-4B-Instruct-2507 \
  --adapter /root/autodl-tmp/qwen3_training/outputs/baseline \
  --batch-size 8
```

如果显存有余量，可把 `--batch-size` 调到16；显存不足则降到4。程序使用4位
加载，默认只读取本地基础模型，不会偷偷联网下载。

## 中断与继续

每完成一个批次，程序立即追加写入并刷新：

```text
server_output/predictions.partial.jsonl
```

程序意外中断后，原命令重新运行即可。已经得到合法输出的样本会跳过；上次模型
输出格式非法的样本会再尝试一次。不要删除该断点文件。

## 最终产出

```text
server_output/
├── predictions.partial.jsonl     # 可断点续跑的原始判断记录
├── model_judgments.jsonl         # 按原始输入顺序整理后的最终判断
├── invalid_outputs.json          # 仍不符合格式的模型输出
├── business_aspect_profiles.json # 10家 × 14项的程度、充分程度、争议与证据
└── judge_manifest.json           # 模型、数量、词元、耗时和吞吐量
```

`business_aspect_profiles.json` 中每个软偏好包含：程度、通俗档位、证据充分程度、
争议程度、高程度证据、低程度证据和中间证据。当前模型没有判断“条件性”的独立
输出，因此条件性证据明确标记为尚不可用，不会拿中间程度评论冒充。

`useful` 只提供最高20%的轻微加权；`cool` 暂不参与。相同用户针对同一商家同一
偏好的多条有效评论，只保留证据权重最高的一条，避免重复用户放大结果。
