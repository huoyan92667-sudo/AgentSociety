# Qwen3 4B 评论特征训练数据 v1

这份数据由 `teacher_dataset/v2` 的 2,142 条正式教师标签生成，可以直接登记到 LLaMA-Factory。

## 三份模型数据

| 文件 | 样本数 | 用途 |
|---|---:|---|
| `train.jsonl` | 1,723 | 更新模型参数 |
| `validation.jsonl` | 220 | 训练期间比较不同保存版本 |
| `test.jsonl` | 199 | 训练结束后做一次最终评测 |

实际数量没有机械地切成整数80%、10%、10%，因为同一评论可能对应多个特征。程序优先保证同一个 `review_id` 永远只进入一份数据，因此不存在评论跨集合泄漏。

每条模型数据只有三条消息：

1. 教师标注时使用的固定任务说明；
2. 六项评论判断输入；
3. 教师输出的 `relevance` 和 `strength`。

评论编号、商家编号和样本编号只保存在 `split_index.jsonl`，不会出现在模型输入中。

## 配套文件

- `dataset_info.json`：LLaMA-Factory登记三个数据集；
- `distribution.json`：每份数据中14种特征和各档位的真实数量；
- `manifest.json`：构建来源、切分参数、字符长度和文件校验值；
- `split_index.jsonl`：用于从模型数据追溯到原始教师样本。

当前 `test.jsonl` 仍然使用教师标签，并没有全部经过人工复核。因此它可以用于第一轮基线对比，但在形成正式准确率结论前需要人工复核。

训练配置模板位于：

```text
review_evidence/student_training/configs/
```

先运行 `smoke_test.yaml`，成功后再运行 `train_qlora.yaml`。模型下载完成后还要用 Qwen3 的真实分词器统计一次长度；当前清单里的长度只是字符数，不能冒充词元数。
