# Qwen3 学生模型训练数据

这个模块把 `teacher_dataset/v2` 转换成可直接交给 LLaMA-Factory 的本地数据。

它负责四件事：

1. 用 `sample_id` 对齐候选清单和正式教师标签，找回每条样本的真实评论编号；
2. 按 `review_id` 把数据拆成训练、验证、测试三份，防止同一评论跨集合泄漏；
3. 只把教师和学生共同看到的六项输入写进模型数据；
4. 将追溯编号、分布统计和文件校验值单独保存，不交给模型。

## 重新生成

从项目根目录执行：

```powershell
python -m yelp_agent.recommendation_v2.review_evidence.student_training build
```

只检查已有结果：

```powershell
python -m yelp_agent.recommendation_v2.review_evidence.student_training validate
```

默认输出到：

```text
recommendation_v2/data/student_training/qwen3_4b_teacher_v2/v1/
```

其中：

- `train.jsonl`：正式训练数据；
- `validation.jsonl`：训练过程中选择版本使用，不用于更新参数；
- `test.jsonl`：最终对比使用，训练期间不能读取；
- `split_index.jsonl`：每一行对应的评论编号和样本编号，只用于追溯；
- `dataset_info.json`：LLaMA-Factory数据登记；
- `distribution.json`：逐集合、逐特征、逐档位数量；
- `manifest.json`：来源、切分规则、字符长度、文件校验值。

测试集目前仍然是教师标签，不等于人工金标准。正式报告最终准确率前，应当先人工复核测试集。
