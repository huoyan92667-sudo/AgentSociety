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

## 本地评测原始Qwen3 4B

`evaluate_qwen3_baseline.py` 用于测试微调前的原始模型。它默认：

- 从 `D:\model\Qwen3-4B-Instruct-2507` 读取模型；
- 不加载任何LoRA增量文件；
- 使用本目录生成的validation和test；
- 以NF4四位方式加载，并将批量大小设为1；
- 使用与服务器微调模型评测相同的生成方式和指标口径。

格式正确率和内容正确率分开计算。比如原始模型返回
`{"relevance":"3","strength":"0"}` 时，字符串数字导致 `Schema valid` 判错；
但数值含义明确，因此相关程度、强度和联合正确率会按照3和0计算。原始文字和
转换后的计分值都会保存在逐条预测文件中，不会掩盖格式问题。

先用8条validation确认环境和显存：

```powershell
& "D:\anaconda3\python.exe" "src\yelp_agent\recommendation_v2\review_evidence\student_training\evaluate_qwen3_baseline.py" --split validation --max-samples 8
```

试跑成功后评测完整validation和test：

```powershell
& "D:\anaconda3\python.exe" "src\yelp_agent\recommendation_v2\review_evidence\student_training\evaluate_qwen3_baseline.py"
```

这里故意直接运行文件，不使用 `python -m yelp_agent...`。原因是本地显卡环境只需要
PyTorch和Transformers；模块方式会提前加载整个推荐系统及其数据库、配置和数据校验依赖，
与基础模型评测无关，也容易因两个Python环境的依赖版本不同而失败。

默认结果保存在：

```text
recommendation_v2/data/student_training/qwen3_4b_teacher_v2/v1/evaluation/original_model/
```
