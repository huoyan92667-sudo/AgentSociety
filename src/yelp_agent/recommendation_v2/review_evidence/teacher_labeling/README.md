# 教师标注数据

这里负责从已有的Yelp评论候选关系中，为14种软偏好挑选真实评论，供教师模型GLM生成答案，并用于后续Qwen微调。

当前只保留完整的`v2`数据集。它已经包含第一批630条及新增1512条，共2142条；旧`v1`目录已在确认完整合并后删除，训练时不要再额外拼接旧数据。

相关程度、五档强度、14种特征的数值方向，以及标签如何进入商家打分，统一说明在[TRAINING_AND_SCORING_GUIDE.md](TRAINING_AND_SCORING_GUIDE.md)。

## 当前数据放在哪里

```text
data/teacher_dataset/v2/
├── candidates/                 2142条候选及追踪信息
├── labeled/                    2142条正式教师训练样本
├── audit/                      分布统计及需人工检查的清单
├── teacher_runs/glm53flash_v2/ 新增1512条的模型调用记录
├── candidate_manifest.json     候选数量和来源统计
└── reused_labels_manifest.json 第一批630条的复用结果
```

训练只读取`v2/labeled`。不要把`candidates`中的`selection_relevance`和`selection_strength`当成答案；它们只是旧程序为了挑选不同样本使用的粗略分桶。正式答案只能使用`model_output`。

## 数据是怎么生成的

### 1. 生成候选

- 候选来自已有的关键词命中和语义命中结果；
- 每种软偏好生成153条，共2142条；
- 候选包括五档样本、相关但难判断的样本和无关样本；
- 某档候选不足时允许用真实评论补位，但补位仍然不是答案；
- 商家编号、评论编号、星级、点赞数和召回信息只保留在候选外层，方便追溯；
- 真正发给教师和学生的`model_input`只包含特征定义、等级说明、特殊规则和评论文字。

```powershell
python -m yelp_agent.recommendation_v2.review_evidence.teacher_labeling candidates `
  --template-path 'src/yelp_agent/recommendation_v2/review_evidence/training_data/teacher_input_templates.v1.json' `
  --vocabulary-path 'C:\Users\29072\PycharmProjects\AgentSociety\configs\review_aspect_vocabulary.yaml' `
  --candidate-aspects-path 'C:\Users\29072\PycharmProjects\AgentSociety\src\yelp_agent\recommendation_v2\data\review_features\v1\candidate_aspects.parquet' `
  --candidate-reviews-path 'C:\Users\29072\PycharmProjects\AgentSociety\src\yelp_agent\recommendation_v2\data\review_features\v1\candidate_reviews.parquet' `
  --segments-path 'C:\Users\29072\PycharmProjects\AgentSociety\src\yelp_agent\recommendation_v2\data\review_index\v1\review_segments.parquet' `
  --output-root 'src/yelp_agent/recommendation_v2/data/teacher_dataset/v2' `
  --per-bucket 17
```

### 2. 调用教师模型

程序通过Claude Code调用`glm-5.3-flash`，每条只发送`model_input`，严格要求返回相关程度和五档强度。运行支持逐条保存、失败重试和断点续跑。

```powershell
python -m yelp_agent.recommendation_v2.review_evidence.teacher_labeling run `
  --dataset-root 'src/yelp_agent/recommendation_v2/data/teacher_dataset/v2' `
  --template-path 'src/yelp_agent/recommendation_v2/review_evidence/training_data/teacher_input_templates.v1.json' `
  --run-id 'glm53flash_v2' `
  --model 'glm-5.3-flash' `
  --max-workers 4 `
  --max-attempts 3 `
  --request-timeout-seconds 120
```

`v2`中的前630条是通过模型输入完全一致的方式，从第一批正式标签复用而来；匹配失败0条。旧目录删除后不再需要重复执行复用命令。新增1512条的真实调用结果：

- 总尝试次数：1525次；
- 中间格式错误：2次；
- 中间调用失败：11次；
- 重试后的最终缺失：0条；
- 输入词元：1,204,738；
- 输出词元：484,723；
- 思考词元：0；
- 接口记录费用：18.167813美元。

### 3. 生成审计结果

```powershell
python -m yelp_agent.recommendation_v2.review_evidence.teacher_labeling audit `
  --dataset-root 'src/yelp_agent/recommendation_v2/data/teacher_dataset/v2'
```

生成：

- `audit/label_distribution.json`：14种软偏好的相关程度和五档强度数量；
- `audit/disagreements.jsonl`：旧程序粗略分桶与教师输出不一致的评论，供人工抽查。

## 当前完整性和总体分布

- 14种软偏好，每种153条；
- 正式答案2142条；
- 缺失0条；
- 重复样本编号0条；
- 相关程度0：409条；
- 相关程度1：106条；
- 相关程度2：199条；
- 相关程度3：1428条；
- 强度0：413条；
- 强度1：402条；
- 强度2：193条；
- 强度3：340条；
- 强度4：385条；
- 无关且强度为空：409条。

详细到每种软偏好的分布保存在`data/teacher_dataset/v2/audit/label_distribution.json`。
