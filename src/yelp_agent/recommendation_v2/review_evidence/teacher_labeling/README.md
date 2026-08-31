# 教师标注数据

这里负责从已有的 Yelp 评论候选关系中，为 14 个软偏好挑选真实评论，供后续教师模型（GLM）和学生模型（Qwen）使用。

相关程度、五档强度、14种特征的数值方向，以及标签如何进入商家打分，统一说明在 [TRAINING_AND_SCORING_GUIDE.md](TRAINING_AND_SCORING_GUIDE.md)。

## 现在做了什么

- 每种软偏好优先挑选 5 条低到高的程度样本，共 5 个程度桶。
- 另外挑选一小批“相关但说不清”和“完全无关”的评论，帮助模型学会相关性判断。
- 候选来自已有的关键词命中和语义命中结果，不重新把全部评论交给模型。
- 如果某个特征缺少某一档的候选，会按语义分最接近的真实评论补位，并写入 `selection_fallback=true`；这只是为了覆盖五档输入，不能当作标签。
- 通过评论编号读取对应片段；商家编号、评论编号、星级、点赞数和命中词只保存在外层，方便追溯。
- `model_input` 是真正发给教师和学生的内容，两者完全相同，只包含：特征定义、两个等级说明、特殊规则和评论文字。

## 第一步：生成候选

```powershell
python -m yelp_agent.recommendation_v2.review_evidence.teacher_labeling candidates `
  --template-path 'src/yelp_agent/recommendation_v2/review_evidence/training_data/teacher_input_templates.v1.json' `
  --vocabulary-path 'C:\Users\29072\PycharmProjects\AgentSociety\configs\review_aspect_vocabulary.yaml' `
  --candidate-aspects-path 'C:\Users\29072\PycharmProjects\AgentSociety\src\yelp_agent\recommendation_v2\data\review_features\v1\candidate_aspects.parquet' `
  --candidate-reviews-path 'C:\Users\29072\PycharmProjects\AgentSociety\src\yelp_agent\recommendation_v2\data\review_features\v1\candidate_reviews.parquet' `
  --segments-path 'C:\Users\29072\PycharmProjects\AgentSociety\src\yelp_agent\recommendation_v2\data\review_index\v1\review_segments.parquet' `
  --output-root 'src/yelp_agent/recommendation_v2/data/teacher_dataset/v1'
```

生成结果：

- `data/teacher_dataset/v1/candidates/{aspect}.jsonl`：按特征分片的候选评论。
- `data/teacher_dataset/v1/candidate_manifest.json`：数量、来源关系数量和实际补齐情况。

候选文件中的 `selection_strength` 和 `selection_relevance` 只是“挑选时的目标桶”，不是模型答案；真正的答案要等教师模型根据 `model_input` 判断后写入。

## 第二步：教师试跑28条

下面的命令会让Claude Code调用配置好的`glm-5.3-flash`。每种软偏好均匀取2条，共28条；每次模型只接收模板中的共同要求和候选记录里的`model_input`。

```powershell
python -m yelp_agent.recommendation_v2.review_evidence.teacher_labeling run `
  --dataset-root 'src/yelp_agent/recommendation_v2/data/teacher_dataset/v1' `
  --template-path 'src/yelp_agent/recommendation_v2/review_evidence/training_data/teacher_input_templates.v1.json' `
  --run-id 'glm53flash_v1' `
  --model 'glm-5.3-flash' `
  --max-workers 4 `
  --max-attempts 3 `
  --request-timeout-seconds 120 `
  --limit-per-aspect 2
```

本轮的原始返回、失败记录、进度和统计写入：

```text
data/teacher_dataset/v1/teacher_runs/glm53flash_v1/
```

通过校验的正式标签按软偏好写入：

```text
data/teacher_dataset/v1/labeled/{aspect}.jsonl
```

## 第三步：完成全部630条

试跑确认后使用同一个`run-id`再次运行，但去掉`--limit-per-aspect`。程序会读取已经保存的正式标签，只调用剩余评论。

```powershell
python -m yelp_agent.recommendation_v2.review_evidence.teacher_labeling run `
  --dataset-root 'src/yelp_agent/recommendation_v2/data/teacher_dataset/v1' `
  --template-path 'src/yelp_agent/recommendation_v2/review_evidence/training_data/teacher_input_templates.v1.json' `
  --run-id 'glm53flash_v1' `
  --model 'glm-5.3-flash' `
  --max-workers 4 `
  --max-attempts 3 `
  --request-timeout-seconds 120
```

## 第四步：生成不一致清单

```powershell
python -m yelp_agent.recommendation_v2.review_evidence.teacher_labeling audit `
  --dataset-root 'src/yelp_agent/recommendation_v2/data/teacher_dataset/v1'
```

生成：

- `audit/label_distribution.json`：14种软偏好的教师标签数量和缺失情况。
- `audit/disagreements.jsonl`：挑选桶与教师输出不一致、需要人工检查的真实评论。

## 第二批：扩充到2142条

第二批不覆盖第一批。`v1`继续保留630条原始候选、教师结果和调用记录；`v2`包含14种软偏好各153条，共2142条。

### 1. 生成扩充候选

候选组成仍然包括五档候选、相关但难判断的候选和无关候选。这里的档位只是抽样目标，不能作为训练答案。

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

`per-bucket=17`最终得到每种软偏好153条：五个档位各17条、相关但难判断的候选51条、无关候选17条。

### 2. 复用第一批630条正式教师答案

复用时只比较教师和学生真正看到的`model_input`。只有输入完全相同才复用；不按评论编号、旧档位或相似度猜测标签。

```powershell
python -m yelp_agent.recommendation_v2.review_evidence.teacher_labeling reuse `
  --source-dataset-root 'src/yelp_agent/recommendation_v2/data/teacher_dataset/v1' `
  --target-dataset-root 'src/yelp_agent/recommendation_v2/data/teacher_dataset/v2'
```

实际复用630条，匹配失败0条，剩余1512条需要调用教师模型。

### 3. 调用GLM补齐新增1512条

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

程序按已经保存的正式标签续跑，不会重复调用已完成样本。本批实际结果：

- 新增成功标签：1512条；
- 总尝试次数：1525次；
- 首次或中间格式错误：2次；
- 首次或中间调用失败：11次；
- 重试后的最终缺失：0条；
- 输入词元：1,204,738；
- 输出词元：484,723；
- 思考词元：0；
- 接口记录费用：18.167813美元。

### 4. 最终审计结果

```powershell
python -m yelp_agent.recommendation_v2.review_evidence.teacher_labeling audit `
  --dataset-root 'src/yelp_agent/recommendation_v2/data/teacher_dataset/v2'
```

最终共有2142条正式答案，每种软偏好153条，缺失0条。其中：

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
