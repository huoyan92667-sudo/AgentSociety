# 教师标注候选数据

这里负责从已有的 Yelp 评论候选关系中，为 14 个软偏好挑选真实评论，供后续教师模型（GLM）和学生模型（Qwen）使用。

## 现在做了什么

- 每种软偏好优先挑选 5 条低到高的程度样本，共 5 个程度桶。
- 另外挑选一小批“相关但说不清”和“完全无关”的评论，帮助模型学会相关性判断。
- 候选来自已有的关键词命中和语义命中结果，不重新把全部评论交给模型。
- 如果某个特征缺少某一档的候选，会按语义分最接近的真实评论补位，并写入 `selection_fallback=true`；这只是为了覆盖五档输入，不能当作标签。
- 通过评论编号读取对应片段；商家编号、评论编号、星级、点赞数和命中词只保存在外层，方便追溯。
- `model_input` 是真正发给教师和学生的内容，两者完全相同，只包含：特征定义、两个等级说明、特殊规则和评论文字。

## 运行

```powershell
& 'D:\anaconda3\python.exe' `
  'src/yelp_agent/recommendation_v2/review_evidence/teacher_labeling/candidate_sampler.py' `
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
