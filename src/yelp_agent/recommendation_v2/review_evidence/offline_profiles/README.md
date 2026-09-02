# 固定14项商家评论画像：本地准备与服务器判断

这一目录只服务于“固定14项软偏好离线画像”，不替换在线长尾评论检索。

## 分工

本地准备阶段：

1. 固定选择 Yelp 类别含 `Steakhouses`、真实评论不少于100条的评分前10家；
2. 对14项偏好分别执行高端、低端的关键词查找与意思查找；
3. 每家每项高端最多15条、低端最多15条；
4. 按评论编号合并重复片段，并生成与微调训练完全相同的模型消息；
5. 不加载 Qwen，不提前判断相关度和强度。

4090服务器阶段：

1. 加载 Qwen3-4B 基础模型和 LoRA 权重；
2. 对本地准备好的每条关系输出 `relevance` 与 `strength`；
3. 按评论时间、轻微且有上限的 `useful` 加权，并按用户去重；
4. 为每家商家生成14项程度、证据充分程度、争议程度和代表证据。

## 本地生成命令

在项目根目录运行：

```powershell
& 'C:\Users\29072\PycharmProjects\AgentSociety\.venv\Scripts\python.exe' `
  'src\yelp_agent\recommendation_v2\review_evidence\offline_profiles\prepare.py'
```

默认输出到：

```text
src/yelp_agent/recommendation_v2/data/review_evidence/v1/offline_profiles/steakhouses_top10_v1/
```

该目录就是上传单元。上传整个目录后，照其中的 `SERVER_README.md` 执行即可。

## 源码文件

- `fixed_aspects.v1.json`：14项高低端关键词和意思查找语句；只负责粗找候选。
- `prepare.py`：本地选商家、查评论、合并和生成模型输入。
- `server_judge.py`：服务器模型判断、断点续跑和商家画像聚合。
- `SERVER_README.md`：随数据复制到服务器的运行说明。

本地生成数据位于项目的 `data` 目录，按现有规则不提交 Git。源码、固定定义和
测试可以正常提交；真实生成目录需要上传到服务器或另行保存。
