# 500家商户固定14项软偏好画像

这一层把服务器生成的大型JSON转换成后续排序可以快速读取的数据。它不执行推荐，
也不会重新解释模型已经确定的特征方向。

方向唯一来自训练时教师和学生共用的
`review_evidence/training_data/teacher_input_templates.v1.json`：`strength=0`
表示客观刻度最低端，`strength=4`表示最高端，商家聚合后的 `degree` 同样从0到1
递增。例如安静程度越大越安静，拥挤程度越大越拥挤，等位时间越大等得越久，
辣度越大越辣。

生成命令：

```powershell
python -m yelp_agent.recommendation_v2.business_aspect_profiles `
  --source-directory C:\Users\29072\Desktop\important\server_output
```

生成文件：

```text
data/business_aspect_profiles/v1/
├── business_aspect_profiles.sqlite3  # 500家、7000项分数和代表性评论
├── aspect_directions.json            # 14项固定方向与0到4含义
├── supported_businesses.json         # 第一版允许离线软排序的500家
└── manifest.json                     # 来源、数量和文件校验值
```

数据库中的核心表：

- `supported_businesses`：500家支持清单；
- `aspect_directions`：14项方向定义；
- `aspect_scores`：一店一特征一行，共7000行；
- `reviews`：去重后的完整代表性评论，每条正文只保存一次；
- `aspect_evidence`：51725条商家、特征与评论的证据关系。

调用方只通过 `BusinessAspectProfileCatalog` 批量读取分数和证据，不依赖SQLite表结构。

## 正式排序怎么使用

正式硬过滤默认从这500家开始，防止有离线分和没有离线分的商家混排。用户要求与客观刻度方向相同时直接使用 `ranking_degree`，方向相反时使用 `1 - ranking_degree`。例如“更安静”直接读取安静程度，“更热闹”则反向解释。

证据少的极端分不能直接控制排名，正式排序使用：

```text
用于排序的满足程度
= 0.5 +（按用户方向转换后的程度 - 0.5）× 证据充分程度
```

`usable_for_ranking=false` 时固定按0.5处理。争议程度暂时不直接扣分，而是交给最终回答判断是否需要同时说明正面和风险。固定14项只读取这里的离线数据；自由要求仍由评论检索处理。
