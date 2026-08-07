# 第 14 步：用户画像 V1（只读长期记忆）

## 1. 本步骤解决什么问题

同一个用户可能在不同时间产生不同偏好。推荐任务发生在某个
`cutoff_time` 时，系统只能使用该时间之前的访问、评分和评论证据。

用户画像 V1 因此不是“每个用户永久只有一行”，而是：

```text
一个任务时间点
→ 一份冻结画像
→ 只包含该时间点以前的证据
```

这份画像是只读长期记忆。本步骤不会让对话临时要求写入长期画像，
也没有实现画像的增删改接口。

## 2. 输入数据

构建过程只读取以下文件：

- `businesses.parquet`：商家类别、价格属性、邮编和坐标。
- `interactions.parquet`：选定 5000 名用户的 139,402 条访问、评分和评论。
- `aspect_records.parquet`：第 13 步生成的 99,073 条评论方面证据。
- `rolling_train_contexts.parquet`：17,920 个训练时间点。
- `temporal_contexts.parquet`：5,000 个 validation 和 5,000 个 test 时间点。

构建器不会读取 617,000 多条全量商家评论，也不接收 ground truth 路径。

## 3. PreferenceSignal 怎样生成

每一种类别、评论方面、价格或常见区域都会形成一条
`PreferenceSignal`。它保存：

```text
kind / value / score / confidence / evidence_count
effective_evidence / first_seen / last_confirmed / source
```

### 3.1 原始偏好方向

历史评分先转为 `[-1, 1]` 的方向：

```text
1 星 = -1.0
2 星 = -0.5
3 星 =  0.0
4 星 =  0.5
5 星 =  1.0
```

评论方面则使用：正面 `+1`、负面 `-1`、中性或混合 `0`。

### 3.2 时间衰减

越近的证据权重越高，半衰期固定为 365 天：

```text
time_weight = 0.5 ** (距离 cutoff 的天数 / 365)
```

评论方面证据还会乘第 13 步抽取器给出的置信度。

### 3.3 汇总分数和置信度

同一偏好的多条证据做带权平均：

```text
score = Σ(方向 × 时间权重 × 来源置信度)
        / Σ(时间权重 × 来源置信度)
```

`score > 0` 表示喜欢，`score < 0` 表示不喜欢。置信度同时考虑有效证据量
和证据方向的一致程度；少量证据不会得到很高置信度，多条方向一致的证据
会逐渐提高置信度。

## 4. 各类画像如何形成

- 类别偏好：把用户对历史商家的评分分配给商家的细粒度类别，排除
  `Restaurants`、`Food` 等宽泛类别。
- 评论方面偏好：聚合用户自己的 Review Aspect，例如
  `food_quality`、`quiet_environment`、`service`。
- 价格偏好：只使用用户评分高于 3 星的历史商家价格层级，选择有效支持
  最多的层级。
- 常见区域：按时间衰减统计历史商家邮编，保留前三个。
- 位置中心：对历史商家经纬度做时间加权平均；缺失时明确返回 `None`。

画像可靠度为历史量和评论方面证据量的组合：

```text
reliability = 0.7 × (1 - exp(-历史条数 / 20))
            + 0.3 × (1 - exp(-方面证据条数 / 20))
```

## 5. Train、validation、test 为什么有多份画像

- train 有 17,920 个滚动任务，所以需要 17,920 个训练时点画像，供后续
  Hybrid V2 学习画像特征。
- validation 有 5,000 个最终验证任务，用于开发和选择策略。
- test 有 5,000 个保留任务，用于冻结后的评估。

这 10,000 份 validation/test 画像只负责各自任务发生时的历史上下文，
不会进入训练，也不会读取任务的正确商家。一个用户在 validation 和 test
拥有两份画像是正常的，因为 test 时间更晚，可以合法包含 validation 行为。

## 6. 产物与真实构建结果

产物目录为 `data/features/user_profiles/v1/`：

- `profile_snapshots.parquet`：画像主体和可靠度。
- `preference_signals.parquet`：可追溯的细粒度偏好证据。
- `task_profile_map.parquet`：任务到冻结画像的精确绑定。
- `manifest.json`：输入、配置、输出 Hash 和统计。

真实构建结果：

| 项目 | 数量 |
|---|---:|
| 用户 | 5,000 |
| train 画像 | 17,920 |
| validation 画像 | 5,000 |
| test 画像 | 5,000 |
| 画像总数 | 27,920 |
| PreferenceSignal | 891,624 |
| 历史条数不一致 | 0 |
| 平均画像可靠度 | 0.5203 |

第二次使用相同输入和配置运行时会校验 Hash，并返回 `skipped`，不会重新生成。

## 7. Agent 怎样读取

`UserProfileStore` 只允许两种读取方式：

```text
get(user_id, 完全一致的 cutoff_time)
for_task(task_id)
```

如果没有完全一致的时间点，它会明确报错，不会拿相邻时间的画像代替。
Agent 的 `GET_USER_PROFILE` 工具已经优先使用这个只读仓库，并通过兼容层把
类别、方面、价格、区域、置信度和可靠度传给现有 Prompt。未配置新仓库时，
旧实验仍可以使用原 Hybrid 画像。

## 8. 已验证的阶段门

- 同一输入和 cutoff 生成字节稳定的画像。
- cutoff 后追加极端评分或方面证据，旧画像完全不变。
- 历史和证据更多时，画像可靠度与偏好置信度提高。
- 缺少价格或坐标时返回未知值，不伪造偏好。
- 所有 27,920 个任务的画像历史条数与冻结上下文一致。
- `GET_USER_PROFILE` 能读取新画像；错误 cutoff 被拒绝。

## 9. 当前限制

- Review Aspect 仍来自第 13 步规则抽取器，画像质量受其准确率影响。
- 价格使用 Yelp 商家静态属性；后续可增加属性时效性说明。
- 本步骤只建立长期历史偏好，不理解“今天想吃牛排”这类当前请求；当前需求
  会在第 18 步 `RecommendationRequest` 中单独建模。
- 本步骤没有调用真实 LLM。
