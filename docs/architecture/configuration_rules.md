# 项目配置应该放在哪里

本文件记录第 9 步的配置整理结果。目标是让每个参数只有一个可信来源，并让每次实验都能回答“当时究竟用了什么配置”。

## 四类设置

| 设置类型 | 放置位置 | 示例 |
|---|---|---|
| 数据、算法和评测参数 | `configs/*.yaml` | 随机种子、TF-IDF 参数、Hybrid 权重、超时和交叉验证折数 |
| 一次运行的输入输出位置 | 各脚本的命令行参数 | `--tasks`、`--output`、`--businesses` |
| 当前 benchmark 的固定规则 | 实际执行代码及校验器 | 20 个候选、LLM 只重排 Top-8、历史最多 30 条、正负历史各最多 4 条 |
| 密钥和模型连接信息 | 环境变量或本地 `.env` | `OPENAI_API_KEY`、`OPENAI_BASE_URL`、`OPENAI_MODEL` |

API Key 不属于实验超参数，禁止写入 YAML、预测、trace、配置快照或 Git。

## YAML 文件职责

- `data.yaml`：Philadelphia 数据范围、用户门槛、候选负样本桶和类别体系。
- `tfidf.yaml`：TF-IDF 词表和用户关键词参数。
- `hybrid.yaml`：Hybrid V1 四项权重、时点质量和位置参数、validation 调权步长。
- `agent.yaml`：是否允许 LLM、温度、单次超时和重试次数。Top-8 等固定协议不在这里伪装成可调参数。
- `training.yaml`：rolling temporal training 的最少历史、每用户任务上限、目标间隔和时间权重规则。
- `evaluation_data_usage.yaml`：只用 validation 开发、Legacy Test 仅作历史比较，以及交叉验证和 bootstrap 规则。

所有 YAML 都通过 `src/yelp_agent/config.py` 的 Pydantic 模型加载。未知字段会直接报错，避免拼写错误被静默忽略；跨字段规则也会在运行开始前验证。

## 实验配置快照

正式的 baseline、Hybrid 和 Agent 脚本会在预测目录写入：

```text
resolved_config.json
```

它包含：

- 该类推荐运行实际使用的共享 `AppConfig`；
- 基于有效配置值计算的 SHA-256 指纹；
- 快照格式版本。

指纹不受 YAML 缩进、字段顺序或配置目录位置影响。已有预测与当前配置指纹不一致时，运行会拒绝静默复用，必须显式使用 `--force` 重建。

配置快照不读取也不保存任何 LLM 环境变量，因此不会包含 API Key。

只属于某个数据阶段的设置使用该阶段自己的 manifest，避免修改无关配置后让旧实验失效。例如 `training.yaml` 会被展开记录在 `rolling_train_manifest.json`，不会进入 baseline、Hybrid 或 Agent 的 `resolved_config.json`。

## 本步骤有意不做的事情

当前仍保留 `scripts/` 下的独立命令，没有提前合并成统一 CLI。rolling temporal train、Item-KNN 和 Hybrid V2 还会引入新的训练入口；统一 CLI 将在 Hybrid V2 完成后处理，避免现在建立随后立即推翻的入口。
