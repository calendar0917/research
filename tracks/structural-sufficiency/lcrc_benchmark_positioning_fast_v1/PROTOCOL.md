# LCRC Benchmark Positioning Fast v1 — 冻结协议

- Round ID：`lcrc-benchmark-positioning-fast-v1`
- 结果标签：`EXPLORATORY_FAST_BENCHMARK`
- 角色：科研工程执行；只复用仓库已有实现、编写适配代码、运行测试/实验、记录结果、生成报告。
- 禁止：文献/SOTA 调研；新模型设计、结构搜索、额外消融、预训练、扩展超参数搜索；根据 Test 结果选择模型/配方/epoch/seed；自行提高预算；覆盖旧实验目录。
- 目标：在有限资源下给出 LCRC 在 13 个常用图学习数据集上的**初步性能位置**，不是 SOTA 复现。

## 1. 实验目录与交付

新目录：`tracks/structural-sufficiency/lcrc_benchmark_positioning_fast_v1/`，包含：

- `PROTOCOL.md`：本协议；
- `configs/protocol.json`：冻结机器可读配置；
- `code/`：数据适配、训练、调度、汇总；
- `tests/`：最小正确性测试；
- `analysis/`：数据来源、划分、哈希、预检查、资源账本；
- `results/`：完成任务指标 CSV；
- `REPORT.md`：中文事实报告，含未完成项和原因。

不提交大型可重建缓存或模型权重。

## 2. 模型与训练规则

主模型：仓库已有 **LCRC-C1 / NO_BOND_MEAN**。

- 唯一结构区别：从 LCRC-A 的局部输入中删除 96 维 `b_mean = b_sum / deg` 块；保留 S1/S2、局部贡献函数、全局 `rho` 及其余结构。
- 冻结 width：64 / 96 / 96 / 32 / 192 / 64（width、atom_out、bond_out、stat_rank、local_hidden、rho_hidden）。
- 数据集可因输入维度和输出维度改变总参数数；不得改变主干、增加 attention、消息传递层或新读出模块。
- 从头训练，无 SSL；AdamW，fp32，gradient clipping = 5。
- 不做 ensemble、EMA、SWA、soup、数据增强、类别重采样或额外 loss。
- 所有归一化统计只在对应 Train 内拟合。
- 每个训练任务只保存 Validation 指标最佳的 checkpoint。
- Test 只在该任务训练、checkpoint 选择和 checkpoint SHA-256 记录完成后评估一次。
- 记录实际 epoch、best epoch、参数量、训练时间、显存峰值、随机种子。
- 不得根据 Test 表现重新运行某个配置。

## 3. TU 实验

### 3.1 数据集

IMDB-BINARY、IMDB-MULTI、COLLAB、REDDIT-BINARY、MUTAG、NCI1、PROTEINS、DD、ENZYMES。

使用官方原始 TUDataset，`cleaned=False`，不使用自行过滤的图集合。

### 3.2 特征

沿用当前 LCRC 的 TU 拓扑结构、节点类型、S1/S2、global statistics 构造语义。

- 保留原始离散节点标签；
- 同时保留可用的连续节点属性；
- 连续属性在 fold 的 Train 上拟合归一化；
- 原始边标签或属性存在时，按现有 LCRC 类型编码机制确定性映射；
- 不因数据集特征维度不同删除原始属性；
- 所有特征构造独立于目标标签。

原始属性无法正确适配时，不得静默丢弃：标记 `INPUT_ADAPTER_BLOCKED` 并保留日志。不做新特征工程或学习式结构预处理。

### 3.3 交叉验证

每个数据集一轮分层 10-fold：

```python
StratifiedKFold(n_splits=10, shuffle=True, random_state=20261104)
```

每个 outer fold：

- outer Test = 当前 fold；
- outer Train = 其余 9 folds；
- 从 outer Train 按类别分层抽取其 1/9 作为 Validation；
- inner holdout `random_state = 1000 + fold_id`；
- 最终约为 Train 80%、Validation 10%、Test 10%；
- 模型初始化种子 `100 + fold_id`；
- 仅训练一个模型，不做内层网格搜索；
- Validation 用于 checkpoint 选择和早停；
- outer Test 仅用于最终评价。

不得用 outer Test 准确率选超参数，不得事后替换某个 fold 的模型。

### 3.4 TU 冻结配方

- Optimizer：AdamW
- Learning rate：`1e-3`
- Weight decay：`1e-4`
- Dropout：`0.1`
- Batch size：`64`
- Loss：CrossEntropyLoss
- Maximum epochs：`120`
- Cosine scheduler：保持历史 TU 200-epoch 调度周期，允许在 120 epoch 提前停止
- Minimum epochs before early stopping：`40`
- Early stopping patience：`25`
- Validation：每 epoch
- Checkpoint selection：最高 Validation Accuracy；并列时优先较低 Validation CE，再优先较早 epoch
- Early stopping：连续 25 个 epoch 未出现严格更高的最佳 Validation Accuracy 则停止

所有 fold 使用相同参数，不允许针对某个数据集的 Test 表现调参。

TU 最终结果报告 10 个 outer Test Accuracy 的平均值和样本标准差（百分比）。额外保留每折 macro-F1，但不用于模型选择。这是单次 10-fold，不是 10 个独立训练种子的统计推断。

## 4. OGB 分子任务

数据集：`ogbg-molhiv`、`ogbg-molbace`、`ogbg-moltox21`。

统一使用 OGB 官方 `get_idx_split()` 的 train/valid/test，不重新划分。训练种子固定为 `[0, 1, 2]`。

使用仓库现有 MoleculeNet R3 训练配方：

- AdamW
- LR `3e-4`
- Weight decay `1e-3`
- Batch size `128`
- Dropout `0.1`
- Gradient clipping `5`
- 原有 per-step cosine 调度周期保持 360 epochs，不因提前截断而重新调整调度跨度
- 不调参

三个任务的最大 epoch / 最小训练 epoch / patience：

| Dataset | Max | Min | Patience |
|---|---:|---:|---:|
| MolHIV | 110 | 60 | 40 |
| MolBACE | 150 | 50 | 35 |
| MolTox21 | 120 | 40 | 30 |

每个 epoch 在官方 Validation 上计算 ROC-AUC。严格改善主指标时更新最佳 checkpoint 和 early-stop 计数；连续 patience 个 epoch 无改善则结束。Test 使用 OGB 官方 Evaluator。

MolTox21 是 12-task 多标签分类；训练时保留缺失标签掩码，不得把 NaN 转成负例。MolBACE、MolTox21 不得使用此前 ChemBench 的 random/scaffold indices 代替 OGB 官方 split。

本轮 LCRC-C1 从已有模型实现适配；保留当前分子输入特征计算方式，不新增 RDKit 目标相关描述符。

最终逐 seed 记录 ROC-AUC，报告三个 seed 的均值和样本标准差。

## 5. LRGB Peptides-func

使用官方 PyG LRGBDataset Peptides-func 固定 train/val/test，不自行生成划分。

- 10 个独立 sigmoid 输出，BCEWithLogitsLoss；
- 评价指标是 10 个标签 AP 的宏平均，使用预测概率或 logits 的排序计算，输出百分比；
- 输入使用数据集提供的 9 列 OGB 风格离散原子特征和 3 列离散键特征；
- 按各字段合法类别数分别 one-hot 编码，不把类别整数当连续物理量；
- 以第一列原子类型、第一列键类型建立 LCRC S1/S2 类型码；
- 其余输入与 LCRC 既有局部统计及全局结构摘要接口连接；
- 结构统计仅使用图，不使用目标；
- 确认无向物理边正反两条记录对应同一键属性。

固定训练参数：

- Seeds `[0,1,2]`
- AdamW，LR `3e-4`
- Weight decay `1e-3`
- Dropout `0.1`
- Batch size `64`
- Maximum epochs `120`
- Cosine scheduler horizon `120`
- Minimum epochs `40`
- Patience `25`
- 以 Validation mean AP 最大选择 checkpoint

如 batch=64 OOM，按预先固定的顺序尝试 32、16。允许变化的仅是 batch size；记录实际值，不调整其他超参数。如果 16 仍 OOM，标记 `RESOURCE_BLOCKED`。

## 6. 大图计算与正确性限制

LCRC S2 原则上精确枚举非回退二跳路径。COLLAB、REDDIT-BINARY、DD 和 Peptides-func 必须先进行数据预处理及训练吞吐预检查。

禁止：随机采样 S2 路径；静默截断高阶节点；删除大图；静默更改图边；通过改变 S2 计算定义换取速度。

允许对缓存和计算做不改变数学语义的内存管理优化。不要求开发新的稀疏矩阵算法；如果现有实现无法在预算内精确执行，跳过并标明。

正式运行前：

1. 对每个大图数据集完成 label-free 缓存建立；
2. 记录每图节点数、边数、S2 路径数分布和最大值；
3. 跑一个不读取 Test 的 5-epoch 训练吞吐检查；
4. 测量实际每 epoch 耗时、CPU 内存、GPU 显存；
5. 如果完整任务预计超过单次 25 分钟，标记 `RESOURCE_BLOCKED`，不启动该任务；
6. 如果单数据集预处理超过 30 分钟，停止该数据集并记录阻塞原因。

不得为了填满结果表突破这些限制。

## 7. 正确性门禁

正式训练前至少验证：

- split 两两互斥、完全覆盖对应数据集；
- TU 每个 outer Test fold 在训练和选 checkpoint 时均不可访问；
- 分子任务官方 split 与缓存的 graph indices 一致；
- train-only normalization；
- 目标标签不进入输入缓存；
- 节点重编号不变性；
- 单图与 batch 前向一致性；
- S2 路径构造与小图手算一致；
- LCRC-C1 结构和原始 C1 一致；
- loss、梯度和 checkpoint reload 正常；
- Test metric 可从保存的预测值独立重算；
- Peptides-func AP 和 OGB ROC-AUC 的维度与缺失值处理正确。

所有测试通过后才运行正式 GPU 作业。失败时只允许修复实现 bug，不允许更改协议。

## 8. 双 A100 调度

机器：`res`，2× NVIDIA A100-SXM4-40GB。每张 GPU 同一时刻最多一个训练进程。

使用仓库现有 `rr` 机制调度，记录 run_id、GPU id、代码 commit 和退出状态。

建议执行顺序：

- **P0：** 实现、准备数据、CPU gates、CUDA smoke、大图 5-epoch 成本测量。
- **P1：** 先完成 MolHIV、MolBACE、MolTox21 的三个 seeds，并完成现有四个 TU 数据集的 10-fold。
- **P2：** 执行 Peptides-func，再执行剩余五个 TU 数据集。

不同 GPU 的任务可以交错排队以提高利用率，不要求两个 GPU 按完全相同顺序运行。

每个训练任务独立保存状态，支持中断后跳过已完成任务；不得覆盖已有 checkpoint 或复用未完成训练指标作为完成结果。

### 资源预算

从正式 benchmark GPU 队列启动时开始计时：

- 墙钟软预算：6 小时；
- 墙钟硬上限：10 小时；
- GPU 总消耗上限：18 A100 GPU-hours；
- 单个完整训练任务上限：25 分钟；
- 不允许为了等待最后几个任务越过硬上限。

达到软预算时停止启动新的大图任务，继续完成已开始且能在硬预算内结束的任务。达到硬上限时不再启动任务，安全停止未完成任务，并正常汇总所有已完成结果。

资源阻塞、缺失或失败的数据集不允许用历史随机划分结果填充。

## 9. 与用户给定截图的比较

用户截图的 reference Ours 数值（百分比）：

TU：

| Dataset | Reference |
|---|---:|
| IMDB-BINARY | 73.87 |
| IMDB-MULTI | 50.97 |
| COLLAB | 80.72 |
| REDDIT-BINARY | 91.14 |
| MUTAG | 89.20 |
| NCI1 | 83.09 |
| PROTEINS | 77.85 |
| DD | 79.48 |
| ENZYMES | 69.73 |

Molecular：

| Dataset | Reference |
|---|---:|
| MolHIV | 81.47 |
| MolBACE | 81.67 |
| MolTox21 | 76.67 |
| Peptides-func | 71.57 |

这些数字仅用于生成直观定位表；其原论文具体数据协议和分数来源未在本任务中独立核验，统一标记：

`SCREENSHOT_REFERENCE_PROTOCOL_UNVERIFIED`

禁止将截图 Ours 行当作已经严格同协议的直接对手。

`REPORT.md` 生成两张主要比较表：

- TU：数据集、截图 reference、LCRC-C1 平均 Accuracy、跨折标准差、差值、协议说明；
- 分子：数据集、截图 reference、LCRC-C1 平均 ROC-AUC/AP、跨 seed 标准差、差值、协议说明。

未完成正式任务的数据集单元格填 `NOT_RUN`，不填估计成绩。

## 10. 交付要求

1. 完整运行和失败清单；
2. 13 个数据集的结果定位表；
3. 每个任务的 best epoch、实际 epoch、实际耗时、GPU 显存、参数数和 checkpoint hash；
4. 每个 TU fold 的 Accuracy、每个分子 seed 的 ROC-AUC/AP；
5. 总 GPU-hours、总墙钟时间、双 GPU 使用情况；
6. 从已有预测与记录直接计算的描述性统计；
7. 未完成项和真实原因；
8. 运行复现命令及代码 commit。

报告只描述观察事实，不推导新模型研究结论，不提出新架构，不自行决定下一轮实验。

验收标准：已完成任务具有可核验的数据来源、独立的验证选择和最终 Test 评分；未完成任务有明确原因；没有预算越界、协议漂移或静默近似。
