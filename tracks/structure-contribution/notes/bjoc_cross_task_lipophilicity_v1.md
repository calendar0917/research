# cscl-bjoc-cross-task-lipophilicity-v1 — 预注册 note

Track: `tracks/structure-contribution` · 建立: 2026-10-14 · 状态: 预注册（GPU 之前冻结）

对应配置: `configs/bjoc_cross_task_lipophilicity_v1.yaml`（GPU 运行前冻结；`prep`
会写 `results/bjoc_cross_task_lipophilicity_v1/protocol_freeze.json` 绑定
config/code/dataset/split 哈希）。

## 0. 单一命题

冻结的 BJOC-Null（one-shot Atom→Bond→Junction，静态 incidence，whole-graph reader）
+ train-only Masked Structural SSL 初始化，是否**跨任务迁移**到
MoleculeNet Lipophilicity（logD 回归，RMSE），并在严格可比的协议下接近强基线？

两个独立问题：

- **Q1（迁移）**：同一 SSL 机制在 ZINC 上稳定有效（P3 − P0），换到 Lipophilicity
  后，Arm P（SSL encoder + 相同 init）相对 Arm S（Scratch）是否仍有可确认增益？
- **Q2（SOTA 可行性）**：在"同 split / 同度量 / 同监督预算 / 单模型 / 无外部分子"
  的协议下，结果与已验证公开数字的差距有多大？是否可以声称 SOTA candidate？

## 1. 从 ZINC 轮继承 / 不继承

**继承**：one-shot Atom→Bond→Junction 编码、Null junction 模式、whole-graph readout、
masked SSL 目标（重建 atom/bond class）、static-incidence context、train-only
标准化、last-5 soup、matched-seed 对照、prediction 冻结与哈希纪律。

**不继承**：ZINC 的原子索引语义（输入维数不同）、ZINC 标签分布与统计量、
任何 ZINC-fitted stats 或 checkpoint、ZINC 的固定 update 数、ZINC 官方 test。
新任务**不复用** ZINC checkpoint；SSL 从新 seed init 重新预训练。

**架构不变**：hidden 宽度、composer、reader、一次前向 DAG 全部与冻结 BJOC 相同；
唯一允许变化的是输入一层的宽度（新词表）与 pretext head 的词表。
参数量 413,505 → **419,713**（差 6,208，全部来自首层输入维数）。

## 2. 数据与标签边界

- 数据: MoleculeNet Lipophilicity（4200 分子，`exp` 为 logD@7.4），官方 DeepChem
  URL，sha256 `aed41590…69bacd`（详见 `BENCHMARK_AUDIT.md`）。
- Split: DeepChem **2.8.0** `ScaffoldSplitter` 逐字算法（无 seed、确定性），
  80/10/10 → **3360/420/420**；与隔离 venv 直接运行 deepchem 2.8.0 的结果
  逐索引比对（`reference_match: true`）；0 个 scaffold/SMILES 跨集合。
- 词表: 只由 **train split 结构** 决定（15 个 (element, aromatic) 原子类，
  未见的 aromatic-Se → UNK；charge/H/chirality/stereo 固定桶 + UNK + MASK）。
  词表在任何 GPU 运行之前冻结在
  `code/bjoc_lipophilicity_adapter_v1.py` 并在 `vocab_v1.json` 留证。
- 标签隔离: 计算 checkout（`res`）上的所有 artifact（raw graph、split、objects、
  standardizer、predictions、checkpoints）**不含 valid/test 逐分子标签**；
  train `exp` 仅以冻结 `split_lipo_v1.json["train_y"]` 形式存在。
  valid/test 逐分子标签只保存在本地 `data/lipophilicity/labels.npy`（vault）。
- 训练/推理命令（ssl / arm / predict / smoke）在检测到 vault 存在时**直接拒绝运行**
  （`assert_score_labels_absent`），CPU gate 会静态检查这条纪律（≥4 个调用点、
  raw artifact 无标签字段）。
- 读取顺序: 先冻结所有 valid 预测（哈希），再一次性读 validation 标签做晋级判定；
  只有判定通过且（若需）seed 扩展完成后，才一次性读 test 标签。
  `validation_read.lock` / `test_read.lock` 记录读取时刻与哈希。

## 3. SSL（train 结构 only）

| 项目 | 冻结值 |
| --- | --- |
| 目标 | `CE(masked atom class) + CE(masked bond class)`，权重 1.0/1.0 |
| mask | atom 15% / bond 15%，**在原始 code 上打 mask 后再全量重建特征** |
| mask token | atom 16 / bond 5（专用槽，与真实类/UNK 不冲突） |
| RNG | `SeedSequence([seed, molecule_id, visit_index])` |
| 优化 | Adam lr 1e-3, wd 1e-5, clip 5.0, batch 128 |
| 预算 | `U = round(96 × N_train / 128) = 2520` updates（暴露对齐规则，GPU 前冻结） |
| checkpoint | 仅最后一步；只导出 3 个 encoder |
| 数据 | 只用 train split 图结构；valid/test 不参与 |

## 4. 监督配方 R3-Lipo（两臂完全相同）

| 项目 | 值 |
| --- | --- |
| 目标 | train-only mean/std 标准化后的 MSE；指标在反标准化后的原始尺度上计算 |
| 优化 | Adam lr 1e-3 → 1e-5（**per-step** cosine），wd 1e-5，clip 5.0 |
| batch / epoch | 128 / 360（27 steps/epoch，9720 updates） |
| soup | 最后 5 个 epoch 参数的均值；**无任何 validation 选模** |
| 两臂共享 | 同一 seed 的完整 init、完全相同 batch 顺序、相同 schedule/soup |

Arm 命名：**S** = Scratch；**P** = 同 seed init + SSL encoder（composer/reader 保持
与 S 完全相同的 seed init 值）。P 在启动时校验 encoder 初始哈希等于 SSL 运行记录
的初始哈希，否则报错退出。

## 5. 基线

- ECFP4 (Morgan r=2, 2048 bit) + RandomForest(500, max_features=1.0)；
- ECFP4 + XGBoost(500, depth 6, lr 0.05, hist)；
- 均在同一 train 索引上训练、同 valid/test 上预测；
- RDKit Crippen logP 只作描述性化学参考（绝不进入模型输入，也不作为 hidden shortcut）；
- Chemprop D-MPNN 等图模型只在时间/预算允许时作为"公平参考"追加，且必须标注其
  训练协议差异；本轮**不作为条件**。

## 6. 晋级门槛（Stage F，GPU 前冻结）

对 validation（seed 平均，单模型）比较 P − S：

1. 相对 RMSE 下降 ≥ 4%；
2. 绝对 RMSE 下降 ≥ 0.03；
3. 每个 seed 的 ΔRMSE < 0；
4. paired molecule-bootstrap 95% CI 上界 < 0（单位 = 分子，seed 方向先平均后 bootstrap）；
5. MAE 不恶化超过 +0.005；
6. 不是单分子驱动：top-1 分子增益占比 ≤ 25% 且 top-10 ≤ 50%；
7. ECFP baseline 已完成；
8. 全部正确性 / 标签隔离审计通过；
9. GPU 预算充足（记账检查）。

决策映射（判词在 GPU 前冻结）：

- 全部通过 → `TRANSFER_SSL_GAIN_PROMISING`（购买 seeds 2/3/4 确认 + 进入 Stage H）；
- 未全过但 bootstrap CI 跨过 0（`ci_low < 0 < ci_high`）→
  `TRANSFER_SSL_PROMISING_UNCONFIRMED`（不购买额外 seed、不读 test）；
- 其余 → `TRANSFER_SSL_NO_CONFIRMED_GAIN`（停止；不做第二个 mask ratio / backbone /
  loss 的搜索，也不切换到 ESOL）。

停止规则：不做 test 驱动的搜索；不把 ensemble 当 headline；失败不换数据集。

## 7. 条件 Stage G 与 Stage H

- Stage G（仅当 Gate 通过）: seeds 2/3/4 重复 S/P（同一配方）；SSL 与监督预算不变；
  可选 compute/GNN control 在预算不足时**第一个**被放弃。
- Stage H: 在全部 checkpoint、predictions、manifest 哈希冻结、validation 决策写入
  之后，**一次性**读取 test 标签；报告逐 seed 单模型 RMSE/MAE、seed mean ± std、
  paired bootstrap、scaffold 子群、参数量、GPU 小时、dataset/split 来源。
  seed-ensemble 只作单独标注的补充数字。

## 8. GPU 预算（4.0 h 上限）

| 项 | 上限 (h) |
| --- | --- |
| CPU gates + GPU smoke | 0.2 |
| seeds 0/1 SSL | 0.8 |
| seeds 0/1 监督（4 个 arm） | 0.7 |
| 条件 seeds 2/3/4 | 1.3 |
| 条件 compute/GNN control | 0.6 |
| 故障余量 | 0.4 |

预算不足时的优先级：main matched confirmation → 放弃 optional control → 停止；
**绝不**通过缩短训练或减少 seed 来维持"5 seed"的表述（若最终只有 2 seed，
在报告中明确声明）。

## 9. 不可比性声明（必须随结果一起出现）

- R3-Lipo **不是** ZINC 轮的 raw-y L1 R3；两者只共享 recipe 结构，不能互称相同；
- 本协议的 split（DeepChem 2.8.0 确定性 scaffold）与 GROVER / Chemprop 论文的
  随机 scaffold 划分不同；与 MoleculeNet 原文的 random split 不同；
- 因此任何跨来源比较都必须标注 split 类型与外部数据使用情况；
  分表呈现，且不使用公开榜单数字来断言 SOTA。
