# LCRC Peptides-func Diagnostic v1 — 冻结执行协议

- Round ID：`lcrc-peptides-diagnostic-v1`
- 结果标签：`EXPLORATORY_VALIDATION_ONLY_DIAGNOSTIC`
- 角色：工程执行。只实现、执行、做正确性检查和事实汇总。
- 禁止：文献/科研设计；修改已有实验假设、配置或评价口径；设计 LCRC-B、新模块或改动 S1/S2；全数据集超参搜索；用 Test 结果选择配置/模型/seed；扩大训练规模。
- 数据集：仅 Peptides-func。

## 0. 权限边界

- 已有实验 `tracks/structural-sufficiency/lcrc_benchmark_positioning_fast_v1/` 与全部历史产物只读；本 round 只复制不覆盖。
- 所有新产物写入本目录：`analysis/`、`results/`、`configs/`、`code/`、`tests/`、`runs/`、`history/`。
- 历史三个 seed 的 Test mean AP（51.70 / 50.68 / 51.80）**只用于核对历史记录**；本轮不允许用任何 Test 数值做选择。
- P1（ECFP4 + LightGBM）只在本轮目录下产生 Validation 结果；不读取 Test、不训练第二套分类器、不尝试其他 fingerprint family。

## 1. P0：已有 LCRC 结果诊断（CPU / 只读）

输入（复制自 res，只读）：

| 文件 | 用途 |
|---|---|
| `history/peptide_seed{0,1,2}/curve.csv` | 每 epoch train loss / Validation mean AP / lr / 秒数 |
| `history/peptide_seed{0,1,2}/run.json` | best epoch、实际 epoch、参数量、耗时、checkpoint hash |
| `history/peptide_seed{0,1,2}/selection.json` | checkpoint 选择记录 |
| `history/peptide_seed{0,1,2}/checkpoint.pt` | 冻结 checkpoint（只评 Train/Validation） |

- P0.1：学习曲线审计，输出 `analysis/learning_curves.csv`、`analysis/learning_curve_summary.csv`、`analysis/learning_curves.png`。
- P0.2：只用官方 Train/Validation 标签，10 个标签的正负样本数、阳性比例、常数预测 AP 基线，输出 `analysis/label_statistics.csv`。
- P0.3：只加载冻结 checkpoint 重新评估 Train 与 Validation 的逐标签 AP；不读取 Test 数据、Test 标签或 Test predictions。若 checkpoint 不可用则记录 `CHECKPOINT_UNAVAILABLE`，不重训历史 seed。

## 2. P1：Count ECFP4 + LightGBM（CPU）

- 数据源：`skfp.datasets.lrgb.load_peptides_func(mol_type="SMILES")` + `load_lrgb_mol_splits("Peptides-func")`（HuggingFace 镜像 `scikit-fingerprints/LRGB_Peptides-func`，本地缓存 `data/lrgb_skfp/`）。
- 行身份必须严格核验：样本数 10873 / 2331 / 2331；split 索引互斥且覆盖 15535 行；标签逐行相同；并由 SMILES 构造的 RDKit 分子与 PyG 图逐原子、逐键（含 9 个原子离散列、3 个键离散列与连接关系）核验行对齐。核验不通过则 P1 停止并输出 `ROW_ALIGNMENT_BLOCKED`。
- 只用官方 Train 训练，Validation 评价；禁止合并 Train+Validation。
- 特征：`skfp.fingerprints.ECFPFingerprint(count=True, radius=2, fp_size=2048)`，默认 RDKit 原子不变量，无额外描述符，不搜索 radius/维数。
- 模型：每标签独立 `LGBMClassifier(n_estimators=500, is_unbalance=True, importance_type="gain", random_state=0, n_jobs=8, verbose=-1)`，其余为 LightGBM 默认值；不调参、不早停、不选树数。
- 指标：`sklearn.metrics.average_precision_score`，报告 10 标签非加权平均。

## 3. P2：一次 LCRC 延长训练（单 GPU）

- 任务：`Peptides-func / LCRC-C1 / seed=0`，从 scratch 训练，不加载历史 checkpoint。
- 架构与输入：沿用 Fast v1 冻结的 C1（NO_BOND_MEAN）与 Peptides-func 特征；宽度、S1/S2、readout、loss、batch、优化器参数不变（校验参数量 = 464,739）。
- 训练配置：max epochs 240；min epochs 80；early-stop patience 40；LR 3e-4；weight decay 1e-3；dropout 0.1；batch 64；BCEWithLogitsLoss；per-step cosine，horizon 240 epochs；checkpoint = 最高 Validation mean AP。
- Test 不可访问：执行路径安装 audit hook，任何打开 `peptides_func_test.pt` / `peptides_func_y_test.npy` / `test_predictions.npz` 的行为立即失败。
- 产物：每 epoch train loss / Validation mean AP / lr / 耗时；前 120 epoch、121–240 epoch、全程最佳 Validation AP 与 epoch；与 Fast v1 seed0 历史最佳 Validation AP 的差值。
- cosine horizon 从 120 改为 240，因此差值包含学习率轨迹变化，不得解释为单纯「增加 epoch」的因果收益。
- GPU 硬预算：1.25 A100 GPU-hours。

## 4. 正确性要求

- 训练前完成：split 核验、标签/行身份核验、fingerprint 仅由结构产生、标签不进入特征。
- 10 标签预测维度与概率范围检查；AP 与 sklearn 独立重算一致。
- LCRC-C1 与 Fast v1 结构、参数量、初始化 hash 一致。
- LCRC 路径不读取 Test。
- 运行配置、seed、代码 commit、数据/checkpoint 哈希全部落盘。

## 5. 交付

`REPORT.md`：只描述事实（学习曲线、标签分布、逐标签 AP、ECFP Validation 数值、延长训练结果、缺失/阻塞原因、资源与哈希、复现命令、三路 Validation 对比表），不给研究路线结论。
