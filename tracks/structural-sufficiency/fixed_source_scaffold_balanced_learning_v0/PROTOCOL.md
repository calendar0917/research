# PROTOCOL — FIXED1 骨架均衡学习短程验证（`fixed-source-scaffold-balanced-learning-v0`）

- Round：`tracks/structural-sufficiency/fixed_source_scaffold_balanced_learning_v0/`
- 协议 id：`fixed-source-scaffold-balanced-learning-v0`
- 本文件在任何正式训练开始之前写定。正式运行后只允许修实现（记录到 `REPORT.md` 的工程日志），
  不允许改公式、对照、评价口径、判定阈值，也不允许追加第二个训练臂。
- 机器可读冻结件：`configs/experiment.json`（由 `code/sb_freeze.py` 生成，只读 Train/Valid 结构与
  Train 标签统计，不含任何 Valid 标签派生的训练权重）。

## 1. 问题

上一轮 `lcrc_structural_failure_localization_v0` 没有找到能解释主要预测失败（相对上一个冻结模型
`lcrc_encoder_boundary_reference_v0` 的 arm B）的共同化学结构；主要失败案例缺少高相似度训练近邻，
但没有因果证据。

本轮只检验一个有限假设：

> 在模型结构完全不变的条件下，让不同训练骨架家族具有相同的总优化权重，能否改善官方 scaffold
> split 上的预测性能？

**不预设该假设成立。** 本轮不引入新特征、新参数、新损失分支、新计算模块，不做超参数搜索，
不重训任何历史臂，不追加 seed。

## 2. 资源与预算

- 数据：仅官方 `ogbg-mollipo` Train 3360 / Valid 420。**Test 不读取、不评分、不进入任何选择**
  （运行时卫兵：所有 batch/eval 入口只接受 `train|valid`）。
- seed 0；只训练 **一个新模型**（一个训练臂）。
- 排队：0（本地执行，无队列）。
- GPU：本机无 CUDA 设备（`torch.cuda.is_available() == False`，AMD 集成显卡）。因此本轮
  GPU 消耗恒为 0 GPU·h（上限 0.25 GPU·h 未被使用），预算约束落到墙钟：**硬上限 40 min**，
  训练器内部硬墙钟守卫 2100 s（超过即停并如实报告 `INCOMPLETE`）。
- 不训练其他数据集；不访问 Test；不进行超参数搜索。

## 3. 历史基准（只读复用，不重训）

`context_conditioned_structural_computation_v0/` mollipo A / `FIXED1`，seed 0
（`runs/mollipo_A_seed0/run.json`，`model_audit.initial_state_sha256 = 8b2df74e…`）：

| 项 | 值 |
|---|---:|
| 参数量 | 294,135 |
| best epoch | 140 |
| best Valid RMSE | 0.7266780069873353 |
| best Valid MAE | 0.49844932380229395 |
| endpoint（epoch 360）Valid RMSE / MAE | 0.7415200258174056 / 0.5060955929875536 |
| last-20 Valid RMSE / MAE | 0.7406963081735681 / 0.5061564662748497 |
| best epoch 的 Train RMSE / MAE | 0.3079864751738112 / 0.23350284132462634 |
| train plan sha256 | `f9c1398c5705ca5a31d2cee418fa1fda6d5bc944ad17451d3a6a98524ae31276` |
| 环境 | NVIDIA A100-SXM4-40GB（历史）；本轮 CPU |

描述性参照（**不是**严格匹配消融，不重训、不参与判定）：

| 参照 | Lipo Valid RMSE | 来源 |
|---|---:|---|
| C1（`method_L`，LCRC 原读出，`lcrc_readout_context_pilot_small_v1` arm A） | 0.7086243164299489 | `lcrc_readout_context_pilot_small_v1/analysis/tables/main_table.csv` 行 `ogbg-mollipo,A` |
| 上一轮共享读出头 GINE（`lcrc_encoder_boundary_reference_v0` arm B） | 0.6780750298150204 | 同 round `analysis/main_table.csv` |

## 4. 数据准备（只用 Train SMILES）

- 真实分子身份与官方 split：`data/ogb_reader/ogbg_mollipo/{mapping/mol.csv.gz, split/scaffold/{train,valid}.csv.gz}`，
  与上一轮结构失败分析相同。cache 侧 SMILES 与 `mapping/mol.csv.gz` 的规范形式逐行核对。
- Bemis–Murcko scaffold：**逐字复用上一轮实现**
  （`lcrc_structural_failure_localization_v0/code/sfl_common.py::scaffold_key`）：
  `Chem.MolFromSmiles(smiles)` → `MurckoScaffold.GetScaffoldForMol(mol)` → `Chem.MolToSmiles(scaffold)`。
  - **chirality**：`MolToSmiles` 默认 `isomericSmiles=True`，即 scaffold 键型/立体标识保留在字符串里；
    这与上一轮记录的 1604 个 Train scaffold / 420 个 Valid scaffold 一致（若用
    `includeChirality=False` 会得到 1581 个 Train scaffold，版本不同，本报告如实记录该差异来源）。
  - **空 scaffold**：无环分子返回空字符串 `""`，全部无环分子共用一个 `""` 组（不拆成单例组）。
  - **多组分**：不剥离盐/不取最大片段；按 RDKit 解析后的整个（可能多片段）分子计算 scaffold。
    Train 中多组分 SMILES 为 0 个，Valid 1 个（只影响支持度诊断）。
  - 只用 Train 的 scaffold 定义分组；Valid 的 scaffold 只用于核对官方 split 性质，不参与权重。
- 统计并报告：K（不同 Train scaffold 数）、每个 scaffold 的 n_s、大小分布、singleton 数量与比例、
  最大 scaffold 的分子数、原始分子等权目标下每个 scaffold 的总权重（= n_s / N_train）。
- 权重对比门禁（**INSUFFICIENT_WEIGHT_CONTRAST**）：若同时满足
  `max_i w_i / min_i w_i < 1.5` 且 `ESS/(N_train) > 0.9`（ESS = (Σw)²/Σw²），
  则说明骨架大小分布几乎均匀、干预无法产生实质权重变化 → 报告 `INSUFFICIENT_WEIGHT_CONTRAST` 并停止，不训练。
- 不利用 Valid 标签设计训练权重；权重公式只接受 scaffold 分组 + N_train + K，不接受任何标签。

## 5. 唯一训练臂 `SCAFFOLD_BALANCED`

- 模型：完全复用 `context_conditioned_structural_computation_v0/code/cs_model.py::build_csc(arm="A")`
  （= 历史 FIXED1：BSModel scale-1 trunk + 两个 `MessageFFN` 阶段，`Q=0`，无 Gate），
  同一 `cs_common.SEED=0`、同一 `initialize_new_modules` 规则、同一 294,135 参数、同一推理计算。
- 数据/训练顺序/优化器/配方完全沿用 CSC v0（`cs_common.RECIPE`）：
  AdamW，单参数组，betas (0.9, 0.999)，eps 1e-8，peak LR 3e-4 → min 1e-5 每步 cosine 无 warmup，
  wd 1e-3，grad clip 5，FP32，dropout 0.10（`h0` 之后），batch 128，**360 epochs**，
  标准化目标（Train-only mean/std，ddof=0），无 EMA/SWA/soup/早停/辅助损失。
  每个 epoch 一个 `np.random.default_rng(0).permutation` 覆盖 Train（0..3359），
  与历史 A **同一冻结 batch 顺序**（`plan_sha256` 必须等于 `f9c1398c…`）。
- 唯一改动：原始标准化 MSE → 固定加权 MSE。

  ```python
  weight_i = N_train / (K_train * scaffold_count_i)          # scaffold_count_i = 分子 i 所属 scaffold 的 Train 分子数
  loss = (weight_i * per_molecule_mse).mean()                # 每个 batch 内按分子求均值，不在 batch 内重标定
  ```

  不做 sqrt 权重、不截断、不动态更新、不做 batch 内权重和归一化、不重采样、不删分子。
  数学性质：全 Train 权重均值严格为 1；每个 scaffold 的总权重严格为 `N_train / K_train`。
- 每 epoch 在 eval 模式对完整 Train / Valid 求原始标签单位指标；Valid RMSE/MAE 一律**普通分子等权**，
  不使用训练权重。checkpoint 按最低完整 Valid RMSE 选择（严格更小才替换，平局取更早 epoch），
  同时保存 final 与最后 20 epoch 平均预测。
- 不评价 Test：`D.evaluate` 只被 `train`/`valid` 调用。

## 6. 正式训练前的必要检查（全通过才训练）

1. 初始状态指纹等于历史 FIXED1 的 `initial_state_sha256`（`8b2df74e…`）与 `base_state_sha256`（`539e0dbe…`）。
2. 参数量 294,135；state_dict 键集合/形状与历史 `checkpoint_best.pt` 完全一致；无 Gate/无新增模块。
   历史 checkpoint 在本机重载后 Valid RMSE 复现 0.726678（tolerance 1e-6，设备差异不改变 FP32 权重）。
3. 全部 Train 分子参与训练：每个 epoch 的行集恰为 0..3359 的置换，360 epoch 总计 1,209,600 行次。
4. 每个 scaffold 的总权重都等于 `N_train/K_train`。
5. 权重均值 = 1（严格），报告 min/max、分位数、ESS。
6. 权重只来自 Train 结构：对 Train 标签做 200 次随机置换后重新计算权重，向量逐位相同；
   权重函数签名不接受标签。
7. 权重全为 1 时（真实 batch 上）加权损失与 `F.mse_loss` 逐位相同，梯度逐位相同。
8. 验证指标不使用训练权重：Valid RMSE 由保存的预测与官方标签手工重算一致；
   评价路径不接受权重参数。
9. 预检计时（真实 batch 吞吐）投影的 360 epoch 墙钟 < 硬上限；否则停止。

## 7. 结果分析口径

主指标：best Valid RMSE（全量 420 分子、普通等权）。同时报告 best Valid MAE、endpoint、last-20、
完整 Valid 曲线、以及同 epoch 的普通（非加权）Train RMSE/MAE（并与历史 A 的同 epoch 值对照）。

分析（不修改主指标）：
- 逐分子平方误差差值（本轮 vs 历史 FIXED1 best checkpoint 的逐分子预测，来自
  `context_conditioned_structural_computation_v0/runs/mollipo_A_seed0/predictions_best.npz`，
  已与 `lcrc_encoder_boundary_reference_v0/analysis/paired_errors.csv` 逐位对齐）；
- 2000 次 paired molecule bootstrap（seed 20261126）的 ΔRMSE / ΔMAE 95% 区间；
- 改善/退化分子数；双方共同平方误差最大的 5% 去除后的敏感性；
- 上一轮 11 个固定案例（`lcrc_structural_failure_localization_v0/analysis/cases.csv` 的 row_id）的预测变化，描述性；
- 训练支持度诊断（**只做一个**）：Morgan r=2、2048-bit；Valid 每个分子的最大 Train Tanimoto；
  以 Train leave-one-out 最大相似度中位数为**固定**分层阈值（不搜索），分相对低/高支持两组，
  报告两组的分子数、RMSE、MAE、平方误差变化；
- 优化诊断：实际梯度裁剪频率、加权 loss 与普通 Train MSE 的曲线，防止把优化异常解释为结构迁移。

## 8. 判读（预注册）

以 `best Valid RMSE` 相对历史 FIXED1（0.7266780069873353）的变化为主：

- `STRUCTURAL_DISTRIBUTION_PROMISING`：Δbest ≤ **−0.03**，且 endpoint 与 last-20 同方向改善，
  且 MAE 无明显退化（ΔValid MAE ≤ +0.005）。同时报告是否低于历史 C1 的 0.7086243164299489。
- `WEAK_OR_TAIL_ONLY_GAIN`：有少量收益，或收益集中于极少数案例，缺乏整体推进。
- `NO_CLEAR_GAIN`：无实际改善或明显退化。
- `INCONCLUSIVE`：数据、实现或预算异常，无法形成可比较结果。

0.03 是资源分配参考，不是统计显著性标准。若低支持组也有改善，只作为"结构分布干预可能有助于外推"
的线索，不宣称证明原因。单 seed、历史已暴露 Valid，只用于探索。

## 9. 交付

`PROTOCOL.md`、`configs/experiment.json`、`code/`、`analysis/`（检查、骨架分布、权重摘要、主表、
Valid 曲线、逐分子比较、paired 比较、支持分层、固定案例、训练诊断、判定、预检计时）、
`runs/scaffold_balanced_seed0/`（曲线、预测、checkpoint + SHA256 manifest）、`REPORT.md`（中文）。

本轮完成后停止：不搜索新的权重公式，不追加 seed，不自动进入下一阶段。
