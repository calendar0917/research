# REPORT — LCRC PCBA 规模试运行（`SCALE_PILOT`）

Round: `lcrc-pcba-scale-pilot-v1`
结果标签：**`SCALE_PILOT`**（不是 `RESOURCE_LIMITED`，也不是正式基准结果）
数据集：`ogbg-molpcba`，官方 `GraphPropPredDataset` + 官方 `get_idx_split()`
模型：历史 **LCRC-C1 / NO_BOND_MEAN**（无 `b_mean` 块）
seed：**0（唯一）**

---

## 1. 结论摘要

| 项 | 结果 |
|---|---|
| 训练状态 | **`COMPLETED`**（39/40 epoch，`EARLY_STOP_PATIENCE`） |
| 最优 Validation AP | **0.23449 @ epoch 29**（OGB 官方 `Evaluator("ogbg-molpcba")`） |
| checkpoint 重载后 AP | 0.2344923（Δ = 4.8e-7） |
| 总 GPU 时间 | **2.137 GPU-hours**（训练）+ 0.008（smoke）≈ **2.15 h**，预算 4.0 h |
| 数据量 | 437,929 分子 / 128 任务 / 11,373,137 原子 / 24,618,372 S1 / 34,369,144 S2 |
| 正确性门槛 | 8 项 smoke gate + 数据/缓存/标签审计**全部通过** |
| 官方 Test | **未读取、未评估**（Test 行在标签工件中全为 NaN） |

一句话：128 任务的掩码损失链路、官方 AP、逐 epoch 选择与 checkpoint 重载都工作正常；
在 39 个 epoch 内 Validation AP 从 0.089 升到 **0.234**，第 29 epoch 后进入平台期，
资源远未用尽。

## 2. 是否完整训练、是否资源受限

* `status = COMPLETED`，`stop_reason = EARLY_STOP_PATIENCE`。
* 最优 epoch 29，`patience = 10` 在第 39 epoch 触发早停；`max_epochs = 40`、`min_epochs = 12` 都已满足。
* 不是 `RESOURCE_LIMITED`：soft（3.3 h）/hard（3.6 h）保护均未触发，
  $4$ GPU-hour 预算用掉约 54%。
* 因此本轮是**完整跑完的探索性训练**，不是被预算截断的残次结果；但只有一个 seed，
  仍不能作为正式基准数字。

## 3. 资源

| 项 | 值 |
|---|---|
| GPU | `res-gpu0` = NVIDIA A100-SXM4-40GB（`rr` pool `res-gpu0`） |
| 训练墙钟 | 7693 s（2.137 h），39 epoch |
| 每 epoch | 197.2 s（train 188 s + valid 10 s），2738 step/epoch |
| GPU 峰值显存（torch） | allocated 86.5 MB / reserved 123.7 MB（batch 128，宽 64 的小模型） |
| 参数量 | 245,184 |
| smoke | 27.3 s（8 项 gate 全通过），投影 40 epoch ≈ 2.22 h |
| 数据预处理（本地 CPU，14 核） | 分块构建 176.5 s + 合并 26.5 s + 审计 13 s ≈ **3.6 min** |
| 缓存体积 | 4.74 GB（`npy-directory-v1`，内存映射读取） |

GPU 预算记账：训练 2.137 h + smoke 0.0076 h ≈ **2.15 GPU-hours ≤ 4.0**。

## 4. 训练曲线

| epoch | train loss | Validation AP | 备注 |
|---|---|---|---|
| 1 | 0.0553 | 0.0889 | |
| 5 | 0.0405 | 0.1781 | |
| 12 | 0.0368 | 0.2110 | 达到 `min_epochs` |
| 19 | 0.0353 | 0.2235 | |
| 23 | 0.0347 | 0.2265 | |
| 26 | 0.0343 | 0.2279 | |
| **29** | **0.0340** | **0.23449** | **最优，保存 checkpoint** |
| 36 | 0.0334 | 0.2340 | |
| 39 | 0.0332 | 0.2320 | 早停（10 epoch 无严格提升） |

* Train loss 单调下降 0.0553 → 0.0332（−40%），**无发散、无非有限值**
  （`non_finite_events = 0`）。
* Validation AP 在前 29 epoch 持续上升，之后在 0.229–0.234 之间震荡；
  末 5 epoch 均值 0.2318，与最优差 0.0027 → 属于**平台期**，不是过拟合崩塌。
* 梯度：`grad_norm_max = 11.35`，仅 4 次触发 clip 5.0（2738×39 = 106,782 step 中）。
* 学习率按历史 per-step cosine（`T_max = 2738 × 360 = 985,680` 更新）只走到周期的 11%，
  所以 lr 仍接近峰值 3e-4（末 epoch 2.9e-4）——历史配方本来就是 360 epoch 视野，
  40 epoch 内的 lr 几乎不衰减是**预期行为**，不是 bug。

## 5. 任务覆盖

| 项 | train | valid |
|---|---|---|
| 已观测条目 | 27,162,842 | 3,477,656 |
| 正例 | 359,975（1.33%） | 55,964 |
| 有正例的任务 | **128 / 128**（最少 21 个） | 127 / 128（1 个任务 valid 无正例） |
| 每任务已观测（中位） | 741 正例 / 约 21 万条目 | — |

* 每个 epoch 的 Validation AP 都用 127 个可评分任务计算（`n_tasks_scored = 127`）。
* `per_task_ap.csv` 共 39 epoch × 128 任务 = 4992 行，其中 4953 行有 AP（缺的 39 行来自
  那个 valid 无正例的任务）。按最优 epoch 29 的 128 行看：42 个任务 AP > 0.3，
  中位 0.200，最高 0.839。
* 缺失标签严格按 NaN 排除：smoke 中 4096 行 × 128 列里 194,160 个 NaN 全部被排除，
  masked BCE 与 float64 参考实现差 5.9e-8；把 NaN 当 0 的反事实值 1.1226 vs 正确 1.1318
  （差 0.0091），说明"当 0 处理"确实会算错，测试因此有区分度。

## 6. 实现继承关系与具体修改

**继承（未改动，并在 import 时断言）**

* 模型：`bp_model.build_c1`（`cont_slice=None`），`MODEL_CONFIG` 宽度 64/32/192/128/96/160/96/64，
  `variant = C1`，`has_b_mean = False`（`smoke.json` 记录 `local_layout` 无 `b_mean`）。
* 训练配方：AdamW、peak lr 3e-4、weight decay 1e-3、dropout 0.10、batch 128、
  grad clip 5.0、per-step cosine（`eta_min = 1e-5`，T_max = steps × 360）。
* 特征：`mb2_rdkit.smiles_to_raw_row_relaxed` → `raw_row_to_graph` → `lcrc_data.build_molecule`
  → `concat_molecules`，spec = 冻结 `lipo_v1`（hash `7270f767…`，atom_feat 51 维、global 42 维、
  S1 80 码、S2 400 码）。`pc_common.assert_frozen_inheritance()` 在每次运行时校验这些常量未被改动。

**为 PCBA 必须做的适配**

| 修改 | 原因 |
|---|---|
| 128 任务输出头 + 逐任务 masked BCE | 历史 `bp_mol.py` 只有 MolHIV/BACE/Tox21 的单任务/12 任务假设 |
| 官方 `Evaluator("ogbg-molpcba")` 作为唯一选择指标 | 128 任务的 AP 必须与官方口径一致 |
| 分块并行构建 + 内存映射目录缓存 | 43.8 万分子无法一次性构造 Python 分子对象（旧路径会在内存/序列化上 OOM） |
| 标签工件把官方 Test 行写成 NaN + 训练前断言 | 保证 Test 不可达（`test_evaluated = False`） |
| Watchdog / soft-stop / 逐 epoch 落盘 | 预算内优雅保存，避免被外层 kill |

**明确未做**：没有换架构、没有 attention、没有预训练、没有额外分子描述符、
没有 class reweighting / focal loss / 数据增强。

## 7. 正确性检查（全部通过）

| 检查 | 结果 |
|---|---|
| 官方数据量/任务数/划分 | 437,929 / 128 / 350343-43793-43793，hash 记录在 `data_audit.json` |
| 标签二值性与缺失结构 | 观测值只取 {0,1}；train 1.33% 正例；test 行全 NaN |
| 缓存不含标签 | `label_like_keys = []`、`labels_present = false` |
| S2 计数恒等式（全量，非抽样） | 11,373,137 个原子逐一 `#S2(root) == Σ_{w∈N(root)}(deg(w)−1)`；S1 总数 = 2×键数 |
| S2 内容一致性（抽样 64 分子） | 从 SMILES 重新枚举的 `path_cat/path_end63/path_end/path_closure` 与缓存**逐元素相等**，无截断 |
| 历史 MolHIV 交叉校验 | 同一冻结实现重算历史缓存 8 个分子，S2 记录完全一致；1,049,163 原子恒等式通过 |
| SMILES 转换 | 437,927 严格解析 + **2 个** unsanitized fallback（row 5823 `CC(=O)O[AlH3]OC(C)=O`、row 303198 `CCCCO[AlH3](OCCCC)OCCCC`，RDKit Al 价态）；**0 个失败、0 个静默丢弃**（记录在 `strict_parse_failures.json`） |
| 输出形状 | `[128, 128]`，全有限 |
| masked loss | 与 float64 参考差 5.9e-8；全 NaN batch → 0.0；NaN 当 0 的反事实明显不同 |
| 梯度 | 全参数有限；`grad_norm_max` 11.35 |
| checkpoint 重载 | 状态逐元素相同，valid AP 差 4.8e-7 |
| 单图 vs 批量 | 一致（gate 通过） |
| 原子重编号不变性 | 29 个原子随机置换后 logits 最大差 1.9e-6 |
| 官方 AP vs 独立实现 | 官方 0.03023963 vs sklearn 0.03023963（Δ = 1.0e-17），NaN 逐任务排除 |
| Test 行不可达 | 43,793 个 test 行全 NaN，训练代码未索引（`label_masking_gate`） |
| 选择规则 | 严格提升才更新；`tie_check.consistent = true`（最大 AP，平分取更早 epoch） |

smoke 只用于正确性、吞吐、显存和数值稳定性，**没有**用它挑超参数。

## 8. 如何解读这个 AP（以及不能怎么解读）

* 本轮是**单一 seed、40 epoch 上限、`SCALE_PILOT`** 的探索性训练。历史 LCRC 的
  MolHIV 结果是在不同数据集（41k 分子 / 1 任务）上得到的，PCBA 是 438k 分子 / 128 任务，
  **两者的 AP 不可比**。
* **不能**把 0.23449 与论文里的 PCBA **Test** AP 直接比较：本轮从未读取 Test 标签，
  也没有做多 seed 平均、没有做超参搜索、没有跑满 360 epoch 的 cosine 周期。
* 曲线在第 29 epoch 后已经平台化，且 lr 仍在峰值附近，说明在**当前配方与预算**下继续
  训练到 40 epoch 收益很小；这既可能反映"模型容量/特征已到位"，也可能只是
  "40 epoch 太少、早停来得太早"。**本轮的证据不足以区分这两者**。
* 因此本报告只声称：**管线在真实 PCBA 规模上数值正确、可复现、资源可控**，
  并给出一个可复用的 seed0 基线点，供后续（未授权的）多 seed / 长周期实验对照。

## 9. 交付物与复现

| 文件 | 内容 |
|---|---|
| `PROTOCOL.md` | 实验定义与实现映射 |
| `analysis/data_audit.json` | 数据量、划分、逐任务标签统计、hash |
| `analysis/feature_cache_audit.json` | 标签无关缓存、结构、全量 S2 恒等式、抽样内容一致、历史交叉校验 |
| `analysis/pcba_cache_build.json` | 分块构建摘要（110 shard、0 失败、176.5 s、14 workers） |
| `analysis/strict_parse_failures.json` | 2 个 fallback 行号与 SMILES |
| `analysis/smoke.json` | 8 项 gate + 计时投影 + 缓存/模型指纹 |
| `analysis/cache_manifest.json`、`cache_host_verify.json` | 缓存/标签/划分 hash 与训练主机上的复核结果 |
| `runs/pcba_seed0/train_curve.csv` | 39 epoch 全量曲线（lr、loss、AP、任务数、耗时、显存） |
| `runs/pcba_seed0/checkpoint.pt` | 最优 Validation AP 权重（epoch 29），sha256 记入 `run.json` |
| `runs/pcba_seed0/selection.json` | 每个 epoch 的选择记录与 tie 规则 |
| `runs/pcba_seed0/per_task_ap.csv` | 逐 epoch × 逐任务 AP 与标签计数 |
| `runs/pcba_seed0/run.json` | 状态、资源、重载校验、缓存/标签/划分指纹、环境 |

训练环境：`res`（a100-2），driver 550.163.01，torch 2.5.1+cu124，python 3.12.14，
commit 见 `run.json`；数据在本地 CPU 构建后按 sha256 校验传输，训练前在 `res` 上
用 `pc_prepare.py --stage verify` 复核（30 s，`status = OK`）。
