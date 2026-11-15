# REPORT — `lcrc-semantic-analysis-repair-bw0-s0-v1`

分析修复 + 训练后机制核验 + 唯一新增训练 `B_N100_lipo_W0`。
所有数字都可用一条入口重生成：`code/rp_analysis.py`（默认只读旧产物，不触发训练）。
旧 20 条训练、旧配置、旧缓存、旧 checkpoint、旧机器表 **未改动**。

* 设计依据：`calendar0917/research@1d6700ea01acbee33e7630f1813c7884d28bee71`
* 本轮提交：`90d04295` → `a0f31420` → `8c3aaa2a`（干净、已 `rr deploy res`）
* 唯一新增训练：`B_N100_lipo_W0`，res GPU1，process backend，run `res-bw0-20261115-142629-25fabfe5`
* GPU：**0.19 A100·h**（正式训练 0.181 + 2-epoch smoke ≈0.01），硬上限 1.0
* 墙钟：约 2 h 40 min（含分析、修复、部署、训练、重算）
* 全部定向检查通过：`analysis/focused_checks.json` `all_ok = true`

---

## 0. 一句话结论

**B_W0 不能解释 LS_W0 的进步。** 在同一个 `B` 初始化内取消 weight decay 使 Val/Test 都
略微变差（+0.0032 / +0.0035），而在 `LS` 初始化内同样取消 weight decay 带来
Val −0.0120 / Test −0.0241 的改善；两个初始化对 wd0 的反应方向相反，
存在**条件依赖（交互）**。修复后 LS_W0 相对 B_W0 在 Val 上有 6.9%、Test 上有 3.8% 的点估计优势，
但 Test 的固定状态重采样区间跨 0（`UNCERTAIN`），不能写成已确立的泛化收益。

同时必须撤回三条旧叙述：**"机制成立"**（旧梯度核用的是新建模型 + `bank_raw`）、
**"编辑解释层成立"**（旧 split 标签 331/823 与官方 scaffold split 不符，
真正的留出 cell 只有 7 个分量）、**"稀有 code 是瓶颈"**（修复后 df 几乎全在 `>20` 桶，
三分位阈值退化）。

---

## 1. 修复了什么，错误是否进入训练

### 1.1 定位的缺陷（本轮全部实测，不是推测）

| # | 缺陷 | 证据 | 影响面 |
| --- | --- | --- | --- |
| A | 合并缓存里 `atom_ptr` 是分子→原子，`path_ptr`/`nbr_ptr` 是**原子**→occurrence；旧代码用**分子**编号索引后两者 | 旧 Train `path_tokens` 10 238 / `end_tokens` 277 312；同一 S2 occurrence 两个不一致计数。修复后 277 312 / 277 312，`nbr` 197 122 | 分析/导出 |
| B | `edit_response` 直接对 pair 误差取均值，LOW_SUPPORT 按 pair 数判定 | 见 §5 权重语义检查 | 分析 |
| C | `gradient_kernel` 新建模型、未加载 checkpoint、用 `bank_raw` 绕开 CodeTable 的固定 affine 与 FP 复合置换 | 见 §4，重放后 `dim J = 13 133` 且 init/best 分开 | 分析 |
| D | 导出 `checkpoint_best_lookup.pt` 前模型已被改成 `last`，key 布局不保证能被原始 C1 严格加载 | 见 §4.3，本轮重新导出并 `strict=True` 验证 | 导出 |
| **E** | **`se_train` 在建模型之后才 `torch.manual_seed`，C1 base 用进程全局 RNG** | `analysis/init_reproducibility.json`：3 个新进程得到 3 个不同的初始化 hash，且都不等于旧 checkpoint 记录的 `init_state_hash` | **训练输入** |

### 1.2 错误有没有进到训练里

* **A–D 只影响后处理/导出**。修复后重放两个生成器 checkpoint，Valid/Test raw `max|Δ| ≤ 3.5e-6`
  与存储的 `pred_*_best` 一致（§4.1），说明训练本体与存储预测没被 reader 缺陷污染。
* **E 是训练输入层面的缺陷**：旧 20 条训练的初始权重不可复现、也不可证明跨 family 共享。
  本轮按任务书要求"发现训练输入缺陷 → 停止受影响比较、记录明确范围、不自行重训 20 条"处理：
  * 所有以"同一非表初始状态"为前提的**旧 pairing 叙述**降级为**未建立**；
  * `B_N100_lipo_W0` 显式载入冻结的、有文档的 C1 初始状态（`build_full_state(seed=0)` →
    `remap_state('C1', …)`，即 `lp_model.shared_c1_init_state`），**与 HOME 轮原始 `B` 臂完全同一个初值对象**：
    HOME `analysis/results/B/metrics.json` 记录的 `c1_init_state_hash = 7d0884f7…`，
    本轮本地重算 `hash(shared_c1_init_state()) = 7d0884f7…`；B_W0 checkpoint 的
    `init_state_hash = 7f5d7570…` 也能被本地重建精确复现（两种 hash 只是 state_dict 前缀约定不同）。
  * `B_Rwd` 无法证明共享该初值，因此 **wd=1e-2 这条坐标的比较带明确初值告警**。

所以：**本轮需要重训，但不是重训 20 条，而是补 1 条可复现的 B_W0**。

---

## 2. 修正后的支持度统计（`support_corrected.json` / `support_before_after.csv`）

`layout_train` 断言通过：`n_molecules=3360`、`n_atoms=90514`、`path_occ=277312`、`nbr_occ=197122`。

| 指标（N100, Train） | 旧（错位读取） | 修正后 |
| --- | --- | --- |
| path occurrences | 10 238 | **277 312** |
| nbr occurrences | 7 281 | **197 122** |
| end occurrences | 277 312 | 277 312 |
| joint occurrences | 10 238 | **277 312** |
| 参考 df 大小 path/end/nbr/joint | 23 / 19 / 20 / 74 | **28 / 24 / 25 / 108** |
| path df∈1–5 占比 | 0.00098 | 0.00009 |
| path df∈6–20 占比 | 0.00469 | **0.0** |
| path df>20 占比 | 0.99433 | **0.99983** |

Valid/Test 同向：path occurrence 1 253→35 554、1 320→36 128。

**结论**：修复后这套 LCRC code 词表在这个数据集上**极稠密** —— 每个出现的 path code
在 Train 里几乎都被 >20 个分子用到。三个预算（N100/N50/N25）都一样：

| 预算 | Fit 分子 | Fit df (path/nbr/end/joint) | path df∈6–20 | joint df∈1–5 |
| --- | --- | --- | --- | --- |
| N100 | 3360 | 28 / 25 / 24 / 108 | 0.0 | 0.00096 |
| N50 | 1680 | 28 / 25 / 24 / 104 | 0.0 | 0.00114 |
| N25 | 840 | 24 / 21 / 21 / 91 | 0.0 | 0.00167 |

**风险分箱不可估**：Fit-LOO 的 `r_joint` 三分位阈值 `q1 = q2 = 0.0`（Train 3360 个分子中只有
41 个 `r_joint > 0`，最大值 0.382），按协议**保留 ties、不另找分位**，因此
`risk_bins.status = NOT_ESTIMABLE`。`has_no_path = 0` 在所有 split/预算下成立
（没有分子缺 S2），所以也不需要 `has_no_path` 旁列。
**由此，"稀有组合是不是瓶颈"这个问题在本轮不能回答** —— 修复后的输入里几乎没有稀有组合。

定向检查（`focused_checks.json`）：`layout` ✓、`collate_identity`（前 50 个 Valid 分子，
与 `PREP_DATA.collate` 逐分子 code Counter 一致）✓、`permutation_invariance`（分子重排 +
原子重编号后每分子计数不变、df 完全一致）✓、`oracle_agreement`（用**独立**的
`lcrc_data.build_molecule` 重建前 20 个 Train 分子，计数/joint/`n_s1`/`n_path` 全等）✓、
`small_molecules`（CC → S1=2/S2=0、CCO → 4/2、C → 0/0，期望值来自任务书而不是本 reader）✓。

---

## 3. 修正后的 CPU 显式统计坐标（`explicit_baseline_corrected.json`）

列定义、列顺序、Scaler、超参完全沿用旧实现，只修 per-molecule occurrence 范围。
矩阵 32 604 列，稀疏度 ≈ 0.9978；每行 path/nbr/end/closure/joint 计数与单分子 oracle **全部相等**
（`row_count_oracle.ok = true`）。Ridge 的列 scale 实际是 **Train 列正最大值**（`max>0 ? max : 1`），
不中心化；`global42` 是**原始** lipo_v1 值，不是"已按模型 scaler 标准化"。

| 数据集 / 模型 | 旧 Test RMSE | 修正后 Test RMSE | 修正后 Test MAE | 旧状态 |
| --- | --- | --- | --- | --- |
| lipo Ridge | 1.0457 | **0.9638** | 0.7742 | `INVALID_FOR_COMPARISON` |
| lipo XGBoost | 0.9869 | **0.8432** | 0.6671 | `INVALID_FOR_COMPARISON` |
| esol Ridge | 1.8274 | **1.0935** | 0.8339 | `INVALID_FOR_COMPARISON` |
| esol XGBoost | 1.5523 | **1.4209** | 0.9776 | `INVALID_FOR_COMPARISON` |

耗时：lipo 建矩阵 0.5 s + Ridge 0.4 s + XGB 2.8 s；esol 0.1 / 0.3 / 2.6 s。

XGBoost 的语义照实说明：稀疏零就是 XGBoost 的 **missing**，走学到的默认方向，
不是"数据存在缺测"。

**判定**：修复后两个固定坐标都明显变强（lipo XGB −0.144、esol Ridge −0.734），
按事实并列为新的性能坐标；但**仍明显弱于**神经路线（FS_R Test 0.7106、LS_W0 0.6982）。
它丢失 root 所有权，所以不称"完全同信息的神经 vs 显式对照"。
旧"CPU 参照已接近神经路线"的说法**撤回**。

---

## 4. 真实训练状态：重放、梯度核、纯查表导出

### 4.1 checkpoint 重放（`checkpoint_replay.json`）

固定 spec/norm、generator seed 17、π400/σ80，按训练时顺序重建，再 `strict=True` 载入原 `state_dict`。

| 模型 | Valid RMSE | Test RMSE | `max|Δ|raw` vs 存储预测 | 身份 |
| --- | --- | --- | --- | --- |
| FS_N100_lipo_R | 0.729531 | 0.710553 | 2.0e-6 | ✅ |
| FP_N100_lipo_R | 0.741194 | 0.697495 | 2.0e-6 | ✅ |

CPU 前向 vs 原 GPU 前向的跨设备差异在容差内，无需 GPU 复核。

### 4.2 修正后的梯度核（`kernel_init_best_corrected.json`）

`E(c)` 直接调用 **实际** `model.base.local.path_emb(c)`（含 θ0 固定 affine、
FP 的复合 code 置换、不 detach）；`s(c)=qᵀE(c)`；`J_c` 对本 bank 真正有影响的 13 133 维
generator 参数求导；方向为 seed 20261115 的 Rademacher/√32，单一方向。
code 组：修正后 Train df 只有 28 个 path code，全部使用（<64），FS/FP/init/best 共用同一组。

重叠数 0 / 1 / 2（按原始 mid/b1/b2 同位置重叠，排除对角线）：

| | 0 重叠 (n=152) | 1 重叠 (n=163) | 2 重叠 (n=63) |
| --- | --- | --- | --- |
| **FS init** | 0.0728 | 0.3007 | 0.6104 |
| **FS best(161)** | 0.0339 | 0.2487 | 0.5659 |
| **变化** | **−0.0389** | **−0.0519** | **−0.0445** |
| **FP init** | 0.1826 | 0.2171 | 0.1751 |
| **FP best(94)** | 0.1688 | 0.2038 | 0.1567 |
| **变化** | −0.0138 | −0.0133 | −0.0184 |

`diag_mean = 1.0`（自洽）。LS 自由表梯度核 `max|off-diag cosine| = 0.0`（实现核验 ✓）。

**能推到哪一层**：
1. 真实 codec 的生成器里，两个 code 的**参数梯度方向**随共享的原始组成成分数量单调分级
   （0 < 1 < 2），这在 **init 就已存在**；
2. 训练把三档耦合**全部压低**（约 −0.04 到 −0.05），但**分级顺序保持**；
3. **置换控制 FP 没有这个分级**（init 已接近平坦 0.18/0.22/0.18），且训练几乎不动它
   （−0.014）。所以分级是**真实组成语义**带来的，不是"任何重参数化都会有"的通用现象。

**不能推到哪一层**：这只能证明**参数耦合 + 语义对齐**。它不能单独证明任务收益，
不能证明化学因果，也不能替代 §6 的性能增量。旧数值（`INIT_RAW_DIAGNOSTIC`）不再是
"训练后机制证据"。

### 4.3 纯查表导出（`export_equivalence.json`）

从**重放通过**的 best 状态显式 reload 后物化六表，去掉 `generator.`/包装器前缀，
映射到 DISSECT 原始 `C1` 的 key/shape，`strict=True` 加载：

| 模型 | 参数 | Valid `max|Δ|raw` | Test `max|Δ|raw` | 等价 |
| --- | --- | --- | --- | --- |
| FS → native | 228 674 | 1.4e-7 | 7.2e-8 | ✅ |
| FP → native | 228 674 | 1.4e-7 | 1.1e-7 | ✅ |

新文件：`data/checkpoints/{FS_R,FP_R}/checkpoint_best_native_lookup.pt`。
旧 `checkpoint_best_lookup.pt` 未被覆盖。**注意**：旧导出既没做 reload-best，也没保证
key 布局可严格加载，所以旧导出的"等价证据"标 `EXPORT_INVALID`；这不否定生成器本身的训练性能。

---

## 5. 真实编辑响应（`edit_coverage.json` / `edit_metrics_corrected.json`）

### 5.1 覆盖：先映射，再指标

复用 EDIT 的 823 对与 mappings，不重挖、不放松合法性。按 **RDKit canonical SMILES** 唯一映射到官方
4200 行（pairs.csv 里的 `smiles_a/b` 不是 RDKit canonical 形式，直接字符串比对会失败），
核验标签与行号一致（823/823 无冲突），按无序 row pair 去重（0 重复），方向固定 `row_a < row_b`。

| 计数 | 本轮 | 任务书参照 |
| --- | --- | --- |
| 唯一边 | 823 | 823 ✅ |
| 端点 | 704 | 704 ✅ |
| 全图连通分量 | 231 | 231 ✅ |
| Train–Train | **791** | 791 ✅ |
| Train–Valid | **5** | 5 ✅ |
| Train–Test | **27** | 27 ✅ |
| 两端均非 Train | **0** | 0 ✅ |
| Train–Test 的 Test 分子 / 全图分量 | 7 / 7 | 7 / 7 ✅ |
| Train–Valid 的 Valid 分子 / 全图分量 | 4 / 4 | 4 / 4 ✅ |

**重要差异（必须记录）**：pairs.csv 自带的 `split_a/split_b` 列有 **331 / 823** 条与官方
scaffold split 不符。按它算会得到 TT=501、Train-Test=147、Valid-Test=19、Test-Test=12、Valid-Valid=5，
并凭空出现 36 个"两端都非 Train"的 pair —— 与参照的"HH=0"直接矛盾。
本轮 split 一律取**官方行归属**，未删任何样本。

### 5.2 指标（best-Valid checkpoint，component 等权）

`CE-MAE`：先在每个 component 内对 `e=|Δpred−Δy|` 取平均，再对**有 pair 的 component 等权**平均。
`pair-MAE` 单列，不称 component 等权。

| 模型 | TT CE-MAE (228 comp) | Train-Test CE-MAE | E_T CE-MAE | ALL CE-MAE (231 comp) |
| --- | --- | --- | --- | --- |
| B_R | 0.3057 | 0.4247 | 0.4247 | 0.3076 |
| **B_W0** | **0.2935** | **0.4230** | **0.4230** | **0.2964** |
| LS_R | 0.3137 | 0.4936 | 0.4936 | 0.3176 |
| LS_W0 | 0.3090 | 0.4993 | 0.4993 | 0.3132 |
| FS_R | 0.2963 | 0.4904 | 0.4904 | 0.2996 |
| FS_W0 | 0.3116 | 0.4608 | 0.4608 | 0.3132 |
| LP_R | 0.3387 | 0.4115 | 0.4115 | 0.3391 |
| FP_R | 0.3434 | 0.3868 | 0.3868 | 0.3420 |
| 零变化基线 `|Δy|` | — | — | — | 0.6457 |

Sign agreement（`|Δy| ≥ 0.1`，component 等权）：FS_R 0.905 > LS_W0 0.888 > B_W0/B_R 0.886 >
FS_W0 0.884 > LS_R 0.873 > LP_R 0.863。零变化基线的 `|Δy|`=0.6457 说明所有模型确实在动预测，
但这只检验"有没有变化"，不构成对 LS/B 的增量证据。

### 5.3 这个目录到底有多少独立留出支持

**`NEEDS_HELDOUT_EVIDENCE`**。真正的留出量是 **27 个 pair / 7 个全图分量 / 7 个 Test 分子**
（Train–Valid 更少：5 / 4 / 4），按 `component < 20` 规则**全部 LOW_SUPPORT**，只描述不给区间。
`TT`(228 comp) 与 `ALL`(231 comp) 支持足够，但**必须明确是训练端点主导**
（791/823 的 pair 有 Train 端点；`ALL` 的 693/704 端点是 Train 分子）—— 不是 823 个独立留出样本。

固定状态 500 次 component `Exp(1)` 共同权重扰动（seed 20261115，全模型共用权重）：

| cell | 比较 | ΔCE-MAE | 95% 区间 | 跨 0 |
| --- | --- | --- | --- | --- |
| ALL | FS_R − LS_R | −0.0180 | [−0.0322, −0.0042] | 否 |
| TT | FS_R − LS_R | −0.0174 | [−0.0314, −0.0025] | 否 |
| ALL | LS_R − LS_W0 | +0.0044 | [−0.0032, +0.0117] | 是 |
| TT | LS_R − LS_W0 | +0.0047 | [−0.0020, +0.0121] | 是 |
| ALL | B_W0 − LS_W0 | −0.0169 | [−0.0304, −0.0036] | 否 |
| TT | B_W0 − LS_W0 | −0.0155 | [−0.0297, −0.0027] | 否 |
| TT | FS_R − FS_W0 | −0.0152 | [−0.0269, −0.0022] | 否 |

**修复后的编辑证据怎么说**：
1. 在覆盖足够的 cell 上，**FS 仍优于 LS**（ALL −0.0180、TT −0.0174，区间不跨 0）——
   这条旧叙述的**方向**保留；
2. 但**最朴素的 `B_W0`（无生成器、无查表）在 TT 和 ALL 上都是最好的**，
   且显著优于 LS_W0 —— 所以"编辑解释层"不是语义共享带来的附加能力；
3. 唯一真正留出的 cell 只有 7 个分量，且在那 7 个分量上 **B_W0/B_R 优于 FS/LS**
   （0.4230 / 0.4247 vs 0.4608–0.4993）。**旧"编辑解释层成立"撤回为具体 cell/权重/覆盖支持下的表述。**
4. 重算后的 CE-MAE 与 pair-MAE 明显不同（例：B_R TT 0.3057 vs 0.2966），
   证明 component 等权不是 pair 等权；两个不等大小 component 的合成检查、
   交换 pair 两端不变、去重不变、共享端点同分量、LOW_SUPPORT 只看 component 数，全部通过。

---

## 6. 性能：两张完整表 + 初始化 × weight decay

两张表**并列全列，不择优拼接**。A = 旧协议的 50 点 update grid（Val 最小、同分更早）；
B = 每 epoch 最佳 Valid。`B_R` 当年没跑 grid，所以 A 列为空，不用 best-epoch 区间冒充 grid 区间。

### 表 A（grid 选点）

| arm | j | Valid RMSE | Test RMSE | Train RMSE |
| --- | --- | --- | --- | --- |
| B_R | — | （无 grid） | — | — |
| **B_W0** | 30 | 0.7470 | 0.7142 | 0.3035 |
| B_Rwd | 30 | 0.7130 | 0.7188 | 0.2782 |
| LS_R | 31 | 0.7115 | 0.7202 | 0.2994 |
| **LS_W0** | 34 | **0.6987** | 0.7108 | 0.2843 |
| LS_Rwd | 27 | 0.7046 | 0.7151 | 0.3266 |
| FS_R | 29 | 0.7357 | 0.7091 | 0.2479 |
| FS_W0 | 21 | 0.7337 | **0.6747** | 0.2976 |
| FS_Rwd | 21 | 0.7286 | 0.6802 | 0.2984 |
| LP_R | 23 | 0.7461 | 0.7040 | 0.3422 |
| FP_R | 13 | 0.7423 | 0.7051 | 0.3819 |
| FP_Rwd | 18 | 0.7440 | 0.6759 | 0.3305 |

### 表 B（每 epoch 最佳 Valid；用于和历史 B / checkpoint / 编辑结果对齐）

| arm | best ep | Valid | Test | Train | last Val/Test | late-20 Val/Test |
| --- | --- | --- | --- | --- | --- | --- |
| B_R | 180 | 0.7371 | 0.7211 | 0.3268 | 0.7530 / 0.7162 | 0.7526 / 0.7186 |
| **B_W0** | 223 | 0.7403 | 0.7246 | 0.2983 | 0.7552 / 0.7106 | 0.7540 / 0.7121 |
| B_Rwd | 216 | 0.7130 | 0.7188 | 0.2782 | 0.7303 / 0.7370 | 0.7321 / 0.7363 |
| LS_R | 213 | 0.7043 | 0.7223 | 0.3118 | 0.7147 / 0.7158 | 0.7146 / 0.7168 |
| **LS_W0** | 227 | **0.6924** | **0.6982** | 0.2964 | 0.7000 / 0.7028 | 0.7002 / 0.7048 |
| LS_Rwd | 278 | 0.7026 | 0.7159 | 0.2711 | 0.7055 / 0.7113 | 0.7058 / 0.7137 |
| FS_R | 161 | 0.7295 | 0.7106 | 0.2917 | 0.7540 / 0.7016 | 0.7519 / 0.7022 |
| FS_W0 | 136 | 0.7289 | 0.6919 | 0.3068 | 0.7659 / 0.7007 | 0.7653 / 0.7016 |
| FS_Rwd | 104 | 0.7270 | 0.7099 | 0.4054 | 0.7469 / 0.6819 | — |
| LP_R | 134 | 0.7385 | 0.7051 | 0.3757 | 0.7721 / 0.6877 | — |
| FP_R | 94 | 0.7412 | 0.6975 | 0.3852 | 0.7801 / 0.7084 | — |
| FP_Rwd | 104 | 0.7349 | 0.7000 | 0.3662 | 0.7651 / 0.6918 | — |

`late-window` 明确是**末 20 个 epoch（341–360）的 per-epoch 均值**，不是"末 5%"，也不是 ensemble。
（N25/N50 预算曲线仍在 `performance_tables.json`，此处不展开。）

### 6.1 初始化效应（同一配方内）

| 配方 | 表 | Δ = B − LS | Val | Test |
| --- | --- | --- | --- | --- |
| R | B | 0.7371 − 0.7043 = **+0.0328** (4.65%) | 区间 [+0.0016, +0.0637] 不跨 0 | −0.0012 跨 0 |
| W0 | B | 0.7403 − 0.6924 = **+0.0479** (6.92%) | [+0.0162, +0.0792] 不跨 0 | **+0.0264 (3.78%)**，区间 [−0.0095, +0.0583] 跨 0 |
| Rwd | B | 0.7130 − 0.7026 = +0.0104 (1.49%) | 跨 0 | +0.0028 跨 0 |
| W0 | A | 0.7470 − 0.6987 = **+0.0483** (6.91%) | — | +0.0034 (0.48%) |
| Rwd | A | 0.7130 − 0.7046 = +0.0084 (1.19%) | — | +0.0036 (0.51%) |

### 6.2 配方 / weight decay 效应（同一初始化内）

| 初始化 | 表 | 比较 | Val Δ | Test Δ |
| --- | --- | --- | --- | --- |
| **B** | B | B_R − B_W0 | **−0.0032**（wd0 更差） | **−0.0035**（wd0 更差） |
| **LS** | B | LS_R − LS_W0 | **+0.0120**（wd0 更好） | **+0.0241**，区间 [+0.0075, +0.0413] 不跨 0 |
| B | B | B_R − B_Rwd | +0.0241 | +0.0024 |
| LS | B | LS_R − LS_Rwd | +0.0018 | +0.0064 |
| FS | B | FS_R − FS_W0 | +0.0007 | +0.0186 |
| FS | B | FS_R − FS_Rwd | +0.0025 | +0.0007 |

**交互明显**：wd0 在 `LS` 上是 Val/Test 双赢（Test 区间不跨 0），在 `B` 上是 Val/Test 双输。
`FS` 的 wd0 在 Test 上也更好（+0.0186）、Val 基本持平 —— 所以"取消 weight decay 有帮助"
不是 LS 独有，但它**确实不是**B 能复制的。

其他闸门：B_W0 训练异常 `nonfinite_events = 0`；`grad_clip_events = 4887`（旧 B_Rwd 同类量级）；
`parameters = 228674`；目标/结构 norm、N100 epoch plan、`shuffle_offset 917000` 与模板一致。

---

## 7. 逐项回答

### Q1 哪些旧结论可保留、哪些撤回、哪些不确定

| 旧结论 | 状态 | 指向产物 |
| --- | --- | --- |
| 存储的 raw 预测与 `metrics.json` 一致（16 臂 × 6 组 = 96 行） | **保留** | `metric_recompute.csv`（96/96 匹配） |
| 生成器 checkpoint 与存储预测同身份 | **保留** | `checkpoint_replay.json`（≤3.5e-6） |
| 生成器可等价导出为纯查表 C1 | **保留（重做后）** | `export_equivalence.json`（≤1.4e-7）；旧导出 `EXPORT_INVALID` |
| 真实 codec 的 code 间梯度耦合随共享成分分级 | **保留，但只到参数耦合层** | `kernel_init_best_corrected.json` |
| "训练后机制成立" | **撤回** | 旧核是新建模型 + `bank_raw`；修正后 init 就分级、训练后反而下降 |
| "初始化有小而稳的 Test 收益"（LS_R 优于 B_R） | **撤回** | 表 B：LS_R Test 0.7223 vs B_R 0.7211（更差 0.0012，区间跨 0） |
| "没有性能进步"（整体） | **部分撤回** | 共享训练确实没有稳健同向增量，但 LS_W0 配置本身有进步（见 Q3） |
| "稀有 code 是瓶颈" | **撤回** | `support_corrected.json`：path df∈6–20 占比 0.0，df>20 占 99.98% |
| "编辑解释层成立" | **撤回为 cell 级表述** | `edit_coverage.json` + `edit_metrics_corrected.json`：留出只有 7 个分量 |
| "CPU 参照已接近神经路线" | **撤回** | `explicit_baseline_corrected.json`：修好后更强，但仍差 0.13–0.23 RMSE |
| 风险分箱/稀有切片结论 | **不可估** | 三分位阈值退化（q1=q2=0） |
| 跨 family 共享同一非表初值 | **未建立** | `init_reproducibility.json` |
| "通用重参数化可以解释一切" | **仍不确定** | FP 核没有分级，但 FP 表 A Test 0.6759/0.7051 也不差；一个置换不足以证明等价 |

### Q2 错误是否进入训练，为什么需要/不需要重训

见 §1.2。A–D 只伤分析/导出；E 伤训练输入（初值不可复现）。
因此：**不重训 20 条**（那会改科学设计、超预算），**只补 1 条初值可控的 B_W0**，
并把所有依赖"共享初值"的旧 pairing 叙述降级。

### Q3 B_W0 是否解释了 LS_W0 的改进？初始化与 wd 是否条件依赖

**没有解释。** 三条独立读数一致：

1. 同初始化内 wd 效应：`B_R → B_W0` Val +0.0032 / Test +0.0035（**变差**）；`LS_R → LS_W0`
   Val −0.0120 / Test −0.0241（**变好**）。
2. 同配方内初始化效应：表 B `B_W0 − LS_W0` = Val +0.0479（6.92%）/ Test +0.0264（3.78%）；
   表 A = Val +0.0483（6.91%）/ Test +0.0034（0.48%）。
3. 交互：wd0 的 Test 收益在 LS 是 −0.0241、在 B 是 +0.0035，差 0.028 —— 方向相反。

**条件依赖成立**：wd0 只在 free-table（LS/FS）初始化上帮忙，在原始 C1（B）上帮倒忙。
所以"LS_W0 的进步 = 取消 weight decay"这个竞争解释**被排除**。

**但**：`B_W0 − LS_W0` 的 Test 区间 [−0.0095, +0.0583] 跨 0，表 A 的 Test 差距只有 0.0034。
按固定分支规则：LS_W0 在 Val/Test 点估计都优于 B_W0 → **条件性初始化候选**
（seed0 / 单数据集 / grid-or-best 两种选点 / Test 区间跨 0），**附 `UNCERTAIN`**，
不自动上规模，也不写成等价证明。

### Q4 真实的独立留出支持有多少？修复后的 CE-MAE 是否仍支持旧叙述

**`NEEDS_HELDOUT_EVIDENCE`。** 27 个 pair / 7 个分量 / 7 个 Test 分子（Train–Valid 5/4/4），
全部 LOW_SUPPORT。`TT`/`ALL` 覆盖足够但是训练端点主导。
CE-MAE 在 TT/ALL 上仍显示 FS 优于 LS（区间不跨 0），**但最朴素的 B_W0 在 TT/ALL 上最好**，
并且在唯一真正留出的 7 个分量上 B_W0/B_R 优于 FS/LS。
所以"编辑解释层"必须退回成：**在训练端点主导的 cell 上，FS 相对 LS 有 −0.017 的 CE-MAE 改善；
在留出 cell 上方向相反且支持不足。**

### Q5 梯度分级在 init 已有多少、训练后变化多少、能推到哪层

init 已有完整分级（FS 0.073 / 0.301 / 0.610），训练后整体下降但顺序保持（0.034 / 0.249 / 0.566），
变化 −0.039 / −0.052 / −0.045。FP（置换控制）init 就接近平坦（0.183 / 0.217 / 0.175），
训练几乎不动（−0.014）。
**能推到**：真实组成 codec 下"共享成分 → 参数梯度同向"这一结构性事实，且它由真实语义而非任意重参数化带来。
**不能推到**：任务收益、化学因果、泛化。LS 的直接表梯度核 off-diagonal 恰为 0，
但这只验证实现，不构成"查表不共享"的科学对照。

### Q6 当前最好的 lookup / 显式参照是什么？下一次决策缺什么证据

* **最好的 lookup（两种选点口径一致）**：`LS_W0` —— 表 B 0.6924 / 0.6982，表 A 0.6987 / 0.7108。
* **最低 Test（不可选）**：`FS_W0` 表 A Test 0.6747，但 Val 0.7337（第 6）；按 Val 规则不能选它。
* **最好的显式 CPU 参照**：lipo XGBoost 0.8432 / esol Ridge 1.0935 —— 仍比 LS_W0 差 0.145。
* **最好的原生 C1**：`B_W0` 表 B 0.7403 / 0.7246（Val 第 5），但在编辑 TT/ALL 上最好。
* 跨臂最优是**开发面板选择**，不是独立最终验证。

**下一次决策最缺的**（按缺口感排序）：
1. **数据覆盖**：真正留出的结构编辑 pair 只有 7 个分量 → 需要新的留出编辑证据（本轮明确不做）。
2. **结构依赖证据**：`LS_W0` 的优势是否随标签预算（N50/N25）稳定、是否跨第二个数据集（ESOL）
   同向 —— 现有 ESOL 的 LS/FS 并未胜过原 B，所以"语义共享"的迁移性仍未建立。
3. **性能证据**：`B_W0 − LS_W0` 的 Test 区间跨 0，需要更多独立结构单元（不是更多 seed）
   才能把 3.8% 的 Test 点估计从 `UNCERTAIN` 变成可判读。

---

## 8. 交付物与可复现入口

```
tracks/structural-sufficiency/lcrc_semantic_analysis_repair_bw0_s0_v1/
  PROTOCOL.md
  configs/lcrc-semantic-analysis-repair-bw0-s0-v1.yaml     # 冻结配置（runner 拒绝漂移）
  protocols/lcrc-semantic-analysis-repair-bw0-s0-v1.yaml
  studies/lcrc-semantic-analysis-repair-bw0-s0-v1.yaml
  code/rp_{common,reader,source,support,explicit,replay,perf,edit,checks,init,train,runner,analysis}.py
  analysis/
    source_manifest.json            metric_recompute.csv
    support_corrected.json          support_before_after.csv      support_molecules.csv
    focused_checks.json             init_reproducibility.json
    explicit_baseline_corrected.json
    checkpoint_replay.json          kernel_init_best_corrected.json  export_equivalence.json
    pair_mapping_checked.csv        pair_components.csv
    edit_coverage.json              edit_metrics_corrected.json     edit_before_after.csv
    performance_tables.json         performance_tables.csv          analysis_run.json
    figures/fig1_kernel_grading.png figures/fig2_performance_panel.png
  runs/arm_B_N100_lipo_W0/          # metrics/curve/grid/pred_*/checkpoint_{best,last}/identity
  data/checkpoints/                 # 拉回的 FS/FP/B_Rwd/LS_Rwd/FS_Rwd best + 两个 native-lookup
```

一条命令重生成全部修复产物（只读旧产物，不会训练）：

```bash
uv run python -m rp_analysis            # 或: uv run python code/rp_analysis.py --sections source,support,...
```

远端执行（全部经 `rr`，无手写 ssh/sbatch/nohup）：

```bash
rr deploy res
rr run res <exp> --pool res-gpu1 --gpus 1 --cpus 4 -- \
  env RESEARCH_TRACK_ROOT=tracks/structural-sufficiency/lcrc_semantic_analysis_repair_bw0_s0_v1 \
      RESEARCH_EXTRA_RUNNER_PATHS=.../code RESEARCH_EXTRA_RUNNERS=rp_runner \
  uv run research run lcrc-semantic-analysis-repair-bw0-s0-v1 --set stage=train
```

## 8.1 两张图（只为改变理解，不做图数凑数）

* `analysis/figures/fig1_kernel_grading.png` —— 修正后的梯度核按共享原始组成成分数
  （0/1/2）的余弦：FS（真实 codec）与 FP（置换控制）在 **init** 与**各自 best** 两点的对比。
  它把 §4.2 的表读成一件事：**分级在 init 就已存在，训练把三档一起压低但顺序不变，
  而置换控制从头到尾接近平坦**。这是纯数字最难一眼看出的结论。
* `analysis/figures/fig2_performance_panel.png` —— 两种选点口径（旧 50 点 grid / 每 epoch 最佳 Valid）
  的 Valid–Test 散点，标出承载本轮判断的 10 个臂。它直观显示：**同一臂在两种口径下都赢**，
  而 FS 的 grid 与 best 的 Test 值差得最远 —— 这正是两张表都要全列的原因。

两图只读 `analysis/*.json`，不做任何重算或再选点；由 `rp_analysis` 的 `figures` 段重生成。

## 9. 明确缺口

* `B_R` 没有旧 grid 轨迹，表 A 缺失该行（不为恢复 grid 重训）。
* `FS_Rwd / LP_R / FP_R / FP_Rwd` 的 `late_window` 在旧 `metrics.json` 里字段缺失，表 B 留空。
* 风险箱、稀有 code 切片：受限于退化阈值，**本轮不可估**，不另找分位。
* `B_Rwd` 的初值无法证明与 B_W0 共享（§1.2 E），所以 wd=1e-2 坐标的比较带初值告警。
* ESOL 未重训、未新增模型；ESOL 侧只有 CPU 显式坐标被修复。

## 10. 停止

按任务书要求，本轮到此停止：不购买 PCBA/HIV，不追加 embedding / 消息传递 / 编辑 loss，
不设计下一轮。下一次研究决策需要什么证据已在 Q6 列出，由研究者决定。
