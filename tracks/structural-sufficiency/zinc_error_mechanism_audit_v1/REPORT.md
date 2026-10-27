# ZINC 强模型误差来源复核 — `zinc_error_mechanism_audit_v1`

**边界**：CPU-only；**official test 未加载**；**没有训练任何模型**；只复用已冻结的
DCC-L / BOSI / ZBO 逐分子预测、已冻结的 KSVD 长环审计表和仓库内唯一的严格
train-internal OOF（compact-v4-hinge）。official valid 是已被反复使用的开发集，
本报告所有 valid 数字都是**事后诊断**，不是确认集，也不能单独支撑泛化声明。

**结论先行**

1. **完美修复全部长环误差的上限**：把 35 个 label-penalty 行的绝对误差全部归零，
   DCC-L 4-seed 均值 MAE 最多降 **0.01308**（0.06992 → 0.05684，占总 AE 的 18.7%；
   九个冻结预测的范围 0.0112–0.0174，单 seed DCC-L 为 0.0113–0.0147）。但其中真正能归给 cycle 项的是 **0.00944**
   （0.06049），且 **0.00509 集中在单行 valid:0172**；去掉 0172 后只剩 **0.00434**。
   所以“理论上最多”是 0.013，**现实可及的环修复量级是 0.004–0.009**。硬上界能把
   DCC-L 均值推到 0.0568（低于 0.060），但 cycle-attributable 的 0.0094 只推到 0.0605，
   去掉 0172 后是 0.0656；把 0.060 的希望寄托在环修复上不可靠。
2. **剔除人工环惩罚后，完全未见局部环境仍是主要难点**：无惩罚子集 MAE 0.05890；
   其中 241 个 any-unseen radius-2 分子（25.0% 行）承担 **50.2%** 的无惩罚 AE，
   MAE 0.11840 vs 已见 0.03910（3.03×）。按分子大小分层后仍成立；signed residual
   只有 +0.0062，bias²/MSE = 0.001，**这是离散度/异方差问题，不是组级均值偏移**。
   它比环惩罚（18.7% 总 AE）更大，是当前第一误差来源。
3. **是否值得在新预注册中加强模型环特征：否，停止环特征搜索。** `global33` 已含
   **池化的环原子最小环直方图**（`ring_size3_frac`…`ring_size8plus_frac`，只统计
   环原子、按环原子数归一化；每原子 one-hot 在 `atom_feat`，不在 `global33`）。精确最长简单环和
   minimum cycle basis 尺度是置换不变的，但基本可由 `global33` 重建
   （valid R² = 0.943 / 0.892 / 0.999）；在严格 train-only OOF 里，它们对
   **9,628 个无惩罚分子**的难度条件信息是 **ΔR² = +0.00006、ΔSpearman = −0.0008**，
   全部表面增益都来自 372 个惩罚行，且在惩罚行内跨折极不稳定。
4. **最值得下一次正式 GPU 实验的未决假设**：不是新环特征，也不是新结构算子，而是
   **“4k→8k 的标签数量斜率是否在 10k 之外继续”**。这是 `zinc_dccl_bottleneck_v1`
   已记录、尚未购买的 10k→12k 标签数量斜率点（冻结 DCC-L，forward 不变，
   ~0.75 A100·h）。它区分“剩余误差是共享函数的估计方差（标签数受限）”与
   “该家族在官方协议下已到结构天花板”。如果外部标签不被允许，正确动作是**停止**，
   而不是再买一个结构/环特征实验。

---

## 0. 输入、口径与不可越过的边界

| 项 | 内容 |
|---|---|
| 预测 | DCC-L 4 seeds（`pred_B-R0-online-s0..s3`，4-seed 均值仅作诊断）、BOSI_B_s0、ZBO_R/N/S_s0 |
| seed | DCC-L 0–3；BOSI_B / ZBO_R/N/S 均为 seed 0；本审计不新增 seed |
| protocol | 不新增训练协议；CPU audit over frozen predictions；valid 是已暴露开发集 |
| 目标 provenance | GVAE：`y = z(logP) + z(SA) + z(cycle)`，`cycle = −max(0, max(nx.cycle_basis(G)) − 6)`，在生成器的 RDKit canonical 原子上计算；非 PyG docstring 的“>6 元环个数” |
| `label_penalty` | `label_excess = max(0, basis_max − 6) > 0`，35/1000 行 |
| `any-unseen` | 分子至少一个原子的 radius-2 签名在 official train 10k 中从未出现（`r2_unseen_atom_frac > 0`），251/1000 行 |
| `low4` | 无 unseen，但含 train 支撑 1–4 的 radius-2 原子，366 行 |
| `full` | 所有 radius-2 原子支撑 ≥5，383 行 |
| `tail` | `y ≤ −3`，65 行 |
| `long7` | 精确最长简单环 ≥7，314 行；置换不变 |
| 计算 | 2 个 CPU 脚本，合计 <20 s；无 GPU；无 official test；历史冻结记录未改 |

**标签隔离**：`logP`、`SA`、`z_cycle`、`label_excess`、`y_without_cycle` **只用于分组或
诊断目标**，从未作为模型输入。OOF 诊断的输入只有 rarity / structure / topology
（`topo_*`）等图导出量。第 3 节的 `global33 + ring scale → y` 是诊断回归，用来检验
“图信息是否存在”，不是模型提案。

**复现**：

```bash
uv run python tracks/structural-sufficiency/zinc_error_mechanism_audit_v1/code/zema_audit.py
uv run python tracks/structural-sufficiency/zinc_error_mechanism_audit_v1/code/zema_decomposition.py
```

---

## 1. Q1 — 当前强模型绝对误差的分解

### 1.1 单组（可重叠）贡献

DCC-L 4-seed 预测均值（诊断集成，MAE 0.06992）；历史 headline 0.07599 是 4 个
per-seed MAE 的平均。`AE share` 是该组绝对误差之和 / 总绝对误差；`signed` 是
`mean(pred − y)`。单 seed / 各模型的硬上界见结论先行 1 与 `analysis/cycle_attribution.csv`，结论同向。

| 组 | n | MAE | AE share | signed | 读法 |
|---|---:|---:|---:|---:|---|
| 全部 | 1000 | 0.06992 | 100% | +0.0068 | — |
| label-penalty | 35 | 0.3738 | **18.7%** | +0.2696 | 3.5% 行，误差质量约 5.3× 行占比 |
| penalty severe（excess≥2） | 5 | 1.6648 | **11.9%** | +1.4286 | 5 行里 0172 一行 7.3% 总 AE |
| tail `y≤−3` | 65 | 0.2310 | 21.5% | +0.1624 | 其中 18 行同时 penalty |
| no-penalty tail | 47 | 0.0815 | **5.5%** | +0.0223 | 去掉惩罚后“极端标签”并不特别难 |
| any-unseen | 251 | 0.1333 | **47.8%** | +0.0220 | 25.1% 行，误差质量约 1.9× 行占比 |
| no-penalty unseen | 241 | 0.1184 | **40.8%** | +0.0062 | 最大单一误差块 |
| low4（已见 1–4） | 366 | 0.0663 | 34.7% | +0.0078 | 行数多，per-row 只 1.7× full |
| full（支撑≥5） | 383 | 0.0318 | 17.4% | −0.0041 | 已低于 0.060 |
| exact longest ≥7 | 314 | 0.1130 | **50.8%** | +0.0087 | 主要是组大；per-row 只 1.6× |
| exact longest ≥10 | 144 | 0.1352 | 27.9% | +0.0024 | 同上 |
| RDKit max ring >6 | 31 | 0.1647 | 7.3% | +0.0628 | 31/31 都是 penalty；不是独立结构组 |
| MCB max >6 | 31 | 0.1647 | 7.3% | +0.0628 | 同上 |
| mean max-basis >6 | 37 | 0.3560 | 18.8% | +0.2546 | 35/37 是 penalty；最接近的置换不变代理 |

关键点：**“long-ring proxy 占 50.8% AE”不能读成“长环结构机制占一半误差”**。
`exact longest ≥7` 有 314 行，其中只有 35 行是 penalty；它的 per-row MAE 只比
非 long 组高 1.6×。真正与 penalty 对齐的窄代理（RDKit/MCB >6）只有 31 行，
几乎就是 penalty 子集本身（precision 1.0，recall 0.886），因此不能作为独立证据。

### 1.2 MECE 分解：penalty × unseen × long7

每个分子只进一格；`long7 = exact longest ≥7`，是置换不变的长环代理。DCC-L 4-seed 均值：

| penalty | unseen | long7 | n | MAE | AE share | signed | 其中 tail n / AE share |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 520 | 0.0363 | 27.0% | −0.0024 | 22 / 1.5% |
| 0 | 0 | 1 | 204 | 0.0464 | 13.5% | −0.0140 | 8 / 0.6% |
| 0 | 1 | 0 | 166 | 0.0939 | 22.3% | +0.0321 | 14 / 2.6% |
| 0 | 1 | 1 | 75 | 0.1727 | 18.5% | −0.0512 | 3 / 0.8% |
| 1 | 0 | 1 | 25 | 0.3264 | 11.7% | +0.2158 | 9 / 9.1% |
| 1 | 1 | 1 | 10 | 0.4923 | 7.0% | +0.4043 | 9 / 6.9% |

没有 `penalty=1, long7=0` 的格子：35 个 penalty 行的最长简单环全都 ≥7。
其余冻结预测（BOSI_B、ZBO_R/N/S、DCC-L 单 seed）的 AE share 排序一致：
无惩罚已见非 long 26–29% > 无惩罚未见非 long 21–23% > 无惩罚未见 long 17–20%
> penalty 已见 8–16% > penalty 未见 5.5–7.5%。所以结论不是 DCC-L 特例。

**分解读法**：
- **极端标签**本身只贡献 5.5% 总 AE（47 个 no-penalty tail，MAE 0.0815，仅比普通
  无惩罚 0.0577 高 1.4×）；18 个 penalty-tail 行贡献 16.0%，才是“极端负标签”的主体。
- **人工环惩罚**贡献 18.7% 总 AE；其中 16.0% 在 penalty-tail，只有 2.7% 在
  penalty-ordinary（17 行，MAE 0.1115）。惩罚的伤害几乎全部来自它与极端负标签的交集。
- **长环代理**（broad exact-longest）贡献 32.0% 总 AE（no-penalty long7 279 行），
  但 per-row 只比非 long 高 0.030（0.0803 vs 0.0502）；它不是一个独立的“长环难”机制。
- **radius-2 any-unseen** 贡献 40.8% 总 AE（no-penalty），是最大的单一 MECE 块；
  low4 只贡献 24.8%（无惩罚），per-row 1.66× full，远弱于 any-unseen 的 3.03×。

### 1.3 惩罚的严重度与 cycle attribution

对每个严重度组，用 `λ = 1 + mean(residual) / mean(cycle_component)` 估计模型对
cycle 项的捕获率；`cycle_delta_mae` 是“把未捕获的 cycle 项按组均值修好”能降的 MAE，
`hard_ceiling` 是把该组所有 AE 归零的硬上界。DCC-L 4-seed 均值：

| 组 | n | mean cycle comp. | mean residual | λ | AE mass | cycle 占组 AE | hard ΔMAE | cycle ΔMAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| excess=1 | 30 | −3.468 | +0.0765 | 0.978 | 4.758 | 48.2% | 0.00476 | 0.00229 |
| excess=2 | 4 | −6.936 | +0.5122 | 0.926 | 3.230 | 63.4% | 0.00323 | 0.00205 |
| excess≥4 | 1 | −20.810 | +5.0942 | 0.755 | 5.094 | 100% | 0.00509 | 0.00509 |
| 全部 penalty | 35 | −4.360 | +0.2696 | 0.938 | 13.082 | 72.1% | **0.01308** | **0.00944** |
| 全部 penalty − 0172 | 34 | −3.876 | +0.1277 | 0.967 | 7.988 | 54.4% | 0.00799 | **0.00434** |

- 模型对 `excess=1` 已捕获 97.8%，所以 30 行只贡献 4.76% 总 AE；其误差里约一半
  不是 cycle 项（chem/noise）。
- `excess≥4` 只有 1 行（valid:0172，y=−20.34，pred=−15.25，AE=5.09），却占
  penalty AE 的 38.9%、总 AE 的 7.3%。它是 4/35 个 order-sensitive 行之一，
  但 50 次随机序中 80% 仍给 label=6（均值 4.8），所以不是纯噪声，而是模型对
  severe penalty 的欠捕获；单点 λ 仍不能当作稳定机制。
- **硬上界 0.01308 把所有 penalty 行上的 chem error 也算成“环误差”**；真正
  cycle-attributable 的是 0.00944。去掉 0172 后只剩 0.00434。因此：
  - 理论硬上界：**0.0131**；
  - 现实的 cycle 修复量级：**0.0094**；
  - 不含单行 0172 的可迁移量级：**0.0043**。
  先前 KSVD 长环审计在 compact-v2（MAE 0.184）上测得：精确 label cycle 项的
  isotonic oracle 是 +0.0247，最佳置换不变代理只有 +0.0117（约一半）。把这个
  “约一半”的比例借到 DCC-L 的 0.0094 上，现实可及约 **0.005**；这与“0.004–0.009”
  的量级一致，而不是 0.013。

---

## 2. Q2 — penalty × unseen 交叉，大小混杂与 signed residual

### 2.1 交叉表（DCC-L 4-seed 均值）

| 组 | n | MAE | AE share | signed | mean y |
|---|---:|---:|---:|---:|---:|
| penalty × unseen | 10 | 0.4923 | 7.0% | +0.4043 | −5.74 |
| penalty × seen | 25 | 0.3264 | 11.7% | +0.2158 | −3.94 |
| no-penalty × unseen | 241 | 0.1184 | **40.8%** | +0.0062 | −0.31 |
| no-penalty × seen | 724 | 0.0391 | 40.5% | −0.0057 | +0.36 |

- **primary read**：`penalty × seen` 仍然 MAE 0.326，说明“环境已见”并不能救惩罚行；
  但 `no-penalty × unseen` 的 40.8% AE 远大于 penalty 的 18.7%，是更大的误差块。
- **signed residual**：no-penalty unseen 的 +0.0062 相对 MAE 0.118 很小；
  `bias²/MSE = 0.001`（99.9% 是方差），而 penalty unseen 的 `bias²/MSE = 0.221`。
  前者不能靠组级平移修正；后者有可辨识的欠预测，但只有 10 行。
- 各冻结预测一致（DCCL 4 seeds + BOSI + ZBO 4 臂）：no-penalty unseen AE share
  39.0–43.7%，MAE 0.118–0.142；no-penalty seen MAE 0.039–0.047。

关键 cell 的跨模型复核（MAE；最后一列是 no-penalty unseen 的 AE share）：

| model | penalty×seen | penalty×unseen | P0×seen | P0×unseen | P0×unseen AE share |
|---|---:|---:|---:|---:|---:|
| DCCL_mean | 0.3264 | 0.4923 | 0.0391 | 0.1184 | 40.8% |
| BOSI_B_s0 | 0.5142 | 0.4499 | 0.0426 | 0.1419 | 41.5% |
| ZBO_R_s0 | 0.3263 | 0.4565 | 0.0434 | 0.1352 | 42.5% |
| ZBO_N_s0 | 0.4154 | 0.5285 | 0.0471 | 0.1320 | 39.0% |
| ZBO_S_s0 | 0.2399 | 0.5203 | 0.0434 | 0.1375 | 43.7% |

BOSI 的 penalty seen/unseen 都更差，ZBO_S 的 penalty seen 最好；但“penalty 行远差于
非 penalty、no-penalty unseen 远差于 seen”的排序在所有模型上不变。ZBO_N 是
global mean-field 臂，它的 no-penalty unseen AE share 最低（39.0%）但 MAE 不低，
说明 AE share 同时受组大小影响，不能单独引用。

### 2.2 分子大小控制

按 no-penalty 子集的训练切点 21/25 分三个 size 档；每格给 MAE（n）：

| size bin（原子数） | P0 U0（已见） | P0 U1（未见） | P1 U0 | P1 U1 |
|---|---|---|---:|---:|
| 10–20（n=266） | 0.0471 (152) | 0.1113 (107) | 0.1148 (4) | 0.1342 (3) |
| 21–24（n=353） | 0.0368 (279) | 0.0932 (66) | 0.9275 (6) | 0.4133 (2) |
| 25–36（n=381） | 0.0372 (293) | 0.1540 (68) | 0.1423 (15) | 0.7387 (5) |

- 在每个 size 档内，no-penalty unseen 都是 seen 的 **2.4–4.1×**；
  no-penalty unseen 的平均分子大小（21.6）反而**小于** seen（23.5），
  所以不是“未见组只是更大”造成的。
- penalty 的 size 分层格 n 只有 2–15，不能做稳定的组内比较；但方向是
  penalty 行在每个 size bin 都更差，且 penalty 的误差集中在 tail/严重度，
  不是单纯 size。
- low4（已见但低支撑）在 no-penalty 下 MAE 0.0491，只有 full 0.0296 的 1.66×；
  any-unseen 才是 3.03×。所以“低支撑”本身不是强风险标志，“完全未见”才是。

---

## 3. Q3 — `global33` 的真实定义与环尺度的“新增图信息”

### 3.1 `global33` 到底有什么

来源：`tracks/structure-contribution/code/structural_relation_objects_v1.py::_global_features`
（`GLOBAL_FEAT_NAMES`，33 维，fixed layout）。环相关的是：

- `cyclomatic_rank` = n_bonds − n_atoms + 1；
- `ring_atom_fraction`、`ring_bond_fraction`、`n_ring_systems`、`log1p_n_ring_atoms`；
- `ring_size3_frac` … `ring_size8plus_frac`：**每个环原子的 minimum ring size 的池化直方图**
  （非环原子不计入），按环原子数归一化；
- `fused_ring_atom_fraction`：环键数 ≥3 的原子比例；
- 另有两个距离量 `mean_distance` / `mean_pair_distance`（实现里二者相同）。

**因此**：`global33` **已经包含池化的环原子最小环分布特征**，但**不包含**
每原子 one-hot 向量；每原子 `min_ring_size` one-hot 在 `atom_feat`
（`atom_blocks()["min_ring_size"]`），不在 `global33`。`global33` 也**不包含**
精确最长简单环、MCB max/total、mean max-basis。审计脚本把这一点写进了
`analysis/audit.json::global33.meta`，避免以后再把“池化直方图”误读成“每原子向量”。

### 3.2 这些尺度是否真的提供新增图信息

用 `train 10k global33 → valid 1000` 的 HistGradientBoosting 诊断（监督映射，
只用图特征，不用标签）：

| target | valid R² | valid MAE | valid Spearman |
|---|---:|---:|---:|
| exact longest simple cycle | 0.943 | 0.165 | 0.859 |
| MCB max | 0.892 | 0.010 | 0.484 |
| MCB total | **0.999** | 0.026 | 0.993 |
| mean max-basis（随机序均值） | 0.747 | 0.031 | 0.488 |
| frac orders with basis ≥7 | 0.913 | 0.011 | 0.314 |

- MCB total 几乎是 `global33` 的确定性函数（R² 0.999），没有独立图信息。
- exact longest 的 5.7% 残差方差很小；结合 §1.1，exact-longest 组绝大多数是
  非 penalty 的稠合环，因此这点残差不是 penalty 信号；`global33` 的
  fused/ring-size 块已经覆盖大部分。
- `mean max-basis` 的 R² 最低（0.747），是“最不冗余”的一个；但它本身是随机序
  均值，是 penalty 的代理，不是通用的长环结构特征。

再用 **train-only 5-fold OOF** 检验“把环尺度加进 `global33` 是否更能预测标签”：

| 特征 | 子集 | n | MAE | R² |
|---|---|---:|---:|---:|
| global33 | all | 10000 | 0.9941 | 0.5063 |
| global33 + ring scale | all | 10000 | 0.9676 | 0.5573 |
| global33 | no-penalty | 9628 | 0.9572 | 0.4738 |
| global33 + ring scale | no-penalty | 9628 | 0.9433 | 0.4895 |
| global33 | penalty | 372 | 1.9508 | −0.1789 |
| global33 + ring scale | penalty | 372 | 1.5963 | 0.1945 |

- 加环尺度对 no-penalty 的 MAE 改善是 **−0.0138**（5 折每折 −0.0019…−0.0326，
  方向稳定），说明它们对“只给 33 维全局描述”的弱基线**不是完全冗余**。
- 但这个增益有两个限定：(i) 基线是 `global33`-only 的 HGB，no-penalty MAE 0.957，
  是 DCC-L 0.070 的 13.6×；0.014 的改善相对该基线只有 1.4%；(ii) 下一节的 compact OOF 已经含 49 个 local/global topology 特征，
  在那里 no-penalty 增益是 **0**。所以“global33 缺它”不等于“强模型缺它”。
- penalty 子集的 R² 从 −0.179 提到 +0.195，主要是环尺度在识别/量化 penalty；
  这是标签机制诊断，不是通用结构收益。

### 3.3 置换不变性

- 精确最长简单环、MCB max/total/mean、`mean_maxbasis`、`frac_ge7` 都是
  **原子置换不变**；KSVD 审计的 5×10 / 20×20 次检查均无变化。
- `nx.cycle_basis` 的**存储顺序** max 不是置换不变的：KSVD 655 行里 34 行不稳定
  （全部 y<−4）；本审计对 35 个 valid penalty 行做 50 次随机原子置换，**4 行**
  出现多个 excess 值：0172（label 6，随机序均值 4.8，80% 匹配）、0214（2/1.64/82%）、
  0224（1/1.08/92%）、0267（1/1.30/70%）。**31/35 完全稳定，且 4 个不稳定行的
  label 都是随机序分布的众数（0172 为 80%）**；所以这更像离散/异质函数，
  而不是不可学的随机标签。
- 结论：候选特征必须是置换不变的；但 label penalty 本身是一个
  basis-order/canonicalisation 的离散函数。置换不变特征可以逼近它的众数，
  不能保证逐位复现生成器顺序。这属于“函数难学/离散”，不是“信息不存在”。
  0172 同时是最难的行和 order-sensitive 的行：它的 label 是众数（80%），
  但仍有 20% 的随机序给 0；模型即使在 80% 的情形下也欠捕获，所以主因是
  severe penalty 的函数难度，order 离散只是附加项。

---

## 4. Q4 — 只做误差机制诊断的 target provenance

本审计严格遵守：

- `logP`、`SA`、`z_cycle`、`label_excess`、`y_without_cycle` **没有进入任何模型输入**；
- 它们只用于：定义 penalty/tail 分组、计算 λ/cycle attribution、解释 0172；
- OOF 诊断输入只有 rarity、size/structure、`topo_*`（图导出量）；
- `global33 + ring scale → y` 是诊断回归，目标是标签，但输入只有图特征；
- 没有用 valid 上的事后相关性去声称“加环特征能泛化”；第 5 节专门说明为什么不能。

---

## 5. Q5 — 严格 train-internal OOF 说了什么

仓库里唯一的严格 label-isolated train-internal OOF 是
`tracks/ksvd/results/oof_difficulty_audit/`：compact-v4-hinge，official train 10k
的 5-fold OOF，inner 7200/800，两个 seed，official valid/test 未加载。**它不是
DCC-L 的 OOF**（模型 MAE 0.175 vs DCC-L 0.070），所以不能把它的结论直接搬到强模型。

把 `topo_longest_simple_cycle`、`topo_mcb_max/mean/total`、`mean_excess`、
`frac_ge7` 加进已有的 difficulty Ridge（rarity + structure + topo_nonring），
按 penalty 状态拆开：

| 子集 | variant | n | R²(log AE) | Spearman(pred, AE) | top-quintile MAE |
|---|---|---:|---:|---:|---:|
| all | base（49 特征） | 10000 | 0.1735 | 0.4019 | 0.3300 |
| all | + new ring（53） | 10000 | 0.1894 | 0.4048 | 0.3679 |
| all | + mean_excess（51） | 10000 | 0.1936 | 0.4065 | 0.3691 |
| **no-penalty** | base | 9628 | 0.1597 | 0.3861 | 0.2501 |
| **no-penalty** | + new ring | 9628 | **0.1597** | **0.3853** | 0.2494 |
| **no-penalty** | + mean_excess | 9628 | **0.1595** | **0.3862** | 0.2503 |
| penalty | base | 372 | 0.3086 | 0.4050 | 2.9311 |
| penalty | + new ring | 372 | 0.2103 | 0.3869 | 2.4295 |
| penalty | + mean_excess | 372 | 0.3253 | 0.3944 | 3.0975 |

- **全部表面增益都来自 372 个 penalty 行**。在 9,628 个无惩罚行上，
  `ΔR² = +0.00006`、`ΔSpearman = −0.0008`；5 折的 ΔR² 在 −0.0017…+0.0013 之间
  符号翻转。`mean_excess` 也一样（ΔR² = −0.00017）。
- 在 penalty 行内，加 4 个新环特征的跨折 ΔR² 介于 −0.269 与 +0.029 之间，极不稳定；`mean_excess` 也有 −0.485 的折。全行 top-quintile MAE
  反而变差（0.330→0.368）。这不是“稳定的条件预测信息”。
- 因此：**没有严格 OOF 证据支持“新增环尺度量对强模型残差有稳定的条件预测信息”**。
  更不能从复用 official valid 的事后相关性声称泛化收益。强模型 OOF 不存在，
  这条线只能以“没有证据”收尾，不能以“有希望”收尾。

---

## 6. Q6 — 三种竞争解释的区分

| 解释 | 可检验预测 | 本审计证据 | 判定 |
|---|---|---|---|
| **环信息缺失** | `global33` 的池化 min-ring 不足；加 exact longest/MCB 应显著改善长环 non-penalty 行 | `global33` 已有池化 min-ring 直方图；每原子 one-hot 在 `atom_feat`；尺度可由 `global33` 重建 R² 0.89–0.999；no-penalty long7 的 per-row 超额只有 +0.010（seen）/+0.079（unseen）；OOF no-penalty ΔR²≈0 | **不是主因**；只对 global33-only 弱基线有 0.014 MAE 的增量，对强模型无证据 |
| **环信息存在但函数难学** | penalty 行即使环境已见也难；误差集中在 severe/离散行；order-sensitive | 25/35 penalty 行环境已见，MAE 0.326；λ 0.978/0.926/0.755；4/35 order-sensitive（label 均为众数），0172 占 penalty AE 39%；penalty-tail 16.0% AE vs penalty-ordinary 2.7% | **对 penalty 切片成立**，但只占 18.7% 总 AE，且通用环特征修不了它 |
| **未见局部环境监督不足** | 去掉 penalty 后 unseen 仍主导；size 控制后仍成立；low4 温和；更多标签应帮助所有支撑层 | no-penalty unseen 241 行、50.2% 无惩罚 AE、MAE 3.03× seen；每个 size 档内 2.4–4.1×；low4 仅 1.66× full；DCC-L 4k→8k 对“仍未见”分子改善 −0.042/−0.038，与“新覆盖”−0.041/−0.059 同量级 | **是当前第一误差来源**；但“监督不足”指共享函数的估计方差，不是“给未见环境补查表项” |

**三者不是互斥的，排序是**：

> 未见局部环境的估计方差（40.8% 总 AE，第一）
> > 人工环惩罚的离散/严重度函数（18.7%，第二，且 0.005 集中在 1 行）
> > 环信息缺失（无证据，仅对 global33-only 弱基线有小增量）。

这也与 `zinc_dccl_bottleneck_v1` 的 A 级证据一致：4k→8k 的 −0.038 在“新覆盖”、
“仍未见”、“4k 已完全支撑”三层都接近，说明多标签改善的是共享函数的估计精度，
不是覆盖补洞；环复杂度在控制“未见比例 + 原子数 + |y|”后 partial ρ 只有 0.16。

### 6.1 与既有审计的一致性（避免把旧负结果重新包装）

- `zinc_generalization_audit_v1`：P3B 上 25.1% 分子有任何 unseen radius-2 环境，
  占 49.8% AE；path-unseen 只有 12.1% 分子 / 27.5% AE；low4 的 MAE 0.0616 远低于
  unseen 0.1513。本审计在 DCC-L 上得到 241/965 = 25.0% 的无惩罚 unseen、50.2% 的
  无惩罚 AE，方向与量级一致。**path token 覆盖不是问题；exact radius-2 环境未见才是。**
- `results/phase1e`：Monitor 上 43/1000 行 label-penalty，占各 SSRC 臂 29–55% AE；
  4/43 行在 30 次随机置换下改变 penalty。本审计在 official valid 上得到 35/1000、
  18.7% AE、4/35 order-sensitive；行数和比例不同（不同 split），但“penalty 行
  集中、少数行 order-sensitive”的结论一致。Phase 1e 的强基线 0.0911 与分组表还显示：
  没有 SSRC 臂在大环组超过强基线 `G_ref`，大环专门化没有缩小与强基线的差距。
- `zinc_block_ownership_v1/v2`：v1 仍是 `UNRESOLVED_DUE_TO_VARIANCE`，v2 仍是
  `SINGLETON_CONTENT_EXPLAINS_THE_GAIN`；本审计不改变两者。v2 的
  `d_RS = −0.000945`（S vs R）说明多键块归属没有可分辨增量；v1 的最大贡献行 0172 是 `n_multi=1`，S≡R，且它是 penalty 行。
  因此 **0172 的 5.09 AE 不能被多键块归属或任何 ZBO 块机制解释**，也不能反过来
  当作“环特征有用”的证据。ZBO_R/N/S 在本审计中只用于确认 penalty/unseen 的
  误差排序在不同强模型上同向，不作为新方向。

---

## 7. 决策：停止环特征搜索；下一次 GPU 实验只买“标签数量斜率点”

### 7.1 明确停止

- **停止**在强模型里加入 exact longest / MCB / mean_maxbasis / 任何新的全局环尺度。
  理由不是“环信息一定不存在”，而是：`global33` 已有池化 min-ring；这些尺度大部分
  可重建；唯一不冗余的 `mean_maxbasis` 是 penalty 代理；严格 OOF 在 no-penalty 上
  条件信息为 0；剩下 18.7% 的 penalty 误差里 0.005 集中在单行 0172；少数行是
  basis-order 离散选择（label 多为众数），不是缺一个全局环尺度。为一个 0.004–0.009 上限、且大部分不可迁移的切片再买一个
  结构特征，不符合证据门槛。
- **不**提出新的环/长环/层次环/环组合架构；不把本轮负结果改写成“换一种环特征再试”。
- **不**读 official test；不重训 DCC-L/BOSI/ZBO；不改 v1/v2 冻结记录。
- 若有人主张“先在强模型上试一下”，先看功效：可迁移的 penalty 切片上限是
  0.004–0.009，且 0.005 在单行 0172；seed/execution drift 带宽是 0.0015–0.0025。
  要在 3.5% 的行上分辨这个量级，需要很大的配对样本；即使成功，也只动 18.7% 的
  AE 中的一部分。结论是 **stop**，不是“再试一次”。
- **什么会推翻这个 stop**（预先写明，避免以后移动门柱）：(i) 强模型自己的
  train-internal OOF 上，环尺度在 **no-penalty** 行的 ΔMAE ≥ 0.002 且 5 折同号；
  (ii) 一个置换不变环特征在 penalty 行上把 AE 降低 > 0.003，且 leave-one-out
  去掉 0172 后仍 > 0.002；(iii) 新的 label provenance 证明 penalty 是置换不变的
  （当前不是）。在这些出现之前，不买。

### 7.2 唯一值得的下一次正式 GPU 实验

**未决假设 H**：`zinc_dccl_bottleneck_v1` 测到的 4k→8k 标签斜率（locked −0.0377/−0.0369，
valid −0.0396/−0.0409）是“小数据局部效应”，还是会在 10k 之外继续？换句话说：
当前 DCC-L 家族的剩余误差是**标签数量/共享函数估计方差**，还是**结构天花板**？

这是本轮唯一同时满足以下条件的假设：

1. 有 A 级配对干预证据（4k→8k，两 seed，方向一致）；
2. 其外推尚未被实验区分（8k→10k 只是描述性 −0.017，10k→12k 未跑）；
3. 不需要新架构、不需要新环特征、不需要 test；
4. 判定规则可以先写死，成本 <1 A100·h。

**设计（已记录在 `zinc_dccl_bottleneck_v1` §E，不是本审计新造的方向）**：

- 冻结 DCC-L `B-R0`，494,193 参数，同 SSL、同 R0、同 47,400 updates、Fit-only 标准化；
- 10k（官方 train）vs 12k（官方 10k + 2k 同分布 ZINC，逐分子用 WL 指纹 + 精确图同构
  排除与 official valid/test 同构者）；
- 2 条轨迹，~0.75 A100·h；official test 全程不读；
- 预注册判定：`Δlocked(12k−10k) ≤ −0.010` → 斜率在 10k 外仍有效，剩余误差是标签数量
  问题；`≥ −0.004` → 4k→8k 斜率是局部的，家族在官方协议下以 0.0663 收尾；
  两者之间 → 只补一个 11k 点。
- **最强反对证据**：外部标签会改写 benchmark 口径，本 track 历史上禁止；若这条不可接受，
  正确动作是**停止**，不买任何新 GPU 实验。不要因为“预算还没花完”去换一个环特征或
  结构算子。

### 7.3 如果只能给一句话

**环特征搜索到此为止；真正还没被区分的是“标签数量斜率是否延续”，而不是“还缺哪个环”。**

---

## 8. 产物索引

| 文件 | 内容 |
|---|---|
| `REPORT.md` | 本报告 |
| `DECISION.md` | STOP / 下一次 GPU 实验的决策记录 |
| `code/zema_audit.py` | 单组、oracle ceiling、pairwise cross-tab、global33 可预测性、置换检查、OOF 主表 |
| `code/zema_decomposition.py` | MECE P×U×L、size 分层、cycle attribution、35 行 order probe、OOF no-penalty 拆分、global33 target OOF |
| `analysis/audit.json` | 主表 + 输入 sha256 + headline |
| `analysis/mece_penalty_unseen_long7.csv` | MECE 分解（全模型） |
| `analysis/penalty_x_unseen_x_size.csv` | P×U×size（两套切点） |
| `analysis/cycle_attribution.csv` | 严重度 λ / hard ceiling / cycle ΔMAE（全模型） |
| `analysis/penalty_order_sensitivity.csv` | 35 个 penalty 行的 50 次随机序 probe |
| `analysis/oof_conditional.csv` | all / no-penalty / penalty 的 OOF 条件信息与逐折 Δ |
| `analysis/oof_ring_correlations_no_penalty.csv` | no-penalty 子集的环特征相关（含 partial） |
| `analysis/global33_target_oof.csv` | global33 vs global33+ring 的 train-only OOF |
| `analysis/decomposition.json` | 上述结果的机器摘要 + 输入 sha256 |

**未做**：GPU、训练、official test、对历史 `zinc_block_ownership_v1/v2` /
`zinc_generalization_audit_v1` / `results/phase1e` / `zinc_dccl_bottleneck_v1` 的任何改写。
