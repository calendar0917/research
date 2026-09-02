# MolHIV KSVD 可行性快验与路线

> 日期：2026-07-25。定位：快速机制筛查，不作为最终 OGB test 报告。

## 快验设置

- 数据：固定 `max_graphs=5000, data_seed=0` 的 label-stratified scaffold 子集。
- 正例数：train/valid/test = 150/10/16；因此子集 AUC 只能用于筛路线。
- patch：CoverageRW，`L=m=8`，每个训练图至多向字典贡献 8 个 patch（图编码仍使用其全部采样 patch）。
- 表示：239 维 permutation-invariant WL-style patch vector，逐 patch L2 normalize。
- KSVD：8 atoms，OMP `T=2`，4 iterations，max-abs coefficient pooling。
- 所有方法使用同一批训练 patch、同一 LR 和 size baseline。

## 关键结果

### 表示修复

446 个实际 patch 的随机重标号检查：

- 旧 BFS adjacency：163/446（36.5%）改变；
- 新 WL patch vector：0/446 改变。

固定同一 5000 子集，`size valid/test = 0.692/0.712`：

| 表示 | size+struct valid | test | valid Δ vs size |
|---|---:|---:|---:|
| legacy topo KSVD | 0.684 | 0.678 | -0.008 |
| normalized topo KSVD | 0.688 | 0.693 | -0.004 |
| normalized invariant-WL KSVD | **0.722** | **0.749** | **+0.030** |

说明“先修置换不变性”是可行且必要的；单纯归一化旧邻接向量不够。

### KSVD 归因

同一 invariant-WL patch 集：

| 字典 | valid Δ vs size | test Δ vs size |
|---|---:|---:|
| random seed 0 | +0.065 | +0.042 |
| random seed 1 | +0.046 | +0.015 |
| random seed 2 | +0.034 | -0.021 |
| PCA | +0.017 | -0.011 |
| KSVD seed 0 | +0.030 | +0.037 |
| KSVD seed 1 | +0.024 | -0.002 |
| KSVD seed 2 | +0.068 | +0.001 |

结论：invariant patch space 本身有信号，但当前无监督 KSVD 尚未稳定优于 random dictionary，字典初始化也明显影响结果。

### 全量 official split sanity check

使用完整 train/valid/test（正例 1232/81/130）、6000 个训练 patch：

| 字典 | valid Δ vs size | test Δ vs size |
|---|---:|---:|
| random seed 0 | -0.0068 | -0.0024 |
| random seed 1 | +0.0071 | -0.0030 |
| KSVD seed 0 | +0.0006 | -0.0049 |
| KSVD seed 1 | -0.0001 | -0.0074 |

因此，纯拓扑、无监督、8-atom 的 KSVD 在全量上基本不提供 size 之外的稳定增益。5000 子集的较大正增益不能外推。

## 路线判断

KSVD 主线仍可行，但研究问题应从“无监督重建能否直接涨 MolHIV”改为：

> 能否通过置换不变、化学标注、分类型的 patch space，以及轻量任务一致性约束，让 K-SVD 学到比随机投影和频繁 motif 更有判别性的稀疏原型？

## 建议路线

1. **KSVD-v2 表示层**：canonical/WL + 完整 OGB atom/bond labels + ring/cycle context；保持固定维 patch vector和稀疏重建。
2. **类型分桶字典**：tree / ring / aromatic / fused-ring 分别学习小字典，避免常见 path atom 淹没稀有环原型。
3. **稳健共享字典**：每图等额 patch；5 个 dictionary seeds；对 atoms 做跨 seed 匹配或 ensemble；报告 atom usage/stability。
4. **KSVD 必须胜过的对照**：random D、PCA、exact motif/WL histogram、k-means codebook；同 patch、同 pooling、同 classifier。
5. **轻量判别式 KSVD**：先做 label-aware patch reweighting / residual selection，再考虑 LC-KSVD；不要直接上重型 FDDL。
6. **融合**：只在结构通道通过归因门槛后，使用 GINE + zero-init residual logit，不做硬 concat。

### Go / no-go

进入 GINE 融合前，KSVD-v2 在 full official split 上应满足：

- valid paired mean 优于 random/PCA；
- 至少 4/5 dictionary seeds 方向一致；
- size residual 为正且 bootstrap CI 不显示明显负效应；
- learned atoms 有可解释、跨 seed 可复现的化学/环 motif；
- test 仅在路线冻结后查看一次。

若 chemical/ring KSVD 仍不能胜过 random/WL histogram，则保留 KSVD 作为可解释压缩模块或机制实验，不再把 MolHIV 增益作为主 claim。

## 2026-07-25 实测升级结果

### 化学/真实环 invariant-WL

新增 `wl_chem`（完整 9 个 OGB atom channels + 3 个 bond channels 的 labeled-WL）和
`wl_chem_ring`（再加 cycle rank、cyclic-edge、3--8/overflow shortest-cycle、aromatic/ring
比例）后，5000 图固定子集上的关键结果如下。size baseline 为 valid/test = 0.6921/0.7122。

| 表示/字典 | valid Δ vs size | test Δ vs size |
|---|---:|---:|
| wl_chem random（3 seeds mean） | +0.0472 | -0.0263 |
| wl_chem KSVD（3 seeds mean） | +0.0238 | **+0.0467** |
| wl_chem_ring random（3 seeds mean） | +0.0421 | -0.0202 |
| wl_chem_ring random-patch（3 seeds mean） | +0.0100 | +0.0199 |
| wl_chem_ring KSVD（3 seeds mean） | +0.0185 | **+0.0418** |

完整 official split（train/valid/test 正例 1232/81/130）上的 `wl_chem_ring`：

| 字典 | valid Δ vs size | test Δ vs size |
|---|---:|---:|
| random（3 seeds mean） | +0.0093 | +0.0058 |
| random-patch（3 seeds mean） | +0.0202 | +0.0012 |
| PCA | -0.0082 | +0.0034 |
| KSVD seeds 0/1/2 | +0.0353 / +0.0357 / +0.0025 | +0.0158 / +0.0028 / +0.0083 |
| KSVD seeds 3/4 | +0.0066 / +0.0074 | +0.0027 / +0.0088 |

full 的 KSVD 五个 seeds 都是正 residual，但效应小；这支持继续做 KSVD 化学路线，
不支持宣称目前无监督字典已经稳定大幅领先。

### Typed KSVD 初筛

按 patch 的 induced cycle rank / aromatic bond 分成 `fused / aromatic / ring / tree` 四类，
每类 8 atoms、OMP T=2、4 K-SVD iterations，5000 子集训练字典每类最多 2000 patches。
`type-count + size`（不含字典激活）baseline valid/test = **0.7591/0.7463**。

| family | valid | test | valid Δ vs size | test Δ vs size |
|---|---:|---:|---:|---:|
| random-patch seed0 | 0.6910 | 0.7198 | -0.0011 | +0.0076 |
| random-patch seed1 | 0.6784 | 0.6876 | -0.0138 | -0.0245 |
| random-patch seed2 | 0.7671 | 0.7043 | +0.0750 | -0.0079 |
| PCA | 0.7955 | 0.7615 | +0.1034 | +0.0493 |
| KSVD seed0 | 0.7518 | 0.7268 | +0.0597 | +0.0146 |
| KSVD seed1 | 0.7304 | 0.7536 | +0.0383 | +0.0415 |
| KSVD seed2 | 0.6292 | 0.7273 | -0.0630 | +0.0151 |

这个 typed 初筛说明“分桶”本身有价值，但**当前 typed K-SVD 激活没有稳定超过
PCA / type-count baseline**；主要问题是 fused 类 patch 太少（训练仅 107），以及
aromatic 类占比过高。因此下一步不应直接扩大 atoms，而应做：

1. aromatic 类内按 bond/atom 语义再平衡 patch 采样；
2. fused 类采用 oversampling 或跨图共享 residual 字典；
3. 先固定 `type-count + size`，只检验字典激活 residual；
4. 对 KSVD 做 label-aware positive patch 重加权，再重跑 5 seeds。

### 五 seed 汇总与当前决策

full official split 上，`wl_chem_ring` KSVD 五 seed 平均：

- valid residual vs size：**+0.01750 AUC**；
- test residual vs size：**+0.00767 AUC**；
- random-patch 五 seed 平均：valid **+0.01222**，test **+0.00182**；
- paired KSVD - random-patch 平均：valid **+0.00528**，test **+0.00585**。

五个 KSVD seeds 相对 size 都为正；相对 paired random-patch，valid 3/5 胜、test 4/5 胜。
所以当前结论是“**有可重复但偏小的 KSVD 化学 residual**”，可以进入 label-aware
reweighting 与 atom stability 阶段，但还不足以直接进入最终 GINE claim。

另外，loader 现在会保存 `original_indices`，node-gate 和 dual-channel PyG pipeline
直接复用这些索引；300 图实测 internal Graph / PyG 的 node count 与 label 全部逐图对齐。

## 2026-07-25 label-aware / discriminative KSVD validation 复核

以下新实验均只编码并评估 official train/valid；没有产生新的 test 指标。

### 50% positive patch reweighting：五 seed

固定同一个 50/50 正负训练 patch 集（4000 columns），8 atoms、OMP T=2、4 次
K-SVD，size valid AUC = **0.67874**。

| seed | random-patch Δ vs size | KSVD Δ vs size | paired KSVD-random |
|---:|---:|---:|---:|
| 0 | +0.01800 | +0.03874 | +0.02074 |
| 1 | +0.03531 | +0.05452 | +0.01921 |
| 2 | -0.00593 | +0.02435 | +0.03027 |
| 3 | +0.01038 | -0.01720 | -0.02759 |
| 4 | +0.01745 | -0.00233 | -0.01977 |
| mean | +0.01504 | **+0.01961** | **+0.00457** |

结论：均值有增益，但只做到 3/5 seed 相对 size 为正、3/5 paired 胜出，未达到
冻结路线要求。atoms 本身并非完全随机：以 seed0 为 reference 的 Hungarian 匹配平均
absolute cosine = **0.87288**、median = **0.94293**。因此瓶颈主要是“稳定 atoms 的
判别方向不足”，而不是完全学不到重复 motif。seed0 的近邻 patch 同时包含纯碳链、
芳香环、非芳香环和含 N/O/卤素的局部结构；top-5 positive fraction 为 0.4--0.8，
尚不足以形成干净的 HIV-specific atom。

### D+ / D- 条件 KSVD

分别从正、负训练图各取 2000 patches 学 8-atom KSVD，并测试 max activation 拼接、
activation difference、全图重构差和 patch-level reconstruction margin。三 seed 结果：

| readout | KSVD mean Δ vs size | random mean Δ vs size | paired mean |
|---|---:|---:|---:|
| concat activation | +0.02301 | +0.02905 | -0.00605 |
| activation diff | +0.00905 | +0.00450 | +0.00455 |
| reconstruction diff | +0.00894 | -0.00014 | +0.00908 |
| patch margin stats | +0.01288 | +0.01092 | +0.00195 |

单 seed 偶尔很高，但方向不稳定；atom permutation alignment 后的三 seed ensemble 也
没有稳定超过 random-patch。条件字典不是当前主线。

### Label-augmented KSVD

在平衡训练 patch 上对 K-SVD 输入追加 `alpha * one_hot(graph_label)` 两行，仅保留
学得字典的 chemical block 做 validation 编码。seed0 小筛选中 alpha=0.3 最好，随后
复核五 seeds：

- KSVD mean Δ vs size = **+0.01577**；
- matched random-patch mean Δ vs size = **+0.01504**；
- paired mean = **+0.00073**；
- KSVD 4/5 相对 size 为正，但只有 3/5 paired 胜出。

判别目标直接追加到 reconstruction target 并未超过原始 chemical/ring KSVD。

### Overcomplete KSVD + train-only atom selection

先学 16 个候选 atoms，再按平衡训练 patch 上正负 sparse-code usage 的 Fisher-style
差异，各选 4 个正相关和 4 个负相关 atom。seed0：

- selected KSVD Δ vs size = **+0.02214**；
- identically selected random-patch Δ vs size = **+0.02296**。

未通过 paired control，不继续多 seed。

### 当前冻结决策

1. **保留的最稳 KSVD 结论仍是无监督 `wl_chem_ring` 统一字典**：full split 五个
   dictionary seeds 相对 size 全为正，mean Δ = +0.01750，paired random-patch mean
   = +0.00528。
2. 50% positive reweighting 可作为 ablation，但不能替代主字典：它提高了部分 seeds
   的上限，却降低了方向一致性。
3. D+/D-、label augmentation、supervised atom selection 均已快速证伪，不再继续扫
   fraction、margin statistic、candidate atom 数或 label weight。
4. 下一条值得投入的路线是 **冻结 chemical/ring KSVD，作为 zero-init residual logit
   融入 GINE**；必须采用 validation-only checkpoint selection，并与同 seed GINE-only
   paired 对比。KSVD 通道保持固定、可归因，不再继续改字典超参数。

## 2026-07-25 KSVD → GINE zero-init residual（full validation-only）

`run_molhiv_dual.py` 已增加：

- 从固定 `.npz` 加载 `wl_chem_ring` KSVD dictionary；
- `fusion=residual`：`GINE_logit + Linear(s_KSVD)`，residual head 权重/偏置零初始化；
- residual learning-rate scale = 0.1、weight decay = 1e-3；
- train-only 标准化 KSVD 特征；
- `--valid-only`：test 不做结构编码、不建 test loader、不写逐 epoch test AUC；
- 显式 DataLoader generator，确保同 seed 的 GINE-only / residual minibatch 顺序完全一致。

固定 `ksvd_pos0.50_seed0`（8 atoms、T=2）结构通道，full official train/valid，30 epochs：

| train seed | GINE-only best valid | KSVD residual best valid | paired Δ |
|---:|---:|---:|---:|
| 0 | 0.80431 | **0.81503** | +0.01072 |
| 1 | **0.82300** | 0.80359 | -0.01941 |
| 2 | **0.81398** | 0.80092 | -0.01306 |
| 3 | 0.79646 | **0.80432** | +0.00786 |
| 4 | 0.78392 | **0.82004** | +0.03612 |
| mean | 0.80433 | **0.80878** | **+0.00444** |

median paired Δ = +0.00786，3/5 seeds 胜出，但 paired sample std = 0.02196；因此它是
目前最值得继续的 fusion 信号，却仍未达到“4/5 一致”冻结门槛。特别要注意：修复
DataLoader RNG 配对之前的结果不能作为正式 paired 统计，上表才是严格版本。

另外，把五个 pos50 dictionary embeddings 直接拼成 40 维 residual，在 seed0 的 valid
只有 **0.79066**（对应 GINE-only 0.80431），已在单 seed screen 阶段否决；不继续多 seed。

### 更新后的路线判断

- **字典层主结论**：无监督 chemical/ring KSVD 仍最稳定；额外判别式字典改造没有可靠收益。
- **融合层主结论**：小学习率、正则化、零初始化 residual 有正的五 seed mean/median，
  但 variance 较大。它可以作为下一阶段唯一主线，不应与更多 dictionary sweep 同时进行。
- 下一步应固定 dictionary 和 residual 超参，改进 GINE checkpoint protocol（例如预先固定
  epoch/patience 或 train 内部 early-stop split），避免“各 seed 取 30 epochs 最大 valid”
  放大方差；完成该 protocol 后再决定是否做一次受限 test evaluation。
