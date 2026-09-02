# MolHIV 上 KSVD 的最终可行路线

> 日期：2026-07-25。结论范围：official scaffold **train/valid** 上的路线验证；不是新的
> OGB test 榜单结果。早期 feasibility 阶段曾查看 test，因此本文所有最终路线都只依据
> validation 冻结，最近的 readout、conditional、discriminative 和 inner-checkpoint 实验均
> 没有编码或评估 test。

## 结论先行

已经找到一条可以冻结、可复现、且通过 matched control 的 KSVD 主路线：

> **permutation-invariant chemical/ring patch → natural shared K-SVD → rich sparse-code
> statistics → balanced logistic regression（附 size residual）**。

完整 official train/valid、5 个 dictionary seeds 的结果：

| 方法 | valid AUC mean ± seed SD | mean Δ vs size | KSVD − random-patch | paired wins |
|---|---:|---:|---:|---:|
| size only | 0.67874 | — | — | — |
| random-patch + rich | 0.68463 ± 0.01616 | +0.00589 | — | — |
| **KSVD + rich** | **0.71213 ± 0.00988** | **+0.03338** | **+0.02750** | **5/5** |
| random-patch + moments | 0.69043 ± 0.01244 | +0.01169 | — | — |
| **KSVD + moments** | **0.70219 ± 0.00099** | **+0.02345** | **+0.01176** | **4/5** |

其中 random-patch 与对应 KSVD 使用同一个训练 patch pool、同一个 seed；K-SVD 的初始化
正是同 seed 抽到的 patch columns（仅带初始化微扰）。因此 5/5 的 rich paired 胜出直接说明
增益不是只有 invariant patch space 或随机原型，而来自 K-SVD 更新后的字典与其稀疏编码。

第二条可保留但应降级表述的路线是：

> **GINE logit + zero-init linear KSVD residual**，用 official train 内层 validation 选 epoch，
> 然后在完整 official-train 子集重训，并且 official valid 只评估一次。

固定 8000 图分层子集、5 个 neural seeds：4/5 paired 胜出，paired mean `+0.00164`、median
`+0.01483`；但 seed2 为 `-0.04273`，sample SD `0.02580`。它说明 KSVD residual 有真实但
高方差的可行信号，适合作为 secondary/exploratory route，暂不应取代 standalone rich
KSVD 作为主结论。

---

## 1. 主路线：Rich-statistics KSVD

### 1.1 固定协议

- 数据：OGB `ogbg-molhiv` official scaffold split；字典和分类器只用 official train；
  official valid 用于路线比较；test 不编码、不评估。
- patch sampler：固定 CoverageRW 默认设置，`max_nodes=8`，每图最多向字典 pool 贡献
  8 个 patch；不再扫描 `p/q/walk_length`。
- patch vector：524 维 `wl_chem_ring`：
  - OGB 9-channel atom labels；
  - OGB 3-channel bond labels；
  - labeled WL；
  - cycle rank、cyclic edge；
  - shortest-cycle histogram；
  - aromatic/ring statistics。
- 不变性实测：真实 MolHIV patch 随机 node relabel 后 0/440 改变。
- patch 逐列 L2 normalize。
- 字典：8 atoms，OMP `T=2`，K-SVD 4 iterations，最多 4000 个 official-train patches。
- classifier：`StandardScaler + LogisticRegression(class_weight="balanced", C=1)`。
- graph size residual：节点数、边数两维，和 KSVD readout 一起输入 classifier。

### 1.2 rich readout 的精确定义

对每张图的 sparse-code matrix `X ∈ R^(8×Npatch)`，每个 atom 计算 10 组统计量：

1. mean `|x|`；
2. max `|x|`；
3. top-3 mean `|x|`；
4. std `|x|`；
5. nonzero usage；
6. q75 `|x|`；
7. q90 `|x|`；
8. mean squared energy；
9. signed mean；
10. winner frequency。

得到 `10 × 8 = 80` 维；再追加 6 个 normalized patch reconstruction-error statistics
（mean/std/q50/q75/q90/max）、patch count 和 `log1p(count)`，共 88 维。最后追加节点数、
边数，classifier 总输入为 **90 维**。

### 1.3 五 seed 明细

| dictionary seed | random-patch rich | KSVD rich | paired Δ |
|---:|---:|---:|---:|
| 0 | 0.70928 | 0.71636 | +0.00708 |
| 1 | 0.66573 | 0.72742 | +0.06169 |
| 2 | 0.68758 | 0.70292 | +0.01533 |
| 3 | 0.68436 | 0.70628 | +0.02192 |
| 4 | 0.67619 | 0.70766 | +0.03147 |
| **mean** | **0.68463** | **0.71213** | **+0.02750** |

五个 KSVD seeds 都高于 size baseline，也全部高于 matched random-patch；这是当前最强的
KSVD 本体归因证据。相同 patch、OMP 和 readout 下的 PCA control 为：moments `0.66871`、
rich `0.65432`，分别低于 size `-0.01003/-0.02442`，因此 rich 增益也不是普通低秩
投影即可复现。

### 1.4 低维稳健 fallback：moments

若更重视 seed 稳定性而不是平均 AUC，可使用 24 维 moments readout：每 atom 的
`mean|x| / std|x| / mean(x²)`，再加两维 size，总计 26 维。

| seed | random-patch moments | KSVD moments | paired Δ |
|---:|---:|---:|---:|
| 0 | 0.71119 | 0.70111 | -0.01008 |
| 1 | 0.68476 | 0.70329 | +0.01853 |
| 2 | 0.68456 | 0.70304 | +0.01848 |
| 3 | 0.69211 | 0.70222 | +0.01012 |
| 4 | 0.67955 | 0.70130 | +0.02175 |
| **mean** | **0.69043** | **0.70219** | **+0.01176** |

KSVD 自身的 seed SD 只有 `0.00099`。它比 rich 少约 0.010 AUC，但更紧凑、更容易作为
稳定压缩模块部署。

### 1.5 非线性 classifier 筛查结论

固定 8000 图子集、5 dictionary seeds 上还比较了 balanced logistic、RBF SVM 和
HistGradientBoosting，以及 max/basic/tail/moments/reconstruction/rich readouts。

- RBF 大多低于自己的 size baseline；
- HGB train AUC 约 0.98--0.998，dictionary-seed 方差很大；
- 8000 子集 official valid 只有 16 个正例，非线性模型的高单点 AUC 不稳定；
- rich logistic 在该小子集上未能清楚胜过 random-patch，但在完整 official valid
  （81 positives）上变为 5/5 paired 胜出。

因此最终路线固定为 logistic，不继续扫 SVM/HGB 超参数。

---

## 2. Secondary route：GINE + zero-init KSVD residual

### 2.1 严格 inner-checkpoint 协议

固定 `max_graphs=8000, data_seed=0`：official train/valid/test 为 6400/800/800；模型没有
编码或评估 official test。

每个 neural seed 和每个 fusion 独立执行：

1. official train 内用固定 `inner_split_seed=1729` 做 85/15 stratified split；
2. 最多训练 30 epochs，仅用 inner-valid 选择 epoch；
3. 重置 model、optimizer、DataLoader RNG；
4. 在全部 6400 official-train graphs 上训练恰好 selected epoch；
5. official valid 只评估一次。

同 seed GINE-only 与 residual 具有相同 base GINE initialization 和 minibatch sequence；
结构特征固定为 `ksvd_seed0` 的 8 维 max-activation，train-only normalization；residual
head 零初始化，LR 为 base LR 的 0.1，weight decay `1e-3`。

### 2.2 五 seed 结果

| neural seed | GINE epoch/AUC | residual epoch/AUC | paired Δ |
|---:|---:|---:|---:|
| 0 | 18 / 0.73860 | 18 / 0.75853 | +0.01993 |
| 1 | 16 / 0.73334 | 17 / 0.74817 | +0.01483 |
| 2 | 20 / 0.75247 | 22 / 0.70974 | -0.04273 |
| 3 | 17 / 0.78787 | 17 / 0.80309 | +0.01523 |
| 4 | 17 / 0.78005 | 17 / 0.78101 | +0.00096 |
| **mean** | **0.75847** | **0.76011** | **+0.00164** |

- wins = 4/5；
- paired median = `+0.01483`；
- paired sample SD = `0.02580`。

结论：方向一致性达到 4/5，但均值被一个大负 outlier 几乎抵消。可以作为“KSVD
residual 在严格 checkpoint 下多数 seed 有益”的 secondary result；不能声称稳定提升
GINE，也不应为了修 seed2 再根据 official valid 后验扫描 residual LR。

---

## 3. 已排除或不再继续扫描的路线

| 路线 | 结论 |
|---|---|
| legacy padded adjacency patches | 非 permutation invariant，真实 relabel 会改变表示 |
| 纯 topology KSVD | full split 基本无 size 之外的稳定增益 |
| typed tree/ring/aromatic/fused KSVD | fused 太少、aromatic 主导，未稳定超过 type-count/PCA |
| 50% positive patch reweighting | KSVD mean Δ 为正，但只 3/5 paired 胜出 |
| `D+ / D−` conditional KSVD | activation/reconstruction margins 均不稳定 |
| label-augmented reconstruction target | 5 seeds paired mean 仅 +0.00073 |
| 16→8 supervised atom selection | seed0 KSVD 不胜 matched random-patch |
| 五字典直接 concat residual | seed0 valid 0.79066，低于 paired GINE 0.80431 |
| RBF SVM / HGB readout | 小 valid 上过拟合或高方差，不作为冻结路线 |

这意味着瓶颈并不是“再多扫一轮字典大小、positive fraction 或 sampler 参数”能解决的；
真正被漏掉并最终奏效的是 **保留完整 sparse-code 分布统计，而不是只取 max activation**。

---

## 4. 复现命令

环境：

```bash
cd /home/calendar/code/research/tracks/ksvd
PY=/home/calendar/.conda/envs/gsn-official/bin/python
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
```

### 4.1 最终 standalone KSVD（full train/valid，五 seeds）

```bash
$PY -m code.run_molhiv_readout_search \
  --max-graphs 0 \
  --data-seed 0 --sampler-seed 0 \
  --dict-seeds 0,1,2,3,4 \
  --n-atoms 8 --sparsity 2 --ksvd-iter 4 \
  --max-train-patches 4000 --max-patches-per-graph 8 \
  --families random_patch,ksvd \
  --readouts max,moments,recon,rich \
  --classifiers logistic \
  --output results/molhiv/ksvd_readout_full_5seeds.json
```

主要产物：

```text
results/molhiv/ksvd_readout_full_5seeds.json
results/molhiv/ksvd_readout_full_5seeds.npz
results/molhiv/ksvd_readout_full_5seeds_summary.json
```

### 4.2 PCA matched readout control

```bash
$PY -m code.run_molhiv_readout_search \
  --max-graphs 0 --data-seed 0 --sampler-seed 0 \
  --dict-seeds 0 --families pca \
  --n-atoms 8 --sparsity 2 --max-train-patches 4000 \
  --max-patches-per-graph 8 \
  --readouts moments,rich --classifiers logistic \
  --output results/molhiv/ksvd_readout_full_pca.json
```

### 4.3 严格 inner-checkpoint residual（每个 seed 两条命令）

```bash
for SEED in 0 1 2 3 4; do
  for FUSION in gine_only residual; do
    $PY -m code.run_molhiv_dual_inner \
      --max-graphs 8000 --data-seed 0 --struct-seed 0 \
      --inner-split-seed 1729 --inner-valid-fraction 0.15 \
      --epochs 30 --seed "$SEED" --fusion "$FUSION" \
      --dictionary-npz results/molhiv/ksvd_readout_n8000_5seeds.npz \
      --dictionary-name ksvd_seed0 --struct-readout max \
      --batch-size 128 --hidden 64 --layers 3 \
      --output "results/molhiv/dual_inner_${FUSION}_n8000_seed${SEED}.json"
  done
done
```

汇总：

```text
results/molhiv/ksvd_residual_inner_n8000_5seeds_summary.json
```

---

## 5. 最终推荐

### 应作为主结果

**KSVD-rich-logistic**：它满足当前冻结门槛——5/5 高于 size，5/5 高于同 seed
random-patch，平均 paired 增益显著大于先前 max pooling 路线，并保留 KSVD 作为方法核心。

### 应作为稳健小模型备选

**KSVD-moments-logistic**：26 维、KSVD seed SD 极低，4/5 胜 random-patch；适合需要更低
维度、更简单解释和更稳字典 seed 的场景。

### 只作为 secondary / future work

**GINE + zero-init KSVD residual**：4/5 wins 说明值得保留，但高方差和 seed2 负 outlier
阻止其成为主 claim。若未来继续，正确方向是预注册的 frozen-GINE/post-hoc residual 或
cross-fitted stacking，而不是再用 official valid 扫 LR、字典 seed 或 fusion 宽度。

## 6. 限制

1. 主路线目前是 validation 结果；由于早期 feasibility 已经查看过 test，不能把后续一次
   test 当作完全 untouched final test。
2. 五个 dictionary seeds 共享同一 official valid molecules，seed consistency 不能替代
   数据级置信区间或外部数据集复核。
3. rich readout 的优势说明 KSVD sparse-code distribution 有信息，但不能自动推出每个
   atom 都是独立、可命名的 HIV motif。
4. neural residual 的 paired mean 很小且方差大，不应只报告 4/5 wins 而隐藏 outlier。
