# REPORT — `zinc_block_ownership_v1`

**一句话结论：机制是活的，方向是正的，但预注册的 seed-0 主门未过。**
R（真实 biconnected 块归属）比 N（全图均场）的 best-valid 好 **0.004884**，
比预注册的 **0.005** 实用门槛少 **0.000116**；配对 ΔMAE 的 95% bootstrap CI
为 **[−0.000306, +0.011554]**，跨零。按预注册停止规则，**不启动 seed 1，不跑 S**。
因此本轮不能宣称“块归属有用”，也不能宣称“块归属无用”。最终标签取
**`UNRESOLVED_DUE_TO_VARIANCE`**（保守读法：方向性 positive，但不足以越过实用门槛）。

- 运行 commit：`2409811b0ac8694cc7755cb79768a3a047112e0c`
- 冻结 config：`configs/zbo_v1.json`，sha256 `151deb021d12ad77d8997d5abf99c3a42012efd9bc73ca9e15f7fac35958c0bd`
- CPU gates：`analysis/cpu_checks.json`，sha256 `177271fb541ab5f732ab1f8303435ee016cfe4e05a147d2130c32595bc27a9fb`，**C1–C15 全 PASS**
- 正式 GPU：R_seed0 + N_seed0，各 47,400 updates / 600 epochs，合计 **1.079 A100·h**（软预算 4 h）
- official test：**从未读取**；本轮不声称 official test < 0.060

---

## 1. 问题与预注册

本轮只问一个机制问题：

> 把每条物理键按真实 biconnected component 归属到块（桥边是单键块），
> 再用共享的 per-block Deep Sets 通道读出 `Σ_B φ(u_B)`，能否让共享非递归结构
> 计算在未见局部环境/块组合上优于**同参数、同初始化、同数据、同日程**的全图均场统计？

三臂在**模型参数与初始 state 完全相同**，只切换块内容的构造方式：

| 臂 | 块内容 | 含义 |
|---|---|---|
| **R** | 每个块的真实一阶/二阶键内容 | 真实归属 |
| **N** | 全图键内容的均场，按块边数比例分配 | 无归属信息；不是同信息对照 |
| **S** | 单键块保留真实内容；多键块在 MULTI 集合内按边数均分 | 单键真实 + 多键均场 |

预注册的 seed-0 主判决（`PROTOCOL.md` §8.2）：

```
d_best = N_best_valid − R_best_valid
PASS iff
  d_best >= 0.005
  and R_best_valid <= 0.082
  and d_W20 >= −0.002 and d_T100 >= −0.002
  and (d_W20 > 0 or d_T100 > 0)
  and ordinary (y > −3) paired ΔMAE >= 0
  and fixed-worst-10-removed paired ΔMAE > 0
  and mechanism_active
```

**停止规则**：seed-0 失败即停；不调宽度/层数/lr/wd/归一化/块类型，不买 seed 1，不跑 S。
本轮严格遵守。

---

## 2. 设计要点（与历史工作的区别）

### 2.1 块划分

- 用独立实现的 **Tarjan 边栈**求 biconnected components；每条边恰好属于一个块；
  桥边形成单键块（`n_edges=1`）。
- 与 NetworkX `biconnected_components`、割点、以及 `zinc_arch_discovery_v1` 旧
  `block_ptr/block_bond_ptr/block_bond_idx` 缓存三方交叉核对。
- **不复用**旧 BlockModel 的 `block_joint`（693 维直方图）、`block_dist/block_overlap`、
  `GlobalAttention` 或 block-cut-tree 距离；旧 BlockModel 的 valid 0.165418 与本轮无关。

### 2.2 块输入与 reader 扩展

```
u_B = [ mean_{e∈B} c_e (96), mean_{e∈B} c_e² (96),
        log1p(n_edges), log1p(n_atoms),
        cyclomatic_rank, log1p(n_articulations) ]  ∈ R^196

h_block = Σ_B φ(u_B),   φ: 196 → 64 → 32, shared, SiLU
y = reader_old(z_old) + reader_ext(h_block)
```

- `reader_ext = Linear(32 → 256, bias=False)`，**零初始化**，与旧 reader 第一层相加；
  数学上等价于 449 列第一层（32 新列从 0 开始），但避免宽矩阵 BLAS 归约顺序破坏
  初始化时的逐位等价。
- 参数 **497,161** ≤ 500k：old 473,889 + block_norm 392 + block_mlp 14,688 +
  reader_ext 8,192。
- 无递归 MP、无 learned junction token、无 attention、无蒸馏、无 test 调参。

### 2.3 初始化纪律

- 先按历史 RNG 顺序构造完整 BOSI-B，再加块编码器与 reader 32 新列；
- 新列零初始化；三臂完整初始 state 逐张量相同；
- 与同 seed 原 BOSI-B 的公共 state max diff = 0，前向 **bitwise identical**；
- seed 0 初始 state hash：`bd18518b396646f14426f92ccf251139d249440e0edc1317c384bab23ae5ca2c`；
- R/N 的首个 batch rows hash 相同：`ea88ffc8959b8a6b909a6d1bed63e2d323ac0c7a28bf1f835b1d54cb8f62b345`。

### 2.4 与历史结论的关系

- BOSI v0：R−B 无重复差异（`NO_GO`）；partner-content P−B +0.01239，说明“内容重要、
  组织不重要”是历史事实。
- BOSI-JM：官方 10k/1k 上 BOSI-B seed0 = 0.082405，差于 DCC-L seed0 +0.003868；
  BOSI 主干在官方协议上不与 DCC-L 打平。
- AAS ERRATA E1/E2：seed≥2 共用 s1 SSL；0.006 功效地板需拆成 encoder/监督/执行漂移/行杠杆。
- 本轮因此把问题收窄成 **R−N 的机制对照**，而不是“再做一个 BOSI 变体冲性能”。

---

## 3. Phase 0 CPU gates（GPU 之前的硬门）

`analysis/cpu_checks.json`，22.1 s，CPU-only，C1–C15 全 PASS：

| Gate | 内容 | 结果 |
|---|---|---|
| C1 | Tarjan / NetworkX / 旧缓存三方块划分一致；每键恰好一次；rank/articulation 正确 | train 10,000 mol / **121,808 blocks**；valid 1,000 / **12,155 blocks**；0 mismatch |
| C2 | 原子重编号、边反向、边顺序、batch 顺序不变性（块通道非零权重） | max pred diff ≤ 1.2e-7 |
| C3 | R/N/S 一阶/二阶总量守恒；S 单键 == R；N/S 分配和 = 1 | float64 最大偏差 ≤ 1.8e-15 |
| C4 | 合成反例：两个三角形 + 桥，同键状态多重集、同块大小，不同归属 | R diff **0.098**；N/S diff ~2e-7；线性池化 ~2e-7 |
| C5 | 真实 ZINC 结构覆盖率 + R/S 数值差 | 见下 |
| C6 | 旧 BlockModel 静态审计（无 `block_joint`/attention 调用） | PASS |
| C7 | 参数 497,161；三臂 state 相同；与 BOSI-B 前向逐位相同 | PASS |
| C8 | 第二步后块梯度非零；无 NaN；reader_ext 离开 0 | PASS |
| C9 | 标签隔离：valid labels 置 NaN 后块缓存与预测不变 | PASS |
| C10 | relation/block 缓存可复现；已知 relation content hash 对齐 | PASS |
| C11 | 手算小图（链、三角形、并环、桥、带弦四边形） | PASS |
| C12 | checkpoint 往返预测逐位一致 | PASS |
| C13 | norm 仅 train 拟合；SSL 只含 atom/bond/junction；无 test | PASS |
| C14 | 固定 worst-10 行（来自冻结 BJM B_s0） | 10 行，anchor MAE 0.082405 |
| C15 | 结构-only 分组阈值与 valid 组计数 | PASS |

### 3.1 真实结构覆盖率与 R/S 差异

- train：98,286 个单键块 + **23,522 个多键块**；**88.32%** 分子含 ≥2 个多键块；
  valid 88.90%。
- 块尺寸（边数）以 6 为主（13,038 块），其次 5（5,855）、10（1,868）、11（1,128）；
  最大 28。
- 在含 ≥2 个多键块的 8,832 个 train 分子、22,409 个多键块上，R 与 S 的块均值相对差：
  mean **0.3542**，median **0.3513**，P90 **0.5067**，**99.92% > 1%**；
  二阶矩 mean 0.4067。
- 因此 GPU 停止规则（结构 <20% 或 R/S 输入差异极弱）未触发。

### 3.2 合成反例说明了什么、没说明什么

两个三角形经一条桥连接（块大小 [3,1,3]）。同一组 7 个键状态向量，只改变“哪个状态属于
哪个三角形”，保持多重集和块大小完全不变：

- R 的 `h_block` 差 **0.0981**；
- N/S 的差 ~2e-7（float32 归约顺序）；
- 纯线性池化 `Σ_B mean content` 的差 ~2e-7。

这说明：**非线性共享 φ 是让归属可区分的必要条件**；线性池化对块归属置换不敏感。
它**不**证明真实化学标签上 R 一定更好，只是一个 operator-level witness。

---

## 4. GPU 协议与执行记录

正式 run 在远端 `res`（a100-2，2×A100-40GB，driver 550.163.01，torch 2.5.1+cu124）：

| 项 | 设置 |
|---|---|
| 数据 | 官方 PyG ZINC `subset=True`；train 10k / valid 1k；official test 不读 |
| 目标 | raw-y，L1 |
| batch | 128；79 updates/epoch；600 epochs；**47,400 updates** |
| 优化 | AdamW 1e-3 → 1e-5 per-step cosine；warmup 500；wd 1e-5；clip 5；FP32 |
| 初始化 | 每个 seed 独立加载同 seed 的 train-only masked-SSL atom/bond encoder；不加载 junction |
| 选模 | 每 epoch full valid；strict minimum；tie 取更早 epoch |
| 并行 | R_seed0（res-gpu0）与 N_seed0（res-gpu1）同时启动 |
| 预算 | R 0.493 h + N 0.586 h = **1.079 A100·h**；smoke 0.002 h；soft 4 h 内 |

冻结 config 的 `git_head_at_freeze` 指向 `e7dd8e6`（config 自身尚未提交时的父 commit）；
正式 run 的 `train.json` 记录实际 clean run commit 为 `2409811b0ac8`，两者都写入产物。
所有 run 的 `official_test_loaded=false`，`git_dirty=false`。

---

## 5. 结果

### 5.1 两条正式轨迹

| 指标 | R_seed0 | N_seed0 | R − N（正 = R 更好） |
|---|---:|---:|---:|
| best-valid | **0.076707141** | 0.081591541 | **+0.004884399** |
| best epoch / step | 387 / 30,573 | 359 / 28,361 | — |
| final-valid | **0.077740899** | 0.084134442 | +0.006393544 |
| W20 | **0.077748307** | 0.083952711 | +0.006204404 |
| T100 | **0.077826765** | 0.084128991 | +0.006302226 |
| train@best（eval mode） | 0.013868 | 0.016376 | — |
| train@final（eval mode） | 0.001885 | 0.000496 | — |
| median AE（best） | 0.036015 | 0.041610 | — |
| P90 AE（best） | 0.152221 | 0.161195 | — |
| 参数量 | 497,161 | 497,161 | 0 |
| GPU h | 0.493 | 0.586 | — |
| initial state hash | `bd18518b…` | `bd18518b…` | 相同 |
| first-batch rows hash | `ea88ffc8…` | `ea88ffc8…` | 相同 |

两条轨迹都强过拟合（train@final < 0.002），说明容量足够；差别不在“训练不动”。

### 5.2 预注册 seed-0 门

| 条件 | 实测 | 结果 |
|---|---:|---|
| `d_best >= 0.005` | **0.004884** | **FAIL**（差 0.000116） |
| `R_best <= 0.082` | 0.076707 | PASS |
| `d_W20 >= −0.002` | +0.006204 | PASS |
| `d_T100 >= −0.002` | +0.006302 | PASS |
| `d_W20 > 0 or d_T100 > 0` | 两者均正 | PASS |
| ordinary (y>−3) ΔMAE ≥ 0 | +0.002089 | PASS |
| fixed-worst-10-removed ΔMAE > 0 | +0.003676 | PASS |
| 单行最大贡献 < 50% ΔMAE | 0.419 | PASS |
| `mechanism_active` | True | PASS |

**唯一失败项是 `d_best`，而且只差 0.000116。** 这不是“机制没接上”，也不是
“R 明显更差”；但在预注册规则下，它仍然不能进入 seed 1 / S。

### 5.3 配对 ΔMAE（best checkpoint，正 = R 更好）

- 全体 1,000 valid：**+0.004884**，95% CI **[−0.000306, +0.011554]**，win rate 53.1%。
- final checkpoint：**+0.006394**，95% CI **[−0.000847, +0.017782]**，win rate 51.9%。

两个 CI 都跨零。点估计方向一致为正，但不能把“R 更好”当成已确认。

### 5.4 结构分组（best checkpoint）

| 分组 | n | ΔMAE (N−R) | 95% CI | 读法 |
|---|---:|---:|---|---|
| ordinary (y>−3) | 935 | +0.002089 | [−0.001847, +0.006091] | 方向正，CI 跨零 |
| tail (y≤−3) | 65 | +0.045089 | [−0.010717, +0.124434] | 尾部点估计大但不稳 |
| support all_supported | 384 | +0.003108 | [−0.001242, +0.008045] | 不显著 |
| support low_support | 365 | **+0.010286** | **[+0.000958, +0.024114]** | 唯一 CI 明确为正的 support 组 |
| support any_unseen | 251 | −0.000252 | [−0.013395, +0.012873] | **无优势** |
| rings≥4 | 997 | +0.004961 | [−0.000204, +0.011587] | 几乎全体，信息量低 |
| penalized proxy ring≥7 | 36 | +0.081480 | [−0.022281, +0.218779] | 小样本，不可下结论 |
| fixed worst-10 | 10 | +0.124490 | [−0.226936, +0.636789] | win rate 0.4，方差极大 |
| drop fixed worst-10 | 990 | +0.003676 | [+0.000125, +0.007408] | 仍为正 |
| drop top-10 by R AE | 990 | +0.004997 | [+0.001486, +0.008772] | 仍为正 |
| drop row172 | 999 | +0.002840 | [−0.001283, +0.006905] | 不显著 |

关键读法：

1. **收益不在单行/固定 worst-10**：drop-fixed 与 drop-top10 仍为正，单行最大贡献
   0.419 < 0.5。但 ordinary 组的 CI 跨零，说明“普遍普通分子”上的证据仍弱。
2. **收益不在 any_unseen**：`any_unseen` 组几乎为零（−0.00025），与“未见局部环境”
   的期待相反。低 support 组（1–4 次出现）有正 CI，但这是“稀有但见过”，不是“未见”。
3. **不是单纯尾部**：tail 点估计大但 CI 极宽；ordinary 点估计小且跨零。

### 5.5 块数量 / 组合新颖性（best checkpoint）

| 分组 | n | ΔMAE (N−R) | 95% CI |
|---|---:|---:|---|
| n_blocks ≤10 | 267 | +0.000990 | [−0.009830, +0.011112] |
| n_blocks 11–12 | 269 | +0.004278 | [−0.008652, +0.023855] |
| n_blocks 13–14 | 258 | **+0.007385** | **[+0.000998, +0.014709]** |
| n_blocks ≥15 | 206 | **+0.007593** | **[+0.000552, +0.015473]** |
| n_multi 1 | 110 | +0.010359 | [−0.018833, +0.054490] |
| n_multi 2 | 476 | +0.003915 | [−0.002029, +0.009419] |
| n_multi ≥3 | 413 | +0.004882 | [−0.001812, +0.011748] |
| block-size combo novel | 47 | +0.063616 | [−0.013972, +0.176821] |
| block-size combo seen | 953 | +0.001988 | [−0.001467, +0.005463] |
| block-signature combo novel | 370 | +0.005636 | [−0.007059, +0.021336] |
| block-signature combo seen | 630 | +0.004443 | [+0.001092, +0.007853] |

读法：正 CI 出现在 **块数较多（≥13）** 与 **见过的 signature 组合**；未见 size combo
（n=47）点估计大但 CI 很宽，不能作为“未见块组合泛化”的证据。

### 5.6 机制诊断

| 量 | R_seed0 | N_seed0 |
|---|---:|---:|
| block grad (step 1) | 1.141e-4 | 1.208e-4 |
| block grad (step 2) | 3.191e-4 | 3.321e-4 |
| reader_ext grad (step 0) | 0.3259 | 0.3470 |
| valid `h_block` std | 1.4388 | 0.1932 |
| valid `φ` std | 0.4604 | 0.0236 |
| reader_ext weight norm | 4.7848 | 4.1014 |
| valid mean \|Δpred\|（R 内容 vs N 内容） | **0.10155** | 0.00357 |
| `mechanism_active` | True | True |

块通道确实被训练、被读出；R 模型对“真实归属 vs 均场”的内容替换非常敏感，
N 模型对同一替换不敏感（因为它训练时看到的就是均场）。这不是“机制未激活”。

### 5.7 与历史数字的定位（仅上下文，不是控制对照）

- BOSI-B 官方 10k/1k seed0：**0.082405**（历史 BJM）。
- DCC-L 4-seed valid 均值：**0.075990**；one-shot test 0.067319；历史最好 test 0.066267。
- 本轮 R_seed0 best-valid：**0.076707**，比 BOSI-B 低 0.005698，比 DCC-L 均值高
  0.000717。参数 497,161，单 seed，未读 test。
- 这些数字不能用来宣称“接近 DCC-L”或“达标”：本轮是机制对照，且 R−N 主门未过。

---

## 6. 判定

### 6.1 最终标签

**`UNRESOLVED_DUE_TO_VARIANCE`**

理由：

- `mechanism_active = True`，排除 `MECHANISM_INACTIVE`；
- seed-0 主门唯一失败项是 `d_best`：0.004884 < 0.005，差 0.000116；
- 配对 ΔMAE CI 跨零（[−0.000306, +0.011554]）；
- W20/T100/final 都正向（+0.0062/+0.0063/+0.0064），ordinary 非负，drop-fixed 为正，
  单行贡献 <50%，所以也不能诚实地写成“ownership 无用”；
- 预注册标签 1–4 都要求 **两个 seed**；本轮只跑了 seed 0，不能套用
  `OWNERSHIP_NOT_USEFUL` 的“两 seed 均 <0.005”定义；
- 预注册标签 5 明确覆盖“符号/CI 不能区分”。因此取 5。

**保守读法**：`d_best` 没过实用门槛，因此 **不能**说“真实块归属带来了可用的泛化收益”。
`R_best 0.0767` 与 DCC-L 均值接近，但这只是单 seed、单主端点未过的观察，不能升级为
`STRUCTURAL_GAIN_BUT_NOT_COMPETITIVE` 或 `PROMISING`（后者要求 R−S 两 seed、平均 ≥0.003）。

### 6.2 为什么没有跑 seed 1 / S

`PROTOCOL.md` §8.2 明确：

> 若失败：**停止**。不调宽度/层数/lr/wd/归一化/块类型，不买 seed 1，不跑 S。

本轮严格执行。S 未运行，因此：

- 不能回答 `SINGLETON_NONLINEARITY_ONLY`；
- 不能把 R−N 的点估计解释为“多块组合泛化”；
- 不能把 R 的 0.0767 与 DCC-L 的 0.075990 做竞争力结论。

### 6.3 负结果的事实价值

1. **机制可激活**：块编码器在第二步就有非零梯度，R 模型对归属替换的 mean|Δpred| 0.1016，
   说明不是“块通道没接上”。
2. **方向性信号存在**：W20/T100/final 均约 +0.006；low-support、块数 ≥13、seen signature
   组合有正 CI；drop-fixed/drop-top10 仍为正。
3. **但主端点未过**：best-valid 的实用门槛差 0.000116，配对 CI 跨零；any_unseen 无优势。
4. **因此不能宣称“块归属在未见组合上泛化”**；最多只能说“在 low-support 与较大块数子群
   有方向性信号，值得在新预注册中检验”。

---

## 7. 任务书五个问题的直接回答

1. **机制是否激活？**
   是。R/N 的 block grad step 1 均 >1e-8，`h_block` std >1e-6，reader_ext 范数 >1e-6，
   R 的 R/N 内容替换 mean|Δpred| = 0.1016。

2. **真实归属 R 是否优于全图均场 N？**
   点估计是（best +0.004884，W20 +0.006204，T100 +0.006302，final +0.006394），
   但 best-valid 未过 0.005 实用门槛，配对 CI 跨零。按预注册：**不能确认**。

3. **R 是否优于 S（单键真实 + 多键均场）？**
   **未跑**。seed-0 失败触发停止规则；没有 R−S 证据，不能判断收益是否主要来自
   单键块的非线性内容。

4. **是否在未见局部环境/块组合上更好？**
   **没有证据**。`any_unseen` ΔMAE = −0.00025（CI 跨零）；block-size combo novel
   n=47 点估计 +0.0636 但 CI 宽跨零。正 CI 主要在 low-support（见过但稀有）和
   块数 ≥13 的子群。不能把 low-support 写成“未见组合泛化”。

5. **是否达到 official test < 0.060 / 是否具有竞争力？**
   **未读 official test，不能回答。** valid best 0.076707 与 DCC-L 4-seed 均值
   0.075990 接近，但这是单 seed、机制主门未过的观察；本轮不声称达标或竞争力。

---

## 8. 本轮不声称

- 不声称“真实块归属一定有用”或“块级组合泛化被证实”；
- 不声称“块归属无用”（主门差 0.000116，且 W20/T100/final 正向，CI 跨零）；
- 不声称 R 优于 S（S 未跑）；
- 不声称在 any_unseen / novel block-size combo 上更好；
- 不声称接近或达到 DCC-L / official test 0.060；
- 不把 N 当作严格同信息对照：N 丢失归属信息，R−N 不是“只改组织、不改信息”的对照；
- 不把两个 seed 共用 s0/s1 SSL 说成独立 SSL 初始化族；
- 不把本轮负结果外推为“所有 block/biconnected 方法都无效”。

---

## 9. 复现与产物

### 9.1 关键文件

| 文件 | 作用 |
|---|---|
| `DESIGN.md` / `PROTOCOL.md` | 设计、预注册协议与停止规则 |
| `code/zbo_blocks.py` | Tarjan/NetworkX、块构造、R/N/S 内容、collate |
| `code/zbo_models.py` | BOSI-B + 块编码器 + 零初始化 reader_ext |
| `code/zbo_prep.py` | 构建 relation/block 缓存（远程写 `cache/prep.json`） |
| `code/zbo_checks.py` | C1–C15 CPU gates |
| `code/zbo_freeze.py` | 冻结 config、重算 code hashes |
| `code/zbo_train.py` | 正式 trainer（47,400 updates） |
| `code/zbo_analyze.py` | 配对 ΔMAE、分组、bootstrap CI、判决辅助 |
| `analysis/cpu_checks.json` | Phase 0 全部门结果 |
| `analysis/zbo_analysis.json` | 本轮分析结果（含 R/N 两条 run） |
| `configs/zbo_v1.json` | 冻结 config（协议、hash、fixed rows、判决规则） |
| `runs/R_s0/`, `runs/N_s0/` | 正式 checkpoint、曲线、train.json（gitignored） |

### 9.2 哈希与校验

- config sha256：`151deb021d12ad77d8997d5abf99c3a42012efd9bc73ca9e15f7fac35958c0bd`
- CPU gates sha256：`177271fb541ab5f732ab1f8303435ee016cfe4e05a147d2130c32595bc27a9fb`
- analysis sha256：`cff9197022be8454caf59d351807f8d3fc4b2f02dcdf9b7b2ff836832140ef0d`
- relation content hash：train `c698a2f3…`，valid `6505e23b…`
- block content hash：train `460d9dff…`，valid `00bb2f59…`
- R best.pt `04065fe2…`；N best.pt `9581c42d…`
- R/N initial state hash 均为 `bd18518b…`

### 9.3 复现命令

```bash
# 本地 CPU gates（无需 GPU）
uv run python tracks/structural-sufficiency/zinc_block_ownership_v1/code/zbo_checks.py
uv run python tracks/structural-sufficiency/zinc_block_ownership_v1/code/zbo_freeze.py

# 远端（经 rr；不要手工 SSH）
rr doctor res
rr deploy res
rr run res zbo-prep --pool res-local --result tracks/structural-sufficiency/zinc_block_ownership_v1/cache -- \
  uv run python tracks/structural-sufficiency/zinc_block_ownership_v1/code/zbo_prep.py
rr run res zbo-r-s0-v1 --pool res-gpu0 --result tracks/structural-sufficiency/zinc_block_ownership_v1/runs/R_s0 -- \
  uv run python tracks/structural-sufficiency/zinc_block_ownership_v1/code/zbo_train.py --arm R --seed 0 --device cuda:0
rr run res zbo-n-s0-v1 --pool res-gpu1 --result tracks/structural-sufficiency/zinc_block_ownership_v1/runs/N_s0 -- \
  uv run python tracks/structural-sufficiency/zinc_block_ownership_v1/code/zbo_train.py --arm N --seed 0 --device cuda:0
rr pull zbo-r-s0-v1
rr pull zbo-n-s0-v1

# 分析（只读 official train/valid；不读 test）
uv run python tracks/structural-sufficiency/zinc_block_ownership_v1/code/zbo_analyze.py
```

### 9.4 复核注意

- analysis 的 `valid_pred_sha256_match_*` 为 `false` 是 **GPU 训练时预测 vs CPU 分析时重算**
  的 float32 归约差异：R/N 的 max|Δpred| 分别为 3.8e-6 / 1.9e-6，不是模型或数据错位。
- `n_multi_0` 只有 1 个 valid 分子，不能作为组证据。
- `fixed worst-10` 的 win rate 只有 0.4，均值 +0.124 由少数大误差行主导；不要用该组
  单独声称改善。
- `analysis/zbo_analysis.json` 的 bootstrap CI 使用固定局部 RNG（seed 20261027），
  与调用顺序无关；artifact 用 `--utc 2026-10-27T12:12:00+0800` 生成，重跑应逐位复现。
  若省略 `--utc`，只有 `analysis_utc` 时间戳会变，科学字段不变。

---

## 10. 最终一句话

**真实块归属通道确实被训练并且方向性为正，但 seed-0 的 best-valid 主门差 0.000116
未过；按预注册停止，不买 seed 1、不跑 S。最终 `UNRESOLVED_DUE_TO_VARIANCE`：
不能宣称 ownership 有用，也不能宣称 ownership 无用。**
