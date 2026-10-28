# DESIGN — LCRC-v1

本文件冻结实现层面的计算图、输入语义、初始化顺序与已知限制。协议/门禁见
`PROTOCOL.md`。参数与数值均来自 `analysis/cpu_checks.json` 和
`configs/lcrc_v1.json`。

---

## 1. 输入（全部 label-free，来自原始图）

### 1.1 原子/键

- `atom_feat_v`：冻结 P3 的 108-d 原子描述（own type、degree、ring、min ring
  size、ring bond、incident bond-type counts、radius-1/2 type marginals、
  radius-2 标量、terminal/fused 角色）。不改写。
- `bond_feat_e`：P3 的 7-d 键描述（键型 one-hot + ring 标志）。
- `bond_index_e=(u,v)`、`bond_type_e∈{1,2,3}`。
- `global33`：冻结 P3 的 33-d 全图 topology 向量；标准化复用 P3 train-only
  mean/scale。

### 1.2 S1：中心原子聚合的邻接原子—键联合信息

对每个原子 `v`，把每条 incident bond 的 `(neighbour atom type, bond type)`
映射到 63 个联合码，使用共享 `Embedding(63, 32)` 求和：

```
S1joint_v = Σ_{u~v} E_nbr[type(u), type(bond_uv)]  ∈ R^32
```

同时保留 learned 键状态的和与均值（见 §2），其中 `b_e` 的输入已含两端原子状态
和键型，因此 S1 不是两个独立边际的简单拼接；`E_nbr` 的 lookup 是
`(neighbour type, bond type)` 的联合码。

### 1.3 S2：rooted 二跳 typed-path 统计

对每个 root `v`，枚举所有有序、非回溯的二跳路径 `v -bt1- w -bt2- u`，`u≠v`
（包括 `u` 与 `v` 相邻的闭合路径；`u=v` 的回溯路径排除）。每条路径贡献：

| 项 | 维度 | 内容 |
|---|---|---|
| `E_path[cat189]` | 32 | `cat189 = type(w)*9 + (bt1-1)*3 + (bt2-1)`，联合 (middle type, bt1, bt2) |
| `E_end[cat63]` | 32 | `cat63 = type(u)*3 + (bt2-1)`，联合 (end type, bt2) |
| **乘法 joint** | 32 | `W_j · (U[cat189] ⊙ V[cat63])`，rank-32 因子化，保留两个半边的**共现**信息 |
| `E_close[closure]` | 8 | `closure = 1[u 与 v 相邻]` |
| `gate[cat189] * a_u` | 96 | 按路径类型门控的二跳端点内容（`gate` 初始化为 1） |

S2 的每个量都按 root 求和。`cat189` 和 `cat63` 是联合码；乘法项不是边际拼接。
对 200 个真实分子的诊断：`cat189` 计数矩阵的 rank-1 能量均值 0.955（说明原始计数
以边际为主），但同步态合成测试显示：相同边际、不同共现的两组路径，乘法项输出
差异 max abs = **1.80**，而边际项差异 = **0.0**。因此联合交互确实存在，而不是被
边际替代。

### 1.4 原子标量（9 个，train-only 标准化）

`log1p(n_d2)`、`log1p(n_paths2)`、closure 比例、degree/4、in_ring、min_ring/8、
ring_bond_count/4、terminal、fused。它们只用于把离散/计数尺度放进局部 MLP；
不引入标签或目标公式。

### 1.5 关系原子对

只取最短路径距离 **≥3** 的无序对 `i<j`；短程内容已由 S1/S2 覆盖。类别为
6 个互斥类：

```
category = 2 * dist_bucket + ring_block
dist_bucket: d==3 -> 0; 4<=d<=5 -> 1; d>=6 (含不可达) -> 2
ring_block: 1 iff i,j 属于同一个"非平凡"双连通块（块顶点数 >=3，即环块）
```

定义在 CPU gate 中固定；200 个真实分子与独立 networkx 枚举逐对一致。
不按 valid 表现更换分类。

真实覆盖（512 个 train 分子）：平均 207.5 个合格对/分子（45–532）；六类非空率
[1.00, 0.945, 1.00, 0.324, 1.00, 0.021]；100% 分子有 ≥2 个非空类。

---

## 2. 共享局部内容编码器

```
a_v = AtomMLP(atom_feat_v)                          # 108 -> 128 -> 96, SSL init
b_e = BondMLP([a_u+a_v, |a_u-a_v|, a_u*a_v, bond])  # 295 -> 160 -> 96, SSL init
S1_v = Σ_{e∋v} b_e,  S1mean_v = S1_v / deg(v)
S2_v = [Σ E_path, Σ E_end, Σ W_j(U⊙V), Σ E_close, Σ gate*a_u]
h_v = LayerNorm( LocalMLP([a_v, S1_v, S1mean_v, S1joint_v, S2_v, scalars_v]) )
      LocalMLP: 529 -> 192 -> 64
```

- 只有一次前向；没有迭代 hidden-state 更新、没有 attention、没有 junction
  encoder/token、没有 per-wedge learned token。
- `atom_encoder`/`bond_encoder` 从仓库已有 **train-only masked-SSL** checkpoint
  加载（seed0 用 s0，seed≥1 用 s1），监督训练中正常微调；不加载任何 DCC-L
  监督权重。
- `path_gate` 初始化为 1，使二跳内容在 step 0 就存在；`joint_proj` 是普通
  Linear，不特殊初始化。

---

## 3. 三个 arm 的 forward

### A

```
y_A = Σ_v f(h_v) + ρ( [global33, mean_v h, mean_v h², log(1+|V|)] )
f: 64 -> 64 -> 1;  ρ: 162 -> 64 -> 1
```

`ρ` 容量刻意很小（10,497 参数）；分析时记录 `|Σ f|` 与 `|ρ|` 的尺度，避免把
A 包装成"局部贡献"但实际由全局 reader 承担。

### R

```
z_ij = φ([h_i+h_j, |h_i-h_j|, h_i⊙h_j]) ∈ R^32      # 192 -> 128 -> 32
M_r = (1/n_r) Σ_{(i,j): cat=r} z_ij                  # 空类 -> 0
Δ_rel = Σ_r head( MLP_shared([M_r, log1p(n_r), 1[n_r>0], E_cat[r]]) )
        MLP_shared: 42 -> 48 -> 24; head: 24 -> 1, 零初始化
y_R = y_A + Δ_rel
```

`head` 最后一层零初始化，所以 step-0 `y_R == y_A`（CPU gate 实测 max abs 0.0）。
`E_cat` 是 6×8 的类别身份 embedding；它让"哪一类拥有哪份内容"在两类计数相同时也
可区分（G5 的 Δ_rel diff 0.00885），而不是仅靠计数差异。共享 MLP + sum 仍然是
"共享关系读出网络"，不是每类独立大 reader。

### N

与 R 相同的 `h`、相同的 `z` 多重集、相同的真实 `n_r`/existence/类别 embedding；
唯一差异：

```
M_r^N = mean_{所有合格对} z_ij   (若 n_r>0；空类仍为 0)
```

即移除"哪份内容属于哪个结构关系"的对应。R/N 参数逐 tensor 相同、初始化逐
tensor 相同、数据顺序相同。

---

## 4. 初始化顺序与可复现性

`build_model(arm, norm, seed)`：

1. `torch.manual_seed(seed)` → 构造全部共享模块（atom/bond encoders、local、
   f、ρ、buffers）。
2. 加载 SSL 的 atom/bond encoder 权重。
3. 若 arm∈{R,N}：`torch.manual_seed(seed+90731)` → 构造 `RelationHead`（最后）。
4. `.to(device)`。

因此：
- A/R/N 的共享参数逐 tensor 相同（关系模块不在共享构造之前消费 RNG）；
- R/N 的 full initial state 逐 tensor 相同；
- 关系模块的初始化不依赖 arm。

数据顺序只由 `Generator(seed+101)` 决定；CPU gate 验证 600 epoch 的 permutation
stream 逐 epoch 相同。

---

## 5. 参数审计

| 模块 | A | R/N |
|---|---:|---:|
| atom_encoder | 26,336 | 26,336 |
| bond_encoder | 62,816 | 62,816 |
| local | 133,645 | 133,645 |
| f | 4,225 | 4,225 |
| rho | 10,497 | 10,497 |
| relation | 0 | 32,145 |
| **total** | **237,519** | **269,664** |

均 <500k。R/N 比 A 多 32,145（+13.5%）；这正是 R−A 不能单独归因于机制的容量差。

---

## 6. 与旧实验的关系（实际 forward，不是命名）

- **不是 R0**：R0 每原子用 3 个半径的 exact 693 类 joint count、3 个独立 MLP、
  宽 reader；本实现只有一次局部 MLP + 原子标量和 + 小 ρ。
- **不是 T**：T 用 P3 junction/atom/bond 的 rank-16 关联张量；本实现没有
  junction 张量，也没有 bond-junction incidence。
- **不是 DCC-L**：没有 junction encoder/state，没有 per-wedge token，没有
  513-d reader；局部读出是 `Σ f(h_v)`。
- **不是 AR 的重跑**：AR 在 DCC-L trunk 上换 reader；本实现没有 DCC-L trunk，
  且新增了 R/N 关系对照。
- **不是 RCC 的重跑**：RCC 是 bond state 上的固定二次型；本实现是原子对内容的
  类别均值 + 归属对照。RCC 的警告仍然适用：关系统计可能与局部池化冗余。本轮
  用 ridge probe 量化：common categories (0,2,4) 的 R 内容约 70.2% 可由 A 的
  池化特征线性重建（初始化时），约 29.8% 不可；h 的 participation ratio 仅
  1.19。这个风险没有被排除，而是作为结果分析的一部分。

---

## 7. 已知限制（训练前写明）

1. 关系类别只有 6 类，且 `d>=6 & ring` 只覆盖 2.1% 分子（0.15 对/分子）；该类
   的统计几乎不进入训练信号。
2. h 在初始化时维度很低（participation ratio 1.19）；若训练后仍低，R 的 pair
   内容会接近低维函数，R≈N 的可能解释是表示退化，而不是"归属无信息"。
3. ridge probe 的 R² 是线性下界；训练后的 h 可能更高维，也可能更低维。
4. 只做单次关系组合；没有多层、没有 attention、没有迭代。结果不能外推到
   "所有关系机制"。
5. 没有对关系类别定义、pair 下界、rank、宽度做网格搜索；这是刻意的预算纪律。
6. valid 是历史已暴露的开发集；DCC-L 参照是历史 4-seed 数字，不是本轮重训；
   test 已 spent。
