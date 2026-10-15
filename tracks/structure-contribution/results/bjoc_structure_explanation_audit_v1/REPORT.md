# BJOC-Null 结构信息流与解释可行性审计 v1 — REPORT

* 命名空间：`bjoc_structure_explanation_audit_v1`（track `tracks/structure-contribution`）
* 协议：`PROTOCOL.md`（运行前冻结；哈希规则、干预语义、停止条件均在其中）
* 冻结模型：BJOC-Null 413,505 params + full-10k masked atom/bond SSL 编码器 + R3 recipe（360 epoch cosine），
  `soup_R3_seed0.pt`（`a021d36e…`）与 `soup_R3_seed1.pt`（`b622824a…`）
* 数据：只使用官方 train 10k 的冻结**结构**缓存 `objects_train10k.pt`（`14326a7a…`，无标签字段）。
  官方 valid/test、Monitor/Locked 标签**从未加载**；本审计不计算任何 MAE/RMSE；无梯度更新、无训练。
* 执行：本地 CPU，`torch 2.5.1+cu124`；全部阶段 + 测试通过。
* 结论标签（可并存）：**`NONLOCAL_MIXING_LIMITS_ATTRIBUTION`（支持）**、
  **`GLOBAL_BYPASS_OR_REDUNDANCY_OBSERVED`（支持，表示层面；reader 使用层面较弱）**、
  `EXISTING_INTERFACE_POTENTIALLY_USABLE`（对连通子结构解释**不支持**）、
  `INCONCLUSIVE`（忠实子结构归因、局部信息是否能提升 MAE）。

---

## 0. 一页结论摘要

1. **`a_i` 不是“第 i 个原子的标签”。** 固定特征已含半径 1/2 的类型直方图，以及由双连通分量决定的
   环特征（`in_ring`、最小环大小、环键计数、fused/ring-system 标志）。类型翻转实验：翻转半径 1 的原子
   只改变 `neighbour_types`，翻转半径 2 的原子只改变 `radius2_types`（16/16 smoke 分子）；
   14-环合成图在切掉**远端**键 (6,7) 后原子 0 的环特征从 `{in_ring=1, min-ring bucket=5}` 变为
   `{0, -1}`；切掉另一个双连通分量上的键则不变。→ 固定特征层已有“分量级/全局”依赖，但它是确定、
   可审计的图算法，不是学习到的信息混合。
2. **严格局部性在 Null 分组层（L3）失效，且是公式必然。** `W_b = n_b·mean_graph(w)`、
   `V_b = n_b·mean_graph(w²)`；`n_b = deg(u)+deg(v)−2` 在全部 10,000 个训练结构、
   249,279 条键、346,358 个 junction 上**逐值精确**成立。任何单个 `w_j` 的编码扰动都会改变**全部分子
   内键**的 `c_b`：平均 24.7/24.7 条键（即 100%），且与图距离**无衰减**（距离分桶 d=0…10 的
   `mean|Δc|` 约 0.015–0.029；Spearman(d,|Δc|)=−0.09/−0.11）。
3. **`c_b` 只能被解释为“以 Bond 为索引的局部项 + 全图 junction 均值项的混合”。** 它绝不满足
   “只依赖该键有限邻域”的局部语义。用零 `W/V` 的同一 composer 反事实分解：`‖c−c_local‖/‖c‖`
   均值 0.632/0.646，中位数 0.594/0.596——约 60% 的 `c_b` 向量范数来自图级项。
4. **四条 reader 通路的干预敏感性（冻结模型诊断，不是“独立贡献”）**：置零时
   `sum_atom`（1.16/1.12）≫ `z_bond`（0.45/0.43）≈ `sum_atom_squared`（0.94/0.69）> `global_topology`
   （0.15/0.16）；训练参照替换时 `z_bond` 升到 0.57，`sum_atom_squared` 降到 0.46–0.47，`global` 最低
   （0.07/0.08）。→ **置零不是通路影响的下界**；方向定性结论（global 最小、atom 最大、
   bond 汇总居中）在两种干预和两个 seed 下一致（per-molecule Δ 相关 0.87–0.99）。
5. **全局旁路/冗余得到支持（表示层面）**：`global_topology`（33 维）被其余 288 维的岭回归
   R²=0.917/0.914 线性恢复；`sum_atom`/`sum_atom_squared` 的 R²=0.95/0.90。这是一个架构性旁路：
   33 维固定全图统计**直接**进入 reader，完全绕过 a/b/w/c 对象层级。
6. **连通子结构解释接口不成立。** Phase C：遮蔽 8 条**连通**键在 `z` 中的贡献，与遮蔽同大小的
   随机连通/任意键集合在统计上无法区分（paired mean 差 0.010 / −0.001，median ≤ 0.002，两个 seed
   符号不一致；p90 0.04–0.08）。连通性没有带来任何独特效应。因此不能实现看似漂亮的 Motif 搜索；
   Phase C 按协议停止在“接口不充分”。
7. **对 MAE 的含义（背景性、非本轮选择依据）**：本审计没有发现任何证据表明“局部/ownership 信息”
   是当前预测误差的瓶颈；这与旧审计的 `NULL_OWNERSHIP_BLINDNESS_VERIFIED`、
   `J_MOMENTS_NO_CLEAR_GAIN` 以及 GLRC/DLP 的 `NO_GAIN` 负结果一致。局部信息是否值得为
   **可解释性**而重构、以及重构是否损失精度，是**未决**问题，需要新的预注册实验。

---

## 1. 审计范围、证据分级与执行边界

**三层证据分级**（报告全文使用）：

1. **公式必然（algebraic）**：从模块公式直接证明，并附实现级验证；对任何权重成立。
2. **冻结模型数值干预（numeric）**：在 R3 seed0/1 冻结权重上的可观测量；受干预分布外性与 Reader
   非线性限制，不能升级为化学因果或独立贡献。
3. **无证据/未决（unsupported）**：本轮没有支持。

**真实计算依赖图**（`structural_relation_objects_v1.py` / `structural_relation_composer_v1.py`）：

```
raw graph
  → 固定结构特征  a_i_raw(108), bond_feat(7), junc_feat(32), g(33)
  → a_i = AtomMLP(a_i_raw)                        (逐点，不扩大感受野)
  → b_uv = BondMLP(a_u,a_v,bond_feat)             (两个原子上下文之并)
  → w_ijk = JuncMLP(a_i,a_j,a_k,b_ij,b_jk,junc_feat)
  → Null 分组：n_b = deg(u)+deg(v)-2; W_b = n_b·mean_G(w); V_b = n_b·mean_G(w²)
  → c_b = PSI(b_b, W_b, V_b, log1p(n_b), bond_feat)
  → reader_in = [Σ_b c_b (96), Σ_i a_i (96), Σ_i a_i² (96), g (33)]  = 321 维
  → y_hat = Reader(reader_in)
```

**审计纪律核对**（全部通过，见 `phase_b_baseline.json`、`source_manifest.json`）：

* checkpoint/缓存 SHA-256 与既有记录一致（`soup_R3_seed0 a021d36e…`、`soup_R3_seed1 b622824a…`、
  `objects_train10k 14326a7a…`、`prep d0fb86a3…`、`pretrain_encoder_s0 006fabc1…`、
  `s1 ff8d6474…`）；参数数 413,505、reader_in 321 维精确。
* `forward(batch)` 与 `reader(forward_features(batch)["reader_in"])` 最大差 **0.0**。
* 原子重标号 4.8e-7、物理键顺序置换 1.2e-7、junction 行置换 1.2–2.4e-7、分子批顺序 0.0（float32
  归约噪声内）；单分子与批推理 4.8e-7。
* 样本规则（协议 §4，label-blind）：对对象张量字节做 SHA-256 排序，诊断集 rank 0–255、
  参照集 rank 256–1279，不重叠；运行前冻结。诊断集结构规模与参照匹配键见
  `sample_manifest.json`。

---

## 2. Phase A — 静态计算依赖

### 2.1 逐层依赖表（完整机器可读版在 `phase_a_dependency.json`）

| 层 | 表示 | 真实依赖来源 | 依赖类别 | 锚点 | 聚合后能否逐实例恢复 | 证据 |
|---|---|---|---|---|---|---|
| L0 | x / edge_index / edge_attr | 原子、键 | 输入 | 每原子/键 | 是 | 定义 |
| L1 | `a_i_raw` (108) | 自身类型；度、邻键类型直方图、2-path（r=1）；邻居类型直方图（r=1）；半径 2 类型直方图（r=2）；三角形（r≤2）；`in_ring`、`min_ring_size`、环键计数、fused（双连通分量，最坏全图） | 逐点+r1+r2+分量级环特征 | 原子 i | 否 | 代码 + A1 探针 |
| L1 | `bond_feat` (7) | 键类型；is_ring / 端点环标志（双连通分量） | 逐点+分量级 | 物理键 | 否 | 代码 |
| L1 | `junc_feat` (32) | 中心度、端点互连、公共邻居（r=2）、环键标志、中心环特征、两键类型 | r2+分量级 | junction | 否 | 代码 |
| L1 | `g` (33) | 全图计数/距离/环统计/类型占比 | 全图摘要 | 无 | 已是摘要 | 代码 |
| L2 | `a_i` (96) | 逐点 MLP | 继承 L1 | 原子 i | 否 | 代码 |
| L2 | `b_uv` (96) | `a_u,a_v,bond_feat` 之并 | 局部（键半径 ≤3）+环标量 | 物理键 | 否 | 代码 |
| L2 | `w_ijk` (96) | `a_i,a_j,a_k,b_ij,b_jk,junc_feat` | 局部（中心 j 半径 ≤3）+环标量 | junction | 否 | 代码 |
| **L3** | `W_b,V_b,n_b` | `n_b` 局部；**`W_b/V_b` 依赖全图 junction 多重集** | **非局部（Null 公式）** | 仅键索引 | 否 | 公式 + 全局扫描 |
| **L4** | `c_b` (96) | 局部 `b_b`/`bond_feat` + 全局 `W_b/V_b` | **锚定在键、但全局混合** | 仅键索引 | 否 | A3 + B3 |
| L5 | `reader_in` (321) | Σc、Σa、Σa²、g | 图级聚合 | 无 | 否 | 代码 + 旧审计碰撞搜索 |
| L6 | `y_hat` | 2 隐层 MLP | 全局 | 无 | 否 | 代码 |

### 2.2 A1 — Atom 表示的真实感受范围

* **类型信息**：16 个 smoke 分子的定点翻转（中心取最大度原子，在其 r=1/r=2/r=3/r=4 邻居中各翻一个
  原子的类型）显示：只有 `neighbour_types` 在 r=1 改变、`radius2_types` 在 r=2 改变；r≥3 无类型块变化；
  自身 `atom_type` 块从不因“别人改变”而改变。这与代码里 `paths2/triangles/n_d2/(degree)` 等是
  类型无关的拓扑量一致。
* **拓扑/环信息**：环相关块是全局（双连通分量级）函数。合成 14-环切掉远端键 (6,7) 后原子 0 的
  `in_ring` 1→0、最小环 bucket 5→−1；双环+桥结构切掉另一环的键则原子 0 的环特征不变。→
  表述应当是“**双连通分量级全局**，不一定是整个分子”。
* **结论（公式必然 + 数值）**：`a_i` 的感受野 = 自身类型 + r≤2 类型统计 + r≤1 拓扑统计 + 分量级环特征。
  把 `a_i` 当作“仅由第 i 个原子标签决定”是错误的。L2 的学习模块（逐点 MLP、键/结的邻域拼接）
  **不扩大**感受野。

### 2.3 A2 — Null 的全局混合与 `n_b` 恒等式

（`phase_a_dependency.json: a2_static_identity / a2_model_formula_check`）

* 全量 10,000 分子、249,279 条键、346,358 个 junction：`n_b = deg(u)+deg(v)−2` 的
  最大绝对偏差 **0.0**；`Σ_b n_b = 2·n_junc` 最大偏差 0.0。→ 在简单分子图与当前
  junction 定义（中心 j 的任意两个邻居 i<k 组成一个 junction，恰好触及两条键）下**严格成立**。
* 冻结模型数值：`W_b` 与 `n_b·mean_G(w)` 逐值相等（差值 0.0），`V_b` 同理（0.0）；
  `Σ_b W_b = 2·n_junc·mean_G(w)` 的 float32 归约差 9.5e-6（容差 4.8e-4，相对 ~2e-7）。
* **必须保留的精确表述**：Null 只在 **Junction 编码之后**丢弃了精确的 Junction→Bond ownership。
  Junction 编码器 `JuncMLP` 仍然读取真实相邻的两个 bond 嵌入与三个原子，所以“哪两个键在哪个中心
  成结”会先进入 `w`，再由 `mean_G(w)` 混合。这不是“完全不使用局部结构关系”。
* **可恢复性**：`n_b` 是 `deg(u)+deg(v)−2` 的确定性函数（局部信息保留）；ownership 本身在图级
  均值后不可逐实例恢复；每键的“哪个 junction 贡献了什么”不可恢复。跨分子碰撞搜索的历史结论
  （`REAL_COLLISION_NOT_FOUND_WITHIN_BUDGET`）说明聚合并非简单到可轻易找到精确碰撞，但这不是
  局部性证据。

### 2.4 A3 — `c_b` 能否代表局部结构

**答：不能。** `c_b` 以真实 Bond 为索引锚点，但它通过 `W_b/V_b` 依赖**全分子的 junction 编码多重集**。

实现级证据（`phase_a_dependency.json: a3_remote_junction_witness`）：取 36 junction / 27 键的分子，
把**任意一个** junction 的编码 `w_j` 加 0.1·单位向量：

* 重算的 `ΔW_b` 与公式 `n_b·(0.1·δ/n_junc)` 最大差 **8.0e-8**（float32 舍入）；
* 分子内 27/27 条键的 `|Δc_b| > 1e-6`（最大 7.9e-4）。

结合 B3（下文）的 64 分子正式统计：与图距离无关，100% 键受影响。→ 结论句应写成：

> **`c_b` 虽以真实 Bond 为锚点，但并不是严格局部的结构表示；它是“该键局部术语 + 全图 junction
> 均值项”的混合。**把“有明确索引”误认为“局部且独立的计算贡献”是不成立的。

---

## 3. Phase B — 冻结模型的信息通路诊断

### 3.1 B1 一致性（`phase_b_baseline.json`）

见 §1 纪律核对。补充：两个 seed 的行为一致；本审计全程 `eval()` + `no_grad()`，未修改任何权重。

### 3.2 B2 四条 reader 输入通路（`phase_b_pathways.json`、`phase_b_deltas.npz`）

样本：诊断 256 分子（哈希 rank 0–255），参照 1024 分子（rank 256–1279）。干预：置零（复现旧审计口径）
与“匹配参照替换”（用 `(n_atoms,n_bonds,n_junctions)` L1 最近参照分子的同一块替换）。`Δpred` 为原始 y 单位。

**表 B2-1 置零（|Δpred|）——与旧审计定性结论一致（旧审计：composer 0.466、atom-sum 0.833、
atom-sq 0.892、global 0.174，16 个 Fit 分子、P3-D-s0；本表为 R3 full-10k，256 个 train 分子）**

| block | seed0 mean | seed0 med | seed0 p90 | seed1 mean | seed1 med | seed1 p90 |
|---|---:|---:|---:|---:|---:|---:|
| `z_bond` (Σc) | 0.452 | 0.379 | 0.907 | 0.429 | 0.343 | 0.843 |
| `sum_atom` | 1.161 | 1.098 | 2.147 | 1.124 | 1.064 | 2.064 |
| `sum_atom_squared` | 0.935 | 0.772 | 1.942 | 0.690 | 0.475 | 1.592 |
| `global_topology` | 0.152 | 0.083 | 0.390 | 0.158 | 0.094 | 0.366 |

**表 B2-2 匹配参照替换（|Δpred|）——新增的分布内对照**

| block | seed0 mean | seed0 med | seed0 p90 | seed1 mean | seed1 med | seed1 p90 |
|---|---:|---:|---:|---:|---:|---:|
| `z_bond` | 0.569 | 0.437 | 1.119 | 0.570 | 0.426 | 1.140 |
| `sum_atom` | 0.951 | 0.710 | 2.027 | 0.950 | 0.747 | 2.116 |
| `sum_atom_squared` | 0.471 | 0.351 | 1.061 | 0.458 | 0.322 | 1.099 |
| `global_topology` | 0.072 | 0.053 | 0.152 | 0.081 | 0.065 | 0.168 |

**关键读数**

* **两 seed 方向一致、量级一致**：8 个（block × 干预）组合的 per-molecule Δ 相关
  Pearson 0.87–0.99、Spearman 0.88–0.99、符号一致率 0.80–0.99。
* **零替换不是下界**：`z_bond` 的参照替换（0.57）比置零（0.43–0.45）更大；
  `sum_atom_squared` 则相反（0.46–0.47 vs 0.69–0.94）。→ 只用置零会系统性误估 `z_bond` 通路。
* **分布外程度**（`phase_b_pathways.json: ood`）：
  * 逐维均值 `|z|`：`sum_atom_squared` 置零 = 2.33/2.30（自然 0.70）——该置零干预本身强 OOD；
    `z_bond` 置零 = 0.07（自然 0.69，该块均值近 0）；参照替换的所有块 ≈ 自然水平（0.66–0.74）。
  * 多变量最近邻比（modified 的最近自然点距离 / 自然的最近自然点距离，321 维标准化空间）：
    0.90–1.50；`sum_atom_squared` 置零最高（1.50/1.49），其余 ≤1.12。
  * 自替换对照：max |Δpred| = 0.0（实现正确性）。
* **通路冗余（岭回归，参照集拟合、诊断集评估；λ=1e-2）**（旧审计同序：
  composer 0.536 / atom 0.968 / atom² 0.927 / global 0.938）：

| block (dim) | seed0 R² from other 3 | seed1 R² from other 3 |
|---|---:|---:|
| `z_bond` (96) | 0.365 | 0.364 |
| `sum_atom` (96) | 0.950 | 0.954 |
| `sum_atom_squared` (96) | 0.899 | 0.905 |
| `global_topology` (33) | **0.917** | **0.914** |

  线性 CKA：`z_bond` 与其它块 0.08–0.18；`sum_atom ↔ sum_atom_squared` 0.39；atom 块与 global 0.34–0.40。

**允许的表述（严格限定）**：以上统称 **frozen-model pathway sensitivity**。因为输入块相关、Reader
非线性、且部分干预 OOD，**不得**称为 Atom/Bond/Global 的“独立贡献”。可说的是：
`global_topology` 通路敏感性最小且近乎可被其余三块线性重建；atom 汇总通路最敏感；`z_bond` 居中且
对自然参照替换最敏感。

### 3.3 B3 — 远端结构信息是否进入当前 Bond 表示（`phase_b_nonlocal.json`）

干预：对选定 junction `j*` 的**编码表示**加 `α·δ`（`δ` = 分子内 `w` 逐维标准差；α∈{0.25,0.5,1,2}），
按原 Null 公式重算 `W,V,c,reader_in,pred`，其余编码冻结。`(e*,j*)` 取图上距离最大的一对
（64 个诊断分子；最大距离 5–15）。

* **公式必然 + 数值验证**：`ΔW_b` 与 `n_b·αδ/n_junc` 逐键符合，平均最大误差 1.2e-7。
  `Σ_b ΔW_b = 2αδ`，**与分子大小无关**（这是一阶 prediction 变化不随规模增长的原因）。
* **所有键受影响、无距离衰减**：

| 距离桶 d | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| seed0 mean\|Δc\| | .0202 | .0270 | .0289 | .0265 | .0269 | .0254 | .0272 | .0253 | .0216 | .0197 | .0182 |
| seed1 mean\|Δc\| | .0190 | .0221 | .0212 | .0217 | .0215 | .0205 | .0213 | .0200 | .0170 | .0154 | .0158 |

  距离与 `|Δc|` 的 Spearman：−0.094（s0）/−0.108（s1）——几乎无结构，远端键与近端键效应相当。
* **剂量-响应近似线性**：`mean|Δc|` = 0.0059/0.0117/0.0234/0.0535（s0，α=0.25→2），
  归一化后 0.0237/0.0234/0.0234/0.0267；`mean|Δpred|` = 0.0023/0.0043/0.0078/0.0170。
* **规模依赖弱**：Spearman(n_bonds, |Δpred|) = −0.095/−0.220；与 `ΣΔW=2δ` 的解析结论一致。
* **组织模式对照（同一冻结权重、同一扰动）**：`organized` 分组下只有 `j*` 的两个 bond 的 `W` 变化
  （平均 2.0 条、最大非零距离 0），而 `mean|Δpred|` 反而更大（0.040/0.032 vs Null 0.0078/0.0112）。
  → Null 的混合是**稀释**而不是抹掉 junction 内容。
* **意义**：这是 BJOC-Null **非局部计算依赖**的直接、可复现证明，也说明其量级可由 `mean_graph(w)`
  公式预测。它**不是**合法分子编辑，也**不是**化学因果实验（协议 §5 已冻结该语义）。

---

## 4. Phase C — 连通子结构解释接口可行性

（`phase_c_interface.json`；32 个诊断分子、两个 seed、k=8）

预先判断（Phase A/B 后）：`c_b` 已因全局混合失去严格局部语义，因此“选中一个连通子结构”在模型的实际
计算里对应的内部干预只能是**遮蔽 `z` 中该子结构的 `c_b` 行**。这是一个有限、明确的干预，故按协议
做最小探针，但结论只限定为 **Bond 汇总通路解释**。

**结果**

| 指标 | seed0 | seed1 |
|---|---:|---:|
| mean \|Δpred\|：连通子图遮蔽 | 0.133 | 0.115 |
| mean \|Δpred\|：随机连通子图（8 次平均） | 0.115 | 0.112 |
| mean \|Δpred\|：随机任意子图（8 次平均） | 0.118 | 0.109 |
| paired（连通 − 随机连通）mean / median / p90 | +0.0104 / +0.0004 / 0.083 | −0.0013 / −0.0014 / 0.044 |
| paired（连通 − 随机任意）mean / median / p90 | +0.0076 / +0.0017 / 0.093 | +0.0071 / +0.0070 / 0.053 |
| 空集遮蔽（校验） | 0.0 | 0.0 |
| 全 Bond 遮蔽（校验） | 0.938 | 0.938 |
| `‖c−c_local‖/‖c‖` mean / median / p90 | 0.632 / 0.594 / 1.062 | 0.646 / 0.596 / 1.079 |
| 用 `z_local`（W=V=0 反事实）替换 `z` 的 mean \|Δpred\| | 0.245 | 0.270 |

**判断（数值 + 公式）**：连通性与随机选择没有可区分的效应（两个 seed 的 paired 符号都不一致，
median ≤ 0.002）；同时每个 `c_b` 的约 60% 范数属于图级项。因此：

* 仅修改 `z_bond` 的结论只能限定为 Bond 汇总通路敏感性；
* **不能**把“遮蔽 `c_b` 行”解释成真实分子的删键/删子结构/官能团替换；
* 现有表示**不具备**“连通子结构 ⇒ 可识别的内部计算”接口。按协议 **停止 Phase C 后续实现**
  （不做 Motif 搜索、不做可学习分割、不训练）。

**接口缺失清单（可直接作为下一轮 preregistration 的缺口说明）**：
1. **缺少局部的、有身份的贡献项**：organized 分组 `W_b=Σ_{j∋b} w_j` 才有明确 ownership，但当前
   R3 是 Null，organized 只在旧审计中作为（未训练）对照出现；
2. **缺少 reader 通路的分离**：`c_b` 的图级项与局部项没有分开进入 reader，任何子集干预都会同时移除
   两者；
3. **缺少忠实度协议**：需要预注册的 deletion/insertion-style faithfulness 指标与随机/连通对照
   （Phase C 已给出对照原型）；
4. **缺少训练信号**：没有任何训练目标使“局部项”可识别或可归因。

---

## 5. 按证据强度回答 Q1–Q5

### Q1. BJOC-Null 的哪一层开始失去严格局部结构依赖？

* **L1（固定特征）就失去“只依赖本原子”的性质**：半径 1/2 的类型直方图 + 双连通分量级环特征
  （公式必然 + 类型翻转探针 + 14-环/双环桥合成见证）。
* **L3（Null 分组）是第一个让学习表示本身非局部的层**：`W_b=n_b·mean_G(w)`、`V_b=n_b·mean_G(w²)`
  使任意键的表示依赖全图 junction 多重集（公式必然；`n_b` 恒等式全量验证；远端扰动无衰减）。
* 精确说法：L1 有**确定性的分量级全局特征**（可审计、无学习混合）；L3 有**不可分离的全图混合**。

### Q2. Bond、Atom 汇总及全局 Reader 通路是否存在明显的信息混合或旁路？证据等级？

* **Bond 通路**：存在**数学必然的全局混合**（`mean_G(w)`），且被数值干预直接观测（100% 键、
  无距离衰减、剂量线性、公式精确）。等级：公式必然 + 数值。
* **Atom 汇总通路**：无学习到的跨原子混合（逐点编码后求和），但每个 `a_i` 携带分量级环标量；
  两块之间及与其它块高度线性冗余（R² 0.95/0.90）。等级：数值（表示层面），reader 使用层面无独立证据。
* **全局通路**：存在**架构性旁路**——33 维固定全图统计直接进入 reader（绕过对象层级）；
  同时它被其余三块线性恢复 R² 0.92，且干预敏感性最小。等级：公式（旁路结构）+ 数值（冗余与灵敏度）。
* 三者都有**混杂/non-identifiability**：输入块相关、Reader 非线性、部分干预 OOD；因此“通路敏感性”
  不是“独立贡献”。等级：数值 + 明确限制。

### Q3. 当前模型是否具备可信的连通子结构解释接口？缺什么？

**没有。** 具体缺失见 §4 清单：ownership 明确的局部项、reader 通路分离、忠实度协议、使局部项可识别
的训练信号。用现有表示做连通子结构解释会不可避免地与全图均值项纠缠（~60% 范数），且连通选择与随机
选择不可区分（数值）。

### Q4. 下一步最值得做什么？（保留预测器研究解释 / 局部信息流重构 / 暂停新架构）

* **保留现有预测器 + 研究解释方法**：仅在**全局/通路级**归因（如四块敏感性、剂量-响应、
  cross-molecule 自然替换）范围内诚实可行；对“连通子结构解释”不可行。
* **局部信息流重构**：是唯一可能产生忠实局部接口的路线（organized grouping 或把 `c_b` 的图级项
  与局部项拆成两条 reader 通路），但属于架构变更 + 重训练；本轮**没有**证据表明它保持精度。
* **暂停新架构**（针对 MAE 路线）：与本轮及旧审计证据一致——局部/ownership 通道（GLRC、DLP、
  junction moments）均未带来稳定增益；全-10k 报告指向 tail/scale composition，而非局部性。
* 综合：**准确性路线暂停新架构；解释性路线若要继续，必须把“局部通道重构”作为一个以解释忠实度为
  主指标、以精度平价为约束的预注册实验，而不是继续刷 MAE。**

### Q5. 哪些已有结果被复现、哪些新增发现改变了理解？

**复现（既有结论一致）**
1. Null ownership 盲性：编码后 ownership 被丢弃；junction 行置换对 Null 是精确 no-op
   （置换后 `c`/`reader_in` 差 <1e-6）。
2. reader 四块置零的定性排序与量级：global 最小、atom 最大、composer 居中
   （旧审计 vs 本审计，模型/样本不同，非逐位复现）。
3. 表示冗余排序：atom 块高度冗余、global 块 R²~0.92、composer 最不冗余（旧 0.536 → 本 0.365，
   方向一致、数值因协议而异）。
4. 无精确跨分子碰撞（旧审计）与本轮“全局混合但压缩”的描述一致。

**新增发现（改变理解）**
1. `c_b` 的非局部性是**公式层**的，不只是 reader 非线性：单 junction 扰动精确改变全部 `c_b`，
   且与距离无关（此前只有“mean_graph 依赖”的文字描述，没有逐键距离谱）。
2. 匹配参照替换表明**置零不是敏感性下界**：`z_bond` 对自然替换更敏感（0.57 vs 0.43–0.45）；
   `sum_atom_squared` 置零本身强 OOD（z=2.3）。这修正了只依赖置零的旧口径。
3. organized 对照（同一权重）把扰动局限在 2 条键且 `|Δpred|` 更大（0.03–0.04 vs 0.008–0.011）——
   Null 是稀释而非消除 junction 内容。
4. Phase C 首次直接显示：**连通子结构遮蔽与随机遮蔽不可区分**，且约 60% 的 `c_b` 范数是图级项。
5. `a_i` 感受野的精确表述：类型统计 r≤2、环特征为双连通分量级（14-环远端切键改变原子 0；
   另一分量切键不改变）。

---

## 6. 结论标签

| 标签 | 状态 | 依据 |
|---|---|---|
| `NONLOCAL_MIXING_LIMITS_ATTRIBUTION` | **支持** | `c_b` 的全局项公式必然；远端扰动 100% 键、无距离衰减；Phase C 连通=随机 |
| `GLOBAL_BYPASS_OR_REDUNDANCY_OBSERVED` | **支持（表示层面）** | `g` 直接进 reader（架构旁路）；其余三块线性恢复 `g` 的 R²=0.92；`g` 干预敏感性最小 |
| `EXISTING_INTERFACE_POTENTIALLY_USABLE` | **不支持**（对连通子结构解释） | 无可分离局部项、无 reader 通路分离、无忠实度训练信号 |
| `INCONCLUSIVE` | 对两件事 | (i) 忠实连通子结构归因在重训练后能否成立；(ii) 局部信息是否能提升 MAE |

标签并存是结论本身的一部分：全局/通路级归因可用，局部结构级归因不可用。

---

## 7. 研究决策建议（一页以内）

**当前是否需要购买 GPU？——对本次“冻结模型结构解释”问题：不需要。** 本轮所有关键证据（公式、
10k 全量 `n_b` 恒等式、两 seed 的 256/1024 分子干预、远端扰动、Phase C）都在本地 CPU 上完成，
总时长数分钟；没有再需要 GPU 的未解子问题。**不要**为“在冻结 R3 上做 Motif 搜索/子结构解释”
购买 GPU——本审计已证明该接口不成立。

**唯一值得考虑购买 GPU 的下一轮（不是本轮，且必须先预注册）：**
一个**局部-全局信道分解**的配对实验。具体最小设计（供决策，不在本任务中启动）：

* 模型：保持 BJOC 对象层级与参数量级；把 `c_b` 改为显式两项：局部项 `c_b^loc = PSI(b_b, n_b, bond_feat)`
  与图级项 `c_b^glob = PSI(...W_b,V_b...)`，分别求和后作为两条 reader 通路（或用 organized grouping
  `W_b=Σ_{j∋b} w_j`）；两个 seed、与 R3 相同的 full-10k + SSL 初始化与配方。
* 预注册主指标有两个，缺一不可：
  1. **精度平价**：相对冻结 R3 的 paired valid 差异不超过预设界（例如 |ΔMAE| ≤ 0.002）；
  2. **忠实度**：在留出结构上，连通子结构遮蔽相对随机连通/任意遮蔽必须显著（预注册方向与阈值），
     且局部项贡献可被 ablation 单独测量。
* 它将检验的**尚未解决的科学命题**：*“在一个与 BJOC-Null 精度平价的模型里，能否存在一个忠实于模型
  实际计算的连通子结构解释接口？”* 若平价失败 → 局部性不是免费的，应把 BJOC-Null 正式描述为
  **全局组成模型**并停止局部解释尝试；若平价与忠实度同时成立 → 解释接口首次可用。
* 预算量级：R3 单次训练 ~0.2 GPU-h（A100），两 seed × 两个臂 ~0.5–1 GPU-h，远低于任何架构搜索。

**不购买/不启动**：新的 junction/ownership 特征族、距离特征、attention/MPNN、为新 MAE 刷新而做的
任何训练。理由是旧有 GLRC/DLP/junction-sufficiency 负结果 + 本轮“局部性非瓶颈”的间接证据。

**如果解释性不是优先项**：以本报告的全局通路敏感性方法（4 块零/自然替换 + OOD 诊断 + 剂量-响应）
作为对外的诚实解释口径，并明确声明“结构级解释不可用”。

---

## 8. 复现与 artifacts

```
uv run python tracks/structure-contribution/code/bjoc_structure_explanation_audit_v1.py prepare
uv run python tracks/structure-contribution/code/bjoc_structure_explanation_audit_v1.py phase-a
uv run python tracks/structure-contribution/code/bjoc_structure_explanation_audit_v1.py phase-b1
uv run python tracks/structure-contribution/code/bjoc_structure_explanation_audit_v1.py phase-b2
uv run python tracks/structure-contribution/code/bjoc_structure_explanation_audit_v1.py phase-b3
uv run python tracks/structure-contribution/code/bjoc_structure_explanation_audit_v1.py phase-c
uv run python tracks/structure-contribution/code/bjoc_structure_explanation_audit_v1.py finalize
uv run pytest -q tracks/structure-contribution/tests/test_bjoc_structure_explanation_audit_v1.py   # 9 passed
```

| artifact | 内容 |
|---|---|
| `PROTOCOL.md` | 运行前冻结的协议 |
| `sample_manifest.json` | 哈希规则、诊断/参照行与基本统计 |
| `source_manifest.json` | checkpoint/缓存/code SHA-256、环境、行索引哈希、历史哈希核对 |
| `phase_a_dependency.json` | 逐层依赖表、A1/A2/A3 |
| `phase_b_baseline.json` | B1 一致性 |
| `phase_b_pathways.json` / `phase_b_deltas.npz` | B2 通路敏感性、OOD、冗余（含 per-molecule Δ） |
| `phase_b_nonlocal.json` / `phase_b_nonlocal_arrays.npz` | B3 远端传播 |
| `phase_c_interface.json` | C 接口探针 |
| `metrics.json` | 汇总 + 决策标签 + 各文件 SHA-256 |
| `code/bjoc_structure_explanation_audit_v1.py` | 审计驱动 |
| `tests/test_bjoc_structure_explanation_audit_v1.py` | 9 个针对性正确性测试（合成分子；checkpoint 测试本地可跳） |

已知限制（必须与结论一起引用）：

* 干预后的 `reader_in` 可能落在训练分布外；B2 同时给出零/自然两种干预与 OOD 度量，但 Reader 非线性
  使“敏感性”不能升级为“独立贡献”。
* B3 的 `w_j` 扰动是编码后的内部反事实，不对应任何分子编辑；它证明的是**计算依赖**，不是化学因果。
* Phase C 只做了 32 分子、k=8；它足以否定“连通性带来独特效应”，但不是对所有子图/规模的穷举。
* 本审计只用两个 R3 seed；`z_bond` 的 seed0/seed1 差异（如 `sum_atom_squared` 置零 0.94 vs 0.69）
  提示部分量级存在 seed 依赖，定性排序稳定。
* 没有做任何真实分子编辑、没有重训练、没有可学习 Motif 分割、没有新监督任务。

## 9. 明确未得出的结论

* 不声称 ownership/局部信息对预测有用或无用的**因果**结论；这需要重训练对照。
* 不声称“遮蔽连通子结构”等于真实删键/删官能团。
* 不把通路敏感性称为独立贡献，不把全局块的线性可恢复性直接外推为 Reader 一定冗余使用它。
* 不把本审计与官方 valid/test 的任何 MAE/RMSE 联系；官方 test 的 0.069822 仅作为冻结模型来源的
  历史背景，未在任何选择中出现。
