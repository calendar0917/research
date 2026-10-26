# BOSI Partner-Content Necessity v1 — 设计

轮次 `zinc_bosi_partner_content_v1`。前置：`zinc_bosi_architecture_v0`（`REPORT.md`，
Locked seed-mean R 0.08171 / B 0.08197 / DCC-L 0.08350，判决 `NO_GO` 针对关系分配）。

**本文件在设计层面冻结一件事：新增且只新增一个 arm，且它只改一个公式。**

---

## 1. 未被 v0 识别的问题

v0 的 R/B 对比回答的是"伙伴矩**怎么分配**到关系组"。它没有回答更前面的一步：

> BOSI-B 之所以能和 DCC-L 打平，**是否真的依赖"从其他邻键获得的显式内容矩"**？
> 还是说：目标键自身的特征 + 原子统计 + 关系计数 + 读出网络已经足够？

这不是性能问题，是**必要性**问题。v0 的 R/B 差异只有 −0.0003（seed-mean），
说明"分配"不重要；但**"伙伴内容本身"是否重要，v0 从未测过**。若伙伴内容其实
不承载信息，那么整个统计主干（R、B、以及 v0 的 `ATTENTION_NEXT.md` 计划）都建立在
一个未被验证的假设上；若它确实承载信息，那么 v0 的 NULL 结论只否定"分配"，
不否定"内容"，主干的定位就要跟着改写。

## 2. 唯一的新 arm：BOSI-P（Partner-content Null）

BOSI-B 现在的矩块是

```
M_B[v,e,r,k] = n[v,e,r] / max(deg(v)-1, 1) * Σ_{f≠e, f∋v} T_k(x[v,f])      (v0 协议修正 A1)
```

P 臂**只**把参与求和的项换掉：

```
M_P[v,e,r,k] = n[v,e,r] * T_k(x[v,e])
```

`T_k` = Chebyshev `[T1,T2,T3]`，`k∈{1,2,3}`，`r∈{T,Q,R,N}`，`n[v,e,r]` = 该关系组内
伙伴半键数。**逐字保留**：`a_v`、`b_e`、`x[v,e]` 的生产路径；四类关系计数；`closure`
原始统计；384 维矩接口；side/compose/reader 的形状与参数；473,889 个参数及其梯度通路；
Fit-only masked-SSL 的 atom/bond 初始化；以及全部 recipe。

### 2.1 这是"替换"，不是"消融"

矩块是一个**有形状的接口**，必须被填满。把伙伴内容删掉有三个选择：

| 方案 | 问题 |
|---|---|
| `M ≡ 0` | 不是"没有伙伴内容"，而是"side MLP 的一个 384 维输入突然恒零"——分布外输入，且与"计数被清零"混淆 |
| 删掉 384 维、`SIDE_IN` 617→233 | 参数变了（−49k），不再是等参数对照，也违反"不改宽度" |
| **`M[r] = n[r]·T_k(x[v,e])`** | 保留接口形状、计数语义、`n=0 ⇒ 0` 的稀疏行为，但组内内容只来自**当前半键**；参数量逐张量不变 |

因此 P 是"把伙伴内容换成自身内容"的**替换式 null**。它的对照含义是精确的：

* **B 与 P 的差别 = 唯一一条通道**：其他邻键 `f≠e` 的 learned half-edge 状态 `x[v,f]`
  是否进入矩块。
* **B 与 P 的共同部分**：目标键自身（`a_v, a_u, b_e, x[v,e]`）、关系计数、closure、
  degree、所有编码器、side/compose/reader、读出。

### 2.2 必须公开的两点，不得事后修改

1. **P 仍保留原始原子特征，包括已预计算的半径统计**（`atom_feat` 108 维）。因此本实验
   只删除"其他邻键的 learned 状态"这一条通道，**不能**声称删除了全部邻域信息。
   严格表述是："伙伴内容不进入矩块"，不是"模型看不见邻域"。
2. **P 不是 B 的严格子集。** B 的广播是均场收缩，P 的 `n[r]·x_self` 是一个
   `计数 ⊗ 自身状态` 的乘性交互，B 没有这一项。所以对比测的是
   "等参数、等接口下，伙伴内容 vs 自身内容"，不是"有统计 vs 无统计"。
3. **量纲可比性**是设计的一部分：`Σ_r M_B = Σ_f T(x_f)`（deg≥2），
   `Σ_r M_P = (deg-1)·T(x_self)`。两者都随 `deg-1` 增长，同一个 `T(·)` 分布，
   所以输入尺度天然接近。CPU 门控会**实测**两侧 RMS 比并原样报告，
   **不允许**事后缩放或调激活来救某一臂。

## 3. 复用与不重训

| 对象 | 用途 | 处理 |
|---|---|---|
| BOSI-B s0/s1 冻结权重与 Locked 预测 | 主对比的另一半 | **只读复用**，用 commit 里的 `locked_read.json` 逐文件 sha256 校验 |
| BOSI-R s0/s1 | 背景（分配无关） | 同上，只读 |
| DCC-L Fit8k s0/s1 冻结预测 | 背景（历史基线） | 只读，校验 `locked_rows` 与文件 sha |
| BOSI-B/R 的 Monitor、Fit 读数 | 报告中的 checkpoint 横评 | B/R 的 **Fit** 读数在冻结步用冻结权重重算（不读 Locked）；Monitor 用 v0 `freeze.json` 记录值并复算校验 |

**不重训任何已有模型。** 不重建关系缓存（v0 的 `bosi_rel_train.pt` 只读复用，
按内容哈希 `c698a2f3…` 校验）。不访问 official valid/test。

## 4. 训练

只有 `P_s0`、`P_s1` 两条。recipe 与 v0/ DCC-L 逐项相同：Fit8k、batch 128、
47,400 updates（753×63）、AdamW 1e-3→1e-5 per-step cosine、warmup 500、wd 1e-5、
clip 5、L1 raw-y、`Generator(seed+101)` 的 batch 顺序、Monitor 严格最小选点。
初始状态与同 seed 的历史 B 必须**逐张量相同**（arm 不消耗 RNG），门控里直接与
`v0/analysis/freeze.json` 记录的 `initial_state_sha256` 对照。

预算：**≤1.5 A100·h**（含 smoke、含失败作业）。若实测超限则停止，**不缩短 updates**。

## 5. 判决（预注册，`Δ = MAE(P) − MAE(B)`，正数 = 伙伴内容有益）

| 标签 | 条件 |
|---|---|
| `PARTNER_CONTENT_SUPPORTED` | 两个 seed 上 `Δ > 0`，seed-mean `Δ ≥ 0.005`，ordinary 与 top-10-removed 方向一致；若 any-unseen 未受益，必须追加"内容有预测价值但未解决未见结构泛化" |
| `PARTNER_CONTENT_NOT_ESTABLISHED` | `Δ` 小、跨 seed 不稳定、或 95% CI 跨零。不得宣称等价 |
| `SIMPLER_BODY_PROMISING` | 两个 seed 上 `Δ < 0` 且 seed-mean `|Δ| ≥ 0.005`，且不由尾部少数分子主导 |

不根据 Locked 结果改统计类型/宽度/激活/超参；不追加第三个 seed；不因结果好就自动
启动 SSL 或 attention。

## 6. 明确不做

架构搜索；新关系类型；新矩阶；新宽度；junction token；attention；重训 B/R/DCC-L；
official valid/test；第三个 seed；在 Locked 上选任何东西。
