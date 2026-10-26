# DESIGN — BOSI-JM：联合矩式的邻键统计（性能优先的架构开发）

**日期：** 2026-10-28
**轮次：** `tracks/structural-sufficiency/zinc_bosi_joint_moment_v1/`
**前身：** `zinc_bosi_architecture_v0`（BOSI-B/R，Fit8k 内部划分，473,889 参数）
**数据：** 官方 PyG ZINC `subset=True`，train 10,000 / valid 1,000。**test 不构造、不加载、不预测、不评分。**

---

## 0. 为什么是这一步

前一阶段（`zinc_bosi_architecture_v0`）在 Fit8k/Monitor/Locked 上得到两条互不冲突的事实：

1. **主干本身与 DCC-L 打平、点估计略优**（Locked seed-mean：BOSI-B 0.081970 vs DCC-L
   0.083498），说明"物理键为对象 + 原子为锚点 + 一次静态统计 + 无 learned junction"
   这个主干值得迁移到官方 10k 协议上直接测一次绝对水平。
2. **关系分配机制（R vs B）关闭**：两 seed 符号相反、幅度 −0.00026。
   `zinc_bosi_partner_content_v1` 进一步定位：拿掉伙伴*内容*使 Locked 变差
   **+0.012390**（两 seed 同向，CI 不跨零，非尾部）；而伙伴内容*怎么分组*价值为零。

第三条事实来自 `AUDIT.md` 与 BOSI 自己的解释：BOSI 的 384 维统计块是
**每通道一阶/二阶/三阶幂和**，它对"伙伴的多维坐标"只保留**逐通道**信息，
丢失跨通道配对。BOSI v0 的 CPU 反例已经写明：伙伴集 `[(0,0),(1,1)]` 与
`[(0,1),(1,0)]` 的**每个坐标的所有阶矩完全相同**，但 `(a+b)` 的二阶矩是
4 vs 2。也就是说，BOSI-B 的接口在信息上比"伙伴集合本身"更弱，
而 partner-content 实验说明伙伴内容正是值钱的那一块。

所以本轮的问题不是"再加一种统计"，而是一个**单变量**、可实现、可证伪的机制问题：

> 把 384 维统计块里的**跨通道配对**显式做出来——对同一个伙伴 `f` 的
> 学到的两个 12 维投影先做外积、再对所有伙伴求和（`J`）——能否在官方 10k
> 协议上把 BOSI-B 的 valid MAE 压下去，并接近 DCC-L？

同时预注册一个**严格等参数的机制对照** `CM`：把同样的两个投影先求和、再做外积。
`J` 与 `C` 的唯一差别就是"先配对再求和"还是"先求和再配对"。这不是两个架构，
而是同一个架构的两种统计收缩，用来把性能差异归因到机制本身。

---

## 1. 三个 arm

### 1.1 BOSI-B（基线，原样）

`code/` 直接 `import bosi_models.BOSI`，用 `arm="B"`，**模块与 `forward` 完全不改**：

```
a_v   = atom_encoder(atom_feat_v)                              108 -> 128 -> 96
b_e   = bond_encoder([au+av,|au-av|,au*av,bond_feat_e])        295 -> 160 -> 96
x[v,e]= tanh(half_encoder([a_v,a_u,b_e,bond_feat_e]))          295 -> 128 -> 32
M_B[v,e,r,k] = n[v,e,r]/max(deg-1,1) * sum_{f!=e} T_k(x[v,f])  (4,3,32) = 384
n[v,e,r]                                                       (4,)
s[v,e]= side([a_v,b_e,x,M_B,n,closure,log1p(deg)])             617 -> 128 -> 96
c[e]  = compose([b_e, s_u+s_v, |s_u-s_v|, s_u*s_v, bond_feat])  391 -> 160 -> 96
z[G]  = [sum c, sum c^2, sum a, sum a^2, global33]              417
yhat  = reader(z)                                              417 -> 256 -> 256 -> 1
```

参数 **473,889**。半键、relation 构造（T>Q>R>N）、Side、Compose、Reader、
关系计数、closure、global33 全部逐字节复用。

### 1.2 BOSI-JM（本轮的候选）

**唯一改变：送入 Side 的 384 维 `M`。** 其余每一个模块、每一层宽度、
每一个参数、reader 的形状与初始化全部保留。

对半键 `(v,e)`，记其他邻键 `f ∈ P(v,e)`，`n = |P(v,e)| = deg(v)-1`：

```
A    = sum_f [T1(x_f), T2(x_f), T3(x_f)]                        96      (k-major)
z_f  = LayerNorm(x_f)                                          no affine, 32
u_jf = tanh(U_j z_f)                                            12
w_jf = tanh(V_j z_f)                                            12
J_j  = sum_f outer(u_jf, w_jf)                                  12 x 12
M_JM = concat(A, vec(J_1), vec(J_2))                            96+144+144 = 384
```

* `T1,T2,T3` 与 BOSI 原版完全相同（Chebyshev，`chebyshev(x)`），
  `A` 就是原版四个关系组矩的**未分组总和**（`A = Σ_r M_R[r]`）。
* `U_j, V_j`：`Linear(32, 12, bias=False)`，四个，共 `4 × 384 = 1,536` 参数。
  **固定声明的初始化**：PyTorch `nn.Linear` 默认 Kaiming-uniform（`a=√5`，无 bias），
  在六个公共 MLP **之后**按 `U_1, V_1, U_2, V_2` 顺序消耗同一个全局 RNG 流。
  因此同 seed 下公共 BOSI 张量与 B 臂逐位相同，JM 与 CM 的完整 state 逐位相同。
* `LayerNorm` 用 `F.layer_norm(x, (32,))`：**无 affine**，`eps=1e-5`（PyTorch 默认），
  偏置方差。它在这里是**固定的归一化**，不是可训练模块。
* `n=0`：`A` 与两个 `J` 严格全零（`index_add_` 空索引）。
  `n=1`：`J_j = outer(u,w)`，正常输出。
* 关系计数 `n[v,e,r]`、`closure[4]`、`log1p(deg)` 全部保留在 Side 输入里，
  所以 Side 的输入维度仍是 `96+96+32+384+4+4+1 = 617`，输出仍是 96。

参数：`473,889 + 1,536 = 475,425`（cap 500,000，headroom 24,575）。

**不做**：learned junction token、迭代状态更新、attention、额外监督任务、
global correction branch、reader 加宽/加深、查表、递归消息传递。
所有结构统计只读原始分子图（bond_index / bond_type）与当前模型编码状态 `x`。

### 1.3 BOSI-CM（严格等参数的机制对照）

共享 JM 的**全部**参数、初始化、数据、配方。只把 `J_j` 换成

```
C_j = outer(sum_f u_jf, sum_f w_jf) / max(n, 1)                 12 x 12
```

`n=0` 时 `sum = 0`，除法用 `max(n,1)=1`，结果严格为零。
参数与 JM **完全相等**（475,425，逐张量同形状同初值）。

**代数关系（本轮的中心恒等式）：**

```
J_j - C_j = sum_f (u_jf - ubar_j)(w_jf - wbar_j)^T ,   ubar_j = (1/n) sum_f u_jf
```

`J` 保留"同一伙伴内部两个投影的配对"，`C` 只保留"两个和的边际配对"。
`n<=1` 时两者**恒等**（`u-ubar=0`）。因此这是一条只作用于 `n>=2` 半键的精确干预：
官方 train 10k 上 `n>=2` 的半键占 **44.70%**（`n=0/1/2/3` 分别
8.25% / 47.05% / 42.21% / 2.49%），远高于 v0 里 R/B 分配只在 19.50% 上有定义。

### 1.4 为什么这不是"再加容量"

JM 与 CM 的参数个数、参数形状、逐张量初值完全相同；JM 相对 B 只多 1,536 个
参数（0.32%），且这 1,536 个参数在 CM 里以同样的方式被训练、被使用。
两者对 Side 的输入维度同为 384。所以：

* **JM − B** 衡量的是"把跨通道配对显式做出来"是否值 1,536 个参数；
* **JM − CM** 衡量的是"先配对后求和"相对"先求和后配对"是否值 0 个参数。

如果 JM 胜过 CM，机制被支持；如果 JM 输给 CM，则配对本身有害，架构关闭；
如果两者相同而都胜过 B，收益来自 `A + 12×12 双线性池化` 这一族接口的容量/归纳偏置，
而不是配对——这一点也必须在报告里写清楚。

---

## 2. 与历史结果的边界

| 参照 | 协议 | valid MAE | 说明 |
|---|---|---:|---|
| DCC-L 4 seed 均值 | 官方 10k | **0.075990** | `zinc_performance_search_v1`，历史 |
| DCC-L seed0 | 官方 10k | **0.078537** | `zinc_dual_channel_composer_v1`（B-R0） |
| P3B seed0 | 官方 10k | **0.076224** | `zinc_recipe_trunk_v2`（xBssl_P3N_s0） |
| BOSI-B（v0，seed-mean，Locked） | Fit8k 内部 | 0.081970 | 不同协议，**不可与上表混排** |

这三条**只作为历史参照**，不当作本轮的匹配对照，也不重训。本轮 B / JM / CM 在
同一 commit、同一配方、同一 seed 下并行训练，唯一的匹配比较是**它们之间**。

指标全部是 raw-y MAE；不做 ensemble、seed 平均、checkpoint soup、TTA。
official valid 只用于训练期间的模型选择与开发评估；它不参与特征、归一化或预训练。
official test **本轮完全不碰**。

---

## 3. 预注册：本轮要回答什么、什么情况下停

**主问题（P1）：** `JM` 的 best-valid 是否优于同 seed 的 `B`？

买 seed 1 的门槛（在 seed0 结果出来后，**先冻结 checkpoint 与 valid 预测**再比较）：

```
A. JM best_valid <= 0.075
或
B. (JM best_valid - B best_valid) <= -0.003  且  JM best_valid <= 0.078
```

同时要求 ordinary（`y > -3`）组没有明显恶化、W20 不出现反方向的大幅退化。
不满足就停在 seed0，不买 seed1、不扫超参、不改 rank/宽度/LR/wd/horizon。

**机制问题（P2）：** 只有在 JM 表现出可信的性能优势后，才以新作业跑
`BOSI-CM seed0`（必要时 `seed1`），用 `JM − CM` 判断"联合配对"本身是否被支持。
未触发的作业不得自动启动。

**预算：** 默认 5 A100 GPU·h（含 smoke、计时、失败作业、后续有条件运行）。
超过即停，不静默放宽、不缩短 47,400 updates。

**计算需求核对：** BOSI v0 在 Fit8k（63 steps/epoch）上实测 1.68–2.72 s/epoch；
本轮 79 steps/epoch（1.25×）→ 预计 0.4–0.6 h/run。seed0 两条 ≈ 1.0 h，
seed1 两条 ≈ 1.0 h，CM 两条 ≈ 1.0 h，加 smoke/计时 ≈ 0.1 h，总计 ≈ 3.1 h < 5 h。
若实测超出，按预算规则停止而不缩短日程。

---

## 4. 风险与事前判断

1. **BOSI 主干的有效信号集中在 side 通道**（v0：side 置零 ⇒ 0.31，换臂只值 0.011–0.031）。
   JM 改的正是这个通道的输入，方向正确。
2. **v0 的教训是"接口形状可以被忽略"**：R/B 的统计分布差异真实存在（相对差 13.3%），
   但训练后不产生精度差异。JM 的输入差异比 R/B 大得多（`n>=2` 上 97.0% 的半键
   在初始化时就有 >1e-6 的差异，JM/CM 联合块的最大相对差 0.427），但**差异大不等于有用**。
   这条风险必须由 JM−CM 而不是由 JM−B 来回答。
3. **`X^T Y` 型 12×12 池化不是新数学**：它是二阶池化 / 双线性池化 / 低秩
   covariance 的既有形式，也与"二阶矩"文献重叠。本轮的贡献只在
   **受控的机制隔离**与**同协议下的实测**，不声称原创算法。
4. **`n<=1` 时 JM≡CM**：44.70% 的 `n>=2` 覆盖率是本设计的上限来源，
   比 v0 的 19.50% 高，但仍不到一半。如果实测无差异，主因很可能是这个覆盖率，
   届时不得用放大系数、加 rank、加阶数事后救援。

---

## 5. 交付

`code/`（`bjm_common.py` / `bjm_models.py` / `bjm_data` 复用 `bosi_data` /
`bjm_prep.py` / `bjm_checks.py` / `bjm_train.py` / `bjm_analyze.py`）、
`configs/bjm_v1.json`、`PROTOCOL.md`、`analysis/cpu_checks.json`、
`analysis/summary.json`、`REPORT.md`，以及 gitignore 的 `runs/`、`cache/` 下的
checkpoint / 曲线 / manifest / hash。历史轮次的文档与结果不改写。
