# BOSI Partner-Content Necessity v1 — 预注册协议

轮次 `zinc_bosi_partner_content_v1`。配置 `configs/bosi_p_v1.json`（内含 `DESIGN.md`
的 sha256 `58429233…`）。**本文件在第一次 GPU 作业之前冻结。** 之后只允许追加
"偏差"记录，不允许修改判决规则、评估集、预算或对照。

设计基线：`calendar0917/research@a269d14`，即 BOSI v0 报告（Locked seed-mean
R 0.081714 / B 0.081970 / DCC-L 0.083498，判决 `NO_GO` 针对关系分配）。

---

## 1. 要回答的问题

v0 的 R/B 只证明"伙伴矩**怎么分配**不重要"。它没有测过更前面的一步：

> B 之所以能和 DCC-L 打平，是否真的依赖**从其他邻键获得的显式内容矩**？
> 目标键自身特征 + 原子统计 + 关系计数 + 读出网络是否已经足够？

这是**必要性**实验：不搜索架构，不优化指标，只问"去掉这条通道会不会变差、是否可重复"。

## 2. 复用（只读，逐文件 sha256 校验）

| 对象 | 来源 | 用途 |
|---|---|---|
| BOSI-B s0/s1 冻结权重 + Locked 预测 | `zinc_bosi_architecture_v0/{runs,analysis}` | 主对比的另一半 |
| BOSI-R s0/s1 同上 | 同上 | 背景 |
| DCC-L Fit8k s0/s1 冻结预测 | `zinc_dccl_bottleneck_v1/runs/fit8k_s{0,1}/predictions_best_monitor.pt` | 历史基线 |
| 关系缓存 | `zinc_bosi_architecture_v0/cache/bosi_rel_train.pt` | 内容哈希 `c698a2f3…`，**不重建** |
| official-train 缓存 / 划分 | `zinc_arch_discovery_v1/cache/ad_train.pt`、`bjoc_predictive_bottleneck_v1/split_a.json` | 数据与划分 |

**不重训 B / R / DCC-L。所有既有模型不再对 Locked 前向一次**：复用的是 v0 已记录的
预测文件，只做哈希校验。本轮唯一的新 Locked 前向来自 P。

## 3. 唯一改动

```
B:  M[r,k] = n[r] / max(deg-1,1) * Σ_{f≠e, f∋v} T_k(x[v,f])
P:  M[r,k] = n[r] * T_k(x[v,e])
```

`T_k∈{T1,T2,T3}`（Chebyshev），`r∈{T,Q,R,N}`，`n` 为组内伙伴数。其余**逐字不变**：
`a_v`、`b_e`、`x[v,e]`、`n`、`closure`、384 维 M 接口、side/compose/reader、
473,889 参数、Fit-only masked-SSL 的 atom/bond 初始化、全部 recipe。
没有宽度/归一化/优化器/学习率/矩阶变化。

这是**替换式 null**（不是消融）：矩块接口必须被填满，把伙伴内容换成目标键自身内容，
从而只隔离"其他邻键 learned 状态是否进入 M"这一条通道。P 仍保留 `atom_feat` 108 维
（含半径统计）与邻居原子 `a_u`，所以**不得**表述为"删除了全部邻域信息"。

## 4. CPU 门控（已执行，`analysis/cpu_checks.json`，10/10 → `CPU_GO`）

| 门 | 判据 | 实测 |
|---|---|---|
| C1 参数/初始 | 逐张量同形、同 state hash、等于历史 B/R 同 seed 的初始 hash、473,889 | 全等；seed0 `ceeb598f…`、seed1 `c93ea85b…`；张量差 0.0 |
| C2 空组 | `n[r]=0 ⇒ M[r]=0`（精确） | 9,243 个空槽，违规 0 |
| C3 伙伴内容 | 改动 `partner_half` 后 P 位同、B 移动 | 内容位移 1.57–1.68；P ΔM=0.0 / Δy=0.0；B ΔM=3.20/3.28、Δy=2.1e-3/2.6e-3 |
| C3b 关系置换 | 直方图保持置换后 P 不变 | Δn=ΔM=Δy=0.0 |
| C4 自身状态 | 扰动目标自身 `x` → 仅该行 M 变，且符合闭式 `n[r]·ΔT_k` | 仅 1 行变化，闭式误差 0.0 |
| C5 尺度 | 有限、非零、RMS 比在 [1/8, 8] | RMS B 0.41645 / P 0.41946 → **比 1.0072** |
| C6 对称/批次 | relabel/reverse/edge order/batch order/recombine < 1e-5 | R 3.6e-7、B 3.0e-7、P 1.8e-7 |
| C7 梯度 | P 无可失活参数，六模块梯度全非零 | 0 个 dead tensor，5 步 loss 1.2929→0.9827 |
| C8 标签隔离 | 训练批不含 Monitor/Locked 行；Locked 标签投毒无影响 | 512 行训练全在 Fit8k；eval 1000 行全在 Monitor；投毒后 loss 与 Monitor 预测差 **0.0** |
| C9 接口/预算 | M=[·,384]、n=[·,4]、SIDE_IN 617、READER 417、arms={R,B,P} | 全部符合 |

尺度可比（1.007）是本实验可信的前提：P 与 B 的矩块输入幅度几乎相同，
**不存在"某一臂输入更小所以更难学"的混淆，也没有做任何事后缩放**。

## 5. GPU 计划

* 只训练 `P_s0`、`P_s1`，**串行**在单卡上（v0 实测两卡并行的 CPU 争用使单 run 成本
  上升约 20%，串行更省且更可复现）。
* recipe 与 v0 逐项相同：Fit8k、batch 128、47,400 updates（753×63）、
  AdamW 1e-3→1e-5 per-step cosine、warmup 500、wd 1e-5、clip 5、L1 raw-y、
  `Generator(seed+101)` 的 batch 顺序、Monitor 严格最小选点。**不缩短 updates。**
* 预算 **≤1.5 A100·h**（含 smoke 与失败作业）。若实测将超限 → 停止并记录，不改配方。
* 初始状态与同 seed 历史 B 的 `initial_state_sha256` 必须相同（C1 已验）。
* 训练期禁止读 Locked；Monitor 只用于选 checkpoint。

## 6. 评分流程（顺序不可逆）

1. `p_freeze.py`：冻结 P 的 checkpoint / Monitor / **Fit** 读数与哈希；校验复用文件的
   sha256；校验 P 与历史 B 的初始状态、首 batch 行号一致。若 `locked_read.json` 已存在则拒绝。
2. `p_locked.py`：一次性读取 Locked，给 P 的 best/final 打分；同一次读取内做冻结权重
   干预（换臂 B/R、M 置零）。二次调用只校验不重算。
3. `p_analyze.py`：只用冻结文件产出全部表与判决。

Locked 是**历史已暴露**的内部划分（DCC-L 4 seeds、ESOC、v0 都读过），本轮的读取
不是新的盲测确认，不得这样表述。

## 7. 指标与判决（预注册）

`Δ = MAE(P) − MAE(B)`，**正数 = 伙伴内容有益**。n=1,000 Locked 分子。

指标：overall、ordinary（`y>−3`）、tail（`y≤−3`）、去掉配对最差 10 个分子、
low_support、any_unseen、all_supported、Monitor、Fit、final checkpoint。
配对 bootstrap 10,000 次（固定分子重采样）**只描述分子抽样，不包含训练随机性**。

| 标签 | 条件 |
|---|---|
| `PARTNER_CONTENT_SUPPORTED` | 两 seed `Δ>0`；seed-mean `Δ≥0.005`；ordinary 与 top-10-removed 方向一致。若 any-unseen 未受益，报告必须写明"内容有预测价值但未解决未见结构泛化" |
| `PARTNER_CONTENT_NOT_ESTABLISHED` | `Δ` 小、跨 seed 不稳定、或 95% CI 跨零。**不得**宣称与 B 等价；也不得再以"伙伴矩是关键贡献"为由追加模型 |
| `SIMPLER_BODY_PROMISING` | 两 seed `Δ<0` 且 seed-mean `|Δ|≥0.005`，且非尾部主导（去掉 top-10 后仍为负） |

不根据 Locked 结果改统计类型/宽度/激活/超参；不追加第三个 seed；不扩样本、不改划分。

## 8. 本轮之后

* 只有 `PARTNER_CONTENT_SUPPORTED` 且 any-unseen 未退化，才允许写
  `NEXT_SSL_HYPOTHESIS.md`（**只写设计，不启动任何 SSL GPU 作业**）。
* 任何情况下都不启动 attention / junction token / 关系分配的新实验。

## 9. 禁止清单

重训 B/R/DCC-L；official valid/test 访问；新随机划分；用 Monitor/Locked 标签选架构、
超参、epoch、seed（Monitor 仅按上表选 checkpoint）；第三个 seed；learned junction；
递归 MP；Transformer/attention；新查表；外部预训练；新标签分量；蒸馏；ensemble；
加宽网络、加矩阶、改关系类型、改归一化、或对某一臂事后缩放；启动 SSL 作业。
