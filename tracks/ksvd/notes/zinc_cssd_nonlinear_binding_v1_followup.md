# Followup — `zinc_cssd_nonlinear_binding_v1` 报告解释补充（2026-10-07）

**性质**：可追溯的解释补充，不覆盖 `results/zinc_cssd_nonlinear_binding_v1/REPORT.md`，
不改任何不可变 run 记录 / 原始数值 / 预测文件。原始报告 commit `97b414bddaf29f3e2f2691ea9edd820fcca43356`。
本补充只读旧产物 + 保存的预测文件；没有启动任何新训练，没有重评分历史 held-out。
（本机仍未安装 research-judgment skill；按同原则在本地完成判断，未声称调用外部审查工具。）

## 1. G_N / G_E 的符号约定（修正文字，不改正数值）

代码定义（`zinc_cssd_nonlinear_binding_v1._contrast_stats`）：
`G_N = 0.5*((e_C00 − e_C10) + (e_C01 − e_C11))`，即 **G = sep MAE − joint MAE，
正值 = joint 更好**。旧报告 §2 表头写"正 = joint 更差"，与代码定义**相反**，
属文字笔误。正确读法：G_E = −0.00678（CI95 [−0.01035, −0.00320]，不含 0）表示
**E joint 分支更差**；G_N = −0.00230（CI 跨 0）方向一致但不确定。原始数值
（select_eval.json）无误，只有报告表头的符号描述需要按本条修正。

## 2. joint 臂只有 select seed 0；A/C00 在 confirm 上 seed 方向翻转

- joint 臂（C10/C01/C11）只在 select 上各跑过 seed 0（7 个 body run =
  5×seed0 + A/C00 seed1）；**joint 臂没有 seed 1 数据**，其"负结果"是单 seed
  select + 机制干预层面的，不是跨 seed 稳定的。
- A vs C00：select（seed0）C00 优 +0.0058；confirm 上 seed0 C00 优 −0.0058、
  seed1 A 反超 +0.0040（confirm_eval.json）。合并 ~+0.0009，远小于单 seed
  噪声（±0.005）。
- 因此正确的口径是：**"本候选族（池化前非线性 joint 绑定）在本预算内的证据
  不支持继续购买"**；不能写成"结构—语义融合问题被证伪"（joint 臂从未在
  confirm / seed1 检验），也不能写成"C00 是已确认的安全替换"（seed 方向翻转）。
  C00 的 sep 机制精确打乱不变性仍然成立（机制性质，非性能优势）。

## 3. 干预后的 ΔMAE ≠ 预测变化量；两者已按保存的 base/shuffle 预测分开核对

`interventions.json` 里的 `delta` 是 **select y_raw MAE 的变化**，不是 |Δpred|。
按保存的 `select_predictions{,_n,_e}.npz` 逐行核对（seed 0 soup，select 999 行）：

| 臂 | 干预 | MAE base→shuf（ΔMAE） | \|Δpred\| mean / med / p95 / max |
|---|---|---|---|
| C10 | n-shuffle | 0.147699→0.147721（+2.2e-5） | 6.8e-4 / 5.2e-4 / 1.9e-3 / 5.6e-3 |
| C10 | e-shuffle | 0.147699→0.147699（+1.3e-9） | 0 / 0 / 0 / 1e-6（精确不变） |
| C01 | n-shuffle | 0.152176→0.152176（−4.6e-9） | 0 / 0 / 0 / 1e-6（精确不变） |
| C01 | e-shuffle | 0.152176→0.152139（−3.7e-5） | 7.4e-4 / 5.0e-4 / 2.4e-3 / 6.2e-3 |
| C11 | n-shuffle | 0.153132→0.153048（−8.4e-5） | 8.1e-4 / 5.8e-4 / 2.4e-3 / 5.5e-3 |
| C11 | e-shuffle | 0.153132→0.153049（−8.3e-5） | 1.8e-3 / 1.1e-3 / 5.6e-3 / 1.8e-2 |

读法：joint 分支被配对打乱后**预测确实会动**（均值 7e-4–1.8e-3），但
ΔMAE 是这些变化的**有符号聚合**（L1 下可正可负，C01/C11 打乱后 MAE 反而略降）；
旧报告的"joint 配对贡献近置换不变"应精确表述为：**预测响应量级 ~1e-3 且
对 MAE 的净效应 ≤1e-4、方向不定**——依赖存在，有利作用不存在。C10-e/C01-n
精确不变是 sep 机制闭环的直接验证（对应分支对配对不可见）。

## 4. pairing_changed_fraction 是批次内配对 occurrence 行的改变比例

`intervention_stats` 的 `rows` 是**批内 occurrence 行**（node-occurrence 行或
bond-occurrence 行），`changed_fraction` = 这些行中 shuffle 后原子类型/化学
tuple 真正改变的行占比（~0.24/0.28）。它**不是**"分子被改变的比例"，也不是
图级比例；引用它时应写"配对 occurrence 行改变比例"。

## 5. 已排除与未排除的竞争解释；branch seed 口径

- 已排除（select_eval.json training_health + checks.json）：训练病理（分支
  失活 / 梯度不可达 / 非有限参数 / 冻结基底漂移）、C↔C 容量差（36 参数）、
  C00 精确不变性排除实现泄漏。
- **未排除**：优化景观/功能容量的其他解释只被部分触及；"梯度可达、参数
  接近"只能排除明显失活，不能宣称所有优化与功能容量解释均已排除。
- branch 模块（node_branch/edge_branch）由 `branch_seed_for(arm, seed)` =
  `20261007 + 100*step(arm) + seed` 的**私有 generator** 初始化，是
  **arm-dependent** 的：同 seed 下不同臂的私有分支初始化**不同**（这是设计，
  使臂间只比较共享骨架 + 各自私有分支）。旧报告"五臂共享模块 state 哈希
  完全一致"只覆盖共享骨架，不能写成"所有未改动私有分支都初始化相同"。
  本轮（basis reuse）的下游骨架两臂完全同构，将改为逐 tensor 全 state 对齐
  （见本轮 preregistration）。

## 6. A 臂的融合接口口径

A = 上轮恢复**原槽乘积绑定**（W_A_S/W_A_C/W_E_S/W_E_C + slot encoders）于
canonical Full 446-D 融合布局（`sc.FULL`，fusion_in 446）的强度参照，
其融合接口由 M_COMP 的 native110 deploy 布局**扩回 446**。因此 A 不是
"最强 native M_COMP 原样复现"，而是"CSSD 路径上的 Full446 参照"。
本轮（zinc_cssd_basis_reuse_v1）把 A 的这个架构**原样冻结**为唯一下游骨架，
不再改接口、不再挑 seed。

## 7. select 与 confirm 的分布差异（本轮 Q 定点核对的附带发现，详见
`zinc_cssd_nonlinear_binding_v1_q_spotcheck.md`）

select999 含 2 个深长环尾部分子（gid=3775、gid=1424，各携带 ~13.6–14.6%
的臂内 select 总绝对误差，合计 ~28%），confirm1000 没有任何 |err|>5 的行
（max 2.13）。select MAE（0.144–0.153）与 confirm MAE（0.086–0.092）的
巨大差异主要是**这两个尾部样本造成的分布差**，不是训练过程差异；跨集比较
（如"select 单 seed 领先未在 confirm 复现"）必须带上这个分布事实。

## 引用

- 原始数值：`results/zinc_cssd_nonlinear_binding_v1/{select_eval,confirm_eval,interventions,checks}.json`、
  `runs/<arm>_s<seed>/{select,confirm}_predictions*.npz`（全部保持原样）。
- 代码：`experiments/luyin16/zinc_cssd_nonlinear_binding_v1.py`（`_contrast_stats`、
  `intervention_stats`、`branch_seed_for`）。
