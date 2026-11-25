# LCRC 计算互补性与结构函数共性研究 v0 — PROTOCOL

- Round: `tracks/structural-sufficiency/lcrc_computational_commonality_v0/`
- 数据：`ogbg-molpcba` 官方 scaffold split（Train 350343 / Valid 43793 / Test 43793，128 任务）
- 指标：官方 OGB macro AP（masked；只对官方可评价任务取平均，127 个）
- Test 角色：`DEVELOPMENT_TEST / POST_HOC_EXPOSED_TEST_DIAGNOSIS`（本轮不用于任何权重/超参/架构选择）
- 本轮不训练任何预测模型，不改变历史结果，不运行 PCQM4Mv2

## 冻结模型集合

核心：`W2_s0`、`W2_s1`、`FIXED1_s0`、`FIXED1_s1`、`DYNAMIC_s0`、`DYNAMIC_s1`（各自历史 best-Valid checkpoint）。

扩展：`S3_s0`、`RC_REAL_s0`、`BOC_FULL_s0`（历史冻结 logits 已在盘）。
`CPC_s0`、`RHIC_s0` 的 Test logits 存在，但其 Valid logits 从未落盘且需要额外的路径结构缓存才能补齐推理，
本轮**未纳入**主分析，记录在 `analysis/asset_inventory.json`。

历史训练 horizon 差异（会影响组合，不宣称架构因果）：W2_s0 / DYNAMIC_s0 为历史 40-epoch 配方；
W2_s1、FIXED1_s0/s1、DYNAMIC_s1 为 DSCR 的 25-epoch 前缀（frozen 40-epoch 余弦的前缀，不重压缩）。

## Part A：预测组合

1. 逐任务百分位排序分数（全 split 分子、含该任务标签缺失的分子）：
   `r_{m,t}(x) = (average_rank(logit_{m,t}(x)) - 0.5) / N`，相同 logits 取平均秩。
   这是对**整个候选集合**的无标签传导式排序组合，不是逐样本独立部署方案。
2. 主指标：排序分数上的官方 AP。敏感性分析：原始 logits 平均。
3. 组合（等权）：
   - 同构：`W2_s0+W2_s1`、`FIXED1_s0+FIXED1_s1`、`DYNAMIC_s0+DYNAMIC_s1`
   - 异构：`W2_s0+FIXED1_s0`、`W2_s0+DYNAMIC_s0`、`FIXED1_s0+DYNAMIC_s0`、`W2_s0+S3_s0`、`W2_s0+RC_REAL_s0`
   - 多模型：`CORE` = W2/FIXED1/DYNAMIC 的 seed0 等权；`EXTENDED` = 全部可用 seed0 架构等权
   不搜索最优子集，不按 Test 剔除模型。
4. 任务权重（仅对 `W2+FIXED1`、`W2+DYNAMIC`，候选 `w ∈ {0,0.25,0.5,0.75,1}`）：
   - `EQUAL`：w=0.5，不拟合
   - `GLOBAL`：Valid macro AP 最优权重
   - `TASK`：逐任务 Valid AP 最优；并列取最接近 GLOBAL、再取较小 w；正例或负例 < 20 直接 GLOBAL；
     收缩 `w_final = α w_task + (1-α) w_global`，`α = n_pos/(n_pos+100)`，不搜索收缩系数
   - `Valid` 交叉拟合：Bemis–Murcko scaffold 3 折（按 scaffold 大小平衡分子数，固定哈希顺序），
     每折用另外两折拟合权重，预测当前折，拼接得 OOF AP；随后用完整 Valid 拟合权重并评价 Test。
5. Oracle（仅参照）：(a) 逐任务 Valid 最优单模型 → Test；(b) 逐任务 Test 事后最优单模型（POST_HOC_ORACLE）；
   (c) 两个核心模型对的 Test 事后最优离散权重（POST_HOC_ORACLE）。
6. 统计：2000 次配对任务 bootstrap；去掉 |ΔAP| 最大的 1/3/5 个任务的敏感性分析；
   Valid/Test 方向一致率；`Valid/Test 正例均 ≥ 20` 的高支持子集（不替代官方指标）。
7. 判定参考（+0.005 AP 为初步实用性参考，非显著性阈值）：
   `LARGE_COMPLEMENTARITY` / `TASK_SPECIFIC_SIGNAL` / `GENERIC_ENSEMBLE_GAIN` / `LIMITED_COMPLEMENTARITY`
   （可并存，也可 `INCONCLUSIVE`）。必须报告与 0.28、0.29 AP 的距离。

## Part B：冻结结构函数

教师：`W2_s0` best-Valid checkpoint（eval、冻结、无梯度）。

`T(p,e) = W_type @ path_emb(p) + W_end @ end_emb(e) + W_joint @ joint_proj(path_u(p) ⊙ path_v(e))`，
其中 `W_*` 是 local MLP 第一层权重中 `s2_type` / `s2_end` / `s2_joint` 的列块；`joint_proj` 的 bias 按原实现保留；
local MLP 自身 bias 只加一次。只研究这三项，不声称分解完整 S2 / R2。

必须先在真实分子上重建：第一层 preactivation、`h0`、最终 logits（FP32 累加容差内），否则 Part B 停止。

真实结构成分（按实际 codec 解码）：`p → (middle atom type, bond1, bond2)`，`e → (end atom type, bond2)`；
若 `bond2` 与 `e` 中字段完全重复则只保留一次（本轮代码内逐 split 校验并记录）。

归纳模型（目标为冻结教师 `T_total` / `T_joint` 向量，不使用任务标签）：MEAN、CODE_ADD、SEM_1、SEM_2、
NULL_1/NULL_2（同特征/同维度的频率分层固定置换控制）。固定 Ridge：float64、不惩罚截距、
`mean squared error + 1e-4 ||β||²`、不搜索超参数。三种加权：类型等权（主）、occurrence、分子支持。

评价协议：
1. 全已知类型拟合（相对 RMS、cosine、相对 MEAN 的 skill、设计矩阵秩、自由系数数）；
2. 类型留出：`doc_freq ≥ 3` 的键按文档频率分桶、固定 hash 取 20% 作类型留出；Fit/Holdout 内部分别做 NULL 置换；
   报告「完整键未见但各成分均已见」与「存在未见成分」两个子集。

真实前向替换：`T(p,e) → T_hat(p,e)`（第一层减去对应块投影、按原根节点与重数加回；其余全部不变），
Train 未见 `(p,e)` 保留教师原 `T` 并单独报告覆盖率；禁止微调、重拟合读出或标签校准。
评价：ORIGINAL / MEAN / SEM_2 / NULL_2（优先），CODE_ADD / SEM_1（次优先）；
官方 Valid/Test macro AP、ΔAP、逐任务 AP、logits RMS 变化、逐任务 Spearman、Top 1% 重合、
preactivation/h0 重建误差、替换比例；Train 使用固定哈希 4096 分子仅作描述性监测。

初步实用性参考（筛选，非等价性证明）：真实结构描述相对随机控制的类型留出误差改善 ≥ 10%；
替换后 Valid/Test AP 下降均 ≤ 0.002；替换覆盖充分；结果不来自大量未替换路径或不重要计算块。

## 综合判定

二维矩阵 A+/A− × B+/B−（`FUNCTION_RECONSTRUCTION`、`SEMANTIC_REUSE`、`FUNCTIONAL_PRESERVATION`、
`NONTRIVIAL_INTERACTION`），A 侧标签为第 7 节四种现象，B 侧为上述四项；证据冲突时标 `INCONCLUSIVE`。
不因本轮结果启动下一版训练。
