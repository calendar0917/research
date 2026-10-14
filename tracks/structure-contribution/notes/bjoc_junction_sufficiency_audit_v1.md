# cscl-bjoc-junction-sufficiency-audit-v1 — 预注册与设计说明

Track: `tracks/structure-contribution` · 建立 / 冻结: 2026-10-14 · 状态: 预注册（训练前冻结，commit `82f76cf`）

协议配置: `configs/bjoc_junction_sufficiency_audit_v1.yaml`（`config_sha256` 写入
`results/bjoc_junction_sufficiency_audit_v1/protocol_freeze.json`）。

## 1. 单一命题

针对导师讨论提出的问题（K-SVD 必要性之后的下一个机制问题）：
**当前 BJOC-Null 的竞争性精度里，有多少根本不需要“学习到的 Junction 局部内容”？
以及 Null 把哪些结构组织信息不可逆地丢掉了、丢在哪一层？**

- **问题 A（预测价值）**：只改变 Junction 信息进入 Bond Composer 的方式，
  不新增任何可学习模块，比较三个功能臂：
  - `A0 NO_LEARNED_J_CONTENT`：`W_e = 0, V_e = 0, q_e = 0`；
  - `A1 GLOBAL_J_MOMENTS`：`W_e = μ_G, V_e = ν_G, q_e = 0`（把一个分子的
    Junction 一阶/二阶矩广播到每条键）；
  - `A2 P3_NULL_REFERENCE`：历史 `W_e = n_e·μ_G, V_e = n_e·ν_G,
    q_e = log1p(n_e)`，只读复用冻结的 P3-D-s0 模型与 Monitor 预测。
- **问题 B（结构信息）**：B1 代数盲性（所有权重排下的 Null 不变性）、
  B2 层级表示审计（Level 0–3）、B3 真实分子碰撞搜索、
  B4 干预测试、B5 合成已知真值（复用历史反例，不重跑 Organized-vs-Null）。

本轮是**面向导师讨论的机制探索**，不是 SOTA 搜索：单 seed、不换 split、
不调超参、不做 full-10k refit、不读 Locked、official valid 不使用、
official test 绝不加载。

## 2. 数据与标签边界

- **Split D**（`arch-breakthrough` 轮预注册，salt `cscl-bjoc-arch-breakthrough-v1-repl`，
  Fit 8000 / Monitor 1000 / Locked 1000；duplicate SMILES 组不跨集合）。
  该 split 已在 `cscl-bjoc-task-adaptive-ssl-v1` 轮被训练与 Monitor 评分过，
  因此本轮 Monitor 属于**历史暴露的探索性开发数据**，不是盲测，也不能与
  official split 数字并列比较。只解释 within-split 配对对照。
- 结构/标签访问：
  - 训练：只读 Fit raw-y（8,000 行）；
  - Monitor 标签：**本轮只读一次**，在所有新预测文件冻结并 hash 入 manifest 之后；
  - Locked：**本轮绝不读取**；
  - official valid：本轮完全不加载；official test：无代码路径。
- 标准化、SSL encoder 重用都基于 Fit-only；复用资产的 sha256 在运行时逐项重验，
  hash 不符即拒绝训练（`verify-inputs` + `freeze`）。
- 复用（只读）：`objects_fit_D.pt` / `objects_monitor_D.pt`（结构字段，LabelGuard 下构建）、
  `pretrain_D_s0.pt`（Fit-only masked-SSL encoder）、`soup_P3_D_s0.pt`、
  `predictions/monitor_P3_D_s0.npz`、`monitor_y_D.npy`（历史标签文件，hash 校验后复用）。

## 3. 问题 A：臂、对照与判定规则

- 三个臂共享：同一 scratch init（hash 断言）、同一 seed-0 Fit-only SSL encoder
  （hash 断言）、同一 batch 顺序（hash 断言）、同一 R3 配方
  （360 epochs / 22,680 updates、per-step cosine、Adam、lr 1e-3→1e-5、
  weight decay 1e-5、clip 5、batch 128、raw-y L1、最后 5 个 epoch 参数平均）。
- 唯一新训练轨迹：A0、A1（seed 0）。A2 只读复用，不重训。
- 对照（Monitor，paired molecule bootstrap 5,000 次，seed 0）：
  `Δ10 = MAE(A1) − MAE(A0)`（Junction 内容的边际价值）、
  `Δ21 = MAE(A2) − MAE(A1)`（每键 count 条件化的边际价值）、
  `Δ20 = MAE(A2) − MAE(A0)`（合计）。
- 判定：仅当点估计 ≤ **−0.002** 且 95% CI 上界 **< 0** 时给
  `J_MOMENTS_USEFUL_EXPLORATORY` / `COUNT_CONDITIONING_USEFUL_EXPLORATORY`；
  否则给对应的 `*_NO_CLEAR_GAIN`。任一对照 |Δ| ≤ 0.002 且 CI 跨 0 时附加
  `INCONCLUSIVE_SINGLE_SEED`。决策词汇表冻结为：
  `J_MOMENTS_USEFUL_EXPLORATORY` / `J_MOMENTS_NO_CLEAR_GAIN` /
  `COUNT_CONDITIONING_USEFUL_EXPLORATORY` / `COUNT_CONDITIONING_NO_CLEAR_GAIN` /
  `NULL_OWNERSHIP_BLINDNESS_VERIFIED` / `REAL_REPRESENTATION_COLLISION_FOUND` /
  `REAL_COLLISION_NOT_FOUND_WITHIN_BUDGET` / `INCONCLUSIVE_SINGLE_SEED`。
- 指标：raw-y MAE、RMSE、median、P90、ordinary（y > −3）、tail（y ≤ −3）、
  去掉 Top-10 误差分子后的 MAE、Fit MAE、训练 wall-clock、paired bootstrap CI。

### 透明性声明（预注册）

A0/A1 vs A2 是**功能消融**，不是活跃容量匹配对照：A0 会把 112,896 个
Junction-encoder 参数置为“结构上不可用”（梯度恒为 0），A1 只保留全局矩。
报告必须同时给出：分配参数量、活跃条目数（前向/反向中非零梯度的参数条目）、
各模块梯度范数、死通道数、前向耗时。A1/A2 都不恢复真实的 Junction-to-Bond
归属关系；归属问题已在历史轮（Organized-vs-Null、GLRC）研究过，本轮不重跑。

## 4. Stage A（CPU，任何 GPU 前的正确性门）

`stage-a` 在本地 CPU 上执行 17 项门（G1–G17），全部通过才允许训练/评分：

- G1：`A2 ≡ src.BJOC`（同一 forward 委托，逐位一致），并且 A2 重新加载历史 soup
  后的 Monitor 预测与冻结 `monitor_P3_D_s0.npz` 的 max|Δ| ≤ 1e-5；
- G2/G3：A0 的 W/V/q 严格为 0，且对 junction 内容互换完全不敏感；
- G4/G5/G6：A1 的 W/V 在分子内均匀；junction 行置换下 A1/A2 不变；
- G7：A2 的 `n_b` 与 bincount 精确一致；
- G8/G9：junction encoder 在 A1/A2 下梯度非零、在 A0 下无梯度；
- G10–G12：原子/键重标记、batch 组成不变性；
- G13：save/load 回放一致；G14：只用 Fit 标准化的证据；G15：标签隔离；
  G16：无 official valid/test 路径；G17：无禁用模块。
- 另记录 init/trained 两个相位的参数-梯度审计（逐模块梯度范数、
  composer 列块梯度、活跃条目计数、前向耗时）。

## 5. 问题 B：判定规则（预注册）

- **B1 代数盲性**：固定每条 Bond 的自身表示、每条 Bond 的 Junction 计数、
  全图 Junction embedding 多重集与其他内容时，Null 的 `W_e = n_e·μ_G`、
  `V_e = n_e·ν_G`、`q_e = log1p(n_e)` 与所有权重排无关；
  用历史 4-bond/4-junction 反例做 worked example，并在真实分子上做
  post-encoding incidence 交换（B4-Test2）。
- **B2 层级审计**：分析子集在计算任何表示之前冻结为 Fit 前 2000 行 + 全部 1000
  Monitor 行（label-blind）。Level 0：原子/键/degree/Junction 描述符直方图 +
  33 维 global 向量；Level 1：各对象 embedding 的一阶/二阶矩、每键计数多重集；
  Level 2：A0/A1/A2 下 `Σ_b c_b`；Level 3：321 维 reader 输入
  （`[Σc_b 96, Σa_i 96, Σa_i² 96, g_G 33]`）。诊断：有效秩、PCA 90/99% 维数、
  block 间 ridge R²（Fit-only 拟合）、Fit 标准化空间距离分位数。
- **B3 碰撞搜索**：official train 10000 的**结构字段**（x / edge_index / edge_attr），
  不读标签。精确键 = 33 维 global float32 字节 + 原子类型直方图 + 键类型直方图 +
  degree 直方图 + 每键 Junction 计数多重集，且 canonical SMILES 不同；
  若不存在精确碰撞，则报告 signature 空间最近邻（≤200 候选、最多报告 20 对），
  并对报告对做 NetworkX 类型图同构复核。裁决：
  - `REAL_REPRESENTATION_COLLISION_FOUND` 需要“精确结构统计碰撞 + canonical
    SMILES 不同（非同构） + reader 输入 Fit 标准化 L2 距离 ≤ Fit 不同结构对的
    1st percentile”；
  - 否则 `REAL_COLLISION_NOT_FOUND_WITHIN_BUDGET`，并如数报告最近邻距离。
  近邻永远不称为“相同”。
- **B4 干预**：junction 行置换（预期不变，历史已知，仅复核）、
  post-encoding 归属交换（Null 不变 / Organized 变化）、
  真实近碰撞对比、reader 输入 4 个 block 的 neutralisation（越界输入敏感性诊断，
  不是化学归因）。
- **B5 合成已知真值**：复用 `structural_relation_composer_v1` 的对象级反例，
  并构造可扩展族（每个键恰被两个 Junction 触及、每个 Junction 触及相同无序键类型对，
  从而编码后 `w` 多重集固定；只改归属）。目标 `M` 只依赖 Junction 多重集，
  目标 `O` 依赖真实归属；只做确定性可区分性检查，不训练。

## 6. 预算与执行

- GPU 硬上限 **1.5 GPU-hours**（按 rr job wall-clock）；新轨迹仅 A0/A1 各约
  0.17 h，合计约 0.34 h；A2 只读。Part B 与全部门禁全部在 CPU 完成。
- CPU 侧：`stage-a` 17 门、`grad-audit`（init/trained）、预测冻结、
  Monitor 单次评分、B1–B5。
- 已声明偏离（写入 `post_freeze_amendments.json`）：训练与 Monitor 评分前的
  **driver 级**修订（B3 行对齐/近邻归属修正、B5 族构造、finalize 聚合、
  `b-algebraic` 的 detach 修复）——这些修订不改 `code/bjoc_junction_sufficiency_models_v1.py`、
  不改 config 的臂定义/门槛/预算，模型侧 hash 与 config hash 不变；
  训练轨迹在冻结 commit `82f76cf` 上执行并记录各自 commit/hash。

## 7. 停止纪律

- 不因 Monitor 分数改臂定义、门槛、λ、seed 或重训；不追更好的 MAE；
- 不读 Locked、不加载 official valid/test；
- 不重跑已证过的 Organized-vs-Null；
- 单 seed 结论一律限定为探索性，不写成总体声明。
