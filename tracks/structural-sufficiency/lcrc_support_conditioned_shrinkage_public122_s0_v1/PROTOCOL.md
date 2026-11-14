# PROTOCOL — 支持度条件化的联合块收缩（冻结执行协议）

- round: `lcrc-support-conditioned-shrinkage-public122-s0-v1`
- 本文件在**任何正式训练之前**写出。科学常量以本文件 + `code/sc_common.py` + `configs/experiment.yaml` 为准。
- 性质：**新的路线实验**（不是旧 S 家族的数值修复，也不是第三轮冻结读出探针）。
- 主数据 Lipo（官方 `random122`，seed 0），条件下 BACE（官方 `scaffold122`）。
- Test 全程可见，用于本文件规定的固定采购判断；统一标 **DEVELOPMENT_TEST / TEST_USED_FOR_ROUTE_SELECTION**，不称独立最终测试。

---

## 0. 研究问题

对根 `v` 的二跳 path occurrence，联合统计块 `J` 由完整联合布局给出（`J_full`）；同一数据按
`(root, bt2)` 分组的边际算子给出 `J_marg`（冻结 B5）。本轮唯一的改动是把**联合块**按一个
**跨训练分子结构支持度**得到的冻结先验 `a_v` 在两者之间插值：

```
J_new(v) = J_marg(v) + a_v * (J_full(v) - J_marg(v))
```

`type / end / closure / content / S1 / atom_scalar / global42 / f / rho / dropout` 全部维持 C1。
**`content`（C_v）必须仍是 FULL**，因此 `a=0` 端点**不是**完整 B5，只是"仅 J 走 marginal_bt2"。

- **H1**：按较低结构支持度加强收缩（`SELECT`），有超过**两种均匀收缩**（`UNIFORM`、`UNIFORM_OOF`）的预测价值。
- 竞争解释：一般输入收缩（两种 UNIFORM 即其对照）、配方收益、干预实际太弱（`INTERVENTION_WEAK_OR_UNUSED`）、
  收益只在少数高影响行、支持代理与所需规律无关（`REVERSE` 方向检验）。

核心比较是 **SELECT 对两种 UNIFORM**，不只是 SELECT 对较弱普通模型。

## 1. 资源、边界与授权

| 项 | 值 |
|---|---|
| 墙钟 | 目标 2–3 h，硬上限 4 h |
| GPU·h | 软上限 2，硬上限 3 A100·h（按 `rr` 占卡 wall 计，含失败/等待） |
| 正式新训练 | 最多 15 条，全部 seed 0 |
| 冒烟 | 最多 8 个短作业，各 ≤8 updates，只查实现 |
| 卡 | `res-gpu0` / `res-gpu1`，每卡同时最多一条本轮进程，进程内 `cuda:0`；不用 `res-2`；不动其他任务 |

**0 条也是允许的科学结果**：若结构可行性门槛（§3）失败，就停止训练采购并交付结构可行性结论。
不增 seed、不改官方划分、不加预训练/辅助目标/额外特征通道/PNA/新性质/第四跳，不搜学习率、`κ`、
收缩强度或任何第三候选粒度。技术性重试保持同一科学规格并计成本；不因阴性结果重试。

## 2. 被收缩对象与"何时可干预"的结构判定（与权重无关）

对根 `v`，`D_v(c,e)` 是其全部二跳 path occurrence 中 `(path_cat=c, path_end63=e)` 的**整数计数**（保留重复次数）。
分组 `b = path_cat % N_BOND_TYPES`（`N_BOND_TYPES = 5`，与 `marginal_bt2` 源码一致；不自行重解码键型）。

```
N_{v,b}(c,e) = D_v 在 b 组内的表            n_b = ΣN      n_c = Σ_e N      n_e = Σ_c N
R_{v,b}(c,e) = N_{v,b}(c,e) - n_c n_e / n_b               空组为 0
```

- 若所有整数 `n_b N - n_c n_e` 都为 0 ⇒ 该根 **inactive**（此时 `J_full ≡ J_marg`，逐位相等），否则 **active**。
- `r_v = ||R||_F`（float64）。只在**每组实际出现的 c 值集合 × e 值集合**的小笛卡尔积上计算，
  绝不为每根分配全码空间（`n_path_codes × n_nbr_codes`）的稠密矩阵。**未出现**的 `(c,e)` 对
  仍属于 R（其项 `-n_c n_e/n_b` 一般非零），这一点决定了 `r_v` 的数值。
- 恒等式（不做成"真实协方差"）：`J_full - J_marg = Σ_b W( Σ_{c,e} R_{v,b}(c,e) U_c ⊙ V_e )`，bias 相消。
  已在 `tests` 与 `sc_checks.residual_linear_map` 里按 ≤1e-5 核验。
- **inactive 根所有臂 `a=1`**，直接走 `J_full`：不把数值舍入差解释成作用。

## 3. 支持键、独立结构计数与训练内可行性（先于任何训练）

### 3.1 两个候选键（固定优先，不增加第三种）

```
K_FINE(v)  = ( canonical_float32(atom_feat[v], 51 列), sorted tuple(c, e, D_v(c,e)) )
K_JOINT(v) = (                                       sorted tuple(c, e, D_v(c,e)) )
```

`K_JOINT` 是**整根的联合计数模式**，不是单条 `(c,e)` 粗键（旧 S 修正表的粗键只有约 114 类，不能当"完整局部模式"）。
两键都不含 `atom_id / path_end 索引 / row_id / SMILES / global / 标签`。AF51 沿用签名规范：
有限 float32、`-0` 归 `+0`、显式小端字节序；哈希只作索引，命中比较**原内容**
（`analysis/preflight.json::stage_keys.collision_check_*` 记录每个 digest 与原内容一一对应）。

### 3.2 支持计数单位 = fit 中的**不同 canonical 分子结构**

- 用原发布 SMILES 原样解析，去 atom-map 编号后 `MolToSmiles(canonical=True, isomericSmiles=True)` 作为结构 group；
  不脱盐、不做 tautomer 归并、不改模型输入。RDKit 版本与 group 映射记录在 `analysis/preflight.json`。
- 对每个 group 汇总其**所有根**的 key 集合，只计一次：同分子多根、多路径、重复记录都不增加 `c`。
- `c_T(s) = fit 中包含 s 的 canonical group 数`；未知 key = 0；计数 ≤ fit 的不同结构数。
- 同 canonical group 的输入键多重集必须一致；不一致则报告身份问题并**停止**（本轮 Lipo/BACE 均无重复结构，
  检查数 0，见 preflight）。
- Train 查询也用同一完整 fit 表（不采用 Train 专用 LOO 系数）⇒ 存在**自计数乐观性**，报告如实列出。
- N25 表**不使用**被排除 Train 的结构计数（`tests::test_fit25_support_excludes_held_out_structures`）。

### 3.3 参考剂量权重与固定加权 mid-CDF

每个 fit 记录 `G` 内，active 根 `w_v = r_v / Σ_{u∈G} r_u`（无 active 根的记录权重全 0；有 active 根的记录总权重 1）。
`W = Σ_fit roots w_v`。

```
F_T(x) = [ Σ_v w_v 1(c_T(K_v) < x) + 0.5 Σ_v w_v 1(c_T(K_v) = x) ] / W      (float64)
```

`F_T(0) = 0`（未知 key）。支持计数去结构重复 与 CDF 匹配实际 fit 记录，是两个明确不同的口径。

### 3.4 正式门控（仅 active 根）

| 臂 | `a_v` |
|---|---|
| `BASE` | 1（走原算子，无门控模块） |
| `UNIFORM` | 0.70 |
| `UNIFORM_OOF` | `a_oof`（见 §3.6） |
| `SELECT` | `0.70 + 0.60*(F_T(c_T(K_v)) - 0.5)`，范围 [0.40, 1.00] |
| `REVERSE` | `0.70 - 0.60*(F_T(c_T(K_v)) - 0.5)`，范围 [0.40, 1.00] |

⇒ SELECT 低支持收缩更多，REVERSE 高支持收缩更多；三者在 fit 参考权重下平均 `a` **严格等于 0.70**（容差 1e-10，
`tests::test_alpha_mean_is_uniform_dose`）。不声称完整门控分布或训练后**实际 J 剂量**也匹配——后者实测并报告。
`a` 是冻结结构先验，**无可学习参数**；五臂参数量都是 228 674。

### 3.5 Train-only 5-fold 可行性（零训练、零预测标签）

fold = `int(sha256('lcrc-support-fold-v1|' + canonical_smiles), 16) % 5`，重复结构全在一 fold。
每次只用另外 4 fold 重建 `c` 与 `F`，查询被留 fold（不读标签、不训练、不构造新 benchmark split）。
合并 OOF 查询，权重仍是每 fit 记录内部 `r` 归一化。对每个候选 key、每个 N 必须同时满足：

| 门槛 | 内容 |
|---|---|
| E1 | 有 active 根的不同 canonical groups ≥ **100**，且 active keys ≥ **20** |
| E2 | 完整 fit 上加权 `a_SELECT` 的 `q90-q10` ≥ **0.20** |
| E3 | 合并 OOF 查询的加权 `a_SELECT` 的 `q90-q10` ≥ **0.20** |
| E4 | OOF 未知 key（`c=0`）加权占比 ≤ **0.60** |
| E5 | OOF 中 `c≤2` 与 `c≥5` 的加权占比分别 ≥ **0.10** |

加权分位数 = CDF 首次 ≥ q 的最小观测值，不插值。这些是**干预可判读性**标准，不是统计显著性或自然常数。

### 3.6 第五臂 `UNIFORM_OOF`

把 OOF 查询的 `a_SELECT` 按同一 `w` 加权平均，得到**唯一常数** `a_oof`；训练/所有查询的 active 根一律 `a=a_oof`，
inactive 仍 1。它**只用 Train 结构**估计"新分子通常会受多强的收缩"，不看 Valid/Test 结构或标签；不要求 fit 平均 = 0.70。
保留它正是为了控制"SELECT 在留出集平均收缩更强"的解释。`a_oof` 与 0.70 都事先定义，**不按效果二选一**。

### 3.7 冻结的 key 选择规则（只看 Train 结构 + 上述门槛）

1. 若某 key 在 fit100 与 fit25 都通过 ⇒ 取其中 `K_FINE` 优先、`K_JOINT` 次之，购买两 N。
2. 否则若某 key 在 fit100 通过 ⇒ 同优先级购买 fit100，fit25 记 `NOT_RUN_NO_TARGETABLE_SUPPORT`。
3. 若两个 key 在 fit100 都失败 ⇒ **停止**科学训练，交付 `NO_TARGETABLE_SUPPORT_UNDER_REGISTERED_KEYS`
   （即使 fit25 单独通过也不临时改主实验）。

不得看 Valid/Test 分布、标签、历史逐行误差或预测增益来换 key 或放宽门槛。冻结 key 后才生成 Valid/Test 系数；
报告其分布，**不据此换方案**。

**本轮已冻结的结果**（`analysis/key_selection.json`、`analysis/freeze.json`）：

| 项 | 值 |
|---|---|
| 状态 | `KEYS_PASS_BOTH_N` |
| 选定 key | **`K_FINE`**（两 N 两 key 全通过；按优先级取 `K_FINE`） |
| 购买 | Lipo fit100 五臂 + fit25 五臂（10 条） |
| `a_oof` | fit100 = 0.687 279 882 3，fit25 = 0.660 228 581 9 |
| support 表 sha256 | fit100 `d8ed150b…`，fit25 `b8d8bd12…` |

## 4. 实现身份与有效干预检查（先行，全部零训练或冒烟）

1. 单分子重复同 key、重复记录不增加支持；不同结构共享 key 才增加；未知映射 0；计数上界正确。
2. 改任何标签不改变支持表/CDF/key 选择（`targets()` 被禁用时支持流程仍可完成）；N25 之外 Train 结构不进入其支持表；group folds 无重复结构穿越。
3. 原子重编号、batch 顺序/大小、同批其它分子不改变本分子的 key、系数或 eval 输出（容差 2e-5）。用 `atom_ptr/path_ptr` 正确取根，不把 root 索引当分子索引。
4. `R` 整数判定、`J_full-J_marg` 恒等式（≤1e-5）、`a=1`→C1（逐位）、`a=0`→仅 J 的 `marginal_bt2`；其他局部块逐项与 C1 一致；`content` 强制 FULL。`BASE` 直接走原算子。
5. SELECT/UNIFORM/REVERSE 结构加权平均 `a`=0.70；`UNIFORM_OOF` 的 active 常数 = 冻结 `a_oof`；REVERSE 单调方向相反。K 一致 ⇒ `a` 一致；充分签名一致 ⇒ K 一致；不依赖 row_id。
6. 训练前共享 state、batch 计划、dropout RNG 一致；评估/诊断保存恢复全部 RNG 与 model mode；所有原有参数可训练。
7. 在固定初始权重下，取 fit 分子按公开 row 排序前 64 个，检查实际 J 变化、local 输入变化与一次梯度路径。
   若 active R 与门控跨度通过而实际 J 变化全部 ≤1e-8 ⇒ `INTERVENTION_WEAK_OR_UNUSED`（实现退化），停止定位，
   不靠加大强度救配置。初始与选中/最终权重都实测：`a` 分布、`‖J_full-J_marg‖`、`‖(1-a)(J_full-J_marg)‖`、后者/`‖J_full‖`，
   按分子先聚合再报分布。强度匹配只是结构参考权重下的平均匹配。

## 5. 正式清单与固定采购

Lipo fit100：`BASE / UNIFORM / UNIFORM_OOF / SELECT / REVERSE` 五条，**必须全完成后比较**（不只看有利臂）。
fit25：同五臂（上游诊断固定的 840 行嵌套子集，不重抽、不补 fit50）。

Lipo 推进门槛（在任一已购买 N 上满足即可）：
1. `R_SEL` 下 SELECT 相对**新 BASE** 的 Valid 与 Test RMSE 均改善 ≥1%；两 split 的 MAE 均不恶化超过 1%；
2. SELECT 相对 `UNIFORM`、`UNIFORM_OOF`、`REVERSE` 的 RMSE 增益在 Valid/Test 均为正。

这是后续预算门槛，不是显著性定义。只过一 split、只靠 Test oracle、仅优于较弱旧 R1 都不采购。
过门槛且资源允许 ⇒ 只购买一次 BACE fit100 五臂（≤5 条）：沿用 Lipo 已冻结的**同一种 key**（不另选粒度），
用 BACE Train 结构重建支持/CDF 并先过 §3.5 门槛，失败记 `NOT_RUN_NO_TARGETABLE_SUPPORT`；
主指标 ROC-AUC，配方固定 R1（仅 `weight_decay=1e-3`），原始 0/1 `BCEWithLogits`，不加 class weights；
AUC 增益 ≥0.005 且 Valid/Test 双向优于 BASE 与两种 UNIFORM、REVERSE 才算跨任务兼容。
BACE 门槛：`K_FINE` 已通过（`a_oof=0.6943`）、`K_JOINT` 因 E5 low-share=0.075 未通过——若采购，仍用 `K_FINE`。
资源预计不足以完整购买五臂 ⇒ `NOT_PURCHASED_BY_BUDGET`，不只跑 SELECT。

调度：每个 N 先 `BASE` 与 `SELECT`，再填两种 UNIFORM 与 `REVERSE`；两卡各自串行、空卡接下一条已登记且未提交的作业；
消息式 fail-closed launch ledger（查不到/暂时空的队列 ≠ 未启动；未确认终态不得重复提交；每作业独立 `out_dir`）。
Lipo 未过门槛即停止，不通过 BACE 寻找阳性。不启动新 D-MPNN 训练；最多 15 条正式训练，未购买项写原因、不记成效果 0。

## 6. 训练、归一化与选点（全部固定）

C1：原 228 674 参数、输入 spec 不变、dropout 0.10（`h` 上一次调用供 `f/rho` 共用）、FP32、MSE。
Lipo 用 R3（peak_lr 1e-3 → min 1e-5、wd 1e-2、betas (.9,.999)、eps 1e-8、clip 5、batch 128）；
BACE 若采购用 R1（仅 wd=1e-3）。
初始化 = 上游 `FULL seed0 → C1 remap` 的同一可训练 state；同任务同 N 五臂**共享**该初始 state 及完整 batch 计划；
支持 buffer 不消耗训练 RNG。各 N 都 `U0 = 360*ceil(3360/128) = 9720` updates，per-step cosine 固定到 U0，
无 warmup/续训/重启/EMA/SWA（fit25 完成更多 epoch 是固定 update 预算的结果，不偷改 horizon）。

归一化：结构 norm 用**全部官方 Train 的无标签结构**拟合一次；目标 norm 用 **Lipo 固定 840 行标签**拟合一次；
各 N 共用，hash 与上游诊断轮一致（`analysis/freeze.json`）。训练损失只用当前 fit 标签；支持只用当前 fit 结构；
禁止用被排除的 Train 标签做训练/选点。官方 Valid/Test 照常可见。

每完成一个 fit epoch 评估完整 `train_fit` / Valid / Test；另在 `round(j*U0/50), j=1..50` 的共同 grid 评估并保存逐分子预测，
step0 另存但不进主候选池。主 `R_SEL` = 这 50 点内 Valid RMSE 最小（BACE 最大 ROC-AUC），同分取较早 ⇒ 各 N 主选择机会相同。
辅 `A_SEL` = Valid MAE（BACE BCE）最小；`W_SEL` = 5 点滑动均值主指标最好窗口的**末状态**；两者只作敏感性。
可另记每 epoch best-Valid 作次表，不与 50 点主表混称，不用 Test 决定 epoch。保存主选中、各辅规则选中、最终权重与匹配预测。

## 7. 只做能改变判断的分析

gain 正号表示更好：`RMSE_control - RMSE_candidate`，`AUC_candidate - AUC_control`。

- **A 主表**：每任务/N 五臂的 Train_fit/Valid/Test、MAE 或 PR-AUC、选中 update、参数量、GPU 占卡时间。
  主比较四组 `SELECT vs BASE/UNIFORM/UNIFORM_OOF/REVERSE`；另报三对照 vs BASE。旧 C1_R3 / D-MPNN 参照独立标源，
  不与本轮新数字相减。
- **B 不确定性**：固定选中权重做 paired canonical-structure cluster bootstrap 2000 次，同任务所有比较共用索引；
  组内全部评估行共同保留，指标仍按官方行权重。另做 Murcko-scaffold cluster 敏感性（空 scaffold 按 canonical 结构各自分组），
  报 cluster 数与最大 cluster 比例。不把单 seed 的评估区间解释为训练稳定性；不对 1800/9720 相关步做 iid 检验。
  区间跨 0 如实写；过线只能称 provisional。
- **C 低标签**：两 N 都跑则比较 SELECT 对 BASE/两种 UNIFORM 的收益随 N 如何变；计算同一行上的 `gain25-gain100` 配对 bootstrap。
  只用 N25 有效 ⇒ 限定为 `LOW_LABEL_ONLY`，不称全量 Lipo 性能创新。
- **D 支持解释**：分子负担 `B_G = Σ_v w_v(1 - a_SELECT(v))`（无 active 根记 0 并单列）。用 **Train-only OOF** 的
  `B_G` 的 q25/q75 冻结低/高负担阈值，套到 Valid/Test；分组对照 BASE/两种 UNIFORM 的逐行 ΔSSE、RMSE、MAE、人数与区间。
  某组 <30 个不同 canonical 结构 ⇒ 只报 `LOW_SUPPORT`，不作组间机制主张；不按 Test 误差找新阈值。
  报根/分子 unknown-key 比例、自计数与 OOF 支持差异、无 active 根比例；不把支持计数当独立实验标签数。
- **E 影响度**：headline 始终全样本；按本轮 BASE 绝对误差降序（ties 公开 row id）固定删除最大 1%/5%，五臂删同一行集；
  另列最大正/负 ΔSSE 各 3 行及受益/受损比例。辅 A/W 选择下方向是否保持、最后 10 个 grid 点中位数单列。
- **F 结构追溯的最小保留**：选中 SELECT 上用固定权重把 `a` 设 1，仅检查预测变化（不当重训收益证据）；
  用已存在的五对输入别名检查新 `a` 与输出仍相同（容差 2e-5）。`a` 是固定训练表下 `Σ_v` 的函数，
  未变 `Σ` 的根仍有相同 `a/q`；`Δp = s_y[Σ_变根 Δq + Δρ]` 用一个可恢复案例验证算术即可。

## 8. 预定结论与停止

| 触发 | 状态 |
|---|---|
| 无 key 通过 | `NO_TARGETABLE_SUPPORT_UNDER_REGISTERED_KEYS`（代理/对照可行性失败，不说"稀有结构不重要"） |
| 门控改变 J/预测很弱 | `INTERVENTION_WEAK_OR_UNUSED`（区分实施退化与训练后不使用；不调强度救结果） |
| SELECT 好于 BASE 但未在两 split 同时超过两种 UNIFORM | `GENERIC_CONTRACTION_SIGNAL`；若只有 `UNIFORM_OOF` 复现收益，优先整体收缩剂量解释 |
| SELECT 好于 BASE 但不稳定优于 REVERSE | `DIRECTION_NOT_IDENTIFIED` |
| 过 §5 门槛、方向与分组兼容 | `SUPPORT_DIRECTED_SIGNAL_THIS_CONFIG`（区间跨 0 则 `UNCERTAIN`，最多购买已授权 BACE） |
| 只 N25 有效 | `LOW_LABEL_ONLY` |
| 只一 split / 一种选点好 | `SPLIT_OR_SELECTION_SENSITIVE` |
| 五臂无可兑现收益 | `NO_PREDICTIVE_GAIN_THIS_CONFIG`（停止这项先验；不自动搜第三 key、更多 alpha 或 seed） |

先报告观测，再给投入判断；禁止用排除法宣布"表达力不足"。
即使阳性，也只支持**这一**结构代理、算子、配方、split 与单 seed 下的学习偏置；不证明校准置信度、
普遍稀有结构因果、VC 复杂度下降、化学因果；不声称击败所有可能的均匀收缩强度。
`a` 改变了学习函数，不能仅因参数量相同声称完整函数族表达力相同。

## 9. 交付

`PROTOCOL.md`、`configs/experiment.yaml`、`code/`、`tests/`、
`analysis/{preflight,key_selection,freeze,source_manifest,run_manifest,comparisons,intervention,support_groups,influence,budget}.json`、
主表 csv、最多 3 张核心图、中文 `REPORT.md`（开头回答：①代理是否有可判别性？②选择性是否超过两种均匀及反向？
③低标签与全量是否一致？④下一步保留什么、停止什么？）。
入口支持 `prepare/preflight/smoke/train/analyze/report` 与安全恢复已完成作业；远端训练只依赖已 materialize 的结构/标签缓存
（RDKit 解析、canonical/scaffold 计算放本地 CPU）。

---

## 附录 A — 实现映射与事后记录（只在训练开始后追加）

（训练开始后追加，不改上方科学常量。）
