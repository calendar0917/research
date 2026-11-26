# 结构响应路线澄清 8h v0 — 冻结协议

Round id: `structural-response-route-clarification-8h-v0`
Directory: `tracks/structural-sufficiency/structural_response_route_clarification_8h_v0/`

本文件在**第一个正式训练启动前**冻结。它只落地任务书
（`/home/calendar/Downloads/structural_response_route_8h_prompt.txt`）里已经给定的
科学定义；执行 Agent 不新增公式、隐状态数、通道数、可加原子头、字典发现、
IG/oracle 训练、teacher、蒸馏、额外属性特征或超参搜索。

---

## 0. 本轮要买的证据

| 编号 | 问题 | 对应阶段 |
|---|---|---|
| A1 | 从头训练后 `H_FULL` 与 `H_IND`、`ETA_FULL` 与 `ETA_STAR` 的联合计算是否分别有收益、是否配合 | A（4 臂） |
| B1 | 在同一严格真实编辑挑战上，结构算子与训练目标各贡献什么 | B（D0 九臂） |
| C1 | 所选路线是否对另一实测性质保留方向、是否需要全图根响应 | C（ESOL 3 臂）、R、P |

两条评价轴并列：整分子预测（能否接近强预测参照）与结构响应（真实分子
A→B 的 `f(B)−f(A)` 能否预测 `y(B)−y(A)`，并在其他环境复用）。“解释”在此仅指
可明确描述、可在真实分子上核验的结构替换响应；不声称唯一原子贡献、化学因果
机制、通道化学命名或自动发现结构词典。

## 1. 资产与数据身份（只读）

| 资产 | 路径 | SHA-256 |
|---|---|---|
| 原始 Lipo 缓存 | `tracks/structural-query-learning/data/ogbg_mollipo_sq-pilot-cache-v1.npz` | `fe002287…3c7c` |
| 片段缓存 Lipo | `response_dictionary_small_s0_v1/data/ogbg_mollipo_rd_frag_v1.npz` | `ba6e9a41…a1d8` |
| 片段缓存 ESOL | `response_dictionary_small_s0_v1/data/ogbg_molesol_rd_frag_v1.npz` | `fac644d0…a62d` |
| D0 划分 | `acra_relation_generalization_pilot_v1/data/split_train_sub_dev.json` | `675289f6…e845` |
| 严格编辑目录 | `response_dictionary_task_transfer_s0_v1/analysis/structure/catalogue_pairs.csv` | `64186ac5…4bcc` |

复用代码（只读，SHA 记录在 `checkpoints/manifest.json` 与本目录 `code/` 的导入处）：
`full_graph_cavity_response_pilot_v0/code/{fgcr_model,fgcr_inference}.py`、
`fgcr_joint_response_increment_audit_v0/code/run_audit.py`（局部参考张量代数）、
`response_dictionary_small_s0_v1/code/{rd_model,rd_train,rd_common}.py`、
`response_dictionary_lipo_followup_s0_v1/code/fu_model.py`（`FUSION_MLP`）、
`gnn_route_clarification_8h_v1/code/{gr_models,gr_common}.py`（`G0`/`Rf`）。

**Test 访问。**官方 npz 是单一归档，`load_bundle` 一次物化整个文件（含
`test_idx` 与 Test 标签/特征），随后**立即从工作 payload 中删除 Test 索引数组**并
记录该事实（`checks.json: data_test_block_dropped`）。Test 行不进入任何 plan、
forward、scaler、选模或指标。本协议与全部产物不含任何 Test 预测或指标。

D0：`train_sub_rows=2856`（Fit）、`dev_rows=504`（Dev），均为 official Lipo Train 的
原始 row ID，互斥且并为 3360。映射按原始 ID，不使用 split-local 索引。

## 2. 先冻结的严格响应挑战（设计阶段计数，执行时复算）

全 catalogue 2597 对；排除任一端点落在 official Valid/Test 后得 **2133** 对、
**391** 个配对连通分量。D0 映射（执行时复算一致，见 `analysis/support.json`）：

| 范围 | 对 | 分子 | Dev 分子 | 2133 图分量 |
|---|---|---|---|---|
| Fit–Fit | 1619 | 1146 | 0 | 340 |
| Fit–Dev | 238 | 222 | 93 | 44 |
| Dev–Dev | 276 | 202 | 202 | 62 |

Fit–Fit 的 1619 边图自身恰好 **340** 个连通分量，训练采样只用这张图。

Dev–Dev 的 exact-transform 支持（`transform_id`）：Fit 出现过 123 对；剔除被评估
pair 所属整个 2133 图分量后仍有其他 Fit 分量支持 **119 对 / 51 分量 / 65 transform**；
至少两个其他分量支持 **97 对 / 45 分量 / 46 transform**。未获其他分量支持
**157 对**（与 119 互斥覆盖 276）。

固定评价集：主集 = 276；复用支持集 = 119；较强支持敏感性 = 97；未支持 = 157。
Fit–Dev 238 为次要评价，一端已训练，不称两端留出。**不据标签大小、预测、误差或
模型是否覆盖 selected view 改变任何集合。**

方向：catalogue 的 `transform_id = s_key_a > s_key_b`，故每行存储的 A→B 方向**就是**
canonical source→target 方向；校验所有 2597 行（`transform_id` 首/次分量等于
`s_key_a`/`s_key_b`、无退化 `s_key_a==s_key_b`、无重复无序分子对）后
orientation sign 恒为 +1。`row_lo/row_hi` 只是无序对 ID，无方向含义。

## 3. 阶段 A：Lipo 从头训练 H × eta 四臂

四个正式单元，全部 official Train3360 → Valid420，seed 0：

| ID | H | eta |
|---|---|---|
| `A_IS` | `H_IND` | `ETA_STAR` |
| `A_IJ` | `H_IND` | `ETA_FULL` |
| `A_FS` | `H_FULL` | `ETA_STAR` |
| `A_FJ` | `H_FULL` | `ETA_FULL` |

逐通道定义（binary `s,t∈{0,1}`，物理无向边只计一次）：

```
log p_i(s)  = log_softmax(u_i)(s)
H_FULL      = logZ − Σ_i logsumexp_s u_i(s)
H_IND       = Σ_(i,j) logsumexp_(s,t)[log p_i(s)+log p_j(t)+psi_ij(s,t)]
ETA_FULL_i  = posterior_logodds_i − [u_i(1) − u_i(0)]
m_(j→i)(s)  = logsumexp_t[log p_j(t)+psi_ij(s,t)]
ETA_STAR_i  = Σ_j [m_(j→i)(1) − m_(j→i)(0)]
```

H 的两个 readout 槽位（`H`、`H/(1+n_bonds)`）同步替换；eta 在 `eta_norm` **前**替换。
`A_IS` 不执行全图 belief；其余需 `H_FULL` 或 `ETA_FULL` 的条件每 batch 只执行一次
原 exact belief。保留断开分量与孤立原子，不删图、不近似 BP、不裁剪分子。

配方 FGCR-R0：AdamW，peak 3e-4，min 1e-5，wd 1e-4，batch 64，eval 256，warmup 5
epoch，clip 5，dropout 0.1，FP32，TF32 off，无 AMP，200 全 epoch 原 cosine，无早停，
MSE（均值/总体标准差仅由当前训练集拟合）。每 epoch 全量 Train/Valid eval；strict
min Valid RMSE 选 best，完全相同时取较早 epoch。保存 best/final 与逐 epoch 原始单位
RMSE/MAE；`last20` 指最后 20 个逐 epoch 指标的**均值**，不是预测 ensemble。
原模型参数布局 100,926；四臂共享同一随机初始 `state_dict`、同一 batch plan、同一
逐 epoch 分子顺序。不从历史 `best.pt` 微调。

报告：`g_eta^I = RMSE(A_IS) − RMSE(A_IJ)`、`g_eta^F = RMSE(A_FS) − RMSE(A_FJ)`、
`g_H^S = RMSE(A_IS) − RMSE(A_FS)`、`g_H^J = RMSE(A_IJ) − RMSE(A_FJ)`；eta 增益在
H_FULL 与 H_IND 之间的**中中差用 MSE** 报告（单次训练产物的描述性交互，不写成普遍
因果交互）。每项并列 best / 同步 epoch / final / last20；2000 次配对分子 bootstrap、
MAE、改善分子比例；共同最大平均平方误差 5% 剔除只作敏感性。

## 4. 冻结后续模型分支（只看 A 的分子结果，不看 D0 编辑结果）

σ_A = official Lipo Train 标签标准差 = **1.211539840098397**。
对每个固定 H，eta 学习候选信号需**同时**满足：

1. best 的 `g_eta ≥ 0.015 σ_A`（= 0.0181731）；
2. final 或 last20 至少一个同方向；
3. `FULL` 的 best MAE 不比 `STAR` 差超过 `0.01 σ_A`（= 0.0121154）。

`B_ref = min(可用且已重放或逐预测核验的 G0/Rf、RESP Valid RMSE)`；两者都不可核验则
`REFERENCE_BLOCKED`。

**ROOT**：A 四臂完整，至少一个固定 H 同时满足三条信号且其 `FULL` Valid RMSE
`≤ 1.10 B_ref`。
- 只一个 H 合格 → 用它；两个都合格 → 默认 `H_IND`，仅当 `H_FULL` 的 FULL RMSE 比
  `H_IND` 的 FULL 再低 `≥ 0.01 σ_A` 才选 `H_FULL`。
- 固定 D0 三行：`ROOT_STAR`、`ROOT_FULL`、`GINE_G0`。

**RESPONSE**：A 与参照完整可核验但不满足 ROOT 条件（含“有联合增益但仍明显不具
竞争力”）。固定 D0 三行：`RESP_PRODUCT`、`RESP_FUSION_MLP`、`GINE_G0`。
不通过新 loss 自动抢救 FGCR。

A 因工程/资产/预算不完整或参照为 `REFERENCE_BLOCKED` → `A_UNRESOLVED`，为继续 B/C
按预设工程回退走 RESPONSE，但**不得**解释为 FGCR 已被证明缺乏收益或不具竞争力。

两个 RESP 算子完全复用父模型，均 198,262 参数，`latent=16`：

```
PRODUCT     T = Σ_j w_j q_j k_j / sqrt(16),   q = tanh(U z_S), k = tanh(V z_C)
FUSION_MLP  T = Σ_j w_j tanh((U z_S + V z_C)_j / sqrt(2)) / sqrt(16)
s(C,S)      = b(C) + a(S) + T,   F(G) = 原固定 ≤4 selected views 的平均
                                   （无合法 view 用原 fallback head）
```

共享原 canonical 随机初始化与零初始化 `w`（hash `ccc8171d…072c`），不用监督
checkpoint。前几个优化步核验 `w` 确实离开 0。

**在写 `analysis/branch_decision.json` 之后才开始 D0 正式训练**；不因 D0 某个编辑
子集不漂亮而切换另一整套九臂。

## 5. 阶段 B：D0 三模型 × 三目标（九个正式单元）

|  | BASE | ENDPOINT | DELTA |
|---|---|---|---|
| 所选结构算子 1 | `B_1_B` | `B_1_E` | `B_1_D` |
| 所选结构算子 2 | `B_2_B` | `B_2_E` | `B_2_D` |
| `GINE_G0` | `B_G_B` | `B_G_E` | `B_G_D` |

全部从随机初值在 Fit2856 训练；Dev504 只作分子选模与最终响应评价。**绝不**使用
official Train3360 训练好的 checkpoint 作 D0 模型或初始化。配方：ROOT 行沿用
FGCR-R0（200 epoch、batch 64）；RESP 两行沿用 `rd_small_s0`（360 epoch、batch 128、
peak 1e-3、min 1e-5、wd 1e-5、warmup 5、clip 5、dropout 0.1、AdamW
betas(.9,.999)、eps 1e-8、原 cosine）；GINE 沿用 `Rf`（同 RESP 但用 `gr_common`
的 `peak*(0.1+0.9 s/W)` warmup）。无早停/EMA/SWA/soup/预训练/teacher；目标 scaler
仅用 Fit2856（mean 2.185815826311436，std 1.2139664740467941）。

损失唯一定义（`e_i = f_std(G_i) − y_std_i`，同一个 Fit σ）：

```
L0 = mean_base(e_train_i^2)
LE = 0.5 · E_q[e_eval_A^2 + e_eval_B^2]
LD = 0.5 · E_q[(e_eval_B − e_eval_A)^2]
BASE:     L = L0
ENDPOINT: L = (L0 + λLE)/(1+λ)
DELTA:    L = (L0 + λLD)/(1+λ)          λ = 0.5 固定
```

因此同一状态、同一辅助预测下 `L_DELTA − L_ENDPOINT = −λ/(1+λ)·E_q[e_A e_B]`
（`checks.json: target_identity_and_swap_invariance` 用 float64 恒等式核验）。

pair sampler：只用 Fit–Fit 1619 的 **340** 个 Fit-only 连通分量；先均匀抽分量、
再均匀抽该分量一条边，有放回，每基础步 64 对；`seed = 23017`；一条冻结 schedule
（步数上限 9000）写入 `analysis/pair_schedule.npz`，所有目标/家族取同一随机流前缀。
端点在一次辅助前向中去重，再用索引恢复 64 对及其重复权重。端点、y 与 sampler 状态
全部来自 Fit。基础 L0 仍只对原基础 batch 求均值。

辅助前向纪律：基础 batch 正常 `model.train()` 算 L0 → 辅助 pair forward 临时
`model.eval()`（保持 autograd，不用 `no_grad`/`detach`）→ 恢复原训练状态。
辅助前向不更新 BatchNorm running stats、不消耗 dropout RNG；ENDPOINT/DELTA 完全
相同。BASE 跳过辅助路径；已核验跳过不改变基础梯度、模型 buffer 或 RNG 状态。
基础与辅助加权后**一次** backward/clip/step。

报告边界：D−E 只识别既定严格编辑图上的残差交叉项收益；本轮无随机配对/随机编辑图
控制，不声称收益特异依赖化学编辑关系而非一般残差耦合；GINE 对照只检验“是否特定
结构模型专属”。

checkpoint 选择：每 epoch 在全部 Dev504 上按分子 RMSE 选 best，规则全臂一致；保存
final 与 last20 分子指标。**先冻结九个 best/final 的 SHA，再统一打开编辑评价**；
不按编辑 MAE 选 epoch、λ、架构或另挑 checkpoint。

## 6. 阶段 B 的评价与可复用响应证据

每个 best 与 final，在相同冻结 pair 集合上计算
`delta_hat = f(B) − f(A)`、`delta_y = y(B) − y(A)`（原始性质单位）。

主编辑指标：**component-equal edit_MAE**——先在 2133 边图每个评估分量内部对 pair 取
MAE，再对有评价 pair 的分量等权平均。同时报 pair-equal edit_MAE/edit_RMSE、真实
`|delta_y| ≥ 0.1` 子集上的方向正确率与子集大小（不单独凭方向正确率宣布成功）。

必要参照：
1. `ZERO`：`delta_hat = 0`。
2. `TRAIN_TRANSFORM_MEAN`：对每个评估 pair 只取同 `transform_id` 的 Fit–Fit 真实
   delta，**排除评估 pair 所在的整个 2133 边图分量**，先在支持分量内平均、
   再对支持分量等权平均；无其他分量支持时记 NA（不用 0 或全局均值填补）。
   公平比较只固定在 119 子集。
3. GINE 三目标结果。

不确定性与敏感性：2000 次 paired **component** bootstrap（rng 1703，同一比较共用
抽样；分量等权、不给 pair 当独立样本）；对 119/97/157 等子集只在该子集实际有
评价 pair 的非空分量上重采样并报实际分量数；分量数 < 30 的子组只作低支持描述；
额外列出剔除“两模型平均分量 MAE 最大”的那 1 个分量后的差值（不分别删各自最差）。

误差分解（不另训练，用 `raw residual e_i = f_raw(G_i) − y_raw_i`）：

```
E[(e_B − e_A)^2] = E[e_A^2] + E[e_B^2] − 2 E[e_A e_B]
```

按 pair-equal 与 component-equal 两种权重各报一次，另报两端均值偏差与中心化协方差；
只描述 delta 改善来自端点误差下降还是共同误差抵消，不把“残差更相关”命名为化学机制。

结构响应卡：全量 276 对保存原始 row ID、exact core/端口/变量片段、transform、分量、
Fit 支持、真实 delta、九个模型预测 delta 与误差。报告展示 12 例：6 个按 pair ID
预选、来自不同分量的 supported 案例；3 个晋级候选 D 相对 E 最大改善、3 个最大恶化
（按结果选出的诊断，不当总体证据；无晋级候选时按结构行 1 的 D/E 固定展示）。

## 7. 阶段 C：ESOL 三个固定 BASE 单元

使用第 4 节已冻结的同一三行模型，只训练 BASE。official Train902 / Valid113，seed 0，
Train-only scaler；各自沿用第 5 节的完整配方与 epoch 数；只按 ESOL 合法 categorical
schema 适配 embedding 表，匹配算子两臂相同参数布局与初始化；不从 Lipo 监督权重
迁移。保留原图全部分量与原始属性。exact 推断先依 Train/Valid 结构生成计划，
`smax` 取实际所需最大 clique size、最高允许 9（induced width ≤ 8），不因默认
`smax=4` 静默截断或删图；超限则 ROOT 的 ESOL 配对整体标
`TOPOLOGY_OR_BUDGET_BLOCKED`，不只跑便宜一臂。不构建 ESOL 编辑目录、不扫另一 λ、
不延伸 FreeSolv/HIV。

## 8. 条件分支 R：一个精确半径 2 模型

仅 ROOT 分支触发，优先级低于完整 D0 九臂与 ESOL 三臂，预算目标 ≤ 2 GPU·h。

保持第 4 节所选 H、所有潜势/参数/读出/200 epoch 配方，与 A 对应的 STAR/FULL 可比，
只把 eta 换成

```
ETA_R2(i) = 在完整诱导半径 2 球 B2(i) 上的精确 posterior_logodds_i − unary_logodds_i
```

保留球内全部真实边；球外不接入势或消息；内部原子/键的原始 OGB 特征与势参数照原图
使用，不在裁剪后重算 degree/ring 或其他属性；每个 root 只取该 root 节点结果。可缓存
无标签拓扑/batch 计划/gather 索引，但每个 step 必须对当前势值计算并反传。不用有限
步 BP 替代。

判读：`R2_PRACTICALLY_CLOSE` 当 best 的 `|RMSE_R2 − RMSE_FULL| ≤ 0.01σ_A` 且
final/last20 两项 R2 的劣势均不超过 `0.02σ_A`（预设实用接近标准，非统计等价）。
完整 200 epoch 投影不满足预算则跳过，不缩短 R2 后与完整 FULL 比较。

## 9. 条件分支 P：唯一达标目标接回 official Lipo

优先级高于 R；最多一个模型家族/算子，最多三个正式单元，目标 ≤ 1.5 GPU·h。
对 D0 每行定义 `gain_E_D = CE_E − CE_D`、`gain_B_D = CE_BASE − CE_D`（各自按分子
RMSE 选中的 best checkpoint）。晋级需**同时**满足：

1. `gain_E_D ≥ max(0.015 σ_Fit, 0.05 · CE_E)`；
2. `gain_B_D ≥ max(0.015 σ_Fit, 0.05 · CE_BASE)`（D 必须对原训练目标产生净收益）；
3. pair-equal edit_MAE 中 D 同时不劣于 E 与 BASE；
4. D 的 Dev 分子 RMSE `≤ 1.03 ×` BASE 的 Dev RMSE，且 MAE 不比 BASE 差超过
   `0.01 σ_Fit`；
5. best 的 supported119 CE_edit_MAE_D ≤ CE_edit_MAE_E，且 E-final 主 CE 减
   D-final 主 CE 非负。

σ_Fit = **1.2139664740467941**。阈值是预设资源门槛，不改。若一个共同分量剔除后完全
反向，标 `FRAGILE`。多行合格按 `(CE_E − CE_D)/CE_E` 最大者选唯一行；相差 ≤0.01
时优先 D 分子 RMSE 更低者，再相同选实测推理更快者。允许 GINE 胜出。所选模型用
official Train3360 从随机初值重跑 BASE/ENDPOINT/DELTA 完整配方，pairs 为全部 2133
official-Train 对，q 用其 391 分量，目标/scaler 按 official Train 重算；三臂在同一
Valid420 按分子 RMSE 选模。若原阶段 A 已存在严格相同 BASE 则可复用并记录同一性。
P 主要评价整分子性能；official Valid–Valid 严格编辑只有 6 对，不作新的响应确认集。

## 10. 工程核验（正式训练前）

见 `code/checks.py` 与 `analysis/checks.json`：数据 SHA/映射/互斥/Fit-only 标准化/
Test 不进入 forward；catalogue 定义与 2133/1619/238/276 及 119/97 支持复算；FGCR
四臂参数量与初始 `state_dict` 逐位相同、batch plan 相同、FULL 重放与
0.7128937193209034 一致；极小图 float64 brute force 核对 H/eta、零 coupling 为 0、
单边/互不相交单边 FULL=local、含相邻边的链/环可不相等；原子置换与键反向/顺序不变、
单图与 batch 一致、孤立原子与多分量保留；belief 次数符合定义；梯度有限且关键
潜势/encoder/head 实际更新；目标恒等式与端点交换不变；辅助端点严格 Fit-only；
E/D 完全相同 pair plan；辅助 eval forward autograd 有效、BN buffer 与训练 RNG 不被
修改、恢复 train 状态正确；checkpoint 重载复现；指标原始单位与 scaler 一致。

数值容差按 dtype 与运算规模设置；关键恒等式用 float64。不因 float32 求和舍入反复
重启正式训练。真正改变科学定义的修复须标明，并使受影响的配对单元从相同初值统一
重跑；所有失败成本计入预算。

## 11. 调度、预算与不完全任务

软包络 14 GPU·h（硬上限 16）：工程/预跑 0.5、A 2.0、B 7.0、C 1.0、P 1.5、R 2.0。
两卡并行 1 h 计 2 GPU·h；预跑、失败、重启、训练、GPU 分析都计入。
正式训练默认 16 个配置（A4 + B9 + C3），P ≤ 3、R ≤ 1。墙钟目标 7–8 h，7 h 15 后
不再启动新训练。预算紧时按顺序砍 R、P，再砍整个 ESOL 包；不削减 D0 主矩阵的关键
E 控制；不事后降低 epoch、减少数据、换精度或用半程结果冒充阴性证据。

每次启动保证整个匹配单元能在保留窗口前结束：A 四臂是一个核心组；D0 每个模型的
B/E/D 是一个三臂组；ESOL 结构算子两臂必须一起可完成；P 的 E/D 必须匹配完整。

## 12. 预设路线判读（填写证据，不临场发明后续模型）

Q1 `H_FULL`/`ETA_FULL` 是否优于匹配 STAR/IND、H 与 eta 是否配合；Q2 差异是否值得成本、
是否接近应用参照；Q3 同样端点暴露下 DELTA 是否额外优于 ENDPOINT；Q4 结构模型收益
是否超过 GINE 也能取得的收益、还是训练目标更关键；Q5 整分子性能与真实编辑响应能否
同时保留、119/97 支持组是否支持复用；Q6 ESOL/R2 是否支持原路线方向、未执行部分
明确保留未知。

性能位置在同一数据切分内用本轮 GINE 及可用强参照计算误差比：≤1.05 接近，
1.05–1.10 有代价，>1.10 明显差距（资源标准，不是公认 SOTA 门槛）。

最终建议只能取：`PRIORITIZE_JOINT_ROOT_RESPONSE`、`PRIORITIZE_FINITE_CONTEXT_RESPONSE`、
`PRIORITIZE_FRAGMENT_CONDITIONAL_RESPONSE`、`PRIORITIZE_RESPONSE_ALIGNED_OBJECTIVE`、
`NO_CANDIDATE_WITH_BOTH_PERFORMANCE_AND_RESPONSE_ADVANTAGE`、`UNRESOLVED`。
允许 `UNRESOLVED`，不强推冠军；不给出新潜势/新网络/新增数据集的自动续跑方案。

## 13. 交付

`PROTOCOL.md`、`configs/experiment.json`、`code/`、`analysis/{branch_decision,support,checks}.json`、
`main_molecule_table.csv`、`edit_response_table.csv`、`paired_contrasts.csv`、
`predictions/`、`curves/`、`checkpoints/manifest.json`、`resource_ledger.json`、`REPORT.md`。
