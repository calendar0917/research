# DSCR v0 执行报告 — Direct Structural Computation and Recurrence

- Round：`tracks/structural-sufficiency/direct_structural_computation_recurrence_v0/`
- 协议：`PROTOCOL.md`（在任何正式训练前冻结，commit `ffe232c7`）；协议 id `dscr-v0`
- 数据集：`ogbg-molpcba` 官方 scaffold split（Train 350,343 / Valid 43,793 / Test 43,793，128 任务，masked BCE，官方 OGB macro AP）
- Test 角色：`DEVELOPMENT_TEST / EXPOSED_TEST_ENGINEERING_COMPARISON_SET`。每个 epoch 全量评分，**但从不参与** checkpoint、horizon、配置或任何选择；本轮不构成盲测确认。
- T0 = `2026-11-25T00:33+08:00`；首个正式训练 `01:15:34`；最后一个正式训练结束 `06:23:56`；停止新训练上限 `07:48`；硬截止 `08:33`。**未触发任何时限降级。**
- 共同 horizon：**H = 25**（8 单元计划，由 `analysis/preflight_timing.json` + `analysis/schedule.json` 仅按计时冻结，见 `configs/horizon.json`）。
- `C* = FIXED1`，`K* = SELF`（按协议 §7：seed-0 四候选的最高官方 Valid macro AP，差距 >0.001 无需 tie-break；只看 Valid）。
- 本轮新训练 GPU·h ≈ **9.6**（8 个单元；目标 ≤14、硬上限 16）；含预检/检查/诊断约 **0.3**。

---

## 第一页：十个问题与路线判定

**P1｜新的点式深度 SELF 是否优于 W2？**
方向一致为正，但未达预注册门槛：SELF−W2 = **+0.0040**（seed 0，CI 含 0）与 **+0.0106**（seed 1）；协议要求两个配对 seed 均 ≥+0.005，故 `WEAK_OR_UNCONFIRMED_LOCAL_COMPUTATION_SIGNAL`。

**P2｜固定来源一跳计算 FIXED1 是否优于 SELF？**
是。FIXED1−SELF = **+0.0065** Valid（种子 0，任务 bootstrap CI 不含 0，127 任务中 79 改善/48 退化）；seed 1 配对 C*−K* = **+0.0055** Valid / **+0.0081** Test。→ `FIXED_SOURCE_COMPUTATION_SIGNAL`。

**P3｜FIXED1 是否达到迭代参照 DYNAMIC 的水平？**
是。两 seed 的 Valid 差仅 **+0.0008**（seed 0）与 **+0.0001**（seed 1），远小于 seed 噪声（W2 的 seed 1−0 = −0.0082）。FIXED1 以 1,128,468 参数、只读 `h0` 就复现了 DYNAMIC 的 Valid 水平。

**P4｜真实两跳结构计算 REAL2 是否优于 FIXED1？**
否，且方向相反：REAL2−FIXED1 = **−0.0067** Valid（CI 不含 0；80/127 任务退化）、−0.0015 Test。→ `NO_TWO_HOP_GAIN`。

**P5｜无绑定多重集组合 BAG2 是否优于 FIXED1？**
否：BAG2−FIXED1 = **−0.0051** Valid（CI 含 0 但整体为负），Test +0.0013（CI 含 0）。BAG2 略优于 REAL2（+0.0016 Valid），说明"去掉对应关系"不亏。

**P6｜真实分支对应关系（REAL2 vs BAG2）是否有预测价值？**
否。REAL2−BAG2 = **−0.0016** Valid（78/127 退化）、**−0.0029** Test。冻结权重替换进一步显示：把 REAL2 的读出换成 BAG2 的配对读出（−0.0001）或反向（+0.0002）几乎不改变 AP。→ `BINDING_NO_PREDICTIVE_VALUE`。

**P7｜迭代 DYNAMIC 相对直接两跳 REAL2 是否仍有优势？**
是：DYNAMIC−REAL2 = **+0.0059** Valid、**+0.0068** Test。即"用两跳结构替代迭代"的假设不成立。

**P8｜等预算、同信息契约下 C\* 与 DYNAMIC 谁更优、是否跨 seed 可重复？**
Valid 上两者实质持平（+0.0008 / +0.0001，两 seed 同向），**Test 上出现跨 seed 符号冲突**（seed 0 DYNAMIC +0.0052，seed 1 C\* +0.0065）→ `COMPETITIVE_NONITERATIVE_COMPUTATION` + `SPLIT_CONFLICT`。

**P9｜增益是否广泛，而不是由少数低支持任务驱动？**
是广泛的：高支持任务（Valid 正负例均 ≥20，107 个）上 C\*−W2 = +0.0119/+0.0120（两 seed），REAL2−FIXED1 = −0.0051，去掉 |Δ| 最大的 3 个任务后结论不变（−0.0073）。低支持任务对 macro 的净贡献很小（|贡献| ≤ 0.006）。

**P10｜机制上"为什么"，以及路线建议？**
冻结权重诊断给出清晰答案：**承重的是"读固定来源 h0 的一跳聚合 + 归一化方式"，不是分支对应关系**。BAG2 权重 + REAL2 读出（无 degree 归一化）= +0.0002；同一替换改成 degree 均值 = **−0.0161**；γ（两跳上下文强度）从 0.5 降到 0 掉 0.0028、升到 1 掉 0.0036~0.0045。→ 建议继续**固定来源一跳**路线（`RECOMMEND_CONTINUE`），两跳与多重集路线 `NO_NEW_COMPETITIVE_DIRECT_MECHANISM`。

### 路线判定表

| 路线 | 机制角色 | 支持证据 | 反对证据 | 未决问题 | 判定 |
|---|---|---|---|---|---|
| 严格直接局部计算（W2 / SELF） | 第二阶段完全不读其他节点隐状态 | SELF−W2 Valid +0.0040（s0）/ +0.0106（s1）；Test +0.0031 / +0.0059 | s0 未达预注册 +0.005 门槛；SELF 仍对每条真实入边条件化，不是纯点式深度 | 若要更高点式容量应另立容量对照 | `WEAK_OR_UNCONFIRMED_LOCAL_COMPUTATION_SIGNAL` |
| 固定来源一跳计算（FIXED1） | 第二阶段读真实 1-hop 邻居的 `h0`，绝不读 `h1[j]` | FIXED1−SELF +0.0065（s0，CI 不含 0）；C\*−K\* +0.0055/+0.0081（s1）；C\*−W2 +0.0106 / +0.0160；与 DYNAMIC 差 +0.0008 / +0.0001 | 对 SELF 的 Test 优势仅 +0.0003（s0）；C\*−DYNAMIC 的 Test 两 seed 反号 | 仅 2 个 seed，无第三 seed | **`FIXED_SOURCE_COMPUTATION_SIGNAL` → `RECOMMEND_CONTINUE`** |
| 真实两跳结构组合（REAL2） | 从 `h0` 直接做带根两跳分支计算 | DYNAMIC−REAL2 +0.0059/+0.0068（两跳未替代迭代） | REAL2−FIXED1 −0.0067（CI 不含 0）；REAL2−BAG2 −0.0016 | 仅 1 个 seed；冻结替换已显示对应关系不可识别，故未追加 seed | `NO_TWO_HOP_GAIN → NO_NEW_COMPETITIVE_DIRECT_MECHANISM` |
| 无绑定多重集组合（BAG2） | 同样两个多重集，去掉对应关系 | 绝对 Valid 0.2707，高于 REAL2、低于 FIXED1；最佳 epoch 22、Train AP 最低（更早收敛/更强正则） | BAG2−FIXED1 −0.0051 Valid；Test +0.0013（CI 含 0）；degree 均值替换 −0.0161 | 诊断控制臂，不作候选架构 | `NO_TWO_HOP_GAIN → NO_NEW_COMPETITIVE_DIRECT_MECHANISM` |
| 迭代隐状态更新（DYNAMIC） | 第二阶段读更新后的邻居状态 `h1[j]` | Test 侧 seed 0 +0.0052 vs FIXED1；vs REAL2 两 split 均 +0.006 | 与 FIXED1 的 Valid 差两 seed 均 ≈0；Test 优势在 seed 1 反向（C\* 领先 +0.0065） | 迭代优势与"更宽有效访问范围"不可分离（本轮无"读 h1 但非迭代"臂） | `COMPETITIVE_NONITERATIVE_COMPUTATION`（+`SPLIT_CONFLICT`） |

---

## 1. 执行概览

| 阶段 | 内容 | 状态 |
|---|---|---|
| 复用/冻结 | W2 seed-0（`lcrc_pcba_backbone_scaling_v1/runs/W2`）、DYNAMIC seed-0（`lcrc_pcba_dynamic_v1/runs/W2_DYNAMIC`）只读复用，不重训、不作初始化 | 完成 |
| A 检查 | `analysis/checks.json`：**17/17 PASS**，含数据身份、参数量、初始化同构、epoch-0 等价、方向/编号/顺序不变性、批次隔离、度数统计、真实度数、seed-1 plan 冻结、plan 身份 | 完成 |
| 计时预检 | `analysis/preflight_timing.json`（逐臂 p50 step/eval/checkpoint，双卡调度模拟）、`analysis/schedule.json`、`configs/horizon.json` | 完成 |
| C seed-0 | FIXED1 / REAL2 / SELF / BAG2 四个新臂，各 25 epoch 全量 Train/Valid/Test 评价 | 完成 4/4 |
| D seed-1 | W2 s1、DYNAMIC s1、C\* s1（FIXED1）、K\* s1（SELF） | 完成 4/4 |
| E 追加 | 未执行（见 §9） | 未执行 |
| F 主表/逐任务 | `analysis/main_table.csv`、`per_task_metrics.csv`、`paired_comparisons.json`、`summary.json` | 完成 |
| G 机制诊断 | `analysis/mechanism_diagnostics.json`（读出替换、γ 探针、3 个可追溯分子） | 完成 |

**时间线（本地 +08）**：00:33 T0 → 01:15:34 首个正式训练 → 02:28 首批完成 → 03:50 第二批 → 05:08 DYNAMIC s1 → 06:06 C\* s1 → 06:23 K\* s1 → 06:43 分析 → 06:55 报告。8 单元实际占用 **4.75 h** 墙钟（预检预测 4.76 h）。

### 失败与修复（全部留痕，不重跑 epoch）

| 单元 | 症状 | 原因 | 处理 |
|---|---|---|---|
| FIXED1 s0（首次 02:28） | 末尾自检 `best-checkpoint Valid AP 0.2758469983 != recorded 0.2758455214`（差 1.5e-6） | 第二阶段 `index_add_` 走 CUDA 原子加，求和顺序不固定；同一权重重复评分会差 1e-6~1.4e-5 AP | 把自检阈值从 1e-6 放宽到 1e-4（并记录实测差值），从 `checkpoint_last.pt` 以"无可训练 epoch"的修复路径重写 `run.json`；**未重训任何 epoch** |
| W2 s1（首次 04:54） | 同上，差 1.4e-5 | 同上 | 同处理后 COMPLETED |

两次失败记录保留为 `runs/<unit>/run.failed.json`，`run.json` 内含 `resume_repair`（含首次失败原因、恢复的曲线行数与 best 记录）。**结论：AP 的可复现带为 ~2e-5 量级，远小于本轮讨论的 1e-3~1e-2 效应。**

## 2. 信息边界与资产复用

- 四个新臂 = 冻结历史 W2 主干 + `bond_proj: Linear(13,128)` + 两个 `MessageFFN(128/256)`；参数量全部 **1,128,468**（`A2.2`），W2 为 994,324。
- 私有 RNG 初始化、`fc2` 零初始化：epoch-0 与 W2 **logits 等价**（A2.4 断言 + 各 run 记录：SELF 0.029063365 / FIXED1 0.029063280 / REAL2 0.029063336 / BAG2 0.029063179 vs W2 参考 0.029063332）。
- 顺序计划：seed-0 用历史 W2 plan（`global_plan_sha256 = 06f437…b6f6`，A2.11/A2.12 逐字节一致），seed-1 用 `configs/plan_seed1.json`（`0af6ec…e0`，在任何 seed-1 训练前冻结）。
- 每 epoch 训练/评价 RNG 隔离逐次断言；Test 每 epoch 全量评分但只写入曲线与 `*_logits_best.npy`，从不参与选择（`run.json: test_used_for_selection=false`）。
- 真实度数（`A2.13`）：Train 平均度 **2.157**、最大 5；BAG2 配对/边比 **2.394**，每 batch128 实际 **16,981** 个配对（协议预算 11,955，实测高 42%，仍在吞吐预算内）。

## 3. 主结果

见 `analysis/report_tables.md` 表 1（由 artefacts 直接渲染）。摘要（官方 macro AP，主 checkpoint 由完整 Valid AP 选出）：

| 臂 | seed | best ep | Valid AP | Test AP |
|---|---:|---:|---:|---:|
| W2（历史） | 0 | 24 | 0.265273 | 0.253616 |
| SELF | 0 | 25 | 0.269306 | 0.256719 |
| FIXED1 | 0 | 25 | **0.275847** | 0.257001 |
| REAL2 | 0 | 25 | 0.269148 | 0.255474 |
| BAG2 | 0 | 22 | 0.270738 | **0.258324** |
| DYNAMIC（历史） | 0 | 25 | 0.275064 | **0.262226** |
| W2 | 1 | 23 | 0.257058 | 0.246275 |
| DYNAMIC | 1 | 19 | 0.272981 | 0.253803 |
| C\* = FIXED1 | 1 | 19 | **0.273104** | 0.260301 |
| K\* = SELF | 1 | 24 | 0.267630 | 0.252209 |

所有 AP 来自同一次（重新）评分流程，与训练期记录的逐 epoch 值相差 ≤3e-6。

## 4. 预注册配对对照

完整 16 行（含任务 bootstrap、分子 bootstrap、高支持、去 top-3 敏感性）见 `analysis/report_tables.md` 表 2 与 `analysis/paired_comparisons.json`。关键行：

| 对照 | Valid Δ [95% CI] | Test Δ | 改善/退化 (127) | 高支持 Δ |
|---|---|---|---|---|
| SELF−W2 (s0) | +0.00403 [−0.0032, +0.0109] | +0.00310 [−0.0035, +0.0090] | 78/49 | +0.00603 |
| SELF−W2 (s1) | +0.01057 [+0.0042, +0.0177] | +0.00593 [+0.0013, +0.0105] | 77/50 | +0.00603 |
| FIXED1−SELF (s0) | +0.00654 [+0.0012, +0.0119] | +0.00028 | 79/48 | +0.00589 |
| C\*−K\* (s1) | +0.00548 [+0.0007, +0.0102] | +0.00809 [+0.0013, +0.0148] | 80/47 | +0.00595 |
| C\*−W2 (s0) | +0.01057 [+0.0037, +0.0175] | +0.00339 [−0.0053, +0.0115] | 93/34 | +0.01191 |
| C\*−W2 (s1) | +0.01605 [+0.0099, +0.0224] | +0.01403 [+0.0076, +0.0203] | 85/42 | +0.01197 |
| REAL2−FIXED1 (s0) | −0.00670 [−0.0125, −0.0005] | −0.00153 [−0.0066, +0.0031] | 47/80 | −0.00514 |
| BAG2−FIXED1 (s0) | −0.00511 [−0.0115, +0.0010] | +0.00132 [−0.0044, +0.0069] | 61/66 | −0.00406 |
| REAL2−BAG2 (s0) | −0.00159 [−0.0063, +0.0034] | −0.00285 [−0.0076, +0.0013] | 49/78 | −0.00107 |
| DYNAMIC−FIXED1 (s0) | −0.00078 [−0.0067, +0.0051] | +0.00522 [+0.0015, +0.0090] | 51/76 | −0.00066 |
| DYNAMIC−REAL2 (s0) | +0.00592 [−0.0003, +0.0123] | +0.00675 [+0.0018, +0.0122] | 66/61 | +0.00448 |
| C\*−DYNAMIC (s0) | +0.00078 [−0.0053, +0.0063] | −0.00522 [−0.0092, −0.0015] | 76/51 | +0.00066 |
| C\*−DYNAMIC (s1) | +0.00013 [−0.0046, +0.0050] | +0.00650 [+0.0019, +0.0121] | 58/69 | +0.00070 |
| **seed 噪声** W2 s1−s0 | −0.00822 [−0.0153, −0.0017] | −0.00734 [−0.0158, −0.0003] | 54/73 | −0.00203 |
| **seed 噪声** DYNAMIC s1−s0 | −0.00208 [−0.0097, +0.0053] | −0.00842 [−0.0191, −0.0001] | 62/65 | −0.00201 |

要点：

1. **seed 噪声与效应同量级。** W2 的两个 seed 相差 0.008 Valid / 0.007 Test。因此所有"路线"判断都改用**同 seed、同 plan 的配对**（C\*−W2、C\*−K\*、C\*−DYNAMIC），而不是跨 seed 的绝对值比较。
2. **配对后 C\*−W2 在两个 seed 上都是 +0.010~+0.016，且 CI 不含 0**，这是本轮最稳的结论。
3. **两跳/多重集没有带来任何正收益**，REAL2 在 80/127 个任务上比 FIXED1 更差，且不是少数任务所致（去 top-3 后 Δ 仍为 −0.0073）。
4. **DYNAMIC 的 Valid 优势消失**（−0.0008/+0.0001），只在 Test 的 seed 0 上留下 +0.0052，而 seed 1 反向 −0.0065 → 记为 `SPLIT_CONFLICT`，不能作为迭代优势的证据。

## 5. 任务级证据

`analysis/per_task_metrics.csv` 给出每个单位×split×任务的 AP、正负例数、观测数与支持分层。要点：

- FIXED1 相对 SELF 的改善集中在高支持任务（+0.0059，79/127 改善），低支持任务净贡献 +0.0016。
- REAL2 相对 FIXED1 的退化也是广谱的（80/127），不是个别任务异常。
- Test 上 FIXED1/SELF 的差距几乎消失（+0.0003），说明"一跳相对点式"的收益在 Valid 上更清楚；而 C\*−K\* 在 seed 1 的 Test 上仍 +0.0081，跨 split 方向一致。

## 6. 机制诊断（冻结权重，`analysis/mechanism_diagnostics.json`）

复现性：`own` 读出重算与记录值相差 ≤1.9e-6（gate 1e-4），因此下列替换可信。

| 权重 | 第二阶段算术 | Valid AP | Δ vs 自身 |
|---|---|---:|---:|
| BAG2 | 自身配对读出（交叉积 ÷deg） | 0.270736 | — |
| BAG2 | 换成 REAL2 单跳读出（sum） | 0.270955 | **+0.00022** |
| BAG2 | 换成 REAL2 读出但做 degree 均值 | 0.254625 | **−0.01611** |
| REAL2 | 自身两跳分支读出 | 0.269147 | — |
| REAL2 | 换成 BAG2 配对读出 | 0.269017 | **−0.00013** |
| REAL2 | γ=0（两跳上下文关闭） | 0.266322 | −0.00283 |
| REAL2 | γ=1 | 0.264674 | −0.00447 |
| BAG2 | γ=0 | 0.267885 | −0.00285 |
| BAG2 | γ=1 | 0.267090 | −0.00365 |

结论：

1. **对应关系在冻结权重下不可识别**（±3e-4），与独立训练的 REAL2−BAG2 = −0.0016 方向一致（去掉对应关系不亏）。这不是"对应关系被使用但无增益"，而是**两跳配对结构本身没有被训练成承重结构**。
2. **真正承重的是读出归一化**：同样的读出改成 degree 均值后掉 0.016，是所有干预里最大的单项效应。这解释了为什么 FIXED1（单跳 + sum）与 DYNAMIC 同级，而 REAL2/BAG2（两跳 + 归一化/交叉积）更差。
3. **两跳上下文被弱使用**：γ 从 0.5 降到 0 只掉 0.0028，升到 1 掉 0.0036~0.0045，说明 γ=0.5 附近是平台而非尖锐最优，上下文贡献约 0.003 且不随强度单调增长。
4. `R1 = RMS(h1−h0)/RMS(h0)`、`R2 = RMS(h2−h1)/RMS(h1)`（末 epoch，诊断 batch）：SELF 0.615/0.575、FIXED1 0.648/0.510、REAL2 0.635/0.536、BAG2 0.637/0.531 — 四个新臂的第二阶段都在做与第一阶段同量级的更新，不存在"第二阶段空转"。
5. 可追溯分子（`trace`，任务 31，Valid）：

| 分子 | 类别 | Δlogit(REAL2−BAG2) | 原子数 | 最大度 |
|---|---|---:|---:|---:|
| `CC[C@H](C)[C@H](NC(=O)…)…`（肽类） | REAL2 胜 | +17.58 | 332 | 3 |
| `NC(=O)C1=CN(C2OC(COP(=O)([O-])OP(=O)([O-])OC[C@H]3O…)…`（核苷酸类） | BAG2 胜 | −12.64 | 48 | 4 |
| `CCOC(=O)c1oc2nc3c(cc2c1N)COC(C)(C)C3` | 近平局（高支持度） | +0.05 | 21 | 4 |

每个分子的 `top_atoms`/`top_branches` 记录了贡献最大的 3 个根原子及其 3 条分支（来源原子、内层原子、元素符号、度数、范数），可追溯到具体邻居。差异最大的两个分子都是大分子（332/48 原子），提示两跳差异集中在多分支大分子上，而非普遍机制。

## 7. 路线判定

`analysis/summary.json` 的 `q6_routes` 按协议 §9 的五个条件逐条给出：

| 路线 | 有增益 | 跨 seed 可重复 | 非少数低支持任务 | 信息边界清晰 | 结论 |
|---|---|---|---|---|---|
| 固定来源一跳（FIXED1） | +0.0106 / +0.0160 vs W2；+0.0065 / +0.0055 vs 同容量兄弟 | 是（2 seed 同向） | 是（高支持 +0.0119/+0.0120） | 是（A3 梯度支持表：只读 h0） | **`RECOMMEND_CONTINUE`** |
| 真实两跳（REAL2） | 否（−0.0067 vs FIXED1） | 否（仅 1 seed） | 否（80/127 退化） | 是 | `NO_NEW_COMPETITIVE_DIRECT_MECHANISM` |
| 无绑定多重集（BAG2） | 否（−0.0051 vs FIXED1） | 否 | 否 | 是（诊断臂） | `NO_NEW_COMPETITIVE_DIRECT_MECHANISM` |
| 迭代（DYNAMIC） | 对 FIXED1 无（−0.0008/+0.0001） | 部分（Test 反号） | 否 | 是（读 h1） | `NO_NEW_COMPETITIVE_DIRECT_MECHANISM`（作为"新路线"）；仍是能力参照 |
| 点式深度（SELF） | 弱（+0.0040） | 是但未达门槛 | 是 | 是 | `WEAK_OR_UNCONFIRMED_LOCAL_COMPUTATION_SIGNAL` |

**建议的下一步（不属于本轮结论）**：把"固定来源一跳 + sum 读出"作为新的基线机制继续（它用 1.13M 参数、只读 h0、无迭代，就在 Valid 上匹配 DYNAMIC）；若要检验迭代是否真有独立价值，需要新增"读 h1 但只做一跳、且语义上非迭代"的臂，把"迭代"与"更宽访问范围"分开。两跳/对应关系路线在现有证据下不建议追加预算。

## 8. 预算与 GPU·h 台账

`results/ledger.csv` 与 `results/manifest.json` 记录逐单元墙钟、GPU·h、best epoch、SHA256 与来源（本轮/历史）。

| 单元 | GPU·h | 备注 |
|---|---:|---|
| SELF s0 | 1.179 | |
| FIXED1 s0 (C\*) | 1.231 | 含首次失败尝试的手工累计（4372 s） |
| REAL2 s0 | 1.216 | |
| BAG2 s0 | 1.259 | 与 DYNAMIC s1 有 ~5 min 同卡重叠 |
| W2 s1 | 1.081 | 含首次失败尝试 3835 s |
| DYNAMIC s1 | 1.245 | |
| FIXED1 s1 (C\*) | 1.186 | |
| SELF s1 (K\*) | 1.178 | |
| **小计（新训练）** | **9.58** | 目标 ≤14、硬上限 16 |
| 预检/检查/诊断/分析 | ~0.3 | 计时预检 3 次、全量检查 2 次、机制诊断 2 次、历史只读评分 2×3 split |
| **总计** | **≈9.9** | |

历史 W2 seed-0（1.395 GPU·h）与 DYNAMIC seed-0（1.734 GPU·h）为复用资产，不计入本轮消耗。墙钟：训练 4.75 h（01:15:34→06:23:56，含两次修复），全轮 00:33→约 07:00，在 7 h15 min 停止新训练与 8 h 硬截止内。

## 9. 未执行项与偏离

- **Phase E 未执行**：协议 §8 明确"预算富余本身不是新增实验的理由"，且 06:23 之后剩余墙钟（~85 min）不足以再完成一对 25-epoch 单元而不侵占 45 min 报告预留（07:03 起）。`EXPECTED`：无 seed-2、无 40-epoch 成对延长、无 S3 复用。
- **H = 25 而非 40**：计时预检显示 8 单元 × 40 epoch 需 7.62 h（含 15% 余量后不可行），30 epoch 需 5.72 h（>5.17 h 上限），25 epoch 需 4.76 h 可行。因此按协议降级到 H=25；**从未用精度信息改变 H**。历史 W2/DYNAMIC 的 best epoch（24/25）恰好落在 25-epoch 前缀内，前缀内比较成立。
- **两处自检阈值调整**（1e-6 → 1e-4）与两次 `resume` 修复记录在上文 §1，均在结果产生前后一致执行，未修改任何训练超参、数据或计划。
- **BAG2 的 step 时间**在 run 内为 66.5 ms（与 DYNAMIC s1 同卡重叠 5 min），干净预检值为 39.84 ms p50；报告计算量时以预检为准。

## 10. 不可声称的结论

- 不能说"迭代更新对 PCBA 无用"：本轮只是显示**固定来源一跳在 Valid 上追平 DYNAMIC**，而 DYNAMIC 的 Test 侧优势在 seed 1 反向，样本只有 2 个 seed。
- 不能说"两跳结构在任何设置下都无用"：REAL2 只有 1 个 seed，且其 γ 与读出归一化是预注册固定值；机制诊断只说明**这套两跳配对在当前容量与配方下没有被训练成承重结构**。
- 不能把 Test 当作盲测：Test 每 epoch 都被评分（协议要求），因此任何 Test 数字都是开发集数字。
- 不能把任务 bootstrap 当作 seed 方差：它度量的是任务组成敏感性；seed 方差由 W2/DYNAMIC 的 2 seed 直接给出（0.008 / 0.002 Valid）。

## 11. 产物与复现

| 路径 | 内容 |
|---|---|
| `PROTOCOL.md` | 冻结协议（训练前 commit） |
| `configs/horizon.json`、`configs/plan_seed1.json` | H=25 冻结、seed-1 顺序计划冻结 |
| `code/ds_common.py`、`ds_model.py`、`ds_train.py`、`ds_checks.py`、`ds_preflight.py`、`ds_analyze.py`、`ds_diagnose.py`、`ds_report.py`、`ds_manifest.py` | 全部本轮代码 |
| `analysis/checks.json` | 17/17 检查（含 A2.13 真实度数） |
| `analysis/preflight_timing.json`、`schedule.json` | 计时与调度模拟（H 的唯一依据） |
| `analysis/main_table.csv`、`per_task_metrics.csv` | 主表与逐任务指标 |
| `analysis/paired_comparisons.json` | 全部预注册配对（含任务/分子 bootstrap） |
| `analysis/mechanism_diagnostics.json` | 冻结读出替换、γ 探针、3 个可追溯分子 |
| `analysis/summary.json`、`report_tables.md` | 判定标签、Q6 路线建议、报告表 |
| `results/` | 逐单元曲线/逐任务/`run.json`、`ledger.csv`、`manifest.json`（路径 + SHA256 + commit） |

复现：`analysis/checks.json` 内所有检查可重跑（`--mode full --device cpu`）；主表与对照由 `ds_analyze.py` 从 `runs/*/run.json` + `best_checkpoint_eval.json` 重算；checkpoint 与预测矩阵（每单元约 40 MB）按仓库规范留在 `res` 主机的 `runs/`，SHA256 见 `results/manifest.json`。
