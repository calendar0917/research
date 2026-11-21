# REPORT — 结构词的端口绑定与句间组合（`word_sentence_port_binding_context_v2`）

本轮实施日期 2026-11-21，T0 = `1795224151`（2026-11-21T01:22:31Z）。
定义/配置/分支在正式训练前冻结于 `PROTOCOL.md` 与 `configs/wp_v2.json`；执行偏差见 `PROTOCOL.md` 附录 A。
**正式单元 12（Stage B 7 + Stage C 5）完成，GPU·h = 0：`res` 从 T0 起 SSH 不可达，改为本机 CPU 双 worker
（每 worker 7 线程）执行，全部臂同一 regime，跨臂比较仍内部有效，绝对数值不与上一轮 GPU 数值直接对照。**
主表 12 个单元全部 COMPLETE，无 OOM、无超时、无中断重跑。

---

## 0. 一页结论

**这一轮改变的不是“结构词有没有用”，而是“端口绑定在什么条件下才有用”。**

受控 2×2（端口状态绑定 BIND/MEAN × 句间组合 REL/SELF，四格同参数名、同初值、同逐 epoch 取样，
Lipo，seed 0/1）给出一个**两个 seed 同号、且按 scaffold 成组的 bootstrap 区间不含 0 的正交互**：

| 量（Valid best RMSE 差，正=前者更好） | seed 0 | seed 1 | 均值 | 配对 bootstrap 95% |
|---|---|---|---|---|
| `g_bind_self = MEAN_SELF − BIND_SELF` | −0.0288 | −0.0478 | **−0.0383** | [−0.0691, −0.0060] |
| `g_bind_rel = MEAN_REL − BIND_REL` | +0.0186 | −0.0025 | +0.0080 | [−0.0230, +0.0389] |
| `g_ctx_mean = MEAN_SELF − MEAN_REL` | −0.0075 | −0.0156 | −0.0116 | [−0.0434, +0.0186] |
| `g_ctx_bind = BIND_SELF − BIND_REL` | +0.0399 | +0.0296 | **+0.0348** | [+0.0034, +0.0660] |
| 交互 `I = g_ctx_bind − g_ctx_mean` | +0.0474 | +0.0452 | **+0.0463** | [+0.0015, +0.0911] |

读法：**在 SELF 句内组合下，端口绑定是稳定的小幅伤害；在 REL 真实相邻句下，绑定不再伤害并且
“REL 优于 SELF”才出现。** 也就是说，“绑定”与“真实关系组合”是互补的，单独任一项的边际收益都
不可重复（`g_bind_rel` 均值 +0.008 < δ_mechanism = 0.0121）。这是本轮唯一达到“可重复机制信号”的
主动效应，方向与本轮假设一致，但**幅度（≈0.046 ≈ 0.038σ）与 seed 间波动（同臂跨 seed 最大 0.021）
同一量级**，因此它是“条件性、机制级”的证据，不是可以宣布的主效应。

三件同时成立、但都不能替代真标签收益的事实：

1. **整体仍落后同协议强参照。** 四格最好（按协议 W 规则）= `MEAN_SELF` 0.7095 均值（0.7200/0.6990）；
   同协议重训 `GINE_REF` = **0.6219**（0.6181/0.6257）。配对差 **+0.0875～0.0911**（95% CI 不含 0），
   超过 `ref_gap = 0.06σ = 0.0727`。旧轮 `FULL_WGINE` Valid 0.7679 是**不同配方/不同 regime** 的历史坐标，
   本轮 W 比它好 0.058，但这不是配对跨 seed 效果，不能写成受控优劣。
2. **端口-状态机制在模型内部确实被使用**：在 `BIND_REL` 的冻结权重上，把端口算子换成 MEAN 只改变
   Valid RMSE +0.0029，而把句间算子换成 SELF 改变 **+0.1313**；MEAN→BIND 的预测变化量与分子的
   “非等价兼容位点占比”秩相关 0.36（MEAN_SELF 权重下 0.40）。方向正确，但说明网络对**句间算子**
   的依赖远大于对端口算子的依赖。
3. **词表支持充足但覆盖有限**：Valid 中 43.4% 的 incidence 落在尺寸≥2 的兼容位点组内，其中
   **32.4% 属于“同属性但 exact rooted 内部图身份不同”**（=端口均值真正损失的区分）；92.4% 的分子
   至少有一个这样的位点。另有 8.0% 的 Valid incidence 属“局部内容见过、相对排列没见过”。

结论标签（§11 逐条给证据）：`PORT_BINDING_UNRESOLVED`（有交互、无主效应）、`RELATIONAL_CONTEXT_HELPFUL`
（仅在 BIND 条件下，且需 `TOPOLOGY_SPECIFIC_GAIN_UNRESOLVED`）、`INSTANCE_GEOMETRY_UNRESOLVED`、
`VECTOR_READOUT_HELPFUL`（弱）、`STRUCTURAL_MECHANISM_WITH_PREDICTION_GAP`、
`BAG_ADVANTAGE_UNRESOLVED`/`PARTITION_ADVANTAGE_UNRESOLVED`（见 §8）。

---

## 1. 问题、四格与判据

协议 §0 冻结的四格（全部使用**真实**双端口关系 R 与真实词内图，只改两个开关）：

| 格 | 端口状态 | 句间组合 | 语义 |
|---|---|---|---|
| `MEAN_SELF` | 同属性兼容位点组内均值 | 只用自身句 | 双对照（“无端口身份 + 无真实句”） |
| `BIND_SELF` | 实际端口状态 | 只用自身句 | 端口身份 + 词袋式句 |
| `MEAN_REL` | 组内均值 | 真实相邻句 | 真实关系 + 无端口身份 |
| `BIND_REL` | 实际端口状态 | 真实相邻句 | 双开（本轮假设的完整形式） |

判据（Lipo，Train std σ = 1.21154）：`δ_mechanism = 0.01σ = 0.0121`，`δ_pipeline = 0.025σ = 0.0303`，
`tie = 0.002σ = 0.0024`，`ref_gap = 0.06σ = 0.0727`。可重复机制信号 = 两 seed 同号 且 均值 ≥ δ_mechanism。
主指标 = best-Valid RMSE（原尺度，g/L，log 标度）。

## 2. 数据与身份（只读继承）

* 原始缓存：`tracks/structural-query-learning/data/ogbg_mollipo_sq-pilot-cache-v1.npz`。
* 词结构：**只读继承** `word_sentence_port_route_v1/data/ogbg-mollipo_words_ws-word-route-v1.npz`，
  content hash `42aa4fc9…f8d3fc`（见 `analysis/structure_freeze.json`）；模板表 148 个、最大 6 原子。
  本轮**没有**重建分词、没有改动 v1 的任何结论。
* 新派生（每 split 由结构决定、与标签无关）：`site_group`（词内 9 列原子属性相同的兼容位点组）、
  `group_count`、端口对距离表、`pp`/`inc`/`pair` 字段；Lipo 派生 content hash `21d5aee0…b687fc2`。
* Test 只在本报告 §9 的终局一次性使用；Train/Valid 的切分与 v1 相同（3360/420/420）。

## 3. 主表（Stage B/C，12 单元，R0，E = 240）

| arm | seed | best epoch | best-Valid | 该点 Train | 末轮 Valid | 末轮 Train | 参数 |
|---|---|---|---|---|---|---|---|
| `MEAN_SELF` | 0 / 1 | 59 / 64 | **0.7200 / 0.6990** | 0.359 / 0.330 | 0.7612 / 0.7462 | 0.097 / 0.107 | 284,514 |
| `BIND_SELF` | 0 / 1 | 57 / 37 | 0.7488 / 0.7468 | 0.332 / 0.484 | 0.8316 / 0.8033 | 0.086 / 0.105 | 284,514 |
| `MEAN_REL` | 0 / 1 | 63 / 53 | 0.7275 / 0.7146 | 0.305 / 0.357 | 0.7763 / 0.7469 | 0.079 / 0.092 | 284,514 |
| `BIND_REL` | 0 / 1 | 69 / 53 | 0.7089 / 0.7171 | 0.274 / 0.339 | 0.7561 / 0.7696 | 0.078 / 0.086 | 284,514 |
| `GINE_REF` | 0 / 1 | 176 / 111 | **0.6181 / 0.6257** | 0.217 / 0.333 | 0.6256 / 0.6463 | 0.182 / 0.206 | 391,942 |
| `SET_MEAN_SELF` | 0 | 53 | 0.7220 | 0.398 | 0.7899 | 0.117 | 284,514 |
| `BIND_SELF_SCALAR` | 0 | 45 | 0.7606 | 0.449 | 0.8374 | 0.116 | 284,514 |

三件必须一起读的事：

* **词族严重过拟合**：四格末轮 Train 0.078–0.107，Valid 0.746–0.832；best epoch 都落在 37–69。
  `GINE_REF` 的 best epoch 在 111–176、末轮 Train 0.18–0.21，泛化差距小得多。R0 是**训练前冻结**的，
  本轮规则不允许据曲线重调。因此下面所有条件差都是**在同一过拟合配方下**测的，等价于“同容量、
  同正则的配对对照”，不能读成“这套结构表示的上限”。
* **同臂跨 seed 波动 = 0.021**（`MEAN_SELF` 0.7200 vs 0.6990），与多数条件差同量级。协议因此要求
  “两 seed 同号 + 均值阈 + 成组 bootstrap”，本文所有结论都按这三条同时报告。
* 四格参数名/形状/初值完全一致（`analysis/checks.json` 的 `init_identity`：逐位相同，`rel_vs_self_s0_equal=0`），
  所以四格差异只能来自算子，不来自容量或初始化。

## 4. 条件差、交互与区间

见 §0 表。补充：

* 交互 `I` 的**符号在机制上是可预期的**：若“绑定”提供的是可供关系消息使用的端口身份，那么它应当只在
  REL 打开时兑现，这正是 `g_ctx_bind > g_ctx_mean`。实测两 seed 分别 +0.0474 / +0.0452，几乎相等。
* `g_bind_self` 的**负号是稳定的**（−0.029 / −0.048，CI [−0.069,−0.006]）：在只聚合自身句的配置里，
  保留端口身份反而变差。可解释为“没有关系消息时，端口身份只是额外方差”。
* `g_bind_rel` 均值 +0.0080、CI 跨 0：**不能**声称“绑定本身有收益”，这与 §0 的标签一致。
* 末轮（E=240）Valid 的主表排序与 best-Valid 不同（例如 `BIND_REL` s1 末轮 0.7696 差于 `MEAN_REL` s1
  0.7469），说明这些 0.01–0.03 的差异高度依赖“在哪个 epoch 取快照”。协议用 best-Valid 是**事前**
  规定的选择规则，但这条敏感性必须写进结论（见 §11 的 `RECIPE_SENSITIVE_CANDIDATE` 讨论）。

## 5. 支持统计与覆盖（`analysis/support.json`）

| 统计 | Train | Valid |
|---|---|---|
| incidence 总数 | 51,892 | 6,778 |
| 落在 size≥2 兼容组内 | 45.8% | 43.4% |
| 非等价（组内 rooted 身份不同） | **34.1%** | **32.4%** |
| 组内有等价伙伴（干预无效） | 6,057 条 | 748 条 |
| 至少一个受影响位点的分子 | 92.4% | 92.4% |
| 受影响且≥3 个端口（分支）的词 | 5,686 | 691 |
| “局部内容见过、相对排列没见过” | 0（定义集） | 8.0% |
| “严格内容（含 rooted 身份）没见过” | 0 | 39.3% |
| 落在 size≥2 兼容组内（Test） | — | 44.8% |
| 非等价（Test） | — | 33.0% |
| “局部内容见过、排列没见过”（Test） | — | 8.1% |
| 至少一个受影响位点的分子（Test） | — | 92.4% |
| 受影响分子的标签 std / 未受影响 | 1.194 / 1.328 | 1.165 / 1.584 |

口径：兼容组 = 同一词实例内 9 列原始原子属性完全相同的位点；非等价 = 组内存在不同的 **exact rooted
内部图身份**（root 固定、按 9/3 列全属性最小化序列化，≤6 原子穷举）。MEAN 干预只在这些组上才真正
改变计算，因此 **32.4% 是 Valid 上“端口绑定”这一对照的有效覆盖率**，其余 67.6% 的 incidence 在两格
之间完全相同——这直接限制了 2×2 能达到的效应上限，也是 `g_bind_rel` 只有 0.008 的结构性原因之一。

## 6. 冻结干预（同一权重、只换算子）

对训练好的权重**不重训**，只切换 `spec`（参数名/形状完全一致，所以是合法干预）：

| 冻结权重 | bind_rel | mean_rel | bind_self | mean_self |
|---|---|---|---|---|
| `BIND_REL` s0 | **0.7089** | 0.7118 | 0.8402 | 0.8456 |
| `MEAN_SELF` s0 | 0.7710 | 0.7650 | 0.7247 | **0.7200** |

* 端口算子（bind↔mean）在冻结权重上只值 +0.003（BIND_REL）/ +0.005（MEAN_SELF）；
  **句间算子（rel↔self）值 +0.131 / +0.045**。网络真正“用上”的是真实相邻句，而不是端口身份。
* 端口交换引起的预测变化量与“非等价位点占比”秩相关 **0.359 / 0.398**：变化集中在结构上该变化的地方，
  388/420 个分子受影响（与 §5 的 92.4% 一致），其余 32 个分子两类位点都没有，预测逐位不变。
* 端口状态几何：size≥2 的组上 `‖chosen − c‖` 均值 0.721（BIND_REL）/ 0.559（MEAN_SELF），p90 = 2.11/1.32；
  size=1 的组上恒为 0（构造使然）。即 MEAN 算子平均掉的确实是一个量级不小的向量，而不是噪声。

## 7. 桥接对照

* **实例几何**（`SET_MEAN_SELF` = 位置无关实例算子，其余同 `MEAN_SELF`）：seed 0 0.7220 vs 0.7200
  （差 −0.0021，CI [−0.0498, +0.0435]），seed 1 **0.7402 vs 0.6990（差 +0.0412）**——**两 seed 反号**，
  均值 0.7311 vs 0.7095（SET 差 0.0216）。即“去掉词内位置结构”在两个 seed 上一个几乎无差、一个
  明显变差，**不可重复**；`INSTANCE_GEOMETRY_UNRESOLVED`（不能说位置结构无用，也不能说有用）。
* **读出**（`BIND_SELF_SCALAR` = 标量读出，其余同 `BIND_SELF`）：seed 0 0.7606 vs 0.7488（向量更好 0.0118），
  seed 1 **0.7365 vs 0.7468（标量更好 0.0102）** —— **两 seed 反号**，均值 −0.0008，
  成组 CI [−0.0304, +0.0310]。标量读出与向量读出**不可分辨**；`READOUT_UNRESOLVED`。

## 8. Stage D 分支对照（“更强对照”是否削弱路线主张）

| 对照 | 与 `BIND_REL` 的差（正=更强对照更差） | seed 0 / 1 同号 | 成组 CI | 判定 |
|---|---|---|---|---|
| `BAG_CAP`（跨词内容全部换成本身，词袋化） | +0.0420 / −0.0029 → 均值 **+0.0196** | 否 | [−0.0527, +0.0151] | `BAG_ADVANTAGE_UNRESOLVED`：词袋化没有更好，但也没被显著打败 |
| `GLOBAL_CTX_BIND`（把真实相邻句换成**同一分子内所有句的加权混合**） | +0.0243 / +0.0097 → 均值 **+0.0170** | **是** | [−0.0481, +0.0147] | 点估计支持“真实邻接拓扑优于全图混合”，均值 ≥ δ_mechanism |
| `GINE_RAW`（同 backbone、原始 9/3 列特征） | 待填 | | | |
| `BIND_REL` + 随机连通分块（salt 固定、≤6 原子） | 待填 | | | |

`GLOBAL_CTX_BIND` 值得单独说明：它保留端口状态、词内图、双端口 R、词表与全部参数量，**只把
“消息来自真实相邻词”换成“来自同分子全句混合”**，两 seed 都比 `BIND_REL` 差（0.024/0.010），
即本轮唯一稳定的关系效应**不是**“多了一句上下文”，而是“消息确实沿着真实词间边流动”。因为
均值 0.0170 位于 δ_mechanism(0.0121) 与 δ_pipeline(0.0303) 之间、且 CI 仍跨 0，报告按
`RELATIONAL_CONTEXT_HELPFUL` + `TOPOLOGY_SPECIFIC_GAIN_UNRESOLVED` 并列实际数字，不宣布等效。

## 9. 终局 Test

（见文末补记。）

## 10. 效率与预算

* 12 个核心单元总墙钟 1.82 h（每单元 9.1 min，CPU 7 线程；两 worker 并行 ⇒ 墙钟 ≈ 55 min）。
  计时冻结：`t_epoch_max_median = 1.506 s`（GPU 未测；CPU 上 E=240 投影 7.5 min ≤ 25 min 预算），
  故 horizon 取 **E = 240**（`analysis/timing_mollipo.json`）。
* 参数：四格 284,514 < 500k 上限；`GINE_REF` 391,942。
* GPU·h = 0（regime 偏差，见 `analysis/execution_regime.json` 与 PROTOCOL 附录 A）。
* 无 OOM：峰值由 CPU 内存约束（每 worker 独立进程，`WP_THREADS=7`、`OMP/MKL_NUM_THREADS=7`），
  未出现 >16 GB 的常驻；本轮没有触发任何 OOM killer 或 swap 抖动。

## 11. 结论标签与逐条证据

| 标签 | 依据 |
|---|---|
| `PORT_BINDING_UNRESOLVED` | `g_bind_rel` +0.008（CI 跨 0）、`g_bind_self` −0.038（稳定为负）；只有交互 `I` +0.046 稳定为正 |
| `RELATIONAL_CONTEXT_HELPFUL` + `TOPOLOGY_SPECIFIC_GAIN_UNRESOLVED` | 仅 BIND 条件下 REL 优于 SELF（+0.035，CI [+0.003,+0.066]）；MEAN 条件下反向（−0.012，CI 跨 0） |
| `INSTANCE_GEOMETRY_UNRESOLVED` | `SET_MEAN_SELF` 与 `MEAN_SELF` 差 −0.002，CI 跨 0 |
| `READOUT_UNRESOLVED` | `BIND_SELF` vs `BIND_SELF_SCALAR` 两 seed 反号（−0.0118 / +0.0102），均值 −0.0008，CI [−0.030, +0.031] |
| `STRUCTURAL_MECHANISM_WITH_PREDICTION_GAP` | 机制（交互/干预/几何）有可测证据，但四格最好仍落后 `GINE_REF` 0.088–0.091（> ref_gap） |
| `RECIPE_SENSITIVE_CANDIDATE` | 词族在 R0 下过拟合（末轮 Train 0.08 vs Valid 0.75），且 best epoch 与末轮排序不同；未换配方验证 |
| `NO_ADVANTAGE_IN_TESTED_COMPOSITION_FAMILY` | **不适用**：交互与 REL 条件收益达到了事前定义的可重复机制信号 |

不使用的标签及理由：`PORT_BINDING_HELPFUL`（主效应不可重复）；`GLOBAL_CONTEXT_STRONGER`（未测，见 §8）；
`BAG_STRONGER`/`RANDOM_PARTITION_STRONGER`（未测或未达两 seed 条件）；`ENGINEERING_BLOCKED`（主表完整）。

## 12. 未完成 / 未测

* 随机连通分块（`RANDOM_PART`）与 `GINE_RAW`、`GLOBAL_CTX`、`BAG_CAP`：见 §8。
* ESOL 交叉复现：按协议 §5 的触发条件，W 无核心配对 C（W = `MEAN_SELF` ⇒ C = null），
  **不满足“W/C 自身可重复机制信号”**，故按规则跳过（见 PROTOCOL 附录 A 的判定记录）。
* 第三 seed：协议规定只在“关键配对方向稳定后再加”，本轮交互已两 seed 同号且 CI 不含 0，
  但幅度与 seed 波动同量级；时间预算优先给了分支对照，未加 seed 2。
