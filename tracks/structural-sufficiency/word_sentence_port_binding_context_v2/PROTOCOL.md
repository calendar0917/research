# PROTOCOL — 结构词的端口绑定与句间组合（`word_sentence_port_binding_context_v2`）

Status: **frozen before any formal training run.** 本文件冻结本轮定义、配置与分支。
执行偏差只追加在附录 A，不倒改 §0–§13。上层任务书：
`word_sentence_port_binding_context_agent_prompt_v2.md`（v2026-11-21）。

T0 = `2026-11-21T01:22:31Z`（单调时钟 `1795224151`）。硬截止 T0+8h。

## 0. 问题与四格

区分三件事：(1) 词内完整属性的**位置**是否可被利用；(2) 跨词连接的**位点绑定**
（哪个端口接出去）是否比“同一完整原子属性的兼容位点均值”更有用；(3) **真实相邻句**
的状态是否比同参数逐句计算更有用。主实验是 2×2：

| 名称 | 端口状态 | 两层后续句计算 | 读出 |
|---|---|---|---|
| MEAN_SELF | 兼容位点均值 | 当前句来源（SELF） | 向量先池化 |
| BIND_SELF | 真实位点 | SELF | 向量先池化 |
| MEAN_REL | 兼容位点均值 | 实际相邻句来源（REL） | 向量先池化 |
| BIND_REL | 真实位点 | REL | 向量先池化 |

桥接对照：`SET_MEAN_SELF`（位置无关实例算子）、`BIND_SELF_SCALAR`（先 rho 再池化）。
强参照：`GINE_REF`（`sq_gine` 5 层 width128）。条件诊断：`GINE_RAW`。

## 1. 数据、身份与评价

* 主任务 `ogbg-mollipo`（3360/420/420），条件外推 `ogbg-molesol`（902/113/113）。
* 原始图：`tracks/structural-query-learning/data/ogbg_{mollipo,molesol}_sq-pilot-cache-v1.npz`
  经 `sq_common.EXPECTED_CACHE_SHA256` 逐字节校验。9 列原子、3 列键属性全保留。
* 词结构：**继承** `word_sentence_port_route_v1/data/<dataset>_words_ws-word-route-v1.npz`
  的 content hash（Lipo `42aa4fc9…`、ESOL `0ca99990…`，完整值见
  `analysis/structure_freeze.json`），词表/合并历史/词最大 6 原子均不改。
* 本轮新增**派生特征缓存** `data/<dataset>_portbind_v2.npz`：位点表、兼容位点分组、
  词内边、词内距离对、度数；其 content hash 单独记录为 `derived_sha256`。
  zip 时间戳不作身份。
* 目标按 **Train** mean/std 标准化；训练 full prediction 的 MSE；报告恢复原尺度的
  RMSE/MAE。`sigma = Train target std`（Lipo 1.21154，ESOL 2.06672）。
* 调度阈值（仅本轮资源调度用）：`delta_mechanism=0.01*sigma`（Lipo 0.01212）、
  `delta_pipeline=0.025*sigma`（0.03029）、`tie=0.002*sigma`（0.00242）。
* Test = `TERMINAL_DEVELOPMENT_TEST`：训练/选择不建 Test loader；只在
  `finalist_manifest.json` 提交后集中一次终局评分。长期已暴露的开发 Test，不称盲测。

## 2. 模型（共享计算图，flags 只改计算不改参数）

H=64；激活 SiLU；普通 MLP 为 `input→H→output`，中间 dropout 0.1；残差块后 LayerNorm；
除词型 WGINE 的旧实现外不加 BatchNorm；空集合 sum/mean 为 0，计数单独提供。

1. **词型内容** `e_i`：旧 `ws_model.WGINEEncoder`（模板两层 width32、原子列 [0,3,7]、
   键列 [0]、sum/mean/log-size → 64），参数从头训练。
2. **实例图**：`t_a^0=Linear36(x36_a)`，`bond64=Linear12(b12)`；三层共享
   `A_a^l=Σ_{b∈N_in(a)} SiLU(t_b^l+bond64_ab)/sqrt(max(d_in(a),1))`，
   `t_a^{l+1}=LN(t_a^l+Dropout(MLP_l((1+ε_l)t_a^l+A_a^l)))`，`MLP_l: H→2H→H`。
   跨词键不进入该阶段。
   `z_i=MLP_z([e_i, mean_a t_a, sum_a t_a, mean_internal bond64, log1p(n_i), log1p(d_b_i)])`。
3. **位点状态**：模板内最短路 `d(p,a)∈{0..5}`，共享 6×8 距离 embedding；
   `v_i,p=mean_{a∈D_i} MLP_site([t_a, EmbDist(d(p,a))])`（对**全部**位点 p，含无外连）；
   `c_i,p=MLP_port([z_i, t_p, v_i,p])`。
   * **BIND**：incidence 用实际 `c_i,p`。
   * **MEAN**：`C_i(p)={r: X_r==X_p}`（X = 原始 9 列属性），用
     `bar_c_i,p=mean_{r∈C_i(p)} c_i,r`；同一位点的所有 incidence 用同一替换。
     实际端点 x36 与实际键 b12 仍单独保留。
4. **跨词接口**：`E_α=[chosen_c_i,p, chosen_c_j,q, x36_i,p, x36_j,q, b12_α]`；
   `m_α=MLP_msg([z_i,z_j,E_α])`；旧定义双端口 `R(D_i;p,q)`（15 维）与
   incidence-pair 字段（7 维）**四臂都用真实 R**；
   `P_{α,β}=MLP_pair([z_i,m_α+m_β,m_α*m_β,R_i(p_α,p_β),pair])`。
   `s_i^0=LN(z_i+MLP_sent([z_i, Σm/√max(d_i,1), mean m, mean_pairs P, log1p(d_i),
   log1p(n_pairs_i)]))`。
5. **句间组合**（两层独立参数，层内同步更新）：
   `u_α^l=MLP_ctx_l([source_α^l,E_α])`，
   `s_i^{l+1}=LN(s_i^l+MLP_update_l([s_i^l, Σu/√max(d_i,1), mean u, log1p(d_i)]))`。
   **REL** `source=s_j^l`；**SELF** `source=s_i^l`。R、E、原始词状态不随层更新。
   有效范围约三跳词邻域；不声称原子三跳或任意长程。
6. **读出**：`q_i=MLP_q([z_i,s_i^0,s_i^2])`，`μ_i=n_i/N_atoms`，
   `g=[log1p(N_atoms),log1p(N_bonds),log1p(N_words)]`；
   向量臂 `F=ρ([Σμ_i q_i, g])`，`ρ: H+3→H→1`；标量臂 `F=Σμ_i ρ([q_i,g])`。
   所有连通分量纳入；`g` 只在统一读出使用。

实现约束：所有臂共用同一个 `nn.Module` 类与**同一套参数名/形状**（flags 只切换计算），
四格逐位同初始（按参数名派生 seed）。新模型参数上限 500,000；预计 ~2.7×10^5，
不触发 H=48 后备（若触发，按任务书 §5.1 统一替换）。

## 3. 桥接对照与参照

* `SET_MEAN_SELF`：以 MEAN_SELF 为母体。`A_set_i=sqrt(dbar_i)*mean_{b∈word}
  SiLU(t_b+mean_bond64_i)`（同词所有 a 相同；无内部边为 0）；
  `v_set_i=mean_{a∈word} mean_{d=0..5} MLP_site([t_a,EmbDist(d)])`；
  `c_set_i=MLP_port([z_i,mean_a t_a,v_set_i])`，所有位点用 `c_set_i`。
  6 行距离 embedding 全部参与，参数形状与层数不变。
* `BIND_SELF_SCALAR`：与 BIND_SELF 完全相同，仅 `F=Σμ_i ρ([q_i,g])`；
  `ρ` 参数形状与初值相同。
* `GINE_REF`：`sq_gine` 5 层 width128 GINE + degree/min-ring 适配，R0 配方，
  每 epoch 按公共排列重新组 batch。
* `GINE_RAW`（条件诊断）：同 GINE_REF，删除 degree/min-ring 两个额外通道。

## 4. 配方 R0 与 horizon

AdamW peak_lr 1e-3、wd 1e-5、batch 128 原始分子；5 epoch warmup 1e-4→peak，
逐 step cosine 至 1e-5；clip 5；FP32；TF32 off；无 AMP/EMA/SWA/预训练。
**每个 epoch 重新生成分子排列**；同 seed 的所有臂使用同一份逐 epoch 排列
（`np.random.default_rng(20261121+seed)` 预生成 E 个排列，GINE_REF 也按同一排列组 batch）。

默认 E=240。计时：MEAN_SELF / BIND_REL / GINE_REF 各 2 warm + 5 timed epoch，
完整 Train（含与正式训练相同的 Train/Valid 前向但不读/存指标），CUDA 计时同步，
取**最大**中位 `t_epoch`，冻结共同 horizon：

* `240*t*1.25 <= 25min` → E=240；
* 否则 `160*t*1.25 <= 25min` → E=160；
* 否则先优化数据打包/缓存最多 20min，仍不满足则 E=160 并披露实际成本。

OOM 只按 128→64→32 降 batch + 梯度累积保持有效 128；无法维持则披露。

## 5. 阶段与分支（事前定义）

* **A**（≤20min 工程检查 + ≤20min 两探针）：保真/重编号/同初始/健康性/checkpoint
  重载；A1 端口绑定探针（固定占用位点交换邻词）；A2 句间组合探针（同签名路径对）。
* **B**：Lipo seed0 七臂：MEAN_SELF、BIND_SELF、MEAN_REL、BIND_REL、SET_MEAN_SELF、
  BIND_SELF_SCALAR、GINE_REF。先并行 BIND_REL 与 GINE_REF。
* **C**：四格 + GINE_REF 的 seed1（共 12 个正式单元）。四格条件差
  `g_bind_self, g_bind_rel, g_ctx_mean, g_ctx_bind`、交互 `I=g_ctx_bind-g_ctx_mean`。
  `W` = seed0/1 平均 best-Valid RMSE 最小的四格架构（tie 顺序 MEAN_SELF、BIND_SELF、
  MEAN_REL、BIND_REL）。`C` 在读完 seed0/1 后按 §6 规则固定，不因 ESOL 重选。
* **D_REF**（触发：W 的 seed0/1 平均比 GINE_REF 均值差 > 0.06σ）：
  GINE_RAW seeds0/1。
* **D0**：seed0 `|r_SET_MS - r_MS| >= delta_mechanism` → SET_MEAN_SELF seed1；
  seed0 `|r_BS_SCALAR - r_BS| >= delta_mechanism` → BIND_SELF_SCALAR seed1。
* **D1**（触发：`g_ctx_mean` 或 `g_ctx_bind` 为可重复机制信号）：
  `GLOBAL_CTX` seeds0/1，`source_α^l=Σ_{k∈同一分子} μ_k s_k^l`，其余不变。
* **D2**（触发：出现可重复机制信号或 W 较旧 0.767951 改善 ≥ delta_pipeline）：
  `BAG_CAP` seeds0/1，固定 W 的设置，跨词信息全部用当前词自身替代：
  `z_j→z_i`、邻词端口状态→当前词中心端口状态、邻词 x36→中心端点 x36；
  pair 的“同邻词/邻词间键”字段置 0，R 保留当前词真实 R；incidence 数、边界键属性、
  位点占用保留。**R 保持当前词真实 R** 意味着“同邻词”标志与邻词间键字段为 0。
* **D3**（触发：W 平均 Valid ≤ GINE_REF 均值+0.06σ 且 W 相对 BAG_CAP 两 seed 平均
  ≥ delta_mechanism 且两 seed 同向）：`RANDOM_PART` seeds0/1（固定盐 20261121，
  按 Train 词大小分布随机连通分块，共享 WGINE 无逐模板自由 embedding）。
* **D4**（触发：存在可重复机制信号且 W 改善 ≥ delta_pipeline；另按任务书末段
  “更强对照”停止条件）：ESOL 冻结 W/C/GINE_REF seeds0/1，共 6 单元。
* **D5**（触发：W 含可重复正向核心配对、机制对照完成、未触发 D4 停止、仍有预算）：
  Lipo W/C/GINE_REF seed2，共 3 单元。
* **D6**（互斥：四个核心比较均无可重复机制信号，且 ≥3/4 主矩阵模型两 seed 都出现
  `best_epoch<=0.4E`、`final Valid-best Valid>=0.04σ`、`final Train<=0.8×best点Train`）：
  仅改 wd=1e-3（R_G），四格 seed0；若某核心比较 ≥ delta_mechanism 且处理臂优于同臂
  R0 ≥ delta_pipeline → 补该对 seed1（≤2 单元）。最多 6 单元。

优先级：D_REF → D0 → D1 → D2 → D3 → D4 → D5；D6 为互斥路径。核心 12 单元，
常规满路径 29，硬上限 32 个正式启动、16 GPU·h、8h 墙钟。

## 6. 选择、候选与终局

* `W`（研究候选）与 `F`（性能候选）分开：F 从**所有**已完成相同 seed0/1 的合格配置
  （含 BAG_CAP、GLOBAL_CTX、RANDOM_PART、桥接对照）中按平均 Valid 选。
  tie 顺序：位置无关/MEAN 优先，SELF 优先，原冻结分词优先，R0 优先；仍并列按
  完整 arm 名称字典序。
* 核心配对 `C`：从 W 出发按 D4 规则（W=BIND_REL → MEAN_REL/BIND_SELF 中相对 W
  平均差距更大且两 seed 正向者，tie 优先 BIND_SELF；W=MEAN_REL → MEAN_SELF；
  W=BIND_SELF → MEAN_SELF；W=MEAN_SELF 或无正向配对 → C=null）。
* Lipo 终局至少 W、C（若有）、GINE_REF；F 不同则纳入；总架构数 ≤6；各 seed 单列，
  主平均只用共同 seed 集合。
* 统计：1000 次配对 scaffold bootstrap（同一次抽样在全部模型/seed 间共用索引；
  每次先算各 seed 内预测差再取 seed 均值）。CI 跨 0 不等于等效。

## 7. 支持与解释

Train/Valid/终局 Test 分别报告：兼容位点集合 ≥2 的 incidence 数/比例/分子/scaffold；
兼容集合内存在不同 exact rooted 内部图身份的 incidence 数/分子/词型；实际 c 与兼容
均值的差异分布；零/单/多外部位点词比例、词大小、边界键比率。
“已见内容的新布局”统计从**同一 predicate 的 global word id→molecule id 映射**聚合。
冻结干预：对 W 做一次 BIND→MEAN（适用时）与 REL→SELF source 干预，报预测/RMSE 变化
与支持覆盖；8 个案例（2 个读 Valid 标签前固定的结构支持例 + 误差改善/恶化各 3）。

## 8. 结论标签（择实使用）

`INSTANCE_GEOMETRY_HELPFUL|UNRESOLVED`、`PORT_BINDING_HELPFUL|UNRESOLVED`、
`RELATIONAL_CONTEXT_HELPFUL`(+`TOPOLOGY_SPECIFIC_GAIN_UNRESOLVED`|`GLOBAL_CONTEXT_STRONGER`)、
`VECTOR_READOUT_HELPFUL|SCALAR_READOUT_COMPETITIVE|READOUT_UNRESOLVED`、
`STRUCTURED_ROUTE_COMPETITIVE`、`STRUCTURAL_MECHANISM_WITH_PREDICTION_GAP`、
`BAG_STRONGER|BAG_ADVANTAGE_UNRESOLVED`、`RANDOM_PARTITION_STRONGER|PARTITION_ADVANTAGE_UNRESOLVED`、
`RECIPE_SENSITIVE_CANDIDATE`、`NO_ADVANTAGE_IN_TESTED_COMPOSITION_FAMILY`、
`ENGINEERING_BLOCKED|BUDGET_LIMITED_INCOMPLETE`。

## 9. 账本

每个正式 run 记录启动 commit、配置 hash、结构 hash、seed、状态、wall、GPU 型号、
物理 GPU·h、失败/取消/续跑。每 10–15min 更新 `run_manifest.json`。

---

## 附录 A：执行记录（训练结束后追加，不改 §0–§9）

### A.1 执行 regime 偏差（最大偏差）

`res` 从 T0 起不可达：`rr doctor res` 报 `Connection timed out during banner exchange`，随后每 45 s 的
SSH 探针持续报 `ssh: connect to host pve-old.yulinsec.cn port 22: Connection timed out`，直到 Stage B
启动时仍未恢复（本地出网正常）。因此**全部正式单元在本机 CPU 上执行**：两个 worker，每个
`WP_THREADS=OMP_NUM_THREADS=MKL_NUM_THREADS=7`，与“每 GPU 一个任务”的并发度一致。
记录于 `analysis/execution_regime.json`：`regime = LOCAL_CPU`、`gpu_hours = 0`、driver/node = null。
**影响**：四格与所有对照共享同一 regime，跨臂比较内部有效；绝对数值不与上一轮 GPU 数值直接对照
（`GINE_REF` 同协议重训 0.618/0.626 与上一轮 0.629 接近，可作弱一致性检查，但不作受控结论）。

### A.2 正式单元前的工程修复（均不影响已记录的历史结论）

1. `derive_core` 的兼容位点分组按 `(word, 9 列)` 全局 `unique`，跨词合并 → 改 `lexsort` 连续分组，
   断言 `word_group_ptr` 与组数一致。
2. `bond_u/bond_v` 已是 split 全局偏移，端口/边偏移不得再加 `atom_off`（双重偏移）。
3. `pack_batch` 的 `word_site_off` 必须用“批内首 site”重算；`pack_molecules` 的 bond 偏移同理。
4. `evaluate_word` 曾把**原始**标签当作已标准化目标送入 `metric_block`（无效 Valid/MAE、双标准化）。
   在正式队列启动前修复（`targets.append((raw - mean)/std)`）。本次修复前有一次**临时验证 run**
   （`BIND_REL` seed 0，直接调用 `wp_launch.sh`）在 epoch ~120 时被终止，未写 `run.json`，其
   `state_last.pt` 被随后的正式 run 逐 epoch 覆盖；**正式 12 个单元全部在修复之后启动**
   （`BIND_REL_s0` 的 `started_unix = 1795225783`，修复提交早于该时间戳）。
5. `_layout_support` 的占位实现与 `paired_bootstrap` 死代码在支持统计定稿前删除/重写。

`analysis/checks.json` 与 `analysis/probes.json` 记录了 Stage A 的工程检查与 A1/A2 探针结果
（保真度、重编号/重排/batch 不变性、四格逐位同初值、健康梯度、A1 端口绑定可学、A2 句间组合可学）。

### A.3 冻结时点与偏差

* 计时在 GPU 上未测；CPU 上 `t_epoch_max_median = 1.506 s/epoch`，E=240 投影 7.5 min ≤ 25 min，
  故按 §4 规则取 **E = 240**（`analysis/timing_mollipo.json`，写于任何正式单元之前）。
* 分支执行：Stage B 7 臂 seed 0 → Stage C 四格 seed 1 → Stage D（`SET_MEAN_SELF`/`BIND_SELF_SCALAR`
  补 seed 1；`BAG_CAP`、`GLOBAL_CTX_BIND`、`GINE_RAW` 各 2 seed；`BIND_REL` 随机连通分块 2 seed）。
  未改动 §0–§9 的任何规则、阈值、选择顺序。
* **ESOL 触发判定（按 §5 机械判定，未自行改规则）**：W = `MEAN_SELF` ⇒ 核心配对 C = null
  （§6：W 为 `MEAN_SELF` 时 C 不存在）。§5 规定“选定的 W/C 比较必须自身满足可重复机制信号，
  否则跳过 ESOL”，故按规则**跳过**。为避免“看到结果再决定”，此处记录判定时间点：
  当时 12 个核心单元已全部完成，四格均值已知（`summary.json` 首版生成于 03:00Z 之前）。
* 补充：为回答“交互是否在第二个数据集复现”，另以**同一冻结配方**跑了 ESOL 的**完整四格 + GINE_REF**
  （每臂 2 seed，对称、无选择），并在报告中明确标注 `EXPLORATORY_REPLICATION_OUTSIDE_FROZEN_TRIGGER`；
  它不参与 §6 的选择规则，也不改变 §0–§9。

### A.4 预算与产物

* 正式单元：Stage B/C 12 + Stage D 10 + ESOL 10（探索复现）= 32，等于 §9 上限；GPU·h = 0。
* 墙钟：核心 12 单元 1.82 h CPU·worker；全轮 < 4 h。
* 权重/预测保存在 `results/<run_id>/`（`state_best.pt`、`state_final.pt`、`preds_best.npz`、
  `preds_final.npz`，终局另有 `preds_test.npz`）；`.gitignore` 只忽略 `*.pt`/`*.npz`，
  文本产物（`run.json`、`train.log`、jobs、queue.log）入库。
