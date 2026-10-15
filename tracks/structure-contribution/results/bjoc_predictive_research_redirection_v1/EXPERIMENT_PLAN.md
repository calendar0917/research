# EXPERIMENT_PLAN — H-COVER 预注册实验草案

* 轮次：`bjoc_predictive_research_redirection_v1`
* 状态：**草案（draft）**。研究者确认后才冻结为正式 protocol，并另立 `configs/` 与 runner。
* 假设与动机见 `RESEARCH_PROPOSAL.md`；证据见 `EVIDENCE_MAP.md`。
* 纪律重申：本草案不执行；执行轮**不得**读取 official test / Locked / 任何新评价标签，
  不得改动历史实验结果，不得在 valid 上做 per-arm 之外的任何选择。

---

## 0. 研究问题与模型差异

**研究问题**：在完全相同的 SSL 目标与更新预算、完全相同的 BJOC-Null 架构与 R3 监督配方下，
扩大 SSL 的结构分布 `D_N`（N = 10k / 50k / 220k，`views_per_molecule = 967,680/N` = 96.8 / 19.4 / 4.4）
是否改善 held-out 泛化？若是，机制是"结构覆盖"还是"每分子曝光"？

**模型差异（唯一干预）**：

| 组件 | Base（冻结 R3） | Candidate-50k | Candidate-220k |
|---|---|---|---|
| BJOC 架构 | 413,505 参数 | 同 | 同 |
| SSL objective | 15%/15% mask，`CE_atom + CE_bond` | 逐字同 | 逐字同 |
| SSL 更新数 / batch / 优化器 | 7,560 / 128 / Adam 1e-3, wd 1e-5, clip 5 | 同 | 同 |
| SSL 语料 `D_N` | official ZINC subset train 10k | full ZINC train 前 50k（deterministic） | full ZINC train 220,011（minus valid/test 结构碰撞） |
| fine-tune | R3（360 ep cosine，28,440 更新，raw-y L1，last-5 soup） | 同 | 同 |
| fine-tune 数据 | official train 10k | 同 | 同 |
| 新增参数 | — | 0 | 0 |

---

## Stage 0 — CPU 正确性与泄漏检查（未通过不得买 GPU）

### 0.1 语料构建审计（新）

1. **包含关系**：断言 `D_10k ⊂ D_50k ⊂ D_220k`（行级）；`D_50k` 由冻结 hash 排序（如
   `sha256(str(full_index))`）取前 50k，label-blind；记录三份行清单 hash。
2. **official valid/test 结构排除**：对 `D_220k` 的每一行计算 raw-graph 的 WL/结构指纹，
   与 official valid 1,000 / test 1,000 的结构指纹求交；**碰撞行全部排除**并记录数量与 hash。
   （base 的 10k 训练行本身 ⊂ full train，属于 fine-tune 数据，保留。）
3. **重复审计**：`D_220k` 内 canonical/WL 重复组统计；对与 valid/test 的近似重复（可选，报告 only）。
4. **分布报告**（label-free）：三语料的分子大小、原子类型频率、键类型分布、junction 数量、
   环统计；确认 `D_220k` 对 official-valid 的结构分布是**覆盖扩张**而非分布替换。
5. **吞吐/内存**：实测 `build_mol_objects` 在 220k 上的构建时间与峰值内存；
   若超限，改为流式/分片缓存（不得改变任何特征公式）。

### 0.2 SSL 目标与泄漏检查（继承 + 重跑）

沿用并重跑 `code/bjoc_masked_pretraining_v1.py` 的既有门（在新语料的真实分子上）：

* `MASK_TOKEN_SLOTS_FREE`：mask token（atom 27 / bond 0）在语料中不被占用。
* `MASK_SWAP_INVARIANT`：200/200 atom + 200/200 bond 随机隐藏真值替换 ⇒ 模型输入逐位不变。
* `MASKED_REBUILD_LEAK_FREE`：100/100 重建；隐藏真值不出现在被 mask 对象自己的槽位。
* mask RNG 确定性 + visit 依赖；两次同 (seed, mol_id, visit) 相同、visit 变化则不同。
* `labels_read: false`：SSL 路径 AST 扫描无 `.y` / 标签缓存访问；y 不进入任何函数。
* 单分子信息依赖：被 mask 的 token 不能通过任何输入路径影响自己的 hidden target
  （用 dependency map + 正/负对照 witness 显式验证）。

### 0.3 公平比较检查

* **driver 等价 smoke**：用新 driver 在 `D_10k` 上跑 ≤20 updates，与冻结的 full10k SSL
  轨迹前缀比较（本地 bitwise；远端 ≤1e-6 噪声地板），证明"只换了语料采样"。
* 参数审计：预训练头不计入推理；`parameter_audit == 413,505` exact。
* 更新数审计：SSL 恰好 7,560；fine-tune 恰好 28,440；cosine 偏差 ≤1e-12。
* init 对齐：composer/reader init hash、per-seed batch-order hash 与 `soup_R3_seed{0,1}` 的来源记录一致。
* mask/rebuild 公式 hash 与冻结版本一致（模块 sha256 对比）。
* relabel / 端点交换 / batch 组成 / save-load 不变性 ≤ 1e-6（沿用历史 tolerance）。
* 聚合不变性：junction row permutation 为 no-op（与 `results/bjoc_junction_sufficiency_audit_v1` 一致）。

### 0.4 剂量轴 sanity

* 断言 3 臂的 `molecule_visits = updates × batch = 967,680`，`views_per_molecule` 正确；
* 记录每臂实际 visit 数与 mask visit_index 直方图；
* 记录每臂预训练 CE/acc 曲线（只作机制描述，不作选择）。

**通过判据**：0.1–0.4 全绿，尤其"10k 语料 driver 等价 smoke"与"valid/test 结构排除为 0 碰撞（或已排除）"。

---

## Stage 1 — 配对预测实验

**执行顺序（seed-0 筛选）**：

| 序号 | 臂 | SSL | fine-tune | 备注 |
|---|---|---|---|---|
| 0 | Base | 复用冻结 SSL-10k encoder | 复用 `soup_R3_seed{0,1}` | 0 GPU |
| 1 | C50-s0 | SSL on `D_50k`（seed 0） | R3（seed 0） | 必须 |
| 2 | C220-s0 | SSL on `D_220k`（seed 0） | R3（seed 0） | 主候选 |
| 3 | C*-s1 | 仅对通过 seed-0 gate 的臂（≥1 个；若两个都过则优先 220k） | R3（seed 1） | 预注册条件购买 |

**匹配要求**：与 Base 共享 fine-tune 数据、batch order、composer/reader init、R3 配方；
SSL 共享 objective / mask / 优化器 / 更新数；seed 约定逐字继承（`torch.Generator(seed+101)` 等）。
**禁止**：λ/频率/head/步数/语料大小的事后搜索；一个候选只能有一个语料定义；
不得因 seed-0 结果更换语料子集。

---

## Stage 2 — 指标与不确定性

**主指标**：official-valid（暴露 dev 集）seed-mean raw-y MAE，配对分子级对比 `C − Base`。

**配对 bootstrap**（5,000 次；molecule 单位；seed-mean 误差向量；固定 bootstrap seed）。

**最小实质效应阈值（MDE）与统计显著分开**：

* MDE = **0.004**。依据：(a) R3 配方效应 0.0037 是本系统已采纳的最小正效应；
  (b) 种子波动：R3 两 seed 0.079101/0.080586（差 0.0015）、P 0.084786/0.082313（差 0.0025），
  历史 seed 翻转案例可达 ~0.004；(c) 架构轮预注册 promotion gate 即 0.004；
  (d) <0.003 在两 seed 下不可分辨，不值得 1.5× 后续监督成本。**该阈值先于任何实验固定**。
* "统计显著" = bootstrap CI95 上界 < 0；只有**同时**满足 MDE 才叫"实质改善"。

**预注册 gate（seed-0 筛选，与 Base 配对）**：

1. `Δ_MAE ≤ −0.004`；
2. 不依赖单分子：去掉最大单分子贡献后 Δ 仍 < 0，最大单分子占比 ≤ 40%；
3. `ordinary (y>−3)` Δ ≤ 0；
4. `top-10-error-removed` Δ < 0；
5. RMSE ≤ Base + 0.005；
6. numerics / audits / 语料审计全绿。

**seed-1 确认 gate（对通过者）**：seed-1 Δ < 0 且 seed-mean ≤ −0.004，CI95 上界 < 0；
ordinary/top-10 方向一致；若 seed-1 单独不过但 seed-mean 过，须如实标注（沿用 recipe 轮先例）。

**次指标（描述性）**：RMSE、median/p90、tail（y≤−3）、结构 novelty 分层（Fit 距离四分位）、
覆盖度诊断 `corr(ΔAE, Δcov)`、预训练 CE/acc、per-arm wall/steps、推理开销（应完全相同）。

---

## Stage 3 — 停止条件（先固定）

出现任一情况即停止该路线，不再追加语料规模/步数/种子/变体：

1. 220k 与 50k 均未通过 seed-0 gate；
2. CI95 跨零，或仅单 seed 获益；
3. MAE 略降但 RMSE 或 tail 明显恶化（RMSE > +0.005 或 tail 反向主导）；
4. 收益随曝光减少而下降（50k 优于 220k 且 220k 不优于 Base）→ 判为曝光效应，停止扩语料；
5. 覆盖度诊断无变化（`Δcov` 与 ΔAE 无相关），而收益存在 → 归因失败，停止"覆盖"叙事；
6. 需要增加新模块或新目标才能保住正结果 → 与本机制无关，停止。

部分支持时的上限：若只有 ordinary 改善且 total < 0.004，只能记录为"小效应候选"，
不得直接启动下一轮微调或扩语料。

---

## Stage 4 — 独立确认与数据纪律

* **official valid**：历史研究中多次用于模型选择，只能作为**暴露 dev 集**；本轮仍按
  "预测先冻结 → 单次读取"执行，并在报告中披露 exposure。
* **official test**：**永不加载**（历史已 exposed）。任何未来使用都需新的预注册 + 明确授权。
* **内部 Locked**：不读。
* **降低选择偏差的唯一手段**（不属本草案执行范围）：H-COVER 若通过，下一轮应
  (a) 在**新的内部 grouped split**（未参与本轮筛选）上重做 SSL + R3 与对照；
  (b) 或在**第二个分子端点**（ESOL 或另一个 MoleculeNet 回归任务）上做同协议方向一致性检验。
  两者都需要新的预注册，本草案不自动执行。
* **暴露账本**：每臂 SSL 语料行 hash、fine-tune 行 hash、valid 读取时刻与 y hash、
  label-access 计数（应为 0）全部落盘。

---

## GPU 预算、环境、运行入口

* 主机/pool：`res`，`res-gpu1`（a100-2，单卡，串行）；驱动 550.163.01，torch 2.5.1+cu124。
* 预算：

| 项 | 单价 | 数量 | 小计 |
|---|---|---|---|
| SSL 7,560 updates | ~0.50 GPU-h | 3（50k-s0、220k-s0、胜者-s1） | 1.50 |
| R3 fine-tune 28,440 updates | ~0.20 GPU-h | 3 | 0.60 |
| smoke / prep | ~0.05 GPU-h | — | 0.05 |
| **合计** | | | **~2.15 GPU-h**（上限 3.5） |

* 运行方式（示意，冻结后由 `rr` 执行；本轮不运行）：
  `rr deploy res --pool res-gpu1` →
  `rr run res <prep> --pool res-gpu1`（语料构建 + Stage 0 门）→
  `rr run res <ssl> --set corpus={50k,220k} --set seed=0` →
  `rr run res <r3> ...` → 本地 `predict/freeze/score-valid`。
* 训练 commit 必须与 `protocol_freeze.json` 一致；所有门在远端重跑。

---

## 必须交付的产物（执行轮）

* `protocol_freeze.json`（config/code/tests/语料 hash、预算、gate 常量）；
* `corpus_audit.json`（包含关系、valid/test 碰撞排除、分布统计、行 hash）；
* `stage0_cpu_gates.json`（含 10k driver 等价 smoke）；
* `pretrain_{50k,220k}_s*.json`（updates/visits/molecule_visits/CE 曲线/encoder hash）；
* `train_R3corpus_{50k,220k}_s*.json`（init/batch-order/schedule 审计）；
* `prediction_manifest.json` + `predictions/valid_*.npz`（冻结后单次读取）；
* `valid_metrics.json`、`paired_bootstrap.json`、`coverage_diagnostic.json`、
  `gpu_budget_ledger.json`、`scientific_decision.json`、`REPORT.md`。

---

## 关键正确性测试清单（汇总）

1. 图 relabel / 边顺序不变性（对象层与 SSL loss 层）；
2. SSL 目标确实按预期计算（AST + 数值对照）；
3. 无标签/无隐藏目标泄漏（swap + rebuild + dependency map）；
4. 改动前后输入维度与权重初始化（参数 413,505、init hash 对齐）；
5. 语料扩展不改变模型/训练预算（更新数、batch、LR、schedule 逐项断言）；
6. 合成数据机制 sanity（mask 重建可行、visit 依赖、RNG 确定）；
7. 单条数据的信息依赖范围（masked token 不影响自己的 hidden target）；
8. valid/test 结构排除与语料包含关系；
9. driver 等价 smoke（10k 语料前缀一致）。
