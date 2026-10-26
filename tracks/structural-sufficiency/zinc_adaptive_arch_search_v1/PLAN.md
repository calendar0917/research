# PLAN — `zinc_adaptive_arch_search_v1`

轮次目标：在**无递归消息传递、显式结构对象/统计、单模型 ≤500k 参数**的约束内，
把 ZINC 单模型 official-valid raw-y MAE 推到 DCC-L 家族（4-seed 0.075990，
seed0 0.078537，params 494,193）水平以下，并识别哪一类结构组合计算值得发展成独立方法。

主指标：官方 PyG ZINC `subset=True`（train 10,000 / valid 1,000）raw-y MAE，
单模型、best = official-valid 严格最小选点。**official test 本轮不构造、不加载、
不预测、不评分。**

## 预算与时间

| 项 | 值 |
|---|---|
| 墙钟上限 | 8 h（start 2026-10-26T23:58+08:00） |
| GPU 软/硬预算 | 14 / 16 A100-h，单卡 ≤8 h |
| 已确认单价 | BOSI 主干 600 ep / 47,400 updates ≈ **0.37 GPU-h**（BOSI-B s0 实测） |
| 停止新正式训练 | T+7 h |
| 最后 45–60 min | 结果回收、核验、分析、交付 |

## 候选轴（优先级 A → B → C）

### A. BOSI-QP：target-conditioned pair composition（先买）

命题：BOSI 先把邻键 learned content 聚成固定矩，再在 Side 中与目标键结合。
让**目标半键状态参与每一个 pair 的共享非线性编码**（在聚合之前），是否比
"只对伙伴内容做一次共享非线性变换"更有效？

* `A1`（query-conditioned pair）：
  `t_ef = [q_e, p_f, |q_e-p_f|, q_e*p_f]`，`q_e = x[v,e]`，`p_f = x[v,f]`，
  `w_ef = Phi(t_ef)`（128→64→32，SiLU，共享），`P[v,e] = Σ_{f≠e} w_ef`，
  `P` 追加到 **原封不动的 BOSI-B Side 输入**（617→649）。
* `A0`（matched null）：同参数、同初始化、同优化器、同数据顺序，
  只把 `q_e` 换成**同一伙伴集合的均值** `q_null[v,e]`。

为什么是追加而不是替换：`zinc_bosi_partner_content_v1` 已证明伙伴内容通道是
load-bearing（P−B = +0.01239）。本轮先保留该通道，隔离"前聚合目标条件化"这一项。

预算：488,321 参数（BOSI-B 473,889 + Phi 10,336 + Side 新增 128×32 列 4,096），
≈0.45 GPU-h/run。第一步只买 A1/A0 seed 0（两卡并行）。

判定（预注册，见 `DECISIONS.md` D1）：

* `A1−B_s0 ≤ −0.003` 且 `A1−A0 ≤ −0.001` → DEEPEN（A 家族值得继续）。
* `A1−B_s0 ≤ −0.003` 但 `|A1−A0| < 0.001` → 结论是"前聚合非线性内容编码"有用，
  不是目标条件化；保留计算更简单的版本。
* `A1−B_s0 ≥ +0.005` 或 W20/T100 也同向更差 → 停止 A，转 B。
* 介于两者之间 → 至多一次确认（同 seed 已有对照，只允许补一个 seed 或最小简化）。

### B. SQBC：独立的目标键中心结构组合主体（A 无信号则重建）

要求：不是"在 DCC-L 后加模块"，而是从输入到预测完整成立、无递归的键中心组合体。
必须与 A 的信息流有本质差别（键级而不是半键级；不依赖 384 维矩 bank；pair 编码器
直接读取两端 bond token 与对端原子），并且有**等参数、同输入的对称/目标盲对照**。

冻结前必须先做的 CPU 审计：与 DCC-L 的 junction 归属机制逐项对照，写清
"哪一步计算是新的、哪一步是复刻"；若与 A 等价则不打。

### C. 局部/全局结构分尺度计算（条件启动）

仅当 A/B 的误差分析明确显示"局部组合已经有效、图级结构组合未被表达"（且不是
单个 row 或单一 cycle-tail 伪影）时才启动。设计必须是 block/环对象统计 + 一次
共享 Phi + 图级汇总，并配等参数 flat-global 对照。历史 `zinc_hierarchical_relation_v1`、
`zinc_arch_discovery_v1` 的 block 模型、`zinc_global_operator_search_v1` 的负结果
必须先核对；若只是复刻则跳过，不花 GPU。

## 硬约束（不变）

* 单模型部署参数 ≤500,000（含实际使用的编码器）。
* 主模型不得有多轮 node/edge hidden-state 递归、不得复制 ESA/GRIT/CSA 强结构、
  不得蒸馏、不得用 official test、不得把标签分量作为输入。
* 不允许"新增统计 → 塞进大 MLP → 报告 NO_GO"的循环；每个新主体必须有结构计算解释。
* 配方统一 R0：AdamW lr 1e-3→1e-5（per-step cosine, 500 warmup）、wd 1e-5、
  dropout 0、clip 5、batch 128、L1 raw-y、600 ep = **47,400 updates**、FP32。
* 同 seed 必须对齐：公共初始张量、SSL 源、batch 顺序、训练行、optimizer/scheduler。
* 每个正式模型必须报告：overall-1000、ordinary(y>-3)、tail(y≤-3)、radius-2
  all_supported / low_support / any_unseen、rings≥4、fixed worst-10 与其余 990、
  drop-row172 诊断、median AE / P90 AE / top-10 error share、W20、T100、
  train@best、train@final、参数量、GPU-h、checkpoint hash。

## 本轮执行顺序

1. **A 阶段**（t≈0.5–1.5 h）：A1/A0 seed0 并行 → 立即分析。有信号才深入。
2. **B 阶段**（t≈1.5–4.5 h）：A 无信号则实现并运行 SQBC 主体 + 目标盲对照。
3. **C 阶段**（条件）。若 A/B 都无信号且 C 与历史复刻，直接停止架构搜索。
4. **赢家确认**：剩余算力优先给 seed1/seed2 复现与同 seed 等参数对照。

## 交付物

`PLAN.md`（本文件）、`DECISIONS.md`（每轮 Decision Log）、`REPORT.md`（最终中文总结）、
`configs/aas_v1.json`（冻结协议）、`code/`（模型/训练/门控/分析）、`analysis/`（机器可读
指标、预测哈希、曲线、成本、诊断）、`runs/`（gitignored，含 checkpoint 与逐分子预测）。

---

## 执行附录（实际买了什么、为什么偏离预案）

| # | run | 轴 | 目的 | 状态 |
|---|---|---|---|---|
| 1 | A1_s0 / A0_s0 | A | 预注册首轮（两卡并行） | 完成 |
| 2 | A1_s1 / A0_s1 | A | cross-seed 方向一致性（预注册允许的那一次确认） | 完成 |
| 3 | B_s1 | A 基线 | **偏离**：seed 1 缺少 BOSI-B 的匹配基线，任何 seed-1 对照都会错配 | 完成 |
| 4 | S1_s0 / S2_s0 | B | 预注册主体 + **新增**的池化位置对照 S2（与 S1 同参数同初值，只换池化位置） | 完成 |
| 5 | A0M0_s0 / A0M0_s1 | A 消融 | **新增**：A0 的 moment-bank 消融（同参数同初值，M≡0），回答"新通道是否取代了矩 bank" | 完成 |
| — | S0_s0 | B | 取消：预注册规则只在水平有竞争力时才买"有序 vs 对称" | 未跑 |
| — | A2 / A3 | A | 取消：A 族被证明无统计功效，加深没有基准 | 未跑 |
| — | C | C | 不启动：S2 已否掉"图级组合缺失"的前提 | 未跑 |
| — | seed 2/3 | — | 不可用：只有 `pretrain_encoder_s{0,1}.pt`（seed 2 直接 FileNotFoundError） | 阻塞 |

**两次报废重跑（记账/健壮性，非科学失败）**：
1. `aas-A0-s0` 首跑用 `--device cuda:1`（单卡分配）→ `invalid device ordinal`；以 `cuda:0` 重跑。
2. `aas-S1-s0` / `aas-S2-s0` 首跑在**训练完成后**写 `train.json` 时因
   `EXPECTED_PARAMETERS['S1']` 缺键崩溃（`c453854` 修复）；两者重跑，浪费 ≈0.4 GPU·h。
   附带的副作用是一条有用的观测：同一命令、同一 commit 的 s1 两次运行
   best 分别是 0.086889 / 0.090343（CUDA scatter 的 `index_add_` 非确定性），
   与 §4 的功效论证互相印证。

**预注册 vs 事后**：所有判定规则（A 的 D1–D3、B 的 DEEPEN_B/STOP_B、A0M0 的
MOMENT_REDUNDANT/ESSENTIAL）都在对应 GPU 作业启动**之前**写进 `configs/*.json`；
`aas_freeze.py` 会记录 config/PLAN 的 sha256，`aas_analyze.py` 在 config 与
冻结记录不一致时拒绝分析。事后新增的只有：`paired_by_seed` 统计块、
`drop-top-1-contributor` 列，以及 stage a3 的消融臂（在启动前冻结）。
