# BOSI v0 — 键中心、目标键条件化结构统计：能不能顶替 learned junction？

轮次 `zinc_bosi_architecture_v0`。设计基线 `5950260`（`DESIGN.md` 未改动）。
实现/协议/门控 commit `41bdbe6`，计时参数 `55476bf`，远端结果目录 `c339f47`，
分析侧修正 `329ca54` + `47e80b3`（模型/数据/训练三个文件与 `c339f47` 逐字节相同）。
**官方 valid / test 未构造、未加载、未预测、未评分。**

GPU 预算：A100-SXM4-40GB，**1.825 A100·h**（硬上限 2.5，按每个 job 的 wall time 计，每个 job 占 1 卡），4 个正式 run + 1 个计时 run + 1 个关系缓存 run。

---

## A. 核心判断

**`NO_GO`（针对"目标键条件化的伙伴矩分配"这一机制），但主干本身是"打平"而不是"退步"。**

三句话：

1. **R 和 B 没有可重复差异。** Locked 上 `R−B`：seed0 **−0.00206**、seed1 **+0.00155**，
   seed-mean **−0.00026**；两个 seed 的 95% CI 都跨 0。预注册判决要求"两 seed 同向且
   seed-mean ≥ 0.003"，实际是**方向相反**。→ 真分配（per-relation partition）没有带来收益。
2. **但这个机制在模型里是"活的"，不是"没学会所以没差别"。** 冻结权重上把 R 的关系标签
   做直方图保持置换，预测平均移动 **0.178**（98% 的分子动了）；把 R 权重换成 B 统计，
   Locked 从 0.0804 掉到 **0.1118**（+0.031）。R 确实学会了用"哪个伙伴落在哪个关系组"，
   只是**用了也不更准**。
3. **主干（无 junction 的键中心统计池）与未重训的 DCC-L 打平、点估计略优。**
   Locked seed-mean：R **0.08171** / B **0.08197** / DCC-L **0.08350**。
   R−DCC-L = **−0.00178**，B−DCC-L = **−0.00153**；四个同 seed 对比里 **3 个为负**，
   但**每一个 95% CI 都跨 0**。这是"不可区分"，不是"超过"。相对 RBO/SCM/ESOC/SPS 的
   +0.004…+0.05，这是第一次有键中心/无 junction 结构方案站到 DCC-L 同一水平线上，
   但**没有**达到 0.005 的实用门槛，**不触发任何 0.05x 讨论**。

因此：**统计主干值得保留为"低成本可解释基线"，关系分配机制关闭。**

---

## B. 审计结论（详见 `AUDIT.md`）

前向级别的差异只有一条是真正新的：

| 维度 | RBO | SCM | DCC-L | BOSI |
|---|---|---|---|---|
| 中心 | 原子 | 冻结 DCC-L | 键（learned junction） | 键 |
| 统计条件 | 无（unconditional 幂和） | 无（line-graph 带内内容注意） | 无（junction 由局部几何学习） | **目标键 `(v,e)` 条件化** |
| 矩阶 | 1/2/3（原子根） | — | — | 1/2/3（**每关系组**） |
| 跨通道配对 | 有（`g(rel_ij)·x_i·x_j` 全局和） | 无 | 无 | **有，且按关系组分别求和** |
| 邻居类型 | 无显式分布 | 无 | 隐式 | **显式 `n[r]` + `closure`** |
| 闭合环归属 | 无 | 无 | 隐式 | **显式 R 组（同 biconnected block）** |

两条必须公开的"非新"事实（`AUDIT.md` §4）：

* **B 臂的矩块是 RBO 根矩的仿射像**（n≤3、逐通道）；在给定 `(p_k, x_e)` 时，留一法
  条件化不增加信息。真正新的只有 **`r(v,e,f)` 这个目标键关系划分**。
* "归属 vs 广播"此前已有一个 −0.00383 的单 seed 先验（DCC-L 的 L 臂 vs C 臂），
  以及 ESOC 的 REAL−NULL = −0.0282；本轮的 R−B 正是这个问题的第三次、也是最高功率的一次测试。

风险量化（事前写进 `PROTOCOL.md`）：分配只在 **19.50%** 的半键上有定义；
T 关系占 0.545%、Q 占 0.192%。这决定了本轮的上限本来就在 0.002–0.005 量级。

---

## C. 架构、参数与合规

`code/bosi_models.py`，`named_parameters()` 重算：

| 模块 | 参数 |
|---|---:|
| atom encoder 108→128→96 | 26,336 |
| bond encoder 295→160→96 | 62,816 |
| half encoder 295→128→32 | 42,016 |
| side state（每端点 96 维） | 91,488 |
| symmetric bond compose | 78,176 |
| reader `[Σc, Σc², Σa, Σa², global33]` 417→256→256→1 | 173,057 |
| **合计** | **473,889**（cap 500,000） |

* 关系构造（`code/bosi_data.py`，**不用标签**）：T（远端成键）> Q（额外公共邻居）>
  R（同 biconnected block 且 ≥3 条边）> N（无闭合）。优先序硬编码、互斥、MECE。
* 双臂共享同一初始状态：seed0 `ceeb598f…`、seed1 `c93ea85b…`；R/B 的 `initial_state_sha256`
  完全相同，`first_batch_rows` 完全相同。
* 只从 train-only masked-SSL 加载 `atom_encoder` + `bond_encoder`
  （`bjoc_full10k_pretraining_transfer_v1/pretrain_encoder_s{0,1}.pt`，
  sha256 `006fabc1…` / 另一个 seed），`junction_encoder_loaded=false`，**没有**加载任何
  DCC-L 监督权重。
* 禁用清单逐项确认：无 learned junction encoder、无递归 MP、无 Transformer/attention、
  无新查表、无外部预训练、无新标签分量、无蒸馏、无 ensemble、无残差旁路。

## D. CPU 门控（`analysis/cpu_checks.json`，12/12，`CPU_GO`）

| 门 | 结果 |
|---|---|
| G1 度界/伙伴数 | max deg 4，max 3 partners，**无截断** |
| G2 关系 MECE | `Σ_r n[r] = deg−1`，无重复计数 |
| G3 对原图的结构校验 | T 伙伴行 168 = 期望，Q 40 = 期望，**与 networkx/atom_feat/bond_feat 零不一致** |
| G4 矩的计数器反例 | 反例成立：A 的 `ΣT3` = 0.0 vs B 的 0.5774，说明"逐通道幂和 ≠ 全对和" |
| G5 跨通道投影反例 | 4 vs 2（`(a+b)² ≠ a²+b²`） |
| G6 NumPy 参考实现 | R 的 `M` 与手写参考 max\|Δ\| **1.1e-7** |
| G7 对称/批次不变 | 重编号+键序 max\|Δy\| **2.4e-7**；batch 不变同量级 |
| G8 R/B 同初始 | 同 sha256 `ceeb598f…`，张量差 **0.0** |
| G9 R/B 都活着 | 关系置换使 R 移动 4.2e-4、使 B **恰好 0.0** |
| G10 真实 Fit8k 差异分布 | 分配有定义 19.50%；相对差 mean 2.82% 全体 / 13.28% 有定义者；Bfull 对照 10.73% |
| G11 梯度 | 六个模块梯度**全非零**（训练期 step999：reader 3.39…half_encoder 0.186） |
| G12 无崩塌 | 无尺度爆炸，loss 正常下降 |

`Bfull`（DESIGN 字面版：`n[r] · Sloo`）只在 CPU 诊断里出现，**从未训练**；正式 B 臂用
协议修正 A1 的均值广播 `n[r]/max(deg−1,1) · Sloo`，两臂的 `Σ_r M[r]` 完全相同。

## E. GPU 执行记录

| run | run_id | commit | arm/seed | best epoch | best Monitor | gpu_h |
|---|---|---|---|---:|---:|---:|
| bosi-prep | `bosi-prep-20261026-154034-a58ad218` | 41bdbe6 | relation cache | — | — | 0.005 |
| bosi-timing | `bosi-timing-20261026-154456-0fa5d1f0` | c339f47 | R/s0（10 ep） | 10 | 0.2029 | 0.008 |
| R_s0 | `R_s0-20261026-154833-f1f1eb64` | c339f47 | R/0 | 527 | **0.121329** | 0.391 |
| B_s0 | `B_s0-20261026-154848-9116216c` | c339f47 | B/0 | 549 | **0.124505** | 0.499 |
| R_s1 | `R_s1-20261026-162148-9e527bcf` | 329ca54 | R/1 | 569 | **0.124166** | 0.410 |
| B_s1 | `B_s1-20261026-162227-b23a841d` | 329ca54 | B/1 | 556 | **0.122400** | 0.512 |

配方与 DCC-L 完全一致：Fit8k、batch 128、47,400 updates（753 ep × 63 step）、AdamW
1e-3→1e-5 per-step cosine、warmup 500、wd 1e-5、clip 5、L1 raw-y、generator seed+101、
Monitor 严格最小选点。**没有缩短 updates。**

**买 seed1 的预注册规则**（只看 Monitor）：
停止条件 = `R_s0 > 0.123151+0.012 = 0.135151` **且** `R_s0−B_s0 > −0.003`。
实际 `R_s0 = 0.121329`（不满足左项）、`R_s0−B_s0 = −0.003176`（不满足右项）→ **两个条件都不成立
→ 按规则购买 seed 1**。事后看这一步是必要的：seed1 上 R/B 的符号反了过来。

计时：稳态 **1.68–2.72 s/epoch**（两卡并行时偏慢），单 run 投影 0.36–0.51 h。

## F. Locked 结果（一次性读取，`analysis/locked_read.json`，n=1000）

| run | arm/seed | Locked MAE（best-monitor） | Monitor |
|---|---|---:|---:|
| R_s0 | R/0 | **0.080390** | 0.121329 |
| B_s0 | B/0 | 0.082449 | 0.124505 |
| R_s1 | R/1 | 0.083037 | 0.124166 |
| B_s1 | B/1 | 0.081491 | 0.122400 |
| DCC-L s0（历史，未重训） | — | 0.083573 | 0.123151 |
| DCC-L s1（历史，未重训） | — | 0.083423 | 0.127772 |
| **seed-mean R / B / DCC-L** | | **0.081714 / 0.081970 / 0.083498** | |

配对对比（负 = 前者更好；bootstrap 10,000 次，固定行重采样）：

| 对比 | ΔMAE | 95% CI | 更优分子比例 | 去 top-10 |
|---|---:|---|---:|---:|
| R−B（s0） | −0.002058 | [−0.005559, +0.001307] | 0.505 | −0.000942 |
| R−B（s1） | **+0.001546** | [−0.001693, +0.004873] | 0.492 | +0.001690 |
| R−DCC-L（s0） | −0.003183 | [−0.007361, +0.001072] | 0.548 | −0.002346 |
| R−DCC-L（s1） | −0.000386 | [−0.004016, +0.003185] | 0.504 | +0.000774 |
| B−DCC-L（s0） | −0.001125 | [−0.005465, +0.003245] | 0.502 | −0.001433 |
| B−DCC-L（s1） | −0.001932 | [−0.005725, +0.001817] | 0.499 | −0.000987 |

**每一个 CI 都跨 0。** 更优分子比例全部在 0.49–0.55，即抛硬币。

分组（best-monitor，绝对 MAE）：

| 组 | n | R_s0 | R_s1 | B_s0 | B_s1 |
|---|---:|---:|---:|---:|---:|
| overall | 1000 | 0.08039 | 0.08304 | 0.08245 | 0.08149 |
| ordinary (`y>−3`) | 932 | 0.06490 | 0.06608 | 0.06583 | 0.06493 |
| tail (`y≤−3`) | 68 | 0.29266 | 0.31539 | 0.31029 | 0.30852 |
| low_support | 398 | 0.05825 | 0.06016 | 0.06111 | 0.06022 |
| any_unseen | 287 | 0.15681 | 0.16013 | 0.15805 | 0.15499 |
| all_supported | 315 | 0.03874 | 0.04170 | 0.04053 | 0.04140 |
| has_T_or_Q | 75 | 0.08456 | 0.07899 | 0.07558 | 0.06119 |
| alloc_defined_top_quartile | 252 | 0.08222 | 0.09131 | 0.09013 | 0.08777 |
| rings_ge4 | 212 | 0.12906 | 0.14027 | 0.14050 | 0.13787 |

* 相对 DCC-L，**ordinary 四个对比全为负**（−0.0012…−0.0034），tail 基本持平（±0.01 内摆动）。
  即"不是靠尾部取胜"，但也没有尾部收益。
* `any_unseen`：R_s0 **+0.00146**（比 DCC-L 差），R_s1 −0.00263；B_s1 −0.00777 更好。
  **两 seed 不同向** → "any-unseen 同向为正"这一条不成立。
* `has_T_or_Q`（真的含 T/Q 关系的 75 个分子）上，**R 反而比 B 差**（+0.009 / +0.018）。
  这是本轮最反直觉的一条：在"分配最有信息"的子群里，真分配没有赢，反而输。

## G. 冻结权重诊断（全部推理，无训练）

| run | 换臂后 MAE | M 块置零 | side 全置零 | 关系置换 mean\|Δy\| |
|---|---:|---:|---:|---:|
| R_s0 | 0.111759 | 0.292847 | 0.317174 | **0.177518** |
| B_s0 | 0.092184 | 0.301454 | 0.336774 | **0.000000** |
| R_s1 | 0.109913 | 0.324171 | 0.306990 | **0.173682** |
| B_s1 | 0.091666 | 0.286573 | 0.328032 | **0.000000** |

读法（**这是本轮最有信息量的部分**）：

* **关系置换**：R 移动 0.178（98% 分子），B 恰好 0.0。这证明 R/B 不是"同一模型两种写法"，
  训练后仍然严格按设计分叉——R 的预测确实依赖"哪个伙伴进哪个组"。
* **换臂**：R 权重 + B 统计 → +0.031；B 权重 + R 统计 → +0.011。两个模型学到的**不是同一个函数**。
  结合上一条：**R 把分配用起来了，但用起来没有换来精度。** 这是一条干净的"机制存在但无价值"结论，
  不是"没学会"。
* **M 块置零 / side 全置零**：0.29–0.34，量级相同。这两条**不能**解读为"统计值 0.21 MAE"——
  把 384 维输入整体置零是分布外操作，破坏的是 side MLP 的输入统计而不只是信息。
  它们的正确用法是与"换臂"（同分布、合法统计）对照：**同分布的统计扰动只值 0.011–0.031，
  分布外置零值 0.21。** 本轮报告按这个口径引用。
* `side_zero` 把 side 通道清成只剩 `[a_v, b_e]`（相当于退回"键对编码器 + 对称 compose"），
  误差 0.307–0.337，说明主干里绝大部分信号确实来自 side/statistics 通道。

## H. 判决逐项（预注册条件）

| 条件（`PERFORMANCE_PROMISING`） | 实际 | 成立 |
|---|---|---|
| R 两 seed 都优于 DCC-L | s0 −0.00318、s1 −0.00039 | ✅ |
| R 的 Locked seed-mean 改善 ≥ 0.005 | −0.001785 | ❌ |
| ordinary 同向为正 | 四对比全负 | ✅ |
| any-unseen 同向为正 | s0 +0.00146 | ❌ |
| 去 top-10 后仍为正 | s1 +0.00077 | ❌ |
| R 稳定优于 B（≥0.003，两 seed 同向） | s0 −0.00206 / s1 +0.00155 | ❌ |

→ `PERFORMANCE_PROMISING` 不成立；`STRUCTURE_MECHANISM_ONLY` 也不成立（R 没有稳定优于 B）；
按预注册规则落入 **`NO_GO`**（"R 不优于 B"）。`INCONCLUSIVE` 不适用：R−B 的符号在两 seed
上相反，这不是"区间不够窄"，而是**点估计本身就在 0 附近来回**。

## I. 偏差、意外与限制

1. **远端 checkout 无法 fast-forward**：`rr deploy` 因远端仍停在 rebase 前的 `9a08888` 而失败。
   处理：远端 worktree 干净且 deploy 已把 `refs/rr/deploy/main` 取到 `41bdbe6`，
   故在远端做了一次 `git reset --hard refs/rr/deploy/main`（只影响计算副本，无本地提交丢失），
   之后 deploy 正常。这是基础设施修复，不改任何实验内容。
2. **结果目录污染**：首版脚本把 JSON 写进受版本控制的 `analysis/`，导致远端 worktree 变脏、
   下一次 deploy 被拒。改为 `BOSI_ANALYSIS_DIR` 覆盖到 gitignore 的 `runs/`；本地 curated 副本
   仍在 `analysis/`。`prep.json` 是本地 CPU 版本（含本地路径），远端只写 `runs/`。
3. **seed0 与 seed1 不在同一 commit**：seed0 在 `c339f47`，seed1 在 `329ca54`。
   两者之间**模型/数据/训练三个文件逐字节相同**（`git diff --stat` 为空），改动只涉及
   freeze/locked/analyze 与一个路径 helper。已在 §头部与本节记录，不隐藏。
4. **freeze 容差**：GPU 记录的 Monitor 与 CPU 重算差 5e-9…1.7e-8，故把 freeze 的校验容差
   从 1e-9 放宽到 1e-6。这是数值容差，不是结果筛选；8 个 checkpoint 的差值全部在 1.7e-8 内。
5. **一次性 Locked 读取**：`bosi_locked.py` 在 `freeze.json` 存在且全部 hash 通过后才写
   `locked_read.json`；二次调用只校验不重算。读取前的第一次调用因 `ad_common` 导入顺序失败
   （`ModuleNotFoundError`，**未产生任何预测**），修正后完成唯一一次成功读取（commit `47e80b3`）。
6. **Locked 不是新盲测**：该划分历史上已被 DCC-L（4 seeds）、ESOC、additive-reader 读过。
   本轮所有结论的统计强度都受此限制，不得表述为"盲测确认"。
7. **统计功效**：Locked n=1000，配对 bootstrap CI 半宽约 ±0.004。R−B 与 R−DCC-L 的真值
   若在 0.002 量级，本设计**不可能**分辨。这是事前就知道的（`PROTOCOL.md` §2 风险量化），
   也是为什么 §A 只声明"打平"而不声明"超过"。
8. **单 seed 的 B−DCC-L 也略优**（−0.0015），说明这点"优势"更可能来自**整个主干的架构差异**
   （键中心 + 显式统计 + 全局 33 维），而不是关系分配；而主干差异本身也没有超过噪声。

## J. 复现

```bash
# CPU 门控（约 1 分钟）
python code/bosi_checks.py                     # -> analysis/cpu_checks.json, CPU_GO

# 关系缓存（远端，6 s）
python code/bosi_prep.py                       # -> cache/bosi_rel_train.pt (content c698a2f3…)

# 正式训练（每个 0.39–0.51 A100·h）
python code/bosi_train.py --arm R --seed 0 --device cuda:0
python code/bosi_train.py --arm B --seed 0 --device cuda:0
python code/bosi_train.py --arm R --seed 1 --device cuda:0
python code/bosi_train.py --arm B --seed 1 --device cuda:0

# 冻结 -> 一次性 Locked -> 分析（CPU，约 10 s）
python code/bosi_freeze.py --runs R_s0 B_s0 R_s1 B_s1
python code/bosi_locked.py
python code/bosi_analyze.py                    # -> analysis/{tables.md,summary.json,decision.json}
```

预测数组按仓库惯例不入库（`*.npz` 全局忽略），其 sha256 逐条记在 `locked_read.json`。

产物：`analysis/cpu_checks.json`、`analysis/prep.json`、`analysis/freeze.json`、
`analysis/locked_read.json`、`analysis/locked_preds/*.npz`、`analysis/summary.json`、
`analysis/groups.json`、`analysis/decision.json`、`analysis/tables.md`。
checkpoint sha256（best-monitor）：R_s0 `32dedc27…`、B_s0 `c412813c…`、R_s1 `48d1767c…`、
B_s1 `14fe08c1…`（完整值在 `freeze.json`）。

## K. 明确不做的事

不重训 DCC-L；不追加第三个 seed；不改划分、不扩样本；不加宽网络、不加矩阶数、不增删关系类型；
不启动 Stage 2 attention（`ATTENTION_NEXT.md` 只是设计稿，前提条件未满足）；
不因 0.001–0.002 的弱信号讨论 official 10k 或"0.05x"；不把"打平"写成"超过"。
