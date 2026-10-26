# REPORT — `zinc_bosi_joint_moment_v1`（BOSI-JM：联合矩式邻键统计）

**轮次：** `tracks/structural-sufficiency/zinc_bosi_joint_moment_v1/`
**协议：** `PROTOCOL.md`（`zinc-bosi-joint-moment-v1`，GPU 之前提交）
**配置：** `configs/bjm_v1.json`
**代码 commit：** `fb90e66`（设计 + 协议 + 实现 + 12/12 CPU 门控）、`470b9f2`（远端 JSON
输出改到 gitignored 目录）；**全部 GPU 作业都跑在 `470b9f2` 上**。
**数据：** 官方 PyG ZINC `subset=True`，train 10,000 / valid 1,000，raw-y MAE。
**official test：全程未构造、未加载、未预测、未评分。**

**GPU 实耗：0.938 A100·h**（上限 5.0；两个 smoke 0.011 + B_s0 0.376 + JM_s0 0.552；
`analysis/budget_ledger.json`）。预注册的 seed1 与 CM 作业**均未触发**。

---

## A. 核心判断

**`NO_GO`：联合矩（JM）在官方 10k/1k 协议上没有超过 BOSI-B，反而差 +0.001683；
预注册的 seed-0 门槛两条都不成立，因此停在 seed0，不买 seed1、不跑 CM。**
同时有一条比"输赢"更重要的实测结论：**BOSI 主干本身在官方 10k 协议上并不与
DCC-L 打平**——BOSI-B 0.082405，比 DCC-L seed0 差 +0.003868、比 P3B seed0 差
+0.006181。v0 在 Fit8k 内部划分上看到的"打平"没有迁移过来。

| 问题 | 直接回答 |
|---|---|
| 哪个模型最强？ | **BOSI-B（473,889 参数），official valid best = 0.082405 @ epoch 439** |
| JM 是否超过 B？ | **否。** best_valid 0.084088，差 **+0.001683**；paired 1000 分子 ΔAE 的 95% CI [−0.006670, +0.013801] 跨 0，只有 50.4% 分子更好 |
| JM 是否只是"提高训练拟合"？ | **不是，恰恰相反。** JM 的 train@best 是 **0.019174**，B 是 **0.008536**（B 拟合得更狠）。所以这次的差异不是"过拟合换泛化"，JM 连训练拟合都更弱 |
| JM−CM 是否支持联合配对？ | **没有被检验**（CM 未触发，见 §F）。冻结权重上的 J→C 替换代价 **+0.021379** MAE、100% 分子移动，说明训练后的 JM **确实在用配对**；但"用了"不等于"更准"——这一点与 v0 的 R/B 完全同型 |
| 离 DCC-L / P3B 还有多远？ | BOSI-B 比 DCC-L seed0 差 **+0.003868**、比 DCC-L 四 seed 均值差 **+0.006415**、比 P3B seed0 差 **+0.006181**；JM 更远（+0.005551 / +0.008098 / +0.007864） |
| 下一步扩大还是关闭？ | **关闭这条架构线。** 用户侧"靠联合矩把 BOSI 推到 DCC-L 水平"的假设被本轮证伪；不扫 rank/宽度/LR/wd/horizon，不买 seed1，不跑 CM |

---

## B. 主表（official valid 1,000，raw-y MAE，best = 严格最小 valid）

| run | arm | seed | params | GPU-h | best_valid | best_ep | W20 | T100 | final_valid | train@best | train@final |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B_s0 | B | 0 | 473,889 | 0.373 | **0.082405** | 439 | **0.083203** | **0.083241** | 0.083242 | 0.008536 | 0.000593 |
| JM_s0 | JM | 0 | 475,425 | 0.548 | 0.084088 | 377 | 0.085327 | 0.085420 | 0.085394 | 0.019174 | 0.000476 |

GPU-h 列为 `train.json` 自报的 run wall（0.373 / 0.548）；按 rr job occupancy
（含启动与收尾）则为 0.376 / 0.552，两者合计加两个 smoke 见 §H。

定义与历史轮次一致：`W20` = 全部 20 连续 epoch 窗口中"窗口 valid 中位数"的最小值；
`T100` = 最后 100 epoch 的 valid 中位数；`train@*` = 固定 checkpoint 在完整官方
train 10k 上的 eval-mode MAE（不是 epoch 内 online 值）。

两条轨迹都完整跑满 **600 epochs / 47,400 updates**（79 steps/epoch），没有缩短、
没有重启、没有改 cosine。best checkpoint 与 final 的 valid 差 ≤ 1e-3，说明选点稳定。

**配对对比（负 = 前者更好；10,000 次固定分子重采样 bootstrap）**

| 对比 | ckpt | ΔMAE | 95% CI | CI 含 0 | 前者更优分子比例 | 去 top-10 后 | ordinary | tail | any_unseen | rings≥4 |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| JM − B | best | **+0.001683** | [−0.006670, +0.013801] | 是 | 0.504 | **−0.002447** | **−0.001518** | **+0.047716** | **−0.015418** | **−0.011602** |
| JM − B | final | +0.002151 | [−0.005688, +0.013703] | 是 | 0.511 | −0.002293 | −0.003151 | +0.078417 | −0.010608 | −0.008913 |

**分组绝对 MAE（best checkpoint）**

| 组 | n | B | JM | JM−B |
|---|---:|---:|---:|---:|
| overall | 1000 | 0.082405 | 0.084088 | +0.001683 |
| ordinary `y>-3` | 935 | 0.066988 | 0.065471 | −0.001518 |
| tail `y≤-3` | 65 | 0.304178 | 0.351894 | +0.047716 |
| radius-2 all_supported | 384 | 0.034816 | 0.035842 | +0.001026 |
| radius-2 low_support | 365 | 0.083100 | 0.097233 | +0.014133 |
| radius-2 any_unseen | 251 | 0.154202 | 0.138784 | −0.015418 |
| rings≥4（cycle rank） | 213 | 0.113172 | 0.101569 | −0.011602 |
| median AE | 1000 | 0.037000 | 0.036331 | −0.000669 |

support 只用**官方 train 10k** 的精确 typed radius-2 rooted signature 统计（不看标签）。

---

## C. JM 到底输在哪、赢在哪（全部为 exposed valid 上的描述性事实，不是新证据）

1. **一个分子决定了总体符号。** row 172（`y = −20.340`）的 AE：B 9.4145 → JM 14.1468，
   单项给 overall Δ 贡献 **+0.004732**。**去掉 row 172 后 JM 反而更好 −0.003053。**
   这正是 `zinc_dual_channel_composer_v1` 里拖死 DCC-L-L 的同一个分子
   （那里 AE 1.563 → 6.099）。官方 valid 的这个极值分子对任何总体均值比较都能
   搬动 ±0.005，远大于本轮关心的效应量。
2. **JM 在中位数与"结构上的难组"上略好**：median AE 0.0363 vs 0.0370；
   `any_unseen` −0.015418、`rings≥4` −0.011602、`ordinary` −0.001518。
3. **JM 在 `low_support` 上明显更差**（+0.014133），在 tail 上更差（+0.047716，
   其中 row 172 占 +4.73 AE）。
4. **两者方向不一致，不能讲成"JM 在难结构上更好"**：同一条机制在
   `any_unseen`（−0.015）与 `low_support`（+0.014）上符号相反，符号翻转的
   子群解释力接近于零；这不是可用结论。
5. **趋势**：valid 的 epoch 分位中位数

   | 分位 | B | JM |
   |---|---:|---:|
   | 前 25%（ep 1–150） | 0.0911 | **0.0907** |
   | 25–50%（ep 151–300） | **0.0865** | 0.0875 |
   | 50–75%（ep 301–450） | **0.0840** | 0.0857 |
   | 后 25%（ep 451–600） | **0.0833** | 0.0854 |

   JM 在前 150 epoch 略占先，之后被 B 反超并一路落后；`T100`（0.085420 vs 0.083241）
   与 `W20`（0.085327 vs 0.083203）同向，都不是"选点运气"。
6. **train–valid gap**：B 在 best checkpoint 上 train 0.008536 / valid 0.082405
   （gap 0.0739）；JM 0.019174 / 0.084088（gap 0.0649）。到 final 两条都几乎归零
   （train 0.00059 / 0.00048）而 valid 不再改善，即 **600 epoch 末尾两者都已完全
   记住训练集**；JM 在 best 点记得更少、泛化却更差，这是"机制没有带来有效归纳偏置"
   而不是"过拟合"的直接证据。

---

## D. 机制是否"活着"（冻结权重探针，`analysis/probe.json`，纯推理）

对 JM_s0 的 best checkpoint（CPU 重算 valid = 0.0840881518，与 train.json 差 2.9e-10）：

| 探针 | valid MAE | ΔMAE | mean\|Δy\| | 移动分子比例 |
|---|---:|---:|---:|---:|
| 原样（JM） | 0.084088 | — | — | — |
| **同权重、把 `J` 换成 `C`** | **0.105467** | **+0.021379** | 0.04629 | **1.000** |
| 同权重、边际保持的 w 重配对 | — | — | 0.24443 | 1.000 |
| 把 288 维联合块整体置零（分布外，仅参考） | 0.279236 | +0.195147 | 0.24342 | — |

读法：**训练后的 JM 严重依赖"同一伙伴内部先配对"这一项**。把 `J` 原样换成等参数、
同输入、同权重的 `C`，预测 100% 移动、MAE 掉 0.0214；把配对关系在保持两个边际
多重集完全不变的前提下重排，预测平均移动 0.244。这与 v0 的 R/B 关系置换结论同型：
**机制被用上了，而且量级不小；但用它并没有换来更好的泛化。**
第三行是分布外操作，只作量级参考，不能读成"联合块贡献 0.195 MAE"。

---

## E. CPU 门控与本轮的关键事前事实（`analysis/cpu_checks.json`，12/12 → `CPU_GO`）

| 门 | 结果 |
|---|---|
| G1 参数/形状 | B **473,889**、JM = CM **475,425**（+1,536 = 4×32×12，0.32%）；Side 617→96 不变；M 384；≤500k |
| G2 初始状态 | JM/CM 完整 state 逐张量 max\|Δ\| **0.0**（sha `abb42f6b…`）；三者公共 state 同 sha `f894c18a…`、逐张量 0.0 |
| G3 `n≤1` | 合成路径图 + 真实 valid 上 `n≤1` 行：`J ≡ C` 精确 **0.0** |
| G4 中心化恒等式 | `n≥2` 1,423 行，`J−C = Σ_f(u_f−ū)(w_f−w̄)^T`，float64 参考 max\|Δ\| **3.24e-7**（相对 3.5e-7） |
| G5 配对反例 | 二维集合 `{(.5,.5),(−.5,−.5)}` vs `{(.5,−.5),(−.5,.5)}`：逐坐标 0–6 阶矩完全相同，JM 分开 **1.0**、CM 分开 **0.0**；三伙伴集合 `{(0,0),(1,1),(2,2)}` vs `{(0,2),(1,1),(2,0)}`：JM 5 vs 1、CM 3 vs 3。真实分子上边际保持重配对：JM 移动 1.23e-3（100%），CM 8.9e-8（纯 float32 加法顺序） |
| G6 真实统计 | `n`=0/1/2/3 → **8.25% / 47.05% / 42.21% / 2.49%**；`n≥2` **44.70%**；JM−CM 输入最大相对差 **0.427**，`n≥2` 上 **97.0%** 的半键改变；`Σ_r n[r] = deg−1` 全成立；A 块两臂逐位相同（rms 0.917） |
| G7 不变性 | 原子重编号 / 键向反转 / 键序 / batch 序 / batch 重组 全部 ≤ 1e-5（实测 1e-7 量级） |
| G8 梯度 | 全有限；B 6 个模块、JM/CM 10 个模块梯度全非零；四个投影梯度非零（~4e-3 – 6e-3） |
| G9 batch 完整性 | 无跨分子 partner、无重复 partner、`Σ_r n[r] = 伙伴数` |
| G10 来源 | train/valid split 正确；`norm.json` 由 train 10k 逐坐标重算（mean 1.9e-7 / scale 2.7e-8，唯一零方差坐标 19 用 scale=1.0 的冻结约定）；SSL checkpoint 只含 `atom_encoder`/`bond_encoder`/`junction_encoder`（无 side/compose/reader）；代码中无 test 访问；valid 标签只出现在 5 行 eval/hash 语句 |
| G11 checkpoint | 三臂保存/加载后预测逐位相同 |
| G12 全缓存前向 | B/JM/CM 在完整 train 10k 上前向有限；train/valid 关系缓存内容哈希重建一致（`c698a2f3…` / `6505e23b…`） |

**"若真实 JM−CM 机制输入差异几乎为零则停止"这一停止条件未触发**：`n≥2` 覆盖率
44.70%、初始化时相对差 0.427、97.0% 的 `n≥2` 半键改变。所以本轮**不是**"机制没差别
所以没赢"，而是"机制差别很大、模型也确实在用，但没有赢"。这个区分是本轮最有价值
的产物。

---

## F. seed0 判决（预注册，逐条）

```
A. JM best_valid <= 0.075                    -> 0.084088  ✗
B. JM - B <= -0.003 且 JM best_valid <= 0.078 -> +0.001683 且 0.084088  ✗
guards: ordinary 不恶化                       -> -0.001518  ✓
        W20 不出现反方向大幅退化                -> +0.002124  ✗（同向变差，超过 0.002）
=> buy_seed1 = false
```

两条主条件都不成立，guard 也不同时成立 → **停在 seed0**。
按 §7/§8 的预注册规则：**不买 seed1、不跑 BOSI-CM、不扫 rank/宽度/LR/wd/horizon、
不改选择策略**。因此：

* **BOSI-CM 从未训练**，本轮没有 `CM` 的任何实测数字；
* `JM − CM` 的"训练层面"比较**无法给出**，只能给出 §D 的冻结权重替换代价
  （+0.0214，机制活着）；
* 因此问题"JM−CM 是否支持联合配对机制"的诚实答案是
  **"未检验；已有证据只支持'训练后的 JM 依赖配对'，不支持'配对带来更好的泛化'"**。

预算仍剩 4.06 A100·h。**不用剩余预算事后购买未注册的作业**：这正是
"不得见到分数后更改选择策略"所禁止的动作。

---

## G. 与历史参照的距离（同协议，但历史值，不重训、非本轮匹配对照）

| 模型 | params | official valid | vs BOSI-B | 来源 |
|---|---:|---:|---:|---|
| P3B (xBssl_P3N_s0) | 413,505 | **0.076224** | −0.006181 | `zinc_recipe_trunk_v2` |
| DCC-L 四 seed 均值 | 494,193 | **0.075990** | −0.006415 | `zinc_performance_search_v1` |
| DCC-L seed0 (B-R0) | 494,193 | **0.078537** | −0.003868 | `zinc_dual_channel_composer_v1` |
| **BOSI-B seed0（本轮）** | 473,889 | 0.082405 | — | 本轮 |
| BOSI-JM seed0（本轮） | 475,425 | 0.084088 | +0.001683 | 本轮 |

两条新轨迹都在历史上已暴露的 official valid 上评估，**不是新的盲测**；
上面的差值是点估计，不构成 DCC-L/P3B 的显著性主张。

**新的负面事实（本轮真正新增的）**：v0 的"BOSI 主干与 DCC-L 打平"是在
Fit8k/Monitor/Locked 的内部划分上得到的（Locked seed-mean B 0.081970 vs
DCC-L 0.083498）。迁到官方 10k→1k 后，同一个 B 主干拿到 0.082405，比 DCC-L seed0
差 0.0039。**协议换了、结论也换了**：BOSI 主干相对于 DCC-L 的"打平"至少不能
在官方协议上复现，因此 v0 的"主干值得保留为低成本可解释基线"只能限定在
Fit8k 内部划分那个语境里。

---

## H. 执行记录

| 项 | 值 |
|---|---|
| 设计 commit | `fb90e66`（DESIGN + PROTOCOL + 配置 + 实现 + cpu_checks.json） |
| 运行 commit | `470b9f2`（`BJM_ANALYSIS_DIR` 让远端 JSON 落在 gitignored 目录，避免脏 worktree 阻断 deploy） |
| `rr doctor res` | 10 ok / 0 warn / 0 fail；2×A100-SXM4-40GB，driver 550.163.01，torch 2.5.1+cu124；历史 NVML 故障**未复现** |
| `rr deploy res` | `fb90e66` → `470b9f2`（两次增量 fast-forward） |
| `bjm-prep` | `bjm-prep-20261026-220529-b82e9068`，CPU pool，19 s；train/valid 关系缓存内容哈希与本地一致 |
| `bjm-smoke-B` / `bjm-smoke-JM` | `…-53e440c3` / `…-3a714471`，各 3 epoch；稳态 2.16 / 2.91 s·epoch⁻¹，投影 0.359 / 0.485 h（`analysis/timing.json`） |
| `bjm-B-s0` | `bjm-B-s0-20261026-222808-f8e2e1d4`，res-gpu0，22:28:11→22:50:44，**0.376 h** |
| `bjm-JM-s0` | `bjm-JM-s0-20261026-222831-0e591faf`，res-gpu1，22:28:34→23:01:40，**0.552 h** |
| **合计** | **0.938 A100·h / 5.0**（`analysis/budget_ledger.json`，按 job occupancy × 1 GPU 计） |

smoke 与正式 run 的 warmup 差异需要说明：trainer 沿用历史实现
`warmup = min(500, updates // 20)`，所以 237-update 的 smoke 里 warmup 被压到 11；
**正式 run 的 47,400 updates 下 warmup = 500**，与协议一致（曲线里 step 500 处
lr ≈ 1.0e-3 可核对）。

同 seed 一致性（`runs/*/train.json`）：
`initial_common_state_sha256 = f894c18a…`（B/JM 相同）、`first_batch_rows_sha256`
两条完全相同、`generator_state_sha256` 相同、`train_rows_sha256` /
`valid_rows_sha256` 相同、`norm_sha256` 相同、SSL sha `006fabc1…`。

## I. 复现

```bash
# CPU 门控（~19 s，12/12 -> CPU_GO）
uv run python tracks/structural-sufficiency/zinc_bosi_joint_moment_v1/code/bjm_checks.py --rebuild

# 关系缓存（本地或远端；内容哈希会被核对）
uv run python tracks/structural-sufficiency/zinc_bosi_joint_moment_v1/code/bjm_prep.py

# 正式训练（每条 47,400 updates）
BJM_ANALYSIS_DIR=<gitignored> python .../code/bjm_train.py --arm B  --seed 0 --device cuda:0
BJM_ANALYSIS_DIR=<gitignored> python .../code/bjm_train.py --arm JM --seed 0 --device cuda:0

# 冻结 -> 分析 -> 冻结权重探针（全 CPU，~1 min）
uv run python .../code/bjm_freeze.py --runs B_s0 JM_s0
uv run python .../code/bjm_analyze.py
uv run python .../code/bjm_probe.py --runs JM_s0 B_s0
```

产物：`analysis/{cpu_checks,prep,timing,freeze,summary,decision,groups,probe,budget_ledger,manifest}.json`、
`analysis/tables.md`、`analysis/per_molecule_valid.csv`；
checkpoint 与曲线在 gitignore 的 `runs/{B_s0,JM_s0}/`（`best.pt` sha256
B `5e756a61…`、JM `a781c1f1…`，完整值见 `analysis/manifest.json` 与 `freeze.json`）。

### I.1 一个记录型修正（不影响任何结果）

第一次生成的 `analysis/cpu_checks.json` 是 29 MB：`G3` 把两个 `[H, 288]` 的原始块数组
直接留在了返回值里。改为只保留摘要数字后重新生成，文件 32 KB，**12/12 仍然全过、
所有判定数字逐项相同**（`n_le1_rows` 870 / `max_abs_diff_on_n_le1` 0.0 /
`max_abs_diff_on_n_ge2` 0.9065 / `pass` true）。因为 `cpu_checks.json` 的字节变了，
`configs/bjm_v1.json` 的 `cpu_gates.sha256` 与 `freeze.json` 的 `cpu_gates_sha256`
随之更新，`bjm_freeze.py --force` 与 `bjm_analyze.py` 各重跑一次；
**两次 freeze 的 checkpoint / 预测 sha256 完全相同**（`5e756a61…` / `a781c1f1…`），
所以这是序列化修正，不是结果筛选。`bjm_train.py`、`bjm_models.py`、`bjm_common.py`
与 GPU 运行 commit `470b9f2` **逐字节相同**（`git diff --quiet 470b9f2` 通过）。

## I.2 文档时间线

GPU 作业跑在 `470b9f2`（模型/数据/训练三个文件与 `fb90e66` 在训练路径上无实质差异，
唯一变化是远端 JSON 目录）。`DESIGN.md` 在结果提交里做过**一次纯文档修正**：
§5 交付清单里误写的 `bjm_data` 改成实际存在的文件列表。冻结的设计内容
（问句、机制定义、参数预期、判决规则、禁止清单）**一字未改**；`configs/bjm_v1.json`
的 `design_sha256` / `protocol_sha256` 已按最终字节更新，`PROTOCOL.md` 未改动。

## J. 明确不做的事

不买 seed1；不跑 CM；不扫 rank / 宽度 / reader / LR / wd / horizon；
不改 47,400 updates；不改 valid 选点规则；不重训 DCC-L/P3B；
不动 official test；不改写 v0 / v1 / partner-content / stat-SSL / performance-search
任何历史文档与结论。
