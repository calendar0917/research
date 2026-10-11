# REPORT — cscl-structural-relation-composer-v1 (BJOC)

日期: 2026-10-12 · Track: `structure-contribution` · round: `structural-relation-composer-v1`
协议: `configs/structural_relation_composer_v1.yaml`（正式 GPU 运行前冻结）
驱动: `code/run_structural_relation_composer_v1.py` · 模型: `code/structural_relation_composer_v1.py`
对象: `code/structural_relation_objects_v1.py` · 测试: `tests/test_structural_relation_composer_v1.py` (16 fast passed)
远程: `rr` / host `res` / pool `res-gpu1` / A100-40GB, torch 2.5.1+cu124
完整数值: `round_summary.json`、`locked_dev_eval.json`、`official_valid_eval.json`、
`mechanism_diagnostics.json`、`train_*.json`（本目录，git-ignored）

## 0. 问题与设计

同一套原子/物理键/junction 对象、同一组参数、同一数据顺序下，**保留真实的
junction→bond 归属关系**（`W_b=Σ_{j∈b}w_j`, `V_b=Σw_j²`）是否比**保留全部边缘
分布但打散归属**的对照臂（`W_b=n_b·graph_mean(w)`, `V_b=n_b·graph_mean(w²)`，
`Σ_b W_b`、`Σ_b V_b` 逐分子守恒）带来可泛化的预测增益？同时检验 BJOC 这个
严格 one-shot（无 GNN / attention / 迭代消息传递）对象组合器本身是否可用。

- 参数 413,505（匹配参考 Full-Y 408,651）；模型 DAG 深度 = 对象层级，无回边。
- 新分组 split（按 canonical SMILES，`sha256(seed|smiles)` 排序贪心填充）：
  fit 8000 / monitor 1000 / locked 1000；重复分子不跨界；prep 只用 fit。
- 训练配方（三臂一致）：Adam 1e-3 / wd 1e-5 / bs 128 / 240 epochs / clip 5 /
  仅 raw-y L1 / 末 5 轮（236–240）参数平均 soup。

## 1. 主结果

| 臂 | monitor s0 | monitor s1 | locked dev | valid(全10k 候选) |
|---|---:|---:|---:|---:|
| F — 匹配 Full-Y | 0.11219 | — | 0.10189 | —（历史参考 0.118618） |
| N — BJOC null | 0.10022 | 0.10204 | 0.09996 | 0.103423 |
| O — BJOC organized | **0.09899** | **0.09945** | **0.09723** | 0.108399 |

配对 bootstrap（按分子重采样 4000 次）：

| 对比 | monitor s0 | monitor s1 | locked dev | valid(全10k) |
|---|---:|---:|---:|---:|
| Δ_org = O−N | −0.00123 [−0.00656,+0.00447] | −0.00259 [−0.00859,+0.00323] | −0.00273 [−0.00828,+0.00294] | **+0.00498** [−0.00198,+0.01115] |
| O−F | −0.01320 [−0.02056,−0.00552] | −0.01274 [−0.02163,−0.00291] | −0.00466 [−0.01246,+0.00419] | 见下 |
| N−F | −0.01197 [−0.01964,−0.00444] | −0.01015 [−0.02060,+0.00227] | −0.00193 [−0.01032,+0.00812] | 见下 |

- 分布统计（locked dev, MAE/RMSE/median/p90）：F 0.1019/0.1680/0.0661/0.2161，
  N 0.1000/0.1960/0.0619/0.2052，O 0.0972/0.1831/0.0626/0.1961 —— Full-Y 尾部
  更稳（max 1.47 vs 3.3），BJOC 的均值/中位/p90 更好。
- official valid（1000，单次读取，先冻结再读标签）：N 0.103423 / O 0.108399，
  RMSE 0.2559 / 0.2158；历史 Full-Y 0.118618 / RMSE 0.598707（不同轮次、不同
  seed 与统计，**非匹配臂**，仅作披露参考）。四臂预测与 checkpoint 的 sha256
  已写入 `official_valid_freeze.json` 后才读 y。
- 结论方向：**预测器**一致优于匹配 Full-Y（6/6 匹配对比为负；3/6 的配对 CI
  不含 0，全部在 monitor 上）；**归属效应**方向不稳定且无一 CI 排除 0
  （monitor s0/s1、locked 为负，official valid 为正）。

## 2. 配对与可复现性审计

| 项 | 结果 |
|---|---|
| N/O 初始化哈希 | 三对（s0、s1、full10k）逐对完全相同 `init_state_sha256` |
| 训练超参 | 逐对完全相同（含 `train_shuffle_seed=101`） |
| config 差异 | 唯一差异 = `organization_mode`（organized vs null） |
| 数据顺序 | batch order = f(len(fit_rows), seed 101, epoch)，训练循环无臂相关 RNG |
| 参数量 | 413,505（两臂同）；Full-Y 408,651 |
| 墙钟 | F 1405 s；N/O s0 385/381 s；s1 399/388 s；full10k 493/438 s |
| commit | 950448e0（s0 三臂）、6c7eb168 后（s1）、828c587e（full10k + eval） |
| official valid/test | valid 单次（冻结后）；**test 从未加载** |

## 3. 预注册门槛状态与条件性阶段

- monitor 对 F 改进 ≥0.010：**通过**（O −0.0132）。monitor 对 N 改进 ≥0.005：
  **不通过**（O −0.0012）。locked 方向一致：通过。故本轮**未达到**“强信号”
  标准，归属假设 H-ORG 未被确立。
- 资源规则（seed-1 / full-10k）**未被严格触发**：判定当时 4/4 个 Δ_org 点估计
  为负但 CI 跨零。仍执行，作为门槛后佐证（额外 ≈21 min GPU，远低于预算），
  **未依据其结果做任何臂选择**。
- 轨迹数 7（F、N/O s0、N/O s1、N/O full10k）> 上限 6（seed-1 对为超支）；
  GPU 合计 ≈1.1 h / 5 h。

## 4. 机制诊断（冻结 O 模型，仅 monitor 行）

| 干预 | mean |Δŷ| | MAE |
|---|---:|---:|
| 无（真实归属） | — | 0.09899 |
| 分子内打乱 w（junction 多重集完全不变，归属被破坏） | 0.0784 | 0.13445（+36%，seed1 0.13334） |
| 全 null 赋值 | 0.1408 | 0.18512 |
| 跨分子打乱 w | 0.3430 | 0.35448 |
| 屏蔽原子类型 / radius2 / 环 / 键型 one-hot / 全局 | 0.605 / 0.931 / 0.850 / 0.225 / 0.558 | 0.604 / 0.924 / 0.845 / 0.239 / 0.578 |

反向：null 模型在推理时换成真实归属仅移动 0.0393 —— 冻结的 O 模型对归属的
依赖约为 null 模型的 2 倍，但两臂泛化 MAE 相同：null 臂**学会了补偿**，而不是
“表达不了”。junction 编码器梯度 1.06（一次 fit-batch 反传），非死支。

## 5. 失败与偏差（如实记录）

1. `eval-locked` 首次运行因 subgroup 索引笔误在**写出任何产物之前**崩溃；标签在
   内存中被加载但未计算/查看任何指标，修复后一次性读取，结果照实报告，无二次查看。
2. 训练期发现并修复两个**实现 bug**（对象按行索引 vs 训练按位置索引的错配；
   permute 反事实 `[n,1]` 广播导致 dpred 无意义），均在正式结果写入前修复，
   修复后的 smoke 与本地复现逐位一致。
3. full-10k O 运行在完成全部 240 epoch 后因“无 monitor 的日志格式”以非零码退出；
   其 checkpoint/JSON 已保存，并被冻结哈希后用于 official valid 评估。
4. official valid 在本 repo 历史轮次已暴露（0.118618），因此比较是**受控披露比较**，
   不是盲测；official test 全程未读。
5. 单 seed/臂、单次 locked 读取、GPU 非确定性 → 探索性结论，非训练稳定性证据。

## 6. 判断

**PROMISING_REPRESENTATION_ONLY。**

- 表示侧：严格 one-shot 对象组合器（413k 参数，无 GNN/attention/迭代）在全部
  6 个匹配对比中优于匹配 Full-Y（3 个 CI 不含 0），并在 official valid 上比历史
  Full-Y 低 0.010–0.015 MAE 且 RMSE 从 0.599 降到 0.216–0.256 —— 这部分可复现、
  值得继续。
- 归属侧：机制上可用（分子内打乱归属使 MAE +36%），但泛化收益不成立
  （4 个读出中 3 负 1 正，全部 CI 跨零）。H-ORG 未证实；H-REP（局部表示才是
  瓶颈）获得更好支持。合成实验仍是唯一“归属决定性”的场景（且证明了 null 臂
  不是容量受限）。
- 下一阶段建议：若继续追问归属，应换到“归属是预测上必要”的任务（逐键/逐原子
  性质、可原子映射的目标、多任务监督），而不是继续加 ZINC MAE 的统计功效；
  若继续 0.09 目标，则应做 2 seed × 新 split 的等高功效复现，并补一个**匹配的**
  full-10k Full-Y 臂，才能声称“BJOC 在 official valid 上优于 Full-Y”。

## 7. 前置文献审计（约束相关性）

**GPM**（ICML'25, arXiv 2501.18739，官方仓库 `Zehong-Wang/GPM`）经一手核实：
其流程为 random-walk tokenizer → 序列编码 → **transformer encoder**（README
第 3 步明写），因此**不能**移植进“无 attention/Transformer”的本轮约束；它与
本候选的关键差别是：GPM 无显式归属对象，归属只隐式存在于 token 顺序与注意力中。
其余对照（JT-VAE、MEGAN/GSN、Graphormer、Deep Sets/Set Transformer、
Morgan/ECFP+MLP/RF）按标准事实列于
`notes/structural_relation_composer_v1.md` §10，均不提供“关系对象 + 静态归属
分组 + 无消息传递”的组合。

## 8. 证据边界

- locked dev 是 official train 内部的隔离划分，不是新的盲测集；
- 合成真值只说明可表示性，不说明化学因果；
- reader 是普通 MLP，中间向量**不是**可加贡献，本轮不做 α/δ/γ 归因声明；
- null 臂不是小容量对照：它保留全部原子/键/junction/参数/图级矩，且达到同等 MAE。
