# cscl-bjoc-performance-mechanism-long-v1 — 预注册与执行记录

- Protocol: `cscl-bjoc-performance-mechanism-long-v1`
- Config: `configs/bjoc_performance_mechanism_long_v1.yaml`
- Driver: `code/run_bjoc_performance_mechanism_long_v1.py`
- Training module: `code/bjoc_perfmech_long_v1.py`
- Tests: `tests/test_bjoc_performance_mechanism_long_v1.py`
- Results: `results/bjoc_performance_mechanism_long_v1/`
- 冻结时间：代码/配置在 GPU 训练前提交；`protocol_freeze.json` 在部署后的远端生成。

## 0. 本轮问题与纪律

长期目标不变：无 GNN/Transformer/迭代消息传递的有竞争力分子性质预测，
并逐步建立可信的结构贡献理解。本轮**性能优先**，机制实验只服务于性能与未来论文。

五个问题（Q1–Q5）见 config。核心对照缺口：`Scratch + 360ep cosine`（B3）
从未在 full-10k 上训练过；因此无法判断 R3 recipe 是否只是弥补了 scratch
在旧 recipe 下的欠优化，还是 SSL 在充分退火后仍有增量。

纪律：

- official test 任何形式都不读取；本 driver 的代码路径中没有任何 test 数据入口。
- official valid 是历史已暴露开发集：Stage A 只在 B3 预测冻结后描述性评分，不用于
  本轮新架构选择。
- Stage B/C/D 使用新的 group-aware 内部划分；Fit 仅用于训练，Monitor 只在预测冻结
  并哈希后统一读取一次，Locked 只在最终候选冻结后读取一次。
- 所有 GPU 任务只走 `rr`，仅使用 `res` GPU1（冻结时 GPU0 被外部用户 `zc` 的
  `demo_train.py` 占用 37 GB / 100%，本机无本地 GPU）。

## 1. 资源与冻结时的远端状态（2026-10-13）

- `rr doctor res`：ssh/checkout/uv/env 全 OK；两卡 A100-SXM4-40GB，driver 550.163.01，
  torch 2.5.1+cu124，python 3.12.14。
- GPU0：外部进程 PID 9757（user `zc`）占用 37,255 MiB、100% util，已运行 3 天 20 小时，
  **不可用**；GPU1 空闲（14 MiB, 0%）。
- 远端 checkout `399acdf10754`（main）。已核对远端存在并逐字节哈希的复用件：
  full-10k objects/prep/pretrain_encoder/soup_P、recipe-round soup_R3、split_a、
  canonical smiles、masked-round objects/prep。
- 远端**缺失** `bjoc_full10k_pretraining_transfer_v1/valid_y_read_once.npy`（git-ignored，
  历史上只在本地生成），因此 Stage A 的官方 valid 评分在本地用已有冻结 y 完成；
  该 y 只用于已经暴露的开发集描述性评分。

## 2. Stage A — full-10k 的缺失单元格 B3

- 训练 arm：`B3F`（scratch init，R3 recipe：360 epochs，per-step cosine
  1e-3→1e-5，wd 1e-5，batch 128，clip 5，raw-y L1，last-5 soup，seed 0/1）。
- `stage-a-audit` 在训练前逐项核对：B3 的 scratch full-model /
  composer-reader / encoder 初始化哈希必须等于 full-10k 轮记录的 P 初始化；
  B3 绝不加载 SSL encoder；360-epoch batch order 哈希等于 recipe 轮记录的 R3；
  recipe 每项常数相等；cosine 公式端点相等。审计不通过则拒绝训练。
- B3 预测在训练 job 内 label-blind 写出；本地评分使用历史 B0/P0/P3 预测。
- 判据：`G_SSL_R0 = MAE(P0)-MAE(B0)`，`G_SSL_R3 = MAE(P3)-MAE(B3)`，
  `interaction = G_SSL_R3 - G_SSL_R0`；分类词表见 config（persists / shrinks / none）。

## 3. Stage B — 新内部划分 2×2

- 新 split salt `cscl-bjoc-perfmech-long-v1-B`，canonical-SMILES grouped，
  target 8000/1000/1000；实际规模、group 约束与 hash 记录在
  `split_B.json`。
- **与历史 split A 的重叠（必须披露）**：本地与远端构建结果一致：

  |  | 与 A.fit | 与 A.monitor | 与 A.locked |
  |---|---|---|---|
  | B.fit (8000) | 6432 | 790 | 778 |
  | B.monitor (1000) | 782 | 95 | 123 |
  | B.locked (1000) | 786 | 115 | 99 |

  即 Monitor/Locked 中各约 78–79% 的行在历史 full-10k 训练中作为 fit 行出现过。
  因此本 split 的价值不是“全新数据”，而是：同一 Fit-B 上训练的匹配臂对比，
  且 Monitor/Locked 的**评分标签**在本轮前从未被读取。报告不得把该 split 描述为
  “从未见过的数据”或“新分布测试”。
- Fit-only 标准化；Fit-only masked SSL（7560 updates，与历史相同预算，objective
  与 mask 规则逐项相同）；P0/P3 共享同 seed 的 SSL encoder。
- 四臂同 seed、同 batch order（前 240 epoch 与 R0 一致）、同 soup 规则。
- Monitor 先冻结 8 个预测文件，再统一读标签、评分、写 gate 决策。
- Locked 标签封存至最终候选冻结之后。

## 4. Stage C — bond-composition second-moment readout（C1）

- 动机：reader 当前只收到 `z = Σ c_b`（composer 输出的一阶和）；未收到
  `c_sq = Σ c_b²`。`a` 有 `Σa` 与 `Σa²`，`c` 只有一阶，这是 reader 输入中
  一个不对称的缺口。
- 模型：`BJOCSecondMoment`，reader 输入 `[z, c_sq, a_sum, a_sq, g]`；新第一层
  权重列零初始化，因此初始函数与 P3 逐位相同（实测 max|Δ|=1.04e-7）。
  新增参数 24,576（413,505 → 438,081，+5.9%），容量变化如实披露。
- CPU gate（GPU 前必须通过，已本地运行一次并在远端复跑）：
  - 恒等初始化：PASS（1.04e-7）；
  - 代数构造：同一阶和/计数、不同二阶和（存在性）：PASS；
  - 经验非可恢复性：用 P3 模型在 4,000 个真实结构上的 `c_sq` 对现有
    reader 输入做线性 probe，per-dim test R² median 0.619（训练 0.786），
    中心化相对残差 1.03；说明线性可恢复部分有限。注意：非线性 reader
    在理论上仍可逼近部分统计量；该 gate 只排除“纯代数重写”；
  - 非退化：96 维 std 全部 > 0，finite；
  - 梯度活性：新列梯度 100% 非零（max 4.2e-3），一步 Adam 后预测改变 1.36；
  - 不变性：原子重标号 / 键端点交换 / batch 成员关系：PASS；
  - 无 GNN/attention/迭代模块；参数量审计 438,081。
- GPU 判据：同 split、同 seed 的 P3 为 matched baseline；seed 0 Monitor
  改善 ≤ −0.003 且 ordinary 不恶化、top-10 去掉后仍改善、RMSE 不超差 +0.005
  才购买 seed 1；通过者进入 Locked 主表。

## 5. Stage D / E — 条件购买规则（在 Monitor 读取前冻结）

- Stage D：当且仅当 Stage B 的 `recipe_effect` gate（P3−P0 seed-mean ≤ −0.003
  且两 seed 均为负）通过时，购买第二划分 `...-C` 的 P0/P3 复现；否则记录
  `RECIPE_GAIN_NOT_ROBUST_ACROSS_SPLITS` 并跳过。scratch arms 只在 Stage A/B
  交互方向冲突且预算允许时补。
- Stage E：当且仅当 Stage B 的 SSL gain gate（P3−B3 ≤ −0.005 且两 seed 为负）
  通过、且 atom-only vs atom+bond 能改变预训练策略时，购买 seed 0 的
  atom-only masked SSL（相同结构、相同 mask 规则、相同 7560 updates、相同 R3
  downstream），默认单 seed。

## 6. 预算

- 上限 7.0 GPU-h，目标 6.5；只使用 GPU1，故墙钟≈GPU-hours。
- 指导预算：A 0.5 / B 2.4 / C 0.8 / D 1.5 / E 0.7 / reserve 0.6。
- 实际消耗记录在 `gpu_budget_ledger.json`。

## 7. 偏差与执行日志

（按时间顺序追加；不改写已冻结的判据。）

- 2026-10-13 预注册提交：config/module/driver/tests；本地 CPU 审计
  （splits B、objects B、input audit、Stage-A init/recipe 审计、C1 CPU gate）全部通过。
- Stage B 与 A 的重叠数据来自本地 `split_B.json`；远端重算 `objects:B` 时
  会再次校验 split 行哈希，若不同则拒绝。
- 远端 `prep`（commit `3bbd9ee`）：所有冻结输入在位；split B 与本地逐位一致；
  Stage-A init/recipe 审计双 seed OK。远端 C1 CPU gate 复现本地结论。
- 本地 smoke：SSL 5 updates、B3 1 epoch（CPU）通过，用于验证训练/序列化路径；
  smoke 事件不产生正式 artifact，不计入 GPU-h。

### Stage A 结果（official valid，已暴露开发集，仅描述性）

2×2（seed mean；B3F 为本轮新增方格）：

| arm | seed 0 | seed 1 | mean |
|---|---|---|---|
| B0 scratch + 240 const | 0.100172 | 0.098232 | 0.099202 |
| P0 SSL + 240 const | 0.084786 | 0.082313 | 0.083550 |
| B3 scratch + 360 cosine（本轮） | 0.097775 | 0.096714 | 0.097244 |
| P3 SSL + 360 cosine | 0.079101 | 0.080586 | 0.079844 |

对比：

- `G_SSL_R0 = P0−B0 = −0.01565`，CI95 [−0.02105, −0.00990]；
- `G_SSL_R3 = P3−B3F = −0.01740`，CI95 [−0.02479, −0.00997]；
- interaction `G_SSL_R3 − G_SSL_R0 = −0.00175`（per-seed s0 −0.00329 / s1 −0.00021）；
- recipe effect for scratch `B3F−B0 = −0.00196`，CI95 跨 0；
- recipe effect for SSL `P3−P0 = −0.00371`，CI95 [−0.00639, −0.00101]。

**决策：`SSL_GAIN_PERSISTS_UNDER_R3`。** 在 360ep cosine 下 SSL 增益没有消失，
幅度甚至略大于 240ep constant 条件；但交互项很小且 seed 异质（完全由 seed 0 驱动），
不能声称 R3 放大 SSL。同时 `ordinary_delta = −0.019` 而 `tail_delta = +0.010`
（65 个尾部分子），复现了已记录的“SSL 改善为普通分子、尾部代价”模式。
该结果回答了 Q1，并封闭了历史归因缺口：R3 不是以牺牲 SSL 增益为代价。
GPU 记录：B3F s0 0.1914 h、s1 0.1944 h。

### Stage B/C/E 结果（split B，Monitor 只读一次；CPU 评分于 `res` 登录节点）

四臂 Fit-B 训练（batch order 审计 `full_matches=True`；P0/P3 共享同 seed SSL
encoder，A1 用 atom-only encoder，C1 用 P3 encoder）：

Monitor（n=1000，paired，seed-mean of single models）：

| 对比 | Δ | CI95 | per-seed | ordinary | tail | top-10 removed |
|---|---:|---|---|---:|---:|---:|
| P3−P0（SSL recipe） | **−0.00101** | [−0.00411, +0.00191] | −0.00496 / **+0.00295** | −0.00240 | +0.01667 | −0.00040 |
| B3−B0（scratch recipe） | +0.00161 | [−0.00230, +0.00542] | −0.00061 / +0.00383 | +0.00249 | −0.00958 | +0.00259 |
| P3−B3（R3 SSL gain） | **−0.01659** | [−0.02207, −0.01104] | −0.01744 / −0.01574 | −0.01788 | −0.00023 | −0.01698 |
| P0−B0（R0 SSL gain） | **−0.01397** | [−0.01903, −0.00870] | −0.01309 / −0.01486 | −0.01299 | −0.02647 | −0.01416 |
| interaction | −0.00261 | — | −0.00434 / −0.00088 | | | |

Gates：`recipe_effect=FAIL`、`recipe_effect_scratch=FAIL`、`ssl_effect_r3=PASS`、
`ssl_effect_r0=PASS`；决策 **`RECIPE_GAIN_NOT_ROBUST_ACROSS_SPLITS`** → 按预注册
跳过 Stage D（省 ~1.4 GPU-h）。

Stage C（seed 0）：Monitor Δ(C1−P3) = −0.00194，ordinary/top-10/RMSE 条件全部
通过，但未达 −0.003 阈值 → `C1_GAIN_WEAK_OR_MIXED`，不买 seed 1，不入 Locked 主表。

Stage E（A1，seed 0）：Monitor Δ(A1−P3) = +0.00197 → `BOND_OBJECTIVE_NEUTRAL`
（Locked +0.00101 [−0.00402, +0.00623]）。

最终 Locked 主表（read once，n=1000）：

| arm | seed 0 | seed 1 | seed-mean | RMSE |
|---|---:|---:|---:|---:|
| B0 | 0.110946 | 0.115653 | 0.113300 | 0.298374 |
| B3 | 0.107855 | 0.112969 | 0.110412 | 0.263897 |
| P0 | 0.097056 | 0.090156 | 0.093606 | 0.249366 |
| P3 | 0.090440 | 0.091241 | 0.090841 | 0.235368 |
| A1 | 0.091454 | — | 0.091454 | 0.226296 |

| 对比 | Δ | CI95 | per-seed |
|---|---:|---|---|
| P3−P0 | −0.00277 | [−0.00647, +0.00039] | −0.00662 / +0.00108 |
| B3−B0 | −0.00289 | [−0.01096, +0.00298] | −0.00309 / −0.00268 |
| P3−B3 | −0.01957 | [−0.03098, −0.00964] | −0.01741 / −0.02173 |
| P0−B0 | −0.01969 | [−0.02683, −0.01375] | −0.01389 / −0.02550 |
| A1−P3 | +0.00101 (s0) | [−0.00402, +0.00623] | +0.00101 |

结论：SSL 增益在两种 recipe、两个 seed、Monitor 与 Locked 上全部成立（~0.02）；
R3 recipe 增益在两个 split 上都不能复现（seed 1 反向）。详见
`results/bjoc_performance_mechanism_long_v1/REPORT.md`。

### 预算终账

GPU 总消耗 **3.04 / 7.0 h**（A 0.386、B 1.895、C/E 0.753、predict 0.004；
Stage D 未购买）。Failed predict 尝试两次（共 ~20 s GPU，未计）。

### 预先记录的执行偏差（在看任何 Monitor 标签之前）

- **D1（Stage E 顺序问题）**：预注册购买规则要求“先看 Stage-B Monitor 的
  `ssl_effect_r3` 再决定是否购买 A1”，但协议同时要求“所有预测必须在首次读
  Monitor 标签前冻结”。两者矛盾时，我选择了预测冻结纪律：**无条件训练
  seed-0 的 atom-only SSL 与 A1 臂**，并在 Monitor 读取前把 A1 预测一并冻结。
  代价 ~0.62 GPU-h；好处是 Monitor 只读一次且所有 scoring 的预测都是盲的。
  A1 若不是按购买规则会被跳过，则报告时记为“未购臂”，不纳入主表。
  选此偏差而非读取后补臂，是因为后者无法证明预测未受标签影响。
- D1 的另一面：Stage C 的 C1 预测也必须在 Monitor 读取前冻结
  （`stage-c-predict` 脚本本身会拒绝在 lock 存在后写预测）。
