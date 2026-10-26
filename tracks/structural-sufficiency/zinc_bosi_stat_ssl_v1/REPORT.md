# BOSI Statistics-Only SSL v1 — 只预训练统计三模块的初始化

轮次 `zinc_bosi_stat_ssl_v1`。前置：

* `zinc_bosi_architecture_v0`（`REPORT.md`；关系分配 `NO_GO`；BOSI-B Locked s0 0.082449 /
  s1 0.081491；Monitor s0 0.124505 / s1 0.122400；473,889 参数）；
* `zinc_bosi_partner_content_v1`（`REPORT.md`；`PARTNER_CONTENT_SUPPORTED`，去掉邻键 learned
  内容使 Locked 变差 +0.012390，两 seed 同向、CI 不跨零、非尾部；其 `NEXT_SSL_HYPOTHESIS.md`
  只写了设计，**未启动任何 SSL 作业**）；
* `bjoc_masked_pretraining_v1`（`SELF_SUPERVISED_PRETRAINING_USEFUL`，encoder 初始化
  Locked-dev −0.011984）；`bjoc_task_adaptive_ssl_v1`（`TASSL_NO_GAIN`，训练期持续 SSL 无效）。

代码 commit：`d17f8e3`（协议 + 实现 + 16/16 CPU 门控）、`55ba40d`（pilot 之后的分析符号修正，
不涉及训练/前向）。GPU 作业 commit 见 §E 账本。
**官方 valid / test 未构造、未加载、未预测、未评分。**
GPU 预算：**0.649 A100·h**（上限 4.0）。

---

## A. 核心判断

**`NO_GAIN`。统计三模块的 masked 结构预训练既没有改善泛化，也没有保持中性：它让
Monitor 变差 +0.008184（+6.6% 相对），在预注册的 seed-0 门槛处就停住了；唯一被读取的
Locked 预测同向变差 +0.005307（CI95 [+0.00100, +0.00945]，**不含零**），并且退化集中在
`any_unseen`（+0.00979）与 `has_T_or_Q`（+0.01241）这两个假设最想帮到的子群。**

| 量 | 历史 BOSI-B s0 | BOSI-SSL s0 | 差 |
|---|---|---|---|
| Monitor（best，严格最小）| **0.124505** @ep549 | 0.132689 @ep408 | **+0.008184** |
| Locked best_monitor | **0.082449** | 0.087756 | **+0.005307** |
| Locked final | 0.081969 | 0.084910 | +0.002941 |
| Monitor @ step 0（零监督步）| 1.519651 | 4.206981 | +2.687330 |
| Fit8k MAE @ best_monitor | 0.007698 | 0.018625 | +0.010927 |
| Locked RMSE | 0.283672 | 0.274699 | −0.008973 |

判决表（预注册 §7）：

| 条件 | 结果 |
|---|---|
| 两 seed `Δ<0` | 只有 seed 0，且 `Δ>0` → 不成立 |
| seed-mean `Δ≤−0.005` | 不成立 |
| ordinary / 去 top-10 同向为负 | 均为正（+0.005491 / +0.005875）|
| any-unseen 不恶化 | 恶化 +0.009791 |
| seed-0 Monitor 门槛（≤ B−0.004）| **不成立（+0.008184）** |

因此：**不买 seed 1 的监督训练，不买算力对照 C**（协议 §5）。

三句话：

1. **pretext 本身是成功的，而且真的用上了这条通道。** pilot（1,890 updates / 75.7 s）
   的 atom CE 0.2227 对边际 1.1096（**−0.886 nats**），bond 0.0191 对 0.5859（**−0.589 nats**）；
   把伙伴统计打乱后 CE 分别上升 **+0.652 / +0.586 nats**；非多数类准确率 79.7% / 98.3%
   （不是类别先验）。所以"没学到东西"不能解释后面的失败。
2. **但这个初始化对性质任务是有害的位移。** 零监督步时 Monitor 从 1.5197 变成 4.2070；
   整个 753 epoch 里 SSL 臂的 Monitor 处处高于历史臂（ep200 0.14967 vs 0.13557，
   ep400 0.13656 vs 0.12878，tail-100 中位 0.13324 vs 0.12527），从未追平。
3. **它并没有改变"靠什么预测"，只是把这个通道用得更差。** 在 Locked 上把 384 维伙伴矩块
   置零或把 `partner_half` 打乱，SSL 模型从 0.0878 崩到 **0.6301 / 0.6259**（100% 分子移动，
   平均 |Δ| 0.61）—— 与 BOSI-B 一样，全部预测都经由统计通道；预训练提升了"重建结构"的能力，
   却没有提升（而是降低了）"用统计量预测性质"的能力。

---

## B. 设计与实现（只有一个新东西）

**BOSI-SSL = BOSI-B + (`half_encoder`/`side`/`compose`) 的 masked 结构重建初始化。**
推理架构、参数（473,889）、六个模块、reader、全部 recipe 逐字不变；不新增标签、不新增分子、
训练期不用辅助 loss（`TASSL_NO_GAIN` 已否定该形态）。

* **数据**：只用 Fit8k 8,000 个分子的原始结构。预训练进程拿到的是 `structural_cache()`
  ——一个只含 `atom/bond/global` 与指针的字典；`cache["labels"]` 在该进程内不可达，
  G1 用守卫字典 + 标签投毒（loss 与参数逐位不变）证明。
* **遮蔽**：15% 原子 / 15% 物理键，每类至少 1 个，`SeedSequence([seed, mol, visit])`；
  **只改属性码，不删边**，所以 T/Q/R/N、closure、block、半径统计都是 v0 冻结对象的原样。
* **完整重建**：所有依赖类型的格子（`atom_feat` 0:28/42:46/46:74/74:102、`bond_feat` 0:4、
  `global` 19:23/31/32）由遮蔽后的原始图重算，其余纯拓扑列逐位复制。
* **两个丢弃式头**：原子恢复（输入 = 入射物理键 `c_e` 之和，96→160→28）、
  键恢复（输入 = 该物理键 `c_e`，96→160→4）。**禁止**图级 `global33`/reader 捷径。
* **冻结**：atom/bond encoder 载入历史同 seed 的 train-only masked-SSL checkpoint 并冻结；
  reader 不参与 pretext；可训练参数 247,872（三模块 211,680 + 头 36,192）。
* **配方**：Adam 1e-3 / wd 1e-5 / clip 5 / batch 128 / fp32 / 7,560 updates / 常数 LR /
  **last-step state only**（重建精度从不用于选 checkpoint）。

## C. CPU 泄漏审计：16/16 → `CPU_GO`（`analysis/cpu_checks.json`）

| 门 | 关键实测 |
|---|---|
| G1 数据边界/标签隔离 | Fit/Monitor/Locked 8,000/1,000/1,000，两两重叠 0；标签投毒后 loss 与全部参数**逐位相同**；预训练模块里无 `["labels"]` 引用 |
| G2 重建 == 生产特征 | **全部 8,000 个分子**三个张量 max\|Δ\| = **0.0**；与上游 `build_mol_objects` 在 100 个原始图上 0.0 |
| G3 遮蔽重建 == 完整重建 | 900 个 (分子, visit) 例：三个张量 0.0；`masked_batch` 对参考 collate 的 **17 个张量全部 0.0** |
| G4 关系不变性 | 900 例 × 9 个关系字段差异 **0**；与冻结关系缓存（`c698a2f3…`）一致 |
| G5 词表 | mask token 原子 **27** / 键 **0**，均不在 Fit8k 原始码内（原始 0–20 / 1–3） |
| G6 隐藏真值交换 | **240 次**：240 次输入逐位不变、240 次目标改变；batch 级 12 个张量 Δ=0.0 |
| G7 未遮蔽正控制 | 改 1 个可见邻居原子 → 8 个特征格变化、`Δc`=0.129、atom logit 移动 0.0148 |
| G8 半径统计无泄漏 | 274 个被遮原子自身行只含 token；自身行对 10 组隐藏真值**位同**；纯拓扑列对任意类型重排位同；邻居/distance-2 看到 token（588/588、698/698）|
| G9 目标可定位 | 166/166 atom、180/180 bond 一一对应；180/180 被遮物理键两条半键同步；端点直方图已重算；batch 内无 target 键 |
| G10 梯度 | 两个头分别反传到 `half_encoder`/`side`/`compose` 均 >0；atom/bond encoder 与 reader **恒为 0** |
| G11 推理一致 | 473,889 参数、28 个 state key、六模块、无 head；初始 hash = 历史 `ceeb598f…` |
| G12 对称/批次 | 重编号 0.0、键反转 3.6e-11、键序 1.4e-10、batch 重组 4.4e-7（相对，图级输出量级 414）|
| G13 mask 流 | 同 visit 可复现、异 visit 不同、每类≥1、与 shuffle 流独立 |
| G14 pretext 前向 == 模型前向 | `a,b,x,M,n,s,c` 逐位相同；无 reader |
| G15 无图级捷径 | `global` 投毒 / 删除 `global` / `del model.reader` 后 logits **逐位不变** |
| G16 接口/预算 | 六模块参数、247,872 可训练、SIDE_IN 617、READER 417、arm=B |

## D. 预训练：pilot（先决）与完整作业

**Pilot（seed 0，1,890 updates，75.7 s = 0.0210 A100·h）** —— 四项冻结条件全部通过：

| 条件 | 阈值 | 实测 |
|---|---|---|
| atom vs 边际 | ≥0.10 nats | **0.886**（CE 0.2227 vs 1.1096）|
| bond vs 边际 | ≥0.02 nats | **0.589**（CE 0.0191 vs 0.5859）|
| atom 用伙伴统计 | CE(shuffled)−CE(clean) ≥0.02 | **+0.652** |
| bond 用伙伴统计 | ≥0.02 | **+0.586** |

非多数类：atom 1,079 个原子准确率 **0.7970**（多数类 0.9758）、bond 1,003 个键 **0.9831**；
`M` 置零使 atom CE 0.2227→0.6243。梯度（step 1889）从 atom 头到 half/side/compose
= 0.259 / 0.255 / 0.235，从 bond 头 = 0.052 / 0.054 / 0.044。
吞吐：75.7 s / 1,890 updates（数据装配 56.6 s 占 75%）；**projected full = 4×≈303 s**（实测 306 s，吻合）。

**完整预训练（两个 seed，`S_pre0_full`/`S_pre1_full`，各 7,560 updates / 120 epochs）**：

| seed | wall | GPU·h | 末段 CE atom / bond | probe CE atom / bond | 非多数类 acc (atom) | shuffled 惩罚 (atom / bond) |
|---|---:|---:|---|---|---|---|
| 0 | 306.0 s | 0.0850 | 0.1929 / 0.0156 | 0.1889 / 0.0157 | 0.8285 | +0.669 / +0.730 |
| 1 | 330.7 s | 0.0918 | 0.1886 / 0.0155 | 0.1887 / 0.0162 | 0.8313 | +0.633 / +0.578 |

（120 epoch 相对 pilot 的 25% 预算只把 atom CE 从 0.2227 降到 0.1889 —— pretext 早已饱和。）

## E. GPU 账本（`analysis/freeze.json` → `gpu_ledger`）

| 作业 | commit | wall | A100·h |
|---|---|---:|---:|
| `ssl-pilot-s0` | `d17f8e3` | 75.7 s | 0.0210 |
| `ssl-pre-s0`（全量预训练）| `55ba40d` | 306.0 s | 0.0850 |
| `ssl-pre-s1`（全量预训练，与上并行）| `55ba40d` | 330.7 s | 0.0918 |
| `ssl-S-s0`（BOSI-SSL 监督）| `55ba40d` | 1,624.9 s | 0.4514 |
| **合计** | | | **0.6492 / 4.0** |

执行 regime 与历史 BOSI v0/v1 相同：`res` / `a100-2` / A100-SXM4-40GB / torch 2.5.1+cu124 /
驱动 550.163.01 / 单卡。**未购买**：seed 1 的监督训练、算力对照 C。
C 的事前冻结预算已按机械规则算出并留档（`analysis/control_budget.json`）：
N₀ = round(306.0 s ÷ 34.3 ms/update) = **8,925 额外更新 ≈ 0.0850 GPU·h**（从 B_s0 的
`best_monitor` 继续、batch 流与终值 LR 1e-5 不变）。因为 seed-0 门槛未过，C 未启动。

## F. 监督实验：seed-0 门槛与机制诊断

`S_s0`：47,400 updates / 753 epoch / 1,624.9 s；初值 = 历史 BOSI-B s0 全模型 + 三个模块的
SSL 导出（`modules_changed = [compose, half_encoder, side]`，其余四个模块与 reader 逐位相同）；
首个 batch 行号、初始 hash、参数量与历史 B_s0 全等（`pairing_checks.seed0` 全 true）。

**门槛：`Monitor(S_s0) = 0.132689 > 0.120505 = Monitor(B_s0) − 0.004` → 失败，停止。**

`analysis/diagnostics.json`（描述性，非判决依据，只用 Fit8k/Monitor）：

| 量 | 历史 B_s0 | BOSI-SSL s0 |
|---|---|---|
| Monitor @ step 0 | 1.519651 | **4.206981** |
| Monitor @ ep200 | 0.135566 | 0.149666 |
| Monitor @ ep400 | 0.128780 | 0.136564 |
| best Monitor | 0.124505 @549 | 0.132689 @408 |
| tail-100 中位 Monitor | 0.125266 | 0.133243 |
| Fit8k MAE @ best_monitor | **0.007698** | 0.018625 |

读法：SSL 初始化把模型推到一个**对性质任务更差**的函数区域（t=0 时 Monitor 差 2.69），
监督训练把它拉回来一部分，但 753 epoch 全程低于历史轨迹，最终也没追平；它选出的
best_monitor 同时也是**拟合更差**的那个点（Fit8k 0.0186 vs 0.0077），所以这不是
"更强的拟合导致过拟合"，而是**没学到对性质有用的表示**。
训练还更快（1,625 s vs 1,788 s），与"统计三模块的输入分布被重排后 GPU 利用率变化"一致，
不构成任何优势证据。

## G. Locked：一次读取（`analysis/locked_read.json`）

顺序不可逆：`s_freeze.py` 先冻结 checkpoint / Monitor / Fit 读数与哈希（含复用文件的逐文件
sha256 校验与 GPU 账本），之后 `s_locked.py` 一次性读取 Locked（n=1,000），
再 `s_analyze.py` 只用冻结文件出表。BOSI-B 的 Locked 预测**直接复用 v0 冻结 npz**，未重跑。

| run | ckpt | Locked MAE | RMSE |
|---|---|---:|---:|
| B_s0（复用）| best_monitor | **0.082449** | 0.283672 |
| B_s0（复用）| final | 0.081969 | 0.283819 |
| S_s0 | best_monitor | 0.087756 | 0.274699 |
| S_s0 | final | 0.084910 | 0.275828 |
| B_s1（复用，仅背景）| best_monitor | 0.081491 | 0.275951 |

配对对比 `S_s0 − B_s0`（主判据 = best_monitor）：

| 指标 | 值 |
|---|---|
| ΔMAE | **+0.005307** |
| 配对 bootstrap 10,000 CI95 | **[+0.000996, +0.009452]**（不含零）|
| 更好的分子比例 | 0.469 |
| 去 top-10 后 | +0.005875（n=990）|
| ordinary（932）| +0.005491 |
| tail（68）| +0.002782 |
| low_support（398）| +0.003285 |
| any_unseen（287）| **+0.009791** |
| all_supported（315）| +0.003776 |
| has_T_or_Q（75）| **+0.012409** |
| alloc_defined_top_quartile（252）| +0.007445 |
| rings_ge4（212）| +0.002434 |

**9 个分组的 Δ 全部为正**，且最大的两处正是"未见结构"与"含 T/Q 关系"的子群。
RMSE 略降（0.2837 → 0.2747）但 MAE 上升：少数大误差被压小、多数分子变差——对一个以
MAE 为目标的模型来说，这不是可用的改进。

同一次读取内的冻结权重干预（描述性，不进入判决）：

| 干预（S_s0 best_monitor）| Locked MAE | 平均 \|Δ\| | 移动比例 |
|---|---:|---:|---:|
| 干净 | 0.087756 | — | — |
| 384 维伙伴矩块置零 | 0.630122 | 0.6134 | 100% |
| `partner_half` 批内置乱 | 0.625904 | 0.6087 | 100% |

即：**SSL 初始化没有削弱模型对统计通道的依赖**（和 BOSI-B 一样是 100%），只是把这条通道
学得对性质任务更没用。

## H. 为什么下这个结论（排除清单）

| 可能解释 | 本轮证据 | 结论 |
|---|---|---|
| pretext 不可识别 / 头部只学类别先验 | pilot 与全量都远优于边际（atom CE 比边际低 0.886 / 0.920 nats），非多数类 acc 0.80–0.83，非"猜先验" | **排除** |
| 泄漏或计算路径错误（标签、图级捷径、半径统计）| 16 门审计：8,000 分子重建 0.0、240 次真值交换输入位同、274 个被遮原子自身行位同、`global`/reader 删除后 logits 位同 | **排除** |
| 统计通道没被用到 | 打乱伙伴统计使 pretext CE +0.58~0.73 nats；Locked 干预使 MAE 崩 7 倍 | **排除** |
| pretext 成功但没迁移到性质任务 | Monitor +0.008184、Locked +0.005307（CI 不含零）、所有分组同向为正 | **本轮结论** |
| 只是"更多计算"造成的差异 | SSL 臂总计算 = 0.451 h（监督）+ 0.085 h（预训练）；历史臂 0.497 h。**SSL 臂算得更多却更差**，方向与"计算红利"相反，无需 C 即可排除 | **排除** |

需要明确写下的负面结论：**"只预训练统计三模块的初始化"这一形态，在本任务上没有价值，
而且是负价值。** 这与 `PARTNER_CONTENT_SUPPORTED` 并不矛盾——那条结论说的是
"这条通道的**内容**有量级价值"，不是"这条通道的**参数**值得用结构重建去预训练"。
把二者混为一谈正是本轮要检验的假设，答案是否定的。

## I. 偏差、限制与如实声明

1. **协议修订（GPU 之前）**：`control_budget` 的 C 设计在第一次 GPU 作业之前改写为
   "历史 BOSI-B 的 `best_monitor` + 冻结额外预算"（原稿写的是从初始状态重训 47,400 步）。
   修订发生在 `d17f8e3` 提交、任何 GPU 作业之前；`freeze.json` 里记录的 `protocol_sha256`
   就是修订后的文件，可核对。
2. **`s_analyze.py` 符号修正**（`55ba40d`）：pilot 伙伴统计门的符号写反（记为
   `clean − intervened`）。修正只影响分析脚本，发生在**唯一一次 Locked 读取之前**，
   且修正后 pilot 仍是明确通过（+0.652/+0.586 nats，远超 0.02 阈值），不改变任何判决。
3. **seed 1 的预训练被提前执行**（0.0918 h，与 seed 0 并行）：协议只要求"seed-0 监督
   通过门槛才买 seed 1"。该作业未参与任何决策（seed 1 的监督训练与 Locked 均未发生），
   如实计入账本。
4. **单 seed 描述性 Locked 结果**：判决词表要求两 seed 才能给 `SSL_PROMISING`；本轮在
   seed-0 门槛处停止，因此 Locked 只作为"该失败是否也在 Locked 上成立"的一次描述性读取，
   **不得**被引用为两 seed 结论。
5. **Locked 是历史已暴露的内部划分**（DCC-L 4 seeds、ESOC、additive-reader、BOSI v0/v1 都读过），
   本轮读取不是新的盲测确认。
6. **没有做的事**：没有为救结果去改 mask 比例 / λ / 头结构 / 更新数 / seed 数；
   没有把重建精度当 checkpoint 选择依据；没有把 `any_unseen` 恶化解释成"样本太小"；
   没有启动 attention / learned junction / 关系分配 / 架构搜索；没有读官方 valid/test。

## J. 证据索引

| 文件 | sha256（前 16）| 内容 |
|---|---|---|
| `PROTOCOL.md` | `cd4baa9f8308a60f` | 预注册协议（含 C 修订）|
| `configs/bosi_stat_ssl_v1.json` | `e0dae9b3ad3fb8f2` | 机器可读配置与禁止清单 |
| `analysis/cpu_checks.json` | `ebbbfa1038b8e4ec` | 16 门审计，`CPU_GO` |
| `analysis/freeze.json` | `103db1b494262682` | checkpoint/Monitor/Fit 冻结 + 复用校验 + GPU 账本 |
| `analysis/locked_read.json` | `94654d1a77d3de43` | 唯一一次 Locked 读取与干预 |
| `analysis/summary.json` | `78db46ffffe3cdf5` | 全部表与 bootstrap |
| `analysis/decision.json` | `3b7c89a9dc8418e0` | `NO_GAIN` 与逐条条件 |
| `analysis/control_budget.json` | `b2208c2fab0d52d8` | C 的事前冻结预算（未购买）|
| `analysis/diagnostics.json` | `431433589a507c48` | 初始化点/轨迹/Fit 诊断（描述性）|
| `runs/S_s0/best_monitor.pt` | `c0ed8975469a…` | SSL 臂选中的 checkpoint |
| `runs/S_s0/final.pt` | `4236e766775a…` | SSL 臂终点 |
| `analysis/locked_preds/S_s0_best_monitor.npz` | `fca026f7e045…` | 冻结 Locked 预测 |

## K. 本轮之后

* 不再启动"用结构重建初始化 BOSI 统计模块"的任何变体（换 mask 比例、换头、换目标、
  换 λ 都只是同一假设的重采样）。
* 若还要追 SSL，唯一没被否定的形态是**对编码器/表示层的初始化**（`bjoc_masked_pretraining_v1`
  已给出正结果），而不是对**统计聚合三模块**的初始化；本轮显式说明这两者不能互相外推。
* 允许的下一步只有：把 `PARTNER_CONTENT_SUPPORTED` 的内容通道做成**推理期可用**的形式
  （不新增标签、不改架构前提下的接口工作），或接受 BOSI-B 现状并转向别的假设。
  任何新方向都必须重新预注册 CPU 门控与 GPU 预算。
