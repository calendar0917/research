# SSRL v0 报告

## 直接回答

**没有受益。** 在本轮冻结的配方下（从 LCRC-W3 1M checkpoint 继续训练 5 epoch、8,000 个真实四分子矩形、AdamW 3e-5→1e-5、fresh optimizer state），结构替换监督和环境交互监督都没有带来可测量的改善；固定终点反而比原始 W3 略差，且 PAIR/RECT 辅助损失没有超过 ABS。

| 维度 | 主指标 | 原始 W3（epoch 0） | 最佳训练 checkpoint | 固定终点（epoch 5） | 是否受益 |
|---|---|---:|---:|---:|---|
| A 完整图预测 | 官方 Valid MAE (eV) | 0.127809167 | **无；三臂 best 均为 epoch 0** | ABS 0.130273009 / PAIR 0.130037595 / RECT 0.130612031 | **否** |
| B 实替换 delta | observation-equal MAE (eV) | 0.111144210 | 无；best 均为 epoch 0 | ABS 0.111786680 / PAIR 0.111982226 / RECT 0.112162349 | **否** |
| C 环境交互 D | pair-type macro MAE (eV) | 0.180303864 | 无；best 均为 epoch 0 | ABS 0.180550497 / PAIR 0.181283496 / RECT 0.181269228 | **否** |

三句话结论：

1. **完整图预测没有受益。** 三个臂按完整官方 Valid MAE 选择的最佳 checkpoint 全部是 epoch 0（原始 W3）；5 epoch 终点的 Valid MAE 分别比 epoch 0 差 +0.002463842 / +0.002228427 / +0.002802864 eV。训练子集 MAE 从 0.089772638 降到约 0.0871–0.0879 eV，说明模型确实在拟合本轮 64k 训练行，但泛化变差。
2. **实替换 delta 没有受益。** best 层面因为三臂都选 epoch 0，ABS/PAIR/RECT 与 epoch 0 的对照约为 0（重跑 float32 噪声 <1e-6 eV）；固定终点全部比 epoch 0 更差，且 RECT 比 ABS 更差（observation-equal +0.000375669 eV，pair-type macro +0.000754525 eV，95% CI 均不含 0）。
3. **环境交互 D 没有受益。** 固定终点 RECT 比 epoch 0 差 +0.001653509 eV（observation-equal）/ +0.000965363 eV（pair-type macro）；RECT 比 ABS 差 +0.000677521 / +0.000718731 eV；RECT 与 PAIR 在 observation-equal 上差 +0.000125677 eV，在 pair-type macro 上为 −0.000014268 eV（CI 跨 0），没有可用的相对增益。

## 协议与预算

- 源 checkpoint：`tracks/structural-sufficiency/lcrc_pcqm4mv2_efficiency_width_v1/runs/PCQM_W3_1M/checkpoint_best_valid.pt`，sha256 `f4175f14…`，2,038,167 参数，epoch 39，官方 Valid MAE 0.127809167210 eV。
- 三个臂独立加载同一 checkpoint；AdamW 重新初始化（不是原训练轨迹的无缝恢复）；所有原始参数可更新，没有新增可训练参数。
- 训练计划：8,000 个 molecule-disjoint 真实矩形（32,000 个唯一端点分子，1,594 个 pair type；全部 4 分子互异）；32,000 个普通分子；batch = 16 矩形（64 图）+ 64 普通图 = 128 图；500 steps/epoch。
- 5 epoch 是在正式运行前由 preflight 的纯计时投影冻结的：单 epoch + 完整 Valid 实测 82.8 s，投影两波正式运行 511.3 s（≤600 s），正式运行实测 486.4 s。没有因为中间预测改变 horizon。
- 正式运行：2×A100-40GB，wave 1 并行 ABS/PAIR，wave 2 RECT；总墙钟 486.4 s，GPU·h（进程和）0.1850，低于 0.5 GPU·h 硬上限；本地 CPU 数据准备（plan + 冻结评估目录）也低于 5 分钟目标。
- 没有读取 Test 标签；没有新增 seed、epoch、loss 系数、数据集或模块。

## A. 完整图预测

| 臂 | best epoch | best Valid MAE | final epoch | final Valid MAE | final − epoch 0 |
|---|---:|---:|---:|---:|---:|
| ABS | 0 | 0.127809167 | 5 | 0.130273009 | +0.002463842 |
| PAIR | 0 | 0.127809167 | 5 | 0.130037595 | +0.002228427 |
| RECT | 0 | 0.127809167 | 5 | 0.130612031 | +0.002802864 |

best 值在三臂重跑间有 ≤1.8e-9 eV 的 float32 差异（同一 checkpoint 在不同 GPU 上重放），不影响选择：三臂的 best epoch 都是 0。

逐 epoch 的完整 Valid MAE：

| epoch | ABS | PAIR | RECT |
|---:|---:|---:|---:|
| 0 | 0.127809167 | 0.127809169 | 0.127809167 |
| 1 | 0.128007851 | 0.128788198 | 0.128971122 |
| 2 | 0.129873999 | 0.129606764 | 0.129354114 |
| 3 | 0.130859018 | 0.130772056 | 0.131239195 |
| 4 | 0.129619574 | 0.129770331 | 0.129690912 |
| 5 | 0.130273009 | 0.130037595 | 0.130612031 |

本轮训练子集 MAE（同一 64k 训练行）：

| epoch 0 | ABS final | PAIR final | RECT final |
|---:|---:|---:|---:|
| 0.089772638 | 0.087090377 | 0.087207592 | 0.087894320 |

因此完整图维度的结论是：训练确实降低了训练子集误差，但没有产生一个在完整官方 Valid 上优于原始 W3 的 checkpoint；最佳 checkpoint 仍是 epoch 0。

## B. 实替换 delta

best 层面：三臂 best epoch 均为 0，规定的五个对照（ABS−epoch0、PAIR−ABS、RECT−ABS、RECT−PAIR、RECT−epoch0）全部约为 0（<1e-6 eV），没有训练后的 best checkpoint 可比较。下面只报告固定终点。

固定终点（epoch 5）在全部 36,952 个可评估替换观测、1,677 个 pair type 上：

| 模型 | obs MAE (eV) | macro MAE (eV) | corr | slope | dir obs | dir macro |
|---|---:|---:|---:|---:|---:|---:|
| epoch 0 (W3) | 0.111144210 | 0.116106923 | 0.946135 | 0.912247 | 90.76% | 91.31% |
| ABS final | 0.111786680 | 0.116217530 | 0.945794 | 0.914915 | 90.74% | 91.30% |
| PAIR final | 0.111982226 | 0.116752333 | 0.945865 | 0.916443 | 90.71% | 91.29% |
| RECT final | 0.112162349 | 0.116972054 | 0.945834 | 0.915609 | 90.65% | 91.23% |

终点相对比较（正数 = MAE 变差；95% paired bootstrap，1,000 次）：

| 对照 | obs Δ [CI] | macro Δ [CI] |
|---|---:|---:|
| ABS − epoch 0 | +0.000642470 [+0.000500438, +0.000804265] | +0.000110607 [−0.000279140, +0.000540456] |
| PAIR − ABS | +0.000195546 [+0.000096256, +0.000297816] | +0.000534803 [+0.000302878, +0.000797122] |
| RECT − ABS | +0.000375669 [+0.000264119, +0.000504735] | +0.000754525 [+0.000470513, +0.001034605] |
| RECT − PAIR | +0.000180123 [+0.000090198, +0.000279029] | +0.000219721 [−0.000006030, +0.000451939] |
| RECT − epoch 0 | +0.001018139 [+0.000851158, +0.001214355] | +0.000865132 [+0.000491352, +0.001306252] |

子集（RECT final；仍是正数 = 变差）：

| 子集 | n obs / n pair types | RECT − epoch 0 obs / macro | RECT − ABS obs / macro |
|---|---:|---:|---:|
| 所有匹配观测 | 36,952 / 1,677 | +0.001018139 / +0.000865132 | +0.000375669 / +0.000754525 |
| 未见 exact context | 6,765 / 798 | +0.000564006 / +0.000640429 | +0.000209725 / +0.000772769 |

未见 exact context 的 macro CI 分别跨 0；observation CI 为正。没有子集显示 RECT 优于 epoch 0 或 ABS。

## C. 环境交互 D

best 层面同样没有训练后的 checkpoint：三臂 best epoch 均为 0，五个规定对照全部约为 0（<1e-6 eV）。下面只报告固定终点。

固定终点在全部 59,214 个矩形、1,542 个 pair type 上（pair-type macro 为主指标）：

| 模型 | obs MAE (eV) | macro MAE (eV) | corr | slope | dir obs | dir macro |
|---|---:|---:|---:|---:|---:|---:|
| epoch 0 (W3) | 0.179829687 | 0.180303864 | 0.926096 | 0.879482 | 89.21% | 88.60% |
| ABS final | 0.180805676 | 0.180550497 | 0.925541 | 0.882624 | 89.04% | 88.33% |
| PAIR final | 0.181357520 | 0.181283496 | 0.925439 | 0.884106 | 89.02% | 88.31% |
| RECT final | 0.181483196 | 0.181269228 | 0.925454 | 0.884722 | 88.99% | 88.32% |

终点相对比较（正数 = MAE 变差）：

| 对照 | obs Δ [CI] | macro Δ [CI] |
|---|---:|---:|
| ABS − epoch 0 | +0.000975988 [+0.000791802, +0.001167376] | +0.000246632 [−0.000327139, +0.000805576] |
| PAIR − ABS | +0.000551844 [+0.000435725, +0.000670987] | +0.000732999 [+0.000357424, +0.001099714] |
| RECT − ABS | +0.000677521 [+0.000541279, +0.000816558] | +0.000718731 [+0.000286897, +0.001138609] |
| RECT − PAIR | +0.000125677 [+0.000015790, +0.000237825] | −0.000014268 [−0.000328284, +0.000288607] |
| RECT − epoch 0 | +0.001653509 [+0.001442992, +0.001873835] | +0.000965363 [+0.000354874, +0.001550369] |

子集（RECT final；正数 = 变差）：

| 子集 | n rect / n pair types | RECT − epoch 0 obs / macro | RECT − ABS obs / macro |
|---|---:|---:|---:|
| 全部 | 59,214 / 1,542 | +0.001653509 / +0.000965363 | +0.000677521 / +0.000718731 |
| 双未见上下文 | 1,108 / — | +0.001353079 / +0.002082864 | +0.000017610 / +0.000039988 |
| SSTA molecule-disjoint | 11,930 / — | +0.001631836 / +0.001236823 | +0.000461016 / +0.000705288 |
| 946 个 same-edit 矩形 | 946 / 491 | +0.001311065 / +0.001086082 | +0.000544318 / +0.000351468 |
| 38 个严格矩形（仅描述） | 38 / 27 | −0.000729123 / −0.001619334 | +0.000500221 / +0.000244885 |

946 个 same-edit 矩形没有显示机制性收益：RECT 相对 epoch 0 更差，相对 ABS 的 CI 跨 0。38 个严格矩形的 RECT−epoch 0 点估计为负，但 95% CI 跨 0（obs [−0.003247, +0.001629]；macro [−0.004498, +0.001190]），且 RECT 相对 ABS 仍为正；按协议只作描述，不据此选择子集。双未见上下文和 molecule-disjoint 子集同样没有显示 RECT 优于 ABS。

## 状态解释（best RECT 与 epoch 0）

状态解释只使用固定 hash 选出的 ≤128 个真实替换对（实际 128 个，来自 1,677 个可用 pair type）和第一个 strict same-edit 矩形；每个 checkpoint 对 254 个分子、3,350 个原子做整图前向，不训练解释器。

公式核对（float64）：

| 检查 | 最大误差 | 通过阈值 |
|---|---:|---|
| mean pooling，mean(h) | 2.22e-16 | 1e-10 |
| mean pooling，mean(h²) | 4.44e-16 | 1e-10 |
| attention pooling，per head | 6.79e-13 | 1e-10 |
| ΔF = Δsum_f + Δrho + Δattention | 1.11e-15 eV | 1e-10 eV |
| model 与 state readout | 7.43e-7 eV | 1e-4 eV |

三个完整实例（数值为 eV；正负号按替换方向 A→B）：

| 实例 | true Δ / D | pred Δ / D | 绝对误差 | Δsum_f | Δrho | Δattention |
|---|---:|---:|---:|---:|---:|---:|
| 真实替换 pair 1973（success） | −0.242181 | −0.242675 | 0.000494 | −0.020418 | −0.360570 | +0.138313 |
| 真实替换 pair 609（fail） | +1.760577 | +0.903357 | 0.857219 | +0.240986 | +0.182316 | +0.480055 |
| strict rectangle 555（success） | D −0.549670 | D −0.492442 | 0.057228 | +2.65e-9 | −0.005088 | −0.487354 |

strict rectangle 555 是 `pair_key=555`，`ctx1=10064`，`ctx2=61124`；两个上下文各自的 ΔF 为 −0.068559 和 −0.561001，差为 −0.492442，与分解一致。

**关键事实：三臂的 best epoch 都是 0，所以 `RECT_best` 就是 epoch 0 的权重。** `state_response_examples.json` 中 `epoch_0` 与 `RECT_best` 两份记录数值完全相同；本轮没有“训练后的 RECT 状态”可解释。上面的状态核验说明 W3 的响应核算精确、且响应主要由 `rho`/attention 项承担，而不是结构监督带来的新能力。

## 正确性检查与 provenance

8 项声明检查全部通过（`analysis/checks.json`，`all_pass=true`，clean commit）：

| 检查 | 结果 |
|---|---|
| epoch 0 重现 W3 | PASS；MAE 0.1278091735 vs checkpoint 0.1278091672（差 6.3e-9 eV）；与冻结目录预测 max\|Δ\| 5.5e-7 eV |
| 三臂 bit 级相同初始参数 | PASS；initial state sha256 均为 `91952d5d…` |
| 相同行/次数/batch plan | PASS；plan sha `b21a9987…`，batch plan sha `8bf0297a…`；epoch 0 MAE spread ≤1.8e-9 eV |
| 矩形方向/端点标签/double difference | PASS；8,000 矩形，四分子互异，C1<C2、A_rank<B_rank |
| 所有行在 1M Train 计划内 | PASS；无 Valid/Test 行 |
| 三个 loss 与手工小样本一致 | PASS；最大差 0 |
| PAIR/RECT 辅助梯度有限且非零 | PASS |
| 无 Test 标签访问 | PASS；Test 从未索引，plan 行全在 Train 1M |

数据与评估复用冻结目录：36,998 个替换观测（36,952 可评估）、211,356 个 exact context、59,214 个矩形 / 1,542 个 pair type、946 个 same-edit 矩形、38 个严格矩形；没有重新选择更有利的子集。

主要产物：

- `analysis/response_metrics.json`、`analysis/model_comparison.csv`、`analysis/response_comparisons.csv`、`analysis/response_curves.csv`
- `analysis/state_response_examples.json`、`analysis/checks.json`
- `analysis/preflight_timing.json`、`analysis/formal_timing.json`
- `data/train_plan.npz`、`data/valid_eval_directory.npz`（均已提交，内容哈希在对应 manifest/json 中）
- 各臂 `runs/PCQM_SSRL_{ABS,PAIR,RECT}/` 保存在本地/远端运行目录（checkpoint 与逐 epoch 预测为运行产物，不随 git 传输）

## 解释与限制

- 这是一个**小规模继续训练**实验，不是对结构监督的一般性否定。它说明：在固定的 5 epoch、8,000 矩形、3e-5→1e-5、fresh AdamW 配方下，PAIR/RECT 的结构替换监督没有把 W3 的完整图预测、实替换响应或环境交互响应变得更好。
- 训练子集 MAE 下降而完整 Valid 与响应指标变差，最直接的解释是模型在拟合本轮 64k 训练行，发生了对本轮分布的过拟合；PAIR/RECT 的辅助项没有阻止这一点，反而比 ABS 略差。PAIR/RECT 的回归斜率略微靠近 1，但 MAE、correlation 和 direction accuracy 没有同步改善，因此不能据此声称响应机制变好。
- 按协议没有做学习率、loss 系数、epoch 数或数据子集的二次搜索；也没有因为中间预测改变 horizon。若要判断结构监督是否在更大数据/更低学习率/更长训练下有效，需要下一轮独立实验，本轮不追加。
- 本轮到此完成；没有 Test 读取，没有新增模块或可训练参数。
