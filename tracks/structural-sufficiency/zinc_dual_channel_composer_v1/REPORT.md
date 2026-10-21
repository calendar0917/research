# ZINC P3 dual-channel local/global composer v1 — 报告

**Track:** `tracks/structural-sufficiency/zinc_dual_channel_composer_v1/`
**代码/配置起点 commit:** `504edbb7b4b5`（实验轮开始前的冻结版本）
**状态:** 完成 L/C seed 0 各 600 epochs；valid-only 分析完成；official test 未读取。
**数据:** official PyG ZINC subset=True，train 10,000 / valid 1,000，raw-y MAE。
valid 是反复使用过的开发集，不是新盲测。

---

## 0. 一句话结论

**结果分类 C：ownership-only，no performance。** L 比匹配的 Null-local 对照 C
好 **0.003830**（0.078537 vs 0.082367），达到预注册的 `L−C ≤ −0.002` pilot
阈值；但 L 比 P3B 差 **+0.002313**，C 比 P3B 差 **+0.006144**，两臂都没有
新的性能水平。L 在 signature-unseen 组几乎追平 P3B（−0.000383），但 overall
被 covered 组和 `y≤−3` 尾部（尤其 row 172）拖坏。单 seed pilot 的 ownership
信号值得记录，但**不足以继续购买 seed 1 或扫 width/reader/grouping**；该具体
配置按预注册规则停止。

**official test 全程未读取。** 用户在正式跑完后曾说“看一下 test”，但本 track
的冻结协议和仓库 guardrail 明确禁止读取 official test labels/cache/predictions；
test 已经 spent、不再 pristine。因此没有打开 test，也没有用 test 做任何选择。
若要做 reporting-only terminal evaluation，需要新的 preregistration 和明确授权。

---

## 1. 边界与完整性

- 只训练 L/C seed 0；没有 seed 1、没有第三臂、没有重训历史 baseline。
- P3B 参照为 `zinc_recipe_trunk_v2/runs/xBssl_P3N_s0`，best-valid
  **0.0762237511 @ epoch 377**，413,505 参数；只读取其 valid 预测和 train.json。
- L/C 从同一 seed-0 train-only masked SSL encoder（sha256 `006fabc1...`）和
  同一份 dual 初始 state 出发；L/C `initial_state_sha256` 都是
  `f5c34c285c98b4d8a88361cb7dc1043b5abf07799f4f6121b25a59c904166e33`。
- 不加载 P3B best.pt 作为初始化；不冻结 encoder；不加输出门控。
- 没有新特征、目标公式、外部数据、辅助监督、SSL 任务、loss 或优化器网格；
  没有递归局部 message passing / GNN / attention；局部 composer 只做一次
  per-bond 静态 incidence 聚合。
- 原始预测、checkpoint、标签和历史训练成绩未改动；本 track 只新增自己的
  run 与派生分析。

## 2. 实现与 CPU gate

模型为 P3 Null 全局通路 + 并联局部 composer：

```
c_glob = P3 group_composer([b, Wg, Vg, log1p(n_b), bond_feat])
L: Wl_b = sum_{j touches b} w_j,      Vl_b = sum_{j touches b} w_j^2
C: Wl_b = n_b * graph_mean(w),        Vl_b = n_b * graph_mean(w^2)
c_loc  = LocalComposer([b, Wl, Vl, log1p(n_b), bond_feat])
reader = MLP(513 -> 256 -> 256 -> 1)
```

参数 **494,193**（P3 413,505 + local composer 31,536 + reader delta 49,152），
cap 500,000，headroom **5,807**。reader 前 321 列复制 P3、后 192 列零初始化；
初始 forward 与 P3 的 max abs diff **4.17e-7**。

`analysis/cpu_checks.json` 的 26 项检查全部通过，包括：

- L/C 参数量、state keys/shapes、逐 tensor 初始 state 完全相同；
- p3 base 权重与参照 P3N builder 逐 tensor 一致；reader 前 321 列一致、后 192 列为 0；
- organized 聚合每条 junction 进入两条 bond，`sum_b Wl = 2 sum_j w`、
  `sum_b Vl = 2 sum_j w^2`；Null 对照与原 Null 逐键一致；空 junction 边界有限；
- 图重编号与 batch composition 不变性（≤1e-6，实测 5.96e-8 / 2.38e-7）；
- reader 新列 step 1 有梯度；local composer step 1 梯度为 0（零列导致，预期），
  step 2 非零且有限；
- checkpoint save/load 一致；trainer optimizer 覆盖全部 494,193 参数；
- 无 official test 路径。

## 3. 计时与预算

### 3.1 真实 timing

L/C 各在 A100 上跑了 **3 个完整 10k train epoch**，每 epoch 都做完整 valid-1000，
确认 **79 steps/epoch**。另用 2-epoch replicate 复核，C 确实比 L 慢。

| arm | epoch 1/2/3 seconds | mean | steady (ep2–3) | steps/epoch |
|---|---|---:|---:|---:|
| L | 2.477 / 2.127 / 1.958 | 2.187 | 2.043 | 79 |
| C | 3.026 / 2.518 / 2.417 | 2.654 | 2.467 | 79 |

按“600 × mean epoch × 1.15 + startup + final eval + 30s process reserve”的
保守投影：L **0.4289h**、C **0.5185h**，两臂合计 **0.9474h**；加上已花 timing
会超过原 0.8594h cap，原规则会 STOP。用户在 timing 后明确说“不用管GPU预算，
跑出结果”，因此启动了两条 600-epoch 正式轨迹；walltime 各 1:30:00。

**实际占用（scheduler job occupancy，两卡相加）：**

| item | run id | occupancy h |
|---|---|---:|
| timing L | dcc-timing-L-20261021-110337-aad0a5fa | 0.00444 |
| timing C | dcc-timing-C-20261021-110337-ab4bb2ef | 0.00528 |
| timing2 L | dcc-timing2-L-20261021-110831-8459d70a | 0.00444 |
| timing2 C | dcc-timing2-C-20261021-110831-386f91e3 | 0.00472 |
| formal L | dcc-L-s0-20261021-111139-2abef61e | **0.34833** |
| formal C | dcc-C-s0-20261021-111139-06ba66c8 | **0.40306** |

实际总占用 **0.77028 A100-h**，原 cap 剩余 **0.08913h**。保守投影高估了实际
占用（0.9474 vs 0.7514 formal occupancy），但这是事后事实，不改变“原计划
本应 STOP、后经用户明确要求继续”的记录。`analysis/budget_ledger.json` 保留
了两种口径（run-level 与 job occupancy）和 override 说明。

## 4. 主表

valid-1000，raw-y MAE。P3B 的 GPU-h 是历史 run-level wall；L/C 是 rr job
occupancy。在线 train MAE 是 epoch 内 pre-update running mean；eval-mode train
是固定 checkpoint 在完整 10k train 上的评估。

| model | params | best valid | best ep | final valid | best train (eval) | final train (eval) | best online | final online | GPU-h |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P3B_s0 | 413,505 | **0.076223754** | 377 | 0.078178808 | 0.014680 | 0.000820 | 0.019724 | 0.000583 | 0.3014 |
| L_s0 | 494,193 | 0.078537152 | 489 | 0.079076417 | 0.005510 | 0.000494 | 0.005568 | 0.000614 | 0.3483 |
| C_s0 | 494,193 | 0.082367432 | 482 | 0.082965206 | 0.004906 | 0.000582 | 0.006010 | 0.000634 | 0.4031 |

三个模型都完整跑了 600/600 epochs，没有 INCOMPLETE。checkpoint 重新加载后
valid/train eval-mode MAE 与 train.json 的差都在 1e-8 量级；`valid_best.npz` 的
row=0..999 与 stored best-valid MAE 完全一致（abs diff 0.0）。见
`analysis/checkpoint_verification.json`。

## 5. 配对差（valid-1000，固定 checkpoint）

10,000 次 paired bootstrap，所有对比共用同一组 row resamples。Δ 定义为
`AE(first) − AE(second)`，负值表示前者更好。CI 是对 exposed valid 上固定
checkpoint 的描述性区间，不是跨 seed 确认。

| contrast | mean Δ MAE | 95% CI | 读法 |
|---|---:|---|---|
| L − P3B | **+0.002313** | [−0.005810, +0.013579] | L 更差；CI 跨 0 |
| C − P3B | **+0.006144** | [−0.001295, +0.017241] | C 更差；CI 跨 0 |
| L − C | **−0.003830** | [−0.007780, +0.0000716] | L 更好；CI 上界几乎贴 0 |

按预注册阈值：

- 性能信号 `≤ −0.003`：L **未达到**（+0.002313）；C **未达到**（+0.006144）。
- 归属信号 `L−C ≤ −0.002`：**达到**（−0.003830），但只是 single-seed、
  exposed-valid 的描述性信号，CI 仍跨 0。
- 没有 “C 赢 P3B、L 不比 C 强” 的情况；也没有 “L/C 都赢 P3B” 的情况。

## 6. 固定子群与 tradeoff

所有子群 mask 在 L/C 结果前固定；没有用真实 y 或 unseen 分组做模型路由，
没有按子群选 epoch。表中 MAE 为组内 mean AE。

| group | n | P3B | L | C | L−P3B | C−P3B | L−C |
|---|---:|---:|---:|---:|---:|---:|---:|
| signature-unseen | 251 | 0.151340 | 0.150957 | 0.156079 | **−0.000383** | +0.004739 | −0.005122 |
| signature-covered | 749 | 0.051051 | 0.054268 | 0.057666 | +0.003217 | +0.006614 | −0.003397 |
| ordinary `y>−3` | 935 | 0.068277 | 0.065537 | 0.069504 | **−0.002740** | +0.001228 | −0.003967 |
| tail `y≤−3` | 65 | 0.190540 | 0.265538 | 0.267399 | **+0.074998** | +0.076859 | −0.001861 |
| rings≥4 | 213 | 0.113035 | 0.111694 | 0.115130 | −0.001341 | +0.002095 | −0.003436 |
| rings≥4 & unseen | 47 | 0.314125 | 0.317617 | 0.321616 | +0.003492 | +0.007491 | −0.003999 |
| fixed P3B worst-100 | 100 | 0.359040 | 0.365397 | 0.378193 | **+0.006357** | **+0.019153** | −0.012796 |

**unseen vs covered 的差之差（DoD）**：

- L−P3B：**−0.003600**（L 相对 P3B 的优势在 unseen 比 covered 大 0.0036；
  但 L 在 unseen 的绝对改善只有 −0.000383，covered 反而 +0.003217）。
- C−P3B：−0.001876。
- L−C：−0.001725（L 相对 C 的优势也略偏向 unseen，但幅度小于 0.002）。

**安全性/tradeoff（明确报告，不掩盖）：**

- ordinary `y>−3`：L **改善** −0.002740，C 退化 +0.001228（未超过 +0.002）。
- fixed P3B worst-100 绝对 MAE：L **+0.006357**，C **+0.019153**，两臂都恶化。
- top-100 误差份额：P3B 47.1%、L 51.5%、C 49.9%。这只是描述；不作为硬 gate，
  也不拿份额替代总体 MAE。
- 尾部 `y≤−3` 是 L 总体变差的主因：65 行 MAE 从 0.1905 升到 0.2655。单个
  **row 172**（y=−20.34）的 AE 从 1.563 升到 6.099，单项给 L−P3B 贡献
  **+0.004536**；去掉 row 172 后 L−P3B = **−0.002225**（弱正信号，仍 <0.003）。
  不能把“去掉 row 172”当主结果，但它说明 L 的 overall 损失高度集中在尾部极端个案。
- C 去掉 row 172 后仍为 +0.001486，C 仍差于 P3B。

## 7. Interpretation

### 7.1 性能：没有新水平

L 和 C 都没有达到 `≤−0.003` 的性能信号。L 比 C 好 0.00383，但 L 仍比 P3B
差 0.00231；C 比 P3B 差 0.00614。因此：

- 不能把 L 的 `L−C` 结果包装成“总体性能改善”；
- 不能因为 L 在 ordinary、unseen 或 rings≥4 的局部方向为正，就忽略 tail 和
  fixed worst-100 的退化；
- C 也没有“容量足够就赢”的迹象；C 明显最差。

### 7.2 归属：有 pilot 信号，但不是已确认机制

`L−C = −0.00383` 达到预注册 pilot 阈值，说明在这个训练设置下，**organized
local grouping 相对于同参数、同初始化的 Null-local 对照确实产生了可测差异**；
而且这个差异在 unseen（−0.00512）、rings≥4（−0.00344）、fixed worst-100
（−0.01280）上更大。这个方向和“局部 ownership 有信息”一致。

但必须同时报告反证/限制：

- CI 上界 +7.2e-5，基本贴 0；single seed，不能当跨 seed 确认。
- L 相对 C 的优势没有转化为相对 P3B 的优势；两者都差于 P3B。因此这更像
  “局部通道在优化/读出头上有可测作用”，而不是“Null 分组是泛化瓶颈、并联
  就能修复”。
- C 的 Null-local 对照本身不是 P3B：它多了同参数 local composer 和 reader
  扩展，所以 L−C 只隔离 grouping，不隔离“是否该加这条通路”。
- 冻结 P3B 权重换 grouping 变差（审计的 0.1213 probe）仍然不能证明并联通路
  会更好；本轮的并联结果也没有给出总体性能证据。

### 7.3 未见签名组：方向略偏，但绝对改善可忽略

L 在 signature-unseen 251 组：0.150957 vs P3B 0.151340，**−0.000383**，基本
是持平；C 在该组 +0.004739 更差。DoD（L−P3B）−0.003600 说明 L 的相对优势
确实更集中在 unseen，而不是 covered；但绝对幅度远低于 0.003 性能线，且
unseen 组只占 251 行，不能抵消 covered 组的 +0.003217 和 tail 的 +0.074998。

所以本轮**不能**说“未见签名组被改善了”作为主结论；更准确的说法是：
“L 在 unseen 组相对 P3B 不再恶化（−0.00038），收益方向略偏向该风险组，
但绝对改善可忽略，整体仍被 covered/tail 吃掉。”

### 7.4 仍不知道什么

- 单 seed 无法区分“organized grouping 真有泛化收益”与“这一次优化的随机路径”。
- 没有每个 pattern 的独立标签；样本/结构支持不足（审计 H2）仍未被排除。
- 没有证据证明 Null pooling 是全部泛化问题的病因；也没有证据证明局部组合
  在所有配置下无用。
- 没有做 width/reader/grouping 扫描；不能从这一个配置推断“再宽一点/再深一点”
  会不会更好，也不应在本轮后自动去扫。
- 没有 test 结果；所有结论只针对 exposed valid-1000。

## 8. Decision 与下一轮

**Decision: C — L 赢 C 但未赢 P3B，无新性能水平，结束该配置。**

- 不买 seed 1；不跑 width/reader/grouping 网格；不重开 M/S/T/J/Q/ESR/bond
  composer；不把本轮结果包装成“泛化问题已解决”。
- 如果未来要确认 `L−C` 这个 ownership pilot（不是当前建议），至少需要
  **P3B seed1 + L seed1 + C seed1** 三条同 recipe 轨迹；按本轮实际 occupancy
  粗估约 `0.30 + 0.35 + 0.40 = 1.05h`，另加 timing/启动余量。不能只报
  “加一个 L seed1 约 0.33h”就称跨 seed 受控确认。
- 若要做 official test 的 reporting-only 终点评估，必须先有新 preregistration
  和明确授权；本轮没有做，也不应用 test 来重新选择模型。

## 9. 复现与文件

CPU gate：

```bash
uv run python tracks/structural-sufficiency/zinc_dual_channel_composer_v1/code/dcc_cpu_checks.py
```

正式训练（本 track 的实际远端入口；host 在 experiment 前）：

```bash
rr run res dcc-L-s0 --pool res-gpu0 --gpus 1 --time 1:30:00 \
  --result tracks/structural-sufficiency/zinc_dual_channel_composer_v1/runs -- \
  python tracks/structural-sufficiency/zinc_dual_channel_composer_v1/code/dcc_runner.py \
  train --arm L --seed 0 --recipe B --device cuda:0

rr run res dcc-C-s0 --pool res-gpu1 --gpus 1 --time 1:30:00 \
  --result tracks/structural-sufficiency/zinc_dual_channel_composer_v1/runs -- \
  python tracks/structural-sufficiency/zinc_dual_channel_composer_v1/code/dcc_runner.py \
  train --arm C --seed 0 --recipe B --device cuda:0
```

分析：

```bash
uv run python tracks/structural-sufficiency/zinc_dual_channel_composer_v1/code/dcc_analysis.py
```

产物：

- `runs/L_s0/`、`runs/C_s0/`：best.pt、final.pt、curve.csv、train.json、
  predictions/valid_best.npz、artifacts.json（大 artifact 本地 git-ignore）。
- `analysis/main_table.json` / `.md`、`paired_contrasts.json`、`subgroups.json`、
  `per_molecule_dual.csv`、`model_metrics.json`、`decision.json`、
  `timing.json`、`budget_ledger.json`、`artifact_hashes.json`、
  `checkpoint_verification.json`、`curves.png`。

**Artifact hashes（sha256，完整值见 `analysis/artifact_hashes.json`）：**

| model | best.pt | final.pt | valid_best.npz |
|---|---|---|---|
| L_s0 | `fab04ec0d5fd56f338379d7dcb223e31208338a8a89a335a25a6f0c4f5226cd5` | `bc0ed5ebc1a911385c877b019b512024805718e08c8cc878005556fc38da7c21` | `55e4126d3da925d0067d0d007e7eaee902b7bceea20f3cc102ed515e411f1a56` |
| C_s0 | `0a828c494f75138653ceb5677199f7b17e77943b5251bfd4b901406fe6739067` | `2c512fb0a1e633a041a33e03f5385f7f51ea2fc70580a6bcf3377b717089b69b` | `060b5fe9068a83abec7e1adde4b6bd91017db7043d933bfd231575278b2feb82` |

L/C 初始 state hash（两臂相同）：
`f5c34c285c98b4d8a88361cb7dc1043b5abf07799f4f6121b25a59c904166e33`。

审计 erratum 见
`tracks/structural-sufficiency/zinc_generalization_audit_v1/ERRATUM.md`。
