# RBO v0 — 根内分支算子：值不值得继续？

轮次 `zinc_rooted_branch_operator_v0`。代码 commit `cc2bacf` + `10e7920`（正式训练跑在
`10e7920`），checkpoint sha256 `b4b9b6e8…`。**官方 test 未构造、未加载、未预测、未评分。**
所有评估集都来自 official-train 的冻结划分（`zinc_composition_transfer_v1/cache/support_sets.npz`）。

---

## A. 核心判断

**STOP。RBO 没有表现出任何"r2 未见环境更好预测"的信号；阻碍不是局部构件覆盖，也不是训练/优化。**

- 构件覆盖充足（第一阶段）：Q_target 里 1,637 个 r2 未见原子上，中心—键—邻居三元组
  98.8%、二跳路径 94.4%、删边后的邻居分支多重集 97.3% 在 D_minus 中都有实例；
  329/350 个目标分子的全部 r2 未见原子在分支层面被支持。**不是"没有共享单元可学"。**
- 模型是独立的、合法的（第二阶段）：331,521 参数、r=32、无 r2 key embedding、无消息传递、
  无 P3/DCC composer 旁路；只复用 train-only masked-SSL 的 atom/bond 编码器（89k）。
  CPU 门控全过（重编号+键序不变 2.6e-8、batch 不变 1.5e-8、全参数有梯度、checkpoint 往返 0）。
- 训练路径与参考基本重合（第三阶段）：M 曲线在 epoch 100–250 领先 P3-minus 0.004–0.018，
  epoch 300 之后落后 0.001–0.004；选定点的 eval-mode train 误差 **0.0110**，参考 **0.0106**
  （同 epoch 439 的 online train 为 0.0102 vs 0.0111）。两者都把 7,549 个训练分子拟合到 ~1%，
  **不是欠拟合，也不是优化被破坏。**
- 但唯一重要的迁移指标反向：**Q_target 0.1284 vs P3-minus 0.1243（+0.0041，更差）**，
  Q_background 也 +0.0038。预声明的"改善 ≥0.015"完全没有出现，连方向都不对。

## B. 关键数据

同数据（`F_minus` = B 7,098 + D_minus 451 = 7,549 行）、同配方（recipe B：AdamW
1e-3→1e-5、warmup 500、per-step cosine、wd 1e-5、L1、batch 128、600 ep = 35,400 updates）、
同 SSL 编码器、同 batch 顺序；两边都只用 M 选点。参照是冻结的 `sup_minus_s0/best_monitor.pt`
（本轮从其 checkpoint 重算，M 复现到 1e-8）。

| 集合 | n | RBO | P3-minus | Δ (RBO−P3) | 95% CI | 中位 ΔAE | 更好的分子 |
|---|---:|---:|---:|---:|---|---:|---:|
| M（选点集） | 1000 | 0.082757 | 0.080424 | **+0.00233** | [−0.00298, +0.00770] | +0.00040 | 49.6% |
| Q_target | 350 | 0.128432 | 0.124316 | **+0.00412** | [−0.00811, +0.01632] | +0.00064 | 49.4% |
| Q_background | 650 | 0.104169 | 0.100407 | **+0.00376** | [−0.00321, +0.01078] | +0.00049 | 49.4% |

- best-M：RBO 0.082757 @439；参照 0.080424 @424（P3-minus epoch-600 为 0.080989）。
- 差距是**弥散**的，不是尾部驱动的：去掉 |ΔAE| 最大的 5 行后 Δ 仍为 +0.0027（Q_target）/
  +0.0035（Q_background）；AE>0.5 的行数 8 vs 6（Q_target）。
- 关键比值：Q_target/M = **1.552（RBO） vs 1.546（P3-minus）** —— 未见/已见的误差倍率完全没有改善，
  这正是本轮假设要改变的量。
- 按 r2 未见原子比例分层（Q_target）：<0.05 组 −0.0112 [−0.0265, +0.0044]（n=152，跨零）；
  0.15–0.30 组 **+0.0479** [+0.0093, +0.0893]（n=40）；≥0.30 组 +0.0712（n=8）。
  即：**未见内容越多，越没有优势**（0.15–0.30 与 ≥0.30 两个分层分别 n=40、n=8，配对 CI 都不含 0）。
- key 级：45/90 个目标 key 更好，但按 query 数加权是 +0.0051（更差）；n≥3 的 57 个 key 中
  只有 25 个更好（+0.009）。逐 key 的"更好"来自 n=1 的 key，不构成证据。
- 训练曲线（M，与参照同 epoch）：

| epoch | 25 | 50 | 100 | 150 | 200 | 250 | 300 | 350 | 439 | 600 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RBO | .1657 | .1352 | .1038 | .0920 | .0935 | .0923 | .0859 | .0872 | **.0828** | .0832 |
| P3-minus | .1633 | .1312 | .1222 | .0961 | .1004 | .0982 | .0816 | .0852 | .0816 | .0810 |

**机制确实被使用**（冻结 checkpoint 上只做推理消融，不重训）：关掉根内三体项后
M 0.0828→0.1116（+0.0289）、Q_target 0.1284→0.1505；关掉关系项 +0.0060；把 typed histogram
置零 +0.0061。三个通道都有可观、非零的作用，不能把失败归因于"模块没被用上"。

## C. 失败原因：能区分与不能区分

- **不是局部构件覆盖**（第一阶段数据）：T/P/分支 94–99% 被支持；两分支组合 75% 被支持
  （中位支持次数 2，确实稀薄，但存在）。
- **不是训练路径/优化**：同配方同数据，训练拟合与参照持平，M 轨迹在 100–250 epoch 领先、
  epoch 300 之后落后 0.001–0.004，没有"崩掉"或"收敛不足"的形态；无 NaN、全参数有梯度、
  训练/评估路径一致。
- **不是"容量被浪费在没用的通道"**：三个机制通道的消融都显著改变预测和 M。
- **能确定的是**：在这个"粗粒度分支内容 + 共享低秩投影 + 根内 1/2/3 体对称积 + 关系加权二体"
  的具体实现下，**根内归属与组合并没有把 r2 未见环境的误差降下来**；同时整体水平（M、
  Q_background）还差 3%，所以连"用整体性能换未见性能"的交换都没发生。
- **不能确定（未知）**：(i) 是"机制本身无价值"还是"这个分支表示太粗/秩 32/三阶上限太低"——
  本轮没有做匹配容量的 rank/表示对照；(ii) 是否把监督放到这些环境上（上一轮已证明这是大杠杆）
  才是唯一有效路径，而不是换组合方式；(iii) 参照的 r2 联合直方图（693 维 root_joint）里可能
  残留了"部分组合已见"的信息，本轮的分支哈希表没有等价物，因此不能声称"r2 信息无用"。
  但可以声称：**用被支持的构件 + 显式根内组合，没有复现那部分可用信息。**

边界：单 seed、单数据集（train-only 子集）、单 rank、单阶数上限；Q_target/Q_background 只在
checkpoint 冻结后读一次，未用于选点；未有第二 seed，未做"关掉三体"的重训对照，未跑 full-10k，
未读 official test。

## D. 下一决定

**STOP。** 不买 seed1，不做匹配架构的三体关闭对照，不启动 full ZINC 性能试验。

理由：预声明的探索信号是 Q_target 相对同监督 P3-minus **改善 ≥0.015**，实际是 **+0.0041（更差）**；
M/Q_background 也同向变差 0.002–0.004；且未见/已见误差倍率（1.55）与参照完全相同。
即使把 0.003 的整体差距全部归因于容量，也仍没有任何"相对未见环境"的正向信号可供放大。
如果以后还要碰这个问题，需要的不是这个机制的第二个 seed，而是先回答"r2 环境的信息到底是
缺失还是只是没被监督"——本轮与上一轮的证据都把矛头指向后者（监督位置），而不是组合方式。

---

### 复现与成本

```bash
# 1) CPU 第一阶段（支撑审计，4 s）+ 正确性门控（3 s）
uv run python tracks/structural-sufficiency/zinc_rooted_branch_operator_v0/code/rbo_support.py
uv run python tracks/structural-sufficiency/zinc_rooted_branch_operator_v0/code/rbo_prep.py
uv run python tracks/structural-sufficiency/zinc_rooted_branch_operator_v0/code/rbo_checks.py
# 2) 正式训练（res, res-gpu0, A100-40GB, commit 10e7920）
rr run res rbo-minus-s0 --pool res-gpu0 --gpus 1 --cpus 4 --time 0:50:00 \
  --result .../runs --result .../analysis -- \
  python tracks/structural-sufficiency/zinc_rooted_branch_operator_v0/code/rbo_train.py --seed 0 --device cuda:0
# 3) 冻结后比较（本地 CPU，3 s）
uv run python tracks/structural-sufficiency/zinc_rooted_branch_operator_v0/code/rbo_analyze.py
```

- GPU：job wall 863 s = **0.24 GPU·h**（trainer 自测 850 s），单卡，无失败/取消。
- 墙钟：整轮约 45 min（含设计、CPU 门控、14 min 训练、分析）。
- 证据：`analysis/t0_support.json`（第一阶段）、`t0_prep.json`、`t0_checks.json`（门控+玩具）、
  `summary.json` / `main_table.md` / `per_molecule.csv` / `per_key.json` / `curve.csv`、
  `rr_status.json`（provenance）。checkpoint 与 `train.json` 在 `runs/rbo_minus_s0/`（git-ignored，
  远端保留）。
