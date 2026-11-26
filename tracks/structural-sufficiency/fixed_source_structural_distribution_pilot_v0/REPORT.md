# FSDP v0 — 真实结构组织 vs 分子内结构内容分布（MEANFIELD）短程验证

- Round：`tracks/structural-sufficiency/fixed_source_structural_distribution_pilot_v0/`
- Round ID：`fixed-source-structural-distribution-pilot-v0`；Protocol：`fsdp-v0`（训练前冻结于 `PROTOCOL.md`）
- 参照：`direct_structural_computation_recurrence_v0`（FIXED1，只读复用，不重训）
- 数据：`ogbg-molpcba` 官方 scaffold split，完整 Train / Valid；**全程未访问 Test**（无标签读取、无打分）
- 新模型：仅 `MEANFIELD`，seed 0，12 epoch；参数 1,128,468（与 FIXED1 完全一致）
- 实际计算：`res`（process 后端），单卡 A100-SXM4-40GB，12 epoch 完整跑完，**0.5139 GPU-hour**（软上限 0.8）
- 判定：**`LOCAL_ALLOCATION_ADVANTAGE_SIGNAL`**

---

## 0. 结论摘要

去掉“每条真实边对应一个独立来源内容”的绑定，只保留每个分子的结构内容分布 + 每个中心的真实关系类型计数后，Valid 官方全任务 macro AP 从 **0.269992 降到 0.249413（−0.020575）**。落后量不仅超过预设的 0.008 阈值，而且方向一致地出现在高支持任务（−0.013508，78/107 个任务退化）和 12 个 epoch 的后半段曲线上；任务 bootstrap 的 95% 区间为 [−0.034863, −0.010550]，2000 次重采样中 **没有一次** 让差值变正。

因此本轮证据支持“具体来源内容与真实中心的局部对应关系仍然被 FIXED1 实际利用”。

同时必须强调三点限制：(a) 干预同时改变了局部内容差异性与全局信息访问范围，不能分离；(b) 单 seed、12 epoch、FIXED1 在 epoch 12 仍在上升，差距是否继续扩大无法由本轮判定；(c) MEANFIELD 仍高于 W2（+0.003192，区间跨 0），所以不能反过来说“真实拓扑无用”。

---

## 1. 本轮实现（相对 FIXED1 的唯一改动）

复用内容（逐字、只读）：W2 主干与 S1/S2 局部编码、两个 FIXED1 block、LayerNorm、`bond_proj`、残差与 FFN、`f`/`rho`/四头 attention readout、数据、初始化、训练配方。未使用 RSSC 低秩分支。

每个 block 的有向边消息：

```python
p = SiLU(z_source[src] + edge_msg)                      # 与 FIXED1 逐字相同
group_id = unique((graph_id, raw_bond_feature_signature))
group_mean = segment_sum(p, group_id) / group_count      # 向量化 index_add_
p_meanfield = group_mean[group_id]
m = scatter_sum(p_meanfield, dst)
h_next = h_current + FFN(z_target, m)
```

- block1：`z_source = LN1(h0)`，`z_target = LN1(h0)`；block2：`z_source = LN2(h0)`，`z_target = LN2(h1)`；从不读取 `h1[src]`/`h2[src]`。
- 键属性签名使用**原始 13 维离散 bond feature 的完整元组**（先在冻结缓存上核验：12,309,186×13，取值仅 {0,1}，整数），用 `torch.unique(dim=0)` 精确分组；不使用学习后的 embedding，不做任何分桶。
- 不新增参数、不重算局部结构编码、不加随机置换；不出现逐分子 Python forward 循环。

**信息访问范围声明**：MEANFIELD 保留全部真实有向边、每个中心的真实入边数与入边类型计数，但把每条边的来源内容替换为其组均值，因此同一 (分子, 键属性) 组内是分子范围的信息共享。**它不是与 FIXED1 信息访问等价的变体**；这个差异无法在本轮设计中剥离。

结构性副产物（实测，非假设）：同一物理键的两个方向携带同一原始 bond feature，必然落在同一组，所以有向边组大小恒为偶数（实测 odd-size 组数 = 0），单元素组不可能出现，且整个 layer 对**全局方向翻转不变**（FIXED1 不是：两方向原始消息最大差 3.159）。

## 2. 正确性检查与对比度门控（训练前）

7 组检查全部通过（`allocation_checks.json`，seed 0，实数数据 + 合成图）：

| 检查 | 结果 |
|---|---|
| P1 参数量 + **全部 71 个初始化张量**与 FIXED1 完全一致 | 1,128,468 = 1,128,468；差异张量 0；W2 trunk fingerprint 与记录 seed-0 一致 |
| P2 不跨分子、不跨键属性组混合 | 每组成员同分子同签名 |
| P3 每个 (分子, 键属性) 组消息总和守恒 | float64 相对误差 0（float32 GPU 实测 5.6e-5 绝对 / 5.8e-3 相对） |
| P4 每个中心真实入边数与类型计数不变 | 逐中心、逐 (中心,类型) 计数完全一致 |
| P5 见证：同组不同来源内容确实替换绑定 | FIXED1 两条边内容最大差 11.29 → 置换后 0；中心 2 的聚合确实改变 |
| P6 原子重编号 / 边序 / 方向翻转不变性 | 最大绝对差 1e-16 量级（float64，容差 1e-12） |
| P7 空边图 + 梯度可达 | 聚合恒为 0、输出有限、两 block 与 h0 梯度非零 |
| P8 Test 隔离 | 静态扫描无 Test 打分调用；Test 标签全为 NaN；运行期 `evaluate()` 拒绝非 train/valid |

**真实 Valid 对比度**（冻结样本：官方 Valid 前 1,024 行，不洗牌；模型处于官方初始化状态）：

| 量 | 值 |
|---|---|
| 有向边 / 组数 | 60,204 / 5,593 |
| 组大小 | 均值 10.76，p50 = 8，p90 = 24，max = 80，**全为偶数** |
| 单元素组比例（边/组） | 0 / 0 |
| 被改变的消息比例 | **0.99997**（60,202 / 60,204） |
| 聚合向量变化 RMS（block1，同一状态） | **0.4030** 相对（block2 同状态 0.4030） |
| 组消息总和守恒误差 | 最大绝对 5.6e-5，最大相对 5.8e-3 |
| 门控 | `SUFFICIENT_CONTRAST`（阈值：相对 RMS ≥ 1e-2 且改变比例 ≥ 0.5） |

`block2_relative_rms_change_end_to_end ≈ 1.6e-8` 是初始化时的正常结果：两个 `fc2` 均为零，`h1 = h0`，所以两臂的 block2 输入状态完全相同；同状态对比（0.4030）才是干预幅度。训练曲线里该量全程稳定在 0.432–0.437，说明干预强度没有随时间衰减。

## 3. 训练结果（12 epoch horizon，seed 0）

| model | 参数 | best ep | Valid AP（best） | endpoint Valid AP | last-3 mean | Train AP @12 | Train BCE @12 | Valid BCE @12 | 高支持任务 AP（107） |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **MEANFIELD** | 1,128,468 | 10 | **0.249413** | 0.248112 | 0.248283 | 0.440975 | 0.030246 | 0.046101 | 0.270088 |
| FIXED1（主对照，历史） | 1,128,468 | 12 | 0.269992 | 0.269992 | 0.263256 | 0.447919 | 0.029790 | 0.045382 | 0.283597 |
| SELF（容量控制，历史） | 1,128,468 | 12 | 0.255413 | 0.255413 | 0.253573 | 0.443298 | 0.029936 | 0.045907 | 0.276094 |
| W2（非 MP 基准，历史） | 994,324 | 12 | 0.246224 | 0.246224 | 0.245125 | — | — | 0.045909 | 0.266657 |

主指标始终是**官方全任务 macro AP**；高支持子集（Valid 正例 ≥ 20，107 任务）只作补充。

逐 epoch Valid AP：

| ep | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MEANFIELD | .0977 | .1550 | .1792 | .2047 | .2175 | .2266 | .2350 | .2409 | .2422 | **.2494** | .2473 | .2481 |
| FIXED1 | .0992 | .1529 | .1813 | .2068 | .2217 | .2375 | .2446 | .2521 | .2492 | .2536 | .2662 | **.2700** |
| SELF | .0975 | .1557 | .1801 | .2057 | .2206 | .2317 | .2411 | .2441 | .2479 | .2513 | .2540 | .2554 |
| W2 | .0958 | .1542 | .1784 | .1997 | .2150 | .2260 | .2336 | .2335 | .2397 | .2432 | .2460 | .2462 |

曲线形状是本轮最重要的信息之一：**epoch 1–4 两臂不可区分**（甚至 MEANFIELD 在 ep2 略高），分叉从 epoch 5–6 开始，此后 FIXED1 持续拉开（ep11→ep12 还在 +0.0038），而 MEANFIELD 在 0.242–0.249 附近进入平台。这不是“一开始就学不到”，而是**绑定在训练后段才被大量利用**。

## 4. 逐任务分析（MEANFIELD − FIXED1，127 个可评分任务，配对）

| 量 | 值 |
|---|---|
| macro AP 差 | **−0.020575** |
| 任务中位数差 | −0.008595 |
| 改善 / 退化任务数 | **36 / 91**（改善占比 28.3%） |
| 高支持子集差（107 任务） | **−0.013508**（29 改善 / 78 退化） |
| 任务 bootstrap（2000 次，seed 0） | 均值 −0.020644，95% CI **[−0.034863, −0.010550]**，P(差>0) = 0.000 |
| 去除 \|差\| 最大 3 个任务（108, 9, 28）后的差 | **−0.015449**（剩 124 任务） |

三个最大绝对差任务：task 108（FIXED1 1.000 vs MEANFIELD 0.250，Valid 仅 1 个正例，属低支持）、task 9（+0.179，MEANFIELD 更好）、task 28（−0.126）。剔除这三个任务后差距从 −0.0206 收到 −0.0154，方向与显著性不变，说明结果不是少数极端任务驱动的。

次级配对（同 horizon，仅背景）：MEANFIELD − SELF = **−0.005997**（55/127 改善，CI [−0.0136, +0.0012]）；MEANFIELD − W2 = **+0.003192**（71/127 改善，CI [−0.00567, +0.01283]）。即 MEANFIELD 大体停在“W2 与 SELF 之间、靠近 W2”的位置。

## 5. 冻结干预（与独立训练严格分开报告）

在历史 FIXED1 最佳 checkpoint（`runs/FIXED1_seed0/checkpoint_best.pt`，**epoch 25**，记录 Valid AP 0.275846）上，权重完全不动，只把消息聚合规则换成 MEANFIELD，做一次**完整 Valid 前向**（43,793 行）：

| 量 | 值 |
|---|---|
| FIXED1 原计算（同权重复评） | Valid AP 0.275851，BCE 0.046377 |
| MEANFIELD 替换（同权重） | Valid AP **0.211502**，BCE 0.050091 |
| Δ Valid AP | **−0.064349** |
| 任务方向 | 123/127 退化，4/127 改善 |
| logit 相对变化 RMS | 0.1717 |

这条证据的性质与第 3 节不同：它说明**已经训练好的 FIXED1 权重高度依赖具体来源绑定**（一次性移除损失 0.064 AP），而第 3 节说明**从头在分布化计算下训练 12 epoch 只落后 0.021 AP**。后者小于前者，说明分布化训练确实部分补偿了绑定缺失；但两者都指向同一方向。二者不能互相替代，也不能混为“同一实验的两次读数”。

## 6. 四个必答问题

**Q1 删除具体来源内容的局部对应关系后，性能损失有多大？**
Valid 官方全任务 macro AP −0.020575（−7.6% 相对），高支持子集 −0.013508，任务中位数 −0.008595，91/127 任务退化。同权重冻结干预的损失更大（−0.064349）。Train 侧损失小得多：Train AP@12 −0.006944（−1.6%），Train BCE@12 +0.000456。

**Q2 Train 与 Valid 的变化是否一致？**
方向一致（都变差），量级明显不一致：Train 退化为 −1.6%，Valid 退化为 −7.6%，Train−Valid 差距从 FIXED1 的 0.1779 扩大到 MEANFIELD 的 0.1916。也就是说 MEANFIELD 的拟合能力损失不大，** generalization 损失更大**。曲线层面也一致地表现为“前期无差别、后期分叉”，并且 MEANFIELD 的 Valid BCE 从 epoch 3 起就一直更高。

**Q3 真实局部组织与分子内内容共享，哪种计算方式更值得继续？**
本轮证据支持**真实局部组织（FIXED1 式的来源—中心绑定）继续作为主线**：损失超过预设阈值、跨任务广泛、在高支持任务上同向、后段曲线仍在扩大、且冻结干预独立地给出同方向结论。分布化计算不是没有信息（它仍高于 W2，介于 W2 与 SELF 之间），但作为替代路线目前没有竞争性证据。

**Q4 哪些竞争解释仍然无法区分？**
1. **局部差异性 vs 全局访问范围**：本干预同时移除了“每边独立内容”和“访问范围限于真实一跳”，两者不可分离。因此不能断言“绑定本身有独立因果收益”。
2. **方向信息**：MEANFIELD 的组均值对方向对称（这正是它方向翻转不变的原因），FIXED1 使用有向消息。损失中有一部分可能来自方向对称化，而不是来源身份本身。
3. **消息场的有效多样性/秩**：每组只贡献一个向量，有效消息多样性大幅下降；参数量虽不变，但“表达能力”与“信息损失”在本设计中不可分辨。
4. **horizon 外推**：12 epoch 时 FIXED1 仍在上升、MEANFIELD 已平台；差距会继续扩大还是收敛，本轮无法判断（不追加训练长度是协议规定）。
5. **分组粒度**：键属性签名把同一分子的同类型键全部合并（平均组大小 10.76）。更细的分组（例如按分子内序号、或按局部化学环境）会给出不同折中，本轮未探索。
6. **结构信息仍在**：`h0`、关系类型计数、原始全局描述符仍携带结构信息，且 MEANFIELD > W2，所以“真实拓扑无用”不成立。

## 7. 判定

按冻结规则：AP 差 = −0.020575 < −0.008，且高支持任务同向（−0.013508）⇒ **`LOCAL_ALLOCATION_ADVANTAGE_SIGNAL`**。

该标签表示：在本轮单 seed、12 epoch、PCBA 官方 Train/Valid 的设定下，**具体来源内容与真实中心的局部对应关系带有可测的性能价值**。它不是统计等价性结论，也不是“MEANFIELD 无效”的一般性断言——见 Q4 的六条未分离解释。

按协议，不因该结果追加模型、seed、分组方式或训练长度；本轮到此停止。

## 8. 运行事实与复现入口

- 运行时的 Git commit：`5528f83bb767`（正式训练只执行该提交；结果与报告随后提交为 `973e327d`）
- 墙钟说明：计算本身 0.514 GPU-hour 在预算内，但 `res-2` 队列 41 分钟未获资源（`(Resources)`），最终改在 `res` 执行；加上实现与本地验证，本轮 agent 墙钟超出 75 分钟硬上限，超时全部来自排队与开发，不来自训练本身
- 执行：`rr run res fsdp-meanfield-s0-res2 --pool res-gpu1 -- uv run python tracks/.../code/mf_train.py --out-dir ... --epochs 12`；node `a100-2`，GPU `NVIDIA A100-SXM4-40GB`（CUDA_VISIBLE_DEVICES=1），torch `2.5.1+cu124`，driver 550.163.01
- 训练：32,856 steps = 12 × 2,738，batch 128，冻结 40-epoch cosine 前缀（LR 3.0e-4 → 2.467e-4 @ ep12），fp32，dropout 0.10，clip 5
- 时间/预算：单 epoch 训练 ≈ 131 s；总 GPU 0.5139 h；GPU peak allocated 431 MB
- epoch 0 Valid AP 0.029063165，与未训练 W2 seed-0 参考 0.029063332 一致（差 1.7e-7），确认初始化未漂移
- 最佳 checkpoint（epoch 10）复评：Valid AP 0.249416100（与记录值差 3.0e-6，来自 `index_add_` 归约顺序），Train AP 0.405818 / Train BCE 0.031509
- 运维备注：`res` 上 `fsf-v1` 遗留的未跟踪目录 `fixed_source_role_separation_followup_v1/analysis/` 阻塞了 `rr deploy`，已原样移到 `/home/hxy/cy/rr-leftovers/fsf-v1-analysis-20261126-131023`（未删除、未修改内容）。res-2 上的同配置 Slurm 作业排队 41 分钟未获资源（`(Resources)`），已取消，最终结果来自 `res`，本轮只有这一个执行 regime。

交付文件：

```
PROTOCOL.md                                  configs/round.json
code/mf_common.py  mf_model.py  mf_checks.py  mf_contrast.py
code/mf_train.py   mf_analyze.py  mf_frozen.py
tests/test_meanfield.py                      （10 项，全部通过）
analysis/main_table.csv          analysis/epoch_curves.csv
analysis/paired_comparisons.json analysis/allocation_checks.json
analysis/checkpoint_manifest.json analysis/per_task_metrics.csv
analysis/frozen_intervention.json analysis/summary.json
results/runs/MEANFIELD_seed0/{train_curve.csv, per_task_ap.csv, run.json,
   contrast.json, structural_checks.json, allocation_checks.json,
   best_checkpoint_eval.json, checkpoint_manifest.json,
   checkpoint_best.pt, checkpoint_last.pt, valid_logits_best.npy}
```
