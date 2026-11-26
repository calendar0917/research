# FSR v0 — 固定来源结构计算的中心—邻域角色分离（短程验证）

- Round: `tracks/structural-sufficiency/fixed_source_role_separation_pilot_v0/`
- Protocol id: `fsr-v0`；Round ID `fixed-source-role-separation-pilot-v0`
- 数据：`ogbg-molpcba` 官方 scaffold split（Train 350343 / Valid 43793，128 任务，masked BCE，官方 OGA macro AP）
- seed 0、仅 PCBA、最多两个新训练单元（TIED、RSSC）；**不运行 Test**
- 历史参照：`direct_structural_computation_recurrence_v0/results/runs/FIXED1_seed0`（真实 seed-0 FIXED1 曲线，25 epoch，best Valid AP 0.275847）。只取相同 epoch 前缀，绝不用 25-epoch 最佳值替代前缀值。

本文件在任何正式训练开始之前冻结；训练开始后公式、horizon 规则、判读阈值不得修改。

## 1. 假设

FIXED1 在两个 block 内对中心状态 `z` 与邻域聚合 `m` 使用同一个线性变换 `fc1(z+m)`。解除这一参数绑定（给 `z-m` 一条独立低秩通道）能否改善标准图级预测，同时保持固定来源、明确结构依赖与完整原始预测计算？

## 2. 共享模型（逐字继承 FIXED1）

```
h0 = base.drop(base.local(batch, a, b, scal))
z1 = LN1(h0);  m1[i] = Σ_j SiLU(z1[j] + e_ji);  h1 = h0 + block1(z1, m1)
z_target = LN2(h1);  z_source = LN2(h0)
m2[i] = Σ_j SiLU(z_source[j] + e_ji);  h2 = h1 + block2(z_target, m2)
logits = base.post_local(h2, batch)
```

atom/bond encoder、S1/S2、dropout、`f`、`rho`、四头 attention readout 全部不改；不增加预测头；不把 `h1[src]`/`h2[src]` 作为邻居来源。

## 3. 变化：每 block 一条独立低秩分支

```
B = Linear(128, 32, bias=False)      # 继承 fan-in 均匀初始化 U(±1/√128)
A = Linear(32, 256, bias=False)      # 全零初始化
a = fc1(z + m) + A(B(x)),  out = fc2(SiLU(a))
x = z + m   (TIED)     x = z − m   (RSSC)
```

- 两个 block 都加上式，rank 固定 32，不搜索；无额外激活/门控/注意力/归一化/头。
- 新增参数 `2 × (128×32 + 32×256) = 24,576`；预期总量 `1,128,468 + 24,576 = 1,153,044`（以真实清点为准）。
- `A`/`B` 由独立私有 RNG（seed 20261126）初始化，不推进主干与 dropout RNG；两臂初始参数逐位一致。
- 继承的 W2 主干、bond_proj、两个 `fc1/fc2`、LN 初始化沿用历史规则（私有 generator 17，`fc2` 置零）。所有模型从头训练，不加载历史训练权重。

## 4. 训练（继承 40-epoch scratch recipe）

batch 128；AdamW `beta=(0.9,0.999)`、`eps=1e-8`、wd 1e-3；dropout 0.10；clip 5；FP32；单一 LR group；masked observed-entry BCE；首 epoch 线性 warmup `3e-5→3e-4`，之后按**历史 40 epoch** per-step cosine 到 1e-5（horizon 只取前缀，**不压缩 cosine**）。seed-0 冻结 order plan（`global_plan_sha256 = 06f437…b6f6`），两臂更新数与 batch 序列完全相同。

Horizon：正式训练前先在真实 batch assembly + GPU step + 全量 Validation 上测成本；预计两臂 12 epoch 在预算内则取 12，否则统一降为 8；若仍无法成对完成则停止，不购买单边结果。

评价：每 epoch 全量官方 Valid AP/BCE（eval mode）；在 epoch 4/8/12 中实际经过的节点做全量 Train AP/BCE，并在结束时评价 best checkpoint 的 Train 指标。**不评价 Test。**

checkpoint：共同 horizon 内最高官方 Valid macro AP 为主 checkpoint，平局取更早 epoch；保存 best 与 last。

## 5. 预算

墙钟目标 ~60 min，硬上限 90 min；GPU-hour 软上限 1.6、硬上限 2.0；仅 seed 0；无多 seed、蒸馏、集成、预训练、辅助监督；不抢占他人进程；不自动启动下一轮。

## 6. 正确性检查（训练前）

B0 数据身份（复用历史审计值逐项比对）；B1 参数量；B2 两臂初值逐位一致且继承部分等于历史 FIXED1；B3 `A=0` 精确恢复 FIXED1 初始前向（合成图 + 真实 batch）；B4 非零 `A` 时 `z+m` 相同/`z−m` 不同 → TIED 输出相同、RSSC 可不同；B5 RSSC 双形式数值等价 `W(z+m)+D(z−m) = (W+D)z + (W−D)m`（bias 只加一次）；B6 第二层邻域只读 `LN2(h0)`；B7 原子重编号、边序、键方向反转不变性；B8 A/B 梯度可达（首步 B 允许为零）；B9 空邻域；B10 不读取 Test。

## 7. 报告与比较

主要比较：`RSSC − TIED`、`RSSC − FIXED1(同 horizon)`、`TIED − FIXED1(同 horizon)`，使用 best AP、horizon endpoint AP、最后三个 epoch 平均 AP（历史值取自真实同 epoch 前缀）。若历史前缀曲线不可靠则标记 `REFERENCE_MISSING`，只报告 RSSC−TIED。

任务级：全部官方可评分任务、Valid 正负例均 ≥20 的高支持任务、改善/退化计数、task bootstrap 2000 次区间、去除最大绝对差三个任务后的均值。bootstrap 只表示当前固定训练结果的任务不确定性，不是 seed 方差。

判读（阈值仅作短程探索参考，训练前冻结）：

- `ROLE_SEPARATION_PROMISING`：RSSC 相对 TIED **且** 相对同 horizon FIXED1 均 ≥ +0.003 AP；高支持任务平均差不低于 −0.001；最后三个 epoch 平均 AP 不低于 TIED − 0.001。
- `GENERIC_PARAMETERIZATION_SIGNAL`：TIED 与 RSSC 均 ≥ +0.003 于 FIXED1，但 RSSC−TIED < +0.003。
- `ROLE_DIFFERENCE_WITHOUT_PERFORMANCE_ADVANCE`：RSSC−TIED ≥ +0.003 但 RSSC 未超过 FIXED1（< +0.003）。
- 其他：`NO_CLEAR_ROLE_SEPARATION_SIGNAL`。
- 任务未完整结束 / horizon 不一致：`BUDGET_LIMITED` / `HORIZON_MISMATCH`；历史同 horizon 前缀缺失：`REFERENCE_MISSING`。

## 8. 机制诊断（轻量）

对两个 block 记录 `D = A.weight @ B.weight`、`R = ||D||_F / ||W||_F`；真实 Valid batch 上 `fc1(z+m)` 与低秩修正项的 RMS、二者比例、`z`/`m` 的 RMS、A/B 梯度；RSSC best checkpoint 上做一次冻结前向（新增分支置零）重算 Valid AP 与 logits 变化；最多 3 个真实 Valid 分子的中心/邻域项非线性前分解与重构误差。不做结构词提取、化学解释或规则搜索。

## 9. 交付物

`code/`（实现与检查）、本文件、`analysis/{checks,preflight_timing,main_table,epoch_curves,paired_comparisons,mechanism,summary}.json|csv`、`results/manifest.json`、`REPORT.md`。大体积 checkpoint 留在 `runs/` 并记录路径 + SHA256。
