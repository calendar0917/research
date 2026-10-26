# PROTOCOL — `zinc_bosi_joint_moment_v1`（GPU 之前冻结）

**协议 ID：** `zinc-bosi-joint-moment-v1`
**设计：** 同目录 `DESIGN.md`（在本文件之前提交，不改写）
**CPU gate：** `analysis/cpu_checks.json` — **12/12 PASS → `CPU_GO`**
**冻结时刻：** 见 `configs/bjm_v1.json` 的 `frozen_utc`；本文件在任何正式 GPU 训练之前提交。

**边界：** official **test 全程不构造、不加载、不预测、不评分**。
official valid **仅用于训练期间的模型选择与开发评估**：不进入特征、不进入归一化、
不进入预训练目标、不用于任何结构统计。

---

## 1. 问句

> 把 BOSI 统计块里被丢弃的**跨通道配对**显式做出来——对同一伙伴的两个 12 维学到的
> 投影先外积再求和（`J`）——能否在官方 ZINC-subset 10k/1k 协议上把 BOSI-B 的
> valid MAE 压下去，并靠近 DCC-L？这个收益是否来自"先配对后求和"本身
> （由等参数、同初值的 `CM` 隔离）？

## 2. 数据角色（官方 split，不新建划分）

| 角色 | 行数 | 来源 | 用途 |
|---|---:|---|---|
| official train | 10,000 | `zinc_arch_discovery_v1/cache/ad_train.pt`（sha256 `d85d9875…`）| 监督训练 + global-33 标准化（`norm.json`，`n_fit=10000`，`valid_or_test_used=false`）|
| official valid | 1,000 | `…/cache/ad_valid.pt`（sha256 `6e45fea6…`）| 每 epoch 完整评估 + best checkpoint 选择 + 开发评估 |
| official test | — | **不加载** | — |

结构缓存（label-free，`bosi_data.relation_arrays`）：`cache/bjm_rel_train.pt`
（content sha256 `c698a2f3…`，与 `zinc_bosi_architecture_v0` 逐字节内容一致）、
`cache/bjm_rel_valid.pt`（content sha256 `6505e23b…`）。
两者在 `--rebuild` 下重算内容哈希相同（`G12`）。

**SSL 初始化：** 只从 `bjoc_full10k_pretraining_transfer_v1/pretrain_encoder_s{seed}.pt`
加载 `atom_encoder` 与 `bond_encoder`（sha256 s0 `006fabc1…` / s1 …；两份 checkpoints
的顶层 keys 只有 `atom_encoder`/`bond_encoder`/`junction_encoder`，**没有任何
`side`/`compose`/`reader` 监督权重**，`G10`）。不加载 DCC-L 监督权重，不做额外 SSL。

## 3. 模型（`code/bjm_models.py`，`named_parameters()` 实测）

| 模块 | 形状 | 参数 |
|---|---|---:|
| atom encoder | 108→128→96 | 26,336 |
| bond encoder | 295→160→96 | 62,816 |
| half-edge encoder（tanh） | 295→128→32 | 42,016 |
| side | 617→128→96 | 91,488 |
| bond composer | 391→160→96 | 78,176 |
| graph reader | 417→256→256→1 | 173,057 |
| **BOSI-B 合计** | | **473,889** |
| joint projections `U1,V1,U2,V2` | 32→12，无 bias | 1,536 |
| **BOSI-JM / BOSI-CM 合计** | | **475,425** |

`side` 输入 617 = `a_v(96) + b_e(96) + x(32) + M(384) + n(4) + closure(4) + log1p(deg)(1)`；
`M` 的宽度 B/JM/CM 都是 384，所以 **Side 的输入形状与输出形状不变**。

* `JM` 的 `M = [A(96), vec(J_1)(144), vec(J_2)(144)]`；
* `CM` 的 `M = [A(96), vec(C_1)(144), vec(C_2)(144)]`，`C_j = outer(Σu, Σw)/max(n,1)`；
* `A = Σ_f [T1(x_f), T2(x_f), T3(x_f)]` 两臂逐位相同（`G6.A_block_shared = true`）；
* 四个投影固定使用 `nn.Linear` 默认初始化（Kaiming-uniform，`a=√5`，无 bias），
  在六个公共 MLP 之后按 `U1,V1,U2,V2` 顺序消耗同一 RNG 流；
* `LayerNorm` 无 affine、`eps=1e-5`；
* `n=0` 时 `A` 与 `J`/`C` 严格为零；`n=1` 时 `J == C`；
* `J - C = Σ_f (u_f - ū)(w_f - w̄)^T`（`G4`，float64 参考，max|Δ| 3.2e-7）。

**禁止清单（逐项确认）：** 无 learned junction token、无迭代状态更新、无 attention、
无额外监督任务、无 global correction branch、reader 宽度/深度/参数不变、
无递归消息传递、无外部标签/额外分子/标签公式分量/蒸馏/ensemble/测试集选择。

## 4. 配方（三条轨迹完全一致）

AdamW，lr 1e-3 → 1e-5（**per-step** cosine），warmup 500 updates，wd 1e-5，
dropout 0，grad clip 5，raw-y L1，batch 128，eval batch 256，fp32，无 AMP。

* official train 10,000 → `ceil(10000/128) = 79` steps/epoch，**600 epochs = 47,400 updates**
  （不可缩短）；
* batch 顺序：`torch.Generator().manual_seed(seed + 101)` → 每 epoch `randperm(10000)`；
* 每个 epoch 完整 official valid（1,000 行）；
* best checkpoint = **严格最小 valid MAE**，平局保留更早 epoch；
* 标准化：`norm.json` 的 global-33（只用官方 train 10k 计算，`G10` 逐坐标重算核对）；
* 同 seed：全 arm 共享同一初始状态流（公共参数逐位相同，`G2`）、同一行顺序、
  同一 `first_batch_rows`。

记录：`final`、`W20`（20 连续 epoch 窗口内 valid 中位数的最小值）、
`T100`（最后 100 epoch 的 valid 中位数）、`train@best` / `train@final`
（固定 checkpoint 在完整 10k train 上的 eval-mode MAE）、每 epoch 曲线、
随机状态（generator state / torch RNG state 的 sha256）、split/行号 hash、
initial state hash、first batch 行号、每个 checkpoint 与预测的 sha256。

## 5. CPU gate（`analysis/cpu_checks.json`，12/12 PASS）

| gate | 内容 | 关键数字 |
|---|---|---|
| G1 | 参数量 / Side 输入输出 / M 宽度 / ≤500k | B 473,889；JM=CM 475,425（+1,536）；Side 617→96；M 384 |
| G2 | JM/CM 完整初始 state 逐张量相同，公共参数 = B 臂 | 逐张量 max\|Δ\| **0.0**；三者 common state sha `f894c18a…` |
| G3 | `n≤1` 时 JM ≡ CM | 合成路径图（deg≤2）与真实 valid 上 `n≤1` 行 max\|Δ\| **0.0** |
| G4 | `n≥2`：`J-C = Σ(u-ū)(w-w̄)^T` | 1,423 行，float64 参考 max\|Δ\| **3.24e-7** |
| G5 | 二维配对反例 + 真实分子边际保持重配对 | 反例：JM 分开 1.0 / 4.0，CM 分开 **0.0**；真实 1,082 个半键重配对后 JM 移动 1.23e-3（100% 分子），CM 8.9e-8（float32 加法顺序） |
| G6 | 真实 train 10k 统计（label-free） | n=0/1/2/3 = 8.25/47.05/42.21/2.49%；`n≥2` 占 **44.70%**；JM−CM 输入最大相对差 **0.427**、`n≥2` 上 97.0% 改变；`Σn = deg-1` 全成立 |
| G7 | 结构不变性 | 原子重编号 / 键向反转 / 键序 / batch 序 / batch 重组 ≤ 1e-5 |
| G8 | 梯度 | 全有限；所有预期模块梯度非零；JM/CM 四个投影梯度非零 |
| G9 | batch 完整性 | 无跨分子 partner、无重复 partner、`Σ_r n[r] = deg-1` |
| G10 | 来源审计 | train/valid cache split、norm 由 train 10k 重算（mean 1.9e-7 / scale 2.7e-8）、SSL 只含 encoder、代码中无 test 访问、valid 标签只用于 eval/hash |
| G11 | checkpoint 保存/加载 | 三个 arm 预测逐位相同 |
| G12 | 完整 train-only 缓存前向 | B/JM/CM 全 10k 前向有限；两 split 内容哈希重建一致 |

判定：**`CPU_GO`**。未发现参数超标、初始化分叉、机制退化为零、索引跨分子、
数据来源泄漏或缓存不可复现。

**如果 `n≥2` 上的 JM−CM 输入差异几乎为零，本轮必须停止并报告，不得用放大系数或
增加维度救援。** 实测相对差 0.427、覆盖率 44.70%，该停止条件未触发。

## 6. GPU 执行（`res`，`rr` 管理）

`rr doctor res` 先做体检；两个 pool `res-gpu0` / `res-gpu1`。
本地 commit → `rr deploy res` → 冒烟 + 实测计时 → 正式跑。

顺序与预算（硬上限 **5 A100·h**，含 smoke/计时/失败作业）：

1. `rr run res bjm-smoke-{B,JM,CM} --pool res-gpu0/1 -- 2 epochs`（每条约 5 s，仅验证路径）；
2. `rr run res bjm-timing-* ... --max-epochs 3` 实测 s/epoch，确认 47,400 updates 可跑；
3. **两条正式 run 并行：** `res-gpu0` = BOSI-B seed0，`res-gpu1` = BOSI-JM seed0，
   **同一 commit**，`--time` 给足，不缩短更新数；
4. seed0 判决后才决定是否买 seed1 / 跑 CM。

## 7. seed0 判决（预注册，看完 seed0 不得改设计）

先 `bjm_freeze` 冻结两个 checkpoint、valid 预测与 hash，**再**比较。

```
A. JM best_valid <= 0.075
或
B. (JM best_valid - B best_valid) <= -0.003  且  JM best_valid <= 0.078
```

并且 ordinary（`y > -3`）不出现明显恶化、W20 不出现反方向的大幅退化。

* 满足 → 买 seed1：`res-gpu0` = B seed1，`res-gpu1` = JM seed1（同 commit、同配方）；
* 不满足 → **停在 seed0**：不买 seed、不扫 rank/宽度/LR/wd/horizon、不改选择策略。

## 8. 机制对照（有条件）

只有在 **JM 相对 B 的性能优势在两个 seed 上可信**之后，才以**新作业**跑
`BOSI-CM seed0`（预算允许且 JM−CM 差异同样有希望时再跑 `CM seed1`）。
`CM` 与 `JM` 共用一切，唯一差别是 `J → C`。`JM − CM` 是"联合配对"的净效应。

## 9. 分析（`analysis/summary.json` + `REPORT.md`）

主表每个 arm×seed：`params`、`GPU-hours`、`best_valid`、`best_epoch`、`W20`、
`T100`、`final_valid`、`train@best`、`train@final`。

对比（配对、固定分子重采样的 10,000 次 bootstrap，只描述分子抽样不确定性）：

* `B vs JM` 的逐分子 AE 配对差；
* `JM vs CM`（若执行）；
* 分组：`ordinary`（y>-3）、`tail`（y≤-3）、`seen/unseen radius-2`（support 只用
  **official train 10k** 的精确 typed radius-2 rooted signature 统计）、`ring-rich`
  （cycle rank ≥ 4）、去 top-10 后的敏感性；
* 两 seed 方向是否一致；`train−valid` gap 与随 epoch 的趋势；
* 相对 DCC-L（0.075990 四 seed 均值 / 0.078537 seed0）与 P3B（0.076224 seed0）
  的差距。

**边界：** 官方 valid 在本 track 与历史上已被反复读取，**不是新的盲测**；
CI 只描述固定分子上的抽样不确定性，不作为 seed 复现证据。
**official test 不读。**

## 10. 禁止清单

official test 访问；在 valid 上做架构/超参/epoch 扫描（valid 只按 §4 的严格最小规则
选 checkpoint）；新的随机划分；额外分子/外部标签/新标签公式分量；知识蒸馏；
ensemble / seed 平均 / checkpoint soup / TTA；递归消息传递；learned junction token；
attention；加宽 reader；加 rank、加阶数、放大系数来救援失败；缩短 47,400 updates；
重训 DCC-L 或 P3B。
