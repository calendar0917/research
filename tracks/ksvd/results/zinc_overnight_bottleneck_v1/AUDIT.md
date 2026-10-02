# AUDIT.md — ZINC overnight bottleneck v1, Stage A (定点审计)

日期 2026-10-02/03 · 只读 + 少量 CPU 量级测量 · **官方 test 从未实例化**
· 证据：`audit_stageA.json`、`audit_stageA.py`、`frozen_node_scale_seed0.json`、
`zinc_topology_channel_localization_v1/`、`zinc_small_full_cycle_alignment_v1/`。

## 1. 重放与误差预算（A1）

最新 product control soup 重放（同一 checkpoint、同一 train-median 标定）：

| 项 | 重放值 | 发布值 |
|---|---:|---:|
| raw valid MAE | 0.1102208197 | 0.1102208181 |
| calibrated valid MAE | 0.1090421230 | 0.1090421242 |
| train-fit bias | −0.01920154 | −0.01920073 |

差异 ≤ 2e-8，身份通过（`REPLAY_OK`）。0.09 缺口 = **0.019042**。逐组贡献
（`pair_valid_predictions.csv` + `valid_per_graph_aligned.csv`，组标签来源：
`label_effective_cycle_snapped`，与 pair CSV 的 group 列 1000/1000 完全一致）：

| 组 | n | cal MAE | 贡献 | 占总误差 |
|---|---:|---:|---:|---:|
| G0（无环惩罚） | 965 | 0.087976 | **0.084897** | 77.9% |
| G1（负惩罚，非 172） | 34 | 0.210942 | 0.007172 | 6.6% |
| G172 | 1 | 16.973145 | 0.016973 | 15.6% |
| 合计 | 1000 | 0.109042 | 0.109042 | 100% |

**缺口分布**：单行 id172 贡献 0.016973；即使把该行误差降到 0，总分仅到
0.092069（仍 >0.09）。把 top-5 行误差清零才到 0.088431。G0 中位数 |err|
0.0568，302 行 >0.1，550 行 >0.05。**G0 是唯一有足够算术空间的组**
（0.084897）；G1 整组算术上限 0.007172 < 缺口；id172 单行算术上限
0.016973 < 缺口。算术空间 ≠ 可恢复收益，不能删行。

固定目标分位箱（|err| 分位）：p50 0.0568、p75 0.1165、p90 0.2083、
p95 0.2938、p99 0.5233。top 误差行（cal）：id172 (err 16.97)、29 (1.23)、
753 (0.92)、946 (0.78)、775 (0.72)、375 (0.72)、917 (0.67)。

## 2. 节点链路量级（A2）

canonical fresh init（sha `93c2f23c…`），1024 固定 train 图、23,740 atom
occurrence。单精度前向逐项记录（`audit_stageA.json`）：

| 量 | RMS | 说明 |
|---|---:|---:|
| occurrence 结构投影 `s` | 2.655e-2 | |
| occurrence 原子投影 `a` | 5.977e-2 | |
| 乘法联合 `u0` | 1.589e-4 | 与 slot 同量级 |
| node slot（entry RMS） | 2.469e-4 | 3-shell 聚合后 |
| edge slot | 1.119e-3 | node slot 的 4.5× |
| Sem108 interface | 0.908 | |
| node_out | 0.07256 | 主要来自 node_encoder bias |
| edge_out | 0.07754 | |
| fusion 输入 | 0.456 | |

`node_encoder` 首层：`W1` 范数 4.64，`b1` RMS 0.0486；**pre-activation 信号
RMS 1.365e-4，bias RMS 0.0486，信噪比 ≈ 2.8e-3**（bias 是信号的 ~356×）。
`node_out` 逐列跨样本 std max = 5.30e-5（相对 node_out RMS 0.0726 约
0.07%）。fusion 首层 node 列范数均值 0.327，`|F1b + F_node·mean(node_out)|`
max 0.0916 vs `|F1b|` max 0.0472：node 的**均值**可折进 fusion bias，但跨样本
变化极小。Sem108 interface RMS 0.908 主导 fusion 预激活（逐列 std 0.259）。

**事实**：node 分支在 fresh init 就是 bias 主导（信号/bias ~2.8e-3），不是
训练中期才退化。WD 只作用 W_A_S/W_A_C/node_encoder.*（含 bias）；其余参数
原 w=1e-5。node_out 在 ep10 仍非零、ep40 逐列常量（前轮记录）。

## 3. 竞争解释的证据边界（A2 出口）

| 解释 | 支持证据 | 反对/缺失 | 判定 |
|---|---|---|---|
| A 初始幅值过小（信号被 bias 淹没） | init 信噪比 2.8e-3；固定幅值 residual 只延后未阻止死亡 | residual 那轮未改变初始幅值（系数匹配到 2.4e-3，非单位化），所以没真正测试"幅值" | **强候选，由 2×2 的 N1/N3 直接检验** |
| B 指定 node 参数 Adam WD 收缩 | W_A_S 范数 ep1→ep10 掉 ~100×；原 Adam 每步 wd·p | 未做影子单步；下游常量吸收可独立杀死通道 | **强候选，由 N2/N3 检验** |
| C 下游 fusion 常量吸收 | node 均值可折进 fusion bias；node_out 先于 slot 变常量 | 常量吸收是后果还是原因未定 | **竞争解释** |
| D FP32 相对 bias 分辨率 | init 信号 1.4e-4 ≫ FP32 eps·0.05 ≈ 6e-9 | 不支持作为主导 | 弱 |
| E mask/索引/shared tensor 实现错误 | 身份检查通过；`node_binding_zero` 正确清零；graph_id/split hash 一致 | 未发现 | 未发现 |

**不能提前给唯一根因**：A 与 B 由 2×2 正交检验，C 是二者都可能触发的下游
机制。Stage A 只冻结这些假设与阈值，不做因果归因。

## 4. 拓扑误差的竞争解释（A3，冻结事实）

`zinc_topology_channel_localization_v1`（只读重放，`REPLAY_OK`）：
topo25 对 35 个环图**基本可区分**（34/35 至少一个 top-5 train 近邻惩罚 <0，
top-1 距离中位数 0.0）→ 输入统计歧义不是解释。encoder 无死单元，环图上
reader 对拓扑块的局部灵敏度约为 G0 的 10×。**决定性事实**：`train:3776` 与
`valid:0172` 的 topo25 **逐位相同**、标签惩罚同为 −6，预测却差 21.1；
换共同 T0 仍差约 18 → id172 的失败**不可能只由拓扑通道解释**，必然涉及
其余 R 块（unary/pair/global）或 train/valid 分布差。

T25 精确匹配只限制"该统计块"不可学习，不等于全输入不可学习；低维近邻不
等于原图语义近邻。**因此 T0（把显式 T8 变成独立加性项）不假设它能修复
id172**；它检验的是"显式拓扑贡献不被其他坐标门控后是否更稳地泛化到严重
环样本、且不伤 G0"。

## 5. 对科学比较的影响

- 所有 seed-0 arm 共享 canonical init `93c2f23c…`；`kappa` 冻结在
  `frozen_node_scale_seed0.json`（per-entry 规则，`r_init=2.469e-4`，
  `kappa=4050.24`；远程 CPU 重算 rel diff 3.9e-8，runner 取冻结值）。
- N0 用原幅值/原 WD，与历史 product control 同协议同数据；N0 的 240 结果
  同时给本轮夜间 baseline（与历史 0.109042 并列报告，不作强因果比较）。
- 未做全模型 float64 重算、未写回真实模型、未改变 IHT 选择；未访问 test。
