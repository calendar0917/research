# REPORT — cscl-correctness-v1 seed-0 预测信息实验（O-unit vs O-rich）

日期: 2026-10-10 · Track: `structure-contribution` · round: `cscl-correctness-v1`
目的: **测量预测信息是否充分**（任务 §7），不是证明解释性机制。主指标:
internal dev raw-y MAE（soup；best-epoch 为副表）。

## 1. 主结果（seed 0，`res` GPU1，pool res-gpu1，进程内 cuda:0）

| 臂 | soup dev MAE | best-epoch dev | fit MAE | monitor best | params | input | epochs | wall |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| O-unit（修复后单元表示 + OpaqueModel 头） | **0.35708** | 0.36393 | 0.26291 | — | 36 289 | 类型 embedding+描述子+关系（761+1 类型） | 104 | 59 s |
| O-rich（base573 静态特征 + 等容量 MLP） | **0.33206** | 0.34345 | 0.25340 | — | 29 953 | 573 维 | 107 | 12 s |

- 统一配置：AdamW lr 1e-3 / wd 1e-5、batch 128、≤300 epochs、monitor
  early-stop patience 30、top-5 soup、raw-y 标准化（fit_inner 统计）。
- 硬件/环境：NVIDIA A100-SXM4-40GB，driver 550.163.01，torch 2.5.1+cu124；
  commit 9eb51d023251；official valid/test 未加载；监督仅 raw y。
- **容量与输入不严格等价**（如实声明）：O-unit 36.3k/单元特征 vs O-rich
  30.0k/573 维全局特征；差异方向对 O-unit 略有利（参数更多），但不构成
  严格控制。

## 2. 配对比较（分子级 bootstrap，2000 次，dev 2000 分子）

- 配对差 **O-unit − O-rich = +0.02502**，bootstrap 95% CI **[+0.00124, +0.05078]**
  （CI 不含 0；方向：O-rich 更好）。
- **声明**：这是同一 seed 内部 dev 划分上的分子重采样区间，**不是**跨 seed
  训练稳定性证据；本轮只允许 seed 0。

## 3. 子群错误分布（dev raw-y MAE）

| 子群 | n | O-unit | O-rich | 差 (unit−rich) |
|---|---:|---:|---:|---:|
| ring_units = 0 | 12 | 0.628 | 0.395 | **+0.234** |
| ring_units = 1 | 233 | 0.591 | 0.364 | **+0.227** |
| ring_units ≥ 2 | 1755 | 0.324 | 0.327 | −0.003 |
| n_atoms [0,15) | 44 | 0.628 | 0.387 | +0.241 |
| n_atoms [15,22) | 659 | 0.393 | 0.322 | +0.071 |
| n_atoms [22,30) | 1129 | 0.310 | 0.324 | −0.014 |
| n_atoms [30,40) | 168 | 0.460 | 0.413 | +0.048 |

**读法**：多环分子（ring_units≥2，占 dev 87.6%）上修复后的单元表示与丰富
静态特征**持平**；全部差距集中在 0–1 个环系的分子——单元划分+关系集合
在这些分子上缺少 rich573 全局拓扑/环境统计所携带的信息。

## 4. 对照历史（只作量级参照，不可混表）

- v0 同族 opaque 头（**缺陷特征**，27k params）：soup dev 0.40005 → 修复后
  O-unit 0.35708（**仅正确性修复 +0.043 提升**；两 run GPU 非确定性 ~0.001）。
- v0 B 臂 0.465 / A 0.425 / XGB(单元特征) 0.477 / C 0.431（全部旧缺陷表示）。
- CPU 筛查（同划分，固定配置）：Ridge α=1 dev — v0 特征 0.5026 / v1 特征
  0.4171 / rich 0.3370；XGB — 0.4771 / 0.4532 / 0.3717
  （`results/cscl_v1_screen/screen.json`）。
- ksvd 轨历史 ~0.12–0.19 为不同协议（辅助监督/不同划分），不与 y-only 同表。

## 5. 判断（按任务 §9 规则）

**情况一成立：O-rich 显著好于 O-unit（配对 CI 不含 0）。**

1. 修复后的单元表示仍然缺少一部分强静态特征携带的预测信息——主要在
   少环/无环分子的全局形状与化学环境统计上。
2. 同时，修复本身带来 +0.043 的实质提升，证明 v0 的部分性能差距确实来自
   **表示实现错误**（H-R 成立），而非加性分解。
3. 因此"把差距归因于贡献分解"仍然不被本轮数据支持。

## 6. 运行记录与已知问题

- 正式 run（rr，process backend）：
  - `cscl-v1-ounit-s0-v2-20261010-210307` commit 9eb51d0（含逐分子导出）
  - `cscl-v1-orich-s0-v2-20261010-210715` commit 9eb51d0
  - 前序：`cscl-v1-ounit-s0`（0.35594，无导出）、`cscl-v1-orich-s0-fixed`
    （0.33206）——与上表数字一致性在 GPU 非确定性范围内。
- **废弃 run**：第一次 `cscl-v1-orich-s0`（soup dev 1309）因 rich 特征
  标准化缺陷（未 clip，max|Rz|=2e9）整体作废，属基础设施错误而非科学
  结果；修复为 fit-only z-score + clip ±10（协议常数）后重跑。
- 导出 npz 初版以标准化单位保存 pred；已用 fit_inner y 统计（与 ounit
  json 逐位核对一致）精确重建原始单位（`dev_preds_raw_units.npz`），重建
  MAE 与 json 逐位吻合；导出代码已改为直接存原始单位。
- 逐分子配对分析: `paired_analysis.json`。

## 7. 不变性与纪律声明

- 原子重标号/单元置换不变性：模型级测试
  （`test_opaque_model_unit_permutation_invariance`、全管线重标号测试）通过。
- official test 禁读；本轮未加载 official valid；监督 = raw y only（json
  字段声明）。
- 本轮未训练新的显式贡献模型（B/C 未重训）；两臂均为非 GNN/Transformer 回归头。
