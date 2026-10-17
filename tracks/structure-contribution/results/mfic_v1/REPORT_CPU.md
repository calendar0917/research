# MFIC CPU 科学购买门槛报告

协议：`cscl-bjoc-mfic-v1`（split V，salt `cscl-bjoc-mfic-v1`）
日期：2026-10-17
状态：**CPU 门槛通过，允许进入冻结后的 GPU Stage B**；本文件只记录 CPU 阶段，GPU 结果见 `REPORT.md`。

## 1. 问题与设计

MFIC（Matrix-Function Incidence Composition）测试的是一种**非迭代、非 Attention、解析**的图级结构组合：

1. 由分子对象构造物理重原子邻接矩阵 `A`（键级只用于 `R` 的边权，不进入读写器）；
2. `S = D^{-1/2} A D^{-1/2}`，固定 `alpha = 0.65`；
3. `R = solve(I - alpha*S, I)`，`K = R - I - alpha*S = sum_{k>=2} alpha^k S^k`；
4. 每原子编码 `h_i`（96 维）经 `q_i = Linear(96->16)(h_i)`；
5. 图级二次型 `C = Q^T K Q / max(1,n)`，取 136 个上三角特征；
6. Reader 输入从 P3 的 321 维变为
   `[sum_composer(96), sum_atom(96), upper_tri(C)(136), global(33)] = 361` 维。

三臂：P3（413,505 参数）；MFIC-R（真实 `K`，425,297 参数）；MFIC-U（content-null
`K_U = d*I + o*(11^T - I)`，425,297 参数）。三臂共享相同的 atom/bond/junction 编码器、
Null composer、SSL checkpoint、batch order、22,680 次更新、raw-y L1、R3 配方和 last-5 soup。

CPU 门槛只使用 Fit 7,996 行的 raw y；不读 Monitor/Locked 标签，不读 official valid/test，
不使用任何模型预测。CPU 探针的目的是回答：**在固定 16 维 q 投影和固定 Ridge alpha=1 下，
真实 `K` 的二次型是否比 Base 结构特征提供至少 0.020 的廉价 MAE 信号？**

## 2. Split V

- official train 10k 中排除已知输入不一致行 `[941, 2210, 6071, 9766]`，可用 9,996；
- Fit 7,996 / Monitor 1,000 / Locked 1,000；
- identity = PyG 重建 canonical SMILES；3 个重复身份组，所有重复组都在同一侧；
- split 哈希：`b8adc1b8271f15fc0162258184470613cf4d7fb66d882e6076d259bf2cd0cc90`；
- zone 行哈希：
  - fit `dc786d44bb58f887003fc1ed1d19d4b0d5b16e0531a5b8f09de72f47ecb9f953`
  - monitor `a86cb5a65951e9599e5952fbbb5f5150a2208c484fa4bef76b10c10bc227e0a1`
  - locked `d1576cbdcad8c1881cf473ba647d80726efaf238108c1d6e15d175122f793606`
- 与历史 E/F/S/T/U 无 zone-hash 碰撞、无行完全相同；但这是同一 official-train 10k
  暴露池的重新抽样（`historical_rows_seen_fraction = 1.0`），**不是独立盲测**。
  新 Fit 与 U Fit 重叠 6,421 行（80.3%），与 E/F 各 6,401、S 6,390、T 6,394。

## 3. CPU-Matrix 探针

- 4 折 identity-grouped 分组交叉验证：`sha256(salt|fold|identity)`，贪心最小折；
- 每折仅用训练折原子拟合 16 维 PCA；验证折原子只做 transform；
- CPU-Base：68 维结构特征（28 atom type、4 bond type、global 33、3 raw size）；
- CPU-Matrix：68 + 136 = 204 维，136 维为 `upper_tri(Q^T K Q / max(1,n))`，其中
  `Q` 是训练折 PCA 的 16 维原子投影，**与 MFIC 模型中的 16 维 q 投影同维**；
- 两个臂都用训练折标准化、固定 Ridge `alpha=1`，无超参搜索；
- 不使用模型预测、official valid/test 或 Monitor/Locked 标签。

### 结果

| fold | Base MAE | Matrix MAE | Δ (Matrix-Base) |
|---:|---:|---:|---:|
| 0 | 0.497883 | 0.389163 | **-0.108719** |
| 1 | 0.515845 | 0.402366 | **-0.113480** |
| 2 | 0.478327 | 0.397085 | **-0.081242** |
| 3 | 0.492972 | 0.393073 | **-0.099899** |
| **mean** | **0.496257** | **0.395422** | **-0.100835** |

- 4/4 折同向改善；方向 `['improved','improved','improved','improved']`；
- trimmed-10 平均 Δ = **-0.092055**；
- ordinary（y > -3）平均 Δ = **-0.090967**；
- tail（y <= -3）平均 Δ = **-0.228014**；tail_n 每折 123–147；
- 平均 60.7% 的分子变好；
- 16 维 PCA 平均解释率约 **94.3%**；
- `mean_mae_improvement = 0.100835`，是门槛 0.020 的约 5.0 倍。

需要明确：这是 **fixed-PCA-q + Ridge** 的廉价探针，不是 MFIC 神经模型的预测结果。它证明
真实 `K` 的二次型在 16 维投影下含有 Base 之外的结构信号；它不保证监督训练一定能把该信号
转化为模型收益。CPU 通过只购买 GPU 实验，不购买结论。

## 4. 数学与模型审计

### 4.1 数学门（全部 PASS）

- 单原子 / 单键 / 链 5 / 环 6 / 萘 / 双环 / 链+孤立点：`K` 有限、对称；
- 200 个真实 Fit 分子：`R` 与 `K` 有限、对称，`K != f(n)`；
- `K_U` 在非退化图上与 `K` 不同；
- 单原子 `K = 0`；
- walk-count witness：`K_{3,3}` 与三棱柱在每阶 walk count 相同，但 `K` 可区分；
- 全图 `sum(R)` 不是常数，且**未作为特征**；
- 所有检查见 `math_gates.json`。

### 4.2 模型审计（20/20 PASS）

- P3 = 413,505；MFIC = 425,297；MFIC reader = 158,721；q_projection = 1,552；
- `reader_in = 361`，`C` 为 16x16；
- 三臂 shared modules bitwise equal；MFIC-R/U full init equal；
- `K` 不在 `state_dict`；`K` 只改变 `C` 与 reader 的 C 输入块，隐藏状态其余部分 bitwise 相同；
- 真实 `K` vs `K_U`、`K=0` 都改变预测；
- q_proj、所有参数、reader 首层 136 个 C 通道都有非零梯度，无 dead channel；
- 原子重标号 / 键端点交换 / batch 置换不变性通过；
- 无 GNN / attention / message passing / 迭代模块；
- 见 `model_audit.json`。

## 5. 冻结的 CPU 决策

```json
{
  "decision": "MATRIX_FEATURE_CHEAP_SIGNAL",
  "passed": true,
  "conditions": {
    "C1_mean_mae_improvement_at_least_0.020": true,
    "C2_at_least_3_of_4_folds_negative": true,
    "C3_trimmed10_mean_direction_negative": true,
    "C4_ordinary_mean_direction_negative": true,
    "C5_math_model_and_invariance_gates": true,
    "C6_no_official_or_model_prediction_dependence": true
  }
}
```

因此允许进入 GPU Stage B；但必须保持以下限制：

- 不因 CPU 信号强而放宽 GPU Monitor 门槛；
- 不搜索 `alpha`、PCA 维数、kernel rank、Ridge alpha、Reader 宽度；
- 不把 CPU 探针称为模型收益；
- 不把同一 10k 暴露池称为独立盲测；
- 不引用 Batatia et al. ICLR 2024 的 ZINC 0.063 作为本模型结果（其模型含 GCN 层）。

## 6. 复现

```bash
uv run python tracks/structure-contribution/code/run_mfic_v1.py split
uv run python tracks/structure-contribution/code/run_mfic_v1.py verify-inputs
uv run python tracks/structure-contribution/code/run_mfic_v1.py objects
uv run python tracks/structure-contribution/code/run_mfic_v1.py cpu-signal
uv run python tracks/structure-contribution/code/run_mfic_v1.py math-gates
uv run python tracks/structure-contribution/code/run_mfic_v1.py model-audit
uv run python tracks/structure-contribution/code/run_mfic_v1.py cpu-gate
uv run python tracks/structure-contribution/code/run_mfic_v1.py freeze-pre-gpu
uv run python tracks/structure-contribution/code/run_mfic_v1.py finalize-cpu
```

关键文件：`cpu_signal.json`、`cpu_folds_V.json`、`cpu_gate.json`、`decision_cpu.json`、
`math_gates.json`、`model_audit.json`、`objects_manifest_V.json`、`protocol_freeze.json`。
