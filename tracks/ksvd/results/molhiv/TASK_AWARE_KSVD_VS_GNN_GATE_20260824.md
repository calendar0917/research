# Task-aware KSVD vs strong GNN gate (2026-08-24)

## 目的

这一轮回答两个问题：

1. 当前字典训练/编码是否仍然是无监督的；
2. task-aware patch route 是否已经超过同一数据切分上的强 GNN。

## 训练目标的归因

### Ordinary KSVD baseline

普通 KSVD 的字典 `D` 只通过 patch reconstruction objective 学习：

\[
\min_{D,Z}\|Y-DZ\|_F^2,
\]

并用 OMP 在每个 patch 上求稀疏码 `z`。图标签没有进入字典拟合，也没有进入 OMP。因此它是：

> **无监督字典学习 + 无监督稀疏编码**。

### New task-aware branch

新 pilot 从 ordinary KSVD 初始化 `D`，然后用 soft-threshold code 和图级 readout 联合优化：

```text
graph BCE + 0.2 * patch reconstruction + 0.05 * atom incoherence
```

图标签通过 BCE 反向传播更新字典、阈值和 readout。因此它不是无监督 KSVD，而是：

> **KSVD 初始化的监督 task-aware dictionary/projection learning**。

这里的“稀疏”只来自 soft-threshold；它是否优于 dense projection，需要单独做 matched control，不能从 task-aware 分数本身推出。

## 同一 8k scaffold protocol 的结果

数据与 split 完全沿用：`molhiv_n8000_scaffold_folds3_seed20260726.npz`。task-aware 结果来自同一 patch bank、同一三折；GINE 为 3-layer original-node GINE、hidden 64、30 fixed epochs、seed 0。

| 方法 | fold 0 | fold 1 | fold 2 | mean | sample std |
|---|---:|---:|---:|---:|---:|
| raw patch max | 0.6523 | 0.5516 | 0.6307 | 0.6115 | 0.0530 |
| frozen ordinary KSVD + OMP | 0.6247 | 0.5629 | 0.5862 | 0.5913 | 0.0312 |
| task-aware sparse dictionary | 0.6338 | 0.6200 | **0.6991** | **0.6510** | 0.0423 |
| matched dense task-aware projection | 0.6504 | **0.6324** | 0.6734 | **0.6521** | 0.0206 |
| original-node GINE (strong near-baseline) | **0.7269** | **0.6324** | 0.7215 | **0.6936** | 0.0531 |

差值：

- sparse task-aware − ordinary KSVD: `+0.0597` mean，3/3 folds；
- dense task-aware − ordinary KSVD: `+0.0608` mean，3/3 folds；
- sparse task-aware − dense task-aware: `-0.0011` mean；
- sparse task-aware − GINE: `-0.0426` mean。

## 结论

### 已经成立

1. 普通 reconstruction-only KSVD 在这条 patch-MIL 管线中不是强分类器；它更准确的定位是无监督压缩/基底学习器。
2. 让标签参与字典和编码优化，可以恢复普通 KSVD 丢掉的任务信号；在三折上稳定超过 frozen ordinary KSVD。
3. 当前优势不能归因于“稀疏 KSVD”本身：dense task-aware projection 的均值略高，说明主要增益来自监督表示旋转，而不是 sparse-specific mechanism。

### 尚未成立

1. task-aware patch route 尚未超过同协议的 original-node GINE。
2. 更不能声称已经超过 CIN。已有 full-data internal pilot 的 CIN held-out AUC 为 `0.7684`；已有 official terminal comparison 中 GINE ensemble test 为约 `0.778`，而 KSVD route test 约 `0.770`。
3. 当前 patch route 是 graph-level MIL：patch 内使用了 atom/bond chemical features，但没有保留 node-level states，也没有 occurrence/overlap message passing 或 Transformer。

## 下一步 gate

下一步不再把“监督 dictionary”直接称为 KSVD 胜出，而做一个最小且可归因的升级：

1. 保留 ordinary KSVD、task-aware sparse、dense projection 三个 branch；
2. 将 patch readout 从单一 `max |z|` 扩展为预注册的 signed/magnitude moments，并保持同一参数预算；
3. 在三折 scaffold 上与 GINE、CIN-small 做 paired comparison；
4. 只有 sparse 在多个 fold/seed 同时超过 dense control，才把稀疏编码列为核心贡献；否则将 KSVD 表述为无监督初始化/可解释局部基底，task-aware 表述为监督适配层。

若仍低于 CIN/GINE，合理的终局定位是“参数高效、可解释的 local-evidence branch”，而不是替代强 GNN。只有在严格 scaffold split 上出现稳定 paired gain，才进入 official valid/test。

