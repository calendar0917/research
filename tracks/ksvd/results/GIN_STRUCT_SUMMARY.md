# GIN ± structure (paper-optimistic)

Dataset: **MUTAG**

Protocol: `paper-optimistic-gin-struct-v0` — max held-out Acc on fold test (optimistic; Xu/GIN paper habit)

**Role: self-check only. Do not mix with strict-v0.**

HPs: `{'epochs': 80, 'hidden': 64, 'lr': 0.01, 'dropout': 0.5, 'layers': 4}`

| variant | Acc % (mean±std) |
|---------|------------------|
| gin_only | 94.6 ± 2.5 |
| gin_plus_struct | 87.8 ± 4.1 |

## Structure injection

- Shared KSVD (M0, rich readout) on **train fold only**
- Graph vector z-scored on train, **broadcast-concat to every node** as extra channels
- GIN-0, 4 layers, layer-wise sum scores

## Result (this run)

| variant | Acc % |
|---------|-------|
| **gin_only** | **94.6 ± 2.5** |
| **gin_plus_struct** | **87.8 ± 4.1** |

**解读**：在 paper-optimistic 下，**硬拼接 KSVD 结构到节点特征会拖累 GIN**（约 −7pt）。与 LR 诊断一致：结构通道仍偏噪声。  
原因候选：维数 7→115、结构与原子特征尺度/语义未对齐、广播拼接过粗。

**不是** CIN 同实现；只说明「当前注入方式」无益。

## vs literature

- CIN/GIN paper MUTAG 同属乐观 10-fold 习惯，**代码/超参不同**，数字不可横比排名  
- 本表只回答：同一协议下结构是否帮**这个** GIN  

