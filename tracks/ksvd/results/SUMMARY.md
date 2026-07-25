# KSVD pipeline results

Status: **done** · wall ~20.5s · 2026-07-24  
环境：`paper/.venv`（numpy/scipy/sklearn/torch_geometric）  
入口：`python -m code.run_all`（在 `tracks/ksvd`）

---

## 数据用了什么？

| 阶段 | 数据 | 说明 |
|------|------|------|
| 0–1 | 合成 **ring+chords** n=20, e=23 | 只测采样 / KSVD |
| 2a 合成 | **cycle(+噪声弦) vs path**，各 50 张，n=12 | 结构二分类 |
| 2a 真实 | **MUTAG**（TUDataset，188 图） | sklearn 10-fold **mean** Acc |
| 2b | **ogbg-molhiv** | **未跑**（环境无 `ogb` 包） |

---

## Stage 0 — 采样 `stage0-sample-v0`

合成图，p=q=1 摘要（3 seeds）：

| 方法 | edge_cover | edge_count_per_sg | mean \|S\| |
|------|------------|-------------------|------------|
| B0 一阶 | 1.0 | 0.100 | 3.30 |
| B1 均匀 RW | 1.0 | 0.358 | 8.90 |
| M0 +γ | 1.0 | 0.357 | 8.83 |

- 管道可用；cover 满。  
- M0 与 B1 接近（降权弱信号）。  
- **勿**用 raw `edge_repeat` 直接比 B0（子图更小）。

---

## Stage 1 — KSVD `stage1-ksvd-smoke-v0`

| 方法 | recon_rel | atoms_used | mean nnz/列 |
|------|-----------|------------|-------------|
| B0 | ~0 | 8 | 3.0 |
| B1 | 0.27 | 10 | 3.0 |
| M0 | 0.30 | 10 | 3.0 |

- 稀疏度 T=3 生效，**未塌成 1 原子**。  
- B0 列更简单 → 重构极低；RW 子图更复杂 → 残差更高（正常）。

---

## Stage 2a — 合成 cycle vs path `stage2a-synth-ringpath-v0`

sklearn 5-fold mean Acc：

| 方法 | Acc |
|------|-----|
| B0 + KSVD | 0.550 ± 0.084 |
| B1 + KSVD | 0.590 ± 0.162 |
| M0 + KSVD | 0.500 ± 0.100 |
| **degree_stats** | **1.000 ± 0.000** |

**解读**：该任务可用「边数/度」完美分开（环 vs 路），结构字典在此 **无增益**。说明任务设计过易被宏观统计解决，**不**说明 RW-KSVD 实现必错。

Stage3 在另一 seed/网格上 M0 最好约 **0.81**（p=q=0.5），仍低于度特征天花板叙事；见 `full_pipeline.json` stage3。

---

## Stage 2a — MUTAG `stage2a-tud-skfold-v0`

**协议**：Stratified 10-fold **mean** Acc + 结构-only embedding + LR。  
**不是** Xu/GIN/CIN 的 max-val 乐观协议；**不能**与 CIN 92.7% 横比。

| 方法 | Acc (mean±std) |
|------|----------------|
| B0 + KSVD | 0.665 ± 0.099 |
| B1 + KSVD | 0.660 ± 0.058 |
| M0 + KSVD | 0.649 ± 0.070 |
| **degree_stats** | **0.878 ± 0.058** |

**解读**（与 *A Fair Comparison* 一致）：MUTAG 上简单度/规模特征很强；当前 **仅结构邻接块 + 浅 LR** 的 KSVD 读出 **未超过** degree baseline。管道通了，**方法尚未在下游赢**。

---

## Stage 3 — M0 的 p,q 网格（合成）

Best（json）：`p=0.5, q=0.5`, acc≈**0.81**, recon≈0.02。  
完整网格见 `full_pipeline.json` → `stage3_ablation.grid`。

---

## 未完成

| 项 | 原因 |
|----|------|
| **ogbg-molhiv** 主对标 | 未装 `ogb`；需 `pip install ogb` + 下载后接同一 `pipeline` |
| 节点属性融合 | 阶段约定后置；MUTAG 现为纯结构 |
| 与 CIN 同协议数字 | 需 OGB + 更强读出/融合，非本次 smoke |

---

## 结论（给自己/导师）

1. **工程**：采样 → 诱导邻接 → KSVD → 读出 → CV **全链路已通**。  
2. **科学**：在「度就能分」的设定上，当前结构字典 **没有优势**；符合 fair-comparison 警告。  
3. **下一步真正有意义的**：  
   - 装 OGB 跑 **molhiv**（结构+原子特征融合）；  
   - 或换 **度无法 trivially 分开** 的合成结构任务；  
   - 读出/字典共享/监督微调，而不是只调 p,q。

原始 JSON：`results/full_pipeline.json`。
