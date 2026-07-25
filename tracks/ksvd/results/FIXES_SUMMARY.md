# Fix experiments v1 — 结果与解读

## 实现了哪些修复

| # | 修复 | 代码 |
|---|------|------|
| 1 | 诱导子图 **BFS 排序**（相对纯度排序） | `vectorize.order_nodes_bfs` |
| 2 | **共享字典**：仅在 train fold 上学 \(D\)，全图用同一 \(D\) 编码 | `pipeline.embed_dataset_shared_dict` + `shared_dict_cv` |
| 3 | **Oracle**：全图三角/密度等手写特征 | `graph_oracle_features` |
| 4 | **Patch-pool（无 KSVD）**：patch 的 adj+local stats 均值 | `patch_pool_features` |
| 5 | 可选：把 local stats **拼进 \(Y\)** 再 KSVD | `append_local_stats` |

入口：`python -m code.run_fixes` → `results/fixes.json`

---

## Synth：三角形 vs 长环（同 n、同 \|E\|）

| 方法 | Acc | 说明 |
|------|-----|------|
| degree | 0.575 | 任务仍难被度分开 ✓ |
| **oracle 三角统计** | **1.000** | 标签本质可分 ✓ |
| **patch_pool B0/B1/M0** | **0.98–1.00** | **采样+向量化已含信号** |
| per-graph KSVD（旧设定） | 0.54–0.64 | 与修复前类似，仍差 |
| **shared_bfs KSVD** | **0.855 ± 0.05** | **共享字典关键突破** |
| shared_bfs + localstats in Y | 0.990 | 字典+显式局部统计 |
| **shared_B0_bfs** | **1.000** | 一阶+共享字典足够 |

### 解读（合成）

1. **瓶颈主要不在 RW**，而在 **每图独立字典 → 系数不可比**。  
2. 共享 \(D\) 后，纯邻接 patch 的 KSVD 从 ~0.55 拉到 **~0.86**。  
3. Patch-pool 已近满分 → 信息在 patch 里；KSVD 读出仍可再挤。  
4. B0+共享字典即可完美分 → 此任务不必 RW；RW 的价值要到更大感受野任务再验。

---

## MUTAG

| 方法 | Acc | 备注 |
|------|-----|------|
| attr | 0.857 | |
| degree | 0.878 | |
| oracle 结构手写 | （见 json） | |
| **shared_M0 结构（诚实 CV）** | **0.672** | 略好于旧 ~0.65，仍远低于度 |
| shared_M0 all-D 结构（乐观） | 0.725 | D 见过全数据，偏乐观 |
| s+attr（乐观 D） | 0.809 | 仍 **< attr alone** |
| s+attr+deg（乐观 D） | 0.846 | 仍 **≤ deg** |
| patch_pool M0 | **0.761** | 无字典，高于 per-graph KSVD |
| patch_pool+attr | 0.820 | 仍 < attr |

### 解读（MUTAG）

1. 共享字典有帮助，但 **结构通道仍弱于度/属性**。  
2. Patch-pool > 旧 KSVD → 字典/读出仍在丢信息。  
3. 拼结构仍容易 **稀释属性**（线性头）。  
4. MUTAG 上「可还原字典」尚未变成下游增益；**合成上已证明共享字典方向正确**。

---

## 结论：做到哪一步了

| 目标 | 状态 |
|------|------|
| 管道能跑 | ✅ |
| 可控任务上证明结构信号存在 | ✅（oracle / patch_pool） |
| 证明 **共享字典** 必要 | ✅（0.55→0.86） |
| RW 优于一阶（此任务） | ❌ 不必要（B0 已满分） |
| MUTAG 超过度/属性 | ❌ |
| molhiv | 仍不急 |

### 下一步（仍先做实、不上 molhiv）

1. **读出**：对共享 \(X\) 用直方图/分位数，或小 MLP（仍 CV）。  
2. **监督字典 / 半监督**：类别相关原子。  
3. **合成升级**：需要 \(k\geq 2\) hop 才可分的 motif（逼出 RW）。  
4. MUTAG：结构作 **残差门控** 拼属性，避免硬拼接降分。

---

## 一句话

**修复前**：结构通道≈噪声。  
**修复后**：在合成任务上，**共享字典把 KSVD 从随机拉到可用（~0.86）**；采样本身（patch-pool）已近完美。  
**MUTAG**：诚实共享字典仍明显弱于度/属性——分子任务还要属性友好的融合，不是再堆 RW。
