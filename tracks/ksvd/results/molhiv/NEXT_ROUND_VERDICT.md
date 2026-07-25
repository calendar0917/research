# Next-round 结论（B/C/D/A · n≈5k · 2026-07-25）

> 协议 `molhiv-next-round-v0`。主文件：`next_round_n5000.json` · `NEXT_ROUND_n5000_SUMMARY.md`

## 总览

| 方向 | 结果 | 判定 |
|------|------|------|
| **B** 小字典细扫 + 多种子 residual | Δ(s+size−size) **跨 seed 正负摇摆**，均值≈0 | **不稳定**；上次 n6k seed0 的 +3.7pt **不可当定论** |
| **C** patch + chem hist | 相对纯 topo 略好，但 **几乎不超 size** | **弱对照**；说明纯拓扑/粗化学统计都难单独成主信号 |
| **D** ring_boost | boost=1 略好于 0；2/4 无增益 | **可保留小 boost=1 作默认偏置**，非主突破 |
| **A** 节点门控进 GINE | gine_only test@best≈**0.78**；node_gate≈**0.67–0.76**；graph_concat≈**0.69** | **节点门控未稳定超过 gine_only**（烟测） |

---

## B — 字典 / residual 稳定性

| 设定 | mean Δ valid (3 seeds) | 解读 |
|------|------------------------|------|
| A8T3（本轮 grid 最优） | **≈ −0.01 ± 0.04** | 不显著 |
| A8T2 | **≈ −0.03 ± 0.06** | 不显著 |

- seed 子采样改变 train 分子集合后，residual **可正可负**。  
- **结论：** 结构通道相对 size 的增益 **脆弱**；不能只靠单 seed 宣称“有真结构”。  
- 仍可保留 **A8–A12、T=2–3、max pool** 作为默认超参区，但评估必须 **多种子 + 相对 size**。

## C — chem patch

| feat | Δ vs size (seed0) |
|------|-------------------|
| topo | −0.11 |
| chem | **−0.003** |

- chem 让 s+size **贴齐 size**，却 **没有明显超出**。  
- 粗 atom/bond 直方图进字典 **不是** 通向 CIN 量级的捷径（可能与属性通道冗余）。  
- 更细的官能团/环指纹或 **不进 D、只做节点 PE** 可另开，但本轮不主推“chem 进 Y”。

## D — ring bias

| ring_boost | Δ vs size |
|------------|-----------|
| 0 | −0.11 |
| **1** | **−0.03**（本 seed 最好） |
| 2 | −0.10 |
| 4 | −0.11 |

- 轻度环偏置 **值得留下**（默认 `ring_boost=1` 可试）。  
- 不是独立主贡献。

## A — 节点级融合（n=3k, 20ep 烟测）

| fusion | test@best_val（约） |
|--------|---------------------|
| **gine_only** | **0.78** |
| node_gate | ~0.67–0.76（随 best val 选点不稳） |
| graph_concat | ~0.69 |

- **图级 concat 再次失败。**  
- **节点 gate 未证明优于 gine_only**（短训 + 子集）。  
- 可能原因：B0 节点 patch 信息弱、gate 过早注入、或需监督/可学习字典。  
- 若再试 A：更长训、全量或更大 n、RW 节点 patch、**只在后几层 gate**、或 edge-level。

---

## 综合：现在相信什么

### 仍然成立

1. CoverageRW **工程闭环**通；合成 C4 上机制成立。  
2. molhiv 上 **max pool + 中小字典 + BFS** 是相对不差的默认。  
3. 评估必须含 **size / GINE**；degree 不够。  
4. **图级硬拼结构向量** 基本可判无效。

### 需要下调的预期

1. “结构相对 size +3～4pt” **单 seed 假象风险高** → 多种子后 ≈0。  
2. 无监督拓扑 KSVD **很难**在 molhiv 上成为 GINE 的稳定增益源。  
3. chem 进字典、环偏置、节点 gate **本轮均未打出清晰胜者**。

### 研究叙事上仍可走的路（收窄）

| 优先级 | 方向 | 理由 |
|--------|------|------|
| 1 | **机制/采样论文向**：CoverageRW + 可还原字典 + 过程指标 + 合成表达力 | 已有证据，不依赖 molhiv 增益 |
| 2 | **结构对象升级**：显式小环/cell 候选再字典（真对标 CIN 对象） | 拓扑 KSVD 上限可能到了 |
| 3 | **可学习/任务驱动编码**：轻量 LC 或联合训 D（在 2 之后） | 无监督 residual 已不稳 |
| 4 | 节点/边结构 **后置弱融合**（长训+多种子再验证） | A 未死但本轮未过线 |

### 建议暂停

- 继续大扫 cover / mean pool / 图级 concat / FDDL 全家桶  
- 用单 seed 子集 AUC 讲“分子任务胜利”

---

## 默认配置（若继续工程）

```text
CoverageRW: p=0.5, q=2, L=8, m=8, cover=0.95, ring_boost=1
KSVD: n_atoms=8–12, T=2–3, pool=max, patch_feat=topo
Eval: always report size residual + multi-seed; dual: gine_only primary
```

## 复现

```bash
export PY=/home/calendar/.conda/envs/gsn-official/bin/python
cd tracks/ksvd
$PY -m code.run_molhiv_next_round --max-graphs 5000 --seeds 0,1,2 --epochs 20
```
