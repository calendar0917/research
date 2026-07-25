# 设计消融结论（n=6000 scaffold 子采样 · seed=0）

> 目标：各环节相对比较，找 **可能有效方向**，不刷全量 SOTA。  
> 协议：`molhiv-design-ablation-v0` + residual + dual 烟测。

## 0. 关键基线（先看这个）

| 基线 | valid AUC | test AUC | 含义 |
|------|-----------|----------|------|
| degree | 0.605 | 0.588 | 度直方图 |
| **size (n,\|E\|)** | **0.765** | **0.773** | 分子大小已很强 |
| process_cov | 0.764 | 0.771 | 覆盖/walk 数等过程量 ≈ size |

**警告：** 只报「结构-only > degree」不够；必须压过 **size**，否则可能在编码图规模。

---

## 1. 分环节排名（结构-only valid）

| 环节 | 有效？ | 证据 |
|------|--------|------|
| **pool** | **max 有效** | max 0.637 ≫ mean 0.491；attn≈0.61 一般；rich 无增益 |
| **sampler** | **coverage > B0** | cov 0.637 vs B0 0.448；无早停/uncovered **更差** |
| **bias p,q** | **BFS 略好** | BFS 0.637 > neutral 0.617 > DFS 0.552 |
| **RF L,m** | **小/大两端 val 高，需 residual** | L4m4 val0.70 但 test 弱；L12m12 val0.68 test0.76 |
| **dict** | **小字典更好** | A8T2 **0.702** > A16T3 0.637 > A24 0.619 |
| **cover 抬高** | **否** | no_earlystop / uncovered 掉点 → 不是 cover 不够 |

完整表：`DESIGN_ABLATION_n6000_SUMMARY.md`

---

## 2. 残差检验（相对 size）— 真结构信号

| 设计 | s_only val | **s+size val** | Δ vs size(0.765) |
|------|------------|----------------|------------------|
| **A8T2 + max + L8m8** | 0.702 | **0.803** | **+0.037** |
| default max | 0.676 | 0.787 | +0.022 |
| L4m4 max | 0.697 | 0.781 | +0.016 |
| L12m12 max | 0.679 | 0.704 | **−0.061**（有害） |

**结论：**  
- **小字典 + max pool + 默认 coverage** 在 size 之上仍有 **~+3～4pt valid** → 这是目前最像「真结构」的方向。  
- 一味加大感受野（L12）在 residual 下 **负贡献**（可能混入噪声/规模相关模式）。

---

## 3. 双通道（n=3000 烟测，仅看相对）

| 设定 | test@best_val |
|------|----------------|
| gine_only | **~0.78** |
| concat + default max | ~0.70（更差） |
| concat + A8T2 max | val 虚高、test **~0.58**（过拟合） |

**结论：** 图级 concat **当前无效**；结构通道还不能硬拼进 GINE 头。融合方式要改，或先把 \(s_G\) 做强再融。

---

## 4. 可能有效的方向（优先序）

### 保留 / 推进

1. **CoverageRW 保留**（相对 B0 明显更好）；**不要**为抬 cover 关早停。  
2. **pool = max**（弃 mean；attn 暂不主推）。  
3. **小字典 A8T2（或附近）** — 现残差最强。  
4. **BFS 偏置 (p=0.5,q=2)** 作默认。  
5. **叙事与评估**：结构贡献必须报 **相对 size / GINE**，不能只比 degree。

### 暂缓 / 否决（本轮）

| 方向 | 原因 |
|------|------|
| 抬高 cover / 无早停 | valid 崩 |
| mean pool / 无监督 MIL | 弱或过拟合 |
| 图级 concat 进 GINE | 烟测掉点 |
| 盲目 L12 大 patch | residual 负 |
| FDDL/LC 全家桶 | 基础通道仍弱，后置 |

### 下一刀设计（建议）

| 优先级 | 改什么 | 为什么 |
|--------|--------|--------|
| **P1** | **节点级 \(s_v\)** 门控进 GINE（非图级 concat） | MUTAG 上节点级曾接近 gin；图级已失败 |
| **P2** | patch 向量 **加粗粒度化学标签**（原子类型直方图拼 y，或边类型）作对照 | 纯拓扑上限可能到了 |
| **P3** | 固定 A8T2+max，扫 T/atoms 细网格 + seed×3 | 巩固 residual 是否稳 |
| **P4** | 环/小回路偏置采样（对标 CIN 对象） | L 两端都有信号但未稳定 |

---

## 5. 一句话

**有效环节：coverage 采样 + max 池化 + 偏小字典；结构相对 size 有弱增益（+3～4pt）。**  
**无效/危险：抬 cover、mean/硬 concat、盲目大 RF。**  
下一步重点：**换融合形态（节点级）或换 patch 语义**，不是全量、也不是再堆采样覆盖。

## 复现

```bash
export PY=/home/calendar/.conda/envs/gsn-official/bin/python
cd tracks/ksvd
$PY -m code.run_molhiv_design_ablation --max-graphs 6000 --seed 0
# residual / dual_A8T2 见 results/molhiv/*.json
```
