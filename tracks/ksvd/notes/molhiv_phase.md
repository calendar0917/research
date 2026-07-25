# molhiv 验证阶段（2026-07-25）

> 在采样闭环坐实后，进入 **KSVD 优化 / 池化 / 特征融合** 的可测验证。  
> 主数据：`ogbg-molhiv`。协议：`ogb-molhiv-v0`。

## 1. 已定决策

| 项 | 决定 |
|----|------|
| 主协议 | `ogb-molhiv-v0`：OGB scaffold，test ROC-AUC @ best val，≥3 seeds 烟测 / ≥10 正式 |
| 字典 | **共享 D**（train only）；先 **无监督 KSVD**，LC/FDDL **后置** |
| 池化 | 优先 **MIL attention** vs mean/max；层次聚类第二刀 |
| 融合 | **双通道**：属性 GINE ‖ 结构 \(s_G\)；**不把 atom 特征拼进字典** |
| 负对照 | degree hist、B0 1-hop、随机 patch、结构-only LR |

## 2. 阶段与成功线

| 阶段 | 内容 | 过线 |
|------|------|------|
| **P0** | 依赖 + 数据 + 协议锁 | 能 `load_molhiv`，打印 split 规模 |
| **P1** | 结构-only 探针 | \(s_G\) AUC **显著 >** random / degree |
| **P2** | 双通道 GINE ‖ \(s_G\) | 融合 **≥** GINE-only（多种子） |
| **P3** | 池化消融 mean/max/attn | attn 有稳定 Δ 或可解释 |
| **P4** | （可选）轻量 LC 正则 / 环偏置 | 有增益再写进主线 |

## 3. 明确不做（本阶段）

- FDDL / WDDL 全家桶
- atom 特征拼进 \(Y\)/\(D\)
- 宣称「CIN 量级」未过 P2
- 与 MUTAG Xu 数字混表

## 4. 入口命令

```bash
# 推荐解释器（本机已有 numpy/sklearn）
export PY=/home/calendar/.conda/envs/gsn-official/bin/python
cd tracks/ksvd

# 依赖（molhiv 额外；一次）
$PY -m pip install -r configs/requirements-molhiv.txt

# 合成池化消融（无 ogb）
$PY -m code.run_pool_ablation

# P0: 数据检查
$PY -m code.run_molhiv_probe --check-only

# P1: 结构-only（限量烟测）
$PY -m code.run_molhiv_probe --max-graphs 2000 --pools mean,max,attn

# P2: 双通道（需 torch/PyG/ogb）
$PY -m code.run_molhiv_dual --fusion concat --pool attn --epochs 50
$PY -m code.run_molhiv_dual --fusion gine_only --epochs 50   # 对照
```

## 5. 与上一阶段关系

- 采样默认算法不变：`CoverageRW-KSVD-Readout`（`notes/graph_level_algorithm.md`）
- 本阶段只换 **readout/pool** 与 **下游数据**
- 基线 git tag：见仓库 `main` 首提交 *baseline before molhiv phase*
