# Structural canonical slots：三种子结论

> 日期：2026-08-02  
> 范围：3 graph-bank seeds × 3 cover seeds；每次72张50-node synthetic graphs；3-fold graph isolation；完整45D patch adjacency；`K24/T3/updates25`。

## 1. 核心结论

用户提出的修正成立：不需要把完整 patch adjacency 换成统计 descriptor。对每个 patch 使用以 center 为 distinguished color 的 exact canonical ordering，可以在保留45D完整邻接信息的同时，消除 numeric-ID slot instability，并改善 KSVD 与 stitched graph reconstruction。

三次独立审计均判定：

```text
ADOPT_STRUCTURAL_CANONICAL_SLOTS
selected = ROOTED_CANONICAL
```

## 2. 三种子 reconstruction

| seed | construction observed/full RMSE | rooted canonical observed/full RMSE | observed improvement | full improvement |
|---|---:|---:|---:|---:|
| 1 | 0.34760 / 0.33368 | 0.31316 / 0.31509 | 9.91% | 5.57% |
| 2 | 0.34934 / 0.33336 | 0.31126 / 0.31258 | 10.90% | 6.24% |
| 3 | 0.34833 / 0.33562 | 0.31301 / 0.31664 | 10.14% | 5.66% |
| mean | 0.34842 / 0.33422 | **0.31248 / 0.31477** | **10.32%** | **5.82%** |

首个正式 seed 的其他指标：

| branch | patch error | observed F1 | full recall/F1 |
|---|---:|---:|---:|
| construction | 0.4305 | 0.8803 | 0.8182 / 0.8250 |
| rooted canonical | **0.3882** | **0.9030** | **0.8270 / 0.8455** |

24/24 atoms 在所有 folds 中有效。

## 3. 重编号稳定性

Frozen patch node sets 经全图 relabel 后，各 branch 独立重新生成 slots：

| branch | vector match | code cosine | pooled cosine | 解释 |
|---|---:|---:|---:|---|
| ID sort | 约0.031 | 约0.121 | 约0.645 | numeric ID 不是结构坐标 |
| structural signature | 约0.914 | 约0.950 | 约0.979 | 约50% patches 仍有 signature tie |
| unrooted canonical | 1.000 | 1.000 | 1.000 | 完整 topology 稳定 |
| rooted canonical | **1.000** | **1.000** | **1.000** | 完整 topology + center role 稳定 |
| overlap canonical | 约0.996 | 约0.997 | 约0.999 | automorphism ambiguity 会沿 chain anchors 传播 |

rooted canonical 的 node-order equivariance 平均约 `0.941`、transition-slot match 约 `0.896`，低于 adjacency-vector 的1.0。原因不是 canonical code 读取了 ID，而是约48.8%的 patches 存在可检测的 automorphism-equivalent node maps（包括搜索中被安全剪枝的 twin symmetry）：纯结构可以唯一确定 canonical adjacency，却无法唯一命名结构完全等价的真实节点。

## 4. Relabel + resampling

rooted canonical 的 pooled sparse-code cosine平均约 `0.7045`，没有解决 Beam8 重采样后选择不同 patch sets 的问题。但 reconstruction quality 稳定：

```text
original full RMSE mean          0.31477
relabel-resample full RMSE mean  0.31482
absolute difference              0.00005
```

所以必须区分：

1. canonical slots 已经修复同一 patch topology 的 local-coordinate instability；
2. Beam8 的具体 patch-set 仍不严格 relabel-equivariant；
3. 尽管具体 code bag 不同，重构误差几乎不漂移。

## 5. 为什么 rooted 优于其他规则

- `ID_SORT`：既不稳定又恶化 reconstruction；
- `SIGNATURE`：结构几何较好，但 tie 仍需 ID，不能严格 replay；
- `CANONICAL`：严格稳定且重构改善，但忽略同一 topology 中 center 的角色；
- `ROOTED_CANONICAL`：center singleton color 保留 sampler 给出的局部参照，同时 exact canonicalization 解决其余 slot；
- `OVERLAP_CANONICAL`：强制把上一步 shared slots 作为连续锚点，transition 更稳定，但早期 automorphism choice 会传播，且 reconstruction 明显弱于 rooted canonical。

## 6. 更新后的方法定位

图重构主路线：

```text
Beam8 patch node sets
→ exact rooted canonical local slots
→ complete 45D adjacency vector
→ shared K24/T3 KSVD
→ slot-to-node maps + overlap stitching
```

图表示路线仍可另行使用 invariant descriptor。两者不是替代关系：

- rooted canonical 45D KSVD：优先保留完整 patch topology 和 adjacency reconstruction；
- invariant 22D / invariant-space KSVD：优先获得不依赖 node-map 的 compact structural token。

## 7. 边界与下一步

- 恢复带全局编号的邻接矩阵仍需 slot-to-node/transition sidecar；
- automorphism-equivalent nodes 不存在纯结构唯一身份；
- 若下游要求 code embedding 对 relabel+resampling 也接近1，下一问题是 sampler canonicalization或多采样 set-level aggregation，不应再修改 local slot ordering；
- rooted-canonical coordinate distribution 尚未重新做独立 K/T capacity grid，`K24/T3` 是当前通过三种子的冻结配置，不声称容量全局最优。

## 8. 产物

- 协议：`KSVD_STRUCTURAL_CANONICAL_SLOT_PROTOCOL_20260802.md`；
- 实现：`canonical_slots.py`、`run_structural_canonical_slot_audit.py`；
- tests：`test_canonical_slots.py`、`test_structural_canonical_slot_audit.py`；
- 三次正式结果：`STRUCTURAL_CANONICAL_SLOT_AUDIT_20260802.md`、`STRUCTURAL_CANONICAL_SLOT_AUDIT_SEED2_20260802.md`、`STRUCTURAL_CANONICAL_SLOT_AUDIT_SEED3_20260802.md`。
