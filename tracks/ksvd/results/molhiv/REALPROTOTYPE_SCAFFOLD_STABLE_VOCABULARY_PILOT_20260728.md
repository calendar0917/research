# Real-prototype scaffold-stable vocabulary pilot（2026-07-28）

## 1. 数据边界

- 8,000-graph development subset；只使用其中 6,400 个 official-train graphs。
- 三个 official-train-only Bemis–Murcko scaffold outer folds。
- model seed 固定为 0，固定训练 30 epochs。
- 本轮 official valid/test 编码与评估次数均为 **0**。

## 2. 为什么做这个 pilot

稳定性诊断显示：三个随机真实 prototype banks 的 optimal-matching cosine 只有 `0.557700`，
而 probability ensemble 相对单 bank 均值增加 `+0.007604` AUC；但是 fit→heldout coverage
变化仅 `-0.000231`，prototype usage JS 仅 `0.001544`。因此主要问题不是普通的
跨 scaffold coverage collapse，而是 prototype identity 与 downstream composition 的方差。

本 pilot 把候选池从 256 个随机来源图扩大为每个 outer-fit graph 一个确定性 observed patch，
并比较 random、farthest、uniform facility、真实 scaffold-balanced facility、
shuffled-scaffold facility。所有 selector 均 label-free。

## 3. 三折结果

| Selector | Fold 0 | Fold 1 | Fold 2 | Mean | Std | Heldout coverage |
|---|---:|---:|---:|---:|---:|---:|
| fixed_random | 0.730325 | 0.666421 | 0.782079 | 0.726275 | 0.057935 | 0.704550 |
| farthest | 0.750784 | 0.703799 | 0.776604 | 0.743729 | 0.036912 | 0.695981 |
| uniform_facility | 0.728845 | 0.696718 | 0.778453 | 0.734672 | 0.041178 | 0.809383 |
| scaffold_facility | 0.750577 | 0.687013 | 0.793251 | 0.743614 | 0.053460 | 0.807738 |
| shuffled_scaffold_facility | 0.761523 | 0.665566 | 0.756128 | 0.727739 | 0.053911 | 0.811504 |

严格 scaffold-specific gate：

- scaffold facility − matched random：`+0.017339`，赢 `3/3` folds；通过。
- scaffold facility − uniform facility：`+0.008942`；通过。
- scaffold facility − shuffled scaffold：`+0.015875`；通过。
- scaffold facility − farthest：`-0.000115`；未通过。

因此 **scaffold-specific selector 严格 promotion gate 未通过**；它与 broad-pool farthest
几乎完全打平，而不是明确优于所有 matched controls。

## 4. 更重要的结果

- broad deterministic candidate pool 下，farthest mean AUC 为 `0.743729`，
  scaffold facility 为 `0.743614`；二者都高于旧 3-random-bank ensemble `0.728219`。
- farthest + scaffold-facility probability ensemble：fold AUC `[0.7668856211846231, 0.6993987408926929, 0.8009240966158945]`，mean `0.755736`。
- 二者 probability Pearson mean 只有 `0.730830`，存在明显互补。
- 但把两套 prototype 各取 16 个压成单 bank 后，mean 仅 `0.718881`。

这说明互补性不是简单的 atom union：top-3 assignment、prototype competition 和 MIL readout
会随 vocabulary 整体几何改变。coverage 从约 0.70 提升到约 0.81，也没有自动带来同比例 AUC
提升，所以继续只优化 coverage objective 的收益可能有限。

## 5. 研究判断

1. **有效的新结论**：广覆盖、确定性的 observed-patch candidate pool 是明显改进；早期
   farthest 失败主要不能归因于 farthest 原理本身，也与过窄的 256-candidate random reservoir 有关。
2. **尚未证明**：真实 scaffold weighting 本身优于一般的 broad-pool diversity selection。
3. **下一优先级**：固定 farthest/scaffold 两套强 vocabulary，研究 prototype occurrence 的
   distance-binned / diffusion relational composition，而不是继续调 facility 权重。
4. relation head 应先做低容量 GNN-free 版本，并保留 occurrence-independent MIL、
   farthest、scaffold facility、distance-shuffled relation 作为 matched controls。
