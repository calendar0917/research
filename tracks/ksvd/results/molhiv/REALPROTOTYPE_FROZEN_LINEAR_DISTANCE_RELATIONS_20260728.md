# Frozen-base bounded-linear distance relations（2026-07-28）

## 1. Protocol

- 8,000-graph subset；只使用 6,400 个 official-train graphs 和三个 scaffold folds。
- official valid/test 编码与评估均为 **0**。
- 先训练并冻结 46,658-parameter occurrence PrototypeMIL。
- Relation residual 只有 `2650` 个参数，且 `|residual| <= 0.25`。
- feature standardization 只使用 outer-fit graphs；real 与 distance-shuffled 使用相同训练协议。

## 2. Results

| Vocabulary | Model | Fold 0 | Fold 1 | Fold 2 | Mean |
|---|---|---:|---:|---:|---:|
| farthest | occurrence | 0.750784 | 0.703799 | 0.776604 | 0.743729 |
| farthest | frozen relation | 0.753444 | 0.711120 | 0.780785 | 0.748450 |
| farthest | shuffled | 0.755816 | 0.712612 | 0.778567 | 0.748998 |
| scaffold | occurrence | 0.750577 | 0.687013 | 0.793251 | 0.743614 |
| scaffold | frozen relation | 0.746561 | 0.689482 | 0.793997 | 0.743346 |
| scaffold | shuffled | 0.746525 | 0.680753 | 0.791690 | 0.739656 |

- Farthest relation − occurrence：`+0.004721`，赢 `3/3`；real − shuffled `-0.000549`。
- Scaffold relation − occurrence：`-0.000267`，赢 `2/3`；real − shuffled `+0.003691`。

## 3. Pair ensemble

| Ensemble | Fold 0 | Fold 1 | Fold 2 | Mean |
|---|---:|---:|---:|---:|
| occurrence | 0.766886 | 0.699399 | 0.800924 | 0.755736 |
| frozen relation | 0.767241 | 0.707760 | 0.800319 | 0.758440 |
| shuffled | 0.765932 | 0.699788 | 0.799229 | 0.754983 |

Pair relation − occurrence：`+0.002704`，赢 `2/3`；real − shuffled `+0.003457`。

## 4. Decision

- Baseline reproduction max error：farthest `0.000e+00`，scaffold `0.000e+00`，pair `0.000e+00`。
- Promotion：**False**。

即使冻结 base 并严格限制 residual，当前 full-matrix linear relation 仍未通过 gate；下一步应转向距离尺度消融或固定低秩统计，而不是继续扩大 relation head。