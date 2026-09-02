# KSVD U0-D：无人工原子路线的 dictionary optimization / health audit

> 日期：2026-07-31
>
> 正式结论：**PASS_KSVD_OPTIMIZATION**

## 1. 本实验回答什么

U0-P 已确认 walk-order adjacency patches 暴露 LOW/HIGH regime 信号。U0-D 现在只问：

> 在每个独立数据 replicate 上，只使用一次 deterministic maximin initialization，普通 KSVD 是否比完全相同的 INIT 改善 held-out sparse reconstruction，并保持非坍缩字典？

本阶段不使用 graph labels 训练字典，不做 atom-to-motif matching，也不做多 restart 或模型挑选。

## 2. 冻结配置

- master seeds：`731101, 731102, 731103, 731104, 731105`；
- train / validation / test：每类 `150 / 50 / 100` 图；
- patches per graph：`24`；
- primary signal：15-D walk first-discovery-order induced adjacency upper triangle；
- 只减去 train patch coordinate mean；不做 per-patch normalization；
- `K=12`, `T=2`, `T_min=1`；
- FINAL updates：`25`；
- initialization：一次 deterministic maximin，第一列取 centered norm 最大训练 patch，后续最小化与已选 atoms 的最大 absolute cosine；
- INIT 和 FINAL 使用完全相同的 `D_init`；dead-atom internal seed 固定为 0；
- validation/test 不参与初始化、字典训练或超参数选择。

## 3. held-out reconstruction

| seed | INIT test rel. err | FINAL test rel. err | relative reduction | FINAL test NMSE |
|---:|---:|---:|---:|---:|
| 731101 | 0.6895 | 0.4494 | 0.3482 | 0.2020 |
| 731102 | 0.6833 | 0.4438 | 0.3506 | 0.1969 |
| 731103 | 0.6918 | 0.4483 | 0.3519 | 0.2010 |
| 731104 | 0.6807 | 0.4694 | 0.3104 | 0.2204 |
| 731105 | 0.6992 | 0.4431 | 0.3663 | 0.1963 |

五次 relative reduction：mean `0.3455`，std `0.0187`，minimum `0.3104`；`5/5` 达到 10%。

Registered reconstruction gate：**PASS**。

## 4. FINAL test dictionary health

| seed | mean nnz | nondead atoms | effective atoms | max coherence | max activation share | health |
|---:|---:|---:|---:|---:|---:|---|
| 731101 | 2.0000 | 12/12 | 9.0497 | 0.6869 | 0.2757 | PASS |
| 731102 | 2.0000 | 12/12 | 9.0894 | 0.5559 | 0.2553 | PASS |
| 731103 | 2.0000 | 12/12 | 9.5095 | 0.6151 | 0.2302 | PASS |
| 731104 | 2.0000 | 12/12 | 9.0696 | 0.7309 | 0.2585 | PASS |
| 731105 | 2.0000 | 12/12 | 8.5634 | 0.6648 | 0.2940 | PASS |

Across replicates：nondead gate `5/5`，effective-count gate `5/5`，coherence gate `5/5`。

Registered health gate：**PASS**。

## 5. 跨 replicate 描述性稳定性

这里不设置通过门槛。atom matching 使用 absolute cosine 的 exact maximum-sum assignment；subspace 指标使用 principal-angle cosines。

| stage | mean pair matched atom cosine | minimum pair mean | mean subspace cosine | minimum pair mean |
|---|---:|---:|---:|---:|
| `init` | 0.5505 | 0.4804 | 0.9046 | 0.8737 |
| `final` | 0.8621 | 0.8160 | 0.9215 | 0.8710 |

## 6. 结论边界

**PASS_KSVD_OPTIMIZATION**

Proceed to U1A.  Evaluate graph-level INIT and FINAL code readouts on the same datasets, retaining raw patch, Gaussian, medoid, PCA, simple-stat, and shuffle controls.

该结论只支持：普通单次初始化 KSVD 在当前 walk-induced patch distribution 上学到了更好的 held-out sparse reconstruction basis，且未发生注册定义下的严重坍缩。

它仍不支持：

- 每个 atom 都是合法 adjacency 或可命名 motif；
- FINAL codes 比 INIT codes 更适合图级任务；
- KSVD 超过 edge-count histogram、PCA 或 simple graph statistics；
- 该结论可直接外推到真实大图。

因此下一步必须是 paired U1A：同数据、同初始化、同 classifier protocol 下比较 INIT 与 FINAL，而不是直接展示 FINAL accuracy。
