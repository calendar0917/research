# MolHIV Beam8 multi-cover masked-chemistry gate

> 日期：2026-08-16  
> 范围：OGB official-train 内部 scaffold folds；不编码或评估 official-valid/test。

## 结论

**GENERIC_MULTICOVER_CONTEXT_PASS；BEAM-SPECIFIC ROUTE REJECTED**

Beam8 的不同 seed 确实产生不同覆盖，因此 multi-cover 在工程上成立；但是在严格去除 chemistry canonicalization 泄漏后，同预算的 random-BFS multi-cover 在三个 scaffold folds 上都稳定优于 multi-Beam。当前证据支持“多重重叠局部上下文”，不支持“Beam 搜索本身提供更好的化学预训练上下文”。

## 1. Cover diversity

1000 个 official-train 分子的 8-seed 审计显示：

| cover | eligible changed graphs | unique covers / 8 | patch Jaccard | variable nodes | variable edges |
|---|---:|---:|---:|---:|---:|
| Beam8 | .9639 | 6.267 | .1985 | .7565 | .7551 |
| random BFS | 1.0000 | 7.754 | .1833 | .9637 | .9776 |

所以 multi-Beam 不是重复同一 deterministic cover；它具有真实的边际覆盖变化。

## 2. Leakage diagnosis

第一版 masked-chemistry gate 使用完整原子类别和键类别进行 attributed canonicalization，同时模型能看到 target 的 canonical slot。由于 masked atom 的类别参与了 cover membership / slot 的构造，这会把目标信息泄漏进预训练输入。

该版本被保留为 leakage diagnostic，但不能用于支持下游迁移。严格版本改为：

- cover membership 与 slot canonicalization 只使用二值拓扑；
- atom categories 只作为被遮蔽目标或其他可见原子的上下文；
- 真实 bond categories 只进入重建模型的内部键上下文，不参与 cover 构造；
- HIV 标签完全不进入 item、loss、模型选择或评估。

## 3. Topology-only 3-fold results

每折使用 1000 个 fit-fold 分子做 8 epochs 无标签训练，在 1000 个 heldout scaffold 分子上把每个原子确定性遮蔽一次。四个 matched variants 使用相同模型容量与初始化：single Beam、4-cover Beam、4-cover Beam patch-context shuffle、4-cover random BFS。

三折 heldout 均值：

| variant | mean CE ↓ | mean field acc ↑ | atomic-number CE ↓ | atomic-number acc ↑ |
|---|---:|---:|---:|---:|
| single Beam | .7785 | .7652 | 1.6402 | .6492 |
| multi Beam | .5748 | .7895 | 1.1407 | .7031 |
| multi Beam shuffled | .6468 | .7477 | 1.1687 | .7030 |
| **multi random BFS** | **.4713** | **.8123** | **.9136** | **.7157** |

三折 paired effects 均值：

| comparison | CE reduction | accuracy gain | fold consistency |
|---|---:|---:|---:|
| multi Beam vs single Beam | +.2037 | +.0243 | 3/3 |
| aligned multi Beam vs shuffled | +.0720 | +.0418 | 3/3 |
| multi Beam vs random BFS | **-.1036** | **-.0227** | Beam loses 3/3 |

解释：增加独立覆盖与保留正确 patch-to-atom binding 都有价值，但 Beam 的贪心边际搜索不是价值来源。更随机、覆盖变化更大的 BFS patches 对 masked chemistry 更有效。

## 4. 对 Beam8 / KSVD 的影响

- 不把当前 multi-Beam reconstructor 迁移到 CIN；它在上游表征 gate 已被 random BFS 支配。
- 如果目标是提升 MolHIV，下一步应命名为 generic multi-cover SSL，并以 random BFS 为主 sampler；这不再是 Beam8 的性能路线。
- KSVD 仍可放在连续 SSL patch embeddings 之后测试压缩、原型可解释性或 sample efficiency，但必须与 PCA、random-prototype 和不压缩连续 embedding 比较。KSVD 不能把一个已被 sampler control 支配的 Beam 表示重新解释成 Beam 优势。
- Beam8 若要重新成为独立研究贡献，需要改变 sampler objective（例如 learnable chemistry reconstruction policy、uncertainty/novelty coverage），并在 topology-only、matched-budget random/BFS controls 下先赢 representation gate，再进入 HIV 分类。

## 5. Safety

- `official_valid_evaluations = 0`
- `official_test_evaluations = 0`
- 所有预训练与重建评估均局限于 official-train 的内部 scaffold folds。

Artifacts:

- `tracks/ksvd/code/run_molhiv_beam8_cover_diversity_audit.py`
- `tracks/ksvd/code/run_molhiv_beam8_masked_chemistry_gate.py`
- `tracks/ksvd/results/molhiv/molhiv_beam8_cover_diversity_audit_20260816.json`
- `tracks/ksvd/results/molhiv/molhiv_beam8_masked_chemistry_topology_3fold_summary_20260816.json`

