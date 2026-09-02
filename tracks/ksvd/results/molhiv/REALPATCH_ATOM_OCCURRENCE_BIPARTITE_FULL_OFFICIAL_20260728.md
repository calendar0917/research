# 真实局部原型 atom--occurrence 二部模型：全量 official valid/test

日期：2026-07-28

## 1. 冻结依据

路线已不再要求 KSVD 作为核心词汇。冻结候选采用：

```text
raw radius-2 permutation-invariant patch
  -> official-train-fit PCA64
  -> 32 个确定性的 farthest observed train patches
  -> atom--occurrence--atom 两轮受限传播
  -> atom/occurrence 分别 attention + mean + max readout
```

没有 atom--atom GINE，也没有 occurrence--occurrence GINE。

在进入 official evaluation 前，候选已在 8,000 图、official-train-only 的三个 scaffold folds 上补齐 3 seeds：

| Family | 3 folds x 3 seeds mean AUC |
|---|---:|
| matched farthest node MIL | 0.717316 |
| farthest bipartite | **0.733284** |
| gain | **+0.015968** |

配对胜场为 **6/9**，三个 fold 的 seed-mean 增益分别为：

```text
fold 0  +0.033517
fold 1  +0.011070
fold 2  +0.003316
```

所以它通过了冻结条件：平均增益至少 `+0.005`、至少 `6/9` wins、没有某个 fold 的 seed-mean 整体下降。

## 2. 全量协议

- 完整 `ogbg-molhiv`：41,127 graphs；
- train/valid/test：32,901 / 4,113 / 4,113；
- 原型和 PCA metric 只使用 official train；
- 固定 32 prototypes、top-3 support、hidden 48、2 轮二部传播；
- 固定 30 epochs；
- model seeds：0/1/2；
- 三 seed ensemble：预先声明的 sigmoid probabilities 算术平均；
- matched node MIL 使用完全相同的 prototype vocabulary 和 seeds；
- node MIL 参数：29,234；
- bipartite 参数：64,611。

仓库在更早实验中已经查看过 official test，因此这里是 **controlled frozen evaluation**，不是 untouched test。结果不得用于继续搜索本轮模型的层数、宽度、seed、epoch 或融合权重。

## 3. 单模型结果

### 3.1 Official valid

| Seed | Node MIL | Bipartite | Delta |
|---:|---:|---:|---:|
| 0 | 0.750729 | **0.776712** | +0.025983 |
| 1 | 0.763962 | **0.784594** | +0.020631 |
| 2 | **0.794778** | 0.782741 | -0.012036 |
| Mean | 0.769823 | **0.781349** | **+0.011526** |

Bipartite valid sample std 为 `0.004121`，明显小于 node MIL 的 `0.022602`。

### 3.2 Official test

| Seed | Node MIL | Bipartite | Delta |
|---:|---:|---:|---:|
| 0 | **0.733778** | 0.710570 | -0.023208 |
| 1 | 0.705404 | **0.752135** | +0.046731 |
| 2 | 0.713523 | **0.728056** | +0.014533 |
| Mean | 0.717568 | **0.730254** | **+0.012685** |

Bipartite 在 valid/test 上都以 2/3 seeds 胜 matched node MIL。开发阶段观察到的“二部传播比 node MIL 更有用”因此得到方向一致的全量支持。

## 4. 三 seed probability ensemble

| Family | Official valid AUC | Official test AUC |
|---|---:|---:|
| farthest real-patch node MIL | 0.785769 | 0.734443 |
| **farthest real-patch bipartite** | **0.793553** | **0.737741** |
| bipartite - node | **+0.007783** | **+0.003299** |

二部结构的 ensemble 增益在 valid 和 test 上都为正，但 test 增益只有约 `0.0033`。

固定 seed 的 5,000 次 paired stratified bootstrap（每次分别重采样阳性和阴性分子）显示：

| Split | Ensemble delta | 95% bootstrap interval | P(delta > 0) |
|---|---:|---:|---:|
| official valid | +0.007783 | [-0.026541, +0.041796] | 0.6822 |
| official test | +0.003299 | [-0.032285, +0.041015] | 0.5688 |

区间均跨过 0。因此，matched ensemble 的正差值只能视为方向一致的描述性结果，不能视为已经证明的稳定提升。

## 5. 与已有全量结果的描述性比较

这些 family 并非全部同参数量、同 prototype bank，因此只能用于判断绝对竞争力，不能替代 matched comparison。

| Family | Official valid ensemble | Official test ensemble |
|---|---:|---:|
| farthest bipartite（本轮） | 0.793553 | 0.737741 |
| 旧 graph-balanced observed real-prototype MIL | **0.802488** | 0.767771 |
| fixed-epoch original-node GINE | 0.781385 | **0.777993** |

本轮 bipartite 相对：

- 旧 real-prototype ensemble：valid `-0.008935`，test `-0.030030`；
- GINE ensemble：valid `+0.012168`，test `-0.040252`。

因此，虽然二部传播稳定超过了它自己的 matched node baseline，但当前绝对 test AUC 仍明显不足。

## 6. 结论

### 6.1 得到支持的部分

二部传播的机制判断基本成立：

- 8k scaffold development：`+0.015968`，6/9 wins；
- full official-valid ensemble：`+0.007783`；
- full official-test ensemble：`+0.003299`；
- official valid/test 的单 seed mean 均为 2/3 wins。

因此，PrototypeMIL 丢掉 occurrence 组成关系确实会损失一部分信息，而无需完整 GINE 的受限 atom--occurrence 传播可以恢复其中一部分。

### 6.2 没有得到支持的部分

这套具体实现不能作为新的主结果：

1. bipartite test ensemble 只有 `0.737741`；
2. valid 与 test 相差 `0.055812`；
3. 相对 matched node ensemble 的 test 增益仅 `+0.003299`；
4. 增加了约 35k 参数，却仍低于旧真实原型 ensemble 和 GINE；
5. seed 0 在 test 上反而下降 `-0.023208`，说明二部模型的泛化仍不稳定。

### 6.3 准确的研究定位

当前可以保留的结论是：

> occurrence incidence 是真实但较弱的结构信号；atom--occurrence 二部传播是合理的低层机制，但它不能弥补不稳定或不适合分类的 prototype vocabulary，也没有自动解决 valid--test domain shift。

所以路线 A 的“结构假设”得到部分验证，但这版模型不晋级为最终模型。CIN-like 和更深 GINE 也没有必要基于该结果继续扩张，因为主要限制已经不是传播范围不足，而是 prototype vocabulary 与跨 scaffold/test 泛化。

## 7. 后续边界

本轮 official test 已经完成，不应再根据这些数值调本轮模型。若继续研究，应回到 official-train-only scaffold folds，或转到新的外部数据集，并预先固定新的问题，例如：

- graph-balanced 多 prototype banks 是否能在二部结构中降低 vocabulary 方差；
- 多 bank 的一致 occurrence 是否比单 bank occurrence 更稳定；
- 将二部 residual 限制在很小幅度，是否能保留 node MIL 的稳定性而只补充 incidence 信息。

这些必须作为新的 train-only 实验重新设门槛，不能继续用本次 valid/test 选择设计。

## 8. Artifacts

代码：

- `code/run_molhiv_atom_occurrence_bipartite_official.py`
- `code/summarize_molhiv_atom_occurrence_bipartite_official.py`

结果：

- `results/molhiv/farthest_bipartite_full_official_seed0.json`
- `results/molhiv/farthest_bipartite_full_official_seed1.json`
- `results/molhiv/farthest_bipartite_full_official_seed2.json`
- `results/molhiv/farthest_bipartite_full_official_3seed_summary.json`
