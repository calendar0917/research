# Real-Prototype Compact：冻结 Official Test 结果（2026-07-28）

## 结论

official-valid 的 `0.812457` 没有在 official test 上保持。冻结的 compact real 模型得到：

```text
official test AUC = 0.751461
```

因此，不能再把 valid 的 `0.812457` 直接理解为“已经接近约 0.82 的最终水平”。按相同 terminal-test 口径，本路线当前更合理的最终分数是约 `0.751`。

compact relation 相对 occurrence ensemble 仍有正增益，但 assignment-shuffled control 略高于 real：

```text
compact real - occurrence ensemble       = +0.009322
compact real - assignment-shuffled       = -0.000606
```

这说明 compact 关系特征作为一种附加统计量可能有用，但现有结果仍不能证明“具体原型身份之间的真实配对关系”是增益来源。

## 冻结协议

在编码 official test 前，配置已经写入并封存：

```text
results/molhiv/compact_official_test_freeze_v1.json
SHA256 = 9d2e7cf4c05d3439fe8039608661191a6aa607d667d1a15d7814f54262b8df2f
```

固定配置：

- 只使用 exact official train 拟合，共 32901 个图；
- official valid 不参与重新拟合，也不在 terminal run 中编码；
- seed 0；
- farthest 与 scaffold-facility 各 32 个真实原型；
- uniform gate；
- top-3 正余弦分配；
- exact distance 1/2 compact relation；
- 固定训练 30 epochs；
- 只运行 occurrence、compact real 和 assignment-shuffled control；
- 不运行未通过 valid 门槛的 pair-PCA；
- 不进行 test 后的参数、seed、rank 或模型选择。

## 数据隔离审计

```text
fit indices = exact official train
fit SHA256 = 4e77289653e41be5d2267f54d9f33fda4b70fd622c19b93a282db710a9318b0a

official train graphs = 32901
official valid graphs = 4113
official test graphs  = 4113

official valid nodes = 114300
official test nodes  = 103927

official-valid latent nonzero count = 0
official-test normalized fraction    = 1.0
```

审计文件：

```text
results/molhiv/official_test_latent_isolation_audit_v1.json
```

另外，terminal run 与 official-valid run 的训练部分逐项一致：

- 两个 vocabulary 的 prototype SHA 完全一致；
- 两个 vocabulary 的 train prediction SHA 完全一致；
- compact real 的 train score/residual SHA 完全一致；
- assignment-shuffled 的 train score/residual SHA 完全一致。

因此，test 分数下降不是因为更换了原型、训练配置或随机结果。

## 完整结果

| Predictor | Official valid AUC | Official test AUC | Test - Valid |
|---|---:|---:|---:|
| Farthest occurrence | 0.793326 | 0.733763 | -0.059564 |
| Scaffold-facility occurrence | 0.755962 | 0.721598 | -0.034364 |
| Occurrence probability ensemble | 0.808167 | 0.742139 | -0.066029 |
| Uniform compact real | **0.812457** | **0.751461** | **-0.060996** |
| Uniform compact assignment-shuffled | 0.814726 | 0.752067 | -0.062659 |

Test split：

```text
n = 4113
positives = 130
```

## 不确定性

使用 10000 次固定随机种子的分层 bootstrap，只做结果分析，不参与模型选择：

| Test predictor | AUC | 95% bootstrap interval |
|---|---:|---:|
| Occurrence ensemble | 0.742139 | [0.693459, 0.789343] |
| Compact real | 0.751461 | [0.703912, 0.798566] |
| Assignment-shuffled | 0.752067 | [0.704391, 0.799071] |

配对差值：

| Test difference | Point estimate | 95% bootstrap interval |
|---|---:|---:|
| Compact real - occurrence | +0.009322 | [+0.000365, +0.018110] |
| Compact real - shuffled | -0.000606 | [-0.005775, +0.004703] |

这支持一个较窄的结论：compact 附加统计量相对 occurrence base 可能有稳定的小增益；但 real 和 shuffled 之间无法区分。

## 研究解释

### 1. Valid 的 0.812 不是无效结果，但它高估了 terminal test

valid 只有 81 个正样本，AUC 的不确定性较大；而且这条路线是在 valid 前经过多轮研究选择后才确定的。即使最终 test 配置被冻结，valid 分数仍可能带有模型路线选择造成的乐观偏差。

### 2. Test 的主要下降来自整个 occurrence 表示，而不是 compact 单独失效

occurrence ensemble 从 `0.808167` 降到 `0.742139`，下降 `0.066029`；compact real 从 `0.812457` 降到 `0.751461`，下降 `0.060996`。compact 在更难的 test 上仍增加约 `0.0093`，但无法补足基础表示的跨 split 差距。

### 3. 当前不能宣称“人物/原型身份匹配关系成立”

real 与 assignment-shuffled 的预测相关系数为 `0.991742`，test AUC 也几乎相同，并且 shuffled 略高。这更像是 compact relation head 利用了总体距离分布、共现强度或平滑正则化，而不是依赖某个具体原型 A 与原型 B 的可解释配对。

## 最终判断

- **作为一个无消息传递、真实原型驱动的轻量路线，0.751 test 是有价值的。**
- **作为接近约 0.82 的最终主模型，目前不够。**
- **compact relation 值得保留为组件，但不应再把重点放在当前的具体 prototype-pair identity 上。**
- 后续若继续，应回到 official-train 内部开发，不再根据 official test 调整这条配置。

## 机器结果

```text
results/molhiv/compact_official_test_summary_v1.json
results/molhiv/compact_official_test_uncertainty_v1.json
results/molhiv/scaffold_stable_vocabulary_n41127_official_test_seed0.json
results/molhiv/fixed_vocabulary_relation_gates_n41127_official_test_seed0.json
results/molhiv/compact_official_test_run_v1.log
```
