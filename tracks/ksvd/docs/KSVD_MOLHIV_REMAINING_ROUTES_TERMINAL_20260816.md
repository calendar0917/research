# MolHIV Beam8 / KSVD remaining-routes terminal audit

> 日期：2026-08-16  
> official-train 内部 scaffold folds only；official-valid/test 均未评估。

## 最终结论

**NO_STABLE_BEAM_OR_KSVD_PERFORMANCE_ROUTE**

本轮把此前仍有希望的三类路线全部推进到了 matched controls：multi-cover SSL 到 CIN 的迁移、balanced marginal cover、连续 SSL embeddings 上的 KSVD。它们都产生了局部正信号，但没有形成跨 scaffold folds 稳定的 Beam/KSVD 性能增益。

## 1. SSL chemistry embeddings → CIN

只复制 fold-fit、topology-only masked-chemistry checkpoint 的 atom/bond embeddings；CIN layers、readout 和分类头保持 matched random initialization。三折平均相对 CIN：

| initialization | AUC delta | AP delta | 判定 |
|---|---:|---:|---|
| Beam SSL | +.0003 | -.0022 | reject |
| shuffled Beam SSL | +.0092 | +.0081 | general SSL regularization |
| random-BFS SSL | -.0107 | +.0026 | reject |
| balanced-BFS SSL | +.0017 | +.0028 | scaffold-sensitive |

balanced-BFS 在 folds 0/1 的 AUC 分别提升 +.0122/+.0135，但 fold 2 下降 -.0207。Beam alignment 没有稳定优于 shuffled；因此不把 embedding initialization 作为 Beam-specific 贡献。

## 2. Balanced marginal cover

新的 topology-only sampler 从 randomized-BFS candidate pool 中按 unseen nodes/edges、低 occurrence mass 和 patch novelty 贪心选取覆盖。相对 independent random-BFS，三折 masked reconstruction CE 均改善约 .009–.011，准确率改善约 .004–.007。

这是一个真实但较小的上游表征改善；其 CIN 迁移仍在 fold 2 反转。可以保留为 generic cover engineering，但不继续扫描 scorer 权重。

## 3. KSVD on continuous SSL context embeddings

每个节点只保留“自身原子类别被遮蔽”那次 forward 的连续 context embedding。字典仅在 fold-fit reservoir 上拟合，固定 downstream logistic evaluation：

| family | mean AUC | mean AP |
|---|---:|---:|
| **raw continuous** | **.6217** | .0631 |
| PCA | .5299 | .0684 |
| random dictionary | .5907 | **.0811** |
| KSVD | .5691 | .0598 |

KSVD 在三折都获得比 random dictionary 更低的 reconstruction MSE，但只在 fold 2 赢 AUC，三折平均同时输给 raw continuous 和 random dictionary。当前 KSVD 只保留 reconstruction / prototype interpretation 价值，不保留 MolHIV 性能主张。

## 4. 还值得保留什么

- generic multi-cover SSL：可以作为与 Beam 无关的正则化方向，但需要更强 backbone 或更大数据验证。
- balanced topology cover：上游重建稳定小增益，可用于 future SSL sampler control。
- random prototype dictionary：AP 均值最高，若研究目标转向高召回稀有阳性或原型解释，可单独研究；不能称为 KSVD 增益。
- 若要重新打开 Beam，必须提出不同于当前 coverage/chain/balance 的学习型 policy，并先在 topology-only matched random controls 上通过 representation gate。

## 5. Safety and artifacts

- `official_valid_evaluations = 0`
- `official_test_evaluations = 0`
- `tracks/ksvd/code/run_molhiv_beam8_masked_chemistry_gate.py`
- `tracks/ksvd/code/run_molhiv_cin_beam8.py`
- `tracks/ksvd/code/run_molhiv_ssl_context_ksvd_gate.py`
- `tracks/ksvd/results/molhiv/molhiv_beam8_remaining_routes_terminal_summary_20260816.json`

