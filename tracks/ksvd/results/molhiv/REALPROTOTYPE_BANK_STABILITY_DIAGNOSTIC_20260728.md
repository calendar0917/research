# Real-prototype bank stability diagnostic（2026-07-28）

仅使用 8k subset 的 official-train 三个 scaffold outer folds；official valid/test evaluation 均为 0。

## Aggregate

- prototype ensemble fold AUC: `[0.7482315212011219, 0.677951474853222, 0.7584730100057359]`；mean `0.728219`。
- ensemble 相对单 bank AUC 均值的 fold gains: `[0.016153481273717296, 0.008386975077220638, -0.0017292290697425994]`；mean `+0.007604`。
- bank 间 optimal-matching prototype cosine mean: `0.557700`。
- bank prediction pairwise Pearson mean: `0.677358`。
- heldout − fit graph coverage mean: `-0.000231`。
- fit/heldout prototype-usage JS mean: `0.001544`。
- heldout scaffold-unweighted coverage q10 mean: `0.623112`。

## Fold summary

| Fold | Member AUCs | Ensemble AUC | Ens−member mean | Mean graph prob std |
|---:|---|---:|---:|---:|
| 0 | 0.7122/0.7526/0.7315 | 0.7482 | +0.0162 | 0.00875 |
| 1 | 0.6709/0.6647/0.6730 | 0.6780 | +0.0084 | 0.01470 |
| 2 | 0.7621/0.7538/0.7646 | 0.7585 | -0.0017 | 0.01307 |

## Interpretation rule

- 若 bank matching cosine 高但 predictions 分歧大，主要问题在 downstream optimization。
- 若 bank matching cosine 低且 ensemble gain 大，prototype identity/coverage 是主要方差源。
- 若 heldout coverage 明显下降或 usage JS 很高，应优先做 scaffold-stable vocabulary。
- 若 coverage 稳定但 prediction 分歧仍大，应优先研究 occurrence composition/readout。
