# TCCD-v1 Gate A — Frozen Composition

Pre-registration: `notes/tccd_v1_preregistration.md`.
Result: `results/tccd_v1/gateA_decision.json`.
Commit: `3739c3a`.
Device: remote A100 **GPU1**. Official test: never loaded.
K-SVD refit: **NO**; reused TCCD-v0 `D0`.

## Protocol

* Frozen `D0`, `K=64`, `s=8`, radius 2, five preregistered relations.
* One-shot `OMP(x; D0, s=8)` codes cached once.
* Whole-graph BAG / REL / REL-SHUFFLE features precomputed once.
* Internal split: 8,000 train / 2,000 dev, split seed `20260922`.
* Reader: standardized features + ridge with fixed `alpha=n_train=8000`, identical across arms.
* REL-SHUFFLE preserves the code multiset and relation matrices, but permutes code rows without permuting relations.

## Results

| arm | internal-dev MAE |
|---|---:|
| BAG | 1.078464 |
| REL | 0.750809 |
| REL-SHUFFLE | 0.941584 |

`delta_comp = 0.190775`; `MAE_BAG - MAE_REL = 0.327655`.

Threshold: PASS at `delta_comp >= 0.015`.

**Gate A: PASS.**

Conclusion: assignment-sensitive composition is strongly predictive beyond the shuffled control, and REL also improves over BAG. Proceeding to Gate B was authorized. No seed-1 was needed.
