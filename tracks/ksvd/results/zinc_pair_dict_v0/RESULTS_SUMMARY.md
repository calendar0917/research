# PSD-v0 — Results summary (git-tracked minimal sufficient evidence)

Full evidence: `results/zinc_pair_dict_v0/*.json` (git-ignored, local).
Round brief: `notes/zinc_pair_dict_v0_preregistration.md`.
Analysis: `notes/zinc_pair_dict_v0_analysis.md`. Commit `e1bef7c`, A100, seed 0.
Official ZINC valid/test never used (train-only screen).

## Verdict

> **Pair-state structural representation is not viable (dominated by the raw
> node baseline).** STOP the pair encoder; dictionary (Q2) not started.

## Parameters (real ZINC categories: 21 atom / 3 bond)

| arm | params |
|---|---:|
| `raw` B-Null-Raw d=64 L=4 | 55 681 |
| `pair` d=64 L=3 | 76 609 |
| `raw_wide` d=76 L=4 (param-matched) | 77 977 |

## Stage 1 — 128-graph overfit (train only)

| arm | best train MAE (batch 16, 300 ep) | gate ≤0.15 |
|---|---:|---|
| pair | 0.0626 | PASS |
| raw | 0.0694 | PASS |

(The canonical batch-128/60-epoch probe = only 60 steps; both arms failed 0.15
purely from the step budget → one targeted revision to batch 16, 300 epochs.)

## Stage 2 — train-only screen (2048 dev / 512 monitor, seed 20260922, batch 64)

| arm | params | best monitor MAE | soup monitor MAE | epochs | wall (s) |
|---|---:|---:|---:|---:|---:|
| pair (80 ep) | 76 609 | 0.5177 | 0.5043 | 80 | 133 |
| pair (200 ep, early stop) | 76 609 | 0.5038 | 0.4955 | 143 | 149 |
| raw_wide | 77 977 | 0.4115 | 0.4094 | 80 | 72 |
| raw | 55 681 | 0.4152 | 0.4105 | 80 | 84 |

`pair − raw_wide` = **+0.106** (80 ep) / **+0.088** (best 200 ep); `pair − raw` =
+0.103 / +0.089 ≫ 0.005 practical-effect scale. Pair train@last 0.195 vs monitor
0.513 → overfitting; raw train 0.255 vs monitor 0.415. Pair ~1.8–2.1× slower.

## Mechanism

* Pair-field effective rank 8.5–11.7 (L1–2), 4.6 (L3): not degenerate.
* Graph readout rank: pair `u_G` 1.46; raw sum-pool vector 2.46, raw_wide 2.18 →
  **low rank is shared, not the differentiator; no numerical pathology unique to
  pair.**
* Synthetic C6 vs 2·C3 (1-WL-equivalent): raw node multiset gap 0.0, pair-state
  projection gap 0.051, `u_G` gap 0.043 → mechanism is active.

## Q2 (dictionary)

Not started (Q1 failed). No `z_G`, no matched controls, no compression/budget
claim.

## Cannot say

* Only screens this raw-input pair formulation at this capacity/budget; not
  "pair encoders cannot work on ZINC".
* Official valid/test untouched; full-10000 pair run **not** performed (brief §24).

## Revisit if

A new pre-registration proposes a different object class / parameterisation, or
supplies full-10000 / effective-capacity-matched evidence that the 2048
overfitting is a pure data-size effect, or re-scopes the dictionary question onto
the working `raw` / `raw_wide` representation.
