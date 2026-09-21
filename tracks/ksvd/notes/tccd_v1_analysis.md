# TCCD-v1 — Frozen Composition → End-to-End Dictionary

Pre-registration: `notes/tccd_v1_preregistration.md`.
Lineage: `d863127` → preregistration/code commit `3739c3a` → vectorized
local-training fix `0b6ebc4` → diagnostics `5e3a4cf`.

## Frozen verdict

> **TCCD-v1 supports assignment-sensitive composition and task coupling on the
> frozen reusable vocabulary, but the matched dense latent control decisively
> dominates the sparse task-coupled dictionary. Stop at Gate C. Gate D is NOT
> RUN.**

The TCCD-v0 dictionary was reused. **K-SVD refit: NO.** Official ZINC test was
never loaded or referenced.

## Execution note

The pre-registration required GPU1-only formal experiments. Gate A was completed
on remote A100 GPU1 at commit `3739c3a`. The first remote Gate B attempt was
interrupted before producing a result because the original Python graph-by-graph
batch path was too slow. Per the user's explicit instruction on September 21,
2026, no further server compute was used.

A math-equivalent padded-batch/vectorized path was then added and validated by
17 targeted tests, including exact slow-vs-fast contraction and reconstruction
checks. Gate B, post-B diagnostics, and Gate C were rerun locally on CPU with
the same split, seed, arms, loss, initialization, sparsity, relation operators,
early stopping, and threshold rules. This is a transparent execution-regime
deviation; the local CPU results are matched internal comparisons, not a new
GPU1 benchmark. Gate D was not run.

Interrupted remote run: `tccd-v1-gateB-s0`; no scientific result was read from
that incomplete run.

## Reused artefacts

* Frozen dictionary: `results/tccd_v0/dictionary_gate1.pkl`, shape `(714, 64)`;
  fingerprint `f79596ee0551e4ce1cf2f888bc11b183de3e2949f742838798d80ab8b699ac8a`.
* Canonical train records: `results/tccd_v0/cache/tccd_v0_train_r2_M14_A21_B3_n10000.pkl`.
* Official-train patches: 231,664 total; internal train/dev split
  8,000 / 2,000 using seed `20260922`.
* Frozen values: `K=64`, `s=8`, radius 2, `F=714`, five registered relations.

## Gate A — Frozen Composition

Formal GPU1 result: `results/tccd_v1/gateA_decision.json`, commit `3739c3a`.
The linear reader uses standardized features and fixed ridge `alpha=n_train=8000`
for all arms; the diagnostic alpha curve is not used for selection.

| arm | internal-dev MAE |
|---|---:|
| BAG | 1.078464 |
| REL | 0.750809 |
| REL-SHUFFLE | 0.941584 |

`delta_comp = MAE_REL-SHUFFLE - MAE_REL = 0.190775`.
`MAE_BAG - MAE_REL = 0.327655`.

Threshold: PASS at `>=0.015`; result: **PASS**.

Interpretation: the same frozen sparse-code multiset is materially more useful
when its rows remain assigned to the real relation operators. The composition
signal is not merely a bag-of-environments effect.

## Gate B — Task Coupling

Local CPU matched result: `results/tccd_v1/gateB_decision.json`, commit
`0b6ebc4`.

| arm | best internal-dev MAE | Top-5 soup MAE |
|---|---:|---:|
| FROZEN-D | 1.087302 | 0.912628 |
| TASK-D | 0.938099 | 0.898649 |

`delta_task = 1.087302 - 0.938099 = 0.149203`.
The frozen loss calibration was `lambda_rec=8.446242`.

Threshold: PASS at `>=0.010`; result: **PASS**.

Interpretation: allowing property supervision to update the tied dictionary
improves the sparse model's best internal-dev result by a large margin over the
frozen K-SVD control. This is evidence for task coupling, not evidence that the
sparse inductive bias is uniquely valuable.

## Post-Gate-B continuity / reuse diagnostics

Report-only result: `results/tccd_v1/gateB_diagnostics_seed0.json`.
These metrics do not affect any gate.

| diagnostic | frozen D0 | TASK-D soup |
|---|---:|---:|
| x-space continuity AUC | 0.514375 | 0.514375 |
| code-space continuity AUC | 0.477500 | 0.506250 |
| graded-tail code AUC | 0.542500 | 0.465000 |
| dead atoms | 0 | 32 |
| reused atoms | 64/64 | 31/64 |
| top-8 mass share | 0.287924 | 0.470679 |
| mean coefficient entropy | 1.541724 | 2.025015 |
| atom semantic concentration | 0.610 | 0.560 |

Interpretation: task supervision did not restore a smooth structural manifold.
It slightly raised primary code-space AUC but lowered the graded-tail AUC, and it
substantially reduced reuse / increased concentration and dead atoms. The
predictive gain therefore looks more like task-oriented discriminative
reorganization than recovery of smooth local geometry.

## Gate C — Dictionary Uniqueness

Local CPU matched result: `results/tccd_v1/gateC.json`, commit `5e3a4cf`.

| arm | best internal-dev MAE | Top-5 soup MAE |
|---|---:|---:|
| TASK-D | 0.938099 | 0.898649 |
| DENSE | 0.403933 | 0.387409 |

`MAE_TASK-D - MAE_DENSE = 0.534167`.
The dictionary is worse than dense by far more than the allowed `0.005` margin.
Result: **FAIL / STOP**.

Interpretation: the sparse dictionary's task-coupled representation is not
competitive with the matched generic dense local latent representation. Gate C
therefore rules out a dictionary-specific predictive advantage in this frozen
TCCD-v1 object class.

## Gate D — Absolute performance

**NOT RUN.** Gate C failed, so the preregistered decision tree forbids full-data
TCCD training and any absolute comparison. The canonical GPU1 baseline remains
the reused development reference at valid MAE `0.119818`; no new baseline was
run.

## Answers to Q1–Q4

* **Q1 — composition:** YES. `REL` materially beats `REL-SHUFFLE`; the
  assignment-sensitive composition gain is `0.190775`.
* **Q2 — task coupling:** YES. `TASK-D` beats `FROZEN-D` by `0.149203` in the
  matched local CPU execution regime.
* **Q3 — sparse inductive bias vs dense:** NO. `DENSE` beats `TASK-D` by
  `0.534167`; the dictionary uniqueness gate fails decisively.
* **Q4 — absolute MAE:** NOT TESTED. Gate D was correctly skipped.

## Final scope

### What is now supported

* The reused TCCD-v0 local vocabulary contains assignment-sensitive composition
  information under the frozen `CᵀRC` relation representation.
* Property supervision can reorganize the sparse dictionary enough to improve
  over its frozen K-SVD control.
* The result is obtained without learned message passing, raw graph bypass, or
  handcrafted rescue features.

### What is now ruled out

* The TCCD-v1 sparse dictionary is **not** uniquely competitive with a matched
  dense latent representation.
* The task-coupled dictionary does not recover the smooth local-coordinate
  manifold that TCCD-v0 continuity already rejected.
* Gate D absolute competitiveness is not licensed and was not measured.

### What remains open

Only a genuinely new pre-registration could test a different object class, such
as a permutation-invariant learned local encoder before a dictionary. That is
outside TCCD-v1 and is not authorized automatically. No additional TCCD-v1
rescue, dense-vs-sparse sweep, GNN, reader expansion, loss-weight sweep, or
official-test access is permitted.
