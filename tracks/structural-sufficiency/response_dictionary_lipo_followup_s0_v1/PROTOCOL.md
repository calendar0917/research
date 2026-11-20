# PROTOCOL — Lipo fusion control + frozen dictionary stress test, seed 0

Status: frozen **before** the formal JOINT unit and before the dictionary stress
test. Parent round: `response_dictionary_small_s0_v1` at commit
`185db2465b600b5260f5974a5a500a22c4ec78eb` (read-only). One new training unit,
one seed, official splits, `test_policy: always` (`DEVELOPMENT_TEST`).

## Questions

1. Does the parent RESP advantage on Lipo also beat a **parameter- and
   init-matched ordinary joint non-linear fusion** (JOINT)? This decides whether
   the product form itself is worth pursuing.
2. When the classes that are actually compressed face a **fixed, stronger
   compression pressure** (K=64 -> K=16), does response-based word selection
   preserve more prediction than latent / frequency selection? This decides
   whether to keep investing in "response-selected words".
3. Read-only: how much independent held-out support does the existing 823-pair
   Lipo edit catalogue really provide?

## Fixed reuse (no new modules, no new recipe)

* Parent config `configs/rd_small_s0.json` verbatim: seed 0, 360 epochs, batch
  128 original molecules (tail kept), AdamW peak `1e-3`, `wd=1e-5`,
  `betas=(0.9,0.999)`, `eps=1e-8`, 5-epoch linear warmup `1e-4 -> 1e-3`,
  per-optimizer-step cosine to `1e-5`, grad clip 5, FP32 with TF32 off,
  dropout 0.10, hidden 96, 3 GINE layers, LayerNorm, latent 16.
* Same fragment caches (`ba6e9a41...`, embedded official cache
  `fe002287...`), same fixed ≤4 views, same per-molecule view-mean squared loss,
  same Train order (`361af4df...`), same canonical seed-0 initial state
  (`ccc8171d...`), 198,262 parameters.
* Every epoch evaluates full Train/Val/Test in raw units (RMSE, MAE); best-Val
  selects the checkpoint (strict `<`, earlier on ties); Test never selects.
* No warm start, no extra bias, no dropout/normalisation change, no extra seeds,
  no retraining of the parent arms, no ESOL work.

## JOINT (the only new training unit)

```
q = tanh(U z_S), k = tanh(V z_C)                      # unchanged, same parameters
h_joint = tanh((U z_S + V z_C) / sqrt(2))
T_JOINT = sum_j w[j] h_joint[j] / sqrt(16)
s_JOINT(C,S) = b(C) + a(S) + T_JOINT
F_JOINT(G) = mean over the same fixed view set
```

Bias-free `concat(z_S,z_C) -> 16 (tanh) -> 1` joint MLP with the shared `w`
readout. It is not ADD (`w_j(q_j+k_j)/sqrt(32)`) and it is not RESP. It uses the
**same** `U`, `V`, `w` tensors, so only the operator differs.

Pre-flight (in the formal job, abort on failure):

| check | rule |
|---|---|
| parameters | JOINT/ADD/RESP identical layout and count 198,262 |
| init state | loads the committed S0 state strictly, hash `ccc8171d...` |
| w=0 predictions | JOINT = RESP = ADD on all 420 Val and 32 Train molecules, `<=1e-6` |
| views/order/data | fragment-cache file SHA, embedded official-cache SHA, order SHA prefix `361af4df`, split sizes 3360/420/420, `max views <= 4` |
| single vs batch | 8 Val molecules scored alone equal the batched scores, `<=1e-5` |
| 3-step smoke | three real optimiser steps finite and `w` moves |
| loop reproduction | the copied loop run as RESP for one epoch reproduces the committed S0 `curve.csv` epoch-1 row, `<=1e-4` on loss/RMSE/MAE/lr |

## Frozen dictionary stress test (no training)

Source: parent Lipo best-Val RESP (epoch 63), all weights frozen. Replay first:
native `F(G)` vs the saved parent predictions, and the three parent K=64
dictionaries rebuilt through the parent code vs the saved dictionary
predictions (must match to float round-off).

Only change: per `tau` class, `K = 16` if Train candidates > 64, else
`min(64, candidates)`. The three classes with 442/81/91 candidates go 64 -> 16;
the other 13 stay exact storage. Expected 88 + 3x16 = 136 words over 702
candidates. Everything else (D0, 16-environment Train panel, farthest-first /
tie rules, `[a(S),q(S)]` latent space, response space, simplex projection with
the frozen 64-step Frank–Wolfe, one-hot prototypes, unsupported rule) is
unchanged. No K sweep, no re-selection.

Reported per dictionary and split: full RMSE/MAE; prediction fidelity
`sqrt(mean((F_D - F_RESP)^2))` in raw units on Val/Test (and Train); the same on
the structure-defined subset of molecules that use one of the three compressed
classes; per-view fidelity split by Train-seen / Train-unseen `S` fragments
(this is not new-structure generalisation); prototype/candidate counts, view
coverage, fallback fraction, and per-class view frequency; alpha simplex
violation, Frank–Wolfe steps used, panel residual before/after; worst samples.

Decision rule (this round's investment reference, not a general claim):
`RESPONSE_SELECTOR_ADVANTAGE_AT_THIS_RATE` only if on **both** Val and Test the
`D_RESPONSE` full-`F` fidelity RMSE is at least 20% below the better of
`D_LATENT`/`D_FREQ`, **and** the actual-property Val RMSE increase over RESP is
at most `0.01 * Train_std`. Otherwise
`NO_RESPONSE_SELECTOR_ADVANTAGE_AT_THIS_RATE`. Small differences are reported
as is; three-way equality is never asserted.

Two cheap response diagnostics on the three compressed classes, frozen panel
and Train frequencies `f_S`: `m_S = mean_context R(S,C)` and
`rho = sum_S f_S sum_C (R-m_S)^2 / (sum_S f_S sum_C R^2 + 1e-12)`, plus the
singular values of the frequency-weighted `R` and of the row-demeaned `R` and
the rank needed for 95% energy. These describe this reference/panel, not the
data's structure.

Frozen-model intervention: replace each view's `k(C)` by the mean `k` of the
Train environment panel of that view's `tau`, keeping `a`, `b`, `q`, `w` and
all fallback paths; evaluate full Train/Val/Test and report the prediction
change and the error change, plus the number of classes without a Train panel.

## Edit replay (read-only)

Catalogue `lcrc_lipo_edit_supervision_public122_s0_v1/analysis/pairs.csv`,
mapped to official rows by RDKit canonical SMILES, labels and split re-derived
from the official row id (never from the catalogue's own split columns), pairs
deduplicated as unordered row pairs. No re-mining, no legality change, no
model-based pair selection. Δpred from the saved **full F** best-Val
predictions of RESP, ADD and JOINT. Reported: pair-MAE, component-equal MAE,
endpoint/component counts per cell, zero-change baseline. Cells with fewer than
20 components are `LOW_SUPPORT` and descriptive only. The 27 Train–Test pairs
are 7 Test molecules in 7 components, not 27 independent held-out experiments.

## Comparison and labels

`Gain_product = RMSE_JOINT - RMSE_RESP` at best-Val for Val (primary) and Test
(follows), plus final and last-10 states. Paired molecule bootstrap 2000 draws,
seed 1703, on the two fixed best-Val states (sample uncertainty only). Top-1 and
top-5 Val molecules removed as an appendix.

`PRODUCT_SPECIFIC_SIGNAL_THIS_CONFIG` iff Val gain >= 0.02, Test gain > 0 and the
Val gain stays positive after removing the single most favourable Val molecule;
`SAMPLE_UNCERTAIN` if the 95% interval crosses zero. Otherwise
`NO_PRODUCT_SPECIFIC_ADVANTAGE_SHOWN`. If JOINT is better, the report says the
current evidence supports general joint computation, not the product operator.

No claim of product-operator superiority in general, no chemical meaning for the
"structural words", no cross-seed stability.

## Budget and stop rules

One new training unit (JOINT, 360 epochs) plus short pre-flight/smoke checks;
frozen analyses on CPU. Hard budget 1 A100·h and 2 h wall including smoke and
failures; soft target 30–60 min. Priority pool `res-gpu1`. Existing parent
result directories are read-only and never overwritten. If a pre-flight check
fails, the unit stops and the failure is recorded instead of relaxing the
configuration; if the dictionary identity check fails, no count is forced to
match.
