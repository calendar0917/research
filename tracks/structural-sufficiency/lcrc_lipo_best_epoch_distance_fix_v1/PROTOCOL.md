# Protocol — LCRC best-epoch / distance-fix v1

**Round id:** `lcrc-lipo-best-epoch-distance-fix-v1`
**Host:** `res` (pools `res-gpu0`, `res-gpu1`, one A100 each, `cuda:0` in-process)
**Budget:** ≤ 7 h wall clock, ≤ 14 A100 GPU-hours this round (independently charged)

This round fixes two problems of `lcrc_lipo_composition_discovery_v1` and does
not open a new architecture search:

1. **`DIST-TRUE` was mathematically wrong.** `DistanceCorrection.forward()` wrote
   every bucket sum back to the *target* atom index instead of the *root* atom
   index.  For root `i` the historical "context" was

   ```
   c34_i^historical = h_i * (#roots whose D34 bucket contains i) / |D34_i|
   ```

   i.e. a rescaled copy of the atom's own representation (correlation with `h_i`
   is exactly `1.0`, verified in `analysis/cpu_gates.json`).  The corrected model
   is called **`DIST-TRUE-FIXED`** and the historical `DIST-TRUE` checkpoint and
   report are left untouched.

2. **The last-5 soup checkpoint rule is replaced** by a single best-Validation420
   epoch state dict for *every* model and seed.

## 1. Frozen data

Identical to every previous random42 round, imported read-only:

| item | value |
|---|---|
| split file SHA256 | `1834414a6da73e9d2c5942ad157a345f08653110acfc4b9fef83c3064bc233b4` |
| labels file SHA256 | `1da269356e0e89974a7388d1e430d7e810a0ab1056be5cd747a1c8ed3d2009ca` |
| Split seed | `42`, Train3360 / Validation420 / Test420 |
| target mean / scale (Train3360 only) | `2.2081011904761905` / `1.1906281212426577` |
| LCRC cache content fingerprint | `23ec00be4fb7e3cf93967cd62911bd932159b1d7e0b43226b23cf7c6a3718e03` |

Test420 is **read this round**, by explicit authorisation, and every Test number
is labelled `EXPOSED_TEST_EXPLORATORY`: Test420 has been used by earlier rounds
on this split, so it is not a new blind test.

## 2. Models (six, no new architecture)

| model | family key | definition |
|---|---|---|
| `BASE` | `base` | frozen LCRC-A, no correction |
| `ROLEBLIND` | `roleblind` | historical all-atom context on hetero atoms |
| `G2L-LOO` | `g2l_loo` | ROLEBLIND modules, leave-one-out all-atom context per atom |
| `G2L-ALL` | `g2l_all` | ROLEBLIND modules, all-atom context written to all atoms |
| `DIST-TRUE-FIXED` | `dist_true_fixed` | corrected per-root D34/D5 context |
| `DIST-BLIND` | `dist_blind` | same parameters/routing, bucket content replaced by the leave-one-out all-atom mean |

`DIST-TRUE-FIXED` definition (root atom `i`):

```
D34_i = {j : 3 <= dist(i,j) <= 4}
D5_i  = {j : dist(i,j) >= 5, finite}
c34_i = |D34_i|^{-1} sum_{j in D34_i} h_j        (0 if empty)
c5_i  = |D5_i|^{-1}  sum_{j in D5_i}  h_j        (0 if empty)
u_i   = tanh(W_atom h_i)
p34_i = 1[D34_i != empty] tanh(W_34 c34_i)
p5_i  = 1[D5_i  != empty] tanh(W_5  c5_i)
h_new_i = h_i + 1[D34_i or D5_i non-empty] * W_out(u_i * (p34_i + p5_i))
y     = sum_i f(h_new_i) + rho(global, mean h, mean h^2, log1p(n))   # rho reads original h
```

`DIST-BLIND` uses the identical module, parameter names, parameter count and
same-seed initialisation; only the *content* of each non-empty bucket is
replaced by the molecule's leave-one-out all-atom mean.  It therefore isolates
"real D34/D5 structural content" from "bucket existence + extra capacity".

## 3. Training recipe (frozen R3, one deliberate change)

AdamW, batch 128, 360 epochs (27 steps/epoch, 9,720 updates), peak LR `3e-4` →
`1e-5` per-step cosine, weight decay `1e-3`, gradient clip `5.0`, dropout `0.10`
on the final atom-level local representation, MSE on the Train3360-standardised
target, no early stopping, no SSL.

**Checkpoint rule (the one change):** after every epoch the model is switched to
`eval()` and the complete Validation420 RMSE is computed.  The epoch state dict
is kept (on CPU) only if its Validation RMSE is *strictly* lower than the best
seen so far, so a tie keeps the earlier epoch.  After epoch 360 the recorded
best state dict is restored, Validation420 is recomputed from scratch and must
match bitwise, and only then are `checkpoint.pt`, `curve.csv` and `train.json`
written.  No soup, no SWA, no EMA, no post-hoc averaging, no Test-based epoch or
weight selection, no per-model variations of the rule.

## 4. Paired-comparison controls

For a given seed, all six models share

* the BASE module initialisation (built from `x1_models.build_base(seed)` and
  copied in bitwise; verified by `shared_init_and_distance_twin`),
* the Train3360 shuffle order (`torch.Generator(seed + 917000)`, identical
  `batch_order_sha256` for all six families, checked in `verify`),
* the dropout RNG stream (`torch.manual_seed(seed)`; the final RNG-state hash is
  recorded and must be identical across the six families, checked in `verify`),
* the training recipe, the Train-only normalisation and the Validation/Test row
  order.

## 5. Pre-registered analysis

* **Main ranking:** 3-seed mean Validation420 RMSE.
* **Also reported in full:** 3-seed mean Test420 RMSE, means/stds of Validation
  and Test RMSE and MAE, best epoch per run, R².
* **Paired per-seed differences** (never only model means), gain defined as
  `rmse(reference) - rmse(candidate)`, positive = candidate better:
  1. `ROLEBLIND - BASE`
  2. `G2L-LOO - ROLEBLIND`
  3. `G2L-ALL - ROLEBLIND`
  4. `DIST-TRUE-FIXED - DIST-BLIND`
  5. `DIST-TRUE-FIXED - ROLEBLIND`
* **Wins/3** against BASE per seed, on Validation and on Test.
* **Test protocol:** the 18 best-epoch checkpoints are frozen in a manifest
  (SHA256, best epoch, Validation metrics) *before* any Test label is read this
  round; then all 18 are evaluated on Test420 with unchanged weights, and no
  epoch/model/LR/weight is re-selected afterwards.
* **Stability analysis** from the saved curves: seed spread of Validation RMSE
  per model, distribution of best epochs, whether Validation rises after the
  best epoch while train error keeps falling, single-best-seed versus 3-seed
  mean.
* Historical `PNA` Validation RMSE `≈ 0.554034` and Test RMSE `≈ 0.528874` are
  quoted as references only; their training protocol is different and they are
  not matched controls.
* Historical soup numbers are reference-only and are **not** ranked in the main
  table.

## 6. Guards

* CPU gates (`analysis/cpu_gates.json`) must pass with the exact code hashes and
  config hash that are then frozen (`analysis/protocol_freeze.json`); formal
  training refuses to start otherwise.
* Formal training refuses to run on CPU, refuses to overwrite an existing run,
  and is one-shot per arm.
* `run_be.py verify` re-loads every checkpoint and re-checks Validation420,
  the curve argmin, the state hash, the saved per-molecule predictions, the
  Train3360-only normalisation and the absence of any Test access in training.
* `run_be.py eval-test` refuses to run unless the checkpoint manifest is frozen
  and its SHA256 still matches.

## 7. Out of scope

No `PATH-*`, `G2L-SCALAR`, `G2L-LOCAL-CAP`, no new architecture, no LR /
clipping / dropout search, no EMA, no ensembles, no Test-driven retraining.
Stage D (Lipophilicity scaffold shift) is optional and runs only after the 18
random42 runs, the frozen-manifest Test pass and all verification are complete.
