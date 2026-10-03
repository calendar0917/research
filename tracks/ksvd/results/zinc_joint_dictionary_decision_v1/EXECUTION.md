# EXECUTION — zinc-joint-dictionary-decision-v1

Wall-clock budget: step start `2026-10-03 00:48:19 UTC`, compute cutoff
`06:03 UTC`, final `06:48 UTC`.  Local CPU work is capped at 8 threads / one
heavy worker; remote work goes through `rr` to `res-2` (pool `res2-cu124`,
≤ 2 GPUs, ≤ 6 GPU-hours).

Every launched remote job is listed with **why it was launched** and **which
evidence it depended on**.  Conditional work is never submitted before its
precondition is satisfied.

## 0. Environment / preflight

| step | command | result |
|---|---|---|
| doctor | `rr doctor res-2` | 10 ok / 4 warn / 0 fail; checkout `8657343edc62`; pool `res2-cu124` (c05,c06); torch 2.5.1+cu124 |

Local HEAD at start: `7ab177b` (main).  Work branch
`task/zinc-joint-dictionary-decision-v1`, created from `7ab177b`.

## 1. Stage A — severe-cycle input audit (read-only)

Script: `tracks/ksvd/results/zinc_joint_dictionary_decision_v1/cycle_input_audit.py`
(local, 8 threads, 20.4 s, exit 0).

* A1 replayed `N0_s0 soup_state.pt` → `raw = 0.11073645`, `cal = 0.11120612`
  (published `0.11120613`), train-fitted bias `−0.01204258`; per-row
  re-accounting of the 0 / −1 / ≤ −2 strata sums to `0.11120613` exactly.
* A2 built the exact topology25 equivalence classes for the 70 query keys ×
  {`topo_raw`, `topo_model_input`} and compared against all 10000 train rows.
* A3 ran a molecule-level (canonical group) 2-fold train-only probe:
  exact-class median lookup and a fixed `ExtraTreesRegressor`.

Deliverables (force-added, see below): `cycle_input_classes.csv`,
`cycle_class_members.csv`, `cycle_probe_predictions.csv`,
`cycle_input_decision.json`.

No Stage-A statistic is consumed by X175, the training target, the routing or
the model.

## 2. Stage B — one fixed prototype

Implementation: `tracks/ksvd/experiments/luyin16/zinc_joint_dictionary_decision_v1.py`.

* `F` = canonical `LatentScaleSEM108` Full (fresh, seed 0, scale_seed 0),
  408,651 params, `L1 + H1_LAMBDA·reconstruction`.
* `B` = `X175 → Linear(175,256) → SiLU → Linear(256,144)`, 82,064 local params.
* `D` = shared sparse dictionary 175×256 (K=256, s=64, 10 tied-IHT steps) →
  `Linear(256,144)`, 81,808 local params, `L1 + lambda_rec·relative-recon`.
* B/D replace the **entire** old local-environment module; pair projection,
  relation/distance, unary/pair moments, C6 mask, global/topology branches and
  the reader (`R=814`) are shared and initialise to the **same** tensors.

### 2a. Fold-internal prep (local CPU, one heavy worker, 3.5 min)

`python -m ...zinc_joint_dictionary_decision_v1 --mode prep` computed, **on the
fit part only**:

* Sem108 / global-context / anchor / topology standardisers (invert the
  all-train standardiser to raw, refit per column on the fit roots);
* the structural dictionary `sdb32` (K-SVD, 10 epochs, seed 20260924) and the
  common subspace `q1`, both refit on fit `phi`;
* X175 B1 normalisation (per-column mean/std, floor 1e-3; block RMS rescale);
* the D shared dictionary (seed 20261003, 5 passes, batch 4096, Adam 1e-3,
  relative-reconstruction objective; 230 steps, loss 0.3385 → 0.0136);
* `lambda_rec` (one-shot, 4 fixed fit batches): ratios ≈ 5.0e-3,
  `lambda_rec = 20.0048`.

Artifacts: `prep/fold_objects.npz`, `prep/*.json`.  Because
`tracks/*/results/**` is gitignored, the prep blob was **force-added** so it
reaches the remote compute copy via `rr deploy` (git bundle).

### 2b. B3 availability checks (local CPU)

`--mode checks` → `availability_checks.json`, `mechanism_ok = true`:

* dims: X175 = 175, E = 144, reader input 814;
* params: F 408,651 = shared 97,313 + F local 311,338; B/D local 82,064 / 81,808;
* identical backend init hash `a2b15342…` for F/B/D;
* batch-composition invariance max |Δ| = 0;
* D reconstruction: fit 0.0126, held-out 0.0126 (blocks φ 0.023, sem 0.081,
  size 0.0011; all ≤ 0.50);
* α active (`max|α| = 3.20`); task+rec gradients reach `dictionary` (0.152) and
  `value` (0.449); finite outputs.

## 3. Deploy + remote smoke

| # | job (experiment) | commit | why launched | depends on |
|---|---|---|---|---|
| 1 | `zjd-smoke-D` (fail) | `6ac9b6c` | first CUDA smoke | deploy ok |
| 2 | `zjd-probe*` | `6ac9b6c` | check remote inputs exist | #1 |
| 3 | `zjd-smoke-D` (fail) | `69ba8e9` | smoke after shipping prep | #2 found env cache |
| 4 | `zjd-smoke-D` (ok) | `82b90f3` | 6-epoch smoke exercising the soup window | #3 fixed locally |

The remote did **not** have `zinc_dictionary_real_data_handoff` /
`zinc_long_cycle_audit` / `anchor_stats.json`, so the prep blob (which already
contains every fold-internal object, the split indices and `lambda_rec`) is the
only input shipped.  A separate `zjd-verify` job recomputes the remote X175 and
compares it to `prep_meta.json`'s hashes.

## 4. Stage C — fixed 8000/2000 molecule-level split, seed 0, 240 epochs

Split: `canonical_group_id`, seed 20261003, 20 % dev, stratified by raw cycle
penalty.  Result: 8000 fit / 2000 dev; dev strata 1926 (penalty 0) / 65 (−1) /
9 (≤ −2); no group straddles; no inconsistent group penalties.

| # | job | why | depends on |
|---|---|---|---|
| 5 | `zjd-F-s0` | canonical control | smoke #4 |
| 6 | `zjd-B-s0` | non-dictionary encoder control | smoke #4 |
| 7 | `zjd-D-s0` | the prototype under test | smoke #4 |

All three submitted together (2 concurrent GPUs; the third queues).  Every arm:
Adam lr 1e-3, coupled wd 1e-5, batch 128, clip 5, FP32, no scheduler/AMP/DDP,
fresh init, fixed last-5 soup (236–240), fit-only median bias.

C3 gate is evaluated **after** these three finish; seed 1 is only bought if the
gate passes (see `DECISION.md`).
### 4.1 Outcome

| arm | dev raw | dev cal | wall |
|---|---|---|---|
| F | 0.116365 | 0.118018 | 1363 s |
| B | 0.127340 | 0.128883 | 720 s |
| D | 0.124460 | 0.124505 | 941 s |

`zjd-verify` (submitted alongside; ran after D) recomputed the remote X175 and
matched the local `prep_meta.json` hashes exactly (`MATCH True`), so the three
arms consumed exactly the intended fold-internal inputs.

C3 gate: `D − F total gain = −0.006487` (needs ≥ +0.003) → **FAIL**.  Seed 1 and
Stage D were **not** bought; no conditional job was ever submitted.  See
`REPORT.md` §4 and `DECISION.md`.

Remote GPU total ≈ 0.95 GPU-hours (F 1363 s + B 720 s + D 941 s + smokes/probes
≈ 300 s) against the 6 GPU-hour ceiling.

## 5. Finalisation

* `REPORT.md`, `DECISION.md`, `ERRATA.md`, `EXECUTION.md` written in
  `tracks/ksvd/results/zinc_joint_dictionary_decision_v1/`.
* Stage A CSV/JSON deliverables, the Stage B availability report, the three
  seed-0 run files and the gate file are force-added (the `tracks/*/results/**`
  tree is gitignored) and committed on
  `task/zinc-joint-dictionary-decision-v1`.
* No push, no merge, no history rewrite.
