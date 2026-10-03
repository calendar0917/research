# EXECUTION — zinc-topology-crossfit-diagnostic-v1

Wall-clock budget: start `2026-10-03 03:15 UTC`, stop-new-compute `04:30 UTC`,
final `04:45 UTC` (90 min).  Remote work goes through `rr` to `res-2`
(pool `res2-cu124`, ≤ 2 GPUs, ≤ 1 GPU-hour).  Local CPU ≤ 8 threads.

Local HEAD at start: `ac8ef5b` (main).  Work branch
`task/zinc-topology-crossfit-diagnostic-v1` created from `ac8ef5b`.
Scientific revision deployed: **`240bd2b`** (two commits: Phase-0 + frozen
protocol/runner, then the frozen sub-fold prep blobs + index manifest).

## 0. Preflight

| step | command | result |
|---|---|---|
| doctor | `rr doctor res-2` | 10 ok / 4 warn / 0 fail; checkout `82b90f3`; pool `res2-cu124` (c05,c06); torch 2.5.1+cu124 |
| deploy | `rr deploy res-2 --pool res2-cu124` | deployed `240bd2b` (incremental from `82b90f3`), 20 s |

## 1. Phase 0 — corrected evidence (local CPU, no GPU)

| step | artifact | why |
|---|---|---|
| 0.1 | `phase0_corrected_metrics.py` → `corrected_historical_metrics.json`, `ERRATA.md` | recompute the double-calibrated historical F/B/D dev metrics from the saved per-row arrays; rebuild group contributions, paired gains and bootstrap |
| 0.2 | `phase0_mechanism_check.py` → `mechanism_check.json` | `named_parameters` / `requires_grad` / optimizer param IDs + one real backward for the Full structural + task dictionaries and the X175/D dictionary/value |
| 0.2b | `cycle_input_decision.json`, `cycle_probe_predictions.csv` (read-only) | confirm the exact-topology-25 class lookup recovers `valid:0172`/`753` and that ExtraTrees severe error is dominated by 0172 |

No historical checkpoint was replayed in Phase 0: the original run saved only
`soup_state_sha256`, so the correction is a **cached-prediction recomputation**.

## 2. Phase 1 — frozen split

`--mode prep` (local CPU, 274.9 s, 8 threads):

* outer split reused verbatim from `zinc_joint_dictionary_decision_v1/prep/split.json`
  (fit 8000 / dev 2000; `fit_idx_sha256`/`dev_idx_sha256` asserted unchanged);
* sub-fold A/B inside the 8000 fit, seed `20261003`, greedy penalty-stratum
  balance over canonical groups: A 4000 (3851/130/19), B 4000 (3851/130/19),
  8000 rows kept, no group straddles, no inconsistent group penalties;
* dev strata verified 1926 / 65 / 9;
* data-dependent prep refit on each base training fold only: standardisers,
  `sdb32` structural dictionary, common subspace `q1`, X175 normalisation;
* frozen `prep/subfold_index.npz` ships the outer + sub-fold indices so the
  remote training path does **not** need the handoff/cycle files.

## 3. Phase 2 — two Full bases (remote GPU)

| # | job | why | depends on |
|---|---|---|---|
| 1 | `ztcd-smoke` | exercise prep apply + train loop + state save (4 epochs) | deploy `240bd2b` |
| 2 | `ztcd-FA` | base on sub-fold A | smoke #1 |
| 3 | `ztcd-FB` | base on sub-fold B | smoke #1 |

All three: FP32, no AMP/DDP, Adam lr 1e-3, wd 1e-5, batch 128, clip 5, 240
epochs, fixed last-5 soup (236–240), per-base median train-fold residual folded
into the reader output bias exactly once.  Both real bases ran concurrently
(1 GPU each, Slurm jobs 55808/55809, node c05, driver 525.85.12).

| arm | train rows | dev raw (unfolded) | dev cal (single) | wall |
|---|---|---|---|---|
| F_A | 4000 | 0.167018 | 0.165144 | 619 s |
| F_B | 4000 | 0.170714 | 0.170444 | 622 s |

Identity: shared non-data init hash identical across folds and equal to the
historical canonical Full init (`a2b15342d453a175…`); `D_fit`/subspace differ by
fold as designed (`644ef7f1…` vs `604a2491…`).

## 4. Phase 3 — fixed CPU readout + coverage (local CPU)

`--mode readout` (4 `ExtraTreesRegressor` fits, 2 per fold; ~61 s):

* `P = [p_base_cal]`, `TP = [raw_topology25, p_base_cal]`, target
  `r = y - p_base_cal` on the *other* sub-fold;
* raw_topology25 read from the graph-only cache
  (`zinc_topology_cache/train_topology_features.csv`, hinge mode); verified
  bit-identical to handoff `topo_raw` (max abs diff 0.0) and matching the
  model-input provenance to 6.1e-5 (float32 inverse-standardisation);
* `analyze.py` builds the averaged main comparison, group table, per-fold
  direction, bootstrap and the frozen gate.

Checkpoint replay: `verify_replay.py` loads each saved soup state, folds the
recorded delta once, and reproduces the first 128 saved dev predictions
(max |Δ| ≤ 1.1e-6), so the reported predictions are checkpoint-replayed, not
cache-only.

## 5. Budget

| resource | used | cap |
|---|---|---|
| GPU | 45 s + 619 s + 622 s = **1286 s = 0.357 GPU-h** | ≤ 1 GPU-h |
| concurrent GPUs | 2 | ≤ 2 |
| local CPU | ≤ 8 threads; prep 274.9 s + readout 61 s + verification ≈ 7 min | ≤ 8 threads |
| wall clock | ≈ 38 min to finalisation | ≤ 90 min |

Compute stopped after the four readout fits; no conditional job was submitted.

## 6. Guardrails honoured

* official test never instantiated / loaded / evaluated;
* official-valid **not** newly evaluated (historical predictions read-only for
  the errata only);
* no D-wide, capacity/λ/s sweep, WD/loss/optimizer search, node rescue, extra
  topology features, third base, extra seed, or full-data confirmation;
* no push, no merge, no history rewrite.
