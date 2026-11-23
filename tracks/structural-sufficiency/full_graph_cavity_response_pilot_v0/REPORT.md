# FGCR-v0 REPORT — Full-Graph Cavity Response pilot

Round: `fgcr-v0-full-graph-cavity-response-pilot`
Dataset: `ogbg-mollipo` (official split)
Executed code commit: `f36aa1fc9b1f19609497de77dc266d04c958dcc0`
Run: `results/seed0` (single seed 0, one frozen configuration)

---

## 0. Decision

**`NO_COMPETITIVE_SIGNAL` — `STOP_THIS_IMPLEMENTATION`.**

Best-Valid checkpoint (epoch 96) has official Valid RMSE **0.712894**, above the
frozen `0.70` stop threshold.  No automatic follow-up experiment, second seed,
architecture change, hyperparameter search, ablation or graph-word analysis was
run.  Test was logged every epoch but never used for gradients,
standardization, checkpoint selection, early stopping or the verdict.

---

## 1. Data identity and protocol compliance

| item | value |
|---|---|
| cache | `tracks/structural-query-learning/data/ogbg_mollipo_sq-pilot-cache-v1.npz` |
| SHA-256 | `fe002287fa9c581278bc5430f5b4a5b26b35d9d4d5207d138f15956d93123c7c` |
| Train / Valid / Test | 3360 / 420 / 420 |
| split check | pairwise disjoint; union exactly `0..4199` |
| standardization | Train labels only; mean 2.1621607145, population std 1.2115398401 |
| metric units | original lipophilicity labels; RMSE and MAE |
| atoms / physical bonds | 113,568 / 123,899 |
| multi-component molecules | 1 |
| isolated atoms | 3 (kept, never dropped) |
| unseen Train category | atom column 0 value 33 appears in Valid only; its embedding row exists (Train-derived table size 53) but was never trained |

The cache SHA was verified at every load.  The nine atom categorical fields and
three bond categorical fields were used.  Each physical undirected bond is
stored once, with no self loops; the cache's directed OGB duplicates were
already merged and checked for field equality.  No labels from Valid/Test were
used to fit the scaler or any parameter.

The raw-unit RMSE is the official OGB metric definition; it was additionally
cross-checked against `ogb.graphproppred.Evaluator` in
`analysis/correctness.json`.

---

## 2. Best-Valid checkpoint (epoch 96)

Checkpoint selection: strict minimum of Valid RMSE.  Ties would keep the earlier
epoch; none occurred.

| split | RMSE | MAE | n |
|---|---:|---:|---:|
| Train | **0.472351** | **0.356652** | 3360 |
| Valid | **0.712894** | **0.538783** | 420 |
| Test | **0.709526** | **0.557069** | 420 |

Other best-epoch facts:

| item | value |
|---|---:|
| best epoch | 96 |
| learning rate at best epoch | `1.7020e-4` |
| mean training gradient norm at best epoch | 3.0068 |
| epoch wall time at best epoch | 6.374 s |
| epochs actually run | 136 |
| early stop | yes: best epoch 96, patience 40, stopped at 136 |
| best-Valid checkpoint reload | predictions reproduced to max abs diff `2.86e-6` |

The model is learning a non-trivial signal: Valid RMSE 0.7129 is below the
Train-label standard deviation 1.2115 and below the Train-mean constant
predictor's Valid RMSE 1.2181, and Train RMSE reaches 0.4724.  The Train–Valid gap is large (0.241), so the
single-seed result is consistent with substantial overfitting; this is an
observation, not a mechanism claim.

---

## 3. Per-epoch curves

All 136 epochs were fully evaluated in `model.eval()` on Train, Valid and Test.
Train metrics are full-Train evaluation-mode forwards, not minibatch training
losses.  A compact summary (RMSE only):

| epoch | Train RMSE | Valid RMSE | Test RMSE |
|---:|---:|---:|---:|
| 1 | 1.189352 | 1.197627 | 1.099904 |
| 5 | 1.166102 | 1.186643 | 1.147304 |
| 10 | 0.898691 | 0.963584 | 0.864235 |
| 20 | 0.795377 | 0.885866 | 0.816639 |
| 30 | 0.689271 | 0.799615 | 0.788652 |
| 40 | 0.646711 | 0.776297 | 0.749741 |
| 50 | 0.600875 | 0.760514 | 0.753205 |
| 60 | 0.566311 | 0.752241 | 0.730391 |
| 70 | 0.526424 | 0.745705 | 0.729774 |
| 80 | 0.527076 | 0.739404 | 0.743650 |
| 90 | 0.479527 | 0.732558 | 0.727859 |
| **96** | **0.472351** | **0.712894** | **0.709526** |
| 100 | 0.489700 | 0.737229 | 0.707182 |
| 110 | 0.511843 | 0.761416 | 0.736880 |
| 120 | 0.451427 | 0.743755 | 0.711257 |
| 130 | 0.428613 | 0.728720 | 0.718162 |
| 136 | 0.421679 | 0.725715 | 0.710188 |

The complete 136-row table with Train/Valid/Test RMSE and MAE is in
`results/seed0/metrics.csv` and `analysis/curves_table.md`; the same data are in
`analysis/curves.json`.  The full table is reproduced in the appendix below.

---

## 4. Exact inference coverage and cost

| item | value |
|---|---:|
| graphs with an exact min-fill junction plan | **4200 / 4200** |
| elimination cliques | 113,568 (one per atom) |
| induced-width median / p95 / p99 / max | 2 / 2 / 3 / **3** |
| width distribution | width 1: 5; width 2: 4117; width 3: 78 |
| max clique size | 4 |
| max clique state table | 16 (= 2^4) |
| total clique-state entries | 687,562 |
| graphs with width > 8 | **0** |
| graphs dropped / truncated / sampled | **0** |
| full plan build time on `res` | 4.82 s |

Every molecule has one exact sum-product belief computation per forward pass.
`logZ` and the node `logw` values are read from the **same** calibrated clique
beliefs; there is no second inference pass.  No truncation, sampling,
approximate belief propagation or graph deletion is used.  The independent
min-fill widths and the actual plan agree exactly.

The largest state table is 16, so exact inference is cheap for this dataset.
Per epoch on the A100, exact training plus full three-split evaluation averaged
6.36 s; the training step itself was 5.47 s at the best epoch and the three
split evaluations were 0.71 s / 0.10 s / 0.09 s (Train / Valid / Test).

---

## 5. Numerical health: H, eta and gradients

At the best-Valid checkpoint, on a real batch of 64 Train molecules
(1,725 atoms, one exact belief pass):

| quantity | mean | std | min | max | max abs | non-zero fraction |
|---|---:|---:|---:|---:|---:|---:|
| `H` (64×16) | 1.9232 | 5.2219 | -15.7526 | 17.0910 | 17.0910 | 100% with `|H| > 1e-3` |
| `eta` (1725×16) | 0.0668 | 1.1542 | -6.8841 | 5.5831 | 6.8841 | 99.92% with `|eta| > 1e-3` |

Both are finite and numerically non-degenerate.  `logZ` is finite (range
-1.264 to 77.582 on this batch).  The exact zero-interaction limits pass:

* float64 brute-force comparison on 1–7 node trees, chains, cycles, branches,
  isolated nodes and multi-component graphs: max absolute error **1.07e-14**
  for `logZ` and posterior log-odds;
* with every edge potential set to zero: `H` and `eta` max absolute values
  **1.42e-14** and **1.03e-14** (real batch), and **0 / 3.33e-16** on a
  direct 4-cycle check;
* atom relabelling, bond-direction reversal and bond-storage-order changes:
  prediction differences ≤ **3.55e-15** in float64.

Gradients at the best checkpoint on a real Train batch (standardized MSE loss):

| parameter group | gradient norm |
|---|---:|
| atom embedding | 0.1356 |
| bond embedding | 0.0810 |
| atom potential MLP | 0.7983 |
| bond potential MLP | 1.1428 |
| `eta_norm` | 0.1061 |
| sentence MLP | 1.5198 |
| readout MLP | 1.1521 |
| scalar bias `beta` | 0.0687 |
| node potential tensor `u` | 0.0594 |
| edge potential tensor `j` | 0.1296 |

All listed gradients are finite and non-zero.  The pre-training real-batch
gradient check also passed for every group (including `u` and `j`), and an
AdamW step moved the parameters.  Therefore the exact responses are
differentiable end-to-end and the two potential networks are actually trained.

---

## 6. Distance to the historical official strong baseline

The historical GINE reference on the same official split is approximately
Valid RMSE 0.61–0.64 and Test RMSE 0.66–0.67.  Its training recipe differs, so
this is **descriptive only, not a matched control**.

| reference | Valid RMSE | Test RMSE | note |
|---|---:|---:|---|
| FGCR-v0 best-Valid (epoch 96) | **0.712894** | **0.709526** | this run, official split |
| historical GINE (range) | 0.61–0.64 | 0.66–0.67 | different recipe |
| descriptive gap | +0.073 to +0.103 | +0.040 to +0.050 | FGCR-v0 worse |
| relative descriptive gap | +11.4% to +16.9% | +5.9% to +7.5% | from the range endpoints |

FGCR-v0 is therefore not in the competitive GINE range under this frozen
single-seed configuration.  The D0 RICE/R2 numbers are not comparable to this
official Valid/Test protocol and are not used here.

---

## 7. What this round proves and does not prove

**What it proves (engineering and prediction facts):**

1. The exact full-graph junction-tree cavity-response model can be implemented
   and trained end-to-end on all 4,200 official `ogbg-mollipo` molecules.
   Every graph has an exact plan (max induced width 3, max state table 16); no
   graph was dropped, sampled, truncated or approximated.
2. One calibrated belief pass is sufficient to obtain both `logZ(G)` and every
   node's two-state `logw`; `H` and `eta` are exact, finite, non-degenerate, and
   satisfy the zero-interaction limits.
3. The node and edge potential networks, the sentence MLP and the readout all
   receive finite non-zero gradients; the best checkpoint reload reproduces the
   predictions and metrics.
4. This frozen configuration learns a real signal (Valid 0.7129, Train 0.4724)
   but does **not** reach the historical GINE range.  The frozen verdict is
   `NO_COMPETITIVE_SIGNAL`.

**What it does not prove:**

1. It does **not** show that full-graph cavity responses are generally worse
   than GINE or any other architecture.  This is one dataset, one frozen
   architecture, one seed, and no ablation.  A single failed configuration is
   not a general mechanism result.
2. It does **not** provide a matched comparison to GINE; the training recipes
   differ, and the historical numbers are only descriptive.
3. It does **not** demonstrate autonomous graph-word discovery, causal
   explanation, interpretability, or an original mechanism.  No word types,
   graph-word analysis or mechanism ablation were run.
4. The Train–Valid gap (0.472 vs 0.713) is observed but not attributed; this
   round does not test overfitting controls, capacity, regularization or data
   efficiency.
5. No statistical significance is claimed: one seed, no confidence intervals,
   no bootstrap, no second seed.

---

## 8. Resource accounting and compliance

| item | value |
|---|---:|
| total parameters | **100,926** |
| atom embeddings | 1,280 |
| bond embeddings | 108 |
| atom potential MLP | 11,360 |
| bond potential MLP | 5,488 |
| `eta` LayerNorm | 32 |
| sentence MLP | 32,992 |
| readout MLP | 49,665 |
| scalar readout bias | 1 |
| GPU | NVIDIA A100-SXM4-40GB, driver 550.163.01, torch 2.5.1+cu124 |
| precision | float32; TF32 disabled; no AMP |
| preflight (2 epochs, seed 0) | 13.919 s; projected 200-epoch 0.3866 GPU·h; peak 0.0737 GB |
| formal run | 136 epochs; **864.794 s = 0.2402 GPU·h**; peak **0.0737 GB** |
| formal process wall time | 871.920 s (includes data and 4.82 s plan build) |
| total training/eval GPU time incl. preflight | ~878.7 s = **0.2441 GPU·h** |
| budget | 2 GPU·h; used ~12.2% |

The preflight projection (0.3866 GPU·h) was conservative relative to the actual
formal run (0.2402 GPU·h).  No budget-driven changes were made to batch size,
model width, learning rate, epochs or data.

A final artifact audit (`code/fgcr_audit.py`, output
`analysis/final_audit.json`) re-checks data identity, the topology gate, the
correctness checks, the preflight budget, best-Valid selection, the complete
predictions, the checkpoint and the report.  Status: **PASS**.

Compliance notes:

* code/protocol were committed before deployment and formal execution
  (`f36aa1fc`); the remote executed that commit through `rr deploy`/`rr run`;
* Test metrics were computed and stored every epoch but never used for
  gradients, standardization, checkpoint selection, early stopping or the
  verdict;
* no second seed, architecture search, hyperparameter search, ablation,
  bootstrap, graph-word explanation, D0/D1 comparison or follow-up training was
  run;
* the remote checkout reports `dirty=True` in the run metadata because it
  contains untracked recovery backups and the run's own untracked output files;
  the tracked source is the deployed commit `f36aa1fc`, and no tracked file was
  modified on the remote.

---

## 9. Artifacts

| path | content |
|---|---|
| `results/seed0/best.pt` | best-Valid checkpoint (state dict, config, scaler, metrics, epoch) |
| `results/seed0/metrics.csv` | all 136 epochs: Train/Valid/Test RMSE, MAE, loss, LR, timings, VRAM |
| `results/seed0/predictions_best.npz` | complete predictions for all 4,200 rows at epoch 96 |
| `results/seed0/run.json` | formal run provenance, best metrics, verdict, checkpoint-reload check |
| `results/seed0/train.log` | epoch-by-epoch run log |
| `results/preflight_seed0/` | GPU preflight metrics, JSON and log |
| `analysis/correctness.json` | brute-force, zero-interaction, invariance, gradient, data and single-belief checks |
| `analysis/topology.json` | all-4,200-graph min-fill width and exact-cost preflight |
| `analysis/widths.npy` | per-graph induced widths |
| `analysis/preflight_gpu.json` | GPU cost preflight and 200-epoch projection |
| `analysis/run_formal.json` | formal run result record |
| `analysis/curves.json`, `analysis/curves_table.md` | full per-epoch curves/table |
| `analysis/posthoc_best_checkpoint.json` | best-checkpoint `H`/`eta` statistics and gradient norms |
| `analysis/final_audit.json` | final cross-check of data, topology, checks, budget, selection, predictions, checkpoint and report |

---

## Appendix: complete per-epoch metrics| epoch | Train RMSE | Train MAE | Valid RMSE | Valid MAE | Test RMSE | Test MAE | lr |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.189352 | 0.964541 | 1.197627 | 0.991500 | 1.099904 | 0.911084 | 6.000e-05 |
| 2 | 1.153173 | 0.913101 | 1.166851 | 0.940459 | 1.037996 | 0.842331 | 1.200e-04 |
| 3 | 1.128106 | 0.894959 | 1.152642 | 0.919123 | 1.028625 | 0.834699 | 1.800e-04 |
| 4 | 1.095276 | 0.876804 | 1.127311 | 0.902595 | 1.022186 | 0.830388 | 2.400e-04 |
| 5 | 1.166102 | 0.974117 | 1.186643 | 0.990747 | 1.147304 | 0.956025 | 3.000e-04 |
| 6 | 1.054152 | 0.844498 | 1.092109 | 0.869578 | 0.999239 | 0.813913 | 3.000e-04 |
| 7 | 1.013411 | 0.813616 | 1.055656 | 0.846663 | 0.976741 | 0.793006 | 2.999e-04 |
| 8 | 1.031151 | 0.810039 | 1.094256 | 0.856081 | 0.967579 | 0.780134 | 2.998e-04 |
| 9 | 0.946238 | 0.753319 | 1.021285 | 0.802212 | 0.920005 | 0.735514 | 2.997e-04 |
| 10 | 0.898691 | 0.712703 | 0.963584 | 0.754109 | 0.864235 | 0.696694 | 2.995e-04 |
| 11 | 0.870429 | 0.688189 | 0.947192 | 0.749716 | 0.858231 | 0.691790 | 2.993e-04 |
| 12 | 0.845846 | 0.669797 | 0.915135 | 0.728691 | 0.858441 | 0.691357 | 2.991e-04 |
| 13 | 0.829288 | 0.652686 | 0.900148 | 0.710614 | 0.831390 | 0.663883 | 2.988e-04 |
| 14 | 0.812108 | 0.641324 | 0.873133 | 0.699569 | 0.828826 | 0.662084 | 2.985e-04 |
| 15 | 0.824039 | 0.641095 | 0.904811 | 0.717255 | 0.833460 | 0.666077 | 2.981e-04 |
| 16 | 0.796698 | 0.627806 | 0.869993 | 0.691178 | 0.810330 | 0.645517 | 2.977e-04 |
| 17 | 0.794335 | 0.631338 | 0.863251 | 0.694396 | 0.838151 | 0.671282 | 2.973e-04 |
| 18 | 0.791175 | 0.628024 | 0.861408 | 0.692269 | 0.827989 | 0.659512 | 2.968e-04 |
| 19 | 0.771348 | 0.599825 | 0.850416 | 0.669819 | 0.804106 | 0.630136 | 2.963e-04 |
| 20 | 0.795377 | 0.615312 | 0.885866 | 0.690015 | 0.816639 | 0.647013 | 2.958e-04 |
| 21 | 0.786545 | 0.618846 | 0.882229 | 0.694101 | 0.840620 | 0.671313 | 2.952e-04 |
| 22 | 0.778804 | 0.618912 | 0.862084 | 0.674604 | 0.845990 | 0.670479 | 2.946e-04 |
| 23 | 0.794209 | 0.608621 | 0.890476 | 0.678446 | 0.829562 | 0.651736 | 2.940e-04 |
| 24 | 0.722756 | 0.566854 | 0.813893 | 0.638395 | 0.796140 | 0.621489 | 2.933e-04 |
| 25 | 0.741097 | 0.587888 | 0.832466 | 0.653908 | 0.820605 | 0.647473 | 2.926e-04 |
| 26 | 0.731733 | 0.577049 | 0.823118 | 0.649745 | 0.801146 | 0.625720 | 2.918e-04 |
| 27 | 0.709758 | 0.555490 | 0.814784 | 0.619695 | 0.808369 | 0.630888 | 2.910e-04 |
| 28 | 0.754092 | 0.606436 | 0.846392 | 0.670270 | 0.845489 | 0.672325 | 2.902e-04 |
| 29 | 0.690967 | 0.539385 | 0.798522 | 0.622280 | 0.780146 | 0.605315 | 2.893e-04 |
| 30 | 0.689271 | 0.534209 | 0.799615 | 0.620481 | 0.788652 | 0.620160 | 2.884e-04 |
| 31 | 0.691558 | 0.534966 | 0.811625 | 0.625382 | 0.779266 | 0.598860 | 2.875e-04 |
| 32 | 0.708510 | 0.559376 | 0.814686 | 0.624542 | 0.820347 | 0.633168 | 2.865e-04 |
| 33 | 0.672012 | 0.527069 | 0.791123 | 0.610889 | 0.778375 | 0.600844 | 2.855e-04 |
| 34 | 0.698764 | 0.535697 | 0.833008 | 0.631681 | 0.791699 | 0.617403 | 2.845e-04 |
| 35 | 0.756104 | 0.582187 | 0.879521 | 0.673135 | 0.807566 | 0.631974 | 2.834e-04 |
| 36 | 0.652305 | 0.500141 | 0.789185 | 0.603917 | 0.768669 | 0.594551 | 2.823e-04 |
| 37 | 0.712202 | 0.562110 | 0.828333 | 0.630010 | 0.823838 | 0.638668 | 2.812e-04 |
| 38 | 0.644034 | 0.493695 | 0.786250 | 0.588128 | 0.781661 | 0.597507 | 2.800e-04 |
| 39 | 0.718636 | 0.553057 | 0.854163 | 0.651689 | 0.821595 | 0.631834 | 2.788e-04 |
| 40 | 0.646711 | 0.502854 | 0.776297 | 0.594926 | 0.749741 | 0.580338 | 2.776e-04 |
| 41 | 0.631763 | 0.482717 | 0.777982 | 0.590903 | 0.759699 | 0.588381 | 2.763e-04 |
| 42 | 0.637794 | 0.497844 | 0.780434 | 0.596622 | 0.775532 | 0.593159 | 2.750e-04 |
| 43 | 0.623648 | 0.476297 | 0.772448 | 0.575325 | 0.781365 | 0.587015 | 2.737e-04 |
| 44 | 0.621485 | 0.480703 | 0.770904 | 0.573085 | 0.772289 | 0.583304 | 2.723e-04 |
| 45 | 0.646625 | 0.497588 | 0.804767 | 0.614077 | 0.794553 | 0.599184 | 2.709e-04 |
| 46 | 0.633814 | 0.485311 | 0.778891 | 0.588002 | 0.774328 | 0.583463 | 2.695e-04 |
| 47 | 0.602262 | 0.462718 | 0.753201 | 0.568593 | 0.737003 | 0.564518 | 2.681e-04 |
| 48 | 0.597866 | 0.456796 | 0.759129 | 0.564277 | 0.759969 | 0.578237 | 2.666e-04 |
| 49 | 0.597956 | 0.457777 | 0.751174 | 0.567493 | 0.743052 | 0.573417 | 2.651e-04 |
| 50 | 0.600875 | 0.465530 | 0.760514 | 0.576012 | 0.753205 | 0.578639 | 2.636e-04 |
| 51 | 0.621473 | 0.476582 | 0.781863 | 0.589077 | 0.749153 | 0.576431 | 2.620e-04 |
| 52 | 0.622438 | 0.475206 | 0.771396 | 0.570159 | 0.779661 | 0.600215 | 2.604e-04 |
| 53 | 0.592828 | 0.458616 | 0.747240 | 0.571835 | 0.751668 | 0.578956 | 2.588e-04 |
| 54 | 0.586767 | 0.445364 | 0.760957 | 0.567048 | 0.735367 | 0.570025 | 2.571e-04 |
| 55 | 0.591100 | 0.449516 | 0.760224 | 0.570757 | 0.739673 | 0.578488 | 2.555e-04 |
| 56 | 0.567342 | 0.430667 | 0.745454 | 0.553676 | 0.735471 | 0.556057 | 2.538e-04 |
| 57 | 0.692355 | 0.537098 | 0.855765 | 0.645814 | 0.848284 | 0.653846 | 2.521e-04 |
| 58 | 0.596759 | 0.451291 | 0.765695 | 0.569374 | 0.757566 | 0.581022 | 2.503e-04 |
| 59 | 0.563872 | 0.426423 | 0.742745 | 0.552292 | 0.746763 | 0.573595 | 2.485e-04 |
| 60 | 0.566311 | 0.431328 | 0.752241 | 0.561496 | 0.730391 | 0.558630 | 2.467e-04 |
| 61 | 0.574122 | 0.445180 | 0.767162 | 0.572920 | 0.783204 | 0.583896 | 2.449e-04 |
| 62 | 0.574435 | 0.442252 | 0.752445 | 0.574554 | 0.740335 | 0.578445 | 2.431e-04 |
| 63 | 0.541513 | 0.409414 | 0.740290 | 0.555385 | 0.735801 | 0.563896 | 2.412e-04 |
| 64 | 0.556956 | 0.423909 | 0.758342 | 0.569268 | 0.721136 | 0.555412 | 2.393e-04 |
| 65 | 0.538169 | 0.408300 | 0.743192 | 0.561887 | 0.731434 | 0.552764 | 2.374e-04 |
| 66 | 0.541241 | 0.409413 | 0.741591 | 0.558032 | 0.751482 | 0.578347 | 2.355e-04 |
| 67 | 0.544725 | 0.415715 | 0.746955 | 0.561253 | 0.717145 | 0.554293 | 2.335e-04 |
| 68 | 0.541027 | 0.411403 | 0.759885 | 0.564434 | 0.738776 | 0.562586 | 2.315e-04 |
| 69 | 0.548366 | 0.414897 | 0.744044 | 0.557798 | 0.751295 | 0.582041 | 2.296e-04 |
| 70 | 0.526424 | 0.401350 | 0.745705 | 0.558128 | 0.729774 | 0.550029 | 2.275e-04 |
| 71 | 0.525635 | 0.402060 | 0.737035 | 0.552694 | 0.724153 | 0.560865 | 2.255e-04 |
| 72 | 0.533534 | 0.405917 | 0.749671 | 0.567131 | 0.733883 | 0.558514 | 2.235e-04 |
| 73 | 0.612000 | 0.471110 | 0.778976 | 0.585185 | 0.782612 | 0.611023 | 2.214e-04 |
| 74 | 0.512385 | 0.391533 | 0.739230 | 0.557563 | 0.738592 | 0.559157 | 2.193e-04 |
| 75 | 0.527565 | 0.400662 | 0.747742 | 0.565656 | 0.711827 | 0.551053 | 2.172e-04 |
| 76 | 0.586774 | 0.450499 | 0.787868 | 0.597318 | 0.773580 | 0.608458 | 2.151e-04 |
| 77 | 0.529713 | 0.402907 | 0.751226 | 0.564166 | 0.740294 | 0.582840 | 2.129e-04 |
| 78 | 0.523595 | 0.397586 | 0.738273 | 0.558858 | 0.718026 | 0.551063 | 2.108e-04 |
| 79 | 0.508328 | 0.385172 | 0.745601 | 0.561953 | 0.735304 | 0.553892 | 2.086e-04 |
| 80 | 0.527076 | 0.407948 | 0.739404 | 0.570118 | 0.743650 | 0.567276 | 2.065e-04 |
| 81 | 0.522810 | 0.394928 | 0.738478 | 0.551777 | 0.736345 | 0.570367 | 2.043e-04 |
| 82 | 0.488510 | 0.369543 | 0.730617 | 0.546589 | 0.726139 | 0.554336 | 2.021e-04 |
| 83 | 0.519386 | 0.393334 | 0.756824 | 0.568013 | 0.749563 | 0.573296 | 1.998e-04 |
| 84 | 0.504707 | 0.382520 | 0.752687 | 0.565743 | 0.727793 | 0.555560 | 1.976e-04 |
| 85 | 0.520197 | 0.396134 | 0.750443 | 0.570852 | 0.746363 | 0.582011 | 1.954e-04 |
| 86 | 0.503651 | 0.388627 | 0.731765 | 0.559647 | 0.706656 | 0.548329 | 1.931e-04 |
| 87 | 0.490277 | 0.374390 | 0.727049 | 0.550696 | 0.706587 | 0.540502 | 1.909e-04 |
| 88 | 0.520950 | 0.396060 | 0.755903 | 0.573831 | 0.742127 | 0.575419 | 1.886e-04 |
| 89 | 0.479607 | 0.362930 | 0.724769 | 0.544371 | 0.729631 | 0.563229 | 1.863e-04 |
| 90 | 0.479527 | 0.361670 | 0.732558 | 0.552581 | 0.727859 | 0.561649 | 1.840e-04 |
| 91 | 0.486518 | 0.369558 | 0.733547 | 0.556344 | 0.728738 | 0.562171 | 1.818e-04 |
| 92 | 0.506081 | 0.382606 | 0.740161 | 0.558951 | 0.729886 | 0.571130 | 1.795e-04 |
| 93 | 0.496670 | 0.374760 | 0.740065 | 0.556495 | 0.734647 | 0.568885 | 1.771e-04 |
| 94 | 0.487542 | 0.369211 | 0.732373 | 0.551851 | 0.709666 | 0.557984 | 1.748e-04 |
| 95 | 0.470881 | 0.358153 | 0.721458 | 0.549456 | 0.710359 | 0.550565 | 1.725e-04 |
| 96 | 0.472351 | 0.356652 | 0.712894 | 0.538783 | 0.709526 | 0.557069 | 1.702e-04 |
| 97 | 0.472311 | 0.356879 | 0.716051 | 0.543247 | 0.712522 | 0.561543 | 1.679e-04 |
| 98 | 0.471718 | 0.355861 | 0.727007 | 0.556093 | 0.711645 | 0.553193 | 1.655e-04 |
| 99 | 0.476698 | 0.360658 | 0.728532 | 0.545442 | 0.713736 | 0.561565 | 1.632e-04 |
| 100 | 0.489700 | 0.372223 | 0.737229 | 0.560616 | 0.707182 | 0.550735 | 1.609e-04 |
| 101 | 0.470305 | 0.354697 | 0.724098 | 0.545751 | 0.710244 | 0.555332 | 1.585e-04 |
| 102 | 0.460376 | 0.347323 | 0.729966 | 0.551980 | 0.711795 | 0.554714 | 1.562e-04 |
| 103 | 0.484819 | 0.367043 | 0.736491 | 0.557971 | 0.724707 | 0.568637 | 1.539e-04 |
| 104 | 0.453327 | 0.342697 | 0.723241 | 0.544177 | 0.711555 | 0.546478 | 1.515e-04 |
| 105 | 0.463758 | 0.347623 | 0.725452 | 0.548740 | 0.719162 | 0.558070 | 1.492e-04 |
| 106 | 0.458343 | 0.348876 | 0.719992 | 0.548260 | 0.718633 | 0.561828 | 1.469e-04 |
| 107 | 0.454246 | 0.343365 | 0.731351 | 0.553616 | 0.715360 | 0.553207 | 1.445e-04 |
| 108 | 0.470126 | 0.354035 | 0.740664 | 0.554504 | 0.728053 | 0.569503 | 1.422e-04 |
| 109 | 0.449600 | 0.337989 | 0.730289 | 0.548891 | 0.723493 | 0.557989 | 1.399e-04 |
| 110 | 0.511843 | 0.392444 | 0.761416 | 0.576577 | 0.736880 | 0.587755 | 1.376e-04 |
| 111 | 0.448226 | 0.339473 | 0.727094 | 0.549960 | 0.709141 | 0.551860 | 1.352e-04 |
| 112 | 0.465302 | 0.353890 | 0.736232 | 0.557371 | 0.717328 | 0.557412 | 1.329e-04 |
| 113 | 0.466405 | 0.353317 | 0.742420 | 0.556158 | 0.709524 | 0.555757 | 1.306e-04 |
| 114 | 0.458293 | 0.346319 | 0.735908 | 0.554278 | 0.726673 | 0.568596 | 1.283e-04 |
| 115 | 0.462992 | 0.351291 | 0.748001 | 0.563074 | 0.728341 | 0.564555 | 1.260e-04 |
| 116 | 0.441103 | 0.332866 | 0.721472 | 0.541479 | 0.711948 | 0.552230 | 1.238e-04 |
| 117 | 0.439057 | 0.331036 | 0.724109 | 0.545208 | 0.713305 | 0.551942 | 1.215e-04 |
| 118 | 0.453429 | 0.342100 | 0.736502 | 0.552332 | 0.716757 | 0.562552 | 1.192e-04 |
| 119 | 0.446598 | 0.336725 | 0.731753 | 0.552628 | 0.714508 | 0.556222 | 1.169e-04 |
| 120 | 0.451427 | 0.341652 | 0.743755 | 0.565637 | 0.711257 | 0.549820 | 1.147e-04 |
| 121 | 0.441472 | 0.331865 | 0.735693 | 0.555325 | 0.718661 | 0.557851 | 1.125e-04 |
| 122 | 0.435908 | 0.330106 | 0.735189 | 0.554388 | 0.718284 | 0.551119 | 1.102e-04 |
| 123 | 0.432925 | 0.327405 | 0.721613 | 0.541105 | 0.702358 | 0.545055 | 1.080e-04 |
| 124 | 0.430315 | 0.324601 | 0.725852 | 0.546101 | 0.705796 | 0.546796 | 1.058e-04 |
| 125 | 0.430989 | 0.325243 | 0.723160 | 0.543683 | 0.715886 | 0.555659 | 1.036e-04 |
| 126 | 0.457587 | 0.347102 | 0.743070 | 0.556435 | 0.737333 | 0.576152 | 1.014e-04 |
| 127 | 0.433975 | 0.327762 | 0.721740 | 0.541297 | 0.714859 | 0.554484 | 9.928e-05 |
| 128 | 0.428496 | 0.324144 | 0.733292 | 0.554126 | 0.727022 | 0.558137 | 9.713e-05 |
| 129 | 0.427544 | 0.323393 | 0.727037 | 0.544473 | 0.714162 | 0.552121 | 9.500e-05 |
| 130 | 0.428613 | 0.325354 | 0.728720 | 0.550899 | 0.718162 | 0.552136 | 9.288e-05 |
| 131 | 0.431354 | 0.324564 | 0.732399 | 0.547565 | 0.721979 | 0.560669 | 9.078e-05 |
| 132 | 0.421726 | 0.318305 | 0.729798 | 0.550367 | 0.718729 | 0.551262 | 8.869e-05 |
| 133 | 0.423584 | 0.319443 | 0.722295 | 0.541853 | 0.708764 | 0.551796 | 8.662e-05 |
| 134 | 0.423970 | 0.319608 | 0.731771 | 0.549890 | 0.710506 | 0.546945 | 8.457e-05 |
| 135 | 0.422608 | 0.318290 | 0.732810 | 0.548859 | 0.718381 | 0.554252 | 8.254e-05 |
| 136 | 0.421679 | 0.318866 | 0.725715 | 0.548363 | 0.710188 | 0.549392 | 8.052e-05 |
