# PROTOCOL — `zinc_direct_bond_relation_seed0_v1`

Frozen before any O/T dev metric is read. This is an exploratory, fixed-interface screen on a reused internal split, not an independent test. Exactly two formal trajectories: O and T, seed 0. No pooling/dictionary/path-feature/control/model-selection or hyperparameter attempt is part of this round.

## 1. Question and estimand

On the frozen working skeleton, does direct access by the static pair composer to the chemical bond type directly connecting two environments improve generalization when fit examples, initialization, capacity (within the prescribed input expansion), schedule, optimization, and execution regime are matched?

This is **not** a claim that the source model has no bond information: each environment already encodes its internal local bonds. The intervention exposes only the direct edge type between the two represented environments. It is not a dictionary-advantage or a message-passing experiment.

## 2. Frozen source and data

* Source experiment/module and trained/init reference: `tracks/ksvd/results/zinc_local_tuple_fresh_fold_replication_seed0_v1/`; training implementation at commit `a5400de`, preregistered fresh-fold objects at `2e4ee3f` (both are ancestors of this round's source revision).
* Frozen fold: fit SHA256 `2a21cb8771f6e24cfb4a5cc50602cf0b4390db9c4367f8bb910052f9f0aebbcb`; dev SHA256 `270ab4126b0f0413f1f6ff7ada2e9914bbaca91ae50cc65bbd8ca2673d8f9357`.
* Frozen position schedule SHA256 `7b11a529584a8bdaa8e2f17a8c8413cdc6677714ae855339b5984e9fe6938e65`; global graph-ID stream SHA256 `69187f13fba5def82ee2f28765c1c46ac4129844d3f76ac769e018d18ae097d2`.
* Targets, fit-only constants, payload, prep, C6 mask, and train-only encoded/environment caches are inherited. Refit the body input scalers on the same frozen 8000 fit rows using the source builder and verify exact equality with frozen `fresh_prep.npz`.
* Load only official-train encoded/environment caches and raw official-train edge data. Do not call generic `load_split` (it deserializes encoded valid); do not instantiate/load official valid/test. The 2000-row internal dev is a reused train partition, not independent validation.
* Direct-bond beta is the primitive raw 4-D `pair_relation[:,19:23]` one-hot, independently rederived from raw train `edge_index`/`edge_attr` for all train graphs before model input. Confirm one-hot/zero, non-adjacent zero, reverse-edge consistency, graph-local batching and stable global graph order. No scaler is fitted for beta.

## 3. Exactly two arms

Let `r_ij` denote the frozen P1 15-D relation (`pair_relation` columns `0:14` plus `18`); the C6 path-count mask is retained. The raw `path_bond_mean` columns `14:18` remain forbidden.

* **O (matched control):** relation encoder input `[r_ij; 0_4]`.
* **T (candidate):** relation encoder input `[r_ij; beta_ij]`, where `beta_ij` is the primitive four-category one-hot for a direct undirected edge and all zero for a non-edge. Both ordered pair directions receive the same beta.

The first relation Linear expands `15→96` to `19→96`; copy the original 15 weights and bias, initialize the four additional columns exactly zero. All other model components, including the 48-D relation output, stay fixed. Expected parameter count is 297,883 per arm (source M 297,499 + 4×96). O/T states are byte-identical at initialization; both initial functions reproduce source M within `1e-5`.

C6 keeps global atom/bond histogram and relation path-count masking. Beta enters once in the static pair composition and downstream existing graph aggregation. No pair-to-centre write-back, pair-to-centre route, propagation/message passing, extra pair categories, distance×bond crosses, graph bond histogram, auxiliary task/loss, scaler, normalization, LayerNorm, capacity change beyond the mandated four input columns, or WD change.

## 4. Required pre-run checks

All checks are fit-only and must be recorded before formal training:

1. Replay source M init on CPU; verify O/T expanded states, O/T initial predictions, and source-function equivalence (tolerance `1e-5`).
2. Check beta for multiple distinct bond types, separate-graph vs batched rows, non-adjacent zeros, adjacent one-hot, reverse directions, global offsets and restored graph order.
3. Independently derive beta from raw train `edge_index`/`edge_attr`; verify actual 19-D model input, confirm `path_bond_mean` is excluded and beta is unstandardized.
4. Real L1 backward on fit smoke batches: T's four new columns receive nonzero task gradient; O's new-column gradient is exactly zero. Discard all smoke weights.
5. Hooks/witnesses verify environments are computed once, pair composition once, and changing beta cannot change `E` (no write-back).
6. Train/eval/replay use the real 19-D path and never re-mask to 15 dimensions.
7. Freeze/bootstrap/replay use private deterministic schedules; diagnostics do not alter the formal training RNG.

Any failed identity, provenance, data, input-path, or gradient check means `IMPLEMENTATION_FAILED`; no formal O/T score is then interpreted.

## 5. Formal training and budget

For each arm independently from source M's untrained `M_init_state` (never soup warm-start): seed 0; FP32; Adam lr `1e-3`; coupled weight decay `1e-5`; global clip `5.0`; batch 128; 240 epochs; L1 on the same frozen `g`; 15,120 optimizer steps; raw arithmetic soup over epochs 236–240. No dev epoch selection, top-5, early stopping, horizon extension or other training changes.

Execution is only `res-2`, pool `res2-cu124`, CUDA/A100 regime; no AMP or DDP. One GPU per arm, distinct physical GPU UUIDs if concurrent; max 2 GPUs. Remote process threads ≤4 each (combined ≤8); local CPU threads ≤8. Wall time ≤120 minutes from round start; no new compute after minute 90. Total allocation GPU time ≤0.8 GPU-hours. Stop/cancel any unneeded jobs; final status must be terminal. Record Slurm allocation/job/node, GPU UUID/PCI, driver, Python and Torch/CUDA in each run. A pre-formal fit-only GPU smoke is allowed and its state is discarded.

## 6. Evaluation and frozen gate

Only the raw soup is the main model. Per arm compute one fit-only bias `b = median(g_fit − pred_raw_fit)` and `pred_cal = pred_raw + b`; never use dev labels for calibration. Report raw and calibrated fit/dev overall MAE, dev G0 (`k=0`) MAE, each bias, and fit-to-dev gap. Metric is the internal chemical component `g=y-c`, not official ZINC `y`.

Gain is `MAE(O) − MAE(T)`, positive favoring T. The primary endpoint is dev G0 calibrated g-MAE; overall calibrated dev g-MAE is the required practical companion. Freeze before scores: paired row bootstrap, 1000 draws, seed `20261007`; resample only G0 rows for G0 and all dev rows for overall, using the same sampled row indices for O/T within each draw. Verify identical-arm zero and swapped-arm mirror witnesses.

The gate passes only if all three hold:

1. G0 calibrated gain ≥ `0.003`;
2. overall calibrated gain ≥ `0.003`;
3. G0 calibrated paired 95% CI lower bound > 0.

Classification: gate pass → `DIRECT_BOND_INPUT_CANDIDATE`; append `CALIBRATION_DEPENDENT` when the calibrated gate passes but either raw point gain is not positive. If gate fails and both calibrated point gains are positive → `DIRECTIONAL_NOT_CONFIRMED`; if either is clearly negative → `NO_GAIN_FOR_THIS_INTERFACE`. No additional training after a failed gate. Call practical equivalence only if both calibrated G0 and overall CIs lie entirely inside ±0.003. Invalid identities/streams/replay or incomplete jobs are `IMPLEMENTATION_FAILED`/`NOT_COMPLETED`, not a performance verdict.

Also report the four mutually-exclusive groups `k=0, -1, -2, ≤-3` with n, MAE, sum absolute error and total-dev contribution; group contributions must sum to MAE and group gain contributions to overall gain. Report per-arm raw→cal changes and the identity `cal_gain = raw_gain + [(raw_MAE_T−cal_MAE_T)−(raw_MAE_O−cal_MAE_O)]` as accounting, not causal decomposition. Pre-fixed sensitivity: delete O's largest-cal-error dev row; this never replaces the gate.

## 7. Conditional diagnostic and stop rule

Only if T passes the frozen gate, evaluate one input-only dev diagnostic on T's soup: replace each adjacent pair's native beta with that graph's average undirected bond-type one-hot mixture; keep non-adjacent beta zero and preserve the adjacent-pair count. Keep native-beta replay and label-free forward comparison. This distinguishes a pair-specific reliance signal from a graph-marginal-compatible signal, but does not prove what a separately trained marginal-only model would learn. Then stop; write only a design note on whether a separate confirmation is worth buying.

No third run, seed/fold/config/optimizer/WD/encoder/pooling/dictionary/path-feature attempt, threshold change, rescue, or official-valid/test evaluation. Negative results close only this fixed input interface; they do not answer a general “bottleneck” question.
