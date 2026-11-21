# PROTOCOL — direct complete local objects, seed 0 (`direct_local_object_s0_v1`)

Pre-registration written before any new model was trained. T0 =
`2026-11-21T21:23:58+08:00`. The task brief fixes the question, model, recipe,
checks, budget and stop rules; this file only records them and the one allowed
time branch.

## Question

On official `ogbg-mollipo` (3360 / 420 / 420), can the **complete 5/6-atom local
induced object**, read directly by an ordinary MLP and pooled over its symmetric
numberings, predict better than a **matched 3/4-atom object** with the same
network, parameters and number of updates — and how far is it from the
same-data GINE reference?

This is a feasibility reference, not a new method claim. It contains no word bag,
no learned word hidden vectors, no teacher, no zero-initialised residual, no
message passing, no fingerprints and no GINE component.

## Objects (frozen)

Every object is `(molecule, root, atom_set)` with the **complete induced**
subgraph on `atom_set`, including all internal physical bonds and all raw
attributes. The root is in the set and is pinned to slot 0.

| arm | core objects | small-component padding |
|---|---|---|
| `DIRECT_S56` | every unique 5/6-atom `(root, atom_set)` in the old `swf_sent.npz` (3 316 287 rows full corpus) | complete connected components of size 1–4, every component atom as root |
| `MATCHED_W34` | every unique 3/4-atom `(root, atom_set)` in the old `swf_base.npz` (1 593 359 rows full corpus) | complete connected components of size 1–2, every component atom as root |

No cover/frame-pair multiplicity, no sampling, no truncation, no new word
selection rule. Old padding is truncated by `sent_n` / `word_n` before use.
Objects are sorted by `(size, atom_set, root)` inside each molecule but the sort
is not part of the canonicalisation.

## Symmetric numbering (frozen)

For object size `k` and the fixed slot-pair order
`(01,02,03,04,05,12,13,14,15,23,24,25,34,35,45)`, enumerate all `(k-1)!`
permutations with the root fixed at slot 0, define the adjacency bit integer
`b(A) = sum_e A[pair_e] 2^e`, take the minimum over permutations, and keep **all**
permutations that attain it. Each retained permutation is a frame carrying the
complete atom, bond and mask attributes. The object output is the
`1/n_frames`-weighted average of the frame outputs.

A precomputed label-free LUT over the `2^(k(k-1)/2)` adjacency patterns (k=1..6;
32 768 patterns for k=6) supplies the canonical bits and winning permutation
lists. No per-object graph-isomorphism call is made; no single arbitrary
canonical permutation is selected.

## Network (fixed, identical for both arms)

* atom: 9 categorical fields, embedding 8 each → 72; `AtomMLP 72→64→32` (SiLU).
* bond: 3 fields, embedding 8 each → 24; `BondMLP 24→32→16` (SiLU).
* frame input 474 = 6×32 atoms + 15×16 bonds (masked) + 6 node mask + 15 edge mask
  + 15 adjacency + 6 size one-hot; `FrameEncoder 474→256→128` (SiLU, LayerNorm),
  **no dropout in the frame path**.
* object `h_o = sum_f weight_f * FrameEncoder(x_f)`, weights sum to 1.
* two buckets per arm: `DIRECT_S56` k≤5 / k=6; `MATCHED_W34` k≤3 / k=4.
  `u_b = sum_{o∈b} h_o / mu_b`, `v_b = sum_{o∈b} h_o / max(1,n_b)`,
  `mu_b = max(1, mean_Train n_b)`.
* graph input 516 = `[u0,v0,log1p(n0), u1,v1,log1p(n1), log1p(n_atoms), log1p(n_bonds)]`;
  `GraphHead 516→256→128→1`, SiLU + dropout 0.10 after the first two layers,
  **normal non-zero output initialisation**.
* official OGB category dimensions; Train-unseen embedding rows are zeroed.
* expected parameter count 329 705; both arms exactly the same.

Supervision is one molecule-level MSE on standardised targets. All gradients
flow through the object average and frame encoder. Per-object Python network
calls are forbidden; forward/backward are batched. Frame inputs are deduplicated
exactly by the complete raw input within a step and across the cache (object rows
are not deduplicated); duplicate-frame computation may not be reused across
optimizer steps.

## Recipe (frozen, one seed)

seed 0 only; python/numpy/torch seeded, cudnn deterministic; TF32 enabled and
recorded; FP32. batch 64, eval batch 64. AdamW LR 1e-3, wd 1e-4, clip 5. Five
epochs linear warmup from 1e-4, then cosine to 1e-5. Common horizon 240 unless
the one cost branch below changes it to 160. Same pre-generated Train molecule
order (`numpy.default_rng(20261121)`), no early stopping per arm.

Every epoch evaluates Train, Valid and Test in `eval()`/`no_grad()` with dropout
off, using the Train-only target scaler and original label units. Checkpoint
selection is Valid RMSE only. Test is visible throughout by explicit user
authorisation but is not used to change the recipe or pick epochs.

## Checks (gating, before formal training)

At most 8 minutes, on a normal non-zero output head:

1. official split/row_id, Train-only scaler, molecule-local bond numbering, old
   core counts, padding truncation, atom coverage, frame-mask consistency;
2. LUT vs brute force (k=1..5 exhaustive, k=6 fixed 500-mask sample);
3. re-labelling invariance on 2 synthetic graphs (6-ring with a 2-atom branch)
   and 3 real molecules per arm: complete raw frame-input+weight multisets and
   whole-net output; CPU FP64 ≤1e-10, optional GPU FP32 ≤1e-4;
4. single vs shuffled multi-molecule batch equality, and scatter vs explicit
   object-loop pooling;
5. one real Train batch (64 molecules): finite non-zero gradients in atom, bond,
   frame encoder and graph head, and a real optimizer step that moves the frame
   encoder.

## Budget and the one time branch

* Hard wall 120 min from T0; no new training launched after T0+90 min; training
  saved/stopped by T0+105 min; at least 15 min reserved for delivery.
* At most 2 planned neural units (the two arms), at most 1 engineering rerun,
  ≤2 GPU·h, one process per physical GPU. GPU1 (`7c4f57e5-...`) runs
  `DIRECT_S56`; GPU0 (`5b51f95e-...`) runs `MATCHED_W34`.
* Each arm is two launches of the same run, not a restart: `stage-a` (epochs 1–5)
  saves model/optimizer/RNG; after a common horizon decision, `continue` resumes
  at epoch 6.
* Cost estimate = median epoch-3..5 “Train + three-split eval” seconds × 240 ×
  1.25. If >45 min, the **common** horizon becomes 160 (cosine recomputed; warmup
  unchanged, no restart). If 160 is still >45 min, at most 10 min of pure
  engineering optimisation; otherwise `COST_BLOCKER`, save partial curves and
  stop. No object/frame reduction and no per-epoch eval removal.
* If `res` is unreachable after two attempts within 3 min, finish local-only
  implementation/cache and report the block; do not switch to long local CPU
  training.

## Analysis (fixed)

Main table includes both new arms and the read-only
`WORD_PREFIX` / `BAG_CONT` / `GINE_RAW_REF` coordinates. Report
`e_L−e_S`, `e_L−e_P`, `e_L−e_G` and the Test counterparts. 1 000 molecule-level
paired bootstrap resamples (same indices for both models) on Valid and Test; this
is sample-composition sensitivity only. Report object/frame/dedup/coverage
counts, all hashes, actual horizon, epoch times, wall and GPU·h. On each best
state, evaluate once with all `h_o` set to zero and only the four structural
counts kept; report prediction RMS and RMSE change. No count-only retraining.

Decision gate (first applicable row, completed runs only):

| condition | code |
|---|---|
| `e_L ≤ 0.67` and `e_S−e_L ≥ 0.015` | `LARGER_OBJECT_PROMISING` |
| `min(e_L,e_S) ≤ 0.67` | `DIRECT_LOCAL_PREDICTION_PROMISING_SCOPE_UNRESOLVED` |
| `min(e_L,e_S) > 0.67`, `e_S−e_L ≥ 0.015`, `e_L < e_P−0.01` | `CONTEXT_SIGNAL_WITH_PREDICTION_GAP` |
| `min(e_L,e_S) ≥ e_P−0.01` | `STOP_THIS_DIRECT_FAMILY` |
| otherwise | `MODEST_LOCAL_SIGNAL_STOP` |

Incomplete arms, failed checks or a cost block are engineering/budget outcomes,
not evidence that structural objects fail. Add `SPLIT_CONFLICT` if a Valid gain
comes with a >0.03 Test regression against the old word bag or the matched
control. No new seeds, datasets, widths, objectives, ESOL runs or mechanism
sweeps; this round closes after the result table.

## Execution record (appended after the fact)

* T0 `2026-11-21T21:23:58+08:00`; prepare/checks 21:48–21:49; stage A rerun
  21:56–21:57; common 240-epoch stage B 22:00:04–22:19:57; final delivery before
  22:35, inside the 120-minute wall.
* Cost estimate after stage A: `DIRECT_S56` 24.0 min and `MATCHED_W34` 23.2 min
  for 240 epochs (safety 1.25) — both below 45 min, so the **240-epoch** branch
  was kept and both arms completed all 240 epochs.
* Both arms used the same 240-epoch horizon, the same Train molecule order and
  the same 5-epoch warmup. No per-arm early stopping.
* All formal training ran on `res` (a100-2, driver 550.163.01): GPU1 for
  `DIRECT_S56`, GPU0 for `MATCHED_W34`; one task process per GPU. No OOM.
* Cache, checks, curves, predictions and `run.json` were pulled back; large cache,
  weights and predictions are local and gitignored. Only code/config/protocol/
  report, curves/run records and lightweight `analysis/` are committed.

## Deviations

1. **Output path bug (the single engineering rerun).** `C.run_dir` initially
   appended a repo-relative `--out` path to `results/`, so the first stage-A
   launch wrote to a nested path. `run_dir` now resolves absolute paths,
   repo-relative paths, and bare run names separately. Stage A was rerun once for
   both arms on the fixed code. The model, data, seed, recipe and order were
   unchanged; the rerun reproduced the first stage-A metrics epoch by epoch.
   This is the one allowed engineering rerun; the first nested-path artifacts
   are not used.
2. **Initial-state hash bookkeeping.** `run.json:init_state_hash` hashes the
   whole `state_dict`, which includes the arm-specific constant `mu_bucket`
   buffer, so the two arms' whole-state hashes differ. The trainable-parameter
   hashes are identical and are recorded separately
   (`init_parameter_hash = 3d8d8cb1064efc9ca7908a5e40c348ec2fae6a79ff260b0771580e8998a24a99`).
   No training parameter differed at initialisation.
3. No other deviations. No new seeds, datasets, widths, objectives or
   mechanism arms were run. The stage-A path rerun is the only extra neural
   launch beyond the two planned unit sets.
