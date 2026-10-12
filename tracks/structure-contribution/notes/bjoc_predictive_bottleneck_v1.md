# cscl-predictive-bottleneck-diagnosis-v1 — design / pre-registration note

**Status (at freeze):** CPU phases (B, C, split, gates) complete; no GPU trajectory has
run.  Protocol `configs/bjoc_predictive_bottleneck_v1.yaml` frozen and committed before
any GPU.  **Official valid/test are never loaded in any form.**

## 1. Why this round exists

The BJOC programme has a long sequence of negative module results (TypedPathRing,
junction dropout, additive IJCM, cross-bond interaction).  Every one of them added or
changed a structural channel and failed to move held-out MAE.  This round stops that
search pattern and spends the minimum formal budget to separate:

1. **sample efficiency** — the official train split has only ~10k molecules;
2. **structure ownership** — BJOC-Null's graph-level junction summary may destroy
   relational structure that BJOC-Organized preserves;
3. **architectural inductive bias** — the one-shot object encoding + global readout
   may fit the training rows and fail to generalise, in which case another local
   feature will not help.

plus an independent CPU audit of **input observability** (are different-y molecules
mapped to the same raw input?).

## 2. Frozen split `cscl-predictive-bottleneck-v1-A`

Canonical-SMILES grouped greedy split of the official train 10000 (salt
`cscl-predictive-bottleneck-v1-A`): Fit8k 8000 / Monitor 1000 / Locked 1000, duplicate
molecules never cross sides.  Nested Fit4k = the SAME group order, whole groups, greedy
to 4000 (label-blind).  Actual sizes are exactly 8000/1000/1000 and 4000.

Frozen hashes (also in the config `frozen:` block):

| set | rows | sha256 (prefix) |
|---|---:|---|
| Fit8k | 8000 | `1cf6a2ec73ff5dfc…` |
| Fit4k | 4000 | `3f6b5d69097476b0…` |
| Monitor | 1000 | `d493dc7fc547e062…` |
| Locked | 1000 | `9cbefd5e84cf7f4e…` |

The global-feature standardiser is fitted separately per training size on that size's
own fit rows (`prep_fit8k.json` / `prep_fit4k.json`); it is the only fitted preprocessing
(atom/bond/junction features are raw).

## 3. Frozen update-count protocol

`T = 240 × ceil(n_fit8k / 128) = 15120` for **every** trajectory.

| trajectory | fit rows | steps/epoch | steps run | soups |
|---|---:|---:|---:|---|
| N4-s0, N4-s1 | 4000 | 32 | 15120 (≈472.5 epochs) | `short` = steps [7552, 7584, 7616, 7648, 7680]; `stepmatched` = [14868, 14931, 14994, 15057, 15120] |
| N8-s0/s1, O8-s0/s1 | 8000 | 63 | 15120 (240 epochs) | last-5 epoch endpoints [14868, 14931, 14994, 15057, 15120] |

The N4 soups are two pre-frozen readouts of **one** trajectory, never two trainings.
Snapshots are taken after the exact frozen update numbers; no checkpoint is selected by
monitor.  The sample-size contrast is `N8 − N4-stepmatched` at identical update count;
`N4-stepmatched − N4-short` isolates extra optimisation on the same 4k molecules.

Trajectories (max 6): N4-s0, N4-s1, N8-s0, N8-s1, O8-s0, O8-s1.  Arms are BJOC-Null /
BJOC-Organized only, 413505 parameters, raw-y L1, Adam 1e-3 / wd 1e-5 / batch 128 /
clip 5, no scheduler, no dropout, FP32, last-5 soup.  Eight readout models.

## 4. Pre-GPU correctness gates (local, 22/22 PASS)

`results/bjoc_predictive_bottleneck_v1/correctness_gates.json`

- split nesting / hashes / disjointness / Fit4k never touches monitor or locked;
- frozen step plan recomputation;
- normalisation strictly fit-only, per size, hashes frozen;
- N and O: 413505 parameters each, identical init state hash per seed, identical state
  dict keys/shapes, only `organization_mode` differs;
- identical batch order for N and O at the same seed;
- batch-composition consistency, atom-relabelling invariance, finite forward,
  save/load replay bit-identical;
- objects cache hash unchanged and 8 rows rebuilt from raw PyG bit-identical;
- source scans: no official val/test load call, no component supervision tokens, no
  historical-output supervision tokens;
- provenance hashes recorded.

## 5. Phase B (CPU, read-only) — where the historical O−N monitor gain lives

Cross-bond split-A seed 0 monitor (N/O, same fit/seed/recipe):

- MAE N 0.098266 → O 0.089843, Δ = **−0.008423**, CI95 [−0.017744, −0.001120];
- O better on only **53.4%** of molecules;
- top-1-removed Δ −0.005132, top-5-removed −0.003558, top-10-removed −0.004404, so a
  large part of the mean difference comes from a small number of large-N-error rows;
- ordinary y>−3 (n=931) Δ −0.010169 CI excludes 0; tail y≤−3 (n=69) Δ **+0.015137**;
- the O-fixed rows are mostly large-N-error molecules; the P channel (from the
  cross-bond round) does not further reduce error on its top-100 prediction-difference
  rows (mean Δerr −0.0018, 47% improved) while N→O does (−0.0581, 55% improved).

Historical matched comparisons: the composer round's N/O monitor deltas are −0.001230
(seed 0) / −0.002588 (seed 1) on a different split with CIs crossing zero; the composer
locked (seed-0-only) delta is −0.002732 CI crossing zero; the full-10k official-valid
N/O difference (+0.00498, O worse) is another training condition and is background only.
Reading: the only CI-supported O>N gain so far is the single cross-bond split-A seed-0
monitor comparison, and it is partly extreme-row driven.  This is exactly why the round
buys a two-seed matched O8/N8 contrast on a new split.

## 6. Phase C (CPU) — input observability

Model input layout audit (`input_observability_audit.json`): atom 108-dim block layout,
bond 7-dim, junction 32-dim, global 33-dim, all deterministic functions of the raw
labelled graph.  The raw snapshot stores no stereochemistry, formal charge, explicit H,
bond stereo, isotope or 3D information; `edge_attr` in this snapshot is {1,2,3}; atom
codes are 0..20.

Graph collision audit (`graph_collision_audit.json`):

- WL candidates → strict VF2 labelled isomorphism: **3 groups / 6 rows**;
- those 3 groups also have bit-identical canonical model-input hashes, and every one has
  **identical y** (range 0.0); the empirical indistinguishable floor is **0.0**;
- there are 4 duplicate-canonical-SMILES groups; the one with inconsistent y (rows
  941/2264, y range 0.01974) is **not** labelled-isomorphic and does **not** share the
  model input (the raw ZINC atom-type codes differ), so it is not an input collision.

Verdict: `INPUT_COLLISION_FLOOR_NOT_ESTABLISHED`.

## 7. Read discipline

- fit y: training only; monitor y: standard training-time readout + frozen paired
  analysis; locked y: one read in `score` after every prediction file is hashed;
- no locked-based model selection, no re-training after the locked read;
- official valid/test never accessed; historical official-test 0.092233 never used for
  any decision;
- molecule bootstrap does not cover training randomness; two seeds cannot estimate the
  seed distribution;
- Fit4k/8k are two controlled points, not a learning curve; no extrapolation to
  20k/50k/100k.

## 8. Phase D (GPU) — six trajectories, matched updates

All six trajectories ran from commit `fd26214` on the remote A100 (pool `res-gpu1`), 413,505
params each, raw-y L1, T = 15120 for every trajectory and every readout; N4 ran 473 epochs
(32 steps/epoch), N8/O8 240 epochs (63 steps/epoch).  Wall: 398/395 s per N4 seed, 364–391 s per
8k seed; total ≈ 0.64 GPU-h, plus ≈2 min for the label-blind prediction stage.

Locked seed-mean MAE: N4short 0.155688, N4stepmatched 0.147022, N8 0.109719, O8 0.103300.
Monitor seed-mean MAE: 0.197306 / 0.192280 / 0.151196 / 0.144556.

Frozen primary contrasts (locked, paired molecule bootstrap):

| contrast | seed-mean Δ | CI95 | per-seed | top-10 removed |
|---|---:|---|---:|---:|
| N4stepmatched − N4short | −0.008665 | [−0.011539, −0.005733] | −0.012666 / −0.004665 | −0.008210 |
| N8 − N4stepmatched | −0.037304 | [−0.043517, −0.031045] | −0.033340 / −0.041267 | −0.036507 |
| O8 − N8 | −0.006419 | [−0.010001, −0.002772] | −0.013698 / **+0.000861** | −0.005735 |

The 4k trajectories overfit (N4stepmatched fit 0.0293 vs N8 0.0397) while generalizing much
worse; the data effect is ~4x the extra-optimization effect and survives top-10-removal in both
the ordinary and tail subgroups.  The ownership direction is average-negative but fails the
frozen two-seed condition (seed1 +0.0009); it is recorded as a `near_miss_signal`.

## 9. Phase E — common errors, decision, report

Common errors (locked, four primary models): four-way top-10% intersection 36/100, mean pairwise
Jaccard 0.414, pairwise residual correlation 0.907–0.949, error-sign agreement 93–98%; seed
stability N 53/100 (Jaccard 0.361), O 58/100 (0.408); of the molecules high-error in both N seeds
O fixes only 8/53, and of those high-error in both O seeds N fixes 2/58.  The dominant failure
mode is a shared, seed-stable model-family error, not an input collision.

Locked labels were read exactly once in `score` after all 16 prediction files were hashed
(`prediction_manifest.json` combined sha256 `8c9298d820771ea1…`; `locked_read.lock`); monitor was
never used for checkpoint selection; official valid/test never accessed.

**Decision: `SAMPLE_EFFICIENCY_BOTTLENECK_SUPPORTED`** (`scientific_decision.json`; frozen rule
from the config).  Next round: label-blind structural utilization experiments (masked graph-attribute
pretraining) with pre-registered stop conditions; no new architecture until that fails.

Provenance disclosure: the freeze manifest was written at the pre-commit working-tree revision
(driver_sha256 `34e86a87…`, HEAD `67db049`); the trained commit `fd26214` has driver_sha256
`bc617c2b…` because a driver edit occurred between `freeze` and the commit.  The config frozen
block is byte-identical (config sha256 `4769d7bf…` unchanged), remote `splits/objects/steps`
reproduced the frozen hashes, and the only post-training edits are analysis-only
(`cmd_common_errors` / `cmd_decision`; `git diff fd26214` touches no training or prediction code).
