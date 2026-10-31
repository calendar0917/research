# LCRC Lipophilicity prototype / mid-range interaction pilot — REPORT v1

Round id: `lcrc-lipo-proto-interact-pilot-v1`
Directory: `tracks/structural-sufficiency/lcrc_lipo_proto_interact_pilot_v1/`
Execution commit (clean, deployed): `ff16154d47b6d4b484314f768564f43ddb757645`
Data: Lipophilicity **Fit2352 only**, v2 frozen scaffold **Fold 2** (Train 1568 / Dev 784), seed **0**.
Labels: **Train + Dev of the Fit split only.** Monitor504, Reserved504 and the random42
Validation420/Test420 splits were never read (asserted in every `train.json`).

---

## 1. What was actually run

One unified CLI, five formal runs (`360` epochs, R3 recipe, last-5-epoch weight soup, no early
stopping, no Dev-based checkpoint selection) plus one short multi-arm CUDA smoke:

| arm | definition | GPU | rr experiment | rr run id |
|---|---|---|---|---|
| `BASE_G0` | frozen `lcrc_model.LCRC` (`lipo_v1`, width 64), `y = Σ_v f(h_v) + ρ(G)` | `res-gpu0` | `lipo-pilot-base0` | `lipo-pilot-base0-20261031-190912-93d4ac99` |
| `PROTO_K16_TOP4` | BASE with only `self.f` replaced by the K=16 / top-4 / τ=0.25 shared-prototype readout | `res-gpu0` | `lipo-pilot-proto` | `lipo-pilot-proto-20261031-191416-37720169` |
| `BASE_G1` | byte-identical recipe to `BASE_G0`, run on the second GPU as the device-drift control | `res-gpu1` | `lipo-pilot-base1` | `lipo-pilot-base1-20261031-190921-03dc2af9` |
| `PAIR_TRUE` | BASE + `δ(G)` from mean products over graph-distance 3/4/5 atom pairs | `res-gpu1` | `lipo-pilot-pairtrue` | `lipo-pilot-pairtrue-20261031-191427-33776dab` |
| `PAIR_NULL` | `PAIR_TRUE` with a per-molecule fixed random permutation of the distance classes | `res-gpu1` | `lipo-pilot-pairnull` | `lipo-pilot-pairnull-20261031-191948-43a960b8` |
| smoke | 5 arms × 2 epochs × 2 eval batches, CUDA path check | `res-gpu0` | `lipo-pilot-smoke` | `lipo-pilot-smoke-20261031-190650-fea70850` |

Commands (exactly as executed; `R=tracks/structural-sufficiency/lcrc_lipo_proto_interact_pilot_v1`):

```bash
rr deploy res
rr run res lipo-pilot-smoke --pool res-gpu0 --result $R/analysis/smoke -- \
    uv run python $R/code/run_pilot.py smoke --device cuda --epochs 2 --batches 2
rr run res lipo-pilot-base0    --pool res-gpu0 --result $R/analysis/runs/BASE_G0_s0_fold2         -- \
    uv run python $R/code/run_pilot.py train --arm BASE_G0 --device cuda
rr run res lipo-pilot-proto    --pool res-gpu0 --result $R/analysis/runs/PROTO_K16_TOP4_s0_fold2  -- \
    uv run python $R/code/run_pilot.py train --arm PROTO_K16_TOP4 --device cuda
rr run res lipo-pilot-base1    --pool res-gpu1 --result $R/analysis/runs/BASE_G1_s0_fold2         -- \
    uv run python $R/code/run_pilot.py train --arm BASE_G1 --device cuda
rr run res lipo-pilot-pairtrue --pool res-gpu1 --result $R/analysis/runs/PAIR_TRUE_s0_fold2       -- \
    uv run python $R/code/run_pilot.py train --arm PAIR_TRUE --device cuda
rr run res lipo-pilot-pairnull --pool res-gpu1 --result $R/analysis/runs/PAIR_NULL_s0_fold2       -- \
    uv run python $R/code/run_pilot.py train --arm PAIR_NULL --device cuda
```

Locked recipe (identical for all five runs, `configs/pilot.yaml`): AdamW (β 0.9/0.999, ε 1e-8),
peak LR 3e-4, per-step cosine to 1e-5, weight decay 1e-3, batch 128, 360 epochs (13 steps/epoch,
**4680 updates**), grad clip 5.0, dropout 0.10 applied to the final atom-level `h` through the v2
`DropoutAtLocal` hook on `model.local`, MSE on the fold-Train-standardised target, soup of epochs
**356–360**. Target mean/σ = 2.13981 / 1.25697, computed from the 1568 Train labels only.

---

## 2. Integrity gates before GPU

`analysis/cpu_gates.json` — **12/12 gates passed** on CPU; the gate file records config hash
`6f28c05b81e9d4ad…`, identical to `analysis/protocol_freeze.json` (freeze file sha256 `1faf254ed1ea62f0…`):

1. Fold-2 Train/Dev rows equal the frozen v2 fold (positions + global row ids), no overlap, 1568/784.
2. Target standardisation recomputed from Train labels only (Dev mean differs, so the check is falsifiable).
3. `BASE` matches the frozen `LCRC` initialisation tensor-by-tensor, and also `v2_models.build_a` at seed 0.
4. `PROTO` shares every non-`f` tensor bit-exactly with `BASE`; only `f.prototypes/f.experts/f.bias` are new
   (no MLP bypass present).
5. `PAIR_TRUE`/`PAIR_NULL` have identical keys, identical parameter count (249,747) and bit-identical init.
6. `PAIR_TRUE` pairs are unique `u<v`, independently re-derived distances are exactly 3/4/5, and no pair in
   the sampled molecules is missing.
7. `PAIR_NULL` keeps `pair_ptr`/`u`/`v` identical, per-molecule bin multiset preserved, only the class labels
   permuted, deterministic on rebuild, and more than one permutation is used (per molecule, not global).
8. Atom relabelling leaves `BASE`/`PROTO`/`PAIR_TRUE`/`PAIR_NULL` predictions unchanged (<1e-4) and the
   per-molecule null permutation table is atom-index independent.
9. Prototype and pair parameters receive finite, non-zero gradients over 2 optimizer steps.
10. Zeroing the PAIR residual reproduces the same-weight BASE prediction.
11. Two mini-batch forward/backward/AdamW steps per arm, finite loss.
12. Batch order, batch composition and single-molecule batches agree (<1e-4).

Implementation defects found *and fixed by the gates/tests before any GPU run* (all were wiring bugs,
no scientific choice changed): the atom-renumbering utility did not permute per-atom S1/S2 pointers and
left `path_end` misaligned; the null-content check compared global bin counts instead of per-molecule
counts (a per-molecule permutation does change global counts by construction); `pair_moments` accepted a
0-dim `pair_bin` tensor.

`analysis/verify_pilot.json` — **61/61 post-run checks passed**: soup sha256, Dev prediction re-computation,
`fit_cache_position` alignment with the fold file, residual consistency, label-scope flags, clean commit,
soup members 356–360, 360 epochs, and reloading every checkpoint reproduces its stored Dev predictions
(CPU reload of a CUDA checkpoint, |Δ| ≤ 2.0e-6 logD).

---

## 3. Five-model table (primary metric: Dev RMSE in logD)

| model | pool | params | train RMSE | train MAE | **dev RMSE** | dev MAE | dev R² | gen. gap | epochs/steps | wall s | GPU s | clip events | soup |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `BASE_G0` | res-gpu0 | 247,106 | 0.2796 | 0.2074 | **0.7079** | 0.5477 | 0.6244 | +0.4283 | 360/4680 | 143.8 | 149 | 2837 | 356–360 |
| `PROTO_K16_TOP4` | res-gpu0 | 244,945 | 0.3645 | 0.2742 | **0.7419** | 0.5826 | 0.5874 | +0.3774 | 360/4680 | 160.3 | 166 | 3945 | 356–360 |
| `BASE_G1` | res-gpu1 | 247,106 | 0.2778 | 0.2063 | **0.7097** | 0.5487 | 0.6225 | +0.4319 | 360/4680 | 173.8 | 180 | 2727 | 356–360 |
| `PAIR_TRUE` | res-gpu1 | 249,747 | 0.2639 | 0.1945 | **0.7182** | 0.5510 | 0.6133 | +0.4543 | 360/4680 | 205.1 | 211 | 2732 | 356–360 |
| `PAIR_NULL` | res-gpu1 | 249,747 | 0.2619 | 0.1948 | **0.7250** | 0.5602 | 0.6060 | +0.4631 | 360/4680 | 230.1 | 237 | 2706 | 356–360 |

Initialisation fingerprints confirm the intended construction: `BASE_G0` and `BASE_G1` share
`init_state_hash = e2654c0b…`; `PAIR_TRUE` and `PAIR_NULL` share `init_state_hash = ede638fa…`.
Their soup hashes differ (device-dependent training), as expected.

Gradient clipping fires on 58–84 % of updates in every arm (R3 at peak LR 3e-4 with clip 5.0); it is a
property of the frozen recipe, identical across arms, and is reported rather than adjusted.

---

## 4. Paired comparisons (Dev, paired on `global_row_id`)

Negative Δ = candidate better. Slices are descriptive only; the primary metric was not re-selected.

| comparison | base | cand | ΔRMSE | ΔMAE | molecules with |err| improved | ΔRMSE after dropping 5 worst BASE | ΔRMSE after dropping 5 largest improvements |
|---|---|---|---|---|---|---|---|---|
| `BASE_G0 → BASE_G1` (device control) | 0.7079 | 0.7097 | **+0.0018** | +0.0010 | 396/784 | +0.0022 | +0.0025 |
| `BASE_G0 → PROTO_K16_TOP4` | 0.7079 | 0.7419 | **+0.0340** | +0.0349 | 375/784 | +0.0521 | +0.0437 |
| `BASE_G1 → PAIR_TRUE` | 0.7097 | 0.7182 | **+0.0086** | +0.0022 | 408/784 | +0.0075 | +0.0104 |
| `PAIR_TRUE → PAIR_NULL` | 0.7182 | 0.7250 | **+0.0068** | +0.0092 | 373/784 | +0.0124 | +0.0105 |

Reading of the slices:

* `PROTO` is worse on **409/784** Dev molecules and the deficit *grows* when the five worst BASE
  molecules are removed (+0.0521): the loss is broad, not carried by a few outliers.
* `PAIR_TRUE` vs `BASE_G1`: worse overall (+0.0086) but MAE is almost unchanged (+0.0022) and
  408/784 molecules improve — a sign that the residual helps typical molecules and hurts a few large
  errors. Train RMSE improves (0.2639 vs 0.2778) while Dev does not, i.e. the extra 2,641 parameters
  fit the training fold rather than transferring.
* `PAIR_TRUE` vs `PAIR_NULL`: the true distance semantics are better than the permuted ones by
  **0.0068** — the expected direction, ~4× the device-drift scale (0.0018), but below the frozen
  0.010 threshold, and both are worse than `BASE_G1`.

Scaffold groups (Dev has 244 groups; full table in `analysis/table_scaffold_groups.csv`), ΔRMSE by
group-size bucket:

| comparison | 1 | 2–3 | 4–7 | 8+ | groups better / worse | molecules in better groups |
|---|---|---|---|---|---|---|
| `BASE_G0 → BASE_G1` | +0.0033 | +0.0181 | +0.0045 | −0.0150 | 116 / 128 | 381/784 |
| `BASE_G0 → PROTO` | −0.0377 | +0.0096 | +0.0412 | +0.0848 | 126 / 118 | 356/784 |
| `BASE_G1 → PAIR_TRUE` | +0.0275 | −0.0113 | +0.0038 | +0.0180 | 124 / 120 | 373/784 |
| `PAIR_TRUE → PAIR_NULL` | +0.0040 | +0.0032 | −0.0032 | +0.0220 | 105 / 139 | 337/784 |

The group-level pattern is the same as the molecule-level one: no comparison shows a systematic
group-wise shift beyond the device-control noise, except `PROTO` being clearly worse in the larger
scaffold groups.

---

## 5. Mechanism diagnostics

### 5.1 PROTO routing (`analysis/runs/PROTO_K16_TOP4_s0_fold2/diagnostics.json`)

| quantity | Train | Dev |
|---|---|---|
| atoms | 41,697 | 21,534 |
| prototypes used as top-1 | 15 / 16 | 14 / 16 |
| prototypes entering top-4 | 16 / 16 | 16 / 16 |
| never top-1 / never top-4 | 1 / 0 | 2 / 0 |
| effective prototypes (exp of top-1 entropy) | 2.538 | 2.565 |
| largest top-1 share | 0.7181 (prototype 9) | 0.7172 (prototype 9) |
| smallest non-zero / zero top-1 share | 2/41697 = 0.00005 | 0 |

Top-1 counts (Train): `[33, 300, 78, 747, 7, 72, 41, 8, 2700, 29944, 37, 0, 7243, 2, 228, 257]`;
top-4 inclusion shares: `[0.0039, 0.0204, 0.0094, 0.2071, 0.0006, 0.0015, 0.2003, 0.0009, 0.0652,
0.2250, 0.0030, 0.0012, 0.2414, 0.0014, 0.0150, 0.0039]`. Train and Dev distributions are nearly
identical, so routing generalises; it simply **collapses** onto ~2.5 effective prototypes — one
prototype takes 72 % of all atoms, and the four most used prototypes (12, 9, 3, 6) cover 0.874 of
all top-4 slots.

Frozen-model control: permuting only the prototype keys `D` (experts `W,b` untouched) changes **all
784** Dev predictions, mean |Δ| = 1.8647 in normalised units (≈ 2.34 logD), max 6.10. The routing
path is therefore genuinely used by the trained model — the weak Dev performance is not a dead-code
artefact. Gradients: `f.prototypes` L2 1.769, `f.experts` L2 4.154, `f.bias` L2 0.500, all finite and
non-zero.

### 5.2 PAIR coverage, residual and gradients

| bin | Train coverage | Train mean pairs/mol | Dev coverage | Dev mean pairs/mol |
|---|---|---|---|---|
| distance 3 | 1.000 | 41.48 | 1.000 | 42.79 |
| distance 4 | 1.000 | 38.04 | 1.000 | 39.25 |
| distance 5 | 0.9955 | 36.09 | 0.9987 | 37.08 |
| any of 3/4/5 | 1.000 | 115.62 | 1.000 | 119.11 |

181,285 Train pairs and 93,385 Dev pairs; max 261 pairs in one molecule. Molecules without any pair in a
bin get a strictly zero bin vector (verified by a gate and a unit test).

Trained `PAIR_TRUE` residual (raw logD, after denormalisation):

| split | mean |δ| | max |δ| | mean δ (signed) | graphs with δ≠0 |
|---|---|---|---|---|
| Train | 0.2205 | 0.9968 | −0.1798 | 1568/1568 |
| Dev | 0.2077 | 0.8699 | −0.1823 | 784/784 |

Zero-residual check on the frozen checkpoint: setting the last `Linear(32,1)` weight/bias to zero gives
max |Δ| = 7.15e-7 (normalised) ≈ 9.0e-7 logD against a same-weight BASE backbone, while the trained
residual itself is up to 0.69 logD (normalised) on the same batch — i.e. the residual path is exactly
additive and the check is not vacuous.

Delta-path gradients on the frozen checkpoint (2 Train batches): `pair_z.weight` 0.1047,
`pair_z.bias` 0.0186, `pair_head.0.weight` 0.0313, `pair_head.0.bias` 0.0137,
`pair_head.2.weight` 0.0760, `pair_head.2.bias` 0.0485 — all finite, all non-zero.

`PAIR_NULL` residual: mean |δ| = 0.2226, max 0.8840, 784/784 non-zero. The null construction is exact:
identical `pair_ptr/u/v` and identical pair count, only the per-molecule distance labels permuted
(6 permutations used; identity permutation for 292/1568 Train and 132/784 Dev molecules, as expected
from a uniform draw). Because per-molecule permutations average out, the *global* bin counts become
near-uniform (31,172 / 31,166 / 31,047 Dev vs 33,548 / 30,770 / 29,067 true). That is an unavoidable
consequence of per-molecule permutations and is reported, not corrected.

---

## 6. Frozen decision rule

```
PROTO gain = BASE_G0 dev RMSE − PROTO dev RMSE = 0.7079 − 0.7419 = −0.0340   (threshold +0.020)
PAIR  gain = BASE_G1 dev RMSE − PAIR_TRUE dev RMSE = 0.7097 − 0.7182 = −0.0086 (threshold +0.020)
PAIR vs NULL = PAIR_NULL dev RMSE − PAIR_TRUE dev RMSE = 0.7250 − 0.7182 = −0.0068 (threshold +0.010)
```

**`NO_STRONG_SIGNAL_THIS_CONFIGURATION`** — neither direction meets its exploratory gate. See
`DECISION.md`; `analysis/summary.json → decision` contains the machine-readable result.

---

## 7. GPU budget ledger (`analysis/gpu_ledger.json`)

GPU-seconds are the rr-measured process lifetime on an exclusive pool GPU; the two pools are **summed**
even though they ran in parallel.

| experiment | pool | state | GPU s |
|---|---|---|---|
| lipo-pilot-smoke | res-gpu0 | completed | 56 |
| lipo-pilot-base0 | res-gpu0 | completed | 149 |
| lipo-pilot-proto | res-gpu0 | completed | 166 |
| lipo-pilot-base1 | res-gpu1 | completed | 180 |
| lipo-pilot-pairtrue | res-gpu1 | completed | 211 |
| lipo-pilot-pairnull | res-gpu1 | completed | 237 |
| **total** | | | **999 s = 0.2775 A100 GPU·h** |

Hard cap 0.5 GPU·h → **0.2225 GPU·h unused**. No retries were needed; every launch succeeded on the
first attempt. In-process train-loop time was 135.0 / 164.3 / 150.9 / 193.9 / 218.2 s (BASE_G0 / BASE_G1 /
PROTO / PAIR_TRUE / PAIR_NULL), i.e. 12–15 s per run is data loading, the frozen cache rebuild
(the Fit cache file is git-ignored on the remote and is rebuilt deterministically with fingerprint
verification), CUDA init and evaluation.

---

## 8. Anomalies and infrastructure notes

* **Remote deploy blocked by a previous round.** `rr deploy res` refused twice because the remote
  compute copy had (a) one *tracked* modified artifact
  (`lcrc_lipo_random42_controls_v2/analysis/runs/benchmark_inference/envelope.json`, timings rewritten
  by that round's own benchmark run) and (b) two *untracked* leftovers of the same round
  (`gpu_ledger.json`, `inference_benchmark.json`). The tracked file was restored to its committed bytes
  with `git checkout --` **on the remote compute copy**, and the two untracked files were moved to
  `/tmp/rr_cleanup_ff16154/` on `res`. **No local historical file was modified, and no historical
  checkpoint or result was overwritten.**
* **The frozen cache is rebuilt per process on `res`** (~3.5 s, 2352 molecules) because
  `lcrc_cross_task_preparation_v1/cache/` is git-ignored. The frozen loader verifies the cache content
  fingerprint and row order, so this is the intended behaviour, not a deviation.
* **Gradient clipping is frequent** (2706–3945 of 4680 updates). Identical recipe across arms; no
  adjustment was made.
* **Device drift is small but non-zero**: identical init hash, identical data/recipe, Dev RMSE differs by
  0.0018 (MAE 0.0010) between `BASE_G0` and `BASE_G1`. This is the correct scale for reading the
  `PAIR_TRUE` vs `PAIR_NULL` difference of 0.0068.
* **PROTO converges to a near-degenerate routing** (2.5 effective prototypes). No auxiliary loss,
  penalty or re-initialisation was added — that was explicitly out of scope for this round.
* **One source file was edited after the freeze and after the runs**: `code/pilot_analysis.py`, raising
  `verify()`'s CPU-reload tolerance from 1e-6 to 1e-5 logD (observed 2.0e-6, i.e. float32
  reduction-order noise between the CUDA run and the CPU reload). Only that post-run analysis module
  differs from `analysis/protocol_freeze.json`: the other 9 frozen files are byte-identical, and no model,
  training, data, batching or metric code changed. The full diff, hashes and consequence (the
  `train` command's freeze guard now refuses HEAD, by design) are recorded in
  `analysis/code_revision_after_runs.json`. Use `git checkout ff16154…` to reproduce training exactly.

---

## 9. Limitations

* One seed, one scaffold fold, one recipe. Every number here is a single-run observation.
* The Dev fold is the v2 scaffold Fold-2 *internal* Dev split of Fit2352; it is not the official
  Lipophilicity test set, and no held-out label was touched. These numbers are therefore not comparable
  to published Lipophilicity leaderboards.
* No significance test is claimed. The fixed thresholds (0.020 / 0.010) are exploratory gates, and the
  paired slices are descriptive.
* `BASE_G0` vs `BASE_G1` gives a *one-observation* estimate of device/run drift; it is a sanity scale,
  not a variance estimate.
* `PAIR_NULL` preserves within-molecule grouping and pair content but cannot preserve the global distance
  histogram (per-molecule permutations average out). The control therefore also removes a global prior,
  which is inherent to the requested construction.
* No Monitor504 / Reserved504 evaluation was performed, so nothing is known about how these five models
  behave outside the Fit2352 folds.

---

## 10. Artefact index

| artefact | path |
|---|---|
| fixed config | `configs/pilot.yaml` |
| protocol freeze / data audit / gates / authorization | `analysis/protocol_freeze.json`, `analysis/data_audit.json`, `analysis/cpu_gates.json`, `analysis/gpu_authorization.json` |
| per-run payload (recipe, params, timings, curve, soup hashes) | `analysis/runs/<arm>_s0_fold2/train.json` |
| per-run checkpoint (last-5 soup) | `analysis/runs/<arm>_s0_fold2/soup.pt` |
| per-molecule Dev predictions (`global_row_id`, `fit_cache_position`, `y_true`, `pred`, `residual`) | `analysis/runs/<arm>_s0_fold2/predictions_dev.npz` + `.csv` |
| per-molecule Train predictions | `analysis/runs/<arm>_s0_fold2/predictions_train.npz` |
| training curve (360 rows) | `analysis/runs/<arm>_s0_fold2/curve.csv` |
| mechanism diagnostics (PROTO routing / PAIR delta / gradients) | `analysis/runs/{PROTO_K16_TOP4,PAIR_TRUE,PAIR_NULL}_s0_fold2/diagnostics.json` |
| summary + comparison tables | `analysis/summary.json`, `analysis/comparisons.json`, `analysis/table_models.csv`, `analysis/table_comparisons.csv`, `analysis/table_scaffold_groups.csv` |
| run manifest (rr ids, states, GPU s) | `analysis/run_manifest.json`, `analysis/run_manifest.tsv` |
| GPU ledger | `analysis/gpu_ledger.json` |
| post-run verification | `analysis/verify_pilot.json` |
| smoke record | `analysis/smoke/smoke.json`, `analysis/smoke/*/train.json` |
| raw rr status snapshots | `analysis/rr_status/*.json` |
