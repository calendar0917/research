# PROTOCOL — LCRC-S3 v1: explicit three-hop structure and a model-selection upgrade

Round id: `lcrc-pcba-s3-v1`
Status: **frozen before the first formal training step.** Stage 0 (structure,
cost and distinguishability audits, no model training) is complete and its
measured numbers are quoted here; §12 records them as fixed inputs, not as
results that may still change. The only later edits permitted are (a) measured
GPU timings written into §9 and (b) the amendment log in §13. Nothing in §5,
§6, §7, §10 or §11 may change after the first training step.

## 1. The single question

W2 (`LCRC-A-Wide-2x`, 994,324 parameters) performs one atom encode, one bond
encode and one-shot S1/S2 local statistics, with **no iterative message
passing**. Its best official Validation AP is `0.2652733161` at epoch 24
(epoch-40 AP `0.2618669181`).

Does adding an explicit, one-shot **three-bond path** encoding — a simple path
`(r, u, v, x)` over four distinct atoms, i.e. information that a two-hop
neighbourhood does not contain — improve official Validation AP beyond a
**near-parameter-matched FFN capacity control**?

The contrast of interest is `S3 − F`, not `S3 − W2`: W2 is the reference, F is
the control that absorbs "just add 24 k parameters". No other question is asked
in this round, and no other experiment is run.

## 2. Frozen reference (read-only)

| item | value |
|---|---|
| round | `lcrc-pcba-backbone-scaling-v1`, arm `W2` |
| run directory | `tracks/structural-sufficiency/lcrc_pcba_backbone_scaling_v1/runs/W2` |
| parameters | 994,324 |
| seed / epochs / batch | 0 / 40 / 128 |
| batch-plan SHA-256 (global) | `06f437299fa451e449202a990d28f3b8d101e7aec709bcd72a878c8de864b6f6` |
| consumed prefix SHA-256 (40 epochs) | `1a24dfca54898dd2b0e7c58d2b2f1956df67b233fdceaa7a376f81a2377a5f58` |
| best Validation AP | `0.26527331610028143` at epoch **24** |
| epoch-40 Validation AP | `0.26186691813350904` |
| best checkpoint SHA-256 | `d86fe5e56e898d382acf66eb0533beb915f8ca39144c6c1c74e25e147667d2bd` |
| epoch-40 checkpoint SHA-256 | `7fbdbd16f7b2077e77cdb85488bb2506f2e4ee98333d40fbcf7558074e26598a` |
| trunk-init fingerprint | `5f7101a8f378e7c9283caaade8d61eee733c0c9f3cfb65b5584c8a1e495ef63d` |
| GPU hours (historical) | 1.3949 |
| epoch seconds (mean, historical) | 124.86 |

W2 is **never retrained**. Only read-only checkpoint reloading, one full
Validation evaluation of each of the two historical checkpoints, and a
comparison against their recorded reports are allowed
(`analysis/w2_best_replay.json`). The historical run's own recorded curve is the
source of its per-epoch Validation AP.

## 3. The S3 structure: exact definition

For a molecule with atoms `V` and real bonds `E` (five one-based bond types),
the S3 path set is

```
P = { (r, u, v, x) : (r,u) ∈ E, (u,v) ∈ E, (v,x) ∈ E,
                    |{r,u,v,x}| = 4,  r is the root }
```

with, for every path, the record

```
root r, mid1 u, mid2 v, end x
b1 = type(r,u), b2 = type(u,v), b3 = type(v,x)        (ordered, one-based)
mid1_type = atom_type(u), mid2_type = atom_type(v)    (16-type vocabulary)
d  = dist_G(r, x) ∈ {1, 2, 3}                         (BFS on the same graph)
```

Properties that are part of the definition, not of an implementation:

* **exactly three real bonds**, four **distinct** atoms — walks are excluded
  (`x ≠ r` rules out backtracking, so a path never reuses a bond);
* **rooted**: every path is emitted under its root, and each root's paths are
  contiguous (`ptr` per root), so a per-root reduction is well defined;
* **`d = 1`** means the three-hop walk closes a ring of length 3; `d = 2` closes
  length 4; `d = 3` means the endpoints are at the same distance a two-hop
  neighbourhood cannot see along this route;
* **no dedup, no truncation, no sampling, no random subsampling**: path
  multiplicity to the same endpoint (two distinct three-bond routes) is
  preserved, and the maximum observed count is 1,048 paths for one molecule;
* **molecule-local atom ids** in the cache; a batch offsets them into a global
  id space exactly once, in `collate_s3`; nothing crosses a molecule boundary;
* the S3 tensors are **label-free** and cover **Train ∪ Valid only** (394,136
  rows = 350,343 + 43,793); the cache records `labels_present=false` and
  `test_rows_present=false`, and the row-id lookup returns `-1` outside scope.

The independent checks in `analysis/stage0_gates.json`:

| check | result |
|---|---|
| synthetic ring/chain cases (cyclobutane 8×d1, cyclopentane 10×d2, cyclohexane 12×d3, butane 2×d3, hexane 6×d3) | pass |
| path-by-path verification against the raw graph (256 molecules, 24,214 paths) | pass |
| atom-renumbering equivariance of the enumeration (128 molecules, 11,754 paths) | pass |
| per-root counts vs the independent closed-form identity (all 394,136 rows) | pass |
| exact full-cache path count | 39,754,052 |
| cache size | 0.452 GiB (485.3 MB) |
| cache build time | 110.4 s (projected 2.41 min ≤ 90 min) |

### 3.1 Why this is not a two-hop feature

**Rooted two-ball vs three-ball distinguishability** (label-free, on the fixed
4,096-molecule Train sample, exact VF2 isomorphism classes):

| quantity | value |
|---|---|
| two-ball classes | 8,573 |
| two-ball classes with ≥ 2 distinct three-ball classes | 3,780 (44.1 %) |
| atoms in an informative class | 93.5 % |
| molecules with at least one informative atom | 99.6 % |
| max three-ball variants per two-ball class | 512 |

Six diverse witness pairs (different molecules and different size strata) were
verified twice: the two rooted balls are isomorphic, and the three-hop-rooted
balls are **not** isomorphic (`analysis/structure_novelty.json`, each pair
carries `verified_2ball_identical=true`, `verified_3ball_different=true`). A
synthetic witness is also verified: cyclohexane rooted at 0 vs n-hexane rooted
at 2 — two balls isomorphic (both a 4-atom path), three balls not, 12 vs 6 S3
paths, `d ∈ {1,2}` vs `d = 3`.

The claim is scoped exactly to this: S3 adds *explicit local structural input*
that the S1/S2 interface cannot express. It says nothing about assay
information, and no part of this round shows that such input is predictive.

The frozen claim is also **not** "S3 is the only way to see three hops": a
two-hop WL colouring or an extra propagation step could in principle recover
part of it. S3 is a *one-shot explicit* encoding and is compared with a
capacity control, not with a two-hop baseline.

## 4. Coverage and cost gates (all measured in stage 0, no training)

| gate | threshold | measured | status |
|---|---|---|---|
| molecules in the sampled Train set with ≥ 1 path of `d = 3` | ≥ 0.30 | **0.99438** | pass |
| verified 2-ball/3-ball witness pair | required | 6/6 + synthetic | pass |
| projected cache size | ≤ 20 GiB | **0.452 GiB** | pass |
| projected preprocessing time | ≤ 90 min | **2.41 min** | pass |
| truncation | forbidden | none (max 1,048 paths/row) | pass |
| S3 epoch time | ≤ 2.5 × W2 epoch | measured in §9 | pending |

Stage-0 verdict: `S3_PREFLIGHT_STRUCTURE_PASSED`.

Sample definition: 4,096 Train rows, label-free, deterministic, size-stratified
(seven fixed atom-count strata, minimum draw per stratum, largest-remainder
allocation, within-stratum systematic selection, no RNG); sample SHA-256
`88e4356ae4ab…` recorded in `analysis/path_count_distribution.json`.

## 5. The two arms

Both arms start **from scratch** from the same untrained W2 seed-0 state (the
historical W2 initialisation, rebuilt bitwise). Neither arm loads a trained W2
checkpoint. Exactly two arms exist in this round; there is no third arm, no
multi-seed run, no S4/S5 extension and no hyper-parameter rescue.

### 5.1 Arm `S3` — LCRC-S3

Inserted exactly once, after the single `local` block, before the single
dropout:

```
z_p   = SiLU( W_e a_x                       # a = frozen atom-encoder output, dim 192
            + E_mid1[mid1_type] + E_mid2[mid2_type]
            + E_bond1[b1-1] + E_bond2[b2-1] + E_bond3[b3-1]
            + E_dist[d-1] )                 # 48 + 48 + 48 = dim 48
s_r   = [ mean_p∈r z_p , sum_p∈r z_p / sqrt(m_r) , log1p(m_r) ]   # 97 dims, m_r = #paths of root r
r_r   = Linear(97, 128)(s_r)               # zero-initialised
h     = drop( h_raw + r_r )
```

* `W_e: Linear(192, 48)`; the six embeddings are **factored** (`16×48`,
  `16×48`, `5×48` ×3, `3×48`) — there is no joint path-type vocabulary;
* atom `a_x` is the *static* atom-encoder output; it is gathered, never
  recomputed from a propagated state;
* per-root aggregation is a scatter-add over the root index, so a molecule with
  `m_r = 0` paths gets `s_r = 0` and therefore `r_r = 0` from the zero
  initialisation;
* `d = 1` is represented by the same `E_dist` slot as every other distance —
  ring closure is a distance class, not a special case;
* parameters: `9,264 + 768 + 768 + 240·3 + 144 + 12,544 = 24,208`; total
  `1,018,532`.

### 5.2 Arm `F` — capacity control

```
h = drop( h_raw + Linear(93,128)( SiLU( Linear(128,93)( LayerNorm(128)(h_raw) ) ) ) )
```

with the last `Linear` zero-initialised. Nothing is gathered, nothing is
rooted, no path or bond tensor is read. `H = 93` is chosen because
`Δ(H) = 257H + 384` and `|24,208 − 24,285| = 77` is the minimum over all
positive integers (`94` would be 180 away): `F` has **24,285** new parameters,
total **1,018,609**, a gap of **0.318 %** of the S3 branch, inside the 1 % gate.

### 5.3 Initialisation

1. rebuild the historical W2 seed-0 model (`bs_model.build_model`, seed 0);
2. construct the new modules (their default init is discarded);
3. copy the whole W2 base state **bitwise** into the arm (verified tensor by
   tensor, `original_w2_subset.all_bitwise_equal = true`);
4. re-initialise **only** the new tensors from a **private generator**
   (`seed·1,000,003 + offset`, offset 901 for S3, 902 for F): Kaiming-uniform
   `W_e`/`fc1`, `Normal(0, 1/√48)` embeddings, default `LayerNorm`;
5. zero the S3 output layer (`out.weight`, `out.bias`) and the FFN's last
   `Linear` (`fc2.weight`, `fc2.bias`).

Consequence, verified: **epoch-0 logits equal W2's logits exactly** (max
absolute difference `0.0` on CPU for both arms; tolerance `2e-5` on CUDA), and
the base trunk is bitwise unchanged. The global torch RNG is not advanced by
step 4 (asserted and recorded).

The zero output layer also means the first optimiser step changes only
`out.weight/out.bias` (`fc2` for F); `W_e` and all six embeddings receive their
first non-zero gradient from step 2 onward. That is expected, is recorded in
`analysis/initialization_parity.json`, and is not a defect.

## 6. Training protocol (identical for both arms)

| item | value |
|---|---|
| optimiser | AdamW, `betas=(0.9, 0.999)`, `eps=1e-8` |
| learning rate | `3e-4`, 1-epoch linear warmup `3e-5 → 3e-4`, then per-step cosine to `1e-5` over the 40-epoch horizon |
| weight decay | `1e-3` |
| dropout | `0.10` (the frozen single dropout) |
| gradient clipping | global norm `5.0` |
| precision | FP32 |
| loss | the original `masked BCEWithLogits` over observed entries; NaN targets never enter |
| batch size | 128, the historical epoch permutation, **same plan SHA-256 as W2** |
| epochs | 40, no early stopping |
| seeds | data order / init / dropout all seed 0 |
| Validation evaluation | **every epoch 1…40** (deterministic pass, no gradient) |

Both runs assert at start-up that their plan meta equals
`C.W2["global_plan_sha256"]` and `C.W2["consumed40_prefix_sha256"]`; a mismatch
aborts before training.

### 6.1 Model selection (the upgrade over the historical round)

For each arm, `e* = argmax_{e ∈ 1..40} AP_valid(e)`, **earliest epoch wins
exact ties**. The run stores `checkpoint_best.pt`, `best_epoch`, `best_valid_ap`,
`best_checkpoint_sha256`, the full Validation curve, and additionally
`checkpoint_epoch40.pt`, `AP40`, Train AP40 and Valid BCE40.

The historical W2 record is used **as recorded** (best epoch 24, AP
`0.2652733161`) and is independently re-verified by replaying its two
checkpoints read-only (`analysis/w2_best_replay.json`, tolerance `1e-4`).
Epoch-40 numbers are reported as an auxiliary endpoint only: **no epoch-40
result may replace the selected checkpoint** in any comparison.

### 6.2 What is never touched

Official Test (43,793 rows) is never scored, never indexed, never loaded into
the S3 cache, and no Test number appears in this round. No architecture
selection uses Test. There is no multi-seed run and no post-hoc threshold
tuning.

## 7. Audits (all recorded under `analysis/`)

| # | audit | where |
|---|---|---|
| 1 | exactly three real bonds, four distinct atoms, ordering, `end ≠ root`, `d∈{1,2,3}` | `tests/test_s3_paths.py` |
| 2 | synthetic ring/chain closed forms | `stage0_gates.json`, tests |
| 3 | per-root counts vs independent closed form | `stage0_gates.json` |
| 4 | no cross-molecule paths (`check_s3_batch_isolation`) | `architecture_audit.json` |
| 5 | atom-permutation equivariance of the whole model | `architecture_audit.json` (tolerance `2e-5`) |
| 6 | bond edit ⇒ encoding changes | `architecture_audit.json` |
| 7 | no iterative message passing (one `local`, one `drop`, minimal-input branch call) | `architecture_audit.json` |
| 8 | epoch-0 logits == W2 | `initialization_parity.json` |
| 9 | bitwise W2 base state in both arms | `initialization_parity.json` |
| 10 | gradients on all new tensors (from step 2) | `initialization_parity.json` |
| 11 | empty-S3 safety and pathless-root convention | `architecture_audit.json`, tests |
| 12 | no NaN/Inf in logits, loss or branch output | trainer asserts |
| 13 | cache row-id ↔ batch consistency | `architecture_audit.json` |
| 14 | Train-only normalisation unchanged | inherited (no renormalisation in this round) |
| 15 | no Test access anywhere | `test_evaluated=false` in every artifact |
| 16 | checkpoint resume reproduces state and logits | `initialization_parity.json` |
| 17 | GPU profiler: op count does not scale with path count (no per-path Python loop) | `gpu_profile.json` |
| 18 | epoch cost ≤ 2.5 × W2 | `gpu_cost_probe.json` |
| 19 | W2 replay reproduces AP and hashes | `w2_best_replay.json` |
| 20 | parameter gap ≤ 1 % of new parameters | `initialization_parity.json`, tests |

Registered behaviour: in audit 5, atom renumbering changes only the order of
floating-point summation, so the recorded tolerance is `2e-5` (observed
`2.9e-6` on CPU), not exact equality.

## 8. Data discipline

* the S3 cache is built from the frozen PCBA feature cache
  (`lcrc_pcba_scale_pilot_v1/cache/pcba_v1`), Train ∪ Valid only;
* the cache stores no labels, and the builder asserts the row scope has no Test
  overlap;
* the label artifact is read through the inherited loader, which masks Test
  targets to NaN; the evaluator scores the 127 Validation tasks that have
  observed labels (task 45 has zero positives and is unscorable);
* the Training/Validation normaliser is the inherited LCRC normaliser, fitted
  on Train only; this round does not refit it.

## 9. GPU budget and execution regime

* host `res` (single box, pools `res-gpu0` / `res-gpu1`, one A100 per pool),
  all execution through `rr`; no manual SSH, no external processes touched;
* S3 → pool `res-gpu1`, F → pool `res-gpu0`, launched in parallel (separate
  GPUs, independent memory); if the second pool is unavailable, the two runs
  are serialised on `res-gpu1` without changing anything else;
* budget: target **≤ 6 GPU-hours**, hard cap **≤ 10 GPU-hours**, charged per
  job from the run records (`analysis/gpu_ledger.json`);
* an epoch-cost probe (`gpu_cost_probe.json`) measures a 20-step training
  window per arm and projects the epoch time before the formal runs start;
* the formal jobs are watchdogged (soft/hard wall-clock) and are crash-visible:
  a failure leaves `run.json` with `status=FAILED`.

## 10. Pre-registered decision rules

`D_best_S3 = AP_best(S3) − AP_best(W2)`, `D_best_F = AP_best(F) − AP_best(W2)`,
`D_mech = AP_best(S3) − AP_best(F)`, where `AP_best(X)` is the official
Validation macro AP at `X`'s own selected best-Validation epoch. Support strata
refer to Validation positive counts per task; the high-support stratum is the
107 tasks with ≥ 20 positives (the fixed reference set).

Rules are applied **in this order**; the first match is the label.

| # | label | condition |
|---|---|---|
| 1 | `PROMISING_S3_SIGNAL` | `D_best_S3 ≥ +0.005` **and** `D_mech ≥ +0.003` **and** on the 107 high-support tasks `mean(S3 − W2) ≥ +0.003` **and** ≥ 60/107 improved |
| 2 | `REGRESSION` | `D_best_S3 ≤ −0.002` |
| 3 | `NO_CLEAR_GAIN` | `D_best_S3 < +0.002` and not (high-support mean ≥ +0.003 and ≥ 60/107 improved) |
| 4 | `CAPACITY_ONLY` | `D_best_F > 0` and `D_mech ≤ +0.003` |
| 5 | `WEAK_OR_MIXED` | everything else |

Interpretation, frozen with the rules:

* `PROMISING_S3_SIGNAL` means "the three-hop input is worth confirming on a
  train-only scaffold-grouped development split", not "S3 generalises better".
  That confirmation is **not** run in this round.
* `CAPACITY_ONLY` means the gain is ordinary capacity; the S3 mechanism has no
  material advantage over a generic residual MLP of the same size.
* `REGRESSION` is a statement about this exact recipe, not a proof that
  explicit three-hop structure is useless.
* `NO_CLEAR_GAIN` / `WEAK_OR_MIXED` end the S3 route. No fallback to S4/S5,
  no larger rank, no retuning, no multi-seed rescue in this round.
* The thresholds are research priorities, not significance tests; the round has
  one seed, so no claim of statistical significance is made.

## 11. Deliverables

`PROTOCOL.md`, `README.md`, `REPORT.md` (Chinese), `code/s3_common.py`,
`code/s3_paths.py`, `code/s3_model.py`, `code/s3_train.py`, `code/s3_audit.py`,
`code/s3_compare.py`, `tests/test_s3_paths.py`, `tests/test_s3_model.py`,
`tests/test_s3_training.py`, and the `analysis/` artifacts:

`structure_novelty.json`, `path_count_distribution.json`,
`cache_cost_projection.json`, `stage0_gates.json`, `s3_cache_audit.json`,
`gpu_preflight.json`, `gpu_cost_probe.json`, `gpu_profile.json`,
`initialization_parity.json`, `architecture_audit.json`, `w2_best_replay.json`,
`s3_selected.json`, `ffn_selected.json`, `best_valid_comparison.json`,
`epoch40_comparison.json`, `per_task_deltas.csv`, `gpu_ledger.json`,
`s3_ablation.json`, `decision.json`, `training_parity.json`.

`runs/S3/` and `runs/F/` are git-ignored (checkpoints and curves), but every
hash needed to identify them is committed in the analysis artifacts.

## 12. Stage-0 inputs frozen by this protocol

* the 4,096-row label-free sample and its SHA-256;
* the exact full-cache path count 39,754,052 and the 485.3 MB cache size;
* the 0.452 GiB cache and 2.41 min preprocessing projections;
* the distinguishability result (8,573 / 3,780 / 93.5 % / 99.6 % / 6 witnesses);
* `S3_ADDED = 24,208`, `F_ADDED = 24,285`, gap `77` (0.318 %).

## 13. Amendment log

| date | change | reason |
|---|---|---|
| — | none | the protocol is unamended at freeze time |

Post-freeze entries may only record (a) measured GPU timings, (b) an
infrastructure failure and its re-run with the same commit and the same
protocol, (c) the actual `rr` job ids. No scientific parameter is editable.
