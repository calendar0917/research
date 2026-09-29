# Pre-registration — `e2e_dictenv_capacity_localization_v2`

Workstream Z (ZINC dictionary-environment line), CPU only, no GPU, no SSH, no
remote compute.  The official ZINC test split is **never** loaded
(`official_test_loaded = false` in every payload).

Round name: `e2e_dictenv_capacity_localization_v2`

Question of this round (v1 is **not** re-opened):

> Can we run a *discriminative* local capacity experiment on the frozen
> CSSD-q1 representation — i.e. one that does not itself destroy the trained
> baseline — and, under that repaired protocol, does any of the three frozen
> capacity directions F / R / G beat the absolute anchor `M_start = 0.130028`?

v1 (`e2e_dictenv_capacity_localization_v1`) returned
`NO_CLEAR_CAPACITY_LOCALIZATION` but the screen had no power: the pure M0
continuation control under fresh `Adam(lr = 1e-3)` degraded from `0.130028`
(step 0) to `0.135621` (best) and `0.156383` (epoch 40), so every arm was
measured inside a degradation regime.  v2 repairs the **adapter only**; it does
not add a new architecture, does not sweep rank/width/LR, and does not combine
candidates.

Implementation (frozen with this document):
`tracks/ksvd/experiments/luyin16/e2e_dictenv_capacity_localization_v2.py`;
runner:
`tracks/ksvd/experiments/luyin16/zinc_e2e_dictenv_capacity_localization_v2.py`;
focused tests:
`tracks/ksvd/tests/test_e2e_dictenv_capacity_localization_v2.py`.

Nothing in this document may be edited after the first training process starts.
No threshold, gate, scale, seed, horizon or tie-break may be changed after
seeing a result.

---

## 0. Frozen context (not re-opened)

Frozen base: `CAP-BASE = CSSD-q1` (C6 clean mask, train-only q1 common structural
coordinate, q1-orthogonal residual sparse dictionary K=32/s=8/IHT-10, paired
node/edge structure-semantic binding, full clean relation, first/second moment
readout; 97 727 parameters).

| item | value |
|---|---|
| CAP-BASE seed-0 soup valid MAE `M_start` | `0.13002798487985273` (re-asserted in preflight) |
| CSSD soup checkpoint | `results/e2e_dictenv_common_subspace_dictionary_v1/training/checkpoints/CSSD-Q1-seed0_soup_state.pt` |
| checkpoint sha256 | `5fc41ab49eca10bf9cf31ef033f8fbe3dbb96ef0eb394116d7a9ca6bf3aa145d` |
| train-only q1 subspace | `results/e2e_dictenv_common_subspace_dictionary_v1/common_subspace.json` (sha256 `36636ce92836bdb8d023cc91b3f532f8d4c46a57d457514c92d90c68028f6c24`) |
| historical reference only | FINAL-CLEAN sparse seed-0 soup `0.12849851670576026` (never a matched control) |
| data | frozen phi65 cache; official ZINC train 10 000 / valid 1 000 |

The baseline is **not** retrained.  No v1 conclusion is re-litigated; no
sparse-vs-dense, IHT, K/s/q, mask, node/edge-binding or readout question is
re-opened.

---

## 1. Hard constraints

* CPU only: `CUDA_VISIBLE_DEVICES=""`, `torch.device("cpu")`,
  `cpu_only_guard` at every stage entry; no GPU, no CUDA, no SSH, no remote
  compute.
* Official test never loaded; `official_test_blocker` on every payload.
* `docs/luyin/luyin19.txt` untouched (never edited, moved or staged).
* Exactly one seed (0).  No seed 1/2 anywhere.
* One frozen architecture per candidate; no rank/head/width/depth sweep.
* No learning-rate rescue: the single differential protocol below is the only
  warm-adaptation protocol tested.  If it fails, the warm-screen route stops.

---

## 2. Budget (hard ceiling)

```text
Phase A: 1 x M0 calibration x 20 epochs
Phase B: M0 / F / R / G x 40 epochs          (only if A passes)
Phase C: 1 x winner x 320 epochs from scratch (only if a candidate passes)
maximum 500 training epochs
```

Fail paths: calibration fail -> 20 epochs only; no screen winner -> 180 epochs
only; exactly one full run maximum.  Concurrency: wave 1 = `M0, F, R`
(3 processes x 4 threads), wave 2 = `G` (1 x 4); reduce concurrency if
contention is observed.  Phase C = 1 process x 4 threads.

---

## 3. Phase A — M0 warm-adaptation calibration (frozen)

Start: CAP-BASE seed-0 Top-5 soup.  Fresh optimizer:

```text
Adam(lr = 1e-4, weight_decay = 1e-5), batch 128, clip 5.0
loss = L1(y_hat, y) + H1_LAMBDA * reconstruction   (CSSD objective, unchanged)
epochs = 20 (fixed)
```

Per-epoch record: train MAE, valid MAE, reconstruction loss (full diagnostic and
optimised term), total loss, base gradient norm, base update norm, wall time.
Final record (exactly these fields):

```text
start_mae, calibration_best, calibration_soup (Top-5 over epochs 1-20),
calibration_last5 (mean valid epochs 16-20), epoch20,
delta_best, delta_soup, delta_last5, late_slope (linear fit epochs 16-20),
C1_pass, C2_pass, C3_pass, overall_pass
```

### Calibration gate (all three must hold)

```text
C1  soup(1:20)          <= 0.1320
C2  mean(valid 16:20)   <= 0.1350
C3  epoch20 <= 0.137  AND  slope(valid, 16:20) <= +5e-4 / epoch
```

`overall_pass` -> `WARM_ADAPTATION_PROTOCOL_VALIDATED`; otherwise
`WARM_ADAPTATION_PROTOCOL_UNSTABLE`, the round stops, `screening/NOT_RUN.json`
and `full/NOT_RUN.json` are written and **no** F/R/G arm is run.  No
`lr=5e-5`, `lr=1e-5`, optimizer-state resume or from-scratch replacement screen
is authorised in this round.

---

## 4. Residual-augmentation contract and near-zero initialisation (frozen)

Every candidate is a **strict residual augmentation** of CAP-BASE:

```text
zeroing the candidate's new residual projection
    -> prediction == CAP-BASE prediction exactly
```

This is asserted for F, R and G in the frozen step-0 audit
(`residual_zero_max_abs_prediction_delta == 0`) and in the focused tests.

The v1 architectures are reused **unchanged**; only the residual-projection
*scale* changes (a scaling-only initialisation fix, expressly allowed):

| candidate | v1 residual scale | v1 step-0 mean shift | v2 scale | predicted shift |
|---|---|---|---|---|
| F (`F_NP`, `F_EP`) | `FUSION_PROJ_INIT = 0.01` | 0.01053 | `0.0008` | ~0.0008 |
| R (`pair_blocks.*.fc2.weight/bias`) | `RELATION_RESIDUAL_INIT = 0.05` | 0.01948 | `0.0016` | ~0.0006 |
| G (appended reader columns) | `READOUT_READER_INIT = 0.01` | 0.00113 | `0.01` (unchanged) | ~0.0011 |

Frozen target: `mean |prediction(candidate) - prediction(base)| <= 0.002`
(hard), preferred `<= 0.001`; every new module must have finite, strictly
positive step-0 gradient.  If a candidate failed the hard bound or had a dead
(identically zero) gradient, only its initialisation scale may be corrected and
the audit re-run; architecture, width, rank, depth and parameter names must not
change.  The screen starts only after the audit passes for all candidates.

### Step-0 audit record (per F/R/G)

```text
base MAE, candidate MAE, Δ MAE
mean |prediction shift|, median |prediction shift|, max |prediction shift|
per-new-module gradient norms (single deterministic train batch, eval mode)
residual-zero identity max |prediction delta|
shared-state bit-identity report
```

### Phase B metric definitions (frozen)

Per epoch, in addition to Phase A fields, the screen logs the **base /
new-capacity** split of gradient norm and update norm (pre-clip gradient,
post-step parameter delta, group means).  Windows and primary metrics:

```text
screen soup  = Top-5 by valid MAE over epochs 21-40   (primary)
last10       = mean valid MAE over epochs 31-40        (secondary)
best valid   = best single-epoch valid MAE over 1-40
epoch40      = last epoch valid MAE
```

---

## 5. Frozen candidates (identical to v1 except the residual scale)

### F — multi-rank factorized structure-semantic residual fusion

```text
u = u_base + W_NP [ (z W^h_NS) odot (q W^h_NC) / sqrt(24) ]_{h=1..4}
e = e_base + W_EP [ (g W^h_ES) odot (b W^h_EC) / sqrt(24) ]_{h=1..4}
```

`H = 4`, `d_h = 24`, `u_base`/`e_base` reuse the original CSSD binding, added
params 29 568, total 127 295.

### R — stronger static pair composition

```text
h^(l+1) = h^(l) + W2^(l) silu( (1 + tanh(gamma^(l)(r))) odot W1^(l) LN(h^(l)) + beta^(l)(r) )
```

two blocks, hidden 256, FiLM on the existing encoded relation, `gamma = beta = 0`
at initialisation, added params 34 400, total 132 127.  Still single-pass: no
message passing, no write-back, no relation refresh.

### G — gated DeepSets readout augmentation

```text
H_E  = sum_i sigmoid(g_E(E_i)) odot v_E(E_i) / (sum_i sigmoid(g_E(E_i)) + 1e-6)
H_P  = sum_ij sigmoid(g_P(q_ij)) odot v_P(q_ij) / (sum_ij sigmoid(g_P(q_ij)) + 1e-6)
repr = [ existing moments ; H_E ; H_P ; existing global/topology ]
```

summary dim 192, node + pair summaries, all existing moments kept, added
params 30 336 (25 344 summaries + 4 992 appended reader projection), total
128 063.  The CAP-BASE reader stack keeps its names, shapes and bit-identical
weights; the appended columns are the explicit bias-free
`reader.summary_proj` (functionally identical to v1's widened first layer).

### Frozen architecture fingerprints (name:shape sha256, construction-time)

| candidate | added | total | shape fingerprint |
|---|---|---|---|
| M0 | 0 | 97 727 | `4d4d8cebe82f143d860dbfe98b6714e3c3fa3189aa6acc22dde45cc8822096db` |
| F | 29 568 | 127 295 | `e00737abb3b9b7f2e6587c504247bd4206654bfced81f1f8e40d2a0f0815f555` |
| R | 34 400 | 132 127 | `326cf787c72accf3d99d5393fb5b761c18f2bc45cb4f7b679666890b618338fa` |
| G | 30 336 | 128 063 | `f0606a2116039de1c370bb3fb722227682eec1cd48fb2fbc4aec84415d7fce4a` |

Added-parameter ratio max/min 1.1634 <= 1.5; all totals inside the preferred
120 k-160 k band.  v2 parameter counts must equal the v1 counts exactly
(asserted in preflight and in `test_added_parameter_counts_equal_v1`).

---

## 6. Phase B — repaired differential-LR screen (frozen)

Four arms, each exactly 40 epochs, all warm-started from the same CAP-BASE soup:

```text
M0  CAP-BASE continuation control
F / R / G  section 5
```

Fresh optimizer, **no inherited state**, strict two-group partition:

```text
base parameters (every trainable parameter present in the CAP-BASE checkpoint):
    lr = 1e-4
new capacity parameters (F: F_*; R: pair_blocks.*; G: summary_* + reader.summary_proj):
    lr = 1e-3
all groups: weight_decay = 1e-5, Adam, batch 128, clip 5.0
```

`every trainable parameter belongs to exactly one group` is a frozen test.  The
same seed, minibatch order (`seed 0 + TRAIN_SHUFFLE_OFFSET`, shuffle on) and
eval order (`EVAL_SHUFFLE_OFFSET`, shuffle off) as CSSD-q1 seed 0 are used;
construction keeps the global RNG stream identical to CAP-BASE (frozen test), so
dropout streams match across arms.

Primary metric: `M0, M_F, M_R, M_G` = Top-5 soup over epochs 21-40.
Secondary: last-10 means.

Control comparison is **both** matched and absolute:

```text
delta_soup_vs_M0   = M_C - M_0
delta_vs_start     = M_C - 0.130028   (absolute anchor, always reported)
```

---

## 7. Phase B gate and winner rule (frozen)

A candidate C is `CAPACITY_SIGNAL` iff **all** hold:

```text
S1  M_C - M_0        <= -0.003      (matched control)
S2  M_C              <=  0.1270     (absolute anchor M_start = 0.130028)
S3  Last10_C - Last10_0 <= -0.003   (late-window confirmation)
S4  new-branch gradient finite and > 0, new-branch update norm > 0 (frozen
    per-epoch logs; "real branch usage", not dead initialisation)
```

`M_C <= 0.123` upgrades the record to `STRONG_CAPACITY_SIGNAL`.

Winner:

1. no candidate passes -> `LOCAL_CAPACITY_AUGMENTATION_NOT_SUPPORTED`; write
   `full/NOT_RUN.json`; stop.  No widened rank/hidden/depth, no combination, no
   LR change.
2. one passes -> winner;
3. several pass -> lowest screening Top-5 soup wins; if the best two are within
   `0.002`, the frozen tie order **F > R > G** decides.

At most one full run; `F+R`, `F+G`, `R+G`, `F+R+G` are forbidden.

---

## 8. Phase C — the single full winner run (frozen)

Only if a winner exists: one seed-0 trajectory **from scratch**, exactly 320
epochs, identical to the CSSD-q1 seed-0 protocol (same split, seed, data order,
objective, `H1_LAMBDA`, K/s/IHT, C6 mask, Adam `lr = 1e-3`, wd `1e-5`, batch
128, clip 5.0, Top-5 soup over the whole curve).  The only change is the winner
capacity module, including its frozen initialisation scale — the warm screen is
never reported as the final model.

Early stop is allowed **only** for: NaN/Inf; loss divergence; permanent zero
capacity gradient; dictionary numerical failure; valid MAE worse than the
matched historical CSSD curve by `> 0.03` for `>= 40` consecutive epochs with no
improving trend.

Interpretation bands (`M_W` = winner full Top-5 soup):

```text
M_W > 0.127        -> SHORT_SCREEN_SIGNAL_DID_NOT_TRANSFER
0.125 < M_W <=0.127-> FULL_CAPACITY_GAIN_NOT_ESTABLISHED   (neutral)
M_W <= 0.125       -> CAPACITY_DIRECTION_SUPPORTED_SINGLE_SEED
M_W <= 0.120       -> NEW_PERFORMANCE_BAND_SINGLE_SEED
M_W <= 0.110       -> MAJOR_CAPACITY_BOTTLENECK_IDENTIFIED
```

The mentor's ~0.06-0.062 reference band is **not** a gate this round; the
question is only whether a capacity direction leaves the 0.12-0.13 plateau.

---

## 9. Post-full mechanism audit (frozen, cheap)

Only on a completed full winner, official-valid split only, winner mask merged
into every probe mask:

* **P1** dictionary coordinate row shuffle (distribution preserving), 3 seeds;
* **P2** assignment-preserving node correspondence shuffle, 3 seeds;
* **P3** assignment-preserving edge correspondence shuffle, 3 seeds;
* **P4** relation row shuffle (`all`, `distance`, `overlap`, `boundary`), 3 seeds;
* **P5** winner branch disable (`capacity_off`): Δ MAE, mean/max shift.

Candidate-specific (no rescue): F per-head output norm / gradient norm /
pairwise output cosine; R per-block residual norm, per-block disable Δ, FiLM
modulation norm; G node/pair summary norm, gate statistics, per-summary disable
Δ, `reader.summary_proj` norm.

---

## 10. Forbidden this round

New F/R/G architectures; head-count/rank/hidden/depth sweeps; LR or optimizer
sweeps (no second warm-adaptation protocol, no `5e-5`, `3e-4`, `1e-5`, `3e-3`);
from-scratch training of all four arms; F+R / F+G / R+G / F+R+G; sparse vs
dense; IHT modifications; CSSD q2; new handcrafted topology or chemistry
features; seeds 1/2; official test.  No post-hoc threshold, scale or gate change
after the first training process starts.

---

## 11. Result layout

```text
tracks/ksvd/results/e2e_dictenv_capacity_localization_v2/
    preregistration_snapshot.json
    parameter_budget.json
    architecture_fingerprints.json
    preflight.json
    init_audit.json
    calibration/
        m0/{curve.csv, soup.json, result.json, soup_state.pt}
        calibration_summary.json
        calibration_decision.json
    screening/
        m0/ fusion/ relation/ readout/{result.json, curve.csv, soup.json, soup_state.pt}
        screening_summary.csv
        decision.json
        NOT_RUN.json            (only when calibration fails)
    full/
        winner/{curve.csv, soup.json, final.json, checkpoints/}
        NOT_RUN.json            (only when no winner)
    mechanism/
        probes.json, probes.csv, candidate_specific.json
    analysis_tables.md
    summary.json
    REPORT.md
    DECISION.md
```

## 12. Frozen implementation rule

The preregistration sha256 is written into `preregistration_snapshot.json` by
`preflight` before the first training process.  After that only clearly labelled
harness/provenance fixes are allowed and they may not change a decision.
Determinism cross-check: the M0 screen arm's epochs 1-20 must reproduce the
Phase-A calibration curve exactly; the comparison is recorded in
`screening/decision.json` (`calibration_vs_screen_m0`).

## 13. Questions the final report must answer

1. Does `Adam(lr = 1e-4)` M0 hold the CSSD soup basin (calibration gate)?
2. Was the v1 failure mainly the `lr = 1e-3` warm restart?
3. Under the stable protocol, do F/R/G produce a real absolute improvement
   rather than only slowing drift?
4. Does any candidate beat both the matched M0 by `>= 0.003` and the original
   checkpoint to `<= 0.1270`?
5. If a winner exists, does the short warm signal reproduce from scratch?
6. Does the winner still depend on dictionary coordinate, node/edge semantic
   correspondence and relation?
7. If nobody passes, can simple local widening be excluded as the main route out
   of the 0.13 plateau (and should the next round test one-shot static
   composition itself, e.g. controlled iterative environment composition)?
