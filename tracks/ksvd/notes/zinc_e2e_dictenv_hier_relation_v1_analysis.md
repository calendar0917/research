# Analysis — ZINC E2E-DictEnv-Hier-Relation-v1 (single-model hierarchical static relation dictionary, seed 0, local CPU)

Preregistration: `notes/zinc_e2e_dictenv_hier_relation_v1_preregistration.md`
(committed `a90584b`, before the formal runs).
Authoring/executed revision: `cc4cf83` (final tracked tree, four commits:
`a90584b` prereg+code, `f25984a`/`934ed97`/`cc4cf83` runner/report bugs found by a
two-epoch dry run **before** any formal run). Runs:
prepare `20261001-173750-21b32689` (scratch, 30.9 s), screen
`20261001-173831-2814f28e` (screen, 1328.3 s wall, 320/320 epochs).
Evidence: `results/e2e_dictenv_hier_relation_v1/{REPORT.md, DECISION.md,
summary.json, run_HIER-RELATION.json, interventions.json, correctness.json,
smoke.json, curve_HIER-RELATION.csv}`.

## 1. Verdict

`HIERREL_STOP`. Soup official-valid MAE **0.182343094** (members
291/294/313/316/320 = 0.193753 / 0.195304 / 0.194067 / 0.193210 / 0.195031;
best single epoch 0.193210 @ 316). The pre-registered band rule
(`<= 0.115 / <= 0.120 / <= 0.1233 / else stop`) puts the candidate **0.0590 MAE
above the borderline edge** — not a near miss. The band gap is not a horizon
artifact: no epoch in the last 220 had valid MAE below 0.19, the best epoch is
316 of 320, and the last-20 mean is 0.2022.

**The band and the mechanism are separate.** Mechanism readout (below) is
*positive*: this round's relation dictionary is **not** inert, unlike the
previous Joint709 round. That does not move the absolute verdict.

| quantity | value |
|---|---|
| soup valid MAE | 0.182343094 |
| soup train MAE (official train 10 000) | 0.091181269 |
| train–valid gap | -0.091162 |
| valid: first / last / last-20 mean | 0.574420 / 0.195031 / 0.202198 |
| min valid over epochs 101–320 | 0.193210 (epoch 316) |
| epochs / seconds per epoch / wall | 320/320, 4.128 s, 1320.9 s |
| peak RSS / parameters | 1.77 GB, 145 961 trainable (expected 145 961) |

Historical context only, **not a matched control** and not usable as one: the
previous round of this track on the same official split/protocol
(Joint709, `claim-e2e-dictenv-joint709-absolute-v1-stop-inert-coordinate-20261001`)
reached 0.132442 with a *different architecture* — it inherited the frozen
Sem108 + φ65 pipeline, a 49-wide coordinate, and roughly 100 k parameters in a
much larger reader. The present round is a from-scratch 146 k-parameter model
whose only input path is one linear 712→128 dictionary plus one 586→64→32→1
head. The 0.182 result therefore says nothing about the older pipelines; it only
says that **this** single-model static candidate does not enter a better band.

## 2. Why this is a negative result and not an implementation failure

Every pre-registered correctness gate passed (`correctness.json`,
`all_passed = true`): 712-D input / 228-D relation object / 586-D readout widths;
each undirected physical edge counted exactly once with the right type and graph
ownership (249 279 train / 24 846 valid unique edges, spot-checked against raw
ZINC); endpoint swap 0.0, relabel 7.08e-08, batch composition 0.0 against a
1e-5 tolerance; grouping never crosses graphs and singleton groups have exactly
`delta = 0`; endpoint re-pairing leaves `z`, `x_hat`, node pooling and `mu`
bitwise unchanged while the relation object and the prediction move (no
write-back); both dictionaries receive finite non-zero **MAE-only** gradients and
their normalised directions actually move under a MAE-only Adam step; the
reconstruction formulas match a hand recomputation with relation code
`l0 = 8` exactly; zero/shuffle interventions touch only the target branch; the
official test split was never instantiated (`official_test_loaded = false`).
Smoke passed (3 epochs on 2048/512). The run was 320/320 epochs with a stable
4.0–4.2 s/epoch and 1.77 GB peak RSS.

So the failure is architectural, not a broken gate, a truncated horizon, a dead
channel, or a NaN/denormal collapse.

## 3. What the model actually learned (mechanism, reported separately)

All four endpoint interventions on the soup are **far** above the pre-registered
`RMS > 1e-4` sensitivity floor (official valid, 1000 molecules):

| intervention (soup) | valid MAE | ΔMAE | Δ pred RMS | mean abs Δpred |
|---|---|---|---|---|
| `zero_beta` (relation codes) | 0.655395 | **+0.473052** | -0.2754 | 0.6119 |
| `zero_node_pool` | 0.827275 | **+0.644932** | -0.6154 | 0.7882 |
| `zero_delta_rel` (deviation channel) | 0.412621 | **+0.230278** | -0.1181 | 0.3539 |
| `permute_endpoints_11` | 0.388458 | +0.206115 | -0.1183 | 0.3353 |
| `permute_endpoints_22` | 0.382102 | +0.199758 | -0.1085 | 0.3213 |

The relation code channel, the shared/deviated grouping channel (`delta`) and
the within-(molecule, atom type) endpoint pairing are all load-bearing: replacing
`delta` by the group mean plus re-pairing endpoints inside the group degrades
valid MAE by 0.20–0.23, and switching the whole relation code off degrades it by
0.47. This is a genuine difference from the previous round, where the analogous
probe was **exactly 0.0** (denormal-floor inert coordinate).

Both dictionaries were genuinely trained, not frozen at their spectral
initialisation:

| quantity | node `D_N` | relation `D_R` |
|---|---|---|
| column-normalised cosine, init → soup | 0.625 | **0.265** |
| ‖soup − init‖ (normalised columns) | 9.80 | 9.70 |
| raw norm init → soup | 11.31 → 26.49 | 8.00 → 15.36 |
| mean pairwise column coherence (soup) | -0.0001 | +0.0014 |
| soup vs best-state cosine | 0.9986 | 0.9975 |

The relation dictionary rotated almost completely away from its spectral
initialisation (cosine 0.27) but stayed incoherent (no collapsed atoms), and the
Top-5 soup is a mild average of nearly identical late states (cosine 0.998), i.e.
the 0.0109 soup gain over the best single epoch is variance reduction, not
weight-space diversity.

One internal-consistency caveat is recorded rather than smoothed over: the
per-molecule reconstruction error of the *initial* dictionary varies strongly
with the batch used. Molecule 0 alone gives `R_N = 0.0055`, whereas the first 256
train molecules (5 923 nodes, the probe batch) give `R_N = 0.3411`; the
`correctness` gate happened to run on a single-molecule batch and therefore
quotes 0.004538. Both numbers are the same quantity computed by the same
formula; the gate's value is not representative of the population and §4 uses
the 256-molecule value. The gate itself (hand recomputation matching the model
output to 0.0) is unaffected.

Code health at the endpoint (official valid): node code norm mean 1.846,
per-coordinate variance 0.0244, activation rate 0.989 of the 128 coordinates
**all kept, no top-k**; relation codes `l0 = 8.00` exactly, norm mean 1.063,
top-1 share 0.448, effective atoms 16.9 of 64. So the sparse relation code is
neither collapsed to one atom nor close to dense — the tied-IHT s=8 regime is
being used as designed.

MAE-only gradients into both normalised dictionaries stay finite and non-zero
for all 320 epochs, and the *task* term dominates the auxiliary reconstruction
terms by a wide margin (epoch 320: ‖∂MAE/∂D̄_N‖ = 3.12, ‖∂MAE/∂D̄_R‖ = 0.404,
versus a λ_N = 0.05 / λ_R = 0.02 auxiliary weight):

| epoch | grad D̄_N | grad D̄_R | z norm | β norm | β l0 | R_N | R_R | ΔMAE zero_beta | ΔMAE zero_node | ΔMAE zero_delta |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.269 | 0.037 | 1.844 | 1.338 | 8.00 | 0.341 | 0.402 | +0.020 | -0.018 | +0.009 |
| 5 | 2.309 | 0.460 | 1.867 | 1.390 | 8.00 | 0.422 | 0.575 | +0.180 | +0.742 | +0.182 |
| 20 | 0.624 | 0.157 | 1.849 | 1.186 | 8.00 | 0.416 | 0.327 | +0.255 | +0.831 | +0.185 |
| 80 | 4.484 | 0.646 | 1.844 | 1.124 | 8.00 | 0.332 | 0.231 | +0.318 | +0.733 | +0.182 |
| 160 | 10.414 | 1.343 | 1.857 | 1.097 | 8.00 | 0.203 | 0.193 | +0.358 | +0.667 | +0.160 |
| 240 | 7.298 | 0.898 | 1.856 | 1.081 | 8.00 | 0.170 | 0.181 | +0.416 | +0.665 | +0.220 |
| 320 | 3.121 | 0.404 | 1.866 | 1.094 | 8.00 | 0.142 | 0.182 | +0.496 | +0.728 | +0.255 |

The probe is the same 256-molecule train batch every time and the *channel
dependence grows monotonically* over training (zero_beta ΔMAE 0.020 → 0.496,
zero_delta 0.009 → 0.255): the model starts by ignoring the relation code and
ends by relying on it. That is the opposite trajectory of the Joint709 collapse.

Relation input block energy in the *scaled* relation object drifts away from the
frozen calibration (fit-time 1.000 per block) because the node dictionary keeps
moving while the scaler is frozen: `mu` 1.27 → 1.07, `delta` 0.79 → 0.83,
`mu_delta` **5.71 → 4.47** (peak 8.58 at epoch 5), `bond` 1.02 (frozen one-hot).
The cross-term block is ~4.5× hotter than the calibration target for the whole
run. This is a recorded property of the frozen-scaler design, not a defect: the
relation dictionary absorbs the drifting scale, and no refit was authorised.

## 4. Where the error actually is

The node dictionary is a linear 128-atom map of a 712-D input, so its
reconstruction error is a real capacity statement. On the first 256 train
molecules with the **initial** dictionary, `R_N = 0.3411` overall, and the
residual is almost entirely in two blocks:

| input block | dims | share of residual (init) | per-column R |
|---|---|---|---|
| struct residual (frozen 65) | 65 | 0.6 % | 0.013 |
| Sem108 | 108 | **48.3 %** | 0.445 |
| RoleCorr | 536 | **51.0 %** | 0.665 |
| common1 (subspace) | 1 | 0.0 % | 0.000 |
| size2 | 2 | 0.1 % | 0.007 |

The spectral initialisation keeps the globally high-energy directions and misses
exactly the sparse, high-magnitude-per-row columns of the Sem108/RoleCorr blocks
(e.g. columns 85 and 113 carry their full energy as residual). The λ_N = 0.05
term then does its job: `R_N` falls 0.341 → 0.143 on the probe batch
(0.079 full train, 0.071 valid at the soup), while the relation code keeps
`R_R ≈ 0.18–0.27` and never gets close to the node channel's quality. **The
auxiliary reconstruction objective was active and useful; it simply is not the
binding constraint on the prediction error.**

The binding constraint is visible in the train/valid gap: soup train MAE
0.0912 versus valid 0.1823 (-0.0912), with train min 0.1385 and no epoch in the
last 220 below 0.19 valid. A 146 k-parameter model fits 10 000 molecules much
better than it generalises — this is an overfitting/regularisation regime, not
the underfitting regime the older inherited pipelines were in (their train and
valid curves moved together). Combined with the mechanism evidence (all channels
alive, code health fine, all gates passed), the honest reading is: *the
hierarchical static relation dictionary works as a mechanism inside this model,
and the model is still 0.06 MAE short of the frozen borderline band.*

## 5. What this round does and does not claim

Established: (a) the exact architecture of §2 of the preregistration is
implemented, gated and reproducible; (b) with the relation object built from a
learned node dictionary's shared/deviated environments, the sparse relation code,
the deviation channel and the endpoint pairing all carry prediction at the
endpoint (interventions 0.20–0.65 MAE, 4 orders of magnitude above the RMS
floor); (c) the absolute performance of this candidate is `stop`
(0.182343 > 0.1233), with a train/valid gap that identifies overfitting rather
than a dead channel as the limitation.

Not claimed: no matched-training control exists (no dense/PCA arm, no
random/shuffled-dictionary arm), so **nothing** is said about sparse-vs-dense or
dictionary-vs-no-dictionary; no significance claim (one seed); no official-test
statement; no comparison against A/B/C, Sem108, Joint709 or the legacy RPD
numbers as if the architectures were matched; the earlier rounds' numbers may
not be used as a control for this one.

The negative band result and the positive mechanism result must be reported
together and never merged into "the relation route works" or "the dictionaries
are useless".

## 6. One recommended next question (not authorised by this round)

The mechanism is alive and the capacity is 0.06 MAE short, with the gap sitting
in generalisation rather than in a dead channel. The single most informative
next purchase is therefore **not** another relation-dictionary variant
(larger K, different s, stacked layers, message passing — all still forbidden
by design), but a *diagnosis of where the 0.06 comes from*: a capacity/regularity
split of the same frozen architecture (e.g. train-set size 10 000 vs 5 000 vs
2 500 and/or dropout/weight-decay-only variation at a fixed epoch budget) would
say whether the 146 k-parameter static model is data-limited or
regularisation-limited, before any further structural idea is bought. That
question needs its own preregistration, its own matched arms and, if it is meant
to be a performance claim, paired seeds.

## 7. Provenance

* Revision `cc4cf83dbce3c40977c36ff5125a47c73afd525c`; tracked tree unchanged
  during the run (only untracked result files and the two pre-existing untracked
  user result dirs).
* Frozen inputs re-verified at run time: joint scaler `0ea418fa…`, joint cache
  sha256 `b7da4437…` (train) / `224adf5a…` (valid), common subspace
  `36636ce9…`, env caches and role-correspondence caches; joint sample
  recomputation versus the cache 8.98e-7 (train) / 4.53e-7 (valid) against a
  1e-5 tolerance.
* Prepared objects: node input sha256 `0ec77547…` (train) / `f9762156…` (valid);
  extra scaler (`53cc14eb…`, fit on 231 664 train rows): common block energy
  1.000000, size block energy 2.000000 over its two columns (weight 1/√2),
  per-column train rms [1.000000, 1.000000, 1.000000], valid block energy
  1.00022 / 0.98825, i.e. the two train-frozen blocks are scale-consistent on
  valid to 1.2 %; relation scaler fit on all 249 279 train edges with
  scaled block energy 1.000 each; projection `P` sha256 `e22f9978…` (seed
  20261001); node dictionary init sha256 `08f24b38…`, relation dictionary init
  sha256 `55c31d86…`, init state sha256
  `926cddc6bd3520e26e482f2e4dc9b8c5d90b864b7afaed149ee2515fad3d4943`, soup
  `757596b2…`, best `769ce9f8…`. Both spectral initialisations had effective rank
  equal to the atom count (128 / 64), zero supplements, orthogonality error
  < 2.3e-15.
* Env occurrence ratio: 1 232 844 directed occurrence entries → 249 279 unique
  undirected physical edges (4.9456) on train, 122 934 → 24 846 (4.9478) on
  valid.
* A pre-existing convention nuance is recorded, not repaired: the upstream
  `cache_meta_joint_*.json` field `sha256_f32` hashes the float64 array *before*
  float32 serialisation, so it does not reproduce against the shipped file (the
  shipped-file hashes above are the ones this round verifies).
* Wall clock: prepare 30.9 s, screen 1328.3 s (4.128 s/epoch), peak RSS 1.77 GB,
  CPU 8 threads.
