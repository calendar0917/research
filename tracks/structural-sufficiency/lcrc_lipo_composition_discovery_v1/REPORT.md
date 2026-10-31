# LCRC Lipophilicity Composition Discovery v1

**Protocol:** `lcrc-lipo-composition-discovery-v1`
**Decision:** `NO_PROMISING_NEW_ARCHITECTURE`
**Frozen strong reference:** `ROLEBLIND_s0`, Validation420 RMSE `0.534232`
**Test420:** not read for metrics; no Test prediction, residual or Test metric was computed.

## 1. Short answer

Eight non-iterative structure-composition corrections were trained on the frozen
Lipophilicity random42 split with the frozen LCRC recipe.  None beat the strong
`ROLEBLIND_s0` reference by the predeclared `0.008` Validation-RMSE gate.  The
closest new arm, `g2l_loo_s0`, was `0.00296` **worse** than `ROLEBLIND_s0`
(paired descriptive 95% interval `[-0.02059, +0.01446]`, centred below zero).
The two strict mechanism pairs were also negative:

* `PATH-BIND` was `0.00190` worse than `PATH-UNBOUND`; the paired interval
  `[-0.01816, +0.01423]` includes zero.  Binding changes the per-root summary
  materially (mean binding-gap norm `0.364`, about `28%` of the summary norm)
  but does not improve Validation RMSE.
* `DIST-TRUE` was `0.04489` worse than `DIST-BLIND`; the paired interval
  `[-0.07874, -0.01022]` is entirely below zero.  Actual D34/D5 structural
  content is harmful here, not helpful.

The one positive mechanism result is the existing `ROLEBLIND` mechanism:
`ROLEBLIND_s0` beat its graph-level local-cap control by `0.04085`
(interval `[+0.01259, +0.06865]`).  That says all-atom context is better than
an atom-local context, but it is not a new architecture and was already known
from the context-mechanism round.

A mandatory seed check changes the interpretation of the whole screen.
`ROLEBLIND_s1` reached `0.586765`, which is **worse than `BASE_s1`**
(`0.576780`, gain to ROLEBLIND `-0.00998`, interval `[-0.03900, +0.01774]`) and
`0.05253` worse than `ROLEBLIND_s0` (interval `[-0.09589, -0.01177]`).  The
seed-0 advantage of ROLEBLIND over the new arms is therefore not a stable
architecture ranking: it is within a very large seed-to-seed spread.  No new
model is promoted, no Stage-III scaffold and no PATH-SELECT run were started,
and Test420 remains unread for metrics.

## 2. Question and constraints

The round asks whether stronger **non-iterative molecular structure-composition
models** can be built from the frozen LCRC local structural statistics and the
S2 shortest-path statistics, and which composition mechanisms beat their strict
matched controls on Lipophilicity random42.

Constraints observed:

* no ordinary message passing, generic Transformer, multi-layer full-graph
  attention, or pretraining in the main experiment;
* no attention was used at all;
* all new heads are single-shot corrections over a frozen local encoder, not
  iterative graph networks;
* no retry of `PROTO_K16_TOP4` or the terminal `PAIR_TRUE` global residual;
* these modules are not called a dictionary and no transferable primitive
  dictionary is claimed;
* frozen splits, frozen labels, frozen cache, frozen Train-only target
  normalisation and frozen training recipe;
* Validation420 is the screening metric; Test420 was not used for metrics;
* one GPU host (`res`), pools `res-gpu0` / `res-gpu1`, one training process per
  pool at a time, 14 A100 GPU-hour hard cap.

## 3. Frozen data and protocol

| item | value |
|---|---|
| split file SHA256 | `1834414a6da73e9d2c5942ad157a345f08653110acfc4b9fef83c3064bc233b4` |
| labels file SHA256 | `1da269356e0e89974a7388d1e430d7e810a0ab1056be5cd747a1c8ed3d2009ca` |
| Test rows SHA256 | `4bc449106a64833973b8e8d3d16097bf5d2906c788d7dfcdf02a4b3a30ffa49b` |
| Test label vector SHA256 | `a0724b4c1ca6f7bb6b69b720ac2d020825685d9666544c7312e0f24d7e3a4021` |
| Train3360 | 3360 molecules |
| Validation420 | 420 molecules |
| Test420 | 420 molecules, `EXPOSED_TEST_EXPLORATORY` |
| target mean (Train3360 only) | `2.2081011904761905` |
| target scale (Train3360 only) | `1.1906281212426577` |
| LCRC cache fingerprint | `23ec00be4fb7e3cf93967cd62911bd932159b1d7e0b43226b23cf7c6a3718e03` |

The frozen audit verifies the split hashes, row order, label vectors,
Train-only target normalisation and the LCRC cache.  The Test label vector is
hash-checked in that audit, but no Test prediction, Test metric, Test residual
or Test-based selection was produced.  `analysis/eval_screen/` contains
Validation420 only.  Because no candidate was promoted, there is no Test
evaluation directory for this round.

### Frozen training recipe

AdamW, batch 128, 360 epochs (27 steps/epoch, 9,720 updates), peak LR `3e-4`
to `1e-5` per-step cosine, weight decay `1e-3`, gradient clip `5.0`, MSE on the
Train3360-standardised target, dropout `0.10` on the final atom-level local
representation, and a last-five-epoch state-dict soup.  No early stopping and
no Validation-based checkpoint selection.

### Models

`BASE` is the frozen LCRC model.  `CONTEXT` and `ROLEBLIND` are the historical
frozen mechanisms.  All new corrections are zero-initialised at their output
projection, so the initial model is bitwise `BASE`; CPU gates verify that
zeroing each correction output exactly recovers `BASE` at the same weights.

| family | new parameters | correction summary |
|---|---:|---|
| `BASE` | 0 | no correction |
| `CONTEXT` / `ROLEBLIND` | 8,352 | historical hetero-only context correction; ROLEBLIND uses all-atom mean for both role vectors |
| `g2l_all` | 8,352 | all-atom mean role vectors; correction added to **all** atoms |
| `g2l_loo` | 8,352 | per-atom leave-one-out mean role vectors; correction added to hetero atoms only |
| `g2l_local_cap` | 8,352 | role vectors from the atom's own `h_i`; correction added to hetero atoms only |
| `g2l_scalar` | 6,273 | graph-level scalar `delta` from all-atom mean role vectors; added to every atom's readout |
| `dist_true` | 8,352 | D34/D5 mean structural content feeds the correction |
| `dist_blind` | 8,352 | same D34/D5 existence/routing, but content replaced by the leave-one-out all-atom mean |
| `path_bind` | 8,192 | `mean_p(v_p * r_p)` over S2 path instances |
| `path_unbound` | 8,192 | `mean_p(v_p) * mean_p(r_p)` over the same S2 path instances |

The distance buckets are label-free and defined per root atom:

* `D34`: atoms at shortest-path distance 3 or 4 (excluding self and distance
  <= 2, finite only);
* `D5`: atoms at shortest-path distance >= 5 (same exclusions).

The S2 path representation uses the frozen path category (16 values) and the
two endpoint roles (5 values each), with learned embeddings and a learned
content vector.  `PATH-BIND` and `PATH-UNBOUND` share the same embeddings,
content encoder and output projection; only the per-root aggregation differs.

## 4. CPU gates and CUDA smoke

`analysis/cpu_gates.json`: **16/16 PASS**.

The gates cover data hashes, Train-only target normalisation, shared
initialisation/parameter identity, exact BASE recovery when the correction is
zeroed, independently recomputed distance buckets, path-category semantics,
BIND/UNBOUND invariants, batch/cross-molecule invariance, atom-renumbering and
path-order invariance, finite non-zero gradients, rho reading the original `h`,
dropout activity, frozen-baseline reproduction, immutable frozen inference,
metric arithmetic, and no Test access in training/model code.

`analysis/smoke_cuda/` contains an 8-arm CUDA smoke run on `res-gpu0`
(1 epoch, 64 Train / 32 Validation molecules).  All eight arms completed and
all gradient probes passed.

## 5. Stage-I Validation420 screen

Sorted by Validation RMSE.  `gain` is `ROLEBLIND_s0 RMSE - model RMSE`; a
positive gain would mean the new model is better.  All values are raw logD.

| model | family | params | Train RMSE | Valid RMSE | Valid MAE | Valid R² | gain vs ROLEBLIND_s0 | wall s |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `roleblind_s0` | roleblind | 255,458 | 0.228647 | **0.534232** | 0.386601 | 0.811159 | 0.000000 | 402.3 |
| `g2l_loo_s0` | g2l_loo | 255,458 | 0.232173 | 0.537189 | 0.389804 | 0.809063 | -0.002957 | 654.4 |
| `dist_blind_s0` | dist_blind | 255,458 | 0.207587 | 0.546511 | 0.399412 | 0.802379 | -0.012279 | 610.6 |
| `g2l_all_s0` | g2l_all | 255,458 | 0.215922 | 0.546849 | 0.401390 | 0.802134 | -0.012617 | 637.2 |
| `g2l_scalar_s0` | g2l_scalar | 253,379 | 0.220311 | 0.550770 | 0.398828 | 0.799286 | -0.016538 | 634.6 |
| `path_unbound_s0` | path_unbound | 255,298 | 0.248463 | 0.572556 | 0.415662 | 0.783094 | -0.038324 | 633.4 |
| `path_bind_s0` | path_bind | 255,298 | 0.243096 | 0.574454 | 0.413803 | 0.781653 | -0.040222 | 623.3 |
| `g2l_local_cap_s0` | g2l_local_cap | 255,458 | 0.251674 | 0.575080 | 0.416431 | 0.781177 | -0.040848 | 636.4 |
| `base_s0` | base | 247,106 | 0.253621 | 0.580416 | 0.427854 | 0.777097 | -0.046184 | 312.0 |
| `dist_true_s0` | dist_true | 255,458 | 0.251040 | 0.591401 | 0.430884 | 0.768580 | -0.057169 | 614.9 |

The best new arm is still below the strong reference.  The paired descriptive
interval for `ROLEBLIND_s0` minus `g2l_loo_s0` is `[-0.02059, +0.01446]`;
the point estimate favours ROLEBLIND but the interval includes zero.  For
`g2l_all_s0` the corresponding interval is `[-0.04061, +0.01535]`.

### Interpretation of the G2L variants

The ordering inside the G2L family is informative even though none is a
promotion candidate:

* `g2l_loo` (0.5372) is closest to `ROLEBLIND_s0`;
* `g2l_all` (0.5468) is worse, so adding the correction to carbon/halogen atoms
  as well as hetero atoms did not help;
* `g2l_scalar` (0.5508) is only slightly worse than `g2l_all`, despite having
  no atom-specific correction vector.  Much of the useful signal is therefore
  graph-level or shared, not an atom-specific composition code;
* `g2l_local_cap` (0.5751) is much worse, confirming that a context vector made
  only from the atom's own `h_i` is insufficient.

The `ROLEBLIND` vs `g2l_local_cap` pair formalises the last point: all-atom
aggregation is worth `+0.04085` RMSE over the local cap (interval
`[+0.01259, +0.06865]`).  This is evidence for the existing all-atom context
mechanism, not evidence for a new architecture.

## 6. Strict mechanism pairs

`gain_true_vs_control > 0` means the true mechanism is better than its matched
control.  Intervals are paired molecule-level percentile bootstraps
(descriptive only).

| pair | true | control | RMSE true | RMSE control | gain | 95% interval | true better on | gate |
|---|---|---|---:|---:|---:|---|---:|---|
| PATH-BIND vs PATH-UNBOUND | `path_bind_s0` | `path_unbound_s0` | 0.574454 | 0.572556 | -0.001898 | `[-0.01816, +0.01423]` | 202/420 | fail |
| DIST-TRUE vs DIST-BLIND | `dist_true_s0` | `dist_blind_s0` | 0.591401 | 0.546511 | -0.044890 | `[-0.07874, -0.01022]` | 193/420 | fail |
| ROLEBLIND vs LOCAL-CAP | `roleblind_s0` | `g2l_local_cap_s0` | 0.534232 | 0.575080 | +0.040848 | `[+0.01259, +0.06865]` | 232/420 | pass (existing mechanism) |

### PATH-BIND

The binding gap is real but not useful for this target.  On Validation420, for
roots with paths (`11,099` roots):

| statistic | PATH-BIND | PATH-UNBOUND |
|---|---:|---:|
| mean `norm(bind - marginal)` | 0.3637 | 0.3519 |
| 95th percentile gap norm | 0.9860 | 0.9692 |
| mean relative gap | 0.2833 | 0.2752 |
| mean bound norm | 1.5254 | 1.5351 |

Both models reach essentially the same Validation RMSE (`0.57445` vs
`0.57256`), and the paired interval includes zero.  `PATH-BIND` fits the
training set slightly better (`0.24310` vs `0.24846`) but generalises slightly
worse.  The added relational specificity therefore behaves like extra
capacity/overfitting, not like a useful composition mechanism.  There is no
basis for PATH-SELECT.

### DIST-TRUE

`DIST-TRUE` is unambiguously worse than its matched control.  The control keeps
the same D34/D5 existence and routing but replaces the bucket content with the
leave-one-out all-atom mean; that control reaches `0.54651`, while true D34/D5
content reaches `0.59140`.  The paired interval is entirely below zero.  This
does not support a claim that short/long-range structural content is a useful
non-iterative composition primitive for Lipophilicity under this recipe.

## 7. Stage-II seed check

Only `ROLEBLIND_s1` was run, because no new model passed the Stage-I
performance gate.  It was paired with the already frozen `BASE_s1`.

| model | family | seed | Train RMSE | Valid RMSE | Valid MAE | Valid R² |
|---|---|---:|---:|---:|---:|---:|
| `base_s0` | base | 0 | 0.253621 | 0.580416 | 0.427854 | 0.777097 |
| `base_s1` | base | 1 | 0.268944 | 0.576780 | 0.422146 | 0.779881 |
| `roleblind_s0` | roleblind | 0 | 0.228647 | **0.534232** | 0.386601 | 0.811159 |
| `roleblind_s1` | roleblind | 1 | 0.254568 | 0.586765 | 0.429352 | 0.772194 |

Key differences:

* `ROLEBLIND_s1` is `0.009985` **worse** than `BASE_s1` (paired interval
  `[-0.03900, +0.01774]`); the point estimate does not reproduce the
  ROLEBLIND advantage at seed 1.
* `ROLEBLIND_s1` is `0.052533` worse than `ROLEBLIND_s0` (interval
  `[-0.09589, -0.01177]`).
* The ROLEBLIND seed spread is `0.052533`; the BASE seed spread is only
  `-0.003636`.
* Correction-zeroed ROLEBLIND readout is very weak: `1.22530` for seed 0 and
  `1.26158` for seed 1, versus full-model `0.53423` / `0.58676`.  The learned
  correction is carrying most of the signal, and its usefulness is seed
  sensitive.

This seed check is the most important result for interpretation.  Single-seed
Validation differences among the new arms (0.003-0.04) are below or comparable
to the observed ROLEBLIND seed spread.  The screen can reject the two strict
mechanism pairs as stated, but it cannot establish a stable ranking of the
non-promoted G2L variants.  It also means `ROLEBLIND_s0`'s apparent edge over
`g2l_loo_s0` should not be treated as a reliable architecture conclusion.

## 8. Diagnostics and correction usage

All eight new arms passed the post-training diagnostics:

* finite non-zero gradients on every new tensor and parameter updates after
  two probe steps;
* `rho` received statistics of the original `h`, not `h_new`;
* DropoutAtLocal masked atoms in train mode and was the identity in eval mode;
* zeroing the correction output recovered BASE exactly at initialisation;
* no non-finite training events;
* gradient clipping was active (about 4.5k-4.8k clip events per 360-epoch run),
  so the recipe was operating in its usual clipped regime.

Validation correction magnitudes (`change_l2_mean` / `change_l2_p95`):

| model | change mean | change p95 |
|---|---:|---:|
| `g2l_all_s0` | 1.055 | 2.187 |
| `g2l_loo_s0` | 0.612 | 3.470 |
| `g2l_local_cap_s0` | 0.582 | 4.255 |
| `g2l_scalar_s0` | 0.000 (additive scalar) | 0.000 |
| `dist_true_s0` | 1.391 | 2.738 |
| `dist_blind_s0` | 1.310 | 2.472 |
| `path_bind_s0` | 0.900 | 1.934 |
| `path_unbound_s0` | 0.862 | 2.049 |

D34/D5 coverage on Validation420: `66,946` D34 entries over `11,098` atoms
with at least one D34 target; `178,594` D5 entries over `11,019` atoms.  Path
coverage: every atom has at least one S2 path instance in the frozen cache.

Correction-zeroed Validation RMSE is tabulated in
`analysis/table_correction_ablation.csv`.  The largest degradation is for
`g2l_scalar` (+0.760), `g2l_loo` (+0.716), `ROLEBLIND_s1` (+0.675) and
`ROLEBLIND_s0` (+0.691); the smallest is for `PATH-BIND` (+0.091) and
`PATH-UNBOUND` (+0.099).  A large correction-zeroed degradation means the base
path is not independently competitive once the correction is removed; it is
not a measure of mechanism quality.

## 9. Decision

`NO_PROMISING_NEW_ARCHITECTURE`.

* No new arm passed the predeclared Stage-I performance gate
  (`gain >= 0.008` vs `ROLEBLIND_s0` and MAE change `<= 0.005`).
* No strict mechanism pair supports its true branch:
  PATH-BIND is slightly worse than PATH-UNBOUND, and DIST-TRUE is clearly
  worse than DIST-BLIND.
* The only positive strict pair is the existing ROLEBLIND-vs-local-cap
  comparison, which supports all-atom context aggregation, not a new model.
* `ROLEBLIND_s1` is worse than `BASE_s1` and much worse than
  `ROLEBLIND_s0`; the seed spread is large enough that the seed-0 ranking of
  the new arms is not a reliable architecture ranking.
* Stage-III scaffold and PATH-SELECT were not run.
* Test420 was not used for metrics.  No Test predictions or Test metrics exist
  for this round.

The honest conclusion is not that "composition does not matter"; it is that
these particular non-iterative composition corrections did not produce a
reliable gain over the strong ROLEBLIND reference on this frozen split and
recipe.  The one robust signal is that a graph-level/all-atom context is much
better than an atom-local cap, which was already the motivation for
ROLEBLIND.

## 10. Reproduction and artifacts

Key commands (local CPU gates, then remote CUDA):

```bash
uv run python tracks/structural-sufficiency/lcrc_lipo_composition_discovery_v1/code/run_dc.py audit
uv run python tracks/structural-sufficiency/lcrc_lipo_composition_discovery_v1/code/run_dc.py gates
uv run python tracks/structural-sufficiency/lcrc_lipo_composition_discovery_v1/code/run_dc.py freeze
uv run pytest -q tracks/structural-sufficiency/lcrc_lipo_composition_discovery_v1/tests
```

Formal arms were launched through `rr` on `res-gpu0` / `res-gpu1`, one process
per pool at a time, with `--device cuda` and the pool's visible device mapped
to `cuda:0` by `rr`.  The failed first `ROLEBLIND_s1` attempt is retained in
the ledger and status artifacts; it failed only in the post-training
correction-disabled audit, was fixed without changing the model or training
math, and was rerun from scratch.

GPU accounting from `analysis/gpu_ledger.json`:

* `11` rr runs, including the CUDA smoke, eight Stage-I arms, one failed
  ROLEBLIND_s1 attempt and its successful rerun;
* total charged `1.7836` A100 GPU-hours of the `14.0` cap;
* all runs on `a100-2`, A100-SXM4-40GB, driver `550.163.01`,
  torch `2.5.1+cu124`.

Artifacts:

| path | contents |
|---|---|
| `PROTOCOL.md` | frozen protocol and decision rules |
| `configs/discovery.yaml` | arm/pool/recipe configuration |
| `code/dc_common.py` | frozen constants, hashes, arm specs |
| `code/dc_data.py` | frozen views, D34/D5 buckets, collation/permutation |
| `code/dc_models.py` | BASE/CONTEXT/ROLEBLIND and the eight new corrections |
| `code/dc_train.py` | shared training loop and diagnostics |
| `code/dc_evaluate.py` | frozen/new-model evaluation |
| `code/dc_analysis.py` | screen, mechanism, seed and group tables |
| `code/dc_gates.py` | 16 CPU gates |
| `code/dc_ledger.py` | GPU ledger |
| `analysis/data_audit.json` | split/label/cache audit |
| `analysis/cpu_gates.json` | 16/16 gate results |
| `analysis/protocol_freeze.json` | final code/config/protocol freeze |
| `analysis/protocol_freeze_stage1.json` | Stage-I freeze before the post-hoc analysis-only fix |
| `analysis/runs/` | eight Stage-I runs plus `roleblind_s1` checkpoints and train logs |
| `analysis/smoke_cpu/`, `analysis/smoke_cuda/` | smoke evidence |
| `analysis/eval_screen/` | Validation420 metrics, predictions and bootstrap |
| `analysis/table_screen.csv` | Stage-I screen |
| `analysis/table_mechanism.csv` | strict mechanism pairs |
| `analysis/table_seed.csv` | seed check |
| `analysis/table_groups.csv` | descriptive scaffold/size/TPSA groups |
| `analysis/table_correction_ablation.csv` | full vs correction-zeroed Validation RMSE |
| `analysis/posthoc_path_binding.json` | binding-gap diagnostic |
| `analysis/decision.json` | machine-readable decision |
| `analysis/gpu_ledger.json` | GPU-hour ledger |
| `analysis/rr_status/` | raw `rr status --json` records |

## 11. Limitations

* Only one seed was trained for each new arm.  The mandatory ROLEBLIND seed
  check demonstrates a `0.0525` seed spread, so the single-seed differences
  among the new arms are not a reliable ranking.
* ROLEBLIND seed 0 is the frozen reference, but seed 1 is worse than BASE
  seed 1.  The screen is therefore a screen against a strong seed-0 draw, not
  a stable estimate of the ROLEBLIND mechanism's expected advantage.
* Validation420 is only 420 molecules; the paired intervals are descriptive
  and were not used for multiplicity-adjusted significance claims.
* Test420 was deliberately not read for metrics.  No Test generalisation
  claim is made.
* The G2L variants are single-shot correction heads, not full architectures.
  Their negative result does not rule out iterative or pretrained composition
  models, nor does it rule out a different readout/regularisation for the same
  structural statistics.
* The PATH-BIND result is a negative result for this target and recipe; the
  binding gap is present, but it did not translate into Validation gain.
