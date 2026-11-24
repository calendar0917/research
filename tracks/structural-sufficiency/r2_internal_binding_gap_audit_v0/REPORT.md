# R2 internal-binding gap audit v0

**Round:** `r2_internal_binding_gap_audit_v0`
**Status:** read-only diagnostic; no model, feature, loss, checkpoint, or training run was created
**Base commit:** `189d5dd1e4e90bed5555cc4f23d57216a1e59dcc`
**Analysis commit:** `393f2b2e0947e6986687acdc02589c1cec3c4dc0`
**Date:** 2026-11-24 UTC
**Device:** local CPU only; no `rr` job; 0 GPU-h; 0 training runs

## 0. Answer in one paragraph

`G_EGO2_5` has a positive **point-estimate** Dev advantage over `R2_REF` on both D0
seeds and on D1/s0, but it is not a stable population-level advantage: every paired
scaffold-cluster 95% interval for the RMSE/MAE gain includes zero, only 52.6–55.4%
of molecules favour EGO2, and the D0 per-molecule direction agrees across seeds for
just 54.6% of molecules (Pearson 0.253, Spearman 0.155). The advantage is also
concentrated: on D0/s1, removing the top 5% of molecules by `|delta|` leaves only
3.3% of the net MAE gain (those molecules are 18.5% of total absolute delta), and
removing the single most influential molecule reduces the MAE gain from `+0.0129`
to `+0.0085`. The pre-specified R2 topology strata do **not**
show a positive, adequately supported, D0-consistent concentration in non-tree or
reconvergence structure: `closure_22` high-vs-zero is null/negative,
`reconvergence_2` is too sparse on Dev (4 and 6 non-zero molecules), and the
ring-fraction contrast is consistently negative. The frozen EGO2 model does read
distance-2–distance-2 edges (54.4/54.4/50.8% molecule coverage; mean
`|Δprediction|` 0.023–0.039; removal worsens Dev RMSE by 0.0028–0.0070), but that
input sensitivity does not localise the R2-vs-EGO2 gap to a missing R2 binding
operator. **Decision: `STRUCTURAL_BINDING_BOTTLENECK_NOT_LOCALIZED`.** The current
evidence is better classified as **indistinguishable from nonlinear
computation/optimisation differences** than as evidence for "missing structural
binding".

## 1. Scope and frozen inputs

This audit asks why the local radius-2 GNN `G_EGO2_5` beats the frozen R2 sentence
trunk `R2_REF` on D0, and whether the gap is plausibly tied to real internal
topology binding inside R2. It does not design or train a new operator. It uses only
the pre-existing best-Dev checkpoints and their committed per-molecule predictions;
no checkpoint was reselected.

| item | frozen source |
|---|---|
| R2 model / data | `tracks/structural-sufficiency/r2_sentence_replication_v1/code/` |
| route-sprint code | `tracks/structural-sufficiency/structure_route_resolution_8h_v1/code/` |
| GWIC-v0 structure | `tracks/structural-sufficiency/graph_word_incidence_composition_pilot_v0/code/` |
| D0 split | `acra_relation_generalization_pilot_v1/data/split_train_sub_dev.json` |
| D1 split | `structural_path_composition_route_v1/data/split_train_sub_dev_d1.json` |

| split | seed | R2 run | EGO2 run | best epoch (R2 / EGO2) |
|---|---:|---|---|---:|
| D0 | 0 | `R2_REF_D0_S0` | `G_EGO2_5_D0_S0` | 80 / 99 |
| D0 | 1 | `R2_REF_D0_S1` | `G_EGO2_5_D0_S1` | 117 / 70 |
| D1 | 0 | `R2_REF_D1_S0` | `G_EGO2_5_D1_S0` | 98 / 133 |

Every checkpoint epoch equals its manifest `best_epoch`, and every stored
`predictions_best_dev.npz` covers exactly the frozen 504 Dev rows. D0 and D1 are
reported separately and are never pooled; 63 Dev rows overlap between D0 and D1 and
are not treated as independent observations. No official Valid/Test label was
used; the frozen loader asserts those entries are NaN.

## 2. Data-identity correction

The parent `REPORT.md` and `analysis/queue_lock.json` recorded the **D0** identity in
their D1 summary field:

- recorded: `60ac13c04fb0dab484fc8e7cc28f2a85450794ed3ebeeb0a87d1b41c3532559e`
- expected D1: `c94e2d4bd92e51ab91119923d61e955f5a2f686fc1751883e858fce88dab6bee`

This was checked before any D1 comparison. The D1 split file, the frozen loader,
Fit/Dev row hashes, and target scaler are correct:

| field | D1 value |
|---|---|
| identity | `c94e2d4b…dab6bee` |
| Fit rows sha256 | `b9f74030…4afcebebbc` |
| Dev rows sha256 | `342c1de0…820b0380a6` |
| scaler | mean `2.151278011472791`, std `1.2159778136886414`, n `2856` |

All **nine** D1 runs under `structure_route_resolution_8h_v1/results/`
(`E_MOB_D1_S0`, `F_MOB_BIG_D1_S0`, `F_MOB_D1_S0`, `F_RAW_D1_S0`,
`G_EGO2_5_D1_S0`, `G_EGO2_5_D1_S1`, `G_FULL5_D1_S0`, `G_FULL5_D1_S1`,
`R2_REF_D1_S0`) carry the correct D1 identity, correct Fit/Dev hashes, and correct
scaler in their manifests. The D1 R2 and EGO2/s0 checkpoints replay to the stored
predictions (Section 3). The discrepancy is therefore a **summary-field/text
erratum only**, not a wrong-row training/evaluation event. An append-only
`structure_route_resolution_8h_v1/ERRATA.md` was added; the original report and
queue lock were not rewritten. No retraining was performed and D1 comparisons were
allowed to proceed.

## 3. Frozen checkpoint replay

`analysis/replay.json`, criterion `max |predicted - stored| < 1e-5`:

| cell | R2 run | R2 max `\|Δ\|` | R2 RMSE | EGO2 run | EGO2 max `\|Δ\|` | EGO2 RMSE |
|---|---|---:|---:|---|---:|---:|
| D0/s0 | `R2_REF_D0_S0` | 1.34e-6 | 0.547679 | `G_EGO2_5_D0_S0` | 1.16e-6 | 0.516106 |
| D0/s1 | `R2_REF_D0_S1` | 1.45e-6 | 0.561031 | `G_EGO2_5_D0_S1` | 9.59e-7 | 0.535283 |
| D1/s0 | `R2_REF_D1_S0` | 1.41e-6 | 0.563802 | `G_EGO2_5_D1_S0` | 1.38e-6 | 0.556328 |

All six checkpoints pass; the audit is working on the intended predictions.

## 4. Same-molecule prediction gap

For molecule `i`, `delta_i = |R2_i - y_i| - |EGO2_i - y_i|`; positive means EGO2 is
better. The paired scaffold-cluster bootstrap uses 5,000 draws, seed 1703, resamples
whole Bemis–Murcko scaffold clusters, and uses the same draw indices for R2 and EGO2.

| cell | R2 RMSE | EGO2 RMSE | RMSE gain [95% CI] | R2 MAE | EGO2 MAE | MAE gain [95% CI] | EGO2 better |
|---|---:|---:|---:|---:|---:|---:|---:|
| D0/s0 | 0.547679 | 0.516106 | **+0.031574** [−0.009137, +0.071161] | 0.417100 | 0.386761 | **+0.030339** [−0.005309, +0.065444] | 52.6% [46.7, 58.2] |
| D0/s1 | 0.561031 | 0.535283 | **+0.025748** [−0.023642, +0.074665] | 0.427264 | 0.414341 | **+0.012923** [−0.025678, +0.049920] | 53.6% [48.3, 58.5] |
| D1/s0 | 0.563802 | 0.556328 | **+0.007473** [−0.032567, +0.044601] | 0.431411 | 0.410396 | **+0.021015** [−0.010831, +0.049717] | 55.4% [49.8, 60.6] |

All point estimates are positive, but **every interval includes zero**. The D1
MAE interval is also wide; D1 is not independent of D0. The positive sign is
consistent with EGO2's extra nonlinear message-passing capacity, but this
comparison cannot separate capacity from topology binding. The rest of the audit
therefore asks whether the advantage is *concentrated* in real R2 non-tree or
reconvergence structure, rather than treating the aggregate sign as evidence for a
binding mechanism.

### 4.1 Tails, single-molecule influence, concentration

| cell | max positive delta | max negative delta | MAE gain after removing max `\|delta\|` | top-5% share of `sum \|delta\|` | gain retained after removing top 5% | dominated? |
|---|---:|---:|---:|---:|---:|---|
| D0/s0 | +1.6713 | −1.1348 | +0.027077 | 19.3% | +0.010022 (33.0%) | yes |
| D0/s1 | +2.2196 | −1.6434 | +0.008536 | 18.5% | −0.0004 (3.3%) | yes |
| D1/s0 | +1.1545 | −1.5737 | +0.024186 | 18.2% | +0.0167 (79.7%) | no |

On D0/s1 the advantage is almost entirely a top-5% effect: after removing those
molecules the MAE gain is slightly **negative**. On D0/s0 it remains positive but
drops to roughly one third. The largest single D0 molecule (`row_id 456`,
`Cc1ncc(CO)c(CO)c1O`, 12 atoms) is EGO2-better by `+1.671` on s0 and `+2.219` on
s1; removing it alone lowers the D0/s1 gain from `+0.0129` to `+0.0085`. These are
descriptive examples, not evidence of a chemical mechanism.

### 4.2 D0 two-seed direction consistency

| statistic | value |
|---|---:|
| both `delta > 0` | 153 / 504 |
| both `delta < 0` | 122 / 504 |
| mixed sign | 229 / 504 |
| sign agreement (non-ties) | 54.6% |
| Pearson(`delta_s0`, `delta_s1`) | 0.253 |
| Spearman | 0.155 |
| same sign, both `\|delta\| >= 0.01` | 263 / 472 (55.7%) |
| same sign, both `\|delta\| >= 0.05` | 199 / 345 (57.7%) |

The per-molecule advantage is only weakly reproducible across D0 seeds. The
aggregate RMSE is similar across seeds, but the identity of the molecules driving
the difference is not. A per-molecule "EGO2 wins" claim is therefore not stable,
even before considering D1.

## 5. Label-blind R2 topology strata

All features use only frozen structure arrays and no labels. For each molecule and
root `v`, on the real induced radius-2 ego `R2(v)`:

| feature | definition |
|---|---|
| `closure_11` | induced edges with both endpoints at distance 1 from `v` |
| `closure_22` | induced edges with both endpoints at distance 2 from `v` |
| `reconvergence_2` | distance-2 nodes adjacent to at least two distinct distance-1 nodes |
| `n_nodes`, `n_edges` | nodes and induced edges in `R2(v)` |
| `cycle_rank` | `E - V + C` on the induced ego (union-find) |
| `supports3`, `supports4` | connected induced atom supports of size 3/4 containing `v` inside `R2(v)` |
| `ring_frac` | molecule-level `mean(in_cycle)` |

Per molecule we store the mean over roots, the fraction of roots with a non-zero
value, and the total over roots. `n_atoms` and graph cycle rank are descriptive
controls. Thresholds are computed **only on Fit** and then applied unchanged to
Dev.

### 5.1 Fit-frozen thresholds

| split | `closure_22` nonzero median | `reconvergence_2` nonzero median | ring-fraction median | atom-count median | Fit `closure_22` zero/nonzero | Fit `reconvergence_2` zero/nonzero |
|---|---:|---:|---:|---:|---:|---:|
| D0 | 0.208333 | 0.142857 | 0.691392 | 27.0 | 1417 / 1439 | 2821 / 35 |
| D1 | 0.208333 | 0.142857 | 0.692308 | 27.0 | 1399 / 1457 | 2823 / 33 |

Dev stratum sizes:

| split | `closure_22` zero / nz_low / nz_high | `reconvergence_2` zero / nz_low / nz_high | ring low / high (zero) |
|---|---:|---:|---:|
| D0 | 230 / 148 / 126 | 500 / 3 / 1 | 247 / 257 (0) |
| D1 | 248 / 139 / 117 | 498 / 3 / 3 | 250 / 254 (1) |

`reconvergence_2` non-zero strata are below the pre-specified support floor
(20 molecules / 10 clusters) and are marked **insufficient support**; their numbers
are reported but not interpreted. The D1 `ring_frac_zero` stratum has one molecule
and is likewise not interpreted.

### 5.2 Pre-specified contrasts

For count features, `Δ_f = gain(nz_high) − gain(zero)`; for ring fraction,
`Δ_f = gain(high) − gain(low)`. Positive means EGO2's advantage is larger in the
more structured stratum. The intervals use the same joint scaffold-cluster
bootstrap over all Dev clusters.

| feature | D0/s0 Δ [95% CI] | D0/s1 Δ [95% CI] | D1/s0 Δ [95% CI] | support | reading |
|---|---:|---:|---:|---|---|
| `closure_22` | −0.0048 [−0.0954, +0.0905] | −0.0222 [−0.1239, +0.0842] | −0.0783 [−0.1649, +0.0070] | all strata ok | not D0-consistent (s0 magnitude < 0.005); null/negative |
| `reconvergence_2` | +0.0958 [+0.0606, +0.1309] | +0.1653 [+0.1285, +0.2053] | +0.1109 [−0.4500, +0.5519] | **insufficient** (n=1 high on D0, n=3 on D1) | positive but unsupported and dominated |
| `ring_frac` | −0.0203 [−0.0835, +0.0440] | −0.0608 [−0.1321, +0.0119] | −0.0239 [−0.0930, +0.0462] | low/high ok | D0-consistent but **negative**; no edge-specific sensitivity probe |

The `closure_22` D0/s0 value is **−0.0048**, below the pre-specified 0.005
magnitude margin, so the contrast is not D0-consistent. The D1 `closure_22`
contrast is negative and larger in magnitude, but that does not rescue a positive
structural story.

Stratum-level MAE gains (positive = EGO2 better):

| cell | `closure_22` zero / nz_low / nz_high | `reconvergence_2` zero / nz_low / nz_high | ring low / high |
|---|---:|---:|---:|
| D0/s0 | +0.0287 / +0.0384 / +0.0239 | +0.0280 / +0.3936 / +0.1238 | +0.0407 / +0.0204 |
| D0/s1 | +0.0267 / −0.0012 / +0.0045 | +0.0123 / +0.0547 / +0.1776 | +0.0439 / −0.0169 |
| D1/s0 | +0.0267 / +0.0720 / −0.0516 | +0.0204 / +0.0124 / +0.1313 | +0.0330 / +0.0092 |

### 5.3 Support and dominance diagnostics

- `closure_22`: all compared strata pass the size/cluster floor, but the
  high-vs-zero contrast is not positive and D0-consistent. On D0/s0 the nz_high
  stratum is flagged dominated; on D0/s1 the zero stratum is flagged dominated.
- `reconvergence_2`: the non-zero strata have 1–3 molecules, so support fails in
  every cell. The D0 point estimates are positive, but they are driven by one or
  three molecules and the top-5% diagnostic flags the high stratum as dominated.
- `ring_frac`: all compared strata pass the size floor, but the contrast is negative
  on both D0 seeds and on D1; the high-ring stratum has the **smaller** EGO2 gain.
  The high-ring stratum is dominated on D0/s0; the low/high strata are not dominated
  on D0/s1 or D1. This is the only adequately supported D0-consistent contrast, and
  its sign is opposite to a "more structure, more EGO2 advantage" story.

### 5.4 Descriptive controls

Atom-count strata (Fit median 27) show a larger EGO2 MAE gain in the high-atom
stratum in all three cells (`+0.0402` vs `+0.0189` on D0/s0, `+0.0338` vs `−0.0113`
on D0/s1, `+0.0239` vs `+0.0184` on D1/s0). This is a descriptive control, not a
pre-specified structural contrast; it is consistent with a generic size/composition
contribution and does not isolate closure. The closure-zero × atom-count cross-tab
is in `analysis/stratified_control_atomcount.csv`; it does not show a consistent
closure effect within either atom-count half. Other descriptive
features (`closure_11`, `n_nodes`, `n_edges`, `cycle_rank`, `supports3`,
`supports4`) are in `analysis/strata_descriptive.csv` and were not searched for a
significant subset.

## 6. Frozen EGO2 topology sensitivity

Only the three trained EGO2 checkpoints were used, with no gradients and no
parameter updates. For every Dev molecule, three ego tensors were compared:

1. the original packed radius-2 ego graph;
2. the same graph with every distance-2–distance-2 edge removed (`drop22`);
3. the same graph with every distance-1–distance-1 edge removed (`drop11`).

Only ego connectivity (`src`, `dst`, `bond`) changes. Atom features, bond features,
root indices, and whole-molecule graph scalars are exactly the original ones. The
packed ego node sets and pointers are unchanged, so root representations remain
aligned one-to-one. The unfiltered implementation reproduces
`sr_gnn.ego_block(st, 2)` exactly; original predictions replay with max `|Δ|`
`1.16e-6`, `9.59e-7`, and `1.38e-6` on the three cells; molecules/roots with no
removed edge show exactly zero prediction and root change.

These perturbed graphs are **not legal chemical edits**. They are input-sensitivity
probes only, and are never interpreted as the effect of deleting real bonds.

### 6.1 `drop22` (distance-2–distance-2 edges)

| cell | molecule coverage | removed edge fraction | mean `\|Δprediction\|` | root L2 mean | RMSE before | RMSE after | `ΔRMSE` | interpret? |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| D0/s0 | 54.4% | 2.22% | 0.023192 | 0.140414 | 0.516106 | 0.519271 | +0.003165 | yes |
| D0/s1 | 54.4% | 2.22% | 0.029373 | 0.124794 | 0.535283 | 0.538107 | +0.002824 | yes |
| D1/s0 | 50.8% | 2.08% | 0.039079 | 0.167211 | 0.556328 | 0.563325 | +0.006997 | yes |

Coverage is above the 10% floor, so the performance change is interpretable as an
input-sensitivity result. The model clearly reads these edges: removing ~2% of ego
edge occurrences changes predictions by 0.023–0.039 on average and worsens Dev RMSE
by 0.0028–0.0070. This is evidence that the trained EGO2 uses d2–d2 connectivity;
it is **not** evidence that R2 lacks the corresponding binding, because R2 is a
different architecture and the perturbation is not applied to R2.

### 6.2 `drop11` (distance-1–distance-1 edges)

| cell | molecule coverage | removed edge fraction | mean `\|Δprediction\|` | root L2 mean | performance interpretation |
|---|---:|---:|---:|---:|---|
| D0/s0 | 5.2% | 0.11% | 0.002036 | 0.007655 | **not interpreted** (coverage < 10%) |
| D0/s1 | 5.2% | 0.11% | 0.000867 | 0.005678 | **not interpreted** (coverage < 10%) |
| D1/s0 | 3.8% | 0.08% | 0.004051 | 0.009767 | **not interpreted** (coverage < 10%) |

The RMSE changes for `drop11` are recorded in `analysis/sensitivity_summary.json`
but are not interpreted, per the pre-specified coverage rule. The low coverage is
itself the relevant result: the d1–d1 edge type is rare in these egos, so this probe
cannot support a performance claim either way.

## 7. The four required questions

### Q1. Is the EGO2 advantage positive on both D0 seeds, and what is the D1 direction?

Yes in sign, no in robust magnitude. D0/s0 MAE gain `+0.0303`, D0/s1 `+0.0129`,
D1/s0 `+0.0210`; all three RMSE and MAE intervals include zero. The D0 seed
per-molecule sign agreement is 54.6%, and the D0/s1 gain is almost entirely a
top-5% effect. The D1 direction is positive but its RMSE gain is only `+0.0075`
with an interval `[−0.0326, +0.0446]`.

### Q2. Is the advantage concentrated in real R2 non-tree/reconvergence structure?

No supported positive concentration was found. `closure_22` high-vs-zero is
`−0.0048` on D0/s0, `−0.0222` on D0/s1, and `−0.0783` on D1/s0. The only positive
D0-consistent **primary** contrast is `reconvergence_2`, but its high stratum
contains one molecule on each D0 seed and three on D1/s0, so it fails the support
floor and is dominated. The only adequately supported D0-consistent contrast is ring fraction,
and it is **negative** on both seeds (`−0.0203`, `−0.0608`) and on D1
(`−0.0239`); the high-ring stratum has the smaller EGO2 gain. This is the opposite
of "more real non-tree structure → larger EGO2 advantage".

### Q3. Does the frozen EGO2 model actually read the non-tree connections?

Yes for d2–d2 edges: 50.8–54.4% molecule coverage, mean `|Δprediction|`
0.023–0.039, and `ΔRMSE` `+0.0028` to `+0.0070` when those edges are removed. That
is a non-degenerate input sensitivity. For d1–d1 edges the coverage is only
3.8–5.2%, so the performance change is not interpreted. Crucially, this sensitivity
is measured **inside EGO2**; it does not show that the R2-vs-EGO2 gap is caused by
R2 lacking those connections.

### Q4. Does the evidence support "missing structural binding" or "still indistinguishable from nonlinear computation/optimisation differences"?

It is better classified as **still indistinguishable from nonlinear
computation/optimisation differences**. R2 and EGO2 differ in architecture,
parameterisation, readout, and optimisation; the audit is not a controlled
structural intervention on R2. The pre-specified structure contrasts do not isolate
a positive EGO2 advantage in the real R2 non-tree/reconvergence strata, the only
supported D0-consistent contrast points the other way, and the per-molecule
advantage is only weakly reproducible across D0 seeds. The d2–d2 sensitivity result
shows that EGO2 uses those edges, not that R2's missing binding is the bottleneck.

## 8. Decision

**`STRUCTURAL_BINDING_BOTTLENECK_NOT_LOCALIZED`**

No primary feature satisfies all of the pre-specified conditions:

| condition | `closure_22` | `reconvergence_2` | `ring_frac` |
|---|---|---|---|
| overall D0 advantage positive on both seeds | yes | yes | yes |
| contrast D0-consistent (same sign, both `\|Δ\| >= 0.005`) | no (s0 `−0.0048`) | yes | yes |
| D1 not clearly reversed | yes | yes | yes |
| matching sensitivity non-degenerate | yes (`drop22`) | indirect only | **no edge-specific probe** |
| not dominated / support floor | no (dominated in one D0 seed) | **no** (1–3 molecules) | no (dominated on D0/s0) |
| **recommend operator?** | **no** | **no** | **no** |

The decision is therefore the pre-specified fallback. No new R2 binding operator is
recommended from this audit, and none is designed here. A future controlled test
would need a matched-parameterisation comparison and a pre-registered intervention
on R2 itself; this observational R2-vs-EGO2 comparison cannot supply that causal
evidence.

## 9. Reproduction and run identity

```bash
cd tracks/structural-sufficiency/r2_internal_binding_gap_audit_v0
uv run python code/run_all.py
uv run pytest -q tests
```

`run_all.py` runs, in order: `ib_replay.py`, `ib_identity.py`, `ib_gap.py`,
`ib_topology.py`, `ib_sensitivity.py`, `ib_summary.py`. The full pipeline took
51.8 s wall on local CPU, used no GPU, and completed with `all_ok=true`
(`analysis/run_all.json`). Peak memory was bounded by processing one split at a
time and batching EGO2 inference; the replay process peak RSS was ~0.51 GB.

| item | value |
|---|---|
| base commit | `189d5dd1e4e90bed5555cc4f23d57216a1e59dcc` |
| protocol sha256 | `c36446a5397e703bee486bbe48c89f4c86158ee683a2fa5e4a81bb23cdfe561c` |
| code sha256 (`ib_*.py` + `run_all.py`) | `913cb83ff200bd242c079023606e602ed700873e29da7a81e080fece311bf491` |
| environment | Python 3.12.15, torch 2.5.1+cu124, CPU, 16 cores, no CUDA device |
| training runs | 0 |
| GPU-hours | 0 |
| official Valid/Test labels | not used (frozen loader asserts NaN) |
| checkpoint selection | pre-existing best-Dev checkpoints only |

The complete output manifest is in `analysis/run.json` and `analysis/run_all.json`.
Key tables: `gap_metrics.csv`, `gap_per_molecule.csv`, `gap_tails.csv`,
`gap_d0_seed_consistency.csv`, `topology_molecules_D0.csv`,
`topology_molecules_D1.csv`, `strata_*.csv`, `strata_contrasts_*.csv`,
`strata_descriptive.csv`, `stratified_control_atomcount.csv`,
`sensitivity_summary.json`, `sensitivity_molecules.csv`, `decision.json`.

## 10. Limitations

1. **Architecture confound.** R2 is a fixed-radius sentence trunk with no message
   passing; EGO2 is a five-layer GNN with per-atom readout. Their difference cannot
   be attributed to topology binding alone.
2. **Not a controlled intervention on R2.** The audit describes R2 structure and
   probes EGO2 inputs; it does not remove or add a binding operator in R2.
3. **Sparse reconvergence strata.** Only 4 D0 and 6 D1 Dev molecules have a non-zero
   per-molecule `reconvergence_2` mean. The `reconvergence_2` contrast is therefore
   reported but not interpretable, regardless of its point estimate.
4. **Perturbations are not chemistry.** `drop22`/`drop11` are input-sensitivity
   probes, not legal chemical edits, and their performance changes are not bond
   deletion effects.
5. **Dev-only, non-independent splits.** All metrics are on held-out Dev rows.
   D0 and D1 share 63 rows and are never pooled. No official Valid/Test labels were
   used.
6. **Associational strata.** The topology contrasts are descriptive associations,
   not chemical causal effects. No post-hoc subset search was performed.
7. **Coverage-limited d1–d1 probe.** The d1–d1 removal covers only 3.8–5.2% of Dev
   molecules, so its performance change is not interpreted.

## Appendix A — exact primary stratum table

Positive gain = EGO2 better. `support=False` means fewer than 20 molecules or fewer
than 10 clusters; such rows are not interpreted.

| cell | feature | stratum | n | clusters | MAE gain | 95% CI | support | dominated |
|---|---|---|---:|---:|---:|---:|---|---|
| D0/s0 | `closure_22` | zero | 230 | 89 | +0.028684 | [-0.025746, +0.081831] | yes | no |
| D0/s0 | `closure_22` | nz_low | 148 | 81 | +0.038399 | [-0.014500, +0.096128] | yes | yes |
| D0/s0 | `closure_22` | nz_high | 126 | 69 | +0.023894 | [-0.048799, +0.102091] | yes | yes |
| D0/s0 | `reconvergence_2` | zero | 500 | 227 | +0.027973 | [-0.007057, +0.063029] | yes | yes |
| D0/s0 | `reconvergence_2` | nz_low | 3 | 3 | +0.393586 | [+0.008693, +1.078947] | **no** | yes |
| D0/s0 | `reconvergence_2` | nz_high | 1 | 1 | +0.123762 | [+0.123762, +0.123762] | **no** | yes |
| D0/s0 | `ring_frac` | low | 247 | 105 | +0.040689 | [-0.002648, +0.083503] | yes | no |
| D0/s0 | `ring_frac` | high | 257 | 145 | +0.020392 | [-0.029781, +0.071617] | yes | yes |
| D0/s1 | `closure_22` | zero | 230 | 89 | +0.026654 | [-0.030814, +0.083078] | yes | yes |
| D0/s1 | `closure_22` | nz_low | 148 | 81 | -0.001206 | [-0.057351, +0.055776] | yes | no |
| D0/s1 | `closure_22` | nz_high | 126 | 69 | +0.004455 | [-0.079191, +0.093963] | yes | no |
| D0/s1 | `reconvergence_2` | zero | 500 | 227 | +0.012343 | [-0.026279, +0.049121] | yes | yes |
| D0/s1 | `reconvergence_2` | nz_low | 3 | 3 | +0.054650 | [-0.475632, +0.390799] | **no** | no |
| D0/s1 | `reconvergence_2` | nz_high | 1 | 1 | +0.177603 | [+0.177603, +0.177603] | **no** | yes |
| D0/s1 | `ring_frac` | low | 247 | 105 | +0.043947 | [-0.013971, +0.102090] | yes | no |
| D0/s1 | `ring_frac` | high | 257 | 145 | -0.016893 | [-0.062334, +0.025892] | yes | no |
| D1/s0 | `closure_22` | zero | 248 | 120 | +0.026708 | [-0.018757, +0.067431] | yes | no |
| D1/s0 | `closure_22` | nz_low | 139 | 71 | +0.071950 | [+0.018398, +0.118451] | yes | no |
| D1/s0 | `closure_22` | nz_high | 117 | 82 | -0.051564 | [-0.124988, +0.019394] | yes | no |
| D1/s0 | `reconvergence_2` | zero | 498 | 261 | +0.020403 | [-0.011710, +0.049514] | yes | no |
| D1/s0 | `reconvergence_2` | nz_low | 3 | 3 | +0.012354 | [-0.322659, +0.241948] | **no** | no |
| D1/s0 | `reconvergence_2` | nz_high | 3 | 3 | +0.131349 | [-0.433370, +0.575032] | **no** | yes |
| D1/s0 | `ring_frac` | low | 250 | 121 | +0.033038 | [-0.015001, +0.078727] | yes | no |
| D1/s0 | `ring_frac` | high | 254 | 175 | +0.009182 | [-0.037672, +0.054267] | yes | no |
| D1/s0 | `ring_frac` | zero | 1 | 1 | +0.339961 | [+0.339961, +0.339961] | **no** | yes |

## Appendix B — output files

```
analysis/identity_audit.json
analysis/replay.json
analysis/gap_metrics.csv
analysis/gap_per_molecule.csv
analysis/gap_tails.csv
analysis/gap_d0_seed_consistency.csv
analysis/gap_summary.json
analysis/bootstrap_gap_draws.npz
analysis/topology_molecules_D0.csv
analysis/topology_molecules_D1.csv
analysis/topology_thresholds.json
analysis/strata_D0_S0.csv
analysis/strata_D0_S1.csv
analysis/strata_D1_S0.csv
analysis/strata_contrasts_D0_S0.csv
analysis/strata_contrasts_D0_S1.csv
analysis/strata_contrasts_D1_S0.csv
analysis/strata_d0_consistency.csv
analysis/strata_descriptive.csv
analysis/stratified_control_atomcount.csv
analysis/sensitivity_summary.json
analysis/sensitivity_molecules.csv
analysis/sensitivity_roots_D0_S0_drop11.npz
analysis/sensitivity_roots_D0_S0_drop22.npz
analysis/sensitivity_roots_D0_S1_drop11.npz
analysis/sensitivity_roots_D0_S1_drop22.npz
analysis/sensitivity_roots_D1_S0_drop11.npz
analysis/sensitivity_roots_D1_S0_drop22.npz
analysis/decision.json
analysis/run.json
analysis/run_all.json
```