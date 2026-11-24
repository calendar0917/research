# R2 internal-binding gap audit, v0 — frozen protocol

Round: `r2_internal_binding_gap_audit_v0`
Status: read-only diagnostic. No model, feature, loss, checkpoint, or training run is
created in this round. Official Valid/Test labels are never read.

## 0. Question and scope

The parent round `structure_route_resolution_8h_v1` found that the radius-2 local
GNN `G_EGO2_5` has lower Dev RMSE than the frozen R2 sentence trunk on D0/s0 and
D0/s1 (and a smaller advantage on D1/s0). This audit asks whether that gap is
plausibly tied to **real internal topology inside the radius-2 R2 sentence** — in
particular non-tree edges and branch reconvergence — or whether the evidence still
cannot separate a structural-binding explanation from generic nonlinear-computation
and optimisation differences.

The audit is deliberately narrow:

- no new model or operator is designed;
- no training, fine-tuning, distillation, or checkpoint selection is performed;
- only the pre-existing best checkpoints and their committed per-molecule
  predictions are used;
- D0 and D1 are reported separately and are never pooled, because 63 Dev rows
  overlap between the two splits.

## 1. Frozen inputs

Code and data are pinned through the parent loaders; every run re-verifies them.

| item | path |
|---|---|
| R2 code / loader | `tracks/structural-sufficiency/r2_sentence_replication_v1/code/` |
| route-sprint code | `tracks/structural-sufficiency/structure_route_resolution_8h_v1/code/` |
| frozen GWIC-v0 model/structure | `tracks/structural-sufficiency/graph_word_incidence_composition_pilot_v0/code/` |
| D0 split | `acra_relation_generalization_pilot_v1/data/split_train_sub_dev.json` |
| D1 split | `structural_path_composition_route_v1/data/split_train_sub_dev_d1.json` |

Checkpoints (best-Dev checkpoint in each existing run; selected before this audit):

| split | seed | R2 run | EGO2 run |
|---|---:|---|---|
| D0 | 0 | `R2_REF_D0_S0` | `G_EGO2_5_D0_S0` |
| D0 | 1 | `R2_REF_D0_S1` | `G_EGO2_5_D0_S1` |
| D1 | 0 | `R2_REF_D1_S0` | `G_EGO2_5_D1_S0` |

Each run must satisfy `checkpoint.epoch == manifest.best_epoch` and its stored
`predictions_best_dev.npz` must cover exactly the frozen Dev rows. No alternative
checkpoint is considered.

Expected split identities:

- D0 `60ac13c04fb0dab484fc8e7cc28f2a85450794ed3ebeeb0a87d1b41c3532559e`
- D1 `c94e2d4bd92e51ab91119923d61e955f5a2f686fc1751883e858fce88dab6bee`

## 2. Data-identity correction (performed before any gap analysis)

The parent `REPORT.md` and `analysis/queue_lock.json` appear to record the D0
identity in a D1 summary field. Before using any D1 result:

1. Recompute the D1 identity, Fit-row hash, Dev-row hash, and target standardiser
   from the frozen D1 split file and the frozen loader.
2. Inspect every D1 `run manifesto` under `structure_route_resolution_8h_v1/results/`:
   `data.identity_sha256`, Fit/Dev row hashes, scaler mean/std/n, and checkpoint
   `data`/`split` fields.
3. Replay the D1 R2 and EGO2 best checkpoints on the D1 Dev rows and compare to
   their stored predictions.
4. If every run-level field and replay agrees with the frozen D1 rows, classify the
   discrepancy as a **summary-field/text erratum only**, append
   `structure_route_resolution_8h_v1/ERRATA.md`, and continue. Do not modify the
   original report or queue lock and do not retrain.
5. If any run actually trained or evaluated on D0 rows while labelled D1, stop all
   D1 comparisons and report the run-level failure. No D1 result may then be used.

## 3. Same-molecule prediction gap

For each of the three split/seed cells:

- Join R2 and EGO2 predictions on `row_id`; require identical row order, identical
  labels, and labels equal to the frozen bundle's Dev labels.
- Report RMSE and MAE for R2 and EGO2 in original label units.
- Define the per-molecule absolute-error difference

  `delta_i = |R2_i - y_i| - |EGO2_i - y_i|`

  so positive `delta_i` means EGO2 is better on molecule `i`.
- Report the mean `delta` (equal to `MAE_R2 - MAE_EGO2`), the proportion of
  molecules with `delta_i > 0`, ties, and the largest positive/negative tails.
- Report the total difference after removing the single most influential molecule
  (`argmax_i |delta_i|`), plus separate removals of the largest positive and largest
  negative `delta`.
- Paired scaffold-cluster bootstrap: 5,000 draws, seed 1703, resampling whole
  Bemis–Murcko scaffold clusters with replacement and keeping all molecules of a
  drawn cluster. The same draw indices are used for R2 and EGO2. Report percentile
  95% intervals for the RMSE gain (`RMSE_R2 - RMSE_EGO2`), MAE gain, and proportion
  EGO2-better. A cluster is the scaffold string stored in the frozen split file.
- D0 seed consistency: for each D0 Dev molecule compare `delta_s0` and `delta_s1`.
  Report sign counts, sign agreement among non-ties, Pearson/Spearman correlation,
  and agreement restricted to `|delta| >= 0.01` and `>= 0.05`. Write the full
  per-molecule table; report the top few molecules by `min(delta_s0, delta_s1)` as
  descriptive examples only.

D1 is reported separately. The 63 overlapping D0/D1 Dev molecules are recorded for
transparency but never treated as independent observations.

## 4. Label-blind R2 topology description

For each molecule and each root `v`, on the real induced radius-2 ego `R2(v)`:

- `closure_11` — number of induced edges whose two endpoints are both at distance 1
  from `v`;
- `closure_22` — number of induced edges whose two endpoints are both at distance 2
  from `v`;
- `reconvergence_2` — number of distance-2 nodes adjacent to at least two distinct
  distance-1 nodes;
- `n_nodes` — number of nodes in `R2(v)`;
- `n_edges` — number of induced edges in `R2(v)`;
- `cycle_rank` — `E - V + C` on the induced ego (computed with union-find; `C` is
  normally 1);
- `supports3`, `supports4` — number of distinct connected induced atom supports of
  size 3 and 4 that contain `v` and lie inside `R2(v)`.

For each molecule report, for every root-level feature, the mean over roots, the
fraction of roots with a non-zero value, and the total over roots. Also report
`n_atoms`, graph cycle rank, and ring-atom fraction (`mean(in_cycle)`).

These features use only frozen structure arrays; no label is read.

### 4.1 Frozen strata

Thresholds are computed **only on the Fit rows of each split** and then applied to
that split's Dev rows. Thresholds are not recomputed on Dev and are not tuned.

- `closure_22` and `reconvergence_2`: primary statistic is the per-molecule mean.
  `zero` = mean exactly 0; `nonzero` = mean > 0. The median of the nonzero Fit means
  splits `nonzero` into `nz_low` (`<= median`) and `nz_high` (`> median`).
- Ring-atom fraction: Fit median splits Dev into `ring_low` (`<= median`) and
  `ring_high` (`> median`). The zero/nonzero split is reported only if a non-empty
  zero group exists.
- Atom count: Fit median splits Dev into `atoms_low` / `atoms_high`; this is a
  descriptive control only.

The three pre-specified primary strata are `closure_22`, `reconvergence_2`, and
ring-atom fraction. `closure_11`, `n_nodes`, `n_edges`, `cycle_rank`, `supports3`,
`supports4`, and atom count are descriptive and are not searched for a significant
subset.

### 4.2 Stratum reporting

For each primary feature, split/seed, and stratum, report:

- number of molecules and scaffold clusters;
- R2 and EGO2 RMSE/MAE and the MAE gain (`mean delta`);
- paired scaffold-cluster bootstrap 95% interval for the MAE gain and the RMSE gain;
- the proportion of molecules where EGO2 is better;
- concentration diagnostics: `max |delta| / sum |delta|`, top-5% share of
  `sum |delta|`, MAE gain after removing the single most influential molecule, and
  MAE gain after removing the top 5% of molecules by `|delta|`.

Strata with fewer than 20 molecules or fewer than 10 scaffold clusters are reported
but marked **insufficient support** and are not interpreted.

### 4.3 Structure-related contrast

For count features (`closure_22`, `reconvergence_2`) the pre-specified contrast is

`Delta_f = gain(nz_high) - gain(zero)`

where `gain(s) = MAE_R2(s) - MAE_EGO2(s)` (positive means EGO2 better). For ring
fraction it is `gain(ring_high) - gain(ring_low)`. Positive `Delta_f` means EGO2's
advantage is larger in the more structured stratum. Intervals for `Delta_f` use the
same joint scaffold-cluster bootstrap over all Dev clusters, computing both stratum
gains on each draw.

A contrast is **D0-consistent** if `Delta_s0` and `Delta_s1` have the same sign and
both magnitudes are at least 0.005. It is **clearly reversed on D1** if the D1 sign
is opposite and `|Delta_D1| >= 0.01`; otherwise D1 is **not clearly reversed**.

These are descriptive associations. A positive stratum contrast is not a chemical
causal effect and is not by itself evidence for a structural-binding operator.

## 5. Frozen EGO2 topology-sensitivity check

Only the three trained EGO2 checkpoints are used, with no gradient and no parameter
update. For every Dev molecule, three ego tensors are compared:

1. the original packed radius-2 ego graph;
2. the same ego graph with every distance-2–distance-2 edge removed;
3. the same ego graph with every distance-1–distance-1 edge removed.

Only the ego connectivity tensors (`src`, `dst`, `bond`) change. Atom features, bond
features, root indices, and whole-molecule graph scalars are exactly the original
ones. The packed ego node sets and pointers are unchanged, so root representations
are aligned one-to-one.

Requirements:

- the unfiltered implementation must reproduce `sr_gnn.ego_block(st, 2)` exactly
  (`atom`, `src`, `dst`, `bond`, `ptr`);
- original frozen predictions must replay against the stored Dev predictions with
  maximum absolute difference `< 1e-5`;
- molecules with no removed edge of the requested type must show **zero** prediction
  and root-representation change (reported maximum must be `< 1e-6`);
- coverage is reported as the fraction of Dev molecules with at least one removed
  edge, the fraction of roots affected, the removed fraction of all ego-edge
  occurrences, and the total removed edges.

Reported input-sensitivity quantities:

- mean, median, 95th percentile and maximum `|prediction_after - prediction_before|`
  in original label units;
- RMSE before and after, and `RMSE_after - RMSE_before`;
- root-representation change: mean/median/p95/max L2 distance, mean absolute
  coordinate change, and mean cosine similarity;
- the same quantities restricted to molecules/roots where at least one edge was
  removed, plus the zero-removal check.

If a perturbation covers fewer than 10% of Dev molecules, its performance change is
**not interpreted**. The coverage fact, prediction-change magnitude, and root-change
magnitude are still recorded as input sensitivity. The perturbed graphs are not
legal chemical edits and are never interpreted as the effect of deleting real bonds.

## 6. Decision rule and stopping

Answer four questions:

1. **Cross-seed consistency.** Is the EGO2 advantage over R2 positive on both D0
   seeds, and what is the D1 direction? Report the overall paired bootstrap.
2. **Structural concentration.** Is the advantage concentrated in real R2 non-tree
   or reconvergence structure? Use the pre-specified `Delta_f` contrasts and their
   support/dominance diagnostics. Do not search for a better subset.
3. **Input sensitivity.** Does the frozen EGO2 model actually read the non-tree
   connections? Use the `drop22` and `drop11` checks and their coverage. A model
   that is insensitive to these edges cannot support a binding explanation based on
   them.
4. **Evidence class.** Does the current evidence support "missing structural
   binding" or "still cannot distinguish from nonlinear computation/optimisation
   differences"?

A next-step specific R2 structural-binding operator is suggested only if **all** of
the following hold for at least one primary feature:

- the overall D0 EGO2 advantage is positive on both seeds;
- the structure contrast is D0-consistent (same sign, both `|Delta| >= 0.005`);
- D1 is not clearly reversed (same sign or `|Delta_D1| < 0.01`);
- the matching input-sensitivity check is non-degenerate: coverage at least 10% of
  Dev molecules and (mean `|Delta prediction| >= 0.01` or `|Delta RMSE| >= 0.005`);
- the contrast is not dominated by very few molecules: each compared stratum has at
  least 20 molecules and 10 clusters, removing the single most influential molecule
  does not flip the stratum-gain sign, and removing the top 5% by `|delta|` retains
  at least 50% of the absolute gain.

For `reconvergence_2`, the direct perturbation is not one of the two frozen edge
removals; the matching sensitivity is therefore marked **indirect** and uses
`drop22` (the removed edges are incident to distance-2 nodes). For ring fraction the
matching sensitivity is also indirect and uses both perturbations, but `drop11` is
expected to be coverage-limited.

Implementation clarification (recorded before the final report): ring-atom fraction
is an atom-composition statistic, not an edge type. The two frozen edge removals
therefore do not provide a specific sensitivity test for it. The audit still reports
the `drop22`/`drop11` input-sensitivity magnitudes, but the decision rule treats the
ring-fraction sensitivity condition as **not satisfied**; a negative ring-fraction
contrast is never promoted to an operator recommendation on the basis of an
edge-removal probe.

If no feature passes all conditions, the conclusion is
**`STRUCTURAL_BINDING_BOTTLENECK_NOT_LOCALIZED`**. In that case no new model or
operator is designed from this audit. Even if a feature passes, the result is an
associational suggestion for a future controlled test, not proof that R2 lacks a
binding operator; the R2-vs-EGO2 comparison remains confounded by architecture,
parameterisation, and optimisation.

## 7. Outputs and execution

All analysis is CPU-only and runs locally; no `rr` job is needed unless a future
GPU requirement appears. No training is started. Peak memory is bounded by
processing one split at a time and by batching EGO2 inference.

Expected outputs under `analysis/`:

- `identity_audit.json`, `replay.json`;
- `gap_metrics.csv`, `gap_per_molecule.csv`, `gap_tails.csv`,
  `gap_summary.json`, `bootstrap_gap_draws.npz`,
  `gap_d0_seed_consistency.csv`;
- `topology_molecules_D0.csv`, `topology_molecules_D1.csv`,
  `topology_thresholds.json`;
- `strata_D0_S0.csv`, `strata_D0_S1.csv`, `strata_D1_S0.csv`,
  `strata_contrasts_D0_S0.csv`, `strata_contrasts_D0_S1.csv`,
  `strata_contrasts_D1_S0.csv`, `strata_d0_consistency.csv`,
  `strata_descriptive.csv`, `stratified_control_atomcount.csv`;
- `sensitivity_summary.json`, `sensitivity_molecules.csv`,
  sensitivity root arrays;
- `decision.json`, `run.json`, `run_all.json`.

`REPORT.md` reports the numerical results, execution identity, limitations, and the
decision. `ERRATA.md` is appended only if the identity check is text/summary-only;
no historical file is rewritten.

## 8. Stop condition

The round stops after the decision. No follow-up model, no new training, no
official Valid/Test evaluation, and no checkpoint re-selection are performed.