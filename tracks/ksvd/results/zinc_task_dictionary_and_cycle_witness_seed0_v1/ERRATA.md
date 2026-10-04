# ERRATA — zinc-task-dictionary-and-cycle-witness-seed0-v1

Clarifications only. No old report, execution log or JSON is overwritten.

## 1. `conflict_row_fraction = 0.982875` was misnamed

In `zinc_overnight_interface_and_tail_seed0_v1/cpu_conflict_8k.json` the field
`conflict_row_fraction` was computed as `multi.mean()`, i.e. the **row fraction
covered by repeated-input classes**, not by conflicting classes. The old file
is kept byte-for-byte; the corrected decomposition on the same old-fit float32
`T25` input is:

* exact classes: 327;
* repeated classes: 190, repeated row fraction **0.982875** (the old number);
* conflicting classes: **4**, conflicting row fraction **0.001250** (10 rows);
* within every severity group, each exact class is target-consistent
  (`group_only_min_l1 = 0.0`): all conflicts are across severity groups; see
  the next item.

## 2. `5.54946` is the global-optimum output evaluated on `k<=-3`

The old per-group number was the all-fit class-median output — the best
severity-blind predictor on the fit input classes — evaluated on the
`k<=-3` rows. It is **not** the `k<=-3` group's own optimum:

| quantity | value |
| --- | --- |
| `global_min_l1_per_row` (all classes, whole old fit) | 0.0060697235 |
| `global_opt_group_cost[k<=-3]` (same output, tail rows) | 5.5494614660 |
| `group_only_min_l1[k<=-3]` (per-group class medians) | 0.0 |
| `group_only_min_l1` overall | 0.0 |
| `balanced_opt` weighted bound (old severity weights) | k=0 0.0108, all severe groups 0.0 |

Interpretation: there is no *within-severity-group* input conflict at all in
the old fit. Every conflict is across severity groups: one exact `T25` class
contains both `k=0` rows and a penalty row. The tail rows are hard for the
smooth `q_U` head even when their input class is unique, but they are not an
input-limited floor on the group's own error scale. The weighted
"balanced-opt" predictor is descriptive only and is not deployed.

## 3. "the trained joint arms barely react when the pairing is destroyed"

At the MAE level the old statement is defensible; at the prediction level it is
wrong. Replaying `R_SJ`/`R_DJ` on the old internal dev with the trained weights
held fixed and the block switched to marginal:

| arm | dev cal MAE native -> flipped | `|delta_pred|` mean | p95 | max | improved / worsened rows |
| --- | --- | --- | --- | --- | --- |
| `R_SJ` | 0.117110 -> 0.117802 (+0.000692) | 0.018194 | 0.048524 | 0.176037 | 991 / 1009 |
| `R_DJ` | 0.117561 -> 0.117891 (+0.000330) | 0.016416 | 0.045104 | 0.116323 | 986 / 1014 |

Per-row contributions: `R_SJ` improve +0.008188 vs worsen +0.008880 per row
(net -0.000692); `R_DJ` +0.007627 vs +0.007956 (net -0.000330). The pairing
switch moves roughly half the rows by ~0.02 in each direction and cancels out
in the mean. Correct wording: **no net generalization benefit under this
checkpoint and fold**, not "the pairing was unused".

## 4. `A0 - B` cannot be attributed to unfreezing alone

The old `A0` vs `B` difference (`+0.006011` overall cal) mixes at least:
240 extra training epochs, the dense-code linear path replacing the sparse
signed-IHT path, and the reconstruction term / different optimizer trajectory.
This round deliberately adds no training to decompose these factors. The only
supported statement is that the frozen body `B` is a lower bound under this
protocol, not that "unfreezing is the unique/dominant bottleneck".

## 5. The old five fit-tail rows are not all input conflicts

Of the five old-fit `k<=-3` rows, `train:0593` (class 232) and `train:1424`
(class 259) share their exact `T25` with `k=0` rows; `train:1760`, `train:2347`
and `train:3776` have unique `T25` classes in the old fit. `q_B` fits them by
extra weight/optimization, not by resolving an input ambiguity; its two dev
tail rows get worse (q_U 2.31/12.20 -> q_B 28.42/13.60). See
`cycle_class_witnesses.json`.

## 6. Stored-order cycle helper vs official canonical order

The audit table distinguishes `cycle_score_stored_order` (computed on the
stored adjacency order, the same semantics as the repo `_cycle_basis_stats`
helper) from `cycle_score_gvae_order` (upstream canonical-SMILES order). On
`train:2447` the stored-order helper reports a 5-cycle excess while the
canonical order reports none. Sixteen renumberings (seed 20261004) of the eight
witness molecules change the stored-order helper score on 7 of 8
(`renumber_check.json`). This proves the **stored-order helper** is node-order
sensitive; it does not by itself prove that the upstream official labels are
wrong (the upstream pipeline canonicalizes before computing the cycle score).
