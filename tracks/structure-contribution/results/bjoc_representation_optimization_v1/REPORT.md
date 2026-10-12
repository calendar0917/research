# cscl-bjoc-representation-optimization-v1 — REPORT

Track `structure-contribution`. Protocol `cscl-bjoc-representation-optimization-v1`
(pre-registration: `configs/bjoc_representation_optimization_v1.yaml`, frozen manifest:
`results/bjoc_representation_optimization_v1/protocol_freeze_manifest.json`).
Driver: `code/run_bjoc_representation_optimization_v1.py`. Internal splits only — **official valid
and official test were never loaded in any form** (`correctness_gates.json.gate22`, `gate19`).

## 1. Decision

**`TYPED_STRUCTURE_OVERFIT`**

> The typed short-path and ring-block channels make the capacity-matched model fit the training rows
> **better** (fit MAE −0.013760) while making molecules it never saw **worse**
> (monitor +0.015322, locked-dev +0.012389 with paired CI95 [+0.004238, +0.021133]). The extra
> structure is consumed by the network (masking the channels moves predictions by 0.72 on average)
> but it buys memorisation, not generalisation.

Supporting readings recorded in `scientific_decision.json`:

* **Junction content: necessary.** Removing the junction reader at *exactly* matched capacity makes
  both fit (0.051586 vs 0.041495) and held-out error worse (locked 0.115888 vs 0.096063,
  Δ +0.019825, CI95 [+0.013324, +0.026108], worse on every pre-frozen subgroup)
  → band `JUNCTION_NECESSITY_SUPPORTED` (`junction_necessity_analysis.json`). The vocabulary's
  `JUNCTION_SIMPLIFICATION_PROMISING` branch requires the opposite sign, so the round's headline is
  the typed-structure outcome.
* **Optimisation: not the improvable point.** `NO_STABILITY_TRAINING_PURCHASE`
  (`stability_analysis.json`), audited *before* training.
* **No arm beat the baseline.** The pre-registered T promotion rule failed on every informative
  check, so no stage-2 trajectory (seed 1 / split B) was bought.

## 2. What was asked

The terminal BJOC evaluation left three candidate explanations for the fit-vs-held-out gap
(fit ≈ 0.04-0.05, unseen ≈ 0.09-0.14 MAE):

1. **S — optimization stability**: the last 40 epochs are worth a real gain.
2. **J — learned junction content**: the per-centre junction reader earns its parameters.
3. **T — missing typed structure**: the objects cannot express *which* bond types short paths carry
   or how rings are fused, and adding that helps.

## 3. Arms (all 432,705 parameters; identical init for B/T)

| arm | model | new channels | parameters | init hash |
| --- | --- | --- | --- | --- |
| B | `ExtendedBJOC` (published widths) | present, exactly zero | 432,705 | `5fa89bc879fd…` |
| T | `ExtendedBJOC` (published widths) | real, standardised | 432,705 | `5fa89bc879fd…` |
| J | `NoJunctionBJOC` (published input dims) | none | 432,705 | `3c9dcaa0f3b4…` |

* Published BJOC backbone: **413,505** parameters → **+4.64%**, the unavoidable cost of admitting
  the new input rows. B pays the same cost with structurally zero inputs, so **T − B is
  capacity-free**; J is matched to B so **J − B is capacity-free**. Both deviations are disclosed
  in the freeze manifest (`deviations_from_published_backbone`).
* B and T are the same class, same widths, same seed: identical initial state (verified) and
  identical data order, so the contrast is purely the channels.

### New channels (135 atom + 12 bond)

typed length-2/3 **bond-type sequences** rooted at each atom (16 + 64 log1p counts), how many of
those bonds are ring bonds (3 + 4), the far-end atom category of length-3 paths (28), path totals
(2), and Tarjan **biconnected ring-block** descriptors per atom (18) and per bond (12):
block sizes, local cycle rank, attachments, cut-vertex share, spiro / monocyclic / polycyclic
flags, block bond-type fraction, and a capped simple-path fused-edge test.

## 4. Feature-information audit (before training)

* **53.4%** (5,715 / 10,699) of original atom-feature groups are split by the new channels — the new
  blocks are not a relabelling of the old ones (`feature_information_audit.json`).
* Controlled pairs: a fused bicyclic and a spiro bicyclic share every original ring statistic
  (in-ring, ring-bond count, min-ring one-hot) and are separated only by the new ring-block block.
* Sparsity, disclosed: bond type 0 never occurs in ZINC, so of the 9 (27) sequences that can occur,
  **6** (length 2) and **12** (length 3) actually appear; `ring_is_spiro_center` fires on 199 of
  231,664 atom rows.
* Representation caveat found while testing: a rooted path *histogram* is blind to which branch of a
  junction carries which bond type — the multiset of sequences is unchanged when two branches swap
  their bond types. This is a limitation of the chosen encoding and is recorded in the tests.

## 5. Frozen recipe and its one deviation

Adam lr 1e-3, weight decay 1e-5, batch 128, grad clip 5, 240 epochs, FP32, no AMP, no scheduler,
raw-y L1, last-5 parameter soup (epochs 236-240), train-shuffle seed = arm seed + 101.

**Deviation and correction.** The first stage-1 launch ran with global features unstandardised
(mean 0 / scale 1), which is *not* the upstream recipe; it pushed the monitor MAE to ≈0.175. The
launch was cancelled before any trajectory completed and nothing from it is reported. The
standardiser is now fitted on each split's **fit rows only** (same rule as the upstream prep stage)
and is applied identically to every arm, so no contrast was affected. Global norms:
`global_norm_a.json` / `global_norm_b.json`.

## 6. Splits

Canonical-SMILES groups ordered by `sha256(salt | smiles)`, greedy fill to 8,000 fit / 1,000
monitor / 1,000 locked; row-set hashes frozen in `split_a.json` / `split_b.json`.

* Split A (`…-discovery`) and Split B (`…-replication`) each have 8,000/1,000/1,000 with zero
  overlap between their own three parts, and 4 duplicate-SMILES groups (8 rows) kept whole.
* Remote reproduction is **bit-identical** (split JSONs, standardisers, global norms and the
  328 MB objects cache all hash-match the local copies).
* Disclosure: split B's locked molecules were mostly inside split A's fit set (805/1000), so B
  replicates the *fit-set* variation, not the locked rows (`split_crosstab.json`). Split B was
  therefore never triggered (T did not promote).

## 7. Results — Split A, seed 0, 240 epochs, one trajectory per arm

| arm | fit MAE | monitor MAE | locked-dev MAE | wall |
| --- | --- | --- | --- | --- |
| B | 0.041495 | 0.140670 | **0.096063** | 395 s |
| J | 0.051586 | 0.159042 | 0.115888 | 265 s |
| T | **0.027735** | 0.155992 | 0.108452 | 418 s |

Paired (per-molecule absolute-error differences, 5,000-sample bootstrap):

| contrast | fit Δ | monitor Δ [CI95] | locked-dev Δ [CI95] | locked: fraction better |
| --- | --- | --- | --- | --- |
| T − B | **−0.013760** | +0.015322 [+0.00488, +0.02902] | +0.012389 [+0.00424, +0.02113] | 0.470 |
| J − B | +0.010090 | +0.018372 [+0.00719, +0.03215] | +0.019825 [+0.01332, +0.02611] | 0.426 |

Pre-frozen locked-dev subgroups (all degrade for both contrasts):
ordinary molecules `y > −3` (n=920): T +0.01219, J +0.01927 · `n_atoms > 20` (n=744): T +0.01069,
J +0.01850 · `cyclomatic ≥ 3` (n=606): T +0.01164, J +0.02002 · junction-rich (`> median`, n=460):
T +0.01467, J +0.02028. Removing each arm's ten worst rows leaves the T penalty at +0.01077 and the
J penalty at +0.01908, so **no result is single-row driven**; T's worst row carries 2.2% of the
error mass.

Level anchor (background only, not used for any decision): B's locked-dev MAE 0.096063 is in the
same range as the upstream BJOC-N official-test MAE 0.092233, so this internal split behaves like a
comparable-difficulty proxy even though the two numbers are not comparable quantities.

### Pre-registered T promotion rule

| check | value | pass |
| --- | --- | --- |
| monitor Δ ≤ −0.006 | +0.015322 | ✗ |
| locked Δ < 0 | +0.012389 | ✗ |
| ordinary molecules not worse | +0.01219 | ✗ |
| not single-row driven | top-10-removed Δ = +0.01239 → +0.01077, worst share 0.0216 | partially |
| correctness / capacity gates | 22/22 | ✓ |

→ `T_NO_PURCHASE`; no stage-2 trajectory (seed replication, split replication) was purchased.
Junction band: `JUNCTION_NECESSITY_SUPPORTED` (locked J−B = +0.019825, CI lower bound +0.013324 > 0).

## 8. Mechanism — why the typed channels hurt

Three independent diagnostics on the frozen checkpoints:

1. **The channels are consumed, not ignored** (`mechanism_analysis.json`): zeroing all new atom
   channels moves T's monitor predictions by 0.719 on average and raises monitor MAE from 0.156 to
   0.778; zeroing all new bond channels moves them by 0.042. Largest per-block sensitivities:
   `path3_endpoint_type` 0.386, `path2_seq` 0.322, `ring_block_bondtype_frac` 0.219, `path3_seq`
   0.212, `path_totals` 0.177.
2. **It is not a few extreme molecules** (`new_channel_extreme_value_analysis.json`): 2,941/8,000
   fit molecules contain a channel entry with |z| > 10 (max |z| = 304, from the extreme sparsity of the
   sequence/endpoint-type blocks), yet the T fit gain is uniform — Δ = −0.01372 for molecules with such an entry
   vs −0.01378 for those without it, and the top-5% by max |z| carry only 6.7% of the total fit gain.
   The held-out damage is equally broad (every subgroup, 47% of molecules better).
3. **It is not an optimisation pathology** (`gradient_contrast_B_T.json`): at the identical init the
   gradient profiles of B and T are nearly indistinguishable (1.336 vs 1.384), at the final state
   T's gradient norm is *smaller* than B's (2.10 vs 3.04) at a lower loss, and clipping is inactive
   for both at that batch. J, by contrast, is the arm that is still far from converged (gradient norm
   5.84 > clip 5 at the final state, loss 0.0525 vs 0.0372).

Reading: the new channels give the encoders a high-resolution local description that the reader uses
confidently; on training rows that description correlates with the label (so fit improves), on new
molecules it does not, and the model has no mechanism to discount it. The pre-registered gate's
resource conclusion is therefore "do not buy this representation **under this recipe**" — not
"ring/path structure is uninformative in ZINC".

Related audit from before training: `feature_information_audit.json`, `numeric_scale_audit.json`
(only `global_feat` is standardised upstream; 11 junction columns are constant), and
`new_channel_range_audit.json` (held-out rows are *inside* the fitted range: `frac_mon_gt_fitmax ≤ 0.001`
for every block, so the failure is not distribution shift in the channel values).

## 9. Optimisation audit (S) — no purchase

`historical_training_audit.json` + `gradient_audit.json` + `stability_analysis.json`:

* Both historical seeds are **still falling** at epoch 240 (late-segment mean 0.0715 / 0.0687 vs
  0.0797 / 0.0762 for the previous segment) and the monitor keeps improving, so there is no late
  plateau to fix.
* The seed gap (soup fit 0.0521 vs 0.0395) comes from last-5 soup geometry, not instability: the
  per-epoch difference shrinks from 0.0165 (epochs 1-40) to 0.0065 (201-240) while the soup gain over
  the last epoch curve is −0.019 / −0.028.
* Raw gradient norms stay at 5-8 with clipping active on 60-100% of batches and a single Adam step
  moving every coordinate by ≈ lr·√n_params (0.635 vs the closed form 0.643) — the ordinary
  constant-LR Adam regime.
* Decision: **NO_STABILITY_TRAINING_PURCHASE** — the one authorised stabilisation candidate S
  (staged lr 160→1e-3, 200→3e-4, 240→1e-4) was **not** run, saving a trajectory.

## 10. Correctness and discipline

* `correctness_gates.json`: **22/22 gates pass**, including identical B/T initialisation,
  B channels exactly zero, T ≠ zero, `T with zeroed channels == B` exactly, relabelling invariance
  (≤ 3e-8), batch invariance (≤ 6e-8), y-independence of the forward pass, live gradients in every
  module, no junction module in J, no forbidden modules, save/load replay exact, parameter counts
  432705/432705/432705.
* A real bug was found by the relabelling gate: the test helper gathered instead of scattering the
  atom-type permutation, so the "relabelled" molecule was a different graph (0.005 MAE). Fixed, and
  the bug is pinned by `test_relabel_helper_preserves_graph_structure`.
* Supervision is raw-y L1 only (`gate18`): no `g/ell/s/c/k` target, no decomposed loss, no target
  component model anywhere in the round.
* Locked-dev labels were read exactly once, inside `score`, after every prediction file was written
  and hashed; `pred_*_locked.npz` contains `row` and `pred` only. The whole pipeline was rehearsed
  first on monitor rows (`predict --rows monitor`, `score --preview`, `decide --preview`).
* Tests: `tests/test_bjoc_representation_optimization_v1.py` (36 tests, all passing).

## 11. Budget actually consumed

| item | planned | used |
| --- | --- | --- |
| formal trajectories | ≤ 10 | **3** (B/J/T on split A seed 0) |
| GPU wall clock | ≤ 2.5 h | **0.30 h** (1,079 s of formal training + one cancelled launch ≈ 7 min) |
| total wall clock | ≤ 7 h 45 m | ≈ 1 h in this session (plus the pre-restart Phase A/B audits) |
| concurrency | 1 | 1 |

No stage-2 arm was run because the stage-2 gate (T promotes on split A) failed.

## 12. What this changes

* The fit-vs-held-out gap is **not** an optimisation problem in this regime: 240 epochs of constant
  LR with clipping is a healthy regime, and the "last 40 epochs are still productive" reading is
  confirmed but does not imply that a decay schedule buys anything.
* The **junction reader is earning its parameters**: at matched capacity, removing it costs
  +0.0198 locked-dev MAE (CI excluding 0, consistent across subgroups). This is the first
  capacity-matched ablation of the junction content in this track, and it closes the
  "learned junction content may be decorative" hypothesis in the negative.
* The **typed short-path / ring-block representation does not transfer**: it is consumed, it fits
  better, it generalises worse. The next representation round must be gated on a *cheap* monitor
  read and must supply an input-scale / regularisation control for sparse one-hot channels (the
  |z| ≈ 300 entries are the mechanism candidate), because "more local resolution" alone is not a
  win at this data budget.
* No evidence was produced that would justify spending a trajectory on the stabilisation candidate.

## 13. Artifacts

`historical_training_audit.json` · `numeric_scale_audit.json` · `gradient_audit.json` ·
`stability_analysis.json` · `split_a.json` · `split_b.json` · `split_crosstab.json` ·
`objects_summary.json` · `standardiser_a.json` · `standardiser_b.json` · `global_norm_a.json` ·
`global_norm_b.json` · `feature_information_audit.json` · `architecture_parameter_audit.json` ·
`correctness_gates.json` · `protocol_freeze_manifest.json` · `train_B_A_s0.json` ·
`train_J_A_s0.json` · `train_T_A_s0.json` · `train_manifest.json` · `prediction_manifest_A.json` ·
`split_a_metrics.json` · `split_a_paired_analysis.json` · `typed_path_ring_gate.json` ·
`junction_necessity_analysis.json` · `mechanism_analysis.json` ·
`new_channel_range_audit.json` · `new_channel_extreme_value_analysis.json` ·
`gradient_contrast_B_T.json` · `scientific_decision.json` (+ `*_preview.json` rehearsals).

Checkpoints: `checkpoint_{B,J,T}_A_s0.pt` (1.7 MB each). Regime: `res` (a100-2), pool `res-gpu1`,
A100-SXM4-40GB, driver 550.163.01, torch 2.5.1+cu124, python 3.12.14, training commit `e8d440c5`,
decision/report commit recorded in `train_manifest.json`.

## 14. Next steps (not authorised yet)

1. If the typed-structure route is retried, pre-register **one** of: channel-wise robust scaling /
   winsorised standardisation, an input dropout on the new blocks, or a sparsity-aware encoder — and
   require an a-priori monitor read before any locked read.
2. The junction result suggests the productive direction is *inside* the existing junction content
   (what the reader computes per centre), not more object types.
3. Split B remains available (frozen, unread) if a future round needs a replication of a contrast
   whose fit-set overlap with split A is acceptable.
