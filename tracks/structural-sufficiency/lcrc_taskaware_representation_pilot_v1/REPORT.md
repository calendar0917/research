# REPORT — LCRC-A Task-Aware Representation Pilot v1

Round: `lcrc-taskaware-representation-pilot-v1` (Sprint 03, P0–P3).
Host `res` / node `a100-2`, 2 × A100-SXM4-40GB, driver 550.163.01, torch 2.5.1+cu124.
Revisions: `de483098` (5 runs) and `f49b5025` (2 runs, before the audit-path fix —
see §7). Protocol: `PROTOCOL.md` + `analysis/protocol_freeze.json`. Audit:
`FEATURE_GAP_AUDIT.md`. Candidates: `analysis/representation_candidates.md`.

Seven formal runs, seed 0, 360 epochs each, **1.339 A100 GPU-hours** of training
time (1.42 h including four superseded attempts), within the 1.6 h cap. Two Test
splits were read, once each, after the checkpoint freeze; nothing else was
evaluated on Test.

| # | run | dataset / split | objective | best epoch | frozen Validation | job id | checkpoint sha256 | GPU s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `lipo_r123` | Lipophilicity `random123` | masked-cell MSE | 248 | RMSE 0.73773 | `lipo-r123-20261101-225118-03302b1c` | `64991fec199d…` | 273 |
| 2 | `lipo_r124` | Lipophilicity `random124` | masked-cell MSE | 152 | RMSE 0.63320 | `lipo-r124-20261101-225128-cba2a23a` | `f13163165e4b…` | 370 |
| 3 | `bbbp_cv_f0` | BBBP inner-CV fold 0 | masked-cell BCE | 27 | AUC 0.85526 | `bbbp-cv-f0-20261101-230423-299af43a` | `c306b8bc5395…` | 122 |
| 4 | `bbbp_cv_f1` | BBBP inner-CV fold 1 | masked-cell BCE | 12 | AUC 0.90515 | `bbbp-cv-f1-20261101-230433-9abba0da` | `a4b7b756d1ff…` | 153 |
| 5 | `bbbp_cv_f2` | BBBP inner-CV fold 2 | masked-cell BCE | 41 | AUC 0.83226 | `bbbp-cv-f2-20261101-230711-b35cec86` | `f295261e0b20…` | 124 |
| 6 | `tox21_equal_task` | Tox21 published | equal-task BCE | 55 | macro AUC 0.86439 | `tox21-equal-task-20261101-231108-da2c7322` | `8c1e2a21f0f7…` | 681 |
| 7 | `hiv_sqrt_pos` | ogbg-molhiv published | BCE, pos_weight 5.0 | 23 | AUC 0.83002 | `hiv-sqrt-pos-20261101-231234-8865ea86` | `bcfb4e927622…` | 3098 |

Full provenance (split row hashes, cache and normaliser fingerprints, code hashes,
per-epoch curves, restored-state recheck) is in `analysis/checkpoint_manifest.json`;
cost and every attempt in `analysis/gpu_ledger.json` and
`analysis/rr_run_registry.json`; Test exposure in `analysis/test_access_ledger.json`.

---

## 1. P0 — what the input can and cannot see (no GPU time)

The full audit is `FEATURE_GAP_AUDIT.md`; the four results that changed how the
rest of the round was read:

1. **There is exactly one provable representation gap: tetrahedral chirality.**
   The raw-attribute adapter never reads `GetChiralTag`, so atom chirality is
   destroyed *before* the encoder. Constructively: the BACE enantiomer pair
   `…[C@@]3…` / `…[C@]3…` (CIP `R`/`S`) yields byte-identical input tensors **and**
   the identical frozen-checkpoint output `8.033543586730957`. On real data 114 of
   115 Lipophilicity, 30 of 32 BACE, 21 of 23 BBBP and 102 of 139 Tox21
   constitutional-twin pairs have identical model input.
2. **That gap is real but bounded.** Colliding Train pairs with different labels
   impose a Train-RMSE floor of **0.0279** on Lipophilicity — against a Train RMSE
   of 0.28 and Test RMSE of 0.53–0.74. BACE has 60/1210 Train rows in a colliding
   pair. So chirality cannot be the main error source anywhere in this suite; it
   is a documented limitation, not the bottleneck.
3. **The largest *count* of high-need missing information is physicochemical, not
   structural**: TPSA, HBD/HBA, MW and a logP prior are HIGH-need for 3–4 tasks
   each, are absent from the spec, and are deterministic and label-free to
   compute. Everything genuinely high-need and structural is already explicit
   (element, aromaticity, bond order, formal charge, H count, rings, topology),
   which is why LCRC-A is competitive at all.
4. **E/Z survives, tetrahedral doesn't.** A cis/trans flip changes
   `bond_feat[stereo]` and nothing else — S1/S2 codes are stereo-blind
   (`PARTIAL` verdict in `feature_collision_tests.json`). Also worth knowing:
   `atom_type`/`bond_type` in every batch are documentation tensors the forward
   never reads (the information enters via `atom_feat[:, :16]` and the S1/S2 joint
   codes), and there is no dead field (73 probes over 68 declared fields).

Two smaller findings: 3D conformation, tautomer choice and binding pose are
`NOT_DETERMINABLE` from a 2D record (a data ceiling, not a representation gap), and
the frozen alphabet is nowhere near saturated (23–59 of 400 path codes are ever
used per dataset), so "the vocabulary is too small" is not supportable.

## 2. P1 — is the Lipophilicity result a property of the model or of one split?

Three frozen splits of the same 4200 molecules, identical recipe, identical
Train-only preprocessing, one run per arm and split.

| repeat | arm | Validation RMSE | **Test RMSE** | Test − Valid | best epoch |
| --- | --- | --- | --- | --- | --- |
| 122 (Sprint 01, read-only) | LCRC-A | 0.71088 | **0.57275** | −0.138 | 240 |
| 122 (Sprint 02, read-only) | ECFP4+RDKit+XGB | 0.65318 | 0.64771 | −0.005 | – |
| 123 (new) | LCRC-A | 0.73773 | **0.56982** | −0.168 | 248 |
| 123 (new) | ECFP4+RDKit+XGB | 0.69094 | 0.62604 | −0.065 | – |
| 124 (new) | LCRC-A | 0.63320 | **0.52815** | −0.105 | 152 |
| 124 (new) | ECFP4+RDKit+XGB | 0.63534 | 0.60331 | −0.032 | – |
| **mean** | LCRC-A | 0.69394 | **0.55691** | −0.137 | – |
| **mean** | ECFP4+RDKit+XGB | 0.65982 | 0.62569 | −0.034 | – |

What this says, and what it does not:

* **The neural advantage is real and repeat-stable.** LCRC-A wins Test on 3 of 3
  repeats by 0.075 / 0.056 / 0.075 RMSE (mean 0.069); the repeat-to-repeat sd of
  Test RMSE is 0.025 (LCRC) and 0.022 (baseline). The Sprint-01 result was not a
  lucky partition.
* **The Validation ranking is inverted, and reproducibly so.** The baseline wins
  Validation on 3 of 3 repeats (by 0.058 / 0.047 / −0.002) while losing Test on
  3 of 3. That is a *systematic* instrument failure of this 420-row Validation
  split for these two arms, not noise in one split.
* **The neural arm is markedly more split-sensitive**: its Validation sd across
  repeats is 0.054 vs the baseline's 0.028, and its best-epoch estimate wanders
  (152 / 248 / 240) while the baseline has no such parameter. Any future claim of
  a ≤0.05 RMSE improvement on Lipophilicity is therefore unmeasurable with one
  split.
* **Test is systematically easier than Validation here** in all six cells
  (Test − Valid between −0.03 and −0.17). The published split hands the easier 420
  molecules to Test; that is a property of the split, and it means absolute Test
  RMSE cannot be compared across rounds that use different partitions.
* **Exposure.** Lipophilicity Test labels were read in Sprint 01, so both new reads
  are `PUBLIC_SPLIT_ROBUSTNESS`, not an independent blind test; each was scored
  once, after `analysis/checkpoint_manifest.json` was written, and the ledger
  records checkpoint sha256 + split row hash. No selection used Test.

## 3. P2 — is a Train-only cross-validation a better selection instrument for BBBP?

Only the 1631 Train rows of `BBBP.scaffold122` were used. Published Valid/Test
labels were never read (`published_valid_test_rows_used: false` in every run).
Bemis–Murcko scaffolds, 3 folds, seed 0, salt `lcrc-taskaware-bbbp-inner-cv-v1`,
stratified at scaffold level, no scaffold crossing folds, no single-class fold.

| fold | n | pos/neg | scaffolds | best epoch | Train AUC | **held-out AUC** | final-epoch AUC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 545 | 448/97 | 257 | 27 | 0.9587 | **0.8553** | 0.8257 |
| 1 | 542 | 445/97 | 268 | 12 | 0.9140 | **0.9052** | 0.8614 |
| 2 | 544 | 448/96 | 190 | 41 | 0.9696 | **0.8323** | 0.8062 |
| mean ± sd | | | | | | **0.8642 ± 0.0373** | 0.8311 |
| published Validation (Sprint 02, read-only) | 1141 | – | – | 24 | 0.9506 | **0.9733** | – |
| published Test (Sprint 02, read-only) | – | – | – | – | – | **0.7110** | – |

* **Yes, and the diagnosis is the *level*, not the epoch.** The published
  Validation score is 0.973; an independent scaffold-held-out estimate on the same
  Train rows is 0.864 ± 0.037. The distance to the published Test is 0.153 for the
  inner CV and 0.262 for the published Validation. All four selections land in the
  same early-epoch regime (12/27/41 vs 24), so the problem is not "the published
  Valid picks a late overfitted epoch" — it is that the published Valid split
  shares scaffolds with Train and therefore cannot see the generalisation gap at
  all.
* **Resolution.** Across folds the instrument resolves differences of roughly
  ±0.04 AUC at 3 folds; that is the honest bar any BBBP model-selection claim must
  clear.
* **What it does not license.** With one candidate (LCRC-A) and three folds we can
  show the *level* is now calibrated and stable; we did **not** show that the
  Train-only CV *ranks* competing models the way Test would, because that needs
  ≥2 arms × 3 folds. Section 5 turns that into the recommendation for the next
  round instead of asserting it here.

## 4. P3 — two task objectives, one run each

### 4a. Tox21: equal-task loss instead of masked-cell mean

| objective | best epoch | Validation macro ROC-AUC | macro PR-AUC | masked BCE |
| --- | --- | --- | --- | --- |
| masked-cell (Sprint 02, read-only) | 55 | 0.86625 | 0.50382 | 0.18066 |
| **equal-task** (new) | 55 | **0.86439** | **0.49737** | 0.18210 |
| Δ | 0 | −0.0019 | −0.0065 | +0.0014 |

The objective does what it says — the probes show the two losses give different
per-cell weights and different gradients on sparse batches (`4 cells / 2 tasks`:
0.8167 vs 0.6322; empty tasks excluded from the denominator), and both gradients
match the analytic form. But on this split it changes the Validation ranking of
assays by less than a thousandth, i.e. **the dense/sparse assay weighting was not
what limited Tox21**. With 12 assays and a masked-BCE difference of 0.001, the
Tox21 headroom is not in the loss normalisation.

### 4b. OGBG-MolHIV: positive class weight 5.0 from Train labels only

| objective | best epoch | Validation ROC-AUC | PR-AUC | Train ROC-AUC | Train−Valid |
| --- | --- | --- | --- | --- | --- |
| unweighted BCE (Sprint 02, read-only) | 77 | 0.83655 | 0.31545 | 0.99504 | 0.158 |
| **pos_weight 5.0** (new) | 23 | 0.83002 | **0.33174** | 0.94770 | 0.118 |
| Δ | −54 | −0.0065 | **+0.0163** | −0.0473 | −0.040 |

`pos_weight = min(5.0, sqrt(31669/1232)) = 5.0` (raw 5.07, capped), computed from the
Train split only. The trade is exactly the one the imbalance literature predicts:
ROC-AUC gives up 0.007, PR-AUC gains 0.016, and the train/valid gap shrinks by
0.04 because the model can no longer win by predicting "inactive" everywhere. At a
3.74 % positive rate PR-AUC is the decision-relevant axis, so this is a mild win
on the axis that matters and a mild loss on the one that does not — but both
deltas are smaller than one-seed resolution, so the honest label is *directionally
consistent, not established*.
The exposed official HIV Test split was **not** evaluated.

## 5. What the five results say together

The five results are consistent about one thing: **the binding constraint in this
pilot is measurement, not chemistry.**

* Representation: the one provable gap is worth ≤ 0.028 RMSE on Lipophilicity
  (P0); the four missing descriptors are bounded by the free-classical-baseline
  gaps, which are 0.010–0.017 AUC on the three tasks that baseline wins and
  negative on the four it loses.
* Objective: two task-targeted objectives moved their selection metric by ≤ 0.007
  AUC / ≤ 0.016 PR-AUC (P3).
* Split: the same model, same recipe, same data family moves 0.105 RMSE across
  three random partitions (P1), and the single published Validation split ranks the
  two arms backwards on 3 of 3 repeats (P1) and scores 0.262 away from Test on BBBP
  (P2).

In other words: a representation change worth 0.03 would be invisible under the
current ±0.05 selection noise, which is why the round recommends fixing the
instrument first (see the reply's section D).

## 6. Deliverables

| path | content |
| --- | --- |
| `PROTOCOL.md`, `analysis/protocol_freeze.json` | pre-registration written before the first GPU job |
| `FEATURE_GAP_AUDIT.md` | the P0 audit in prose |
| `analysis/feature_gap_matrix.csv` | 24 items × status × measured carrier × task needs |
| `analysis/feature_coverage.json` | label-free coverage for all 48 480 molecules |
| `analysis/feature_collision_tests.json` | constructive chirality/E-Z collisions + real-data twin scan |
| `analysis/collision_label_floor.json` | the error bound a proven collision can explain |
| `analysis/representation_candidates.md` | the two candidates, tiers, bounds, kill criteria |
| `analysis/split_freeze.json` | Lipophilicity 122/123/124 rows + BBBP inner folds |
| `analysis/lipo_repeat_comparison.csv` | P1 table (both arms, three repeats) |
| `analysis/bbbp_inner_cv.csv` | P2 table (folds + published reference) |
| `analysis/taskaware_loss_probes.csv` | P3 probes (toy parity) + both formal runs |
| `analysis/checkpoint_manifest.json` | frozen checkpoints, hashes, gates, references |
| `analysis/test_access_ledger.json` | append-only record of the two Test reads |
| `analysis/gpu_ledger.json`, `analysis/rr_run_registry.json` | cost, attempts, regime, quarantine |
| `analysis/post_freeze_edits.json` | which tooling files moved after the freeze, why, and that the protocol did not |
| `analysis/runs/<key>/` | `train.json`, `curve.csv`, `checkpoint.pt`, predictions, `test_eval.json` |
| `code/`, `tests/` | drivers, audit, tables, 17 tests |

## 7. Engineering record (deviations, not incidents)

1. **Remote deploy blocked by untracked artifacts.** The remote checkout at
   `62afa5a2` carried 62 untracked files (13 MB) in the Sprint-01/02 run
   directories. Nothing was deleted: they were archived to
   `~/cy/quarantine/lcrc_taskaware_sprint03_deployblock_20261101/`
   (tar sha256 `c8eaeeea…`, plus a `git status` snapshot), 40 of the 62 verified
   byte-identical to the incoming commit, the 22 unique ones exist only in the
   archive; only then were the paths cleared and the deploy repeated.
2. **History rewrite before any run.** The first deploy (`c51d5ec8`) was superseded
   by an amend (adding `--key`/`run_key`). Because the remote had already advanced,
   it was hard-reset to the previously deployed base and re-deployed, so **no run
   ever executed from a superseded revision**.
3. **A real reproducibility finding: CUDA `index_add_` is not deterministic.**
   BBBP fold 0 finished all 360 epochs and then aborted in the post-training audit:
   the restored best state reproduced Validation ROC-AUC to 2.3e-5 against a flat
   1e-5 threshold. The difference is exactly one ROC-AUC discordant pair
   (`1/(448·97) = 2.301e-5`): the model aggregates messages with `index_add_`,
   which sums in nondeterministic order on GPU, flipping a near-tie. The audit now
   derives its tolerance from the statistic's resolution
   (`max(1e-5, 1.5/(n_pos·n_neg))`, largest per-task step for multitask),
   re-evaluates the restored state twice to record the observed raw
   nondeterminism, and still uses the flat 1e-5 for regression. Training code, RNG
   stream and checkpoint contents were untouched. The two affected BBBP runs were
   relaunched at the fixed commit; both superseded attempts are in the registry
   with their cost and are not reported as results.
4. **Two argparse failures.** The first launches of both Lipophilicity repeats
   omitted `--dataset` and exited in 5 s with no artifact; recorded with cost.
5. **Budget.** 7 reported runs, 1.339 A100 h of training (1.42 h including the four
   superseded attempts, i.e. 0.081 h of waste), no epoch reduction, seed 0 only,
   priority order honoured (Lipophilicity → BBBP → Tox21 → HIV, HIV started with
   1.04 h of headroom against its 0.86 h cost).
6. **Driver equivalence.** Before any GPU run the new driver was shown to be
   bit-identical to Sprint 02's on the published Tox21 split: a 2-epoch CPU smoke
   through both drivers produced identical values in all 11 numeric curve columns.
7. **Two revisions appear in the manifest.** `lipo_r123`/`lipo_r124` ran at
   `f49b5025`, before the audit-path fix in (3); that fix touches only the
   post-training check, and both runs passed the *older, stricter* threshold, so
   their checkpoints are unaffected. The manifest states each run's commit rather
   than hiding the difference.
