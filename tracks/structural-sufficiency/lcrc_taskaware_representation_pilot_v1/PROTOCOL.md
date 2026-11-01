# PROTOCOL — LCRC-A Task-Aware Representation Pilot v1

Round: `lcrc-taskaware-representation-pilot-v1` (Sprint 03, workstreams P0–P3)
Track: `tracks/structural-sufficiency/lcrc_taskaware_representation_pilot_v1/`
Frozen by `analysis/protocol_freeze.json`, written **before** the first GPU job and
committed unchanged at the revision the jobs ran on.

This document is the *pre-registration*. Everything a score could influence is
fixed here: the row lists, the folds, the two objectives, the recipe, the
checkpoint rule, the run list, the budget and the Test policy. Results live in
`REPORT.md`; the audit lives in `FEATURE_GAP_AUDIT.md`.

---

## 0. Scope and the one thing that is not allowed

The round asks whether the information LCRC-A can see is *sufficient* for these
tasks, and whether a small number of task-targeted changes move it — without
retraining a new representation. Therefore:

* **No new FeatureSpec, no augmented encoder, no representation change.** The
  frozen Lipophilicity spec `lipo_v1` (atom_feat 51, bond_feat 13, atom_scalar 9,
  global 42, nbr_code 80, path_code 400) is shared by all seven datasets and is
  **unchanged**; `tests/test_audit_and_collisions.py::test_frozen_spec_hash_is_unchanged`
  pins its hash.
* The only levers this round are (a) *which rows* a model is fitted and selected
  on, and (b) *which objective* it optimises. Nothing else.
* P0 consumes no GPU time at all.

## 1. Execution regime (one machine, one regime)

| item | value |
| --- | --- |
| host / node | `res` / `a100-2` |
| GPU | 2 × NVIDIA A100-SXM4-40GB (jobs pinned by pool `res-gpu0` / `res-gpu1`) |
| driver | 550.163.01 |
| torch / CUDA | 2.5.1+cu124 / 12.4 (fp32, no TF32 changes, no AMP) |
| python | 3.12.14 |
| commit | `de483098` (see §8 for why not the first commit) |
| queue | none; `rr` process backend |

Every `train.json` records hostname, GPU name and count, torch/CUDA version,
driver regime, commit, dirty flag, code hashes and wall-clock. Numbers from any
other regime are not comparable to these.

## 2. Model and recipe (R3, byte-identical to Sprint 02)

Frozen, no search, no early stopping, no SWA/EMA, no soup, no SSL.

| hyper-parameter | value |
| --- | --- |
| optimiser | AdamW(β=(0.9, 0.999), ε=1e-8) |
| lr schedule | per-step cosine, peak 3e-4 → min 1e-5 |
| weight decay | 1e-3 |
| gradient clip | 5.0 (norm) |
| batch size | 128 |
| epochs | **360, always** — never reduced to fit a budget |
| dropout | 0.1, one site: the final atom-level `h` before `f`/`ρ` |
| loss | task-dependent, see §5–§6 |
| seed | 0 (training + shuffle generator; shuffle seed offset fixed in `mb2_common`) |
| precision | fp32 |

Model: the frozen cross-task LCRC-A (`lipo_v1` spec, rank-32 S1/S2, shared
encoder, `ρ` head). Parameter count: 247 106 (single output), 248 536 (12 Tox21
outputs); the shared encoder initialisation is identical between the two at the
same seed (asserted in `tests/test_data_integrity.py`).

**Checkpoint rule.** Lowest Validation RMSE (regression) / highest Validation
ROC-AUC (binary) / highest Validation macro ROC-AUC (multitask); ties break to
the lower Validation BCE, then to the earlier epoch. The selected epoch is
restored *before* any prediction is reported, and the restored state is
re-evaluated: the metric must reproduce within the analytic resolution of the
statistic (§8.3). No Test-driven, no curve-shopping, no second-best epoch.

## 3. P1 — Lipophilicity split-repeat robustness (2 GPU runs + 2 CPU baselines)

**Question.** Sprint 01's Lipophilicity result is one split
(`Lipop.random122.ind.pkl`): LCRC-A 0.5727 Test RMSE vs ECFP4+RDKit+XGBoost
0.6477, while the Validation ranking was inverted. Is that a property of the
model, or of that one 4200-molecule partition?

**Splits.** Two further frozen repeats from the same published split family,
byte-verified against the Sprint-02 manifest before training:

| repeat | file | Train / Valid / Test | Train rows sha256 |
| --- | --- | --- | --- |
| 122 (reference) | `Lipop.random122.ind.pkl` | 3360 / 420 / 420 | `4e8d42a9…` |
| 123 | `Lipop.random123.ind.pkl` | 3360 / 420 / 420 | (in `analysis/split_freeze.json`) |
| 124 | `Lipop.random124.ind.pkl` | 3360 / 420 / 420 | (in `analysis/split_freeze.json`) |

Row ids, per-split row hashes and file hashes are frozen in
`analysis/split_freeze.json`; the driver re-verifies them at load time and
refuses to train if a byte changes.

**Arms.** (a) LCRC-A, R3, seed 0, one run per repeat on Train, selected on
Validation. (b) The frozen classical baseline — the same ECFP4 (2048) + 217
RDKit-descriptor matrix, the same Train-only median imputation / constant-column
drop, the same XGBoost hyper-parameters as Sprint 02, **no tuning** — fitted on
the same Train rows. Baseline feature construction is label-free, so the same
cached matrix serves all three repeats.

**Reported statistic.** Test RMSE, read **once per repeat, after**
`analysis/checkpoint_manifest.json` freezes both arms
(`analysis/baseline_repeat_manifest.json` for the baseline). Selection never
uses Test.

**Exposure.** Lipophilicity Test labels were already read in Sprint 01. These two
reads are therefore labelled `PUBLIC_SPLIT_ROBUSTNESS`, not an independent blind
test; the ledger records exactly what was read, when, and against which
checkpoint hash.

## 4. P2 — BBBP Train-only inner CV (3 GPU runs)

**Question.** BBBP's published Validation (≈0.973 AUC) and Test (0.711) are
inverted by a wide margin, and `BBBP.scaffold123/124` contain the same molecules
as `122`. If the published Validation cannot rank models for this task, is a
Train-only cross-validation a better selection instrument?

**Inputs.** Only the 1631 rows of the published `BBBP.scaffold122` **Train**
split. The published Valid and Test labels are **not read** — the driver never
constructs their cache.

**Grouping.** Bemis–Murcko scaffold of each Train molecule (RDKit
`MurckoScaffold.MurckoScaffoldSmiles`, no chirality, no MakeMurckoGeneric);
acyclic molecules have an empty scaffold and are each assigned a unique group
key so that they cannot be pooled.

**Folds.** 3-fold, seed 0, salt `lcrc-taskaware-bbbp-inner-cv-v1`, stratified at
the *scaffold* level (each scaffold goes to exactly one fold); a least-loaded
bottleneck scheduler balances positives/negatives so that no fold is
single-class:

| fold | n | pos | neg | scaffolds |
| --- | --- | --- | --- | --- |
| 0 | 545 | 448 | 97 | 257 |
| 1 | 542 | 445 | 97 | 268 |
| 2 | 544 | 448 | 96 | 190 |

Row ids, per-fold row hashes, the algorithm name and the acyclic policy are
frozen in `analysis/split_freeze.json`, and
`tests/test_data_integrity.py::test_frozen_folds_reproduce_from_the_frozen_rows`
rebuilds them from the frozen rows.

**Runs.** One R3/seed-0 run per fold (Train = the other two folds, Validation =
the held-out fold), selected on the held-out fold's ROC-AUC by the standard
checkpoint rule. Three runs, nothing else.

**What it licenses.** Whether the Train-only CV number is *usable* (variance
across folds, distance to the published Validation, whether the three folds agree
on which epoch is best). It cannot license a new BBBP Test claim: that Test has
already been used.

## 5. P3a — Tox21 equal-task objective (1 GPU run)

Sprint 02 trained Tox21 with a **masked cell mean**: every observed (molecule,
assay) cell contributes equally, so the loss is dominated by the densely
labelled assays, and a batch can be dominated by one assay.

```
masked_cell   = Σ_cells mask·BCE / Σ_cells mask
equal_task    = mean over tasks t with ≥1 observed cell in the batch of
                ( Σ_i mask_it·BCE_it / Σ_i mask_it )
```

The equal-task form gives each *assay* the same weight regardless of how many
molecules it labels; tasks with no observed cell in the batch are **excluded
from the denominator**, never counted as zero loss.

Frozen gates (`analysis/taskaware_loss_probes_p0.json`): the two losses differ on
a sparse 2-task batch (0.8167 vs 0.6322) and on a 3-task batch whose third task
is empty; they agree when only one task is present; both gradients match the
analytic weight `mask_it / (n_tasks_present · Σ_i mask_it)`.

Only the loss changes: same split, same recipe, same seed, same checkpoint rule
(Validation macro ROC-AUC), same 360 epochs. Tox21 Test is **not** read.

## 6. P3b — OGBG-MolHIV class-imbalance objective (1 GPU run)

Sprint 02's HIV run reaches Train 0.995 / Valid 0.837 / Test 0.752 AUC with a
3.74 % positive rate and an unweighted BCE. This run changes only the positive
class weight, computed **from Train labels only**:

```
pos_weight = min(5.0, sqrt(n_negative / n_positive))
           = min(5.0, sqrt(31669 / 1232)) = min(5.0, 5.0700) = 5.0   (capped)
```

The cap is part of the frozen definition, not a tuned value; it is applied before
the run and recorded per run. Everything else — split, recipe, seed, checkpoint
rule (Validation ROC-AUC), 360 epochs — is unchanged. **The exposed official HIV
Test split is not evaluated**; the ProbeEval set is not used either.

## 7. Budget, order, and stop rules

| run | workstream | expected A100 h |
| --- | --- | --- |
| `lipo_r123` | P1 | 0.10 |
| `lipo_r124` | P1 | 0.10 |
| `bbbp_cv_f0` / `f1` / `f2` | P2 | 0.03 each |
| `tox21_equal_task` | P3 | 0.19 |
| `hiv_sqrt_pos` | P3 | 0.83 |
| **total expected** | | **1.31** |

Hard cap **1.6 A100 GPU-hours**, **≤7 new GPU runs**, seed 0 only. Priority
order: Lipophilicity → BBBP → Tox21 → HIV. Before launching a job the remaining
budget must cover its expected cost; if not, the job is not started (and the
workstream is reported as not run). Epochs are never reduced to fit. No
hyper-parameter search, no second seed, no new dataset, no re-scoring of an
already-exposed Test split.

## 8. Test policy, provenance gates, and the engineering log

**Test policy.** This round reads exactly two Test splits, both Lipophilicity
repeats (§3), each once, after the checkpoint freeze. Everything else is
Validation (P1, P3) or held-out inner CV (P2). The Sprint-01/02 Test numbers are
carried forward read-only as `EXPOSED_TEST_EXPLORATORY` and are never
re-optimised. `analysis/test_access_ledger.json` is append-only and one entry per
(run, split); `ta_eval.py` refuses a second read and refuses a checkpoint whose
sha256 drifted.

**Provenance gates enforced in code, not by convention**

1. A formal run aborts unless `analysis/protocol_freeze.json` is committed at
   `HEAD` and byte-identical to the local file, the tracked tree is clean, and
   `checkpoint_manifest.json` does not yet exist (the round is closed once
   frozen). `train.json` records the protocol blob hash.
2. Split bytes are re-verified against the frozen manifests at load time.
3. Caches are rebuilt per row list, keyed by a content fingerprint, and carry
   **no label arrays**; the standardiser is fitted on each run's Train rows only
   (`valid_or_test_used: false`) and its fingerprint is recorded.
4. `tests/` (17 tests) covers row correspondence, batch alignment, forward
   wiring, label-free caches, train-only normalisation, the chirality collision,
   loss parity and the recheck bound.

**Engineering log (deviations, all recorded rather than smoothed over)**

* *Deploy block.* The remote checkout at `62afa5a2` carried untracked run
  artifacts (62 files, 13 MB) that collided with the incoming commit. Nothing was
  deleted: the tree was tarred to
  `~/cy/quarantine/lcrc_taskaware_sprint03_deployblock_20261101/deploy_block_untracked_20261101.tar.gz`
  (sha256 `c8eaeeea…`, manifest + `git status` snapshot alongside), 40 of the 62
  files verified byte-identical to the incoming commit, the 22 unique ones exist
  only in that archive, and only then were the paths cleared.
* *Commit amend.* The first deployment (`c51d5ec8`) was superseded by an amend
  (run keys); the remote was hard-reset to the previously deployed base
  `62afa5a2` and re-deployed, so no run ever executed from a superseded revision.
* *Recheck tolerance.* BBBP fold 0 finished training and then aborted in the
  post-training audit: the reloaded best state reproduced Validation ROC-AUC to
  2.3e-5 against a flat 1e-5 threshold. Cause: the model aggregates with
  `index_add_`, which sums in nondeterministic order on CUDA, and ROC-AUC is a
  rank statistic whose resolution on this fold is exactly `1/(448·97) = 2.301e-5`.
  The audit now derives the tolerance from the statistic's resolution, evaluates
  the reloaded state twice to record the observed raw nondeterminism, and keeps
  the flat 1e-5 for regression. Training code, RNG stream and checkpoint content
  are untouched. The affected BBBP runs were relaunched at the fixed commit; the
  two superseded attempts are recorded in `analysis/gpu_ledger.json` with their
  cost and are not reported as results.
* *`--dataset` omission.* The first two launches of the Lipophilicity repeats
  exited in 5 s on an argparse error and produced no artifact; they are in the
  ledger with their cost.

**Revision.** All reported runs come from `de483098` (checkpoint manifest records
each run's `git_commit`). `f49b5025` produced the two Lipophilicity repeat runs
*before* the audit-path fix; that fix touches only the post-training audit, and
both runs' audit passed under the older, stricter threshold (their diffs were
below 1e-5), so their checkpoints are unaffected. This is stated in
`checkpoint_manifest.json` per run rather than hidden.
