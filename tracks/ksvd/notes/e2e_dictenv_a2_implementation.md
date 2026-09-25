# E2E-DictEnv-A2 — implementation note (Attributed Code Value & Compression Control)

Round: **E2E-DictEnv-A2** · preregistration `1813f532247e8111cd7c032dc8ef7ec8c1bab64f`
(frozen, `tracks/ksvd/notes/e2e_dictenv_a2_preregistration.md`) · prior-artifact audit
`tracks/ksvd/notes/e2e_dictenv_a2_prior_artifact_audit.md`.

This note records **what was implemented, how the frozen A1 objects are reused, and
what was verified locally**.  It is a provenance document: every deviation from "call
the frozen A1 function" is listed with its rationale and the pre-registered rule it
implements.

---

## 1. Files

| file | role |
|---|---|
| `tracks/ksvd/experiments/luyin16/e2e_dictenv_a2.py` | round core: frozen constants, the continuity-v2 pair population and statistics, the train-only rank-32 dense reference (delegating the PCA to `sdb_v0`), the liveness gate, the shared-IHT-step rule, and the frozen decision table as a pure function |
| `tracks/ksvd/experiments/luyin16/zinc_e2e_dictenv_a2.py` | runner: 16 stages (identity → qualify → continuity-v2 → Stage 1/2/3/4/5 → decision/report, plus a plumbing smoke) |
| `tracks/ksvd/tests/test_e2e_dictenv_a2.py` | 37 focused CPU tests (reuse discipline, identity, continuity-v2, PCA control, matched execution, coder rule, decision table, official-test blocker, GPU policy) |
| `tracks/ksvd/results/e2e_dictenv_a2/` | artifacts: identity, continuity, decision, report (§29 of the brief) |

No A1 file was modified.  `git diff` against `db1aa48` touches only new files.

---

## 2. Reuse map (what A2 calls instead of re-deriving)

| object / stage | A2 call site | source of truth |
|---|---|---|
| 433-D REAL/INDEP coordinate + train-only scaler | `a1run.arm_coordinate(arm, split)` | A1 `cache/`, `scaler_meta.json` |
| exact `s=8` OMP codes | `a1run.load_or_build_codes(arm, split)` | A1 `cache/omp_{arm}_{split}.pt` |
| frozen K-SVD `K=32` dictionaries | `a1run.load_arm_dictionary(arm)` (+ sha check) | A1 `dictionary_*.pt`, `SDB_DICT_SHA256_F32` |
| Gate-0 audits (correctness / assignment / health / accounting / official-test blocker) | `a1run.correctness_stage`, `assignment_stage`, `health_stage`, `accounting_stage` | A1 runner |
| frozen trainer (Top-5 soup, λ, horizon, clipping, batch order) | `a1run.train_arm(arm, stage=…)` | A1 runner |
| IHT coder qualification | `a1run.iht_diag` | A1 runner |
| Stage 4 selection | `a1run.formal_runs` | A1 runner |
| mechanism interventions | `a1run.mechanism_stage` | A1 runner |
| Stage 5 dense-tied control | `a1run.specificity_stage` | A1 runner |
| attributed-WL continuity pipeline | `tccd._adjacency`, `_patch_graph_for_wl`, `attributed_wl_fingerprint`, `tccd.patch_slot_order`, `a1run._histogram_matrix` | the same shared code A1 used |
| PCA / dense reference | `sdb_v0.fit_pca_rank` / `sdb_v0.PCARank` | SDB-v0 (no A2 PCA) |
| usage statistics | `a1run._usage_stats`, `a1run._SlotCapture` | A1 runner |

`a1_emits_into(root)` re-points `RESULTS_DIR / STATE_DIR / CURVE_DIR / FORMAL_DIR` of the
**A1 runner module** at the A2 results directory for the duration of a frozen-stage call,
so A1 functions write into A2's round directory and **cannot** overwrite A1 artifacts.
A test asserts the four module constants are restored afterwards and that a
`qualify_stage()` run leaves `results/e2e_dictenv_a1/*.json` untouched.

`dictionary_override(arm, D)` serves one frozen dictionary to the frozen trainer
(Stage 2 only) and is arm-scoped + restored.

---

## 3. Operationalisation decisions (all within the frozen spec)

1. **Stage-2 dense reference is the frozen `Vᵀ`, the *centred* codes are attached.**
   Pre-registration §5 froze: *"the model's `D` is set to `Vᵀ`, whose columns are
   unit-norm, and the centred codes are attached as precomputed coordinates, exactly as
   stage 1 attaches frozen OMP codes."*  Implemented literally:
   `D = pca.components.T` (float32; columns unit-norm to ≤3e-8, so the trainer's internal
   `F.normalize(D, dim=0)` reproduces `D` bit-for-bit — `projection_vs_normalized_dictionary_max_abs
   = 0.0`), codes `= (X − mean_train) @ Vᵀ` attached via `stage="omp"` with
   `coding_mode="omp_frozen"` and `freeze_dictionary=True`.
   Both residuals are recorded in `pca_meta_{indep,real}.json`: the affine PCA residual
   (`train_normalized_rec`, `valid_affine_normalized_rec`) **and** the tied residual the
   frozen decoder actually sees (`train_tied_normalized_rec`, `valid_tied_normalized_rec`),
   which differs by the not-added mean term.  No mean is invented: the tied decoder of
   A1 adds none (that is a property of the reused model, not a choice made here).
2. **Stage 1/2 recompute no statistics.**  The screen value is
   `soup["soup_valid_mae"]` straight out of `a1run.train_arm`; `G_pair`/`G_topo` are the
   frozen differences.
3. **Finiteness is fail-closed.**  `_curve_is_finite` returns **False** for a missing,
   empty or non-finite curve.  `train_arm` writes state and curve *before* the run JSON,
   so a present run JSON implies a present curve; a missing curve therefore means
   tampered/cleaned artifacts, and a stage that cannot show a finite curve cannot pass.
4. **§11 matched initialisation is *enforced*, not assumed.**  `matched_init_enforced()`
   hands the frozen `a1.build_model`'s own `reference_state` copy a single reference state
   (the REAL arm, `D` excluded) for every training call.  `identity_stage` proves the
   enforcement is a **no-op** on the frozen A1 initialisation
   (`enforcement_is_a_noop = True` for all three arms), that all 50 shared-shape tensors
   coincide with the reference (`matches_reference_state = True`) and that each arm keeps
   its **own** frozen dictionary (`D_kept_per_arm = True`).  If any of those three failed,
   the blocking identity gate would stop the round — the "same initialisation" claim can
   never silently move the numbers.
5. **Liveness slot instrument.**  The frozen text requires *"node / edge / anchor slot
   gradient norms all > 0"*.  A1's `_SlotCapture` probes the **input tensor** of each slot
   module; the anchor encoder's input tensor is grad-free by construction (verified: the
   probe reports 2 of 3 slots while all three modules receive nonzero parameter
   gradients).  A2 therefore gates on the slot module's **parameter** gradient norm and
   reports A1's input probe next to it (`param_grad_norm`, `slot_grad_norm`,
   `slot_value_std`, `param_tensors`).  A slot is dead iff its parameters receive no task
   gradient; the input probe is used only when the parameter norm is absent.  Everything
   else in the frozen liveness list is unchanged.
6. **Stage order** follows the preregistration's section order: §7 Stage 4 (E2E) →
   §8 mechanism (required once the E2 run completes, independent of `G_pair_E2E`) →
   §9 liveness → §10 conditional Stage 5.  The §8 mechanism runs *before* the §9 liveness
   audit so that a liveness failure cannot suppress a required mechanism number.
7. **Stage-2 alias.**  `formal_stage` writes the A1-compatible `formal_selection.json`
   (consumed by the frozen mechanism/specificity guards) with `primary_pass` re-stated
   under the A2 rule (material threshold **and** finite curves), so the frozen downstream
   guards cannot proceed past a Stage 4 the A2 table has already rejected.
8. **Continuity-v2 pooled row.**  Strata are equal-`n` (4000) capped subsamples of one
   seeded permutation of the valid pair population; the pooled row is the first 16000
   pairs of that same permutation (an unbiased sample of the pooled population, **not**
   the union of the strata — the payload says so explicitly).  The strata themselves are
   pairwise disjoint and partition the size axis (`n_valid_pairs_outside_strata = 0`).
9. **The shared IHT step rule** lives in `e2e_dictenv_a2.select_common_iht_steps`, and
   `coder_stage` re-derives A1's `qualified_steps`/`selected_steps` with it and **raises**
   if A1's artifact disagrees.  Incomplete evidence (a missing candidate step) is refused,
   never inferred.
10. **The decision table is a pure function** (`verdict_from_evidence`) that raises
    `ValueError` on an interrupted evidence combination instead of inventing a verdict.
    Every stage the frozen order leaves unexecuted is written to `stage_status.json` as
    `NOT RUN` with its frozen reason (`_not_run`).

---

## 4. Local verification (CPU, this checkout)

Environment: no GPU locally; every number below is a CPU-side provenance/plumbing check.

### 4.1 Artifact identity (`identity_stage`) — 26/26 entries, 67 s

* 3 dictionary sha256 pins: `TOPO b0c5da98…3dfecd`, `INDEP 400821ee…6ff32d`,
  `REAL c1cafb08…d4b809` — match the tracked `SDB_DICT_SHA256_F32` constant and the A1
  decision record / `STATE.yaml`;
* 10 raw-cache tensor shas (train + valid) match the tracked prefixes and the
  `cache_meta_{train,valid}.json` records; shape cross-checks
  (train 10000/231664/1 418 500, valid 1000/23083) pass;
* scaler `real` and `indep` **recomputed exactly** from the raw train cache
  (`max_abs_scale_difference = 0.0`, masks and block weights identical);
* all six OMP caches re-derived with `sdb.omp_codes(normalize_columns(D), X, s=8)`
  **bit-identically** (full split, no row limit);
* §11 matched-init report: all three arms pass (50 shared tensors, no-op enforcement,
  per-arm dictionaries).

### 4.2 Qualification (`qualify_stage`) — PASS, 32 s

A1's Gate-0 numbers reproduced on the reused artifacts: correctness 19/19, assignment
semantics PASS (REAL sensitive `3.6243410110473633`, INDEP invariant `0.0`), health PASS
(holdout normalised reconstruction `TOPO 1.410e-05`, `INDEP 2.109e-02`, `REAL 1.971e-01`;
used atoms 32/32/31; effective atoms 4.03/18.59/19.99; train↔valid usage Spearman
0.9989/0.9989/0.9993), parameter accounting PASS, official-test blocker PASS.

### 4.3 Continuity-v2 (`continuity_v2_stage`) — diagnostic, 8.7 s

Pool: 2000 rooted official-train patches, seed `20260930`, `n_patch`-cross-checked
(`n_patch_size_mismatch = 0`); 1 999 000 pool pairs → 371 455 valid (equal size,
molecule-disjoint, distinct typed-WL canonical key) → 4000 per stratum + 16000 pooled.

| stratum | n valid pairs | REAL x | INDEP x | TOPO x | REAL code | INDEP code | TOPO code |
|---|---|---|---|---|---|---|---|
| small (≤5) | 76 507 | 0.2494 | 0.2414 | 0.0598 | 0.2282 | 0.4165 | 0.0512 |
| medium (6–8) | 290 545 | 0.2756 | 0.2458 | 0.1249 | 0.2546 | 0.4037 | 0.1228 |
| large (≥9) | 4 403 | 0.2938 | 0.2350 | 0.2187 | 0.2729 | 0.3057 | 0.0861 |
| pooled | 371 455 | 0.2971 | 0.2750 | 0.1350 | 0.2464 | 0.4014 | 0.1329 |

(statistic: Spearman(attributed-WL cosine, −angular distance); see §5 for the A1
numbers these must be compared with.)

**Reconciliation with A1 (the provenance resolution of the audit §3).**  Re-running A1's
exact producing recipe on A1's own pool (1500 patches, seed `20260933`; 200 000 draws,
seed `20260934`) reproduces the analysis-note population exactly:
`n_sampled_pairs = 99102`, `n_equal_size = 18846`, `n_equal_size_molecule_disjoint = 18843`,
and all six strict statistics match A1's `gate0_stratum_diagnostics.json` to `<1e-9`
(REAL x 0.28699 / code 0.26007; INDEP x 0.28085 / code 0.35470; TOPO x 0.12823 /
code 0.13306), as do the "all pairs" rows (REAL 0.31749/0.25925, INDEP 0.32729/0.29616,
TOPO 0.51749/0.53452).  The remaining A1 report figures (REAL x 0.3175 / code 0.2593 …)
are the same statistic on the *unequal-size* population — a population difference, not a
discrepancy (audit §3).

**Pipeline identity.**  A2's own attributed-WL + `patch_slot_order` reconstruction of
A1's pool reproduces A1's similarity matrix with `max_abs_similarity_difference = 0.0`,
identical sizes/molecules and an identical key partition — the new diagnostic is built by
the same pipeline as A1's numbers, so the two are comparable.

### 4.4 Trainer plumbing smoke (`smoke_stage`) — CPU, 24 molecules × 1 epoch

All three OMP arms and the IHT arm ran through the real training loop: dictionaries frozen
in the OMP arms (`dictionary_grad_norm = null`), trainable in the IHT arm
(`||dL/dD|| = 6.48`, movement 0.0165), all slot parameter gradients nonzero, all curves
finite.  Writes only under `results/e2e_dictenv_a2/smoke/` and is never read by a gate,
decision or report number.

### 4.5 Tests

`uv run pytest -q tracks/ksvd/tests/test_e2e_dictenv_a2.py` → **37 passed** (2 marked
`slow`: the full identity audit and the continuity reconciliation + pipeline identity).
A1's own suite still passes (`test_e2e_dictenv_a1.py`), and
`uv run ruff check` is clean on all new files.

Covered explicitly: A1 dictionary sha pins; 433-D REAL/INDEP identity through the frozen
scaler; OMP cache bit-determinism (and the cache is not rewritten); the frozen strata
`{(0,5),(6,8),(9,∞)}` against the train `n_patch` quantiles (q25 = 5, q90 = 8) and the
pre-registered stratum shares; the pair sampler (determinism, equal size,
molecule-disjointness, distinct keys, cap, disjoint populations, `i<j`);
Spearman/angular-distance arithmetic **including the frozen ordinal-rank tie convention**;
train-only PCA32 (width 32, delegation to `sdb_v0`, valid never enters the fit, no refit,
stored-code integrity guard, the tied/affine gap identity); the §11 matched-init
enforcement; loader-seed sharing across arms; parameter accounting equal for INDEP/REAL at
109 263 (TOPO 97 487) with identical trainable sets; the shared-IHT-step rule (including
the "one shared count for all arms" bite and refusal of incomplete evidence); every branch
of the frozen decision table plus its refusal of interrupted evidence; fail-closed curve
checks; Stage-1/2 threshold arithmetic; the report/`stage_status` `NOT RUN` bookkeeping;
the official-test blocker (no `"test"` literal anywhere, `official_test_loaded: false`
everywhere); `CUDA_VISIBLE_DEVICES` must be exactly `1`; and the stage list.

---

## 5. What was *not* run locally, and how it is covered remotely

Stage 1/2/3/4/5 training needs CUDA.  They are **not** marked as run: `stage_status.json`
keeps every unexecuted stage `NOT RUN` with its reason, and no number in this note is
presented as a formal result.

Remote plan (GPU1 only, `bash scripts/run_remote.sh 1 …`):

```
preflight   : ssh res, remote checkout clean, uv sync --frozen, research doctor, ZINC, GPU1 visible/usable
deploy      : scripts/deploy.sh <commit>
smoke       : bash scripts/run_remote.sh 1 python -m tracks.ksvd.experiments.luyin16.zinc_e2e_dictenv_a2 smoke --device cuda
formal      : launch_remote.sh 1 a2-full python -m tracks.ksvd.experiments.luyin16.zinc_e2e_dictenv_a2 all --device cuda
await       : wait_remote.sh
pull        : pull_results.sh → local analysis → claims/decisions/STATE → commit + push
```

`run_all` is resumable per stage (`train_arm` returns its JSON if present), so an
interrupted session is continued, never re-decided.  Seed 1 remains untouched: it is
authorised only if `G_pair_E2E ≥ 0.003` and would run as a separate explicit stage on GPU1,
sequentially.

### Hazards tracked for the remote run

* **Memory.**  `continuity_v2_stage` holds ~1 GB of coordinate rows; `compression_stage`
  ~2.5 GB peak per arm.  Both are transient and were exercised locally within 27 GB.
* **Artifact merge.**  Local CPU-side artifacts under `results/e2e_dictenv_a2/` are
  *validation* artifacts.  The local directory is wiped before pulling the remote results,
  so a passing local identity can never mask a failing remote one.
* **GPU0.**  Never referenced.  The runner refuses any `CUDA_VISIBLE_DEVICES` other than
  `1` before touching CUDA; there is no DDP, no dual-GPU and no parallel CUDA process.
