# PROTOCOL — `lcrc-semantic-analysis-repair-bw0-s0-v1`

Design basis: `calendar0917/research@1d6700ea01acbee33e7630f1813c7884d28bee71`.
Task book: `LCRC_Analysis_Repair_BW0_Agent_Prompt.txt` (frozen; every scientific
choice below is taken from it, not from this round's results).

This round **repairs the analysis, not the research subject**. It adds exactly
one formal neural training and re-derives everything else from stored
checkpoints, stored raw predictions, and the frozen caches.

## 1. Scope

| item | decision |
| --- | --- |
| new formal training | **one**: `B_N100_lipo_W0` (original C1 init, `weight_decay=0`, seed 0) |
| old 20 runs | reused read-only; never retrained |
| new seeds / permutations / hyper-parameters / losses / structures / datasets | none |
| PCBA / HIV / D-MPNN / new edit pairs / molecule deletions | none |
| allowed compute | checkpoint forward, pure-lookup export, CPU Ridge/XGBoost recompute, fixed-state resampling |
| Test split | declared development-evaluation target of this round; selection stays on Validation |

Predecessor rounds, read-only: `lcrc_explicit_semantic_learning_8h_s0_v1` (OLD),
`lcrc_lipo_context_small_s0_v1` (HOME), `lcrc_cross_task_preparation_v1` (PREP),
`lcrc_lipo_model_dissection_v1` (DISSECT), `lcrc_lipo_edit_supervision_public122_s0_v1` (EDIT).

The LCRC method, the data pipeline, the codec, the C1 architecture and the
training loop are **not** redefined here.

## 2. Frozen identities

| constant | value |
| --- | --- |
| datasets | `ogbg-mollipo` scaffold 3360/420/420, raw logD; `ogbg-molesol` 902/113/113 |
| C1 parameters | 228 674 |
| free table / generator / FS parameters | 33 680 / 33 689 / 228 683 |
| codec | 16 atom types, 5 bond types, 400 path codes, 80 neighbour codes, rank 32 |
| C1 recipe | AdamW β(.9,.999) ε1e-8, peak 3e-4, floor 1e-5, batch 128, dropout 0.10, clip 5.0, 360 epochs, 27 steps/epoch, 9720 updates, per-step cosine, `shuffle_offset` 917000 |
| recipes | `R` wd 1e-3, `W0` wd 0, `Rwd` wd 1e-2 |
| generator / permutation seed | 17 / 20261114 |
| bootstrap seed / replicates | 20261115 / 500 |
| kernel direction seed | 20261115, Rademacher/√32, one direction |
| replay / export tolerance | raw `max|Δ| ≤ 1e-5` |
| LS zero-coupling tolerance | `|cos| ≤ 1e-6` |
| LOW_SUPPORT rule | cell has **fewer than 20 components** |
| support df buckets | 0, 1–5, 6–20, >20 (token-weighted per bank) |
| late window | final **20** epochs (the old code's actual window, not "last 5%") |

N25/N50 keep their Fit-only target norm; N100 uses the full-Train norm.

## 3. The defect being repaired

In the concatenated PREP cache, `atom_ptr` is molecule→atom but `path_ptr` and
`nbr_ptr` are **atom→occurrence**. The old `se_data.code_support`,
`se_audit.support_counts`, `se_mechanism` and `se_explicit.build_matrix` indexed
`path_ptr[m:m+2]` / `nbr_ptr[m:m+2]` with a *molecule* index, reading only a
sliver of each molecule's occurrences (old Train `path_tokens` 10 238 vs the true
277 312 occurrences of the same `path_end` bank). `rp_reader.MoleculeReader` is
the single corrected entry point; it asserts both pointer levels before use.

## 4. Registered stages

`prepare` (identity + corrected support), `preflight` (focused checks),
`train` (`B_N100_lipo_W0`), `smoke` (adaptation replay, never a result),
`analyze` (every CPU repair section via `rp_analysis.run_all`).

One documented adaptation of the frozen loop: `se_train` builds the C1 base from
the process-global RNG and seeds only afterwards, so the old arms' initial
weights are not reproducible and not shared (`analysis/init_reproducibility.json`).
`B_W0` therefore loads the frozen documented C1 initial state
(`build_full_state(spec,norm,0)` → `remap_state('C1',…)`), the same init the HOME
round's original `B` arm used. `B_Rwd` cannot be shown to share it, and that
comparison is reported with that caveat instead of being presented as a clean
paired contrast.

## 5. Analysis commitments

* **Support**: N100/N50/N25 df from the corresponding Fit molecules; Fit-LOO
  (`support = df - 1`) for the molecules that built it, no subtraction elsewhere;
  joint = the same `(path, end)` occurrence pair, never a cartesian product;
  molecules with no S2 reported separately as `has_no_path`; frozen Fit-LOO
  terciles **keeping ties** — degenerate thresholds are reported
  `NOT_ESTIMABLE`, with no alternative quantile or threshold searched.
* **Explicit CPU coordinate**: identical columns (42+400+80+80+2+32000), same
  Ridge (`alpha=10`, `lsqr`, raw y) and XGBoost hyper-parameters, same sparse
  zero-means-missing semantics. The realised Ridge scaler is the Train column
  positive maximum (`max>0 ? max : 1`), no centring; globals are raw values.
  Old results marked `INVALID_FOR_COMPARISON`.
* **Mechanism**: replay the real best checkpoints (`strict=True`), `E(c)` from
  the actual `model.base.local.path_emb(c)` (affine buffers and FP composite
  permutation included), `s(c)=qᵀE(c)`, `J_c` over the generator parameters that
  actually influence this bank, cosine grouped by 0/1/2 shared original
  mid/b1/b2 components; init and trained-best reported separately. The LS
  direct-table kernel must be zero-coupled (implementation check only).
* **Export**: re-materialise from the reloaded best state, `strict=True` load
  into the original DISSECT `C1`, raw equivalence and recomputed RMSE/MAE. The
  old `checkpoint_best_lookup.pt` is never overwritten.
* **Edits**: reuse EDIT's 823 pairs and mappings unchanged; map by unique
  RDKit-canonical SMILES to official rows, verify labels, dedupe unordered row
  pairs, fix `row_a < row_b`, compute components once on the full graph. The
  split label of each endpoint is taken from the **official scaffold split**, not
  from the pairs file's stored columns.
* **Performance**: print both tables in full (A = old 50-point grid, B =
  every-epoch best-Valid), plus real `last` and the final-20-epoch late window.
  Initialisation effects are read only inside one recipe, recipe/weight-decay
  effects only inside one initialisation. Uncertainty = 500 common Murcko-scaffold
  `Exp(1)` perturbations with no-ring fallback, shared weights across models
  after `row_id` alignment; fixed-state sample perturbation only.

## 6. Result branches (label only; no new training)

`UNCERTAIN` when a main difference interval crosses 0 (direction and magnitude
still reported); `SPLIT_DEPENDENT` when Val/Test disagree in direction;
`NUMERICALLY_UNRESOLVED` for differences inside the measured replay error;
`NEEDS_HELDOUT_EVIDENCE` when the edit cell has fewer than 20 components. No
"smaller than X% therefore equivalent" threshold is invented.

## 7. Boundaries

Predecessor code, configs, caches, checkpoints and machine tables stay byte
identical. Only `OLD/REPORT.md` gains a short ERRATA pointer. Nothing is
published and no PR is opened.
