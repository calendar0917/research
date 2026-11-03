# LCRC sparse local-mode identity retention — v1 protocol (frozen)

Round id: `lcrc-sparse-identity-v1`.

Written and committed **before** any formal GPU training run, any Validation metric
comparison, and before the Test split is touched. Nothing below is chosen from a
score. The only things that may change after this point are factual records of what
happened (run records, metrics, ledgers). Thresholds, arms, seeds, epochs, splits and
feature definitions are immutable.

Base revision: `origin/main` at or after `3ef9a648`.

## 0. Research question (verbatim scope)

LCRC-A already receives S1/S2 local information, but that information passes through
embedding summation, a local MLP and a graph-level readout. If the **discrete identity
of the S2 patterns** is additionally retained, does it give a reproducible predictive
increment on the official MolHIV task?

Out of scope by construction: long-range information, chemical descriptors, any change
to the frozen LCRC-A trunk, any literature search, any new architecture beyond the four
pre-declared arms below, any hyper-parameter search, any new seeds, any extra arm
appended after seeing a result.

## 1. Data split (no re-splitting)

`ogbg-molhiv`, official OGB scaffold split, unchanged and reused through the frozen
production data paths:

* Train: 32,901 molecules — training + vocabulary construction.
* Validation: 4,113 molecules — checkpoint selection only.
* Test: 4,113 molecules — one-shot final evaluation after every model is frozen.

Reused read-only: `lcrc_predictive_trace_benchmark_v1/code/ptb_data.py`,
`ptb_train_mol.py`, `ptb_model.py`, `lcrc_cross_task_preparation_v1/code/lcrc_data.py`
and the frozen `lipo_v1` FeatureSpec (hash
`7270f7677b349a46bb193174381abcb84f14d1b4f83bd1362d4aa20389a515ee`, re-verified at
runtime). Existing label-free structure caches are consumed as-is; no cache is rebuilt
and no split is re-drawn. No Train-only CV.

Test discipline:

1. No Test label and no Test metric is read during training or checkpoint selection.
2. Every checkpoint, model configuration, code hash, vocabulary/feature hash and the
   Test scoring script hash are frozen into `analysis/checkpoint_manifest.json` first.
3. Only then one single Test pass scores all models/seed simultaneously.
4. No model, seed or threshold is changed or added after that pass.
5. Every Test conclusion is tagged `EXPOSED_TEST_EXPLORATORY`.

## 2. Fixed sparse local pattern definition

### 2.1 Key from the existing S2 records

No new local structure feature is designed. The key of one S2 record is

```
key = (root_atom_type, path_cat, path_end63, path_closure)
```

with `root_atom_type = atom_type[path_ptr_root_atom]` — i.e. the `atom_type` of the atom
whose `path_ptr` range contains that record — and `path_cat`, `path_end63`,
`path_closure` taken verbatim from the production cache (integer codes, no RDKit
re-parse, no new end-point encoding).

Implementation: `code/si_features.py:path_keys`. The pipeline is vectorised over the
whole split cache and is cross-checked against an independent per-molecule loop
(`analysis/feature_checks.json`).

### 2.2 Vocabulary frozen from Train structures only

1. Document frequency = number of **distinct Train molecules** containing the key.
2. Sort by `(-document_frequency, key tuple ascending)`.
3. Take the first `K = 512` keys. `K` is not tuned.
4. Persist all keys, frequencies, the ordering rule and the vocabulary SHA-256.

No label, no Validation row, no Test row enters this step. If fewer than 512 distinct
legal keys existed, the round halts with `FEATURE_INFEASIBLE` and `K` is not adjusted.

### 2.3 Molecule vector

```
c_k(G) = 1[pattern k occurs at least once in G]          (512-dim binary)
x(G)   = c(G) / sqrt(max(1, ||c(G)||_0))                 (frozen normalisation)
```

A molecule with no selected pattern yields the all-zero vector. The same frozen
vocabulary is applied to Train, Validation and Test. Per-split coverage, per-molecule
active-pattern count distribution, out-of-vocabulary path ratio and vector hashes are
recorded in `analysis/feature_checks.json`.

## 3. Fixed arms

### A0 — frozen LCRC-A

The original PTB MolHIV A0 model (`ptb_model.PTBLCRC` variant `A0`, width 64, rank 32,
full S1/S2, LocalMLP, additive local contribution `f`, original `rho`). Parameters and
forward are untouched. Retrained in this round with the same seed pool, so the paired
A0 baseline comes from the identical code path and recipe, not from an older checkpoint.

### WIDE_TRUE — real pattern identity (the single primary new model)

Complete original A0 plus a 512-dim linear sparse residual:

```
y_hat = y_hat_A0 + w^T x(G),      w in R^512, w(0) = 0, no bias
```

No extra embedding / MLP / gate / attention / new structural encoding. No A0
parameter shape changes. A0 trunk and residual train jointly end-to-end with one
optimizer.

### WIDE_NULL — marginal-matched negative control

Identical structure, 512 new parameters, identical initialisation and training recipe.
The single change: the real binary matrix is replaced by a label-blind randomised
matrix obtained by bipartite double-edge swaps, done independently per split:

* sample two existing non-zero positions `(i,a)`, `(j,b)`;
* accept only if `i != j`, `a != b`, and `(i,b)`, `(j,a)` are both zero;
* move both non-zeros to `(i,b)`, `(j,a)`.

Exactly `8 x nnz` swap attempts per split, no adaptive stop. Row sums (per-molecule
active count), column sums (per-split document frequency), total nnz, 512-dim input and
parameter count are strictly conserved. Random seeds: Train `20261103`, Validation
`20261104`, Test `20261105`. The Test randomisation exists only inside the Test scoring
pass. Randomisation never reads a label. WIDE_NULL is a mechanism control, not a
deployable predictor.

### FOLD64 — early-merge control

Same 512 patterns, same binary vectors, same normalisation. The 512 frozen keys are
assigned deterministically to 64 buckets:

1. `SHA-256("lcrc-sparse-identity-v1" + canonical key tuple)` per frozen key;
2. sort all 512 keys by digest ascending;
3. bucket of rank `r` is `r % 64`, so every bucket holds exactly 8 keys.

```
z_j(G) = (1/sqrt 8) * sum_{k: bucket(k)=j} x_k(G)
y_hat  = y_hat_A0 + q^T z(G),   q in R^64, zero-init, no bias
```

Everything else (recipe, seed pool, selection rule) is identical to WIDE_TRUE. FOLD64
has fewer new parameters than WIDE_TRUE, so any difference only tests whether the fixed
folding hurts on this task; it cannot isolate "compression timing" as a cause.

### SPARSE_ONLY — CPU diagnostic baseline

The same 512-dim normalised real vector `x(G)` alone, with

```python
LogisticRegression(penalty="l2", C=0.1, solver="liblinear", class_weight=None,
                   fit_intercept=True, tol=1e-4, max_iter=4000,
                   random_state=20261103)
```

fit on official Train only. No C search, no g42/ECFP, no forest, no hyper-parameter
tuning. It answers only "how much predictive information do these fixed S2 patterns
carry on their own".

## 4. Training protocol

Neural arms: `A0`, `WIDE_TRUE`, `WIDE_NULL`, `FOLD64`; seeds `0, 1, 2`; therefore
`4 x 3 = 12` formal runs plus one CPU SPARSE_ONLY fit.

Recipe (identical to the historical PTB MolHIV recipe, not re-tuned): AdamW
(betas 0.9/0.999, eps 1e-8), peak LR 3e-4, per-step cosine to 1e-5, weight decay 1e-3,
batch 128, dropout 0.10 at the single historical position (final atom-level `h`),
global gradient clip 5.0, unweighted `BCEWithLogitsLoss`, exactly 100 epochs, no
pretraining, no EMA/SWA/soup/ensembling, no class weighting, no early stopping.

Same-seed requirements: A0 initial state bitwise identical; identical data shuffling
(seed-only `torch.Generator`, offset 917,000); the new branch is zero-initialised, so at
step 0 the wide models reproduce the same-seed A0 output within 1e-6 (CPU float32); the
new branch must not consume the global RNG stream that A0 uses (built inside a forked
RNG context), so dropout masks, shuffling and A0 initialisation stay identical.

Checkpoint selection: the existing `ptb_train_mol.Tracker` MolHIV rule —
highest Validation ROC-AUC; tie -> lower Validation BCE; tie -> earlier epoch. AP is
reported but never used for selection. One checkpoint per (arm, seed). All 12 runs run
to completion regardless of Validation outcome.

## 5. Correctness gates (must pass before/independent of the formal comparison)

1. Official Train/Validation row ids, labels and model inputs aligned.
2. Vocabulary produced from Train structures only, no label dependency.
3. Key construction matches the production `path_ptr` / `path_cat` / `path_end63` /
   `path_closure` / `atom_type` records exactly (vectorised vs per-molecule loop).
4. The pattern vector is invariant to atom renumbering and to path record order.
5. Batched and single-molecule computation give the identical vector.
6. WIDE_TRUE and WIDE_NULL have exactly the same parameter count.
7. NULL row sums, column sums and total nnz equal TRUE's, per split.
8. NULL differs from TRUE; the change fraction is recorded.
9. FOLD64 has exactly 64 buckets with exactly 8 keys each.
10. With zero-initialised branches, A0 / WIDE_TRUE / WIDE_NULL / FOLD64 predict the same
    values for the same seed (CPU float32 max abs err <= 1e-6).
11. The new parameters receive non-zero gradient, are in the optimizer, and change after
    one update.
12. Trunk parameters, data caches and training recipe match the historical PTB A0.
13. A saved and reloaded checkpoint reproduces its Validation predictions.
14. No Test label is loaded and no Test metric computed during training.

Additional record (not a gate): whether the previously constructed 18-ring molecule pair
still collides in this 512-dim representation. A collision does not fail the round —
this round is not about long-range expressivity.

## 6. Test evaluation (mandatory)

After all 12 neural checkpoints and the SPARSE_ONLY model are final, freeze and hash:
checkpoint manifest, training/model code, parameter+initialisation hashes, vocabulary and
feature matrix hashes, all best epochs, Validation predictions and metrics, and the Test
scoring script. Then run the single Test pass, which scores every arm and seed:
A0 x3, WIDE_TRUE x3, WIDE_NULL x3, FOLD64 x3, SPARSE_ONLY x1. No Test-triggered
skipping, re-ranking, seed selection or retraining. ROC-AUC via the official OGB
`Evaluator(name="ogbg-molhiv")`, cross-checked with sklearn; AP also reported.

## 7. Analysis and statistics

Primary: ROC-AUC (higher better). Secondary: Average Precision. Per seed, on Validation
and on Test separately:

```
WIDE_TRUE - A0, WIDE_TRUE - WIDE_NULL, WIDE_TRUE - FOLD64, WIDE_NULL - A0, FOLD64 - A0
```

Reported: absolute ROC-AUC/AP per arm and seed, 3-seed mean and sd, paired per-seed
differences, Train ROC-AUC and Train-Validation gap, best epoch, new-branch L1/L2 norm,
distribution of the branch's absolute logit contribution, parameter counts, train time,
peak GPU memory, SPARSE_ONLY Validation/Test ROC-AUC and AP, and a descriptive
(non-paired) distance to previously reported fingerprint/RF numbers. No new RF run.

Paired scaffold bootstrap: 2,000 replicates, Bemis-Murcko scaffold group as the
resampling unit, identical resampled rows for every arm and seed, 3-seed mean ROC-AUC
difference, seed `20261103`, 95% percentile interval, Validation and Test separately.
The bootstrap describes sample uncertainty for fixed checkpoints and a fixed split; it
does not replace training-seed uncertainty and the Test interval is not confirmatory
evidence from a fresh blind test.

## 8. Pre-registered decision

`PREDICTIVE_IDENTITY_SIGNAL` requires **all** of:

1. WIDE_TRUE minus A0 3-seed mean ROC-AUC >= +0.010 on Validation **and** on Test;
2. at least 2 seeds beat their paired A0 on Validation **and** on Test;
3. WIDE_TRUE minus WIDE_NULL 3-seed mean ROC-AUC >= +0.005 on Validation **and** Test;
4. WIDE_TRUE mean Test AP >= A0 mean Test AP;
5. every correctness gate passes.

That verdict means only: under the current configuration, retaining stable S2 pattern
identity shows a predictive signal worth further study. It does not prove compression is
the only bottleneck and it is not a fresh blind-test gain.

`NO_PERFORMANCE_SIGNAL`: WIDE_TRUE minus A0 3-seed mean ROC-AUC <= 0 on Validation or
Test.

`NO_IDENTITY_SPECIFIC_SIGNAL`: WIDE_TRUE has a predictive increment but misses the
pre-registered identity-specific margin over WIDE_NULL.

`MIXED_OR_WEAK_SIGNAL`: partial positive evidence that satisfies none of the above.

`INCOMPLETE`: execution/data/resource problem makes the comparison unavailable.

Thresholds are not changed. A lucky FOLD64 or SPARSE_ONLY result never adds an arm.

## 9. Resources

Host `res`, pools `res-gpu0` and `res-gpu1`, one formal training process per GPU at a
time, launched through `rr` from a committed revision. GPU soft budget 4.0 A100 GPU-h,
hard cap 5.0 GPU-h, both cards billed together. A minimal performance pre-check (timed
short run on the full Train set) precedes the formal runs and is billed to the ledger.
If the projection exceeded the hard cap, the round stops and reports — it never shrinks
the arm count, seed count, epoch count or dataset. A failed formal run is reported and
diagnosed, never silently replaced by an amended plan.

## 10. Deliverables

`PROTOCOL.md`, `REPORT.md`, `DECISION.md`, `code/`, `tests/`,
`analysis/feature_vocabulary.json`, `analysis/feature_checks.json`,
`analysis/null_checks.json`, `analysis/initialization_checks.json`,
`analysis/validation_metrics.json`, `analysis/test_metrics.json`,
`analysis/paired_bootstrap.json`, `analysis/checkpoint_manifest.json`,
`analysis/test_access_ledger.json`, `analysis/predictions/`,
`analysis/gpu_ledger.json`, `analysis/provenance.json`.
