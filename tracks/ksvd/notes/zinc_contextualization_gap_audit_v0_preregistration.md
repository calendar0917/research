# CGA-v0 — Contextualization Gap Audit (pre-registration)

Protocol ID: `zinc_contextualization_gap_audit_v0`
Date: 2026-09-23
Status: frozen before any audit run.
Regime: existing-checkpoint **deterministic inference/export only**; official ZINC
test never loaded.

## 0. Hard budget

```text
MAX NEW FULL TRAINING RUNS = 0
```

Forbidden this round: retraining the backbone, seed sweep, architecture
modification, dictionary rescue, HPO, new end-to-end model, official test.

Allowed: deterministic inference from existing durable checkpoints, state
export, train-only unsupervised / fixed probes, nearest-neighbour analysis,
small CPU/GPU diagnostic regressions, bootstrap statistics.

If a required checkpoint is absent and cannot be recovered from durable/local
artifacts the run stops with `STOP_AND_REPORT_MISSING_ARTIFACT`. No retraining.

## 1. Question

Does the advantage of the strong recurrent pair–centre model come from a
**local/static representation cannot by itself decide** contextual state
refinement? Concretely, do patches with the same / nearly-same local
environment acquire **substantially different contextual states** after the
recurrent computation, and is that contextualization **linked to the recurrent
prediction gain**?

This is an audit, not an architecture round. It decides whether continuing to
search for a sixth local / pair / topology-cross dictionary (a strict-static
substitute) is justified, or whether contextual computation itself is the
bottleneck.

## 2. Checkpoints reused (primary)

Primary family = the matched compact-v4 small-head pair:

| role | builder | seed | state file | params | training commit |
|---|---|---|---|---|---|
| recurrent | `rec.build_recurrent(s)` | 0, 1 | `results/compact_v4_recurrent_pair_centre/states/recurrent_seed{0,1}_selection_state.pt` | 82,115 | `a3515e76` |
| matched one-shot baseline | `shead.build_smallhead(s)` | 0, 1 | `results/compact_v4_recurrent_pair_centre/states/baseline_seed{0,1}_selection_state.pt` | 82,115 | `a3515e76` |

Recorded best official-valid MAE: recurrent seed0 `0.13837560486892472` (ep 199),
seed1 `0.13343997858563672` (ep 234); baseline seed0 `0.14533376283763208`
(ep 176), seed1 `0.13805893784115325` (ep 143). Recorded on CPU
(`torch 2.5.1+cu124`, `numpy 2.1.3`, `torch_threads 4`).

The recurrent and matched baseline share protocol, seed, local/pair machinery,
parameter scale; recurrence is the main difference. DTX / SRDA / SDPK numbers
are **not** used as primary causal comparators.

### 2.1 Auxiliary checkpoint (descriptive only)

B-Null seed 0 (`results/local_token_null/states/lt_null_seed0_selection_state.pt`,
`ltn.build_null(0)`, recorded CUDA best-valid `0.12687269969756015`). It is a
`T=2` recurrent pair–centre model with the molecule-dependent 16-D local token
deleted. It cannot replace the primary 2-seed recurrent analysis; it only checks
whether contextual splitting survives without a sophisticated local token.

## 3. Exported computation states

Deterministic forward over official train (10,000) and official valid (1,000),
batch 128, shuffle `False`, `eval()` + `no_grad()`, **CPU** (matches the
checkpoints' original device and removes atomic-`index_add_` CUDA drift).
For every patch/centre `i` the export captures the real existing tensors:

```text
graph_id, local centre index
patch_cont (static descriptor x_i)
h0_i, h1_i, h2_i
A0_i = last_center_context0 (pair->centre pooled context, 165-D)
A1_i = last_center_context1
```

with the true computation `h0 -> q0 -> A0 -> h1 -> q1 -> A1 -> h2`. Per
molecule it also captures `y`, `yhat_matched_baseline`, `yhat_recurrent`,
`n_atoms = n_patches`, `n_patches`, and (descriptive) `C_q` from `q1-q0`
(`q0/q1` are also exported raw for a fixed 128-molecule verification subset,
since per-pair export of the whole split is unnecessary for the audit's
primary metrics). No new forward computation is invented; every exported tensor
is produced by the checkpoint's own `encode`/`forward`.

`A0/A1` are pair states pooled back to a centre, i.e. exactly the forward
message-passing / pair→centre information flow. They are used **only** as audit
inputs to test whether the contextual shift is explained by centre context.
They are **not** evidence that a strict-static model may legally consume `A0`.

## 4. Integrity gates (must hold before interpretation)

* **A. Prediction reproduction** — re-evaluated valid MAE equals the recorded
  best-valid MAE to `|Δ| ≤ 1e-4` (target: exact CPU reproduction). Also the
  matched-baseline valid MAE is re-evaluated the same way.
* **B. State ordering** — concatenated per-graph `y` equals the run record's
  `valid_targets`; per-graph patch counts sum to the exported patch count; the
  per-graph `graph_id` mapping is verified by a single-molecule re-run spot
  check matching its batched slice.
* **C. h/q/A reconstruction** — recomputing `A0` from the stored `q0` via
  `_pool_pairs_to_centres` matches `last_center_context0` to `≤ 1e-5`, and
  `q0 = _pair_value(h0, …)` matches `last_q0` to `≤ 1e-5`, on the first
  128-molecule valid batch.
* **D. Determinism** — a repeat export of a fixed 256-molecule valid subset
  reproduces `h0/h1/h2`, `q0/q1` and the prediction bit-identically (max `|Δ| = 0`).
* **E. No official test** — every artifact carries
  `official_test_loaded = false`; the official test cache is never opened.

## 5. Audit A — local-neighbour contextual divergence

Train patches form the database; valid patches are queries (no same-molecule
leakage is possible).

Two local spaces, computed separately:

* **Space X** — `x_i = patch_cont` (the inherited pipeline already applies
  train-fit per-dimension standardisation; re-standardised with train mean/std
  as a defensive check).
* **Space H0** — `h0` standardised with train mean/std.

`k = 16` primary (no k chosen from results). For a space `S` with neighbour set
`NN16^S(i)` over train patches:

```text
d0_i^S     = mean_j ||h0_i - h0_j||
d2_i^S     = mean_j ||h2_i - h2_j||
delta_i    = h2_i - h0_i
dDelta_i^S = mean_j ||delta_i - delta_j||
Amp_i^S    = d2_i^S / (d0_i^S + 1e-8)
Dnorm_i^S  = dDelta_i^S / sigma_delta,   sigma_delta = RMS||delta - mean_delta||
R_delta^S  = median_i dDelta_i^S / (sqrt(2) * sigma_delta)
```

`R_delta` is dimensionless: `~1` means h0-neighbour deltas are as different as
random pairs; `~0` means the shift is locally determined.

**Exact-local-state subset (added in Amendment A1).** Because the h0 space
contains many exact duplicates, the audit reports the subset of valid patches
with `d0^{H0} ≤ 1e-6` separately: `n`, fraction, train exact-duplicate count
(median / mean / fraction `≥16`), `d2`, `dDelta`, `dDelta/(sqrt(2)σ_delta)` and
the tie-robust retention ratio. This is the cleanest “same local environment”
probe.

**Neighbour retention** (`H0` vs `H2`):

```text
J_i = |NN16^{H0}(i) ∩ NN16^{H2}(i)| / |NN16^{H0}(i) ∪ NN16^{H2}(i)|
```

Report mean / median / p25 / p75 of `J`, **but J is not decision-bearing**
(Amendment A1): when a query has tens to hundreds of exact h0 duplicates, both
`NN16^{H0}` and `NN16^{H2}` can lie inside the same duplicate group while
overlapping little, so a low `J` is a tie artifact, not evidence of
scrambling. The decision uses the tie-robust retention ratios against the
matched random control instead:

```text
retention_d2_ratio     = mean_i d2^{H0}_i     / mean_i d2_rand_i
retention_dDelta_ratio = mean_i dDelta^{H0}_i / mean_i dDelta_rand_i
```

`≈1` means h0-neighbours are no more similar in h2 than matched random patches
(contextual state not preserved); `≪1` means the local neighbourhood is
largely preserved.

**Matched controls**

* `d0/d2/dDelta` for the same statistic under a **random control**: for every
  valid patch, up to 16 train patches sampled uniformly from the same
  (root-atom category, molecule-size decile) cell (fixed RNG seed
  `20260923`). Root category is recovered as `argmax(patch_cont[108:136])`
  (the root-atom one-hot block of the 146-D shell descriptor). This is the
  context-free dispersion baseline.
* **d0-bin control**: valid patches are binned by `d0^{H0}` deciles; report
  `Amp` and `R_delta` per bin. If `Amp` is roughly flat across `d0` bins the
  divergence is not merely poor nearest-neighbour quality.

Primary space for the divergence/task claim is `H0`; `X` is the raw
local-environment robustness view.

## 6. Audit B — can local state predict the contextual refinement?

Target-free (ZINC `y` is never used here). Target is the shift `delta_i`, not
`h2_i` (which trivially contains `h0`).

* **Local predictor** — `delta_hat_i = mean_{j∈NN16^{H0}(i)} delta_j`;
  `MSE_local = mean_i (1/d)||delta_i - delta_hat_i||^2`; `MAE_local`.
* **Constant baseline** — `delta_bar = mean_{train} delta`;
  `MSE_mean = mean_i (1/d)||delta_i - delta_bar||^2`.
* `P_local = 1 - MSE_local/MSE_mean` (non-parametric out-of-sample R²).
* **Context-aware oracle** — `z0_i = [H0_i ; A0std_i]`,
  `delta_hat_ctx_i = mean_{j∈NN16^{z0}(i)} delta_j`; `MSE_context`;
  `P_context = 1 - MSE_context/MSE_mean`;
  `context_gain = (MSE_local - MSE_context)/MSE_local`;
  `context_abs_gain = (MSE_local - MSE_context)/MSE_mean` (Amendment A1: the
  **absolute** fraction of the shift variance that only context explains, which
  is the decision-bearing quantity; `context_gain` on the residual can be large
  while the absolute increment is negligible).
  `context_share_of_h2_variance = context_abs_gain · Var(delta)/Var(h2)`.
* **Optional A1 oracle** — `z1_i = [H0_i ; A0std_i ; A1std_i]`;
  `P_context_A1`, `context_gain_A1`. Diagnostic ceiling only, never a primary
  decision gate.

`A0` is an audit input only and is never used to train any predictive backbone.

## 7. Audit C — local environments split into contextual roles

Target-free spherical k-means (nearest-prototype partition) on **train `H0`**:
`K = 64`, fixed seed `20260923`, deterministic k-means++ init, Lloyd iterations
until assignment is stable or `max_iter = 100`, cosine geometry. No `y`, no
`K` sweep, no validation model selection, not used for prediction.

For every prototype `k` with `n_k ≥ 50` report:

```text
Var(H0 | k), Var(delta | k), Var(H2 | k)
effective contextual rank of delta within k = participation ratio of Cov(delta|k)
S_k = trace Cov(delta | k) / trace Cov(delta)
within-prototype H0 radius, within-prototype delta radius
```

and the sample-weighted mean and median `S_k`. A tight `H0` prototype with a
high-variance / multi-modal `delta` means one local dictionary atom does not
determine its contextual role.

## 8. Audit D — is contextualization linked to the prediction gain?

Per valid molecule `G`:

```text
err_base(G) = |y_G - yhat_baseline_G|
err_rec(G)  = |y_G - yhat_recurrent_G|
gain(G)     = err_base(G) - err_rec(G)      (gain > 0 = recurrence helps)
C_h(G)      = mean_i ||h2_i - h0_i||        (primary)
C_h1(G)     = mean_i ||h1_i - h0_i||
C_q(G)      = mean_pairs ||q1 - q0||        (descriptive)
C_A(G)      = mean_i ||A1_i - A0_i||        (descriptive)
```

* Primary task link: `Spearman(C_h, gain)`, also `Pearson`.
* Molecule bootstrap: 10,000 resamples, fixed seed `20260923`, 95% percentile CI.
* Quartile contrast: molecules split by `C_h` into Q1..Q4; per-quartile mean
  `gain`; `gain_Q4 - gain_Q1` with bootstrap CI.
* Size confound control (primary covariate = molecule size, `n_atoms = n_patches`):
  rank-residualise `C_h` and `gain` on `rank(size)` by OLS; report
  `Spearman(resid_C_h, resid_gain)`. Also report `Spearman(C_h, size)` and
  `Spearman(gain, size)` descriptively.

## 9. Two-seed replication

Every primary audit is run independently on recurrent seed 0 and seed 1. The
seeds are never pooled before the per-seed report. Report per seed: neighbour
divergence, `P_local`, `P_context`, `context_gain`, `Spearman(C_h, gain)`,
quartile contrast; then a same-sign summary and a rough pooled summary. If the
two seeds conflict the verdict is `SEED_UNSTABLE` and a pooled mean must not
hide it.

Auxiliary strict-static comparison (§12 of the brief), if a zero-cost static
prediction is already available, is descriptive/auxiliary only.

## 10. Pre-registered interpretation (data-driven, no single number)

Per-seed flags (Amendment A1; the pre-amendment rule used the tie-confounded
Jaccard and the relative `context_gain`, and is retained for transparency as
`locally_compilable_strict_preregistered`):

* `DIVERGENCE` — `R_delta^{H0} ≥ 0.50` **or** `retention_dDelta_ratio ≥ 0.60`.
* `LOCAL_INSUFFICIENCY_RELATIVE` — `context_gain ≥ 0.20`.
* `LOCAL_COMPILABLE_ABSOLUTE` — `P_local ≥ 0.50` **and**
  `context_abs_gain < 0.05` **and** `retention_dDelta_ratio ≤ 0.50`.
* `TASK_LINK` — `Spearman(C_h, gain) > 0` **and** the Spearman bootstrap 95% CI
  low `> 0` (stricter than the point-estimate rule; a CI crossing zero is
  “near zero” even if the point estimate is positive).

Cases:

* **Case A — `CONTEXTUALIZATION_BOTTLENECK_SUPPORTED`**: both seeds
  `DIVERGENCE` + `LOCAL_INSUFFICIENCY_RELATIVE` + `TASK_LINK`.
* **Case B — `CONTEXT_EXISTS_BUT_TASK_LINK_UNSUPPORTED`**: both seeds
  `DIVERGENCE`, but `TASK_LINK` fails in both.
* **Case C — `CONTEXTUAL_STATE_LOCALLY_COMPILABLE`**: both seeds
  `LOCAL_COMPILABLE_ABSOLUTE`. Only this authorizes searching for a strict-static compiler.
* **Case D — `INCONCLUSIVE_CONTEXTUALIZATION_AUDIT`**: anything else (seed
  conflict, intermediate `context_abs_gain`, or mutually contradictory metrics).

### Amendment A1 (pre-run measurement robustness)

Discovered in the pre-run CPU dry run: 72% of valid patches have an **exact**
h0 match in train (median 77 exact duplicates), so `Jaccard(NN_h0, NN_h2)` is
not diagnostic (two draws from the same duplicate group overlap little) and
`Amp = d2/(d0+eps)` is degenerate (`d0 = 0`). Amendment A1 therefore: (i) keeps
`J` and the raw amplification as descriptive but excludes them from the
decision; (ii) adds the exact-local-state subset; (iii) uses the matched-random
retention ratios; (iv) adds the absolute `context_abs_gain`; (v) tightens
`TASK_LINK` to require a bootstrap CI that excludes zero. No threshold was
chosen from a seed-0/seed-1 result; the dry run was used only to expose the
measurement degeneracy. Both the original and amended quantities are reported.

A large `||h2-h0||` alone is never evidence. The claim requires the three-way
chain: local unpredictability **+** context-input predictability **+**
task-linked recurrent gain.

## 11. Outputs

`results/zinc_contextualization_gap_audit_v0/`:
`checkpoint_inventory.json`, `export_integrity.json`,
`seed{0,1}_neighbour_divergence.json`, `seed{0,1}_local_predictability.json`,
`seed{0,1}_task_link.json`, `prototype_context_split.json`, `summary.json`,
`RESULTS_SUMMARY.md`, per-patch / per-molecule `.npz` arrays, and figures:
`h0_distance_vs_delta_h_distance.png`, `neighbour_retention_hist.png`,
`contextualization_vs_recurrent_gain_seed0.png`,
`contextualization_vs_recurrent_gain_seed1.png`,
`gain_by_contextualization_quartile.png`, `prototype_context_variance.png`.

## 12. What this round will not do

No new dictionary architecture, entropy rescue, `K` sweep, new ring feature,
new topology cross, new pair algebra, new centre update, static-compiler
training, message-passing training, seed 2, or official test. If the audit
suggests a new hypothesis, it is written as a recommendation only and is **not**
trained this round.
