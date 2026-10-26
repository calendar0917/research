# METHOD — zinc_dcc_information_audit_v1

This is a diagnostic audit of the frozen DCC-L representation, not an
architecture search. The four hypotheses are separated:

- **H-A** algebraic unidentifiability of the pooled local aggregation;
- **H-B** non-redundancy of new joint statistics relative to the complete
  513-d reader input `z` on the real molecule distribution;
- **H-C** a generalizable label-prediction increment from those statistics;
- **H-D** whether any such increment explains low-support / unseen / tail
  errors.

A ⇒ B is not assumed; B ⇒ C is not assumed; C is not model novelty and is not
evidence of reaching the 0.05x target.

## 0. Frozen contract and data access

Frozen before results:

- `AUDIT_PLAN.md` sha256 `ddc6978…` (hypotheses, access protocol, stop rules);
- `configs/audit_v1.json` (statistics, alpha grids, gates, budget);
- `configs/b_phase_freeze.json` (conditional B contract);
- `configs/split_fit8k_probe_audit.json` (label-free split).

Only the official ZINC **train** split (10,000 rows) is used. Official valid is
never loaded for selection/training/gating; official test is never constructed.
A2/A3/A4 are label-free except where explicitly declared. B3 reads ProbeFit
labels to fit diagnostic heads; AuditEval labels are guarded until prediction
hashes are frozen, then read once in B4.

## 1. A1 — forward audit of the frozen DCC-L reader

Checkpoints: D0 (`zinc_dual_channel_composer_v1/runs/L_s0/best.pt`, full-train
DCC-L) and D1 (`zinc_performance_search_v1/runs/B_R0_s1/best.pt`, full-train
B-R0). Both are 494,193-parameter native `DualChannel` models with
`organized_local=True`.

`capture_forward` reproduces `DualChannel.forward` by calling the model's own
encoder, grouping, composer and reader modules in the same order, returning
`a, b, w, Wg, Vg, Wl, Vl, n_b, c_glob, c_loc, z_glob, a_sum, a_sq, g, zl, zl2,
z, pred`. A1 verifies:

1. reader parity with the production forward;
2. `c_loc` uses the real `junc_bond` ownership (2-endpoint incidence);
3. `c_glob` uses graph-level junction means (`n_b * mean(w)`);
4. slice layout and source of every reader block;
5. invariance to atom relabeling, bond permutation, endpoint reversal and batch
   regrouping;
6. no valid/test or cross-split cache usage;
7. non-degeneracy: effective rank / participation ratio, singular values,
   within-graph bond-state variation, constant-channel fraction.

The audit does not generalize low-rank initialization claims to trained
weights; the reported spectra are from the trained checkpoints.

## 2. A2 — algebraic and real-model response audit

### A2.1 strict activation-level counterexample

A fixed 6-bond path line graph with two bond-state assignments that preserve
the multiset (hence `sum(c_loc)` and `sum(c_loc^2)`) but change the
key-state-to-position correspondence. The reader blocks other than `zl/zl2`
are held fixed. An independent float64 implementation verifies equality of
`sum` and `sumsq` and a nonzero change in a correspondence-dependent statistic.
This proves only that a pooled `(sum, sumsq)` reader cannot distinguish these
activations; it is not a real-molecule non-identifiability proof, because real
`c_loc` may encode local correspondence and `global33` exists.

### A2.2 frozen real-model response

200 real train molecules with >=6 bonds, same bond-state multiset, fixed
multiset-preserving rebinding across bond positions. The original pooled
reader input and prediction are checked for invariance; correspondence-
dependent joint statistics are checked for change; real `c_loc` within-graph
variation is reported. This is an information control, not a legal molecule
reconstruction and not a label counterfactual.

## 3. A3 — joint statistic families (pre-fixed)

Only two families are used:

### J_raw (primary 32 dims)

For each of four bond-pair relations:

- R1: line-graph distance 1 (shared atom);
- R2: line-graph distance 2;
- R3: finite line-graph distance >= 3;
- R4: both bonds in a biconnected block with >= 3 edges (same ring-containing
  block).

For each relation, unordered bond-type pair frequencies over the observed
3-type vocabulary `{1,2,3}` (6 pairs), `log1p(pair_count)`, and an empty mask
= 8 dims/relation, 32 total. Frequencies are normalized within relation;
empty relation gives all-zero frequencies and mask 1.

Deviation from the plan's 4-type/48-dim template: the actual train cache has
only bond types `{1,2,3}` (no aromatic type 4). The 4-slot version is computed
only as a compatibility diagnostic and never enters models or gates.

### J_latent (4 dims)

Using the frozen backbone's `c_loc`, within-molecule centering
`h_b = c_loc[b] - mean_b(c_loc)`, then for each relation
`T_r = mean_{(b,c) in R_r} dot(h_b, h_c) / 96`; empty relation → 0.

### Mandatory checks

- independent float64 exhaustive reference (100 molecules);
- graph relabel / bond order / endpoint flip invariance;
- cache vs online consistency;
- empty / single-bond / tree / single-ring / fused-ring / two-component cases;
- before/after training distributions, variance, sparsity;
- correlation and conditional redundancy vs `global33` and `z`;
- null comparisons: bond-type reassignment preserving the histogram (an
  information control, not legal chemistry); latent-row permutation preserving
  the `c_loc` multiset.

## 4. A4 — label-free conditional redundancy

Frozen split: Fit8k / ProbeFit1k / AuditEval1k, structure-only WL content hash
(3 iterations) plus exact isomorphism check on collisions, stratified by
`n_atoms` bins `[≤20,21–24,25–28,≥29]` × `n_ring_systems` bins `[0,1–3,≥4]`,
80/10/10 largest-remainder. Exact duplicate pairs all go to Fit8k, so no
duplicate spans roles.

For each target block (J_raw, J_latent) and feature set (`z`, `global33`,
`z` without the raw global block), fit ridge on Fit8k with 5-fold internal CV
over the fixed alpha grid `[1e-4 … 100]`, standardizing X/Y on Fit8k only.
Score out-of-sample block and per-dimension R² on ProbeFit and AuditEval.
Near-constant dimensions (zero fraction ≥0.99 or std ≤1e-3) are listed and
excluded from the gate; a sensitivity including them is reported. A fixed
small MLP (513→128→out, SiLU, AdamW 1e-3, wd 1e-4, batch 256, max 300 epochs,
patience 20, 1,000-row internal holdout) is a nonlinear sensitivity check.
Residual covariance effective rank is reported on held-out molecules.

**Gate.** If every primary block at every frozen seed has `z→J` block
R² ≥ 0.98 and residual variance share ≤0.02, stop before new GPU work.
Otherwise proceed to B. This is an engineering screen, not a Bayes
sufficiency theorem.

## 5. B — conditional predictive audit

### B1 split

The split is the same frozen label-free Fit8k/ProbeFit/AuditEval file. Fit8k
is used for backbone training and J_hat; ProbeFit for diagnostic-head fitting;
AuditEval only for the one-time evaluation after prediction hashes are frozen.
Old Fit4k/Fit8k/Monitor/Locked results are background only; the new
AuditEval is “this round's internal audit evaluation,” not a claim of freedom
from historical selection bias.

### B2 DCC-L reference

Native `DualChannel` (`organized_local=True`, 494,193 params), Fit8k only,
two supervised seeds. AdamW, lr 1e-3 → 1e-5 cosine, 500-update warm-up,
wd 1e-5, batch 128, raw-y L1, clip 5, dropout 0, exactly 47,400 updates.
Checkpoint is the final update only; no official valid / ProbeFit / AuditEval
selection. Global-33 standardization uses Fit8k rows only. Fit8k-only masked
SSL (same frozen BJOC mechanism, 7,560 updates, batch 128) initializes the
encoders; supervised seed 0 uses SSL seed 0, seed 1 uses SSL seed 1. This is
strictly Fit8k-inductive and does not use ProbeFit/AuditEval structures, even
unlabeled.

### B3 audit heads

Freeze all backbone parameters. On ProbeFit fit three small, equally
regularized residual diagnostics with ridge (fixed alpha grid, 5-fold CV
inside ProbeFit only):

- **Z**: full `z` → `y - f_DCC(z)`;
- **ZJ**: `z` + J_raw + J_latent → same residual;
- **ZN**: `z` + `J_hat` → same residual, where `J_hat` is reconstructed from
  `z` only by ridge fitted on Fit8k (label-free, per backbone seed), matching
  the 36-dim input width of ZJ.

The uncalibrated `f_DCC(z)` is the reference. One fixed
`HuberRegressor(epsilon=1.35, alpha=1e-4)` is a sensitivity check, not a
tuning surface. Per seed: hash checkpoint, freeze ProbeFit diagnostics,
generate AuditEval predictions, hash prediction files, then (only in B4) read
AuditEval labels once and score.

### B4 endpoints

`Gain_J = MAE(Z) - MAE(ZJ)` (primary); `Gain_vs_N = MAE(ZN) - MAE(ZJ)`.
Report both seeds independently; paired per-molecule bootstrap (10,000,
seed 20261026) for fixed-model sample uncertainty (not training-seed
uncertainty); MAE/RMSE/median AE; best-10-worst-row contribution and removal;
both-seed direction; diagnostic-head vs raw DCC gains; ProbeFit-vs-AuditEval
gap; prediction/residual correlations. No test evaluation.

### B5 strata

Support is computed strictly on Fit8k structures: radius-2 typed atom
environment hashes (WL-style 2-iteration subtree hash over atom/bond types).
A molecule is `unseen` if any environment has 0 Fit8k molecules, `low_support`
if any has 1–4, else `full_support`. Diagnostic strata: ordinary `y > -3` vs
tail `y ≤ -3`, size bins, ring-rich (`n_rings ≥ 4`). Cycle-penalty presence is
omitted because no reproducible label-side reconstruction is available in
this round. Label-defined groups are used only after the prediction freeze,
for interpretation, never for training weights/selection/search.

### Decision

The five categories and the A-class thresholds are fixed in
`configs/b_phase_freeze.json` and `AUDIT_PLAN.md`. Diagnostic tags (e.g.
`BENCHMARK_TAIL_DOMINATED`) never override the main class. Only class A would
justify proposing next-round minimal-compute principles; no new architecture
is trained in this round.
