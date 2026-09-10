# Pair endpoint association witness audit (compact-v4-hinge, ZINC)

Status: **NO-GO** (screening). The frozen-feature witness diagnostic finds no
stable signed predictive value from the off-diagonal endpoint association the
current symmetric pair map discards. The endpoint witness is worse than both
the R-only control and the diagonal/current-information control on all 5 outer
folds, so the second frozen backbone seed and the Stage-1b second adapter init
were **not** spent. No backbone was trained, no pair encoder was modified, and
official valid/test were never loaded.

Code: `tracks/ksvd/experiments/luyin16/zinc_pair_endpoint_association_witness.py`
Tests: `tracks/ksvd/tests/test_pair_endpoint_association_witness.py` (10/10 pass)
Results: `tracks/ksvd/results/pair_endpoint_association_witness/`

This answers exactly one question:

> Does the cross-coordinate endpoint association that the current symmetric
> pair parameterization provably discards carry stable, exploitable,
> task-relevant signed predictive value on real ZINC data, beyond the existing
> graph vector R and beyond the diagonal/current-pair information control?

---

## 1. Motivation

The compact-v4-hinge candidate forms a symmetric relation object for every
unordered pair of patch centres and pools it into one fixed 302D graph vector
`R = [unary moments ; pair-bucket moments ; global encoder ; topology hinge]`.
The pair operation is symmetric by construction, which is required, but the
three symmetric endpoint blocks it feeds the pair MLP —

    u_i + u_j,    |u_i - u_j|,    (u_i ⊙ u_j) * gate

— do not in general retain *which* latent coordinates appear together on the
same endpoint. If that discarded cross-coordinate association matters on ZINC,
the study must produce a *task-relevant witness*: a signed predictive value
that the frozen `R` does not already expose. A mathematical non-injectivity is
not, by itself, a model bottleneck. This stage is that witness search.

## 2. Actual pair computation graph

Re-derived from the real code (`zinc_patch_path_pooling.PatchPathModel`, frozen
v4-hinge config), not from a summary:

| quantity | value |
|----------|-------|
| patch state `h_i` | 48D (`patch_encoder` output) |
| pair projection `u_i = P(h_i)` | `Linear(48 → 16, bias=False)` |
| relation descriptor width | 23D |
| relation encoder | `MLPBlock(23 → 32 → 16, dropout 0.05)` |
| distance gate | `Embedding(5 → 16)`, `gate = 1 + tanh(...)` |
| exact-distance feature | `relation[5] = log1p(shortest-path distance)` |
| bucket usage | `bucket = min(max(d,1),5) - 1`, 5 buckets (d = 1,2,3,4,5+) |
| pair input | `cat([u_i+u_j, \|u_i-u_j\|, (u_i⊙u_j)*gate, relation])` = 4×16 = 64D |
| pair encoder | `MLPBlock(64 → 64 → 16, dropout 0.05)` |
| pair state `q_ij` | 16D |
| pair readout | per bucket `[sum ; Σ square ; log1p(count)]` over `q_ij`, 5×33 = 165D |

The relation layout is `[bucket one-hot (5) ; log1p(distance) (1) ; overlap (5)
; boundary/containment (3) ; path bond composition (4) ; log1p(path count) (1)
; adjacent bond one-hot (4)]`. Exact continuous `log1p(distance)` is already
present; the five-bucket pooling is therefore **not** "loss of all exact
distance". `center_context: true`: `q_ij` additionally updates the centre
states before the unary readout. `R` in R^302 = unary 97 + pair moments 165 +
global 32 + topology 8, and is the *input* to `head[0]`.

`u_i` is computed from the **pre-centre-update** patch state — exactly the
tensor that enters pair feature construction (the projection is applied before
`center_update` in `forward`).

## 3. Mathematical non-injectivity

`c_ij = [u_i+u_j, |u_i-u_j|, u_i⊙u_j]` is symmetric under `i↔j` but not
injective on the ordered pair. The canonical counterexample (Test 5):

    u_i = (0,0, 0...)   u_j = (1,1, 0...)     -> c
    u'_i= (0,1, 0...)   u'_j= (1,0, 0...)     -> identical c

Both give sum `(1,1,0)`, abs-difference `(1,1,0)` and product `(0,0,0)`, yet
the symmetric endpoint outer interaction

    A_ij = u_i u_j^T + u_j u_i^T

differs: A_ij is zero in the first case and has off-diagonal entry `A_{01}=1`
in the second. The strictly upper-triangular part (`k < l`) of A_ij is a
120D vector `a_ij`; the diagonal `k = l` equals `2 (u_i ⊙ u_j)` and is already
covered by the third block. So `a_ij` is precisely the discarded
cross-coordinate endpoint association.

## 4. Why algebraic collision is not enough

Continuous latent vectors almost never collide exactly, and the model never
sees `c_ij` alone (it also has the full relation and the graph topology
channel). An algebraic collision therefore only justifies *asking* whether the
discarded part is task-relevant. The final decision here rests entirely on the
predictive witness (Section 10), conditioned on the existing `R`, with a
matched diagonal control.

## 5. Corrected state export extension

New cache version `frozen_state_export_v3_pair_endpoint`
(`results/pair_endpoint_association_witness/state_exports/`). It extends the
already-corrected v2 export (v3 is built from the *same* frozen OOF
checkpoints; nothing was retrained) with two fields:

* `projected_patch_state` — `u_i = P(h_i)` in R^16, one row per patch node;
* `pair_relation` — the raw 23D relation descriptor per pair.

`load_export` refuses any `export_version` other than v3 (Test 9 pins this).
The cache fingerprint includes export version, tokenizer version, config
fingerprint, vocabulary fingerprint, split fingerprint, checkpoint SHA-256,
backbone seed, fold, and the deterministic random-projection fingerprint.

## 6. Integrity gates

`export_integrity_report.json` — **all gates pass on 5/5 folds**:

| gate | meaning | result |
|------|---------|--------|
| G0.1 pair grouping | `n_g(n_g−1)/2` pair rows per graph | PASS |
| G0.2 projected-state correspondence | `u_i` row count == patch count, dim 16 | PASS |
| G0.3 endpoint identity + indexable | source/target graph id equal; global indices in range | PASS |
| G0.4 pair coverage | each unordered pair consumed exactly once, no self-loops | PASS |
| G0.5 reconstruct pair states | `q_ij` rebuilt from `u_i,u_j,relation,bucket` == true `q_ij` (max 0.0) | PASS |
| G0.6 reconstruct R | moments from states ≈ true R (max 1.14e-5) | PASS |
| G0.6b reconstruct prediction | reconstructed R through head ≈ `yhat_0` (max 9.5e-7) | PASS |
| G0.7 node permutation | witness summary max diff 1.24e-7; R 2.9e-6; ŷ 1.2e-7 | PASS |
| G0.8 batch invariance | batch 128 vs 97 + reversed order: `u` mean 0.0; ŷ 4.8e-7 | PASS |

Additionally, inside export build, the export asserts (and records in the
fingerprint) that the exported `u_i` matches the forward pair-projection output
(max 0.0) and that the reconstructed pair encoder input matches the true
forward pair input (max 0.0). Gate-0 failure would have STOPPED the audit.

## 7. Off-diagonal witness definition

For each pair, `a_ij = offdiag_{k<l}(u_i u_j^T + u_j u_i^T)`, in R^120 (no ÷2),
computed from `u_i`, `u_j`. A deterministic random projection `W` in
R^{120×16} is generated once from `PROJECTION_SEED = 20260911` as the
orthonormal columns (QR) of a standard-normal matrix and is **never learned**
(Test 7: identical across calls; frozen for Stage 2). Standardized by the
adapter-fit split (`a_mean`,`a_scale`), projected `z_ij = a_std W` in R^16,
then aggregated per distance bucket into `mean(16) ; std(16)`, giving the
graph summary `W_graph` in R^{160}. Bucket counts are not duplicated (they are
already in `R`).

## 8. R-only and diagonal controls

* **B0** — frozen `yhat_0`, no training.
* **B1 (R-only)** — the historical R-only residual head `R → 13 → 13 → 1`,
  `yhat = yhat_0 + head(R)`, 4,135 params.
* **B2 (diagonal / current-information control)** — same pipeline as E but
  built from `d_ij = u_i ⊙ u_j` (16D, already directly available to the
  current pair encoder): standardize → bucket mean/std → `D_graph` in R^160.
  Head `[R;D_graph] → 9 → 1`, 4,177 params.
* **E (endpoint witness)** — input `[R ; W_graph]`, head `[R;W_graph] → 9 → 1`,
  4,177 params.

B2 and E share an identical architecture and parameter count; E vs B1 differ
by +1.02% (within the pre-registered ±2% budget). Adam(1e-3), full-batch, 400
epochs, best selection checkpoint (patience 50), L1 loss, one deterministic
init. The split manifest is the *same* pre-registered molecule-ID-hash split
as the frozen readout audit: 1200 adapter-fit / 400 adapter-selection / 400
adapter-evaluation per fold.

The primary metrics are

    ΔR    = MAE(B1) - MAE(E)      (endpoint witness value beyond R)
    Δdiag = MAE(B2) - MAE(E)      (value beyond the current-information summary)

## 9. Budget staging

Stage 1 = 1 frozen OOF backbone seed (0) × 5 outer folds × 1 deterministic
adapter init (15 tiny adapters). Stage 1b (second init) only for a borderline
Stage 1; Stage 2 (second frozen backbone seed) only after a clear advance. A
CLEAR NO-GO stops after Stage 1.

## 10. Stage 1 results

Backbone seed 0, adapter init 0. Adapter-evaluation MAE on each fold's 400
molecules:

| fold | B0 | B1 (R-only) | B2 (diag) | E (endpoint) | ΔR | Δdiag |
| ---: | -: | ----------: | --------: | -----------: | -: | ----: |
| 0 | 0.224365 | 0.217165 | 0.227577 | 0.231501 | −0.014335 | −0.003923 |
| 1 | 0.159687 | 0.151446 | 0.156709 | 0.158577 | −0.007131 | −0.001868 |
| 2 | 0.263058 | 0.267327 | 0.272942 | 0.275229 | −0.007902 | −0.002286 |
| 3 | 0.153514 | 0.153222 | 0.153902 | 0.156906 | −0.003685 | −0.003004 |
| 4 | 0.165176 | 0.163337 | 0.162626 | 0.165251 | −0.001914 | −0.002625 |
| mean | 0.193160 | **0.190499** | 0.194751 | 0.197493 | **−0.006994** | **−0.002741** |

* `mean ΔR = −0.00699` (positive in **0/5** folds)
* `mean Δdiag = −0.00274` (non-negative in **0/5** folds)
* `mean B0 − B1 = +0.00266` — the R-only head still helps, as in the previous
  audit; it is the endpoint (and diagonal) summary that hurts.

Stratified molecule-level paired bootstrap over 2000 molecules:

* ΔR mean −0.00699, 95% CI [−0.01039, −0.00359], P(>0) = 0.0001
* Δdiag mean −0.00274, 95% CI [−0.00615, +0.00067], P(>0) = 0.055

The endpoint adapter is worse than B1 on the *selection* split in 5/5 folds
(B1 selection MAE 0.1605/0.1872/0.1669/0.1610/0.1701 vs E
0.1689/0.1956/0.1787/0.1652/0.1815), so the deficit is optimization /
generalization, not evaluation noise. The adapter is non-collapsed
(witness-sensitivity min `max|f(R,W) − f(R,0)| = 0.338`; prediction std ≈ 1.9).

Common-input bulk safety (target-independent rare-patch threshold from the
fit split) fails: E degrades the bulk MAE by mean **+0.0050**, max **+0.0112**
(gate ≤ +0.002).

## 11. Replication decision

Stage 1 satisfies the pre-registered **CLEAR NO-GO**:
`mean ΔR = −0.00699 ≤ +0.0005`, and E beats R-only in `0/5 ≤ 2/5` folds
(`mean Δdiag = −0.00274 ≤ 0` as well). Therefore the second frozen backbone
seed and the Stage-1b second init were **not** spent. Q12 (second-seed
replication) is not applicable.

## 12. Stage 2 if executed

Not executed. No `stage2_fold_results.csv` / `pooled_results.csv` /
`final_bootstrap.json` are produced; Figure 3 is not produced.

## 13. Representation ambiguity analysis

Descriptive and target-free (neighbors defined only by frozen features; no
`y`, residual or improvement used). `c_ij` and `a_ij` are standardized; `c`
neighbors are found within the same distance bucket (version 1) and within a
relation-nearest candidate set (version 2). `k = 16` local neighborhoods.

| measure (mean over folds) | same bucket | relation-restricted |
|---------------------------|------------:|--------------------:|
| median a-dist(c-nearest) / a-dist(random) | 0.317 | 0.495 |
| Pearson(‖Δc‖, ‖Δa‖) | 0.927 | 0.924 |
| fraction “close in c, far in a” | 0.000 | 0.000 |
| normalized conditional Var(a \| kNN c) | 0.252 | — |

Reading: c-space neighbors are ~1/3 as far in a-space as random same-bucket
pairs, and `‖Δc‖` and `‖Δa‖` are almost perfectly monotone — the current pair
map already predicts most of the association. There *is* residual local
ambiguity (exact c-duplicate pairs exist; the conditional a-variance inside a
16-neighborhood is ~25% of total), but it is not the dominant structure, and
Section 10 shows it carries no task-relevant signed signal. The ambiguity
result is **not** a GO gate.

## 14. What is and is not proven

Proven (within this OOF setup):

* The v3 export is measurably correct: exported `u_i` == forward pair
  projection, reconstructed pair input == forward pair input, reconstructed
  `q_ij` == true `q_ij`, and R / prediction reconstruction all pass (Gate 0).
* The off-diagonal endpoint association summary does **not** beat the R-only
  control (ΔR < 0 in 5/5 folds, CI entirely negative) nor the matched
  diagonal/current-information control (Δdiag < 0 in 5/5 folds).
* The representation-level ambiguity is mild-to-moderate: `c` strongly
  predicts `a`, but not perfectly (normalized conditional variance ≈ 0.25).

Not proven:

* That the current pair encoder is *globally* wrong, or that a full outer
  product is optimal.
* That triplet/higher-order interactions are unnecessary in general.
* That a *learned* projection or a full 120D MLP would fail — this audit
  deliberately used a frozen deterministic 16D random projection and does not
  license "try a learned projection" as a rescue.
* That moment pooling is sufficient (see the readout audit's caveat).

Case mapping (mandate §48): this is **Case C — R-only equally good (actually
better)**: `E ≈ B1` fails and `E < B1`; the witness adds no stable value beyond
`R`. (It is also not Case B, because `E < B2`.)

## 15. Final verdict

**NO-GO — off-diagonal endpoint cross-coordinate association.**

Claim (pre-registered NO-GO wording):

> Given the frozen compact-v4-hinge representation, a deterministic
> random-projection summary of the discarded symmetric off-diagonal endpoint
> association provides no reproducible incremental signed predictive value
> beyond the existing graph vector R, and it is worse than a matched
> diagonal/current-pair-information control.

Close the endpoint cross-coordinate association route as the current
bottleneck. Per the mandate, do **not** sweep rank, learn the projection, feed
the full 120D outer product, or reach for attention / triplet networks. The
next question moves upstream in the computation graph (centre-incidence
compression / one-shot centre update / higher-order interaction), again only
after its own task-relevant witness.

---

## Core questions

| # | question | answer |
|---|----------|--------|
| Q1 | real current pair input? | `cat([u_i+u_j, \|u_i-u_j\|, (u_i⊙u_j)*gate, relation_encoder(23→32→16)])`, 64D, from 48D patch state via `Linear(48→16,bias=False)` |
| Q2 | endpoint info mathematically unrecoverable from the current map? | the strictly off-diagonal part of `A_ij = u_i u_j^T + u_j u_i^T` (120D); diagonals are `2(u_i⊙u_j)` and already present |
| Q3 | why does the off-diagonal witness keep unordered-pair symmetry? | `a_ij` is built from the i↔j-symmetric `A_ij`; `_offdiag_outer(l,r)==_offdiag_outer(r,l)` (Test 4) |
| Q4 | real frozen-ZINC endpoint-association ambiguity? | mild-to-moderate: c-nearest a-dist = 0.32× random, Pearson 0.93, normalized conditional Var(a\|c) = 0.25; no “close-c/far-a” pairs |
| Q5 | R-only control MAE? | mean B1 = 0.190499 |
| Q6 | diagonal/current-information control MAE? | mean B2 = 0.194751 |
| Q7 | endpoint witness MAE? | mean E = 0.197493 |
| Q8 | mean ΔR? | **−0.006994** (95% CI [−0.01039, −0.00359], 0/5 folds positive) |
| Q9 | mean Δdiag? | **−0.002741** (0/5 folds non-negative) |
| Q10 | ≥4/5 folds same direction? | no — 0/5 for both |
| Q11 | second backbone needed? | no — Stage 1 clear NO-GO |
| Q12 | second backbone replicate? | not run |
| Q13 | common-input bulk safe? | no — E bulk degradation mean +0.0050, max +0.0112 (> 0.002 gate) |
| Q14 | final verdict? | **NO-GO** |

## Outputs

`results/pair_endpoint_association_witness/`:
`checkpoint_inventory.json`, `export_integrity_report.json`,
`pair_feature_spec.json`, `representation_ambiguity.json`,
`stage_export.json`, `fold_split_manifest.json`, `stage1_fold_results.csv`,
`stage1_bootstrap.json`, `final_decision.json`, `state_exports/*.npz`,
`figures/figure1_representation_ambiguity.png`,
`figures/figure2_per_fold_delta.png`.

Figures are limited to 2 (Figure 3 is Stage-2-only and Stage 2 was not run).
