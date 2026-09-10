# Centre-incidence co-occurrence witness audit (compact-v4-hinge, ZINC)

Status: **NO-GO** (Case B + Case C). The frozen-feature witness diagnostic finds no
task-relevant incremental value in the cross-channel relation co-occurrence that the
current per-centre mean/std compression discards: the covariance witness is **worse than
the matched marginal control** (`mean ΔM = −0.001361`, 2/5 folds positive) and is **worse
than the historical R-only residual head on every fold** (`mean ΔR_hist = −0.011056`,
0/5). The second frozen backbone seed, the second adapter init, and the channel-shuffle
surrogate were **not** spent. No backbone was trained, compact-v4 is unchanged, the pair
encoder is unchanged, there is no second centre update, no attention, no higher-order GNN,
and official valid/test were never loaded.

Code: `tracks/ksvd/experiments/luyin16/zinc_centre_incidence_cooccurrence_witness.py`
Tests: `tracks/ksvd/tests/test_centre_incidence_cooccurrence_witness.py` (12/12 pass)
Results: `tracks/ksvd/results/centre_incidence_cooccurrence_witness/`

This answers exactly one question:

> When compact-v4 compresses the incident pair states of each (centre × distance-bucket)
> cell into per-channel means, standard deviations and a log-count, does the discarded
> **cross-channel relation co-occurrence** carry real signed predictive value on ZINC —
> over and above the existing graph vector `R` and over and above a matched
> marginal-control pathway that re-exposes the centre statistics the model already
> consumes?

---

## 1. Motivation

compact-v4-hinge is not a message-passing network, but it does have exactly one
**centre-level interaction**: after the pair function is evaluated, each patch state is
updated from the pair states incident to it. That update is where "which relations share a
centre" is supposed to enter the representation, and it is the last untested object on the
compute graph that the previous two audits explicitly deferred to.

The predecessor audits closed the two adjacent questions:

* `frozen_conditional_readout_sufficiency_audit.md` — a cheap parameter-matched
  permutation-invariant **set readout** of the frozen pre-pooling states `{h'_i}`, `{q_ij}`
  gave no increment over `R` (NO-GO).
* `pair_endpoint_association_witness_audit.md` — the **off-diagonal endpoint outer-product
  association** discarded by the symmetric pair map gave no increment over `R` and none
  over a diagonal control (NO-GO).

Both of those were about *within-pair* structure. This audit is the first to test
*within-centre* joint structure, which is the one place where the model performs an
explicitly non-linear, order-sensitive-looking aggregation and then immediately throws the
joint information away.

## 2. Actual centre update computation graph

Re-derived from the real code (`zinc_patch_path_pooling.PatchPathModel.forward` and
`_pool_pairs_to_centres`), not from a summary. Frozen v4-hinge config, verified at runtime
from the loaded OOF checkpoints:

| quantity | value |
|---|---|
| patch state `h_i` | 48D (`patch_encoder` output) |
| pair projection | `Linear(48 → 16, bias=False)`, applied to the **pre**-centre-update patch |
| pair input | `cat([u_i+u_j, \|u_i−u_j\|, (u_i⊙u_j)·gate, relation_encoder(23→32→16)])` = 64D |
| pair encoder | `MLPBlock(64 → 64 → 16, dropout 0.05)` |
| pair state `q_ij` | 16D, computed **once**, before the centre update |
| pair unordered? | **yes** — `pair_index` enumerates only `i < j` |
| endpoint contribution | each pair row is duplicated (`cat([source, target])`) and added to **both** endpoint centres |
| bucket | `bucket = min(max(d,1),5) − 1`; 5 buckets (d = 1,2,3,4,5+), from the shortest-path distance |
| centre summary | per bucket: `mean(q)` (16), `std(q)` (16), `log1p(count)` (1) = 33D; 5 × 33 = **165D** |
| std definition | population: `mean = total/count`, `var = count⁻¹Σq² − mean²` clamped ≥ 0, `std = sqrt(var + 1e-8)`, **masked to exactly 0 for empty buckets** |
| empty bucket | mean 0, std 0, log-count 0 (all 33 dims exactly zero) |
| log-count formula | `log1p(count)` with `count` = number of incident pair rows in that bucket |
| centre update input | `[patch (48) ; centre_context (165)]` = **213D** |
| centre update module | `Linear(213→60) → LayerNorm(60) → ReLU → Dropout(0.05) → Linear(60→48)`, final projection zero-initialised |
| residual | **yes**: `patch ← patch + center_update([patch ; context])` |
| after the update | the unary readout is **recomputed** from the updated patch; the pair readout still uses the original `q_ij` |
| pre-head `R` | `[unary moments 97 ; pair-bucket moments 165 ; global 32 ; topology hinge 8]` = 302D |

Two details matter for the audit and are easy to get wrong:

1. `q_ij` fed to the centre pool is the **same tensor** as the one pooled into `R`; the
   centre update does not recompute pairs. So a centre-level witness can only add
   information that is a *joint function of the incident `q_ij` rows*, never new pair
   content.
2. The centre update is **residual with a zero-initialised output layer**, so at
   initialisation it is exactly the identity; whatever it learned is a learned correction.

## 3. What mean/std retains

For a cell `(i,b)` with incident rows `Q_{i,b} = {q_ij}`, the current compression is

    m_{i,b} = [ mean_j q_k ; sqrt(E[q_k²] − E[q_k]²) ; log1p(n) ]  ∈ R^33.

Because `E[q²] = Var(q) + E[q]²`, the summary describes exactly the **first and second
marginal moments per channel** plus the count. Concretely it retains:

* each channel's mean (centre of mass of every relation channel);
* each channel's scale/spread;
* relation mass (how many relations sit in this bucket);
* the exact per-channel first/second moments *as marginals*, and nothing about how they
  were paired.

## 4. What it discards

The current summary is a **per-channel** function of the row set. It is invariant to any
transformation of the rows that preserves each channel's empirical distribution
separately. In particular it discards the **cross-channel co-occurrence**, i.e. whether
channel `k` and channel `l` tend to be active *on the same incident relation*:

* Set A: `q1 = (1,0)`, `q2 = (0,1)` → `mean = (0.5,0.5)`, `std = (0.5,0.5)`,
  `cov(q1,q2) = offdiag = −0.25`;
* Set B: `q1 = (1,1)`, `q2 = (0,0)` → **identical** mean and std **and identical count**,
  but `cov(q1,q2) = offdiag = +0.25`.

Exactly the summary the model computes cannot distinguish these. The full second-order
statistic is the population covariance

    C_{i,b} = E[q qᵀ] − E[q] E[q]ᵀ   (denominator n, not n−1)

with the convention that `C = 0` when `n < 2`. Since the diagonal of `C` is the (squared)
std the model already has, the witness uses only the **strictly upper-triangular part**,

    c_{i,b} = offdiag_{k<l} C_{i,b}   ∈ R^{120},   120 = 16·15/2.

No validity mask is added: the count is already in the summary.

## 5. Why mathematical non-uniqueness is insufficient

The two sets above are a clean algebraic counterexample, and exact marginal collisions
certainly occur in the real frozen states. But the model never sees `m_{i,b}` in isolation:
it sees `m` concatenated across five buckets, mixed by a learned MLP, added residually to a
48D patch state that also already encodes the patch and its chemistry, and finally pooled
upwards into `R`. A collision in the summary is therefore only a licence to *ask* whether
the discarded quantity is **task-relevant**, never a proof that it is a bottleneck. This is
the central discipline of this stage, and the reason the audit ends in a *matched marginal
control* rather than in an architecture.

## 6. Centre export reconstruction

New cache version

    frozen_state_export_v4_centre_incidence      (frozen_state_export_v4_centre_incidence)

built from the **same** frozen OOF checkpoints — nothing was retrained. The previously
corrected v3 export (`u_i`, `q_ij`, pair indices, buckets, raw relation, `R`, `yhat_0`)
was already sufficient to reconstruct the centre incidence assignment, so no new backbone
inference was strictly needed; the v4 export adds the centre-path fields that make the
gates directly checkable:

* `patch_states_pre` — the 48D state entering `center_update` (the pair-projection input);
* `patch_states_post` — `patch + center_update([patch ; centre_context])`;
* `center_context` — the true forward 165D per-centre context.

The fingerprint contains export version, tokenizer version, config fingerprint, vocabulary
fingerprint, split fingerprint, checkpoint SHA-256, backbone seed, fold, and the two
deterministic random-projection fingerprints. `load_export` refuses any other
`export_version`, including all four legacy versions (pinned by a test).

## 7. Integrity gates

`centre_export_integrity_report.json` — **all gates pass on 5/5 folds** (37 gate
instances):

| gate | meaning | result |
|---|---|---|
| G0.1 pair grouping | `n_g(n_g−1)/2` pair rows per graph | PASS (0 mismatches) |
| G0.1b incident coverage | each unordered pair consumed exactly once and contributed to **both** endpoints; no self-loops; `Σ_centres count = 2·n_pairs` | PASS |
| G0.2 bucket identity | exported bucket == relation one-hot argmax == `min(max(round(expm1(log1p d)),1),5)−1` | PASS (0/0 mismatches) |
| G0.3 context reconstruction | reconstructed 165D context == true forward context | PASS, max `3.98e-4`, mean `9.09e-9` |
| G0.4 centre update | `pre + center_update([pre ; reconstructed context]) == post` | PASS, max `5.74e-6` |
| G0.5 R reconstruction | post states + true `q_ij` → pre-head `R` | PASS, max `1.14e-5` |
| G0.6 prediction | reconstructed `R` through `head` == `yhat_0` | PASS, max `9.5e-7` |
| G0.7 row-order invariance | permuting incident pair rows inside each molecule leaves the reconstructed context and the witness summary unchanged | PASS, max `5.4e-19` |
| G0.8 batch invariance | batch 128 vs 97 (reversed order): `R` `0.0`, `yhat_0` `4.8e-7`, centre context `0.0` | PASS |

**On the G0.3 tolerance.** The model builds the context in float32; the offline
reconstruction is float64. The residual is *entirely* in the `std` block of cells with very
few relations and near-zero variance, where `sqrt` amplifies float32 cancellation: the
mean absolute difference is `9.09e-9`, the mean-channel difference is `1.19e-7`, and only
`4.8e-5`–`1.8e-4` of cells exceed `1e-4`. The gate tolerance is `1e-3`; the audit records
these fractions explicitly so the tolerance is auditable rather than hidden. Note that G0.4
(the *functional* gate: the context actually drives the update correctly) passes at
`5.74e-6`, four orders of magnitude tighter, which is the meaningful check.

Gate 0 failure would have STOPPED the audit before any adapter was trained.

## 8. Covariance witness

For each (centre, bucket) cell with `n ≥ 2` incident relations, `c_{i,b} ∈ R^120` is the
strictly upper-triangular population covariance of the incident `q_ij`. Because 120D is far
too much to fit in a diagnostic, the audit uses a **fixed deterministic random projection**

    W_c ∈ R^{120 × 8},   z^C_{i,b} = c_{i,b} W_c   (8D),

generated once as the orthonormal columns (QR) of a standard-normal matrix with
`COV_PROJECTION_SEED = 20260912`, **shared across all centres, buckets and folds, and never
learned**. `c` is standardized per channel by adapter-fit statistics before projection.

The graph-level summary is then, per bucket, the mean and std of `z^C` across the
molecule's centres (cells with `n < 2` contribute exactly `z = 0`):

    C_graph ∈ R^{5 × 2 × 8} = R^{80}.

This is permutation-invariant, has no trainable parameters, is cheap, and is explicitly a
**diagnostic** — it is not a re-implementation of the centre update.

## 9. Marginal matched control

This is the most important control of the round. Without it, a positive
`R + something_centre_shaped` result could simply mean "the adapter was handed a second
route to centre-level information the model already has".

`M_graph` is built with the **identical** machinery — same projection dimension, same
per-bucket mean/std across centres, same adapter interface, same 80D width — but from

    m_{i,b} = [mean(q) ; std(q) ; log1p(count)] ∈ R^33

read **straight from the true forward centre context**, i.e. exactly the information the
model's own centre compression already exposes. Its projection `W_m ∈ R^{33 × 8}`
(`MARG_PROJECTION_SEED = 20260913`) is orthonormal and equally never learned.

Because B2 and E are identical in **architecture, width, parameter count and depth**, the
only difference between them is *whether the adapter sees the discarded cross-channel
structure or a re-exposure of the retained marginals*. That makes `ΔM` the decisive
quantity.

## 10. Budget policy

Stage 0 = measurement / cache correctness only. Stage 1 = 1 frozen OOF backbone seed ×
5 outer folds × 1 deterministic adapter init. Stage 1b (second init) only for a borderline
Stage 1. Stage 2 (second frozen backbone seed) only after a clear advance. A clear
NO-GO stops immediately. The current run is a **Stage-1 CLEAR NO-GO**, so 10 of the 15
pre-registered adapters were trained (B1, B1_hist, B2, E on five folds) and nothing else
was spent.

## 11. Stage 1

Backbone seed 0, adapter init 0. Reused, unmodified, the same pre-registered
target-independent molecule-ID-hash split as the frozen-readout audit (1200 adapter-fit /
400 adapter-selection / 400 adapter-evaluation per fold). Adapter-evaluation MAE on each
fold's 400 molecules:

| fold | B0 | B1 (R-only) | B1_hist (R-only, historical head) | B2 (marginal) | E (covariance) | ΔR | ΔR_hist | ΔM |
| ---: | -: | ----------: | --------------------------------: | ------------: | -------------: | -: | ------: | -: |
| 0 | 0.224365 | 0.233584 | 0.217165 | 0.225145 | 0.231869 | +0.001716 | −0.014704 | −0.006724 |
| 1 | 0.159687 | 0.166880 | 0.151446 | 0.160857 | 0.157774 | +0.009107 | −0.006328 | +0.003083 |
| 2 | 0.263058 | 0.276178 | 0.267327 | 0.277908 | 0.282005 | −0.005827 | −0.014678 | −0.004098 |
| 3 | 0.153514 | 0.167095 | 0.153222 | 0.161404 | 0.159732 | +0.007363 | −0.006510 | +0.001672 |
| 4 | 0.165176 | 0.168647 | 0.163337 | 0.175658 | 0.176397 | −0.007750 | −0.013060 | −0.000739 |
| mean | **0.193160** | 0.202477 | **0.190499** | 0.200194 | 0.201555 | **+0.000922** | **−0.011056** | **−0.001361** |

* `mean ΔR = +0.00092` (95% CI `[−0.00447, +0.00619]`, `P(>0)=0.64`) — positive in 3/5
  folds but far below both the `+0.002` advance gate and even the `+0.0005` clear-NO-GO
  line, with a confidence interval straddling zero.
* `mean ΔM = −0.00136` (95% CI `[−0.00569, +0.00279]`, `P(>0)=0.27`) — positive in **2/5**
  folds. **The covariance witness is worse than the marginal control whose only content is
  information the model already has.**
* `mean ΔR_hist = −0.01106` (95% CI `[−0.01610, −0.00612]`, `P(>0)=0.00`) — the covariance
  witness is worse than the historical R-only head on **0/5** folds.
* The covariance adapter is worse than the marginal control on the *selection* split in
  **5/5** folds, so the deficit is generalization, not a single evaluation draw.
* Both witness adapters are non-collapsed (witness sensitivity min `0.601`, prediction
  std `1.84`), so this is not a dead branch.
* Common-input bulk (target-independent rare-patch threshold from the fit split): E
  degrades the bulk relative to B0 by mean `+0.0111`, max `+0.0191` (gate ≤ +0.002, FAIL).
  The pre-registered covariance-vs-marginal bulk check **also fails** on its strict
  per-fold form (max `+0.0077` > 0.002; fold-wise `+0.0077 / −0.0019 / +0.0053 /
  −0.0001 / −0.0028`). Relative to the matched-budget R-only head the bulk is safe
  (mean `−0.0001`, max `+0.0058`). So the covariance pathway is not merely uninformative
  beyond the marginals — it is **slightly harmful on the common-input bulk** relative to
  its own depth-matched control.

**Depth confound, measured not hidden.** The pre-registered matched-budget 1-layer R-only
head (`302→14→1`, 4,257 params) underperforms B0 here, unlike the historical 2-layer head.
A fixed, non-swept depth sensitivity (`302→14→1` vs `302→13→13→1`, 5/5 folds,
`adapter_depth_sensitivity.json`) measures the extra hidden layer at **+0.01198 mean MAE**, the
same order as the entire `ΔM` scale. The historical-head comparison is therefore
**depth-confounded** and is reported as robustness evidence only. Critically, the primary
mechanism test (E vs B2) is both depth-matched and parameter-matched: E and B2 are the same
module (`382→11→1`, 4,225 params) differing only in their input contents, and B1 differs
from both by +0.75% (within the pre-registered ±2%). The confound therefore cuts *against*
the negative conclusion, which makes the NO-GO conservative.

**Why the frozen pair states are sparse, and what that means.** The pair encoder ends in
ReLU, so `q_ij` has 86–97% exactly-zero entries in the frozen checkpoints (independently
reproduced from the predecessor's v3 export, and recorded under `pair_state_sparsity` in the
integrity report). Consequently only 3–29% of `n ≥ 2` cells have a nonzero covariance row.
This is a property of the *frozen representation*: within a distance bucket, an incident
relation set is usually a sparse, low-rank code, and the cross-channel co-occurrence it can
express is correspondingly limited. It also explains the pooled ambiguity numbers, since
matching all-zero marginals also match their (zero) covariance.

## 12. Replication decision

Stage 1 meets the pre-registered **CLEAR NO-GO** conditions: `mean ΔM = −0.00136 ≤ 0`,
`mean ΔR = +0.00092 ≤ +0.0005` fails as well, and E beats the marginal control in only
`2/5 ≤ 2/5` folds. Therefore:

* the second frozen OOF backbone seed was **NOT** spent;
* the Stage-1b second adapter init was **NOT** spent;
* the optional channel-shuffle surrogate control was **NOT** run (it is gated on E clearly
  beating B2, which did not happen).

Q13 (second backbone needed?) is therefore answered **no**.

## 13. Stage 2 if run

Not executed. No `stage2_fold_results.csv`, `pooled_results.csv`, `final_bootstrap.json`.
Figure 3 is not produced.

## 14. Mechanism control if run

Not executed. `channel_shuffle_control.csv` is not produced; the pre-registered trigger
("only after E clearly beats B2") was not met.

## 15. Representation ambiguity

Descriptive and target-free — neighbours are defined only by frozen features; no `y`,
residual, or improvement enters. `m` and `c` are standardized; `k = 16`; the pool keeps
only informative cells (count ≥ 2 **and** nonzero covariance row) so that artificial
all-zero coincidences do not inflate the ambiguity.

| measure (mean over 5 folds) | value |
|-----------------------------|------:|
| median `‖Δc‖` for `m`-nearest neighbours / median `‖Δc‖` for random cells | 0.282 |
| `Pearson(‖Δm‖, ‖Δc‖)` | 0.596 |
| `Spearman(‖Δm‖, ‖Δc‖)` | 0.684 |
| fraction "close in `m`, far in `c`" | 0.017 |
| normalized conditional `Var(c | 16-NN m)` | 0.285 |

Reading: near-identical centre marginals still allow appreciably different covariance
(`m`-space neighbours are ~28% as far in covariance space as random cells, but the residual
conditional variance inside a 16-neighbourhood is ~29% of the total). So **the discarded
quantity is genuinely not determined by the retained summary** — the suspected information
loss is real. It is simply **not task-relevant**, which is exactly the distinction this
audit was designed to make. Per the mandate, this result is **not** a performance gate and
does not reverse the NO-GO.

## 16. What is and is not proven

Proven (within this frozen OOF setup):

* The v4 export is measurably correct: 37/37 Gate-0 instances pass on 5/5 folds, including
  exact incidence coverage to both endpoints, bucket identity, 165D context reconstruction,
  centre-update reconstruction, `R` reconstruction, prediction reconstruction, incident-row
  order invariance, and batch invariance.
* The centre-incidence covariance summary does **not** beat the matched marginal control
  (`ΔM < 0`, CI straddling zero with `P(>0)=0.27`, 2/5 folds), and does not beat the
  historical R-only residual head (0/5 folds).
* There **is** real, measurable representational ambiguity: the retained marginal summary
  does not determine the discarded covariance (normalized conditional variance ≈ 0.29).
* A mathematically discarded statistic is therefore demonstrably **not** a task-relevant
  bottleneck for this representation — the central scientific point of the round.

Not proven:

* That moment pooling is *sufficient* in general. This is a low-capacity, projection-based
  witness on a single frozen backbone, not an architecture result.
* That a fully learned or larger covariance pathway would fail. This audit deliberately did
  not search projection size, rank, or capacity, and it explicitly does **not** license
  "learn the projection" or "use the full 120D covariance" as a rescue.
* That the centre update is globally well-parameterized, or that a *one-shot* centre update
  is the right object. Those are different questions with different witnesses.
* That higher-order (triplet / triangle) structure is unnecessary in general.

Case mapping (mandate §54): this is **Case B — centre marginal re-access only**
(`E ≤ B2`: the covariance witness does not beat the matched marginal control) **plus
Case-C evidence** (`E ≤ B1_hist` on 5/5 folds). Both map to the same action, so the
conclusion does not depend on which label one prefers.

## 17. Final verdict

**NO-GO — within-centre cross-channel relation co-occurrence is not a task-relevant
bottleneck of compact-v4.**

Claim (pre-registered NO-GO wording):

> Given the frozen compact-v4-hinge representation, a deterministic random-projection
> summary of the cross-channel co-occurrence of incident pair states provides no
> reproducible incremental signed predictive value beyond the existing graph vector `R`
> and, decisively, none beyond a parameter-matched, depth-matched marginal-control pathway
> built from the centre statistics the model already consumes.

Action:

* **Do not** build a covariance / joint-relation / centre-incidence architecture.
* **Do not** test full covariance, a larger or learned covariance projection, attention
  over relations, DeepSets centre encoders, a second centre update, relation-set
  Transformers, or triplet/motif networks on this evidence.
* **Do not** re-open this as "another missing statistic along the compute graph". If this
  witness is negative, the right next move is a **paradigm-level** question (information
  accessibility / learnable parameterization / one-shot centre update as a *hypothesis about
  function class*, not as another pooled statistic), not a third covariance.

---

## Core questions

| # | question | answer |
|---|----------|--------|
| Q1 | How is the centre incidence aggregation really implemented? | `_pool_pairs_to_centres`: for each of 5 buckets, `index_add_` of the `q_ij` rows (and their squares, and a count) onto centre indices, then `mean`, `sqrt(clamp(var,0)+1e-8)·(count>0)`, `log1p(count)`; 5 × 33 = 165D |
| Q2 | How does each pair contribute to the two endpoint centres? | each unordered pair row is duplicated (`cat([source, target])`) and added identically to both endpoints — exactly once to each |
| Q3 | What does the 165D context contain? | per bucket: 16 per-channel means, 16 per-channel population stds, 1 `log1p(count)` |
| Q4 | What joint relation information does it explicitly not contain? | the cross-channel co-occurrence — the off-diagonal part of the incident covariance `C_{i,b} = E[qqᵀ] − E[q]E[q]ᵀ` (120D); any two relation sets with identical per-channel marginals and count are indistinguishable to it |
| Q5 | Is there real marginal→covariance ambiguity in the frozen ZINC states? | **yes, mild-to-moderate**: `m`-nearest neighbours are 0.282× as far in covariance space as random cells; `Pearson = 0.596`, `Spearman = 0.684`; normalized conditional `Var(c \| 16-NN m) = 0.285`. The loss is real but is **not** a performance gate |
| Q6 | B0 MAE? | **0.193160** (frozen `yhat_0`, no training) |
| Q7 | R-only MAE? | **0.202477** (pre-registered matched head `302→14→1`, 4,257 params); the historical 2-layer head gives **0.190499** (4,135 params, depth-confounded) |
| Q8 | Marginal-control MAE? | **0.200194** (`382→11→1`, 4,225 params) |
| Q9 | Covariance witness MAE? | **0.201555** (identical architecture to B2, 4,225 params) |
| Q10 | mean ΔR? | **+0.000922** (95% CI `[−0.00447, +0.00619]`, `P(>0)=0.64`, 3/5 folds). Against the historical R-only head: **−0.011056** (CI `[−0.01610, −0.00612]`, 0/5) |
| Q11 | mean ΔM? | **−0.001361** (95% CI `[−0.00569, +0.00279]`, `P(>0)=0.27`, **2/5** folds positive) — the decisive negative |
| Q12 | Fold direction consistent? | No — `ΔR` and `ΔM` flip sign across folds, and both pooled means sit at or below the clear-NO-GO thresholds |
| Q13 | Second backbone needed? | **No** — Stage 1 is a pre-registered CLEAR NO-GO; seed 1 and the Stage-1b init were not spent |
| Q14 | Common-input bulk safe? | **No.** E degrades the bulk versus B0 (mean `+0.0111`, max `+0.0191`, gate ≤ 0.002, FAIL) and the pre-registered covariance-**vs-marginal** check also FAILS on its strict per-fold form (max `+0.0077` > 0.002). Versus the matched-budget R-only head the bulk is safe (mean `−0.0001`, max `+0.0058`) |
| Q15 | Final verdict? | **NO-GO** (Case B + Case C) |

## Outputs

`results/centre_incidence_cooccurrence_witness/`:
`checkpoint_inventory.json`, `centre_export_integrity_report.json`,
`centre_feature_spec.json`, `representation_ambiguity.json`,
`adapter_depth_sensitivity.json`, `stage_export.json`, `fold_split_manifest.json`,
`stage1_fold_results.csv`, `stage1_bootstrap.json`, `final_decision.json`,
`state_exports/*.npz`, `figures/figure1_representation_ambiguity.png`,
`figures/figure2_per_fold_delta.png`.

Stage-2 files (`stage2_fold_results.csv`, `pooled_results.csv`, `final_bootstrap.json`) and
the surrogate file (`channel_shuffle_control.csv`) are intentionally absent. Figures are
limited to 2 (Figure 3 is Stage-2/surrogate-only and neither ran).
