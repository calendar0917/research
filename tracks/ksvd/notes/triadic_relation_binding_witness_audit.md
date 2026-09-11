# Explicit triadic relation binding witness audit (compact-v4-hinge, ZINC)

Status: **NO-GO** (Case C). The frozen-feature witness diagnostic finds **no**
task-relevant incremental value in explicitly binding the three pair relations of a
patch triple `(q_ij, q_ik, q_jk)`: the true triad witness is **worse than the
parameter-matched, depth-matched unbound control** (`mean ΔU = MAE(B2) − MAE(E) =
−0.001849`, 95% CI `[−0.00294, −0.00079]`, `P(>0)=0.0003`, 0/5 folds) and **worse than
the R-only residual head** (`mean ΔR = MAE(B1) − MAE(E) = −0.001653`, 1/5 folds). The
second frozen backbone seed, the Stage-1b second adapter init, and the mechanism
replacement control were **not** spent. No backbone was trained, compact-v4 is
unchanged, no pair state was recomputed, no patch state was updated, there is no second
round of propagation, no attention, and official valid/test were never loaded.

Code: `tracks/ksvd/experiments/luyin16/zinc_triadic_relation_binding_witness.py`
Tests: `tracks/ksvd/tests/test_triadic_relation_binding_witness.py` (13/13 pass)
Results: `tracks/ksvd/results/triadic_relation_binding_witness/`

This answers exactly one question:

> compact-v4 already represents every pair relation individually; does the **absence of
> an explicit binding that `q_ij`, `q_ik`, `q_jk` belong to the same triple `(i,j,k)`**
> hide task-relevant higher-order structure?

---

## 1. Motivation

Four consecutive "missing local statistic" routes had already failed on this frozen
representation: frozen richer readout, pair endpoint association, centre-incidence
covariance/co-occurrence, and a coordinatewise function basis. Each asked whether some
*discarded scalar/second-moment summary* was task-relevant. This audit asks a
qualitatively different structural question:

> The model knows the pairs individually, but does it know that several pairs belong to
> the same higher-order configuration?

This is a **binding** question, not a marginal-statistic question. It is the last
untested object on the current compute graph that the previous audits explicitly
deferred, and it is the natural next rung of the project's original ladder:

```
explicit local patches  →  explicit structural relations  →  explicit composition
```

## 2. Method-philosophy boundary

The project philosophy is

```
explicit local patches + explicit structural relations + explicit permutation-invariant composition
```

**not**

```
h^0 → q^1 → h^1 → q^2 → h^2 → ...
```

This audit therefore forbids and does not implement: a second round of state
propagation, recomputation of pair states, patch-state updates, a learned backbone,
GNN/Transformer/attention, new molecular descriptors, tokenizer changes, or topology
branch changes. The tested object is an **explicit higher-order relation composition**,
not an implicit propagation depth.

## 3. Why iterative refinement was not chosen

Message passing forms higher-order dependence implicitly by propagating hidden state.
This experiment instead defines an explicit structural object — the three-patch
relation tuple `T_ijk = (q_ij, q_ik, q_jk)` — and asks whether that explicit object
carries predictive value beyond its individual/marginal statistics. This stays on the
"patch / relation / dictionary / explicit composition" line and avoids the confounds of
depth, recurrence and per-node state.

## 4. Actual compact-v4 relation computation (re-derived from real code)

Re-read from `zinc_patch_path_pooling.PatchPathModel.forward`, not from a summary. Frozen
v4-hinge shapes verified at runtime from the loaded OOF checkpoints (the same ones used
by the predecessor audits):

| quantity | value |
|---|---|
| patch state `h_i` | 48D |
| pair projection | `Linear(48 → 16, bias=False)` on the **pre**-centre-update patch |
| pair input | `cat([u_i+u_j, \|u_i−u_j\|, (u_i⊙u_j)·gate, relation_encoder(23→32→16)])` = 64D |
| pair encoder | `MLPBlock(64 → 64 → 16)` |
| pair state `q_ij` | 16D, computed **once**, before the centre update |
| pair unordered? | **yes** — `pair_index` enumerates only `i < j`; each row added to both endpoints |
| bucket | `bucket = min(max(d,1),5) − 1`, 5 buckets |
| centre summary | per bucket `[mean(q) 16 ; population std(q) 16 ; log1p(count) 1]` = 33D; ×5 = **165D** |
| centre update | `patch ← patch + MLP([patch ; 165D])`, final projection zero-init (residual) |
| after update | unary readout recomputed; pair readout still uses the original `q_ij` |
| pre-head `R` | `unary 97 + pair-bucket moments 165 + global 32 + topology 8 = 302D` |

### Q1 — does the model already explicitly bind `q_ij, q_ik, q_jk`?

**No.** The only joint relation operation is the per-centre pool, which binds the pairs
**incident to one centre** (`q_ij` and `q_ik` both flow into centre `i`'s 165D context,
alongside every other incident relation), and the one-shot centre update. The outer
relation `q_jk` is never part of centre `i`'s context; it enters centre `j`'s and centre
`k`'s contexts. There is no tensor, no product, and no pooled summary in which the three
relations of a triple appear together. The three pair states of a triple are therefore
only *implicitly* coupled through the shared updated patch states
(`patch_i' = f(h_i, {q_ij}_j)`, `patch_j' = f(h_j, {q_ij, q_jk})`,
`patch_k' = f(h_k, {q_ik, q_jk})` and the subsequent per-patch moment pooling), never
explicitly bound.

## 5. What pair/centre pooling retains

`R` retains (i) per-patch unary moments, (ii) per-pair-bucket moments of `q_ij`
(first/second moments of each channel), (iii) graph-level globals, and (iv) topology. A
triple-level object appears nowhere. Two molecules whose triples are re-paired but whose
per-pair marginals are unchanged are indistinguishable to the current summary.

## 6. What explicit triadic binding means

For each unordered patch triple `i < j < k`, with

```
a = q_ij,  b = q_ik,  c = q_jk ∈ R^16,
```

the audit forms

```
s1    = a + b + c                                  (16D)
s_abs = |a−μ| + |b−μ| + |c−μ|,  μ = (a+b+c)/3      (16D)
s2    = a⊙b + a⊙c + b⊙c                            (16D)
triad_raw = [s1 ; s_abs ; s2]                      (48D)
```

which is permutation-invariant in `(a,b,c)` and has **no trainable parameters**. To make
invariance exact (including floating-point addition order), the three pair states are put
in a canonical **value-lexicographic** order before the reductions. The fixed
deterministic orthonormal projection `W_T ∈ R^{48×16}` (QR of a seeded standard-normal
matrix, seed `20260914`) gives `z = std(triad_raw) W_T`, and the graph summary is

```
T_graph = [mean(z) ; std(z)] ∈ R^32.
```

No triad count is added (graph size is already inside `R`).

## 7. Difference from centre covariance

The centre-covariance audit asked whether the incident relations **of one centre**,
`{q_ij}_j`, co-vary across channels. This audit asks whether the two centre relations of
a triple (`q_ij, q_ik`, sharing endpoint `i`) close with the **outer** relation `q_jk`,
which is *not incident to `i`*. The outer relation is precisely what centre covariance
never binds to the inner pair. This is a genuine closure/binding question, not a repeat
of covariance.

## 8. Frozen triad construction

All triads come from the existing frozen v4 export
(`frozen_state_export_v4_centre_incidence`), which already contains `q_ij` (`pair_states`),
local pair endpoints, buckets, `R`, `yhat_0`, molecule ids and targets. **No backbone
inference and no pair recomputation** were performed. Pair lookup is by a dense
`(n × n)` table built from the exported local endpoints, so each triple `(i,j,k)`
retrieves exactly its three pairs `(i,j)`, `(i,k)`, `(j,k)`.

## 9. Invariance and integrity gates

`triad_integrity_report.json` — **all gates pass on 5/5 folds (22 gate instances,
seed 0)**:

| gate | meaning | result |
|---|---|---|
| T0.1 triple count | exported triples == `Σ C(n,3)` | PASS (`4,037,863 / 4,087,118 / 4,034,692 / 4,114,144 / 4,079,685` per fold) |
| T0.2 pair completeness | `Σ n_pairs == Σ C(n,2)`, no self-loops, no duplicates, no missing pairs | PASS |
| T0.3 pair lookup integrity | each triple's three `q` equal the direct pair gathers; endpoint order irrelevant | PASS, max diff `0.0` |
| T0.4 triad invariance | `triad_raw` is exactly invariant under all 6 pair-slot permutations | PASS, max diff `0.0` |
| T0.5 projection determinism | `W_T` identical across calls, orthonormal columns | PASS, max diff `0.0`, Gram off-diag `7.8e-16` |
| T0.6 parameter match | B1 vs B2/E within ±3% | PASS, `4135` vs `4189` (+1.31%) |

**Cost.** Full `O(n^3)` enumeration was used (no cap). Per fold, fold 0 has `4.04M`
triads, median `n = 23`, max `n = 36`; a deterministic 100-molecule profile measured
`0.0042 s/molecule` for the true **and** unbound build, projecting to `≈8.3 s` for a
full 2000-molecule fold — far below the pre-registered budget. `sampling_manifest.json`
therefore records `full_enumeration = true`, `sampling_applied = false`.

## 10. Unbound matched control

This is the decisive mechanism control. For every molecule the outer relation `q_jk` is
deterministically permuted across the triples while the two centre relations `q_ij`,
`q_ik` are kept. The permutation is a product of single cycles on same-outer-bucket
groups (seeded by a SHA-256 hash of the molecule id and a fixed seed `20260914`, so it is
target-independent and reproducible); leftover singleton bucket groups are pooled and
permuted, and an isolated leftover becomes a recorded fixed point.

Control quality (`unbound_control_manifest.json`, all triads):

| fold | bucket mismatch | fixed-point rate | same outer row | breakage |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.0 | `4.95e-7` | 0.0258 | 0.9742 |
| 1 | 0.0 | 0.0 | 0.0258 | 0.9742 |
| 2 | 0.0 | 0.0 | 0.0260 | 0.9740 |
| 3 | 0.0 | 0.0 | 0.0258 | 0.9742 |
| 4 | 0.0 | `2.45e-7` | 0.0257 | 0.9743 |

The pair-state **multiset is exactly preserved** (the outer slots are a permutation of
the true outer slots; the inner slots are untouched), the outer distance-bucket
distribution is exactly preserved, and 97.4% of triads are genuinely rebound.

### Q2 — why centre covariance is not equivalent to triadic binding

Centre covariance pools incident pairs **within one centre**; the outer relation is not
incident to the triple's reference centre, so no centre-level statistic can bind
`q_jk` to `{q_ij, q_ik}`. The two audits examine different structural objects.

### Q3 — why this is not message passing

The triad is an explicit, deterministic, parameter-free function of frozen pair states.
No hidden state is propagated, no node state is updated, there is no second round and no
learned depth.

## 11. Compute-budget policy

Staged 1 → 2 → candidate: Stage 0 = triad reconstruction/integrity only; Stage 1 =
seed 0 × 5 folds × 1 adapter init; Stage 1b (second init) only for a borderline Stage 1;
Stage 2 (second frozen backbone seed) only after a clear advance. A clear NO-GO stops
immediately. This run is a **Stage-1 CLEAR NO-GO**, so only the 15 Stage-1 adapters
(B1, B2, E × 5 folds) were trained.

Pre-registered adapters (no sweeps):

| reader | architecture | params |
|---|---|---:|
| B1 R-only | `302 → 13 → 13 → 1` | 4135 |
| B2 unbound control | `(302+32) → 12 → 12 → 1` | 4189 |
| E true triad | `(302+32) → 12 → 12 → 1` | 4189 |

B2 and E are **architecture-identical** and differ only in whether the 32D input is the
true triad summary or the unbound one. One shared witness standardizer (computed from
**true** fit-split triads) is applied to both. All readers use the same L1/Adam
(`lr=1e-3`), full batch, fixed horizon 400, selection-checkpoint protocol, and the same
official-train-only 1200/400/400 molecule-ID-hash split; `assignment_sha256` matches the
predecessor manifests exactly on 5/5 folds.

## 12. Stage 1 results

Backbone seed 0, adapter init 0. Adapter-evaluation MAE on each fold's 400 molecules:

| fold | B0 | B1 (R-only) | B2 (unbound) | E (true) | ΔR | ΔU |
| ---: | -: | ----------: | -----------: | -------: | -: | -: |
| 0 | 0.224365 | 0.217165 | 0.220176 | 0.220625 | −0.003460 | −0.000449 |
| 1 | 0.159687 | 0.151446 | 0.150183 | 0.152604 | −0.001158 | −0.002420 |
| 2 | 0.263058 | 0.267327 | 0.267331 | 0.270879 | −0.003552 | −0.003548 |
| 3 | 0.153514 | 0.153222 | 0.154857 | 0.156476 | −0.003255 | −0.001619 |
| 4 | 0.165176 | 0.163337 | 0.158966 | 0.160176 | +0.003161 | −0.001210 |
| mean | **0.193160** | **0.190499** | **0.190303** | 0.192152 | **−0.001653** | **−0.001849** |

* `mean ΔR = −0.001653`, positive in only **1/5** folds. Against the R-only head the
  triad witness loses.
* `mean ΔU = −0.001849` (95% CI `[−0.002941, −0.000791]`, `P(>0)=0.0003`), positive in
  **0/5** folds. **The true triad binding is worse than the matched unbound control**,
  whose only content is the same pair-state multiset arranged with broken closure. The
  loss is therefore not "extra capacity"; the binding itself is unhelpful.
* The adapters are **not collapsed**: true-witness sensitivity min `0.157`, prediction
  std `≈1.8`; the true/unbound summaries are non-degenerate and differ (median
  `‖T_graph − U_graph‖ = 0.254`, per-channel correlation `0.980`).
* Common-input bulk (target-independent rare-patch threshold from the fit split): E
  degrades the bulk relative to B2 by max `+0.003416` (gate ≤ +0.002, FAIL) and relative
  to B1 by max `+0.005125`. Relative to B0 the bulk is safe (mean `−0.0015`).

## 13. Replication decision

Stage 1 meets the pre-registered **CLEAR NO-GO** conditions (`mean ΔR ≤ +0.0005`,
`mean ΔU ≤ 0`, and E beats unbound in `0/5 ≤ 2/5` folds). Therefore:

* the second frozen OOF backbone seed was **NOT** spent;
* the Stage-1b second adapter init was **NOT** spent;
* the optional mechanism replacement control was **NOT** run.

## 14. Stage 2 if run

Not executed. No `stage2_fold_results.csv`, `pooled_results.csv`, `final_bootstrap.json`.

## 15. What is and is not proven

**Proven (within this frozen OOF setup):**

* The explicit triad construction is correct: `Σ C(n,3)` triads, exact pair lookup,
  exact permutation invariance, deterministic projection, matched parameter budgets
  (22/22 gate instances, 5/5 folds).
* The frozen pair states were never recomputed and no backbone/state was modified.
* The unbound control preserves the pair-state multiset and the distance-bucket
  distribution while breaking 97.4% of true triple closures.
* The true triad witness does **not** beat the unbound matched control (`ΔU < 0`, CI
  entirely below zero, 0/5 folds) and does not beat the R-only head (`ΔR < 0`, 1/5).
* The adapted readers are non-collapsed, so the negative is a genuine measurement.

**Not proven:**

* That higher-order structure is unnecessary in general. This is a low-capacity,
  projection-based witness on a single frozen backbone, with `q_ij` produced by a
  ReLU-bounded pair encoder.
* That a *learned* or *larger* triadic composition would fail. The audit deliberately
  did not search projection size, rank or capacity, and explicitly does **not** license
  those as a rescue.
* That the one-shot centre update is globally well-parameterized, or that the pair
  representation is sufficient.
* That message passing / attention is the right next step. The opposite: it is not
  justified by this evidence.

## 16. Implications for explicit compositional modeling

The explicit ladder patch → pair → triple does **not** pay off at this rung under this
frozen representation. Combined with the four predecessor NO-GOs, there is now no
task-relevant witness for any additional *local/higher-order relation statistic* on the
current compact-v4 compute graph. Per the mandate, the correct action is to stop
adding features to the current architecture and escalate to a paradigm-level question
(representation family / dictionary construction / objective-aligned structured model),
not to reach for attention, a triplet Transformer, a larger triad vector, or message
passing by default.

## 17. Final verdict

**NO-GO — explicit triadic relation binding is not a task-relevant bottleneck of
compact-v4-hinge on ZINC.**

Claim (pre-registered wording):

> Given the frozen compact-v4-hinge representation, an explicit permutation-invariant
> triad binding witness over `(q_ij, q_ik, q_jk)` provides no reproducible incremental
> signed predictive value beyond the existing graph vector `R` and, decisively, none
> beyond a parameter-matched, depth-matched unbound control that carries the same
> pair-state marginals while destroying true triple closure.

Action:

* **Do not** design a triadic composition module (no low-rank triadic encoder, no
  triangle attention, no triplet Transformer, no higher-order motif sweep).
* **Do not** re-open this as "another missing relation statistic".
* The next decision is **representation-family level**, with its own pre-registered
  witness: `representation family / dictionary construction / objective-aligned
  structured model`.

---

## Core questions Q1–Q16

| # | question | answer |
|---|----------|--------|
| Q1 | Does the current model explicitly bind `q_ij, q_ik, q_jk`? | **No.** The only joint operation is per-centre pooling of pairs sharing a centre; `q_jk` never enters centre `i`'s context. No triple tensor/product/pool exists. |
| Q2 | Why is centre covariance not equivalent to triadic binding? | Centre covariance pools incident pairs **within one centre**; the outer relation `q_jk` is not incident to the reference centre, so no centre statistic binds `q_jk` to `{q_ij, q_ik}`. |
| Q3 | Why is this not message passing? | Triads are an explicit, deterministic, parameter-free function of frozen `q_ij`; no hidden state is propagated, no node state updated, no learned depth. |
| Q4 | Full triad enumeration cost? | `4.04–4.11M` triads/fold; 100-molecule profile `0.0042 s/molecule` true+unbound ⇒ `≈8.3 s` per 2000-molecule fold; full enumeration used. |
| Q5 | Is deterministic sampling needed? | **No.** Full `O(n^3)` enumeration is cheap; the 512-triads cap was not applied. |
| Q6 | How is the unbound control constructed? | Keep `q_ij, q_ik`; deterministically permute the outer `q_jk` across triples via molecule-ID-seeded single cycles within same-bucket groups; leftover singletons pooled/permuted; isolated leftover = recorded fixed point. |
| Q7 | Marginal-preservation quality? | Pair-state multiset exactly preserved; bucket mismatch `0.0` on 5/5 folds; fixed-point rate `≤4.95e-7`; breakage `0.974`. |
| Q8 | B0 MAE? | **0.193160**. |
| Q9 | R-only MAE? | **0.190499** (`302 → 13 → 13 → 1`, 4135 params). |
| Q10 | Unbound triad MAE? | **0.190303** (`(302+32) → 12 → 12 → 1`, 4189 params). |
| Q11 | True triad MAE? | **0.192152** (architecture identical to B2). |
| Q12 | mean ΔR? | **−0.001653** (1/5 folds positive). |
| Q13 | mean ΔU? | **−0.001849** (95% CI `[−0.002941, −0.000791]`, `P(>0)=0.0003`, **0/5** positive). |
| Q14 | Fold direction consistent? | Yes, in the negative direction: ΔR positive 1/5, ΔU positive 0/5, both pooled means below the clear-NO-GO thresholds. |
| Q15 | Worth a second backbone? | **No** — Stage 1 is a pre-registered CLEAR NO-GO; seed 1 and Stage-1b were not spent. |
| Q16 | Final: GO / NO-GO / INCONCLUSIVE? | **NO-GO** (Case C). |

---

## Outputs

`results/triadic_relation_binding_witness/`:
`checkpoint_inventory.json`, `triad_integrity_report.json`, `triad_feature_spec.json`,
`runtime_profile.json`, `sampling_manifest.json`, `unbound_control_manifest.json`,
`stage1_fold_results.csv`, `stage1_bootstrap.json`, `final_decision.json`,
`fold_split_manifest.json`, `figures/figure1_per_fold_delta.png`.

Stage-1b / Stage-2 files (`stage1b_init_results.csv`, `stage2_fold_results.csv`,
`pooled_results.csv`, `final_bootstrap.json`) are intentionally absent because the
pre-registered CLEAR NO-GO stopped the audit before those budgets were spent.
