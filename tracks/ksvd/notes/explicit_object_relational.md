# Explicit Object Relational Reasoning (EOR) — late relational reasoning over explicit B0/B1/B2 objects

**Study**: `zinc-context-gap` · **Experiment**: `exp/explicit-object-relational-reasoning`
**Verdict**: **STOP** (pre-registered stop branch) — **not promoted**, **no second seed**.
**Failure mode**: the explicit-object relational encoder **collapsed to a constant
during training** (dead ReLU units), so the structural channel was switched off and
the candidate ran as a structure-free model.

| item | value |
|---|---|
| training commit | `f409a16ffeed36f7374e672e069d59c994f630ea` |
| diagnostic-fix commit | `8e10eb87772689d902c81ee8f50f4027aeef0374` (no effect on training) |
| protocol | deterministic A100, `torch.use_deterministic_algorithms(True)`, seed 1 only |
| candidate params | 84,495 (== B-full exactly); encoder 35,152 (== B-full encoder exactly) |
| seed1 raw valid | 0.131644 @ epoch 125 (early stop @165, wall 4806.3 s, peak GPU 1.288 GB) |
| seed1 Top-5 soup valid | **0.125834** (epochs 125/161/146/162/160) |
| references | B-full seed1 0.118126; explicit-basis (exp-1) seed1 0.120194; B-bag 0.120229; A2 0.122134 |
| sanity / tests | 53/53 sanity checks; 17 new unit tests; 419 affected-suite tests |
| determinism | 6-epoch GPU0 vs GPU1 repro bit-identical (`dbb36565…d9a2d`) |
| official ZINC test | **never loaded**; no MolHIV |

---

## 1. Question

Experiment 1 (`explicit_structural_basis`) replaced B-full's hidden GNN structural
channel with an explicit deterministic B0/B1/B2 basis scored by one shared learned
scalar valuation and pooled **immediately**. It reached 0.120194 — a numeric tie with
the connectivity-free B-bag (0.120229), not B-full (0.118126); its order ablation
showed B1 bond objects dominate but carry only *local endpoint* information. The
diagnosis was a **representational limit**: a per-object scalar cannot propagate
information across objects.

Experiment 2 asks the single pre-registered follow-up:

> Keep the objects explicit and keep their **relations** explicit. Can **late pooling +
> explicit relation-conditioned recurrent reasoning** (`T_obj = 2`) over the B0/B1/B2
> objects recover B-full's connectivity gain?

Honest framing (frozen in the brief): this is *neural relational reasoning over
explicitly enumerated, support-traceable structural objects and relations* — not
"structure discovery", not "not a GNN". No parameter may create, delete or re-weight
an object or a relation.

## 2. Frozen architecture

Reused the **exact** exp-1 object family `B(P) = B0 ∪ B1 ∪ B2` and added an explicit
object–object relation graph plus shared recurrent reasoning:

```
h_i^0     = E_{order(i)}(explicit object i)                         in R^8   (d_obj = 8)
q_ij^t    = Q(h_i^t, h_j^t, r_ij)                                   in R^8   (q_obj = 8)
A_i^t     = fixed invariant stats of q over incident relations
h_i^{t+1} = h_i^t + U([h_i^t, A_i^t])        (U tied across rounds AND orders)
s(P)      = fusion(late per-order pooling of h^2)
e_struct  = b + s(P)·v                                              (rank-1, R^16)
```

* `T_obj = 2` exactly; `Q` and `U` are single weight-tied modules (no per-round copy).
* `A_i^t` uses fixed invariant stats `[mean, std, log1p(count)]`, optionally split by
  ≤3 pre-registered deterministic relation families; **no attention, no softmax**.
* Outer backbone frozen identical to B-full: patch encoder, outer h=64, q=16, T=2
  weight tying, `patch_cont`, parent channel, relation descriptor/encoder, centre
  update, global/topology branches, readout/head.

## 3. Explicit relation graph

Two distinct objects **inside the same patch** get an edge iff one of three
support-derived conditions holds:

* `bond_overlap` — their bond supports intersect;
* `overlap` — their atom supports intersect (share an atom, incl. containment);
* `adjacent` — supports are disjoint but a real molecular bond connects an atom of one
  to an atom of the other.

No edge otherwise. Edges are unique undirected pairs `i < j`. The symmetric descriptor
`r_ij` (width 23) is endpoint-order invariant: order-pair one-hot (6), atom/bond overlap
counts, containment, share-centre, share-root, adjacent flag (6), connecting bond-type
one-hot (4) + has_connecting (1), min atom-graph distance bucket (1), size_min/size_max
(2), family one-hot (3). No learned relation-ID lookup.

**Zero-training sparsity audit** (official train 10,000 / valid 1,000):

| split | molecules | objects | relations | mean rel/mol | p95 | mean edge/object | max edge/object |
|---|---|---|---|---|---|---|---|
| train | 10,000 | 4,025,826 | 28,195,862 | 2,819.59 | 4,193 | 6.944 | 11.160 |
| valid | 1,000 | 401,412 | 2,811,834 | 2,811.83 | 4,153 | 6.953 | 10.572 |

Family totals (train) overlap 12,579,643 / bond_overlap 4,858,825 / adjacent
10,757,394. The graph is support-local and sparse — the edge/object ratio ~7 is far
from the complete-graph O(M²) regime. Relation-cache build 761.6 s (basis 59.2 s).

## 4. Parameter budget

`params` stage: candidate **84,495** (== B-full exactly, Δ0, within 1 %), encoder
**35,152** (== B-full encoder exactly, Δ0). No outer head/h/q/global-branch inflation.
Encoder breakdown:

| block | params |
|---|---|
| primitive embeddings (atom/root/distance/degree/boundary/bond) | 376 |
| per-order object encoders E0/E1/E2 (object_hidden 100) | 10,024 |
| inner relation encoder (23→64→16) | 2,576 |
| inner Q encoder (40→64→8, tied) | 3,144 |
| inner update U (104→79→8, tied, both rounds) | 8,935 |
| late fusion (72→136→1) | 10,065 |
| rank-1 projection (bias + direction) | 32 |
| **total** | **35,152** |

Widths (`object_hidden=100`, `update_hidden=79`, `fusion_hidden=136`) were chosen ONCE
by parameter accounting, never by validation.

## 5. Training protocol and result

Frozen protocol, seed 1 only: deterministic A100, Adam lr 1e-3 wd 1e-5 batch 128,
max 240 epochs, patience 40, no scheduler, L1 loss, grad clip 5, fixed Top-5 soup
(ties → earlier epoch), equal parameter average.

* best raw valid **0.131644** @ epoch 125; early stop @165; wall 4806.3 s; peak GPU 1.288 GB
* Top-5 epochs [125, 161, 146, 162, 160]; Top-5 valid MAE [0.131644, 0.132683, 0.134095, 0.134512, 0.134560]
* **Top-5 soup valid 0.125834** (soup improves on the best checkpoint by 0.005810)

## 6. Gate verdict

Pre-registered gates: Strong GO ≤ 0.117126; Method GO ≤ 0.119126; Partial GO
∈ (0.119126, 0.121126] **and** < 0.120194; STOP ≥ 0.120194 or > 0.121126.

`decide`:

```
seed1_soup_valid                     0.125834
Δ vs B-full seed1 (0.118126)        +0.007707
Δ vs exp-1 explicit basis (0.120194)+0.005639
Δ vs B-bag seed1 (0.120229)         +0.005604
Δ vs A2 seed1 (0.122134)            +0.003699
verdict                              STOP
second_seed_authorized               false
```

The candidate is worse than **every** reference, including the connectivity-free B-bag
and the simpler exp-1 scalar-valuation basis. Per the stop rule: no rescue, no second
seed.

## 7. Mechanism — the inner encoder collapsed (primary finding)

`diagnostics` and `mechanism` show the structural channel is **dead**, not merely
unhelpful:

| quantity | at initialization | after training |
|---|---|---|
| h⁰ object norm (mean) | 1.324 | 4.7e-29 |
| h² object norm (mean) | 2.177 | 7.5e-24 |
| h per-dim std (mean) | 0.352–0.359 | 8.3e-30 |
| h² effective rank | 5.29 | 1.16 |
| s(P) std | ~O(1) | **1.8e-9** (`s_collapse: true`) |
| e_struct row-norm std | ~O(1) | 4.7e-10 |

At initialization every object state is healthy (norm ~1.3–2.2, effective rank
4.7–5.3) and the channel is exactly rank-1 by construction (effective rank 1.0).
After training **all ReLU units in E0/E1/E2 and U are dead**: `h⁰ = 0` and the update
outputs zero, so `e_struct` becomes a constant absorbed by the outer bias. Encoder
weights did move and grow (L2 10.35 → 13.29 in the soup; L1 move fresh→selection 775),
confirming the encoder was trained and *converged* to the dead region rather than
never receiving gradients. This is a genuine dying-ReLU / dead-unit collapse, not a
wiring bug: `model.structural_encoder` is the same module used in the forward pass,
and it is included in `torch.optim.Adam(model.parameters())`.

**Interpretation**: the hypothesis "late relational reasoning over explicit objects
recovers B-full" was **not cleanly tested**, because the optimizer switched the inner
encoder off. The correct claim is that *this* frozen architecture/optimization is
collapse-prone; it is **not** evidence that explicit relational reasoning cannot work.

## 8. Frozen mechanism ablations (no retraining)

Each ablation is applied to the trained states via the frozen hooks:

| ablation | Δ valid (raw) | Δ valid (soup) |
|---|---|---|
| relation-zero | −1.0e-9 | +1.0e-8 |
| round-1-only (T_obj=1) | +1.9e-9 | −4.8e-9 |
| mask `overlap` | −8.6e-9 | +6.7e-9 |
| mask `bond_overlap` | +1.2e-8 | +8.6e-9 |
| mask `adjacent` | +9.5e-10 | +2.0e-8 |

Every ablation is ~1e-8 — numerically indistinguishable from no-op. This is the
signature of a constant channel: **relations, the second round, and every relation
family contribute exactly nothing** to the trained candidate.

Scalar alignments (on 23,083 valid patches):

* s_new(P) vs B-full frozen PC1: Spearman 0.191, Pearson 0.197, R² 0.039 (weak, on a dead channel)
* s_new(P) vs exp-1 explicit-basis scalar: Spearman −0.143, Pearson −0.085, R² 0.007 (uncorrelated)
* s_new(P) from 224 pre-existing deterministic patch descriptors (ridge α=1): valid R² 0.469 — i.e. what little variation exists is explainable by existing descriptors, not by the new relational computation.

## 9. Adversarial witnesses (§18)

1. **Relation-topology intervention witness** — take a real batch and apply 64
   deterministic 2-switch rewires to `eor_edge_i/eor_edge_j` (object set and count
   unchanged). Early explicit-basis pooling delta **exactly 0.0** (objects/features
   untouched), relation topology changed, late EOR output delta **9.20e-4**.
   This isolates "relations among explicit objects" from "new object features".
2. **Connectivity witness** (7-atom pair): same B0 primitive multiset, same B1 local
   primitive multiset, **different B2 basis**, same triple count (37). Early
   explicit-basis delta 6.47e-4, late EOR delta 7.74e-5. (A near-identical-early /
   different-late witness; weaker than the intervention because the B2 basis itself
   changes.)

## 10. Integrity

* 53/53 sanity checks pass on the remote A100 (`failed: []`), including: supports
  identical to exp-1, node-relabel permutation invariance, relations fully
  support-deterministic, no edge for disjoint+unbonded pairs, overlap / containment /
  share-atom / adjacent / graph-distance correctness, endpoint-swap invariance, fixed
  widths, `T_obj = 2`, `Q`/`U` tied, no attention/softmax/certificate/support/object
  gate, late pooling only after h², rank ≤ 1, outer h/q/T = 64/16/2, forward/backward
  finite, deterministic GPU smoke.
* 17 new unit tests (`tracks/ksvd/tests/test_explicit_object_relational.py`) + 419
  affected pooling-model tests pass locally.
* 6-epoch deterministic repro on GPU0 and GPU1: bit-identical selection-state SHA-256
  `dbb365655f8d5dbbe1bc71b97d17f7a775fa5c35810e286c893dc661cedd9a2d`.

## 11. Compute

`profile` (batch 128, 8 batches, A100): EOR forward 0.06284 s vs B-full 0.03200 s →
**1.96×** (under the 2× guard), vs explicit-basis 0.00581 s → 10.82×; peak GPU 744 MB
(profile) / 1.288 GB (training); mean 17.39 objects/patch and 121.81 relations/patch.
The relational machinery is affordable but carries a ~2× latency cost with no gain.

## 12. Interpretation

* The candidate is a **pre-registered STOP**: 0.125834, worse than B-full (+0.007707),
  worse than the exp-1 scalar-valuation basis (+0.005639), worse than the
  connectivity-free B-bag (+0.005604).
* The failure is **optimization collapse of the explicit-object relational encoder**,
  a different failure mode from exp-1 (which had a healthy `s(P)` and a representational
  limit). The two experiments therefore do **not** jointly prove that explicit relational
  reasoning is impossible.
* Keep **B-full** as the frozen ZINC Cell A reference.
* Official ZINC test never loaded; no MolHIV training; no second seed.

## 13. Questions Q1–Q22

1. **Did the candidate reproduce or beat B-full?** No — 0.125834 vs 0.118126 (+0.007707).
2. **Did late relational reasoning recover B-full's connectivity gain?** No; it fell
   below every reference.
3. **Is the structural channel alive or collapsed?** Collapsed to a constant
   (`s` std 1.8e-9, h⁰ = h² ≈ 0).
4. **What is the failure mode: representation or optimization?** Optimization
   (dead ReLU units in the inner encoder), unlike exp-1's representational limit.
5. **Are the objects/relations exactly the pre-registered explicit ones?** Yes — the
   exact exp-1 `B(P)=B0∪B1∪B2` supports, verified identical by sanity.
6. **Is the relation graph sparse and support-derived?** Yes — mean edge/object 6.94
   (train), max 11.16, entirely support-derived; far from O(M²).
7. **Does relation-zero ablation change performance?** No — Δ soup +1.0e-8.
8. **Does the second round matter?** No — round-1-only Δ soup −4.8e-9.
9. **Which relation family matters?** None — all three family masks Δ ≤ 2e-8.
10. **Does s_new align with B-full's frozen PC1?** Weakly (Spearman 0.191, R² 0.039),
    on a near-dead channel — not a reproduction of B-full's axis.
11. **Does s_new align with the exp-1 explicit-basis scalar?** No (Spearman −0.143,
    R² 0.007).
12. **Is s_new predictable from existing deterministic descriptors?** Substantially
    (ridge valid R² 0.469), i.e. no new relational information.
13. **Do the witnesses isolate relations from object features?** Yes — the
    relation-topology intervention has exactly zero early-pooling delta and a 9.2e-4
    late delta.
14. **Compute cost?** 1.96× B-full forward latency, 10.82× exp-1 basis, 744 MB peak.
15. **Is the encoder budget-matched?** Yes — 35,152 params, exactly B-full's encoder.
16. **Deterministic / reproducible?** Yes — bit-identical 6-epoch GPU0/GPU1 repro.
17. **Sanity/test coverage?** 53/53 sanity, 17 new unit tests, 419 affected tests.
18. **Parameter breakdown?** E0/E1/E2 10,024; relation encoder 2,576; Q 3,144; U 8,935;
    fusion 10,065; primitive embeddings 376; rank-1 32 → 35,152.
19. **Is there evidence relations help at init before collapse?** At init the object
    states are healthy (rank 4.7–5.3, norm ~1.3–2.2) and the channel is exactly rank-1;
    the collapse happens under training. This is a collapse-risk finding, not a
    demonstrated relational gain.
20. **Was any forbidden mechanism used?** No B3/B4, rings/motifs, learned objects,
    dense pairs, attention/softmax/Transformer, GRU/LSTM, learned support/existence
    gate, certificate/token lookup, learned clustering; no sweep of h_obj/q_obj/T/
    relation features/optimizer; no outer architecture change.
21. **Official test / MolHIV status?** Official ZINC test **never loaded**; no MolHIV
    training; no second seed.
22. **Single next step?** See §14.

## 14. Single next step

Do **not** promote, do **not** buy a second seed, do **not** sweep widths/rounds/
families, and keep B-full as the frozen Cell A reference. The only justified follow-up
is a **fresh, pre-registered collapse-risk study** (zero or minimal training): test
whether the dead-unit collapse is intrinsic to the tied recurrent ReLU object update
over explicit objects, and if so, pre-register one collapse-avoiding inner update
(non-saturating / residual-gated) with its **own frozen collapse witness** — explicitly
*not* a rescue of this run.

## 15. Artifacts

* code: `tracks/ksvd/experiments/luyin16/explicit_object_relational.py`,
  `tracks/ksvd/experiments/luyin16/zinc_explicit_object_relational.py`,
  pooling wiring in `tracks/ksvd/experiments/luyin16/zinc_patch_path_pooling.py`
* tests: `tracks/ksvd/tests/test_explicit_object_relational.py`
* results: `tracks/ksvd/results/explicit_object_relational/` (sanity, parameter_accounting,
  diagnostics, mechanism, compute_profile, decision, report, runs/, soup_states/, states/,
  curves/, cache/)
* records: claim + decision YAML 2026-09-15
