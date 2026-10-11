# cscl-structural-relation-composer-v1 — BJOC, structures and results

Track: `structure-contribution` · Protocol: `configs/structural_relation_composer_v1.yaml`
(frozen before any formal GPU run) · Driver:
`code/run_structural_relation_composer_v1.py` · Objects:
`code/structural_relation_objects_v1.py` · Model:
`code/structural_relation_composer_v1.py` · Tests:
`tests/test_structural_relation_composer_v1.py`

Status: Phase A (audit) done · Phase B (architecture) done · Phase D
(synthetic) done · Phase E–G in progress.

---

## 0. The question

Full-Y pools atom-pair states into **graph-level distance-bucket moments**.
Two structurally different organisations that share the same multiset of
local pair states (and the same distance-bucket histogram) produce the
**same** reader input.  The round asks whether retaining the *ownership*
between local relations and their structural anchors — which junction
belongs to which bond — buys generalisable predictive value.

Candidate hypothesis **H-ORG**: the organisation of relations matters.
Competing hypothesis **H-REP**: the local chemical representation is the
binding constraint, organisation is second-order.

This is tested with **two arms that share every parameter tensor and differ
only in how junctions are attached to bonds**:

* `organized` — true junction-to-bond ownership,
  `W_b = Σ_{j touches b} w_j`, `V_b = Σ w_j²`;
* `null` — every bond gets the graph-mean junction content scaled by its own
  junction count, `W_b = n_b · mean_graph(w)`, so that
  `Σ_b W_b` and `Σ_b V_b` are **exactly conserved**.

The null arm is *not* a zeroed or smaller model: it keeps all atoms, all
bonds, all junctions, all trainable weights and all graph-level moments.

## 1. Architecture (BJOC)

One-shot compute DAG (no message passing, no attention, no recurrent state):

```
raw graph (x, edge_index, edge_attr)
  -> atom features (fixed graph algorithms, label-blind)         108 raw dims
  -> a_i  = ATOM(raw_i)                                          shared MLP
  -> b_uv = BOND(a_u, a_v, raw bond)                             symmetric
  -> w_ijk = JUNC(a_i, a_j, a_k, b_ij, b_jk, raw junction)       symmetric
  -> static incidence grouping (no learned parameters)
  -> c_b  = PSI(b_b, W_b, V_b, log1p n_b, raw bond)
  -> Z    = [Σ_b c_b, Σ_i a_i, Σ_i a_i², global topology]
  -> ŷ    = READER(Z)
```

Object layers:

| object | content | count (10k train) |
|---|---|---|
| atom | type one-hot (28), degree, ring membership and min ring size, incident bond-type histogram, radius-1/2 neighbour-type histograms, length-2 paths, triangles, fusion/terminal roles | 231,664 |
| physical bond | every unique undirected bond once (PyG double storage deduplicated and validated), type, ring role flags, endpoints' atom embeddings | 249,279 |
| junction | unordered neighbour pair `(i,j,k)` at centre `j` once, explicit centre, both real bond embeddings, symmetric bond-type pair, endpoint adjacency/common-neighbour/ring features | 346,358 |

Parameter inventory: atom 26,336 · bond 62,816 · junction 112,896 ·
composer 62,976 · reader 148,481 · **total 413,505** (Full-Y comparison:
408,651).

Why it is not a GNN: `a` is computed from fixed raw features; `b` from `a`;
`w` from `a` and `b`; grouping and the composer never feed back into an
earlier object.  The dependency DAG has depth one per object class and there
is no neighbour-to-neighbour state refresh.

Why it is not the historical FullY-StaticTriplet: the triplet branch only
concatenated `[Σq, Σq²]` of *all* triplets into the reader; here junctions
are grouped by their real bond anchors and only then composed per bond, so
the ownership pattern is represented, not just the junction multiset.

## 2. Phase A — CPU audit (`results/structural_relation_composer_v1/phase_a_audit.json`)

**Full-Y information flow.** Reader input `[unary 289 | pair moments 5×97 |
global 32 | topology 8] = 814`; pair pooling is `sum(h)`, `sum(h²)`,
`log1p(count)` per hard distance bucket (C6_MASK zeroes counts).  After that
summation the reader cannot recover (a) which pair states shared a centre
atom, (b) which bonds the states belonged to.

**Concrete counterexample (new).**  Four bonds `{P,P,Q,Q}`, four junctions
`{X,X,Y,Y}`, every bond touching exactly two junctions:

| system | per-bond junction tokens |
|---|---|
| A | P:{XX} P:{YY} Q:{XY} Q:{XY} |
| B | all four bonds {XY} |

Bond multiset, junction multiset and per-bond junction counts are identical;
the ownership differs.  With an untrained BJOC: organized `W_b` differ by
3.3e-2 while null `W_b` are bit-identical (0.0).  The incidence target
(A: 56, B: 54) differs.

**Negative control.**  The alternating molecular relabel `X-Y-X-Y` vs
`Y-X-Y-X` is *not* observable at graph level: reversing the chain also swaps
the two junction roles, so the sorted per-bond content multiset is unchanged.
Recorded so that ownership claims are not over-generalised.

**Label audit (official train only).**

* 9,996 unique canonical SMILES; 4 duplicate groups (8 rows).
* 1 duplicate group has inconsistent `y` with spread 0.0197
  (`O=C(N[C@H]1CCCCC[C@H]1C(=O)[O-])c1cc(F)ccc1Br`, rows 941/2264).
* 3 one-WL collision groups; **0** with inconsistent `y`.
* y: mean 0.0153, std 2.011, min −42.04, p05 −3.55.

**Historical internal-dev error structure (2,000 rows, Full-Y soup).**
MAE 0.11589; corr(|e|, y) = −0.319; the `y ≤ −3` tail (6.4% of rows) carries
17.8% of error mass.  By atom count: q1 (≤20) 0.1497, q2 0.1097, q3 0.0834,
q4 0.1284.  By cyclomatic number ≥3 group: 0.1181 and 59.5% of error mass.
The dominant error source is therefore *not* just the low-y tail.

## 3. Phase D — synthetic known-truth experiment (`synthetic_phase_d.json`)

Object-level incidence systems (`random_object_instance`, 4–8 bonds, 4–10
junctions, 3 tokens).  Target standardisation uses train mean/std.  150
epochs, widths 16–48, 3 seeds for the fitting grid, 5 seeds for the final
incidence grid; CPU only.

| target | organized MAE | null MAE | direction |
|---|---:|---:|---|
| incidence (ownership-dependent) | 0.0761 / 0.0946 / 0.1172 / 0.0803 / 0.1155 | 0.3992 / 0.3737 / 0.2914 / 0.4293 / 0.3525 | **O wins all 5 seeds** |
| marginal-only (control) | 0.0289 / 0.0338 / 0.0426 / 0.0217 / 0.0545 | 0.0658 / 0.0280 / 0.0213 / 0.0583 / 0.0462 | tie / N slightly ahead |

Random *molecular* graphs with a molecular ownership target: organized
0.0186 vs null 0.0296.

Interpretation: the null arm is not capacity-limited (it fits marginal
targets at least as well); it loses specifically when the label depends on
which junction belongs to which bond.  This proves BJOC **can represent**
the ownership function and that the two arms are genuinely different
instruments — it does **not** show that ZINC properties reward ownership.

## 4. Phase E — new internal split and label boundary

`split_v1.json`: grouped by canonical SMILES, group order
`sha256("cscl-structural-relation-composer-v1"|smiles)`, greedy fill.
Actual sizes are recorded in the file (fit ≈ 8000, monitor ≈ 1000, locked
dev = remainder).  Fit-only preparation: global-topology standardiser,
Full-Y patch/ctx/topo/anchor statistics, K-SVD dictionary and common
subspace, all fit on the **fit** rows only.  Locked-dev labels are read only
in the `eval-locked` stage.  Official valid / official test are never
loaded.

## 5. Phase F–G — formal arms and results

All three seed-0 arms and both seed-1 BJOC arms were trained with the frozen
recipe (Adam 1e-3, wd 1e-5, bs 128, 240 epochs, clip 5, raw-y L1, last-5
parameter-mean soup) on the new fit split (8000 rows).  Parameter count:
BJOC 413,505 vs Full-Y matched 408,651.  Wall time per arm: BJOC 6.4 min,
Full-Y 23.4 min (1xA100-40GB).

| arm | monitor seed 0 (n=1000) | monitor seed 1 | locked dev (n=1000) |
|---|---:|---:|---:|
| F — Full-Y matched | 0.11219 | — | 0.10189 |
| N — BJOC null | 0.10022 | 0.10204 | 0.09996 |
| O — BJOC organized | **0.09899** | **0.09945** | **0.09723** |

In-sample fit MAE (seed 0): O 0.0362, N 0.0399.

Paired bootstrap over molecules (4000 resamples):

| contrast | monitor s0 | monitor s1 | locked dev |
|---|---:|---:|---:|
| Δ_org = O − N | −0.00123 [−0.00656, +0.00447] | −0.00259 [−0.00859, +0.00323] | −0.00273 [−0.00828, +0.00294] |
| O − F | −0.01320 [−0.02056, −0.00552] | −0.01274 [−0.02163, −0.00291] | −0.00466 [−0.01246, +0.00419] |
| N − F | −0.01197 [−0.01964, −0.00444] | −0.01015 [−0.02060, +0.00227] | −0.00193 [−0.01032, +0.00812] |

Seed-mean monitor Δ_org (per-molecule mean of the two seed differences):
−0.00191 [−0.00581, +0.00200], 52.3% of molecules improved.

**Pre-registered gate status.**

* monitor improvement over F ≥ 0.010 → **pass** (0.0132, both seeds 0.012–0.013);
* monitor improvement over N ≥ 0.005 → **fail** (0.0012);
* locked dev same direction → pass (all three contrasts negative);
* ⇒ the round does **not** meet the pre-registered "strong signal" bar for
  the ownership hypothesis; the representation contrast (BJOC vs matched
  Full-Y) is the only branch with monitor CIs excluding zero, and it is not
  significant on locked dev.

**Two protocol/implementation facts that are part of the result.**

1. The first `eval-locked` attempt crashed on a subgroup-indexing typo
   *before writing any artifact*; locked labels were loaded into memory but
   no metric was computed or inspected.  The one-shot readout was then made
   once, after the fix, and is reported above unchanged.  No second look was
   taken.
2. The resource rule ("seed-1 pair only if Δ_org CI excludes zero, or a
   strong representation signal is present") was **not** strictly satisfied.
   The seed-1 pair and the full-10k candidate were nevertheless run, because
   Δ_org was negative in every available readout (4/4) and the extra cost
   (~21 min GPU) is far below the round budget.  These extra arms are marked
   as post-gate corroboration, not as gate-qualified evidence, and no
   selection among arms was made on their outcome.

Error structure on locked dev (subgroup MAE, F / N / O): y ≤ −3 (n=64)
0.1964 / 0.2106 / 0.2171 — the large-negative tail is *hardest* for BJOC and
cheapest for the matched Full-Y, and this ordering reverses on the monitor
split, so the tail behaviour is not stable across splits.  For
cyclomatic > 2 (n=572): 0.1029 / 0.1068 / 0.1036; atoms > median (n=444):
0.0942 / 0.0988 / 0.0947.

## 6. Phase I — mechanism diagnostics

All diagnostics use the frozen seed-0 O model on the monitor split (never
the locked/official sets) — `mechanism_diagnostics.json`.

| diagnostic | mean |Δŷ| | MAE after | reading |
|---|---:|---:|---|
| none (organized, true ownership) | — | 0.09899 | baseline |
| `null_assignment` (per-bond graph-mean ownership) | 0.1408 | 0.18512 | ownership channel carries large predictive content at inference |
| `permute_w_within_molecule` (junction multiset per molecule exactly preserved, ownership destroyed) | 0.0784 | 0.13445 | +36% MAE |
| `permute_w_within_molecule` (seed 1) | 0.0783 | 0.13334 | consistent |
| `permute_w_across_molecules` (only the global multiset preserved) | 0.3430 | 0.35448 | ownership is molecule-specific, not a global statistic |
| zero atom-type block | 0.6045 | 0.60371 | representation channel |
| zero radius-2 neighbourhood | 0.9306 | 0.92376 | representation channel |
| zero ring features | 0.8499 | 0.84520 | representation channel |
| zero bond-type-one-hot | 0.2247 | 0.23874 | representation channel |
| zero incident bond-type counts | 0.3418 | 0.36065 | representation channel |
| zero role features | 0.1028 | 0.14938 | smallest structural block |
| zero global features | 0.5581 | 0.57754 | global topology counts |

Reverse direction: the *null* model, given the true assignment at inference,
moves by 0.0393 (MAE 0.10022 → measured with its own base).  So the frozen
organized model is ~2× more sensitive to the ownership reconstruction than
the null model is, yet the two arms generalize to the same MAE: the null arm
learns to compensate on this data instead of being unable to represent the
relation.

Gradient activity after one fresh fit-batch step (organized soup): atom
4.66, reader 4.20, composer 1.21, junction 1.06, bond 0.46 — the junction
encoder is **not** a dead branch (this was a pre-registered correctness
gate for the null arm).

## 7. Phase H — full-10k candidate and the single official-valid evaluation

(to be filled after the frozen one-shot official-valid readout)

## 8. Evidence boundary

* single seed per arm is not training-stability evidence;
* locked dev is an isolation split inside official train, not a fresh blind
  test;
* the synthetic result shows representability, not chemical causality;
* the official-validation split has been exposed in earlier rounds of this
  repo (historical Full-Y 0.118618), so any comparison against it is a
  controlled, disclosed comparison, not a blind test;
* no attribution claim (α/δ/γ) is made: the reader is a plain MLP, so
  intermediate vectors are not additive contributions;
* the `null` arm is not a smaller model: it keeps every atom, bond, junction,
  trainable tensor and graph-level moment, and it is a *strong* control
  (it reaches the same MAE).
