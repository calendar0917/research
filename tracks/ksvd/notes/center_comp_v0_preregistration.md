# Preregistration — CENTER-COMP-v0: Center-Preserving Composition Diagnostic

Round name: **CENTER-COMP-v0** (*Center-Preserving Composition Diagnostic*).
Study: `zinc-context-gap`.
Status: frozen before the single formal seed-0 run on 2026-09-22.

## 0. This is a new round

This is a new, self-contained round. It does **not** reopen TCCD-v4, TCCD-v5,
or TCCD-v6, and it does not re-run any existing baseline, seed sweep, or
official-valid / official-test run. All existing results are read as frozen
references from their formal artifacts.

## 1. Question

TCCD-v2's frozen representation plateaus at Top-5 soup MAE `~0.25–0.28` while
the canonical strong GPU1 reference is `~0.119818`. TCCD-v4 showed that the
parameter-free normalized two-hop moment `C^T S^2 C` adds no material signal.
TCCD-v5 showed that moving one shared nonlinear pair transform before global
pair pooling (PRE) does not materially beat placing it after pooling (POST),
and that an assignment-location shuffle barely changes PRE. TCCD-v6 attributed
the entire v5 POST gain to base-reconstructible coordinate exposure.

None of those rounds tested whether the **binding between a concrete center
occurrence and the pair interactions it participates in** carries predictive
information that global pair/statistical contraction destroys. This round asks
exactly one causal question:

> With the **identical, frozen TCCD-v2 local prototype codes**, does retaining
> *which pair relations belong to which concrete center occurrence* before
> graph-level pooling materially break the existing frozen-representation
> ceiling?

This is a **composition sufficiency diagnostic**, not a new local-dictionary
experiment, and not a message-passing architecture search.

## 2. Why this is different from TCCD-v4 and TCCD-v5

* **TCCD-v4** tested `C^T S^2 C`, a higher-order moment in which occurrence
  identity has already been contracted away by global quadratic composition.
  Its failure therefore does not rule out occurrence-preserving composition.
* **TCCD-v5** tested `p_ij -> phi(p_ij) -> global sum` (PRE). Every pair is
  transformed and then dropped into a single global pool; the representation
  never records that pair `(i,j)` and pair `(i,k)` share the same occurrence
  `i`. PRE preserves pair occurrences but not **pair-to-center incidence /
  binding identity**.
* **The single new piece of information in this round** is the incidence
  identity. The path is forced to be

  ```text
  pair -> center -> graph
  ```

  instead of

  ```text
  pair -> graph
  ```

  and the center nonlinearity is applied *before* global pooling.

A concrete illustration: two molecules may have nearly identical prototype
counts and identical A–B and A–C pair counts, while in one molecule a single B
connects to both A and C, and in the other two distinct B occurrences each
connect to one of them. Global pair statistics cannot in general separate
these; center-preserving composition can, because it keeps `(c_i, q_i)`
together.

## 3. Frozen lineage and execution discipline

* Base representation: exact TCCD-v2 PrototypeREL.
* Checkpoint: `tracks/ksvd/results/tccd_v2/prototype_rel_seed0_best.pt`.
* Checkpoint SHA-256:
  `093e0e46d4d5dab7e5558b807d993d1136df58d8f531d6b9b3621e8ab05c0f42`
  (verified before any work; mismatch stops the round).
* Local encoder `W`, prototypes `P`, and temperature are **frozen**. The task
  gradient must not reach them. This round never instantiates a trainable
  local encoder, never refits K-SVD/OMP/IHT, and never uses the CHEM-CONT or
  GRAD-CONT checkpoints.
* Reused pair cache: the exact TCCD-v5 pair cache
  (`tracks/ksvd/results/tccd_v5/cache/pair_all_*.npy`), protocol
  `tccd_v5_occurrence_pair_nonlinearity_v1`, metadata
  `official_test_loaded: false`. It is loaded byte-for-byte through the
  TCCD-v6 label-free loader; it is **not** regenerated.
* Internal split: exact existing `internal_split(10000)`, split seed
  `20260922`, 8000 train / 2000 dev.
* Formal compute: remote A100 **GPU1 only**. GPU0 is forbidden.
* Seed: `0` only. No seed 1, no seed rescue.
* Official ZINC valid and test are blocked throughout and never loaded.

## 4. Reused pair descriptor (unchanged from TCCD-v5)

For every graph with `n` occurrences and every unordered pair `i<j`:

```text
p_ij = [c_i + c_j, |c_i - c_j|, c_i * c_j, r_ij] in R^197
```

with the exact TCCD-v2 relation ordering `[R_int, R_b0, R_b1, R_b2, R_geo]`.
No new shortest-path histogram, ring feature, cycle feature, global descriptor,
B-Full feature, or chemistry statistic is added.

## 5. Pair -> Center -> Graph operator (frozen)

```text
e_ij  = phi(p_ij)                         (shared pair embedding, 16-D)
q_i   = sum_{j != i} e_ij                 (center pair context, 16-D)
u_i   = rho([c_i, q_i])                   (center nonlinear composition, 16-D)
h_center = sum_i u_i                      (global center pool)
h     = [h_base, h_center]
y_hat = Linear(10480, 1)(h)
```

* Every unordered pair `(i,j)` contributes `e_ij` to **both** endpoint
  centers `i` and `j`, exactly once each.
* `phi`: `Linear(197, 64) -> ReLU -> Linear(64, 16)`; `13,712` parameters
  (exact TCCD-v5 registered pair-MLP family; no width sweep).
* `rho`: `Linear(80, 64) -> ReLU -> Linear(64, 16)`; `6,224` parameters
  (one fixed small shared MLP; no depth/width sweep).
* Final reader: `Linear(10480, 1)`; `10,481` parameters. `h_base` is the
  frozen TCCD-v2 10,464-D graph representation read from the reused cache.
* Total trainable parameters: `30,417`.
* Pair-to-center incidence is obtained deterministically at batch time by
  scatter-adding `e_ij` into the endpoint index arrays `pair_i0`, `pair_i1`;
  no large new pair tensor or duplicated pair cache is written.

This operator has exactly one pair-encoding stage, one pair-to-center grouping,
one center nonlinear map, and one global pooling. It is an **explicit
incidence-preserving composition operator**, not a GNN backbone.

## 6. Evaluation-only intervention: CENTER-BIND-SHUFFLE

No second model is trained. On the single trained CENTER-COMP checkpoint, at
evaluation only, for each molecule one deterministic fixed permutation
`pi` of its `n` occurrences is generated (same seed and graph index as the
registered TCCD-v5 permutation helper, `PERM_SEED = 20260922`; identity for
`n < 2`), and the center composition uses

```text
u_i^shuffle = rho([c_i, q_{pi(i)}])
```

while `c_i`, the multiset `{q_i}`, the pair-embedding multiset, the graph size,
and `h_base` are all unchanged. This destroys only **which local environment
occurrence is bound to which center-composition context**. The intervention is
applied to both the Top-5 soup and the best checkpoint.

## 7. Training scope

Only one model is trained: **CENTER-COMP seed 0**. Protocol is the exact
TCCD-v5 lightweight frozen-screen protocol:

* Adam, batch `32`, learning rate `1e-3`, weight decay `1e-5`, gradient clip
  `5`, maximum `240` epochs, patience `40`, equal-weight Top-5 soup.
* Primary metric: Top-5 soup MAE on the fixed internal 2000-graph dev split.
* No end-to-end training, no official-valid, no official-test, no seed 1, no
  architecture or hyperparameter sweep.

## 8. Frozen comparison references (read, never re-run)

| representation | Top-5 soup MAE | status |
|---|---:|---|
| TCCD-v2 BASE | `0.2783639132976532` | existing |
| TCCD-v5 POST | `0.2522333264350891` | existing |
| TCCD-v5 PRE | `0.2514181435108185` | existing |
| TCCD-v5 PRE-SHUFFLE | `0.25175556540489197` | existing eval-only |
| CENTER-COMP | new | this round |
| CENTER-BIND-SHUFFLE | new eval-only | this round |
| canonical GPU1 strong B-Full | `0.119818` | absolute scale reference |

Exact values are read from `tracks/ksvd/results/tccd_v5/stageA_seed0.json` and
`tracks/ksvd/results/tccd_v6/stageA_seed0.json`, not hand-copied as
computation inputs.

## 9. Primary quantities

```text
MAE_prev     = min(MAE_v5_PRE_soup, MAE_v5_POST_soup)
G_center     = MAE_prev - MAE_CENTER_soup
G_bind       = MAE_CENTER_BIND_SHUFFLE_soup - MAE_CENTER_soup
gap_to_strong= MAE_CENTER_soup - 0.119818
```

`G_center` measures the increment beyond the best existing frozen composition.
`G_bind` measures whether that increment actually depends on correct
center binding. Comparing CENTER to BASE alone is not informative, because
TCCD-v5/v6 already show that reader-coordinate effects alone are worth about
`0.026`.

## 10. Preregistered outcome logic

**Outcome A — CENTER IDENTITY IS MATERIAL**

```text
G_center >= 0.010  AND  G_bind >= 0.010
```

Interpretation: preserving which relations share a concrete occurrence, and
which local environment that context belongs to, supplies predictive
information missing from the existing global pair/statistical representation.
Next step: proceed to a fuller assembly representation / compression ladder;
do not return to local-continuity repair.

**Outcome B — CAPACITY GAIN, NOT CENTER INFORMATION**

```text
G_center >= 0.010  AND  G_bind < 0.005
```

Interpretation: the new branch has predictive gain but does not depend on true
center binding; it is most likely another reader/coordinate/capacity effect.
Do not claim composition identity is the bottleneck.

**Outcome B-AMBIGUOUS (intermediate, reported honestly)**

```text
G_center >= 0.010  AND  0.005 <= G_bind < 0.010
```

Interpretation: capacity gain with weak binding evidence; not A. No center
identity claim is made.

**Outcome C — CENTER INFORMATION USED BUT NOT ENOUGH**

```text
G_bind >= 0.010  AND  G_center < 0.010
```

Interpretation: the model does use center binding, but this single layer is
insufficient to break the existing frozen-representation ceiling. Decide
whether a fuller assembly representation is worth testing; do not simply
deepen the reader.

**Outcome D — NO MATERIAL SIGNAL**

```text
G_center < 0.010  AND  G_bind < 0.010
```

Interpretation: one layer of pair-to-center incidence does not supply material
missing signal. This weakens the hypothesis that shared-center identity is the
main `0.25` bottleneck. No automatic second layer or GNN rescue.

## 11. Absolute interpretation

Even if the causal gates pass, report `gap_to_strong`. A drop from
`0.251 -> 0.235` is a causal PASS but must still be reported as: center identity
matters, but does not by itself explain the full `0.25 -> 0.12` gap. A drop to
`~0.18` or lower would be strong evidence that aggressive global composition
compression is a principal performance bottleneck.

## 12. Gate 0 (must pass before training)

1. frozen TCCD-v2 `C` re-derived from the checkpoint matches the reused cache
   on a real subset (max abs diff `< 1e-6`);
2. the pair descriptor is numerically identical to the existing TCCD-v5
   implementation (max abs diff `< 1e-6`);
3. every unordered pair contributes exactly to both endpoint centers;
4. graph relabel permutation invariance of `q`, `h_center`, and prediction;
5. pair-enumeration-order invariance;
6. batching invariance;
7. task gradient reaches `phi`, `rho`, and the final reader;
8. task gradient does not reach `W`, prototypes, or temperature;
9. CENTER-BIND-SHUFFLE preserves the `{q_i}` multiset, the `{c_i}` multiset,
   and `h_base`;
10. official valid / official test blocked and never loaded.

No BASE smoke training is performed. Any Gate-0 failure stops the round.

## 13. Mechanism and stratification reporting (read-only)

* Main table: model, existing/new, soup MAE, delta vs best existing, bind-shuffle
  delta.
* Mechanism table: real CENTER vs CENTER-BIND-SHUFFLE; mean / median / p90
  absolute prediction change; fraction of molecules with material prediction
  change (primary threshold `|delta| >= 0.01`; `>= 0.05` and `>= 0.1` also
  reported).
* Size stratification (read-only, no retraining): dev molecules sorted by
  occurrence count `n` into small / medium / large tertiles; report real MAE,
  shuffle MAE, and `G_bind` per stratum, to check whether the effect grows with
  assembly complexity.

## 14. Explicitly forbidden work

No baseline rerun (TCCD-v2 BASE, TCCD-v4, TCCD-v5 PRE/POST/PRE-SHUFFLE,
TCCD-v6, CHEM-CONT, GRAD-CONT, B-Full/B-Null); no seed sweep; no end-to-end
prototype training; no local continuity loss; no K/temperature/relation/local
checkpoint sweep; no pair/center width or depth sweep; no dropout or
normalization sweep; no iterative node-state updates, multiple propagation
layers, GCN/GINE, Transformer, attention, or recurrent propagation; no new
handcrafted relation, shortest-path histogram, ring/cycle feature, or global
descriptor; no official-valid; no official-test.

## 15. Durable records

Required after analysis:

* this preregistration;
* `tracks/ksvd/results/center_comp_v0/gate0.json`;
* `tracks/ksvd/results/center_comp_v0/stageA_seed0.json`;
* trained CENTER-COMP soup / best checkpoints and the eval-only shuffle
  metrics in the Stage-A JSON;
* targeted tests `tracks/ksvd/tests/test_center_comp_v0.py`;
* analysis note, claim, decision, and `tracks/ksvd/STATE.yaml` update.

Every record states GPU1-only, `official_test_loaded: false`, the frozen
checkpoint SHA, the reused-cache protocol, per-arm metrics and verdict, the
`G_center` / `G_bind` values, and the final outcome letter.
