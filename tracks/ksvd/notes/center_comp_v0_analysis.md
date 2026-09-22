# CENTER-COMP-v0 — Center-Preserving Composition Diagnostic

Preregistration: `notes/center_comp_v0_preregistration.md`.
Preregistration commit: `552e3d6`.
Formal implementation commit: `cf87d1e` (device-fix commit).
Formal result: `results/center_comp_v0/stageA_seed0.json`.
Gate 0: `results/center_comp_v0/gate0.json`.

## Verdict

**Outcome D — NO MATERIAL SIGNAL (borderline binding).** Retaining the
pair-to-center incidence (`pair -> center -> graph`) does **not** break the
existing frozen-representation ceiling: CENTER-COMP Top-5 soup MAE `0.254337`
is *worse* than the best existing frozen composition, TCCD-v5 PRE `0.251418`
(`G_center = -0.002919`). The evaluation-only CENTER-BIND-SHUFFLE raises soup
MAE to `0.263678`, so the trained model **does** use the center binding
(`G_bind = 0.009341`, best-checkpoint `0.007225`), but that effect is just
below the preregistered `0.010` material threshold and does not translate into
a new ceiling.

Because the registered rule fires on `G_center < 0.010` **and**
`G_bind < 0.010`, the case is **D**. The binding estimate is honestly reported
as borderline: it is 93.4% of the material threshold, with large per-molecule
prediction movement, so the correct reading is *"a single incidence layer is
used but is not sufficient"* rather than *"center identity is ignored."*

## Execution integrity

* Formal compute: remote A100 **GPU1 only** (`physical_gpu_requested: 1`,
  `cuda`; GPU0 unused).
* Seed `0`; fixed internal `8000/2000` split, split seed `20260922`.
* Frozen base: exact TCCD-v2 PrototypeREL best checkpoint
  (`tracks/ksvd/results/tccd_v2/prototype_rel_seed0_best.pt`), SHA-256
  `093e0e46d4d5dab7e5558b807d993d1136df58d8f531d6b9b3621e8ab05c0f42`,
  verified before the run.
* Reused pair cache: exact TCCD-v5 cache
  (`tccd_v5_occurrence_pair_nonlinearity_v1`), metadata
  `official_test_loaded: false`; **not** regenerated.
* Pair set: all unordered distinct occurrence pairs `i<j`; no self-pairs; no
  sampling. `2,668,346` pairs, `231,664` occurrences, `5` relations,
  descriptor width `197`.
* Operator: `e_ij=phi(p_ij)` (`197->64->16`, `13,712` params),
  `q_i=sum_{j!=i} e_ij`, `u_i=rho([c_i,q_i])` (`80->64->16`, `6,224` params),
  `h_center=sum_i u_i`, `h=[h_base,h_center]`, head `Linear(10480,1)`
  (`10,481` params). Total trainable `30,417`.
* Initialization checksum `699e9854b9684761eb71613a0b647d82b0eb9e2d5b3461cac9e8616a49b64119`
  (deterministic, matched to the local smoke).
* Best epoch `78`, early stop at epoch `118`; Top-5 soup members
  `[78, 117, 83, 84, 73]`.
* Wall `269.8 s`; peak GPU memory `107.7 MB`.
* Official ZINC valid and test: never loaded (`official_test_loaded: false`).
* No baseline was re-run; only the single CENTER-COMP seed-0 model was trained.

## Gate 0

**PASS** on GPU1 (`results/center_comp_v0/gate0.json`, commit `cf87d1e`).

Data-free checks:

* parameter accounting `phi 13,712 / rho 6,224 / head 10,481 / total 30,417`;
* every unordered pair contributes exactly once to each of its two endpoint
  centers (max abs diff `0.0`);
* pair-swap descriptor invariance, pair-order invariance;
* batching invariance;
* node-relabel invariance of `h_center`;
* CENTER-BIND-SHUFFLE preserves the `{q_i}` multiset (max diff `0.0`),
  the `{c_i}` multiset (max diff `0.0`), and `h_base` (max diff `0.0`), and
  demonstrably changes the binding (`h_center` max diff `0.3103`);
* task gradient reaches `phi`, `rho`, and the reader; frozen local parameters
  are detached with no grad; the trainable parameter names are only
  `phi.*`, `rho.*`, `head.*`;
* official valid/test blocked.

Real-cache checks:

* frozen TCCD-v2 `C` re-derived from the checkpoint matches the reused cache
  (max abs diff `3.28e-07`);
* frozen `h_base` matches at relative diff `1.05e-07` (cross-device float32);
* pair descriptor identical to the exact TCCD-v5 tensor path (max diff `0.0`);
* cache metadata certifies the frozen checkpoint SHA and
  `official_test_loaded: false`.

## Frozen comparison references (read, not re-run)

| representation | Top-5 soup MAE | status |
|---|---:|---|
| TCCD-v2 BASE | `0.2783639132976532` | existing |
| TCCD-v5 POST | `0.2522333264350891` | existing |
| TCCD-v5 PRE | `0.2514181435108185` | existing |
| TCCD-v5 PRE-SHUFFLE | `0.25175556540489197` | existing eval-only |
| **CENTER-COMP** | **`0.2543369984`** | **new** |
| **CENTER-BIND-SHUFFLE** | **`0.2636783788`** | **new eval-only** |
| canonical GPU1 strong B-Full | `0.119818` | absolute scale reference |

## Main table

| model | existing/new | soup MAE | delta vs best existing | bind-shuffle delta |
|---|---|---:|---:|---:|
| TCCD-v5 PRE (best frozen composition) | existing | `0.2514181435` | `0.000000` | `0.0003374219` |
| TCCD-v5 POST | existing | `0.2522333264` | `-0.0008151829` | — |
| CENTER-COMP | new | `0.2543369984` | `-0.0029188549` | `0.0093413804` |
| CENTER-BIND-SHUFFLE | eval-only | `0.2636783788` | — | `0.000000` |
| canonical strong B-Full | existing | `0.119818` | `-0.1316001435` | — |

Derived:

```text
MAE_prev     = min(PRE, POST) = 0.2514181435108185
G_center     = 0.2514181435 - 0.2543370 = -0.0029188549
G_bind(soup) = 0.2636783788 - 0.2543370 =  0.0093413804
G_bind(best) = 0.2706928743 - 0.2634674 =  0.0072254321
gap_to_strong= 0.2543370 - 0.119818    =  0.134519
CENTER - BASE= 0.2783639133 - 0.2543370 =  0.0240269149
CENTER - POST= 0.2522333264 - 0.2543370 = -0.0021036720
```

The center branch adds capacity over BASE (`+0.02403`), but less than the
v5 POST/PRE gain (`+0.0261`), and it does not exceed the best existing frozen
composition. In the v5/v6 frame this is consistent with a coordinate/reader
effect rather than a new information channel.

## Mechanism table

Prediction change under CENTER-BIND-SHUFFLE, dev `n = 2000`:

| checkpoint | mean abs change | median | p90 | max | frac >= 0.01 | frac >= 0.05 | frac >= 0.10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| soup | `0.03652` | `0.01075` | `0.12055` | `0.33964` | `0.5125` | `0.2465` | `0.1415` |
| best | `0.03737` | `0.01066` | `0.12664` | `0.33749` | `0.5145` | `0.2435` | `0.1465` |

The intervention is not a no-op: half of the dev molecules move by at least
`0.01` and about a quarter by at least `0.05`. The aggregate MAE cost `0.009341`
is nonetheless below the preregistered material bound.

## Size stratification (read-only, no retraining)

Dev molecules sorted by occurrence count into tertiles:

| stratum | n range | n graphs | real MAE | shuffle MAE | `G_bind` |
|---|---|---:|---:|---:|---:|
| small | 9–21 | 667 | `0.23914` | `0.25804` | `0.018894` |
| medium | 21–25 | 667 | `0.19659` | `0.20493` | `0.008336` |
| large | 25–37 | 666 | `0.32739` | `0.32817` | `0.000781` |

Counter to the naive expectation that assembly complexity grows with graph
size, the binding effect is **strongest for small molecules and nearly absent
for large molecules**. Large graphs already have the worst real MAE (`0.327`),
and shuffling the center context changes essentially nothing there. This does
not support the idea that the large-graph shortfall is caused by losing
center identity under global contraction.

## Three questions

**Q1 — Does retaining pair-to-center incidence identity beat the existing v5
global pair representation materially?**

No. `G_center = -0.002919`: CENTER-COMP (`0.254337`) is slightly *worse* than
v5 PRE (`0.251418`). It is clearly better than BASE by `0.02403`, but v5/v6
already showed that this order of gain is recoverable from global
coordinate/reader geometry without new occurrence information.

**Q2 — Does the gain depend on correct center binding?**

The trained model clearly uses center binding: `G_bind(soup) = 0.009341`,
`G_bind(best) = 0.007225`, with `51%` of molecules moving by `>= 0.01` under
the shuffle. But `0.009341` is just below the preregistered `0.010` material
threshold, and the binding use does not convert into a better ceiling. Honest
answer: **binding is used, but at a sub-material, insufficient level.**

**Q3 — Even preserving this assembly layer, how far from ~0.12?**

`gap_to_strong = 0.134519`. The incidence layer closes none of the
`0.25 -> 0.12` gap.

## Scientific interpretation

1. **The single-layer incidence operator is not the missing ingredient.** The
   v4/v5/v6 lineage argued that global contraction might be discarding
   occurrence identity. CENTER-COMP directly preserves `(c_i, q_i)` binding
   before pooling; the result does not exceed the best frozen global
   composition. This weakens (does not formally refute) shared-center identity
   as the primary `0.25` bottleneck.
2. **Center binding is real but small.** A `0.0093` soup cost from a
   binding-only permutation is far from zero and is accompanied by large
   per-molecule movement. So the representation does carry binding information;
   it is just not enough, at one layer, to add material predictive signal over
   the existing global summaries.
3. **The effect is not complexity-scaling.** The binding effect decreases with
   graph size, opposite to an "assembly complexity" story. If anything, the
   large-graph error is a different problem than lost incidence.
4. **Consistency with v6.** The center branch gain over BASE (`0.02403`) is
   close to the v6 RECON-coordinate exposure gain (`0.02600`). Combined with
   the sub-material binding cost, the parsimonious reading is that this round
   again mostly re-exposes well-conditioned global structure, not a new
   occurrence-level channel.

Caveats:

* One seed, internal-dev only; no official-valid and no official-test read.
* `G_bind = 0.009341` is 93.4% of the material threshold. A second seed could
  in principle cross it, but no seed was authorized and none is run; the
  preregistered verdict is D.
* The `sum` pooling registered here scales with graph size; this is the
  preregistered operator, not a swept choice.

## Decision

**Outcome D.** Stop the single-layer pair-to-center incidence route. Do not add
a second incidence layer, do not deepen the reader, and do not attempt a GNN
rescue. A fuller assembly object or a local-code sufficiency question requires
a new preregistration.

## Not run (frozen stop)

No baseline rerun, no seed 1, no second incidence layer, no message passing /
GNN / Transformer, no reader sweep, no width/depth sweep, no new handcrafted
relation, no end-to-end prototype training, no official-valid, no official
test.

Evidence:

* `tracks/ksvd/notes/center_comp_v0_preregistration.md`
* `tracks/ksvd/results/center_comp_v0/gate0.json`
* `tracks/ksvd/results/center_comp_v0/stageA_seed0.json`
* `tracks/ksvd/results/center_comp_v0/stageA_decision.json`
* `tracks/ksvd/code/center_comp_v0.py`
* `tracks/ksvd/code/run_center_comp_v0.py`
* `tracks/ksvd/tests/test_center_comp_v0.py`
