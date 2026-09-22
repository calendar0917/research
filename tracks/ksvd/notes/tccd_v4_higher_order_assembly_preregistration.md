# Preregistration — TCCD-v4: Higher-Order Assembly Sufficiency Audit

Round name: **TCCD-v4** (*Higher-Order Assembly Sufficiency Audit*).
Study: `zinc-context-gap`.
Status: frozen before formal experiments.

## 1. Question

This is a strict **assembly-only** round. TCCD-v2 established that a task-learned
local prototype vocabulary plus assignment-sensitive relation contractions is
useful, but its absolute result remains far above the canonical GPU1 reference.
TCCD-v3b then showed that replacing the weak local state with a stronger frozen
local representation did not improve the TCCD-v2 ceiling. The registered
question is therefore:

> Does one parameter-free, normalized **two-hop assembly relation** add
> assignment-sensitive predictive information beyond the frozen TCCD-v2
> relation set, or is early graph-level contraction already too lossy?

The only new graph relation is the full normalized two-hop walk moment
`M2 = C^T S^2 C`.

## 2. Frozen lineage and execution discipline

Base: exact TCCD-v2 PrototypeREL path, not TCCD-v3b.

* Base implementation / checkpoint provenance: TCCD-v2 internal Gate-A commit
  `69a985a4ea3eb184c96c8ddad3857c4e87ee1dda`.
* Reused local encoder: exact TCCD-v2 `714 -> 64` trainable linear local map,
  initialized from the frozen TCCD-v1 dense artifact.
* Reused trained PrototypeREL checkpoint: `tracks/ksvd/results/tccd_v2/prototype_rel_seed0_best.pt`,
  SHA-256 `093e0e46d4d5dab7e5558b807d993d1136df58d8f531d6b9b3621e8ab05c0f42`.
  The corresponding TCCD-v2 Top-5 soup checkpoint is
  `tracks/ksvd/results/tccd_v2/prototype_rel_seed0_soup.pt`, SHA-256
  `e08680327b575771a7099ec69cfbd573b00c2a10fa3d0b5dc967f32f9a57c475`.
* Prototype vocabulary: `K=64`, latent width `d=64`, normalized Gaussian
  initialization with `PROTO_INIT_SEED = 20260922`.
* Temperature: exact TCCD-v2 global scalar
  `tau = 0.05 + 0.95 * sigmoid(a)`, initialized at `0.20`.
* Existing relation set: unchanged TCCD-v2 exact set — overlap/incidence,
  three native bond relations, and global relative-position relation.
* Reader: exact TCCD-v2 lightweight linear head, Adam optimizer, batch `32`,
  learning rate `1e-3`, weight decay `1e-5`, gradient clip `5`, at most `240`
  epochs, patience `40`, Top-5 checkpoint soup.
* Regularizers: exact TCCD-v2 entropy and balance regularizers; no retuning.
* Split: internal fixed split seed `20260922`, `8000` train / `2000` dev.
* Primary seed: `0`. Seed `1` is allowed only by the registered ambiguous gates.
* Formal compute: remote **GPU1 only**. GPU0 is forbidden.
* Official test: blocked for the entire round and never loaded.
* No K-SVD refit, OMP/IHT, local-encoder change, prototype-count change,
  temperature/regularizer change, reader sweep, relation redesign, GNN,
  Transformer, handcrafted cycle/global features, shortest-distance-2 matrix,
  walk-order sweep, or three-hop run before a two-hop pass.

Historical TCCD-v2 internal references, reused only for matched comparison:

* PrototypeREL best internal-dev MAE: `0.2862437069416046`.
* PrototypeREL Top-5 soup internal-dev MAE: `0.26635152101516724`.
* Corrected official-valid reference: best `0.287337`, soup `0.261988`.
* Canonical strong GPU1 reference: `0.119818`.

The official-valid values are not used for architecture selection in this round.

## 3. New relation: normalized two-hop walk operator

For each molecule, construct the native **untyped atom adjacency** `A` from
native bonds, with no self-loop:

```text
A[i,j] = 1 iff atoms i and j share a native bond; A[i,i] = 0.
D[i,i] = sum_j A[i,j].
S = D^(-1/2) A D^(-1/2),
```

where zero inverse degree is used for isolated nodes. The only new relation is

```text
R2 = S @ S
M2 = C.T @ R2 @ C.
```

`R2` is kept in full, including its diagonal. Return walks, triangle/common-
neighbor terms, and degree-normalized walk multiplicity are not removed or
hand-filtered. The implementation uses the same symmetric upper-triangle
vectorization and moment normalization conventions as TCCD-v2.

This round does **not** use `1[shortest_distance(i,j)=2]`.

## 4. Matched topology-mismatch control

The fixed permutation seed is `20260922`. For every graph `G`, generate one
persistent deterministic graph-specific permutation using only `(seed, graph
index, n)` and no target or labels. For `n < 2`, use identity. The same
permutation is reused for train/dev artifacts and all evaluation interventions.

```text
C_pi = P_pi C
M2_mis = C_pi.T @ S^2 @ C_pi.
```

All existing TCCD-v2 relation moments continue to use the real `C`. Only the
new two-hop block is replaced by `M2_mis`. Thus REAL-2 and MIS-2 have identical
feature width, reader parameter count, `S^2`, prototype multiset, and base
relations; only prototype-to-two-hop-topology alignment is destroyed.

## 5. Gate 0 — correctness, before formal runs

Gate 0 must pass all of the following without official-test access:

1. **Permutation invariance:** jointly relabel molecule nodes, `C`, and `S^2`;
   `C.T @ S^2 @ C` and predictions remain invariant.
2. **Batching invariance:** single-graph and batched two-hop computation match.
3. **Exact algebra:** on a small synthetic graph, implementation `R2` equals
   direct `S @ S` to numerical tolerance.
4. **Two-hop sensitivity:** two graphs with closely matched one-hop degree and
   node count but different common-neighbor/two-hop structure produce different
   `C.T @ S^2 @ C`.
5. **Mismatch intervention:** on a nontrivial graph, REAL-2 and MIS-2 moments
   differ materially.
6. **No target leakage:** permutation generation and `R2` construction read no
   `y` or target-derived field.
7. **Official test blocked:** explicit provenance remains false.

Gate 0 FAIL => STOP. No Stage A, Stage B, or three-hop work is authorized.

## 6. Stage A — frozen-representation screen

Use the exact trained TCCD-v2 PrototypeREL checkpoint from the internal matched
run. Freeze local encoder, prototypes, and temperature. Extract `C_G` once per
graph and cache:

* `BASE = h_v2`;
* `REAL-2 = [h_v2, vec_sym(C.T @ S^2 @ C)]`;
* `MIS-2 = [h_v2, vec_sym(C_pi.T @ S^2 @ C_pi)]`.

All three arms use one registered lightweight linear reader family and the same
optimizer/stopping protocol. BASE has its naturally smaller feature width;
REAL-2 and MIS-2 have exactly the same width and reader parameters. The primary
comparison is REAL-2 versus MIS-2; REAL-2 versus BASE is secondary.

Stage A uses only internal `8000/2000` train/dev. Official-valid architecture
selection is forbidden.

Define:

```text
delta_topo = MAE_MIS2 - MAE_REAL2
delta_add  = MAE_BASE  - MAE_REAL2.
```

### Stage A decision

* **STRONG PASS:** both deltas are at least `0.015`; authorize Stage B.
* **FAIL:** either delta is below `0.005`; STOP and forbid `S^3`.
* **AMBIGUOUS:** neither is below `0.005`, but at least one is in
  `[0.005, 0.015)`; run one paired reader seed only. The two-seed means must
  satisfy both deltas `>= 0.010` to authorize Stage B; otherwise STOP.

## 7. Stage B — end-to-end two-hop TCCD

Only after Stage-A authorization. The model is exactly TCCD-v2 except for one
new graph feature block:

```text
h_v4 = [h_v2, vec_sym(C.T @ S^2 @ C)].
```

Task gradients may pass through `M2 -> C -> prototypes/local encoder`. No other
architecture or optimization change is allowed.

The TCCD-v2 internal matched baseline is reused whenever the code path remains
equivalent. Its registered values are best `0.2862437069416046` and soup
`0.26635152101516724`. A single matched BASE seed-0 rerun is allowed only if
code drift prevents exact equivalence; no baseline repeats are otherwise run.

Primary metric:

```text
delta_e2e = MAE_soup,base - MAE_soup,+2hop.
```

Best checkpoint is reported but does not override the soup-first decision.

### Stage B decision

* **STRONG HIGHER-ORDER SIGNAL:** `delta_e2e >= 0.020`; PASS.
* **FAIL:** `delta_e2e < 0.005`; STOP and forbid `S^3`.
* **AMBIGUOUS:** `[0.005, 0.020)`; run one paired seed only. The two-seed soup
  mean must be `>= 0.010` to PASS; otherwise STOP.

Mechanistic evaluation-only intervention on the trained +2hop model:
replace only the new two-hop block with `M2_mis`, keep all base relations real,
and report

```text
delta_intervention = MAE_mismatch - MAE_real.
```

If `delta_intervention < 0.005`, the result cannot claim that the model uses
assignment-sensitive two-hop information; it may only be a feature-width or
reader-capacity effect.

## 8. Conditional Stage C — one-shot three-hop

Stage C is authorized only after a Stage-B STRONG PASS or paired PASS. It runs
one seed `0` only and adds exactly

```text
R3 = S^3, M3 = C.T @ S^3 @ C,
h3 = [h_v2, M2, M3].
```

No change to `S` normalization, `K`, reader, or relation set is allowed. Compare
`+2hop` against `+2hop+3hop`. If Top-5 soup improvement is at least `0.010`,
record that assembly order beyond two hops remains useful; otherwise mark order
expansion saturated and stop. `S^4+` is forbidden.

## 9. Official-valid authorization

Do not run the full official train/valid fit merely because an internal gain is
small. A single official train `10000` -> official valid `1000` GPU1 run is
authorized only if either:

* Route A: final two-hop internal soup `<= 0.24`; or
* Route B: relative to TCCD-v2 internal soup, soup improvement `>= 0.030`.

If Stage C is run, use the final frozen architecture. Official test remains
blocked even after an authorized official-valid run.

## 10. Vocabulary diagnostics

For every trained prototype model that reaches diagnostics, report:

* active prototypes and dead prototypes;
* effective prototype count;
* top-8 assignment mass;
* global usage entropy;
* learned temperature;
* semantic coherence summary.

A performance gain with collapsed vocabulary cannot be called reusable
dictionary success.

## 11. Absolute interpretation

Use the fixed references only for interpretation:

* `soup <= 0.20`: major rescue;
* `0.20 < soup <= 0.24`: partial rescue;
* `0.24 < soup < 0.262`: small but mechanistically meaningful gain;
* `soup >= 0.262`: no practical absolute improvement over the TCCD-v2 soup.

If two-hop fails, the frozen conclusion is not that topology is irrelevant. The
claim is narrower:

> adding a normalized two-hop aggregated moment does not materially overcome
the TCCD ceiling; early occurrence aggregation may be the leading bottleneck.

Only a successful two-hop result can authorize testing whether higher-order
assembly algebra is the missing ingredient.

## 12. Durable records

Required tracked records after local analysis:

* this preregistration;
* Gate 0 result and targeted tests;
* frozen Stage-A result;
* Stage-B result if authorized;
* Stage-C result only if authorized;
* analysis note, claim, decision, and `tracks/ksvd/STATE.yaml` update.

Every formal result records local/remote commits, GPU1, seed, split, runtime,
exact `S`/`S^2` normalization, fixed permutation generation, BASE/REAL2/MIS2,
end-to-end soup, intervention, vocabulary diagnostics, and GO/STOP. No result
may load official test data.
