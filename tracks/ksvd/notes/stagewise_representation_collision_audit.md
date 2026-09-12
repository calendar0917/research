# Compact-v4 Stagewise Representation Collision Audit

**Date:** 2026-09-12
**Scope:** analysis-only / zero-full-training bottleneck localisation audit
**Module:** `tracks/ksvd/experiments/luyin16/zinc_stagewise_representation_collision_audit.py`
**Results:** `tracks/ksvd/results/stagewise_representation_collision_audit/`
**Tests:** `tracks/ksvd/tests/test_stagewise_representation_collision_audit.py`
**Official valid:** not used for stage selection. **Official test:** never loaded.

**Verdict: `Case E — NO CLEAR STAGEWISE REPRESENTATION COLLISION`.**

No pre-registered real computation boundary of the compact-v4-smallhead model
shows a stable, reproducible *target-relevant* neighbourhood-geometry
degradation.  Three boundaries (patch encoder, pair encoder, unary graph
compression) show a material geometry **gain**; the centre update is
seed-unstable at sub-threshold magnitude; the pair graph compression shows a
stable **sub-threshold positive** direction that additionally fails the
collision-consistency requirement.  No architecture work is authorized.

---

## 1. Motivation

The compact-v4-smallhead candidate (82,115 params, `R = 302D`) reached
validation 0.145334 (seed0) / 0.138059 (seed1).  Across many audits a single
recurring question survived: *is the remaining error caused by a specific
computation stage destroying target-relevant representation geometry?*  Every
previous attempt answered this by adding a module (P1 learned composer, P2
relation refresh, cycle cells, covariance/triad/endpoint witnesses) and every
one returned a clean or sub-threshold NO-GO.  This audit changes the question
from "which new module helps?" to "where does the *existing* geometry first
degrade?" — without training anything and without reading the target until the
neighbourhoods are frozen.

## 2. Why local architectural patching was stopped

Existing evidence is binding:

* training insufficiency was already corrected (240-epoch regime);
* large graph-head capacity is not the bottleneck (head compression is
  non-inferior end to end);
* P1 learned relation composition — NO-GO;
* P2 one-shot relation refresh — sub-threshold NO-GO;
* persistent cycle cells — NO-GO;
* covariance / triad / endpoint witnesses — NO-GO;
* simple richer frozen readout — no stable gain;
* global topology — strong positive (out of scope);
* simple lookup low-rank donor — not supported.

Three consecutive clean local-repair NO-GOs (P1, P2, cell) made a fourth local
patch scientifically indefensible.  This audit instead localises the *first*
geometry degradation, if any, along the real graph.

## 3. Why the 82K small-head model is used

The 82,115-parameter compact-v4-smallhead model is the current clean candidate:
it removes 17,498 parameters (80.9 % of the head) while remaining
validation-non-inferior on both development seeds.  Its backbone is
structurally identical to the 99,613-parameter baseline, so the stagewise
geometry question is asked on the cheapest faithful model.  The 99.6K full-head
checkpoints are deliberately **not** mixed into the decision matrix: doing so
would turn this into a "head vs representation geometry" audit.

Primary backbones: `smallhead_seed0_selection_state.pt` and
`smallhead_seed1_selection_state.pt` (both 82,115 parameters, `R = 302D`).

## 4. Scope and test lock

* No new backbone training, architecture, P1/P2/cell rescue, pooling, attention,
  width/depth sweep, or error-subgroup mining.
* No new learned reader (no MLP/ridge/forest/neural set reader).
* Official **valid** was not used for stage selection.
* Official **test** was never loaded (no test MAE, prediction, subgroup, or
  checkpoint exists).
* Only the official-train deterministic `7200 / 800 / 2000` split is used.

## 5. Exact computation stages

Exported from the real instantiated model (not hand-reconstructed), using an
instrumented replica of `PatchPathModel.encode` that is bit-identical to the
original (`R` and `yhat` max abs diff `0.0`, see instrumentation integrity).

| code | key | kind | shape | real producer |
|---|---|---|---|---|
| S0 | `patch_encoder_input` | set | (n_patch, 170) | input to `patch_encoder` |
| S1 | `h0` | set | (n_patch, 48) | output of `patch_encoder` |
| S2 | `pair_encoder_input` | set | (n_pair, 64) | input to `pair_encoder` |
| S3 | `q0` | set | (n_pair, 16) | output of `pair_encoder` |
| S4 | `centre_update_input` | set | (n_patch, 213) | `[h0 ; A_i]` input to `center_update` |
| S5 | `h1` | set | (n_patch, 48) | `h0 + center_update(S4)` |
| S6 | `unary_fixed` | fixed | (97,) | `_pool_nodes` after update |
| S7 | `pair_fixed` | fixed | (165,) | `_pool_pairs` |
| S8 | `global` | fixed | (32,) | `global_encoder` |
| S9 | `topology` | fixed | (8,) | `topology_encoder` |
| S10 | `R` | fixed | (302,) | concat `[U;P;G;T]` |

Dimensions were re-derived from the code (patch_hidden 48, pair_hidden 16,
token_width 16, parent_width 8, shell_width 146, 5 distance buckets) and match
the expected 170/48/64/16/213/48/97/165/32/8/302 exactly.

## 6. Geometry-vs-information caveat

This audit can only establish that **neighbourhood geometry worsened**,
**target-relevant collisions increased**, or a stage is **implicated as a
representation-bottleneck candidate**.  It cannot and does not claim that
information is mathematically destroyed.  `kNN geometry degradation` is **not**
`information-theoretic insufficiency`.  No identical-representation collision
proof is offered.

## 7. Unlabeled Phase U protocol

Only molecular inputs are used.  For each checkpoint and each stage the
instrumented forward extracts the actual numeric tensors.  Pooled object
statistics (mean/variance) are fit **only** on the 7200 reference
(`adapter_fit`) graphs.  Projections, set sketches, pairwise distances and
nearest-neighbour manifests are all built before any target is read.  The
target is a separate Phase Y.

## 8. Set-aware projected-quantile sketch

Variable-size object sets (S0–S5) are summarised by a fixed,
target-independent, set-aware sketch:

1. per-feature normalisation using the 7200 reference objects
   (`(x-mu)/(sigma+1e-6)`);
2. `P = 24` deterministic Gaussian unit projections per stage
   (`base_seed = 1729`, stage seed `sha256(base_seed|stage_name)`);
3. 9 fixed quantiles `0.1..0.9` per projection = 216 dimensions;
4. plus `log1p(n)` = **217D**.

The empty set maps to 216 zeros plus `log1p(0)=0`; no NaN is produced.  Sketch
coordinates are then z-scored using only the 7200 reference sketches, and the
graph-to-graph distance is `||z_a - z_b||_2 / sqrt(217)`.  This is a
deterministic projected-quantile (sliced-distribution-style) sketch, **not** an
exact Wasserstein distance.

## 9. Nearest-neighbor manifest lock

For each checkpoint, stage, and query split, the 8 nearest reference molecules
are stored as `(query_graph_id, stage, seed, split, rank, reference_graph_id,
representation_distance)`.  The four aggregated `neighbors_seed{0,1}_{probe,
selection}.jsonl.gz` files plus 44 per-stage shards are written and SHA-256
hashed **before** the target is read.  `neighbor_manifest_hashes.json` records
the hashes and `neighbor_manifests_hash_locked_before_target_read: true`.

## 10. Target scoring Phase Y

Only after Phase U is locked are the official-train targets read.  For each
representation `Z` and query `i` with frozen neighbours `N_8(i;Z)`:

```
V_8(Z) = mean_i mean_{j in N_8(i;Z)} |y_i - y_j|.
```

Secondary diagnostics are the non-parametric `NN8` median predictor MAE
(diagnostic only, never a model benchmark) and the collision rate.  Inference
uses a deterministic paired bootstrap (B = 2000) over query molecules only.

## 11. Random-neighbor baseline

For each query split and checkpoint, 8 random reference molecules are chosen
with a fixed seed (target-independent).  The resulting `V_rand` (target-only)
is the disagreement scale:

| checkpoint | split | V_rand |
|---|---|---:|
| seed0 | probe | 2.0771 |
| seed0 | selection | 2.1222 |
| seed1 | probe | 2.1099 |
| seed1 | selection | 2.1117 |

## 12. Primary `V_8` / `eta` metric

```
eta(Z) = V_8(Z) / V_rand      (lower is better)
```

All degradation gates use `eta`, not raw distance.  The materiality threshold
is `delta_eta >= +0.02` (two percentage points of the random-neighbour
disagreement scale).  This is a pre-registered materiality threshold, not a
0.02-MAE claim; it was not changed after seeing results.

## 13. Collision metric

With reference pairs frozen at a fixed seed, the 80th percentile of
`|y_a - y_b|` is `tau_far = 3.3656`.  The neighbour collision rate is

```
C_8(Z) = P(|y_i - y_j| >= tau_far | j in N_8(i;Z)).
```

Only reference targets define `tau_far`; the rule was fixed before any
representation result.  Error-based collision mining (hardest molecules, worst
residuals, tails) was not performed.

## 14. Patch encoder contrast

`X_patch -> h0`.  The local patch encoder **substantially improves** geometry:

| checkpoint | split | eta(S0) | eta(S1) | delta_eta |
|---|---|---:|---:|---:|
| seed0 | probe | 0.5591 | 0.3663 | **-0.1927** |
| seed0 | selection | 0.5562 | 0.3690 | -0.1872 |
| seed1 | probe | 0.5783 | 0.3963 | **-0.1820** |
| seed1 | selection | 0.5768 | 0.3889 | -0.1879 |

Bootstrap 95 % CI excludes 0 on both seeds.  This is
`MATERIAL_GEOMETRY_GAIN`; the learned local encoder **creates useful geometry**
(the raw patch input is a much weaker representation).  Descriptive only — not
a licence to widen the encoder.

## 15. Pair encoder contrast

`X_pair -> q0`.  The relation encoder also improves geometry, more modestly:

| checkpoint | split | eta(S2) | eta(S3) | delta_eta |
|---|---|---:|---:|---:|
| seed0 | probe | 0.3492 | 0.3094 | **-0.0397** |
| seed0 | selection | 0.3418 | 0.3024 | -0.0394 |
| seed1 | probe | 0.3266 | 0.2930 | **-0.0336** |
| seed1 | selection | 0.3165 | 0.2727 | -0.0438 |

`MATERIAL_GEOMETRY_GAIN`.  Note the secondary collision rate moves the *other*
way on the probe (`+0.0066` seed0, `+0.0040` seed1), so even this gain is not
monotone across diagnostics.  No relation-encoder degradation is present.

## 16. Centre contextualization contrast

`X_centre -> h1`.  This is the only boundary with a seed direction conflict:

| checkpoint | split | eta(S4) | eta(S5) | delta_eta |
|---|---|---:|---:|---:|
| seed0 | probe | 0.3901 | 0.3971 | +0.0071 |
| seed0 | selection | 0.3755 | 0.3841 | +0.0086 |
| seed1 | probe | 0.3681 | 0.3566 | -0.0115 |
| seed1 | selection | 0.3566 | 0.3419 | -0.0147 |

Both seeds are far below the `+/-0.02` materiality band and the seed0 bootstrap
CI includes 0.  Status: `SEED_UNSTABLE` (sub-threshold).  The centre update is
geometry-neutral-to-mixed, **not** an implicated bottleneck.

## 17. Unary graph compression contrast

`set(h1) -> U_97`.  The current unary moment summary **improves** geometry
relative to the full contextualised patch multiset:

| checkpoint | split | eta(h1) | eta(U) | delta_eta |
|---|---|---:|---:|---:|
| seed0 | probe | 0.3971 | 0.3088 | **-0.0883** |
| seed0 | selection | 0.3841 | 0.2981 | -0.0860 |
| seed1 | probe | 0.3566 | 0.3040 | **-0.0526** |
| seed1 | selection | 0.3419 | 0.2916 | -0.0502 |

`MATERIAL_GEOMETRY_GAIN`.  No unary graph compression bottleneck.

## 18. Pair graph compression contrast

`set(q0) -> P_165`.  This is the only boundary with a stable positive
degradation direction — but it is **sub-threshold** and fails collision
consistency:

| checkpoint | split | eta(q0) | eta(P) | delta_eta | delta_collision |
|---|---|---:|---:|---:|---:|
| seed0 | probe | 0.3094 | 0.3208 | +0.0113 | -0.0109 |
| seed0 | selection | 0.3024 | 0.3139 | +0.0115 | -0.0052 |
| seed1 | probe | 0.2930 | 0.3362 | **+0.0432** | -0.0096 |
| seed1 | selection | 0.2727 | 0.3292 | +0.0565 | -0.0088 |

The pair moment summary does lose some target alignment relative to the raw
relation set, and on seed1 the probe `delta_eta` crosses `+0.02`.  However
seed0 is only `+0.0113` (< `+0.02`), so the two-seed material gate is **not**
met; and the collision rate moves in the opposite (better) direction on every
seed/split, violating the pre-registered collision-consistency clause.  Status:
`GEOMETRY_STABLE_OR_SMALL_EFFECT`, recorded as a stable sub-threshold signal.

## 19. Full R geometry

`R = [U;P;G;T]` is the best target-aligned representation measured:

| checkpoint | split | eta(R) | C_8(R) | NN8 MAE |
|---|---|---:|---:|---:|
| seed0 | probe | 0.2120 | 0.0039 | 0.2814 |
| seed0 | selection | 0.2128 | 0.0052 | 0.2900 |
| seed1 | probe | 0.2124 | 0.0042 | 0.2839 |
| seed1 | selection | 0.2187 | 0.0042 | 0.3084 |

`R` is reported purely as a final geometry anchor; `R`-without-topology was not
recomputed and topology is not re-reviewed (it already has independent strong
positive evidence).

## 20. Seed replication

Direction agreement across the two checkpoints:

| contrast | seed0 delta_eta | seed1 delta_eta | consistent |
|---|---:|---:|---|
| A patch encoder | -0.1927 | -0.1820 | yes |
| B pair encoder | -0.0397 | -0.0336 | yes |
| C centre update | +0.0071 | -0.0115 | **no** |
| D1 unary compression | -0.0883 | -0.0526 | yes |
| D2 pair compression | +0.0113 | +0.0432 | yes |

Only contrast C has a sign conflict, at sub-threshold magnitude.  The
bottleneck-localisation requirement (§37–§43) demands both seeds support the
same localization; no degradation localization exists, so this conflict does
not create a Case F verdict.

## 21. Split replication

The 800 `adapter_selection` replication agrees in direction with the 2000
`train_probe` for every contrast (A/B/D1 negative, D2 positive, C follows each
seed).  No `SPLIT_UNSTABLE` flag fires.

## 22. Projection / k robustness if triggered

No contrast reached the material degradation gate, so per the pre-registered
rule the second projection seed and the `k = 4, 16` robustness reruns were
**not** triggered.  `projection_robustness.json` and `k_robustness.json` are
written as `{"triggered": false, "reason": "no contrast reached the material
gate"}`.  No pseudo-results were fabricated.

## 23. Existing NO-GOs as constraints

The P1 learned composer, P2 one-shot relation refresh, persistent cycle cells,
covariance / triad / endpoint witnesses, richer frozen readout, and the
low-rank lookup donor all remain binding.  Because Case E authorizes no
architecture, these constraints are not tested against a new proposal here.

## 24. Bottleneck localization

Pre-registered priority is the **earliest stable material degradation** over
the largest post-hoc effect.  Ordering A → B → C → D1 → D2:

* A patch encoder: material **gain** (not degradation);
* B pair encoder: material **gain**;
* C centre update: seed-unstable, sub-threshold;
* D1 unary compression: material **gain**;
* D2 pair compression: stable sub-threshold positive, collision-inconsistent.

There is no `first_material_degradation`; the strongest descriptive degradation
(D2) is not material and not collision-consistent.

## 25. What is and is not proven

**Proven (official train only, two checkpoints):**

* the local patch encoder and relation encoder *create* target-relevant
  geometry rather than destroying it;
* the unary moment summary does not lose target alignment;
* the centre update is geometry-neutral-to-mixed and seed-unstable at
  sub-threshold magnitude;
* the pair moment summary has a small, stable, sub-threshold tendency to lose
  target alignment, but the secondary collision metric moves the other way;
* the final `R` is the best target-aligned representation.

**Not proven / not claimed:**

* that any stage destroys information in an information-theoretic sense;
* that `R` is optimal or that the freed parameters should be reallocated;
* that the pair-moment sub-threshold signal generalises beyond seeds 0/1;
* anything about official valid or official test;
* that a new architecture would help.

## 26. Top-1 hypothesis or no authorization

`top1_representation_hypothesis.json`:

```
authorized_for_design: false
implicated_stage: null
decision_case: Case E
next_full_training_budget: "not yet authorized"
```

No representation-level hypothesis is authorized.  The single sub-threshold
signal (pair graph compression, D2) is explicitly **not** authorized because it
fails the two-seed materiality gate and the collision-consistency clause.

## 27. Final verdict

**`Case E — NO CLEAR STAGEWISE REPRESENTATION COLLISION`.**

Target-relevant neighbourhood geometry does not reproducibly degrade at any
pre-registered computation boundary of compact-v4-smallhead.  The correct next
direction is therefore **not** another structural module.  It is to investigate
generalisation / regularisation, training stochasticity, scalar-target-mapping
geometry, and finite-data sample efficiency.  Official test remains locked; zero
full-backbone training was performed.

---

## Q1–Q20 pointer

Machine-readable answers: `results/stagewise_representation_collision_audit/answers_q1_q20.json`.
