# REPORT — ZINC G0 structure-semantics transfer audit v1

**Scope.** Local CPU only (≤8 threads, GPU=0).  Reuses the two frozen Full
soups `F_A`/`F_B` of `zinc-topology-crossfit-diagnostic-v1` (result `8134630`,
science `240bd2b`) and the frozen 8000/2000 outer split + A/B sub-folds.  No new
base training, no warm, no remote job, no optimizer/backward.  Official test was
never instantiated/loaded/evaluated; official-valid was not re-read/re-evaluated.
The two readout arrays `P`/`TP` are the previous round's fixed ExtraTrees
outputs; no tree was refit.

## Part A — corrected paired statistics

Fixed `paired_boot` (one shared resample per iteration) on the same arrays
(1000×, seed `20261003`, A/B averaged before MAE).  Full table in `ERRATA.md`.

| comparison | point | corrected 95 % CI |
|---|---:|---|
| TP over base (gain_base) | −0.026359 | [−0.033471, −0.019069] |
| TP over P (gain_topo) | +0.019743 | [+0.007380, +0.031683] |
| const over base | +0.000099 | [−0.000007, +0.000202] |

The correction tightens the intervals (two now exclude zero) but the frozen gate
still FAILS: `gain_base < 0.003`, `G0_worsening = +0.0315 > 0.001`, both per-fold
TP−base gains negative.  The P readout loses on 1247/2000 rows, TP on 1215/2000;
the worst 50 rows carry ~31 % (P) / ~23 % (TP) of the positive damage.  Dev row
1792 is the illustrative case: a meta neighbour with base prediction 2.1249 and
residual −17.67 is copied onto a dev row with base prediction 2.1254 and target
1.98 (error 0.077 → 5.99).  `squared_error`, `leaf=1`, and tree averaging remain
competing explanations; no trees were retrained to attribute.  `TP > P` is a
readout-pair advantage, not a proven usable topology gain.

## Part B — the single frozen class-median shrinkage correction

Rule: per base model, meta fold = the other sub-fold; `r = y − p_base_cal`;
exact raw graph-only T25 class key (round 1e-6); within a class, median residual
per canonical molecule group first, then median over groups `m_c`; `n_c` distinct
canonical meta molecules; `q_c = n_c/(n_c+5)·m_c` (prior strength **5 fixed**);
unseen class → 0; `Q = (p_A+q_A+p_B+q_B)/2`.  No prediction-value NN, no ridge,
no clipping, no dev selection.

| arm | dev MAE (2000) | G0 MAE (1926) | penalty −1 (65) | severe ≤−2 (9) |
|---|---:|---:|---:|---:|
| base | 0.150606 | 0.123065 | 0.210561 | 5.611470 |
| **Q** | **0.149411** | **0.122829** | **0.190731** | **5.539565** |
| const | 0.150507 | 0.122933 | 0.211170 | 5.613293 |

| gate clause | value | threshold | outcome |
|---|---:|---|---|
| gain Q vs base | +0.001195 | ≥0.003 | **FAIL** |
| gain Q vs const | +0.001096 | ≥0.003 | **FAIL** |
| per-fold Q vs own base | A +0.001238, B +0.000592 | ≥0 | pass |
| G0 worsening | **−0.000236** (improves) | ≤0.001 | pass |
| bootstrap Q−base | [−0.002140, −0.000140] | — | excludes 0 |
| bootstrap Q−const | [−0.002035, −0.000043] | — | excludes 0 |
| overall | | | **FAIL** |

The rule **protects G0** (slightly improves it) and gives a small, consistent,
non-zero net gain, but it is ~2.5× below the pre-registered purchase threshold
and below the const reference threshold.  The gain is concentrated in the
ring/penalty groups; G0 moves by −0.00024.

**Uncovered / severe.**  49/2000 dev rows (F_A) and 40/2000 (F_B) have no exact
meta T25 class; of the 9 severe rows, 5 (435, 464, 874, 1004, 1499) are
uncovered by **both** folds and receive `q = 0` — this scheme can do nothing for
them.  The covered severe rows (260, 898, 1009, 1535) get small class corrections
(dev 1009: 0.580 → 0.312).  Per-class support: 237/244 classes, 69/70 with
`n_c ≥ 5`.  Full per-class supports and per-row corrections are in
`part_b_class_median_shrinkage.json`.

## Part C — G0 combination-transfer audit

### C1 — real computation path and channel health

Structural code is `z = model.code(phi65) = [c~ ; alpha_res]` (width 32+1,
`common_dim = 1`, `s = 8`, `eps = 1e-8`); the support set is over the 32 residual
codes.  Real undirected edges are deduped from the `env_bond_u/v/type`
occurrences (bond type asserted consistent); the edge structural key is the
unordered pair of endpoint support masks, semantics = `env_bond_type`.  The node
key is the single support mask, semantics = `dict_atom`.

Frozen-forward channel health (`part_c_channel_health.json`):

| fold | node encoder out | edge encoder out | interpretation |
|---|---|---|---|
| A | per-dim std **0.0** (constant) | per-dim std 0.017 (varies) | node dead, edge alive |
| B | per-dim std **0.0** (constant) | per-dim std **5.7e-11** (constant) | node dead, edge collapsed |

Weight check: `W_A_S ≈ 1e-18`, `W_A_C = 0`, `node_encoder.0.weight = 0` in both
folds → the node channel is dead by construction; atom information still enters
via **Sem108** (`patch_cont[:, :108]`) and the anchor size2 block.  F_B's edge
channel (`W_E_S ≈ 1.4e-3`, `W_E_C ≈ 1.9e-6`) is also collapsed, so F_B's
structure channel is not load-bearing.  This is itself a cross-fold instability
of the structure-semantics channel.

### C2/C3 — coverage and exposure

Frequencies are counts of **distinct canonical training molecules** (a molecule
counts once per key).  Held-out = each model's own meta-fold G0 (3851 graphs) and
outer-dev G0 (1926 graphs), using that model's own base calibrated predictions.
Class fractions are molecule means; `jrm` = `joint_rare_marginals_seen`.

| fold | view | key | split | joint_seen | jrm | structure_rare | semantic_rare | both_rare | joint unseen |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| A | edge | residual | meta | 0.9985 | 0.0008 | 0.0007 | 0.0000 | 0.0000 | 0.0001 |
| A | edge | descriptor | meta | 0.9981 | 0.0011 | 0.0008 | 0.0000 | 0.0000 | 0.0002 |
| A | edge | residual | dev | 0.9986 | 0.0008 | 0.0006 | 0.0000 | 0.0000 | 0.0001 |
| B | edge | residual | meta | 0.9983 | 0.0011 | 0.0006 | 0.0000 | 0.0000 | 0.0003 |
| B | edge | descriptor | meta | 0.9980 | 0.0013 | 0.0007 | 0.0000 | 0.0000 | 0.0003 |
| A | node | residual | meta | 0.9962 | 0.0023 | 0.0015 | 0.0000 | 0.0000 | 0.0006 |
| B | node | descriptor | meta | 0.9909 | 0.0046 | 0.0044 | 0.0000 | 0.0000 | 0.0024 |

(All 16 fold×view×key×split cells are in `part_c_class_composition.json`; the
node view is the most "rare" and still only ~0.5 % `jrm`.)

**The high group is empty.**  The fixed high cut `≥0.25` selects **0 molecules**
in every one of the 16 cells; the low cut `≤0.05` selects 3717–3843.  Per the
pre-registration the thresholds are **not** changed.  Matching therefore returns
**0 pairs** everywhere (`part_c_matching.json`), so no matched
`MAE_high − MAE_low` and no combination-rare effect can be estimated.  The
maximum per-molecule `jrm` is 0.071–0.231 (node view), and only 8–134 molecules
(out of ~3850) even exceed 0.05.

**Why the proxy has no resolution.**  The structural input has collapsed:
1182 distinct `phi65` rows / 231,664 nodes; 202–212 distinct support masks; 169
distinct edge support pairs; 317 distinct edge joint keys.  With 4000 training
molecules, essentially every key is seen ≥5 times, so `joint_rare_marginals_seen`
— and even `structure_rare` — is almost unpopulated.  The same holds for the
pre-registered full-descriptor key, so this is not an artefact of the support
projection.  Coverage is therefore *saturated*, not the bottleneck; the audit
**cannot** attribute G0 error to combination rarity, and equally cannot clear
structure-semantics fusion (a saturated coarse key carries little information).

### Evidence grading (three levels, kept separate)

1. **Actual forward / code mechanism (solid):** node channel dead in both folds;
   F_B edge channel collapsed; `phi65` and the residual support are
   low-cardinality; exact replay of both soups (max |Δ| ≤ 1.1e-6, see
   `replay_verification.json`).
2. **Association on fixed unseen data (solid but uninformative):** class
   composition of held-out G0 is ~99 % `joint_seen`; the high group is empty, so
   no group contrast exists to test.
3. **Causal improvement conclusions (not available):** would require matched
   groups and a training intervention; neither is supported by this coverage
   proxy.

## Evidence boundaries

* This is a **coverage-proxy negative**, not a negative on structure-semantics
  fusion.  The pre-registered exposure cannot be tested because the high group
  is empty under a collapsed key.
* The F_B structure-channel collapse means F_B's coverage numbers describe a
  channel that does not carry weight; the two folds are not a clean replication.
* Node-view numbers are diagnostic only (dead channel); they do not show that
  reviving nodes would help.
* No official-valid or test score is produced or converted.
* The Part B gain is below the pre-registered purchase threshold; it is not a
  bought gain.

## Deliverables and missing items

Produced: `ERRATA.md`, `part_a_paired_bootstrap.json`,
`part_b_class_median_shrinkage.json`, `part_c_coverage.json`,
`part_c_matching.json`, `part_c_class_composition.json`,
`part_c_per_molecule.npz`, `part_c_channel_health.json`, `identity.json`,
`replay_verification.json`, scripts, `REPORT.md`, `DECISION.md`.
Missing/none: all checkpoints, prep blobs and fit objects were present; no
re-training was needed.  The matched-group table is empty **by construction**
(no high group), which is reported rather than patched.