# E2E-DictEnv-Clean-Mechanism-v1 — pre-registration

Round **E2E-DICTENV-CLEAN-MECHANISM-V1** · protocol `e2e_dictenv_clean_mechanism_v1`
· study `zinc-context-gap` (Workstream Z).  CPU-only, local, `official_test_loaded = false`.

This round continues the closed clarity audit
(`notes/e2e_dictenv_h1_clarity_audit_analysis.md`, commit `72b490a`) and is a
**mechanism-convergence round**, not an architecture search.  It answers four
questions:

* **Q1** is the C6 removal (graph-level atom/bond histogram + four dormant
  statistic blocks) stable across seeds in the matched CPU regime?
* **Q2** is the node branch really a fine per-atom `alpha_v <-> q_v`
  correspondence, or only a shell-level structure-marginal x chemistry-marginal
  interaction?  Is the edge branch really a specific structural-role <-> bond
  correspondence?
* **Q3** what is the minimal environment relation on the clean architecture?
* **Q4** on the final clean architecture, is the sparse tied-IHT dictionary
  coordinate actually better than a matched dense tied coordinate?

The round must not open a fifth question family, no matter how much compute is
left.

## 0. Hard constraints (frozen before any run)

1. **CPU only.** Every entry point sets `torch.set_num_threads(k)` and asserts
   `device.type == "cpu"`.  No CUDA, no `res` SSH, no remote runner.
   `CUDA_VISIBLE_DEVICES=""` is exported by the driver.
2. **No official ZINC test.** Only official train (10,000) and official valid
   (1,000).  `official_test_loaded = false` in every artifact.
3. **Frozen H1 protocol, not re-tuned.**  `decoder=h1`, `K=32`, `s=8`, IHT-10,
   `d_e=48`, SDB-v0 dictionary `sdb32`
   (sha256 `b0c5da98aee5795450927fcd76606281aca947448f77ab186e11de09b33dfecd`),
   `lambda_rec = 33.95873017865987`, Adam `lr=1e-3` / `wd=1e-5`, batch 128,
   gradient clip 5.0, no scheduler, 320 epochs, no early stop, fixed Top-5 soup
   by valid MAE, seeds govern `torch/numpy/random` init and the loader
   (`+91011` train shuffle, `+91012` eval shuffle).  Nothing in this list may
   change between arms of a comparison.
4. **Frozen P1 / P2-ABS implementations stay byte-identical.**  The audit core
   keeps its default behaviour (the only change is an optional
   `model_factory` keyword on `audit.train_cpu`, default `None`; the default
   path, parameter shapes, init order and batch order are unchanged and pinned
   by tests).
5. **Matched arms share initialisation and data order.**  Two arms with the same
   seed must hash to the same initial `state_dict` and consume the same batch
   sequence.  Pinned by tests.
6. **No hyper-parameter / capacity / feature search.**  Forbidden: new
   attention / message passing / recurrence / transformer, new MLP
   depth-width sweep, K / sparsity / IHT-step sweep, OMP, new global or motif
   or ring descriptors, ring features, LR / dropout / batch sweep, official
   test, 2^n combinatorial ablations.
7. **Distribution-preserving probes only for verdicts.**  Raw zero probes are
   OOD corruption and are reported only as cross-checks.  The frozen probe
   registry from the clarity audit is reused verbatim.

## 1. Definitions

### 1.1 `BASE`

The frozen H1 architecture with the identity forward (`mask=None`):
`e2e_dictenv_p2_abs.P2Model.forward`.

### 1.2 `C6` (the candidate clean baseline)

```text
global_context:   zero atom_histogram [30:58] + bond_histogram [58:62]
unary pool:       zero the count block (log1p |V_i|)
pair readout:     zero the per-bucket count block (log1p bucket pair count)
pair relation:    zero the log path-count coordinate (used index 14; raw 18)
```

Everything else unchanged.  The mask is defined by the set
`{atom_histogram, bond_histogram, relation:path_count, unary:count, pair:count}`;
the recorded previous-round JSON lists `bond_histogram` twice because the
previous C6 mask tuple was `GLOBAL_CHEMISTRY_GROUPS + ("bond_histogram",)` —
a duplicate that is idempotent in `_replace_grouped_columns`, so the semantics
is exactly the set above.  The new registry defines it as the set.

`C1` = only `{atom_histogram, bond_histogram}` (fallback arm, §4.4).

### 1.3 `CLEAN_BASE`

`C6` unless the Stage-A gate rejects it, in which case `C1` if its fallback
gate passes, otherwise `BASE`.  The decision and the evidence are written to
`results/e2e_dictenv_clean_mechanism_v1/stage_a_cpu_baseline/clean_base.json`
before any Stage-C/D/F training is launched on it.

### 1.4 `FINAL-CLEAN`

`CLEAN_BASE` plus every later simplification that passed its own gate in this
round (relation simplification from Stage D, node/edge independence from
Stage C).  The composition is frozen in
`results/e2e_dictenv_clean_mechanism_v1/final_clean.json` before Stage F.

### 1.5 Reused seed-0 artifacts

The previous round `e2e_dictenv_h1_clarity_audit/matched_cpu/` remains valid and
is reused, because it satisfies every match condition: same 10,000/1,000 split,
same 320 epochs, same optimizer/lambda/horizon, same Top-5 soup rule, same
paired seed-0 protocol, and a traceable commit.  **Reuse is re-verified at run
time**: loading `BASE_e320_soup_state.pt` / `C6_e320_soup_state.pt` and
re-evaluating with the current code must reproduce the recorded soup MAE
bit-for-bit (verified pre-registration: 0.143297936 and 0.128498517, abs diff
0.0, state sha256 match).  No seed-0 rerun is authorised for form's sake.

### 1.6 Reference regime numbers (context only, never a gate input)

Historical GPU H1 soup `0.123549` (outside this CPU regime).

## 2. Frozen seeds and arms

| stage | arm | seed(s) | epochs | notes |
|---|---|---|---|---|
| A | BASE | 0 (reused), 1, 2 | 320 | matched pair with C6 per seed |
| A | C6 | 0 (reused), 1, 2 | 320 | conditional seed 3 only if C6 gate = UNSTABLE |
| A-fallback | C1 | 0 (reused); 1, 2 only if C6 fails | 320 | nested removal fallback |
| C | CLEAN-NODE-INDEP | 0, then 1, 2 if gate | 320 | C6 mask + node independence null |
| C | CLEAN-EDGE-INDEP | 0, then 1, 2 if gate | 320 | C6 mask + edge independence null |
| D | CLEAN-REL-DIST-BOUNDARY | 0, then 1, 2 if gate | 320 | nested relation simplification |
| D | CLEAN-REL-DIST | screening only unless accepted | 20 adapt | distance only |
| F | FINAL-CLEAN-DENSE-TIED | 0, then 1, 2 if gate | 320 | identical mask / init / data order |
| F | FINAL-CLEAN-SPARSE | = CLEAN_BASE (+ adopted simplifications) | — | reused |
| E | shell 2-slot | conditional, only if §7 conditions hold | 20 adapt then 320 | default not run |

All Stage-A pairs use identical initialisation and data order per seed (same
seed for both arms).  All "warm-start" screening runs use a freshly
initialised Adam state and the matched continuation control from the previous
round (`M0`, 20 epochs: 0.123116) where a comparable control exists; for new
masked arms the control is re-run as `M0` on the arm's own base mask so that
the delta is always against the matched same-protocol continuation.

## 3. Stage A — Q1, multi-seed BASE vs C6

### 3.1 Procedure

1. verify the two reused seed-0 artifacts bit-for-bit (§1.5);
2. train `BASE seed1`, `C6 seed1`, `BASE seed2`, `C6 seed2` at 320 epochs;
3. per seed `delta_s = MAE(C6, s) - MAE(BASE, s)`;
4. report per-seed MAE, delta, mean / median / std / range;
5. apply the frozen gate (§3.2).

### 3.2 C6 gate (frozen)

```text
STRONG_SUPPORT      mean delta <  0    AND >= 2/3 seeds delta < 0    AND no seed delta > +0.010
CLEANNESS_SUPPORT   mean delta <= +0.003 AND >= 2/3 seeds delta <= +0.003 AND no seed delta > +0.010
UNSTABLE            seed range > 0.015, or one seed greatly better and one greatly worse
REJECT              mean delta > +0.003 and the majority of seeds are worse
```

Decision rule: `STRONG_SUPPORT` or `CLEANNESS_SUPPORT` → `CLEAN_BASE = C6`.
`UNSTABLE` → add exactly one further pair (`BASE seed3`, `C6 seed3`) and decide
on the 4-seed paired result with the same thresholds (no further seed
expansion).  `REJECT` → fall back to §4.4.  No small-difference
(0.001–0.003) decision may be made from warm-start runs alone.

### 3.3 Training curves

Every run stores the full per-epoch curve (`train_mae`, `train_rec`,
`valid_mae`, `D` norm) plus wall clock, best epoch, Top-5 members, parameter
count, init-state hash, batch-order digest and `official_test_loaded=false`.
This is required to distinguish "cannot recover" from "converges slower".

### 3.4 Fallback when C6 fails (§4.4)

If the C6 failure appears to come from the four later dormant-block removals
(path count / unary count / pair count), test `BASE vs C1` on seeds 1 and 2
instead (C1 seed 0 reused).  No 2^5 subset search.  If C1 passes its own copy of
the gate, `CLEAN_BASE = C1`; otherwise `CLEAN_BASE = BASE`.

## 4. Stage B — mechanism revalidation on the new checkpoints

For every available seed checkpoint of `BASE` and `CLEAN_BASE` (and the
fallback arms), the full frozen probe registry of the clarity audit is re-run
(`interventions()`, `interventions_fill()`, `interventions_graph_shuffle()`,
`interventions_readout_shuffle()`, `interventions_relation_shuffle()`), without
adding or removing any probe.

Primary read-out set (all distribution-preserving unless noted):

| id | probe | question |
|---|---|---|
| GS1 | graph chemistry row shuffle | graph chemistry shortcut sensitivity |
| EG2 | bond-histogram mean fill | dormant bond histogram |
| PS1 / PS2 | unary first / second row shuffle | moment readout |
| PS3 | unary count row shuffle | count dormancy |
| PS4 / PS5 | pair first / second row shuffle | moment readout |
| PS6 | pair count row shuffle | count dormancy |
| RS1 / RS2 / RS3 / RS4 / RS5 | relation group row shuffle | relation group sensitivity |
| N3 / N4 | node / edge assignment shuffle | correspondence vs marginal |
| N6 / N1 / N2 | code zero / node slot zero / edge role zero | content vs assignment |
| A2 / A4 / A5 | anchor marginal mean fill | retained chemistry |
| T1 | topology25 zero | retained bypass |
| EB2 / EB3 | gate off / pair projection zero | backend |

Each checkpoint is evaluated twice: (a) with its own training mask active
(mechanism of the clean forward) and (b) with the identity mask as base (what
the trained weights would do if the removed channels were re-opened).  (b) is a
diagnostic, not a verdict.

Question set (reported even if the answer is "no change"):
dictionary sensitivity retained? local environment sensitivity increased?
node correspondence changed? edge correspondence retained? relation sensitivity
retained? second-moment readout still important?  A performance gain must never
be auto-interpreted as a better mechanism.

## 5. Stage C — Q2, analytic independence null

### 5.1 Node null (per root i, shell s)

Current paired statistic (exactly the model computation):

```text
a_v = coord_v W_A^S ,  c_v = q_v W_A^C
U_pair[is] = (1/sqrt(D_A)) * sum_{v in P_is} (a_v * c_v)
```

Assignment-independent analytic null (the expectation of `U_pair` under a
uniform random pairing of the same multisets):

```text
U_indep[is] = (1 / (|P_is| sqrt(D_A))) * (sum_v a_v) * (sum_v c_v)
R_node[is] = U_pair[is] - U_indep[is]
```

Cases: `n = 0 -> 0`; `n = 1 -> U_pair == U_indep` exactly.
`shell 0` has one occurrence per root and therefore no assignment freedom; it is
reported separately and is never the primary evidence.

### 5.2 Edge null (per root i, shellpair p)

```text
s_e = g_uv W_E^S ,  c_e = b_uv W_E^C       (g_uv = [c_u+c_v ; |c_u-c_v| ; c_u*c_v])
E_pair[ip]  = (1/sqrt(D_E)) * sum_e (s_e * c_e)
E_indep[ip] = (1 / (|Q_ip| sqrt(D_E))) * (sum_e s_e) * (sum_e c_e)
R_edge[ip]  = E_pair[ip] - E_indep[ip]
```

Only the specific structural-role <-> bond-semantic correspondence is removed;
the role multiset, the bond-type multiset, the slot size and the group are all
preserved.

### 5.3 Stage C1 — frozen analytic diagnostics (no training)

On `CLEAN_BASE` checkpoints (all seeds) and the `BASE` seed-0 checkpoint:

* forward replacement `NODE_PAIR -> NODE_INDEP`, `EDGE_PAIR -> EDGE_INDEP`,
  `BOTH_INDEP`: valid MAE delta, prediction correlation, mean |prediction
  delta|;
* per-slot norms: `||U_pair||`, `||U_indep||`, `||R||`,
  `||R|| / max(||U_pair||, eps)`; counts `n = 0 / 1 / >=2`;
* per-shell node residual distribution and per-shellpair edge residual
  distribution: median, p90, p95 (never only the mean).

Goal: does the previous-round pattern (node fine correspondence weak/moderate,
edge fine correspondence strong) reproduce across seeds?  A cross-seed stable
pattern is the mechanism finding.

### 5.4 Stage C2 — trainable controls

From the `CLEAN_BASE` seed-0 soup checkpoint:

* 20-epoch warm-start adaptation for `CLEAN-NODE-INDEP` and
  `CLEAN-EDGE-INDEP`, each against a matched same-protocol continuation control
  (`M0` on the same mask).  Extend to 40 epochs once only if the last five
  evaluations show a monotone decreasing gap.  Screening only: not used to
  decide sub-0.003 differences.
* at least one matched from-scratch seed-0 320-epoch run for each of
  `CLEAN-NODE-INDEP` and `CLEAN-EDGE-INDEP` (they may run concurrently).

Extension gates (frozen):

```text
NODE-INDEP  seed0 delta <= +0.005 (or better)  -> extend seed1, seed2
            seed0 delta >  +0.015 and adaptation does not recover
                                                  -> stop, record NODE_FINE_CORRESPONDENCE_MATERIALLY_NEEDED
            otherwise                              -> decide from the curve + mechanism probes, declare the reason
EDGE-INDEP  seed0 delta >  +0.015               -> stop (only confirms the edge correspondence is needed)
            seed0 delta <= +0.010               -> extend seed1, seed2
```

`BOTH_INDEP` full training is **not** automatic; it is run only if both
single-side controls are close enough to `CLEAN_BASE` that the interaction
cannot be judged (both deltas `<= +0.005`).  Nothing else triggers it.

## 6. Stage D — Q3, relation simplification

### 6.1 Provenance

`relation_groups.json` is regenerated from the upstream builder
(`zinc_long_range_proxy` / P1 relation assembly) with the same per-coordinate
provenance checks as the clarity audit: distance one-hot rows sum to 1, bucket
argmax identity, log-distance match, overlap block, binary boundary indicator,
log path count.  No semantic group may be guessed from memory.  Since C6
already removes `path_count`, it is never re-added.

### 6.2 Nested candidates (no Cartesian sweep)

```text
REL-FULL-CLEAN      = CLEAN_BASE relation (distance + overlap + boundary)
REL-DIST-BOUNDARY   = zero overlap; keep distance + boundary
REL-DIST            = zero overlap + boundary; keep distance only
```

### 6.3 Screening and training

20-epoch warm-start adaptation (extend to 40 once only for a monotone recovery
trend) for `REL-DIST` and `REL-DIST-BOUNDARY` from the `CLEAN_BASE` seed-0 soup,
against the matched continuation.  If `REL-DIST` is clearly worse (> +0.02 and
not recovering) it is dropped.  The simplest surviving candidate gets one
seed-0 320-epoch run.

Gate for a 320-epoch relation candidate:

```text
delta <= +0.005 -> extend seed1, seed2 (adopt if seed-mean delta <= +0.005)
delta >  +0.015 -> stop
```

The acceptable final answer is "distance + boundary (+ overlap if required)".
Distance-only is not forced if the data reject it.

## 7. Stage E — shell simplification (conditional; default NOT run)

Only if (1) `NODE-INDEP` is supported, (2) the relation level is settled, and
(3) CPU budget remains, the single comparison `3-shell current` vs
`root/non-root 2-slot` may be screened for 20 epochs and only full-trained if
the 2-slot arm is close.  No other shell radius / merge / shellpair experiment
is authorised.  Edge shellpair is untouched.

## 8. Stage F — Q4, dictionary specificity on the final clean architecture

Primary control: the P1 matched **DenseTied** operator, reused unchanged:
`DenseTiedModel.code(phi) = phi @ Dbar`, `Dbar = normalized(D)`, same `D`,
same parameter count (97,487), same mask, same downstream architecture, same
initialisation policy, same data order, same optimizer, same horizon, same
readout.  The only difference is the coding operator.

```text
G_dict = MAE(FINAL-CLEAN-DENSE-TIED) - MAE(FINAL-CLEAN-SPARSE)
```

Seed 0 of `FINAL-CLEAN-SPARSE` already exists (it is `CLEAN_BASE` seed 0 plus
adopted simplifications); it is not rerun.

```text
G_dict >= +0.003 (dense worse) -> extend DenseTied seed1, seed2; paired verification
otherwise                      -> record DICTIONARY_COORDINATE_LOAD_BEARING_BUT_
                                  SPARSE_DICTIONARY_SPECIFICITY_NOT_ESTABLISHED
```

PCA32 is not a default control and is not run unless DenseTied and Sparse are
extremely close and an existing PCA32 protocol can be reused unambiguously.

## 9. Metrics recorded besides MAE

* **node / edge**: paired norm, independence norm, correspondence-residual
  norm, residual/paired ratio, counts, per-shell and per-shellpair statistics
  (median / p90 / p95).
* **dictionary**: active atoms, effective atoms, coordinate sparsity
  (mean nonzeros), reconstruction error, `D` movement (Frobenius from init),
  task-gradient norm on `D`, per-atom usage.
* **training**: full curve, wall clock, seconds/epoch, peak RSS, init-state
  hash, batch-order digest, soup members and member MAEs.

## 10. Statistics discipline

2–4 seeds: report mean, std, median, per-seed paired delta.  No p-values or
significance claims.  The object of interest is effect magnitude, direction
consistency and paired reproducibility.  Differences below ~0.003 are never
decided by short runs.

## 11. Early stopping discipline

Allowed early stop: numerical failure; loss not decreasing; catastrophic
degradation (`> +0.02` vs the matched control with no recovery trend); the
scientific question is already clearly answered.  Not allowed: deciding
0.001–0.003-scale differences from short runs.

## 12. Forbidden in this round

GPU / CUDA / SSH / remote runner; A1 attributed dictionary; joint
structure+attribute dictionary; K64 / sparsity sweep / dictionary capacity
sweep; IHT-step or OMP training replacement; new attention / message passing /
recurrence / transformer; new MLP depth-width / LR / dropout / batch-size
sweeps; new global descriptors / motif / ring features; official ZINC test;
large combinatorial ablation; optimising anything after seeing a result.

## 13. Artifacts and provenance

```
results/e2e_dictenv_clean_mechanism_v1/
    runtime_budget.json
    stage_a_cpu_baseline/        (per-run JSON + states, matched_summary.json, gate.json, clean_base.json)
    stage_b_mechanism/           (probe tables per checkpoint)
    stage_c_independence/        (frozen diagnostics, norm distributions, train controls)
    stage_d_relation/            (relation_groups.json, screen, train)
    stage_f_dictionary_specificity/
    node_edge_independence_diagnostics.json
    final_clean.json
    REPORT.md
    DECISION.md
```

Every run record: `commit`, `device=cpu`, `threads`, `seed`, candidate/spec,
parameters (97,487), epochs, soup members and MAE, wall clock,
`official_test_loaded=false`, init-state hash, batch-order digest.

## 14. Tests (focused)

`tracks/ksvd/tests/test_e2e_dictenv_clean_mechanism_v1.py` covers at least:
BASE identity against the frozen module; exact C6 feature contract; node
analytic null toy example; node `n=0` / `n=1` exact `paired == indep`; node
permutation invariance; edge analytic null toy example; edge `n=0` / `n=1`;
edge endpoint-symmetry preservation; relation coordinate groups and allowed
coordinates; DenseTied reproduction of the P1 definition (parameter equality
and coding identity on a fixed `D`); default `CleanMechModel` bit-identical to
`AuditModel`; CPU-only guard; official-test blocker; matched-init contract for
the new arms; `train_cpu` default behaviour unchanged.

## 15. Language discipline

* independence-null result -> "the per-node alpha<->atom correspondence is/is
  not materially needed"; never "the structure-semantic fusion is unimportant"
  when the null performs the same.
* edge null worse -> "the edge branch contains assignment-specific
  structure<->bond information".
* frozen probes on one checkpoint -> "the model is sensitive/insensitive";
  trainable controls -> "recoverable / not recoverable under the tested CPU
  budget"; matched retraining -> "removable at limited/material cost under the
  tested CPU protocol".
* no cross-regime (CPU vs historical GPU) comparisons; C6 must never be
  described as beating the historical GPU H1.
