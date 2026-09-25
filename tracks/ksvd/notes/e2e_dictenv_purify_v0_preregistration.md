# E2E-DictEnv-Purify-v0 — pre-registration (frozen before implementation and before any run)

Round **E2E-DictEnv-Purify-v0** · protocol `e2e_dictenv_purify_v0` · study
`zinc-context-gap`.

Architecture audit (written first, same commit range):
[`e2e_dictenv_purify_v0_architecture_audit.md`](e2e_dictenv_purify_v0_architecture_audit.md).
Amendment: [`e2e_dictenv_purify_v0_amendment_a1.md`](e2e_dictenv_purify_v0_amendment_a1.md)
(equivalence-gate execution regime; rules unchanged).

This is the frozen specification.  It is written **before** any refactor,
implementation, cache build, training or GPU use.  Any deviation requires a
written amendment here before the affected run.

---

## 1. The single scientific question

> Can the 60 raw-chemistry dimensions of the 62-D local anchor `p_i` (root atom
> identity, patch atom marginal, patch bond marginal) be absorbed by one
> unified attributed structural measure — a fixed constant structural channel
> concatenated with the sparse structural coordinate, valued by the existing
> atom/bond attribute matrices — without materially losing predictive
> performance under the frozen H1 architecture?

```text
structure defines the basis
-> attributes value the basis
-> static composition builds the molecule
```

This round is **not** a performance chase, **not** a feature addition, and
**not** a rescue of the attributed shared-dictionary route.

---

## 2. Hard boundaries inherited from prior frozen verdicts

Closed by A2 and **not reopened**:

```text
433-D attributed dictionary (REAL / INDEP)
K64 rescue, s12 rescue
attributed sparse code formation
shared attributed dictionary
```

Frozen principle: **the sparse code is pure topology only.**

Frozen by §21-§24 of the round order and stated here as hard constraints:

```text
pair projection / 15-D pure-topology relation / distance buckets /
distance gate / pair encoder / pair pooling   -> UNCHANGED
global topology features (cycle / hinge / global structural statistics) -> UNCHANGED
reader (width, depth, activation, no normalization, no dropout, no attention) -> UNCHANGED
phi65 / K=32 / s=8 / IHT=10 / dictionary init / dictionary trainability /
reconstruction objective / lambda -> UNCHANGED
```

---

## 3. Reference (canonical) and the Stage-0 semantic contract

Reference = the H1 config of E2E-DictEnv-P2-ABS:

```text
decoder h1, d_e 48, K 32, s 8, IHT 10, lambda_rec 33.95873017865987, horizon 320
params 97,487
historical context only: Top-5 soup valid MAE 0.12354862861608853
```

Stage 0 must reorganize the reference code into four concept modules without
changing any mathematics:

```text
StructuralBasis            phi65 -> D -> alpha (K=32, s=8, IHT=10, trainable D,
                           reconstruction loss unchanged, topology only)
AttributedLocalMeasure     (structure coordinate x atom attribute),
                           (structural edge relation x bond attribute)
StaticComposer             local environment pooling, pair relation, distance
                           bucket, pair composition, graph aggregation, global
                           structural invariants, global zeroth-order
                           attributed measure
PropertyReader             final prediction only
```

`GlobalStructuralInvariant` (`global_context[0:30]`) and
`GlobalZerothOrderAttributeMeasure` (`global_context[30:62]`) must be named and
documented as two distinct semantic objects; their *encoder* stays fused
(one 62->32->32 MLP) because splitting it would break Stage-0 equivalence and
would confound the single purification principle.  No chemistry is deleted from
the global path this round.

### 3.1 Stage-0 equivalence requirement (blocking)

The refactored reference implementation must load the reference checkpoint and
reproduce the old implementation on the same inputs.  Required comparison
points: `alpha`, node slots, edge slots, pair representation, graph
representation, prediction.

```text
prediction max |old - new| <= 1e-6      (target: bit-identical)
```

If this fails, architecture purification stops; the equivalence bug is fixed
first.  No training may start on a non-equivalent refactor.

---

## 4. The one authorized architecture change

```text
beta_v    = [1 ; alpha_v]                 in R^33   (node)
u_v       = ((beta_v W_A_S) . (q_v W_A_C)) / sqrt(96)      W_A_S: 33 x 96
            -> index_add over the 3 shells (unchanged routing)

gamma_uv  = [1 ; g_uv]                    in R^97   (edge)
g_uv      = [alpha_u+alpha_v ; |alpha_u-alpha_v| ; alpha_u o alpha_v] in R^96
u_uv      = ((gamma_uv W_E_S) . (b_uv W_E_C)) / sqrt(48)   W_E_S: 97 x 48
            -> index_add over the 6 shellpair slots (unchanged routing)
```

Properties of the constant channel, frozen:

```text
value fixed at 1
not a dictionary atom
not part of alpha
not reconstructed
not counted toward the top-8 sparsity constraint
not trainable, no parameters of its own
```

### 4.1 Local anchor after purification

The purified candidate's anchor is **2-D local structural size only**:

```text
[log1p|V_i|, log1p|E_i|]  standardized by the train-fit per-coordinate
statistics of those same two coordinates (the existing reference anchor stats,
size slice indices 60:62)
```

Root atom identity, patch atom marginal and patch bond marginal are removed from
the anchor.  The anchor encoder therefore becomes `2 -> 32 -> 32`; its output
width stays 32 and the fusion input width stays `32 + 3*48 + 6*32 = 368`.

### 4.2 Frozen widths (no recalibration of capacity)

```text
node slot width 96 (3 shells)     unchanged
edge slot width 48 (6 shellpairs) unchanged
anchor output 32                  unchanged
fusion 368 -> 128 -> 48           unchanged
pair / relation / global / topology / reader    unchanged
```

### 4.3 Honest boundary (must be quoted in the result)

> The purification candidate changes the organization of zeroth-order chemistry
> and may expose it through the existing local routing structure; this is an
> intentional representation consolidation, not a claim of byte-equivalent
> feature content.

Any performance change is therefore **not** to be described as "just deleting a
branch".

### 4.4 Parameter rule

```text
purified total trainable params <= reference total   (97,487)
report: reference total, purified total, absolute delta, percent delta
```

Computed ledger: `97,487 -> 95,711` (`-1,776`, `-1.822 %`).  If the measured
candidate count is larger than the reference, stop and fix the implementation;
capacity must not be added to preserve performance.

---

## 5. Initialization fairness (formal runs)

For the formal seed-0 pair, and later the seed-1 pair:

```text
same seed
same shared parameter initialization   (max_abs_diff = 0 on every same-shape
                                        shared tensor after construction)
same dictionary initialization
same batches
same optimizer / lr / weight decay / clipping
same training order
same horizon
same Top-5 soup construction
same physical GPU (GPU1)
```

The candidate-only constant rows (row 0 of `W_A_S` and of `W_E_S`) are
initialized deterministically by a separate generator at the same `kaiming_uniform_(a=sqrt(5))`
scale as the shared rows, and never displace or perturb a shared tensor.  The
existing trained reference checkpoint is **not** loaded to warm-start the
candidate; both arms train from the matched initial state.

---

## 6. Stage 0 outputs (before the purified candidate exists)

```text
notes/e2e_dictenv_purify_v0_architecture_audit.md
notes/e2e_dictenv_purify_v0_preregistration.md
results/e2e_dictenv_purify_v0/information_flow.json
results/e2e_dictenv_purify_v0/parameter_ledger_reference.json
results/e2e_dictenv_purify_v0/semantic_refactor_equivalence.json
```

plus targeted tests.  Stage 0 is committed before the purified candidate is
implemented.

---

## 7. Local targeted tests (required, before any GPU use)

```text
semantic refactor exact equivalence (old vs refactored reference)
constant channel not part of alpha
constant channel not counted toward sparsity
max ||alpha||_0 <= 8
dictionary reconstruction unchanged definition
chemistry relabel leaves phi/alpha unchanged
node permutation invariance
edge endpoint symmetry
graph relabel invariance
local anchor chemistry truly absent
only size remains in the local structural scalar slot
global chemistry unchanged
global topology unchanged
pair relation unchanged
reader unchanged
shared init identity (max_abs_diff = 0)
parameter count (ledger == actual == <= reference)
official test blocker
GPU device explicitness
```

Targeted pytest only; the full suite is not run by default.

---

## 8. GPU discipline

```text
physical GPU1 only
no GPU0, no CUDA_VISIBLE_DEVICES=0, no DDP, no two GPUs, no GPU0/GPU1 parallel
launch form: bash scripts/launch_remote.sh 1 <tag> python -m <module> ...
```

If GPU1 is unavailable: STOP; no fallback to GPU0.  GPU0 foreign processes are
never touched.

---

## 9. Cheap training smoke (both arms, before any formal run)

```text
first 512 official-train molecules, 3 epochs, GPU1, official valid NEVER read
```

Applicable arm requirements:

```text
finite task loss, finite reconstruction loss
nonzero task gradient to D
nonzero gradient to the constant node channel row
nonzero gradient to the constant edge channel row
nonzero gradients to the atom / bond valuation matrices
exact top-8 sparse alpha
dictionary atoms alive
environment effective rank > 1
no NaN / Inf
```

If any core mechanism is dead: STOP.

---

## 10. Formal Stage — seed 0, matched, sequential

Two arms, both freshly trained in this round:

```text
Reference:  exact current H1 (P2-ABS frozen config)
Purified:   constant structural channel + size-only local structural scalar,
            everything else frozen
```

Protocol (identical for both arms):

```text
data          official ZINC train 10000 / official valid 1000
              official test NEVER loaded
seed          0
horizon       320 epochs, fixed, no early stop
optimizer     Adam, lr 1e-3, weight_decay 1e-5, batch 128, grad clip 5
loss          L1 + 33.95873017865987 * L_rec
soup          fixed Top-5 equal-weight soup by official-valid MAE
device        GPU1 only, sequential (one CUDA process at a time)
```

### 10.1 Primary quantity

```text
Delta_purification = MAE_purified - MAE_reference        (seed 0, Top-5 soup)
```

Smaller is better.  This is a **non-inferiority** test, not an improvement test.

### 10.2 Reference reproduction sanity (checked before reading the candidate)

```text
if (fresh reference seed-0 soup) - 0.12354862861608853 > 0.005:
    verdict REFERENCE_REPRODUCTION_FAILURE, STOP, diagnose data / artifact /
    dictionary / initialization / CUDA regime / protocol
```

The candidate is not interpreted while the baseline drifts.

### 10.3 Seed-0 decision rule (frozen)

| condition | action |
|---|---|
| `Delta <= +0.002` | candidate seed-0 non-inferiority **PASS**; buy paired seed 1 confirmation |
| `+0.002 < Delta <= +0.004` | ambiguous; still buy paired seed 1 (one seed cannot decide a cleanup) |
| `Delta > +0.004` | `PURIFICATION_PERFORMANCE_FAILURE`; STOP; no seed 1 |

Forbidden rescues after a failure: widening the decoder, changing the constant
channel dimension, adding a residual, restoring part of the anchor, tuning
lambda, seed hunting, new features.

### 10.4 Paired seed 1 (only if authorized by 10.3)

```text
reference seed 1 and purified seed 1, same initialization convention, same
batch order, same GPU1, same 320-epoch protocol
```

Never compare a purified seed-1 run against a historical baseline seed.

### 10.5 Final performance verdict (two seeds)

```text
mean Delta = mean over seeds of (MAE_purified - MAE_reference)
```

| condition | verdict |
|---|---|
| `mean Delta <= +0.002` and neither seed worse than `+0.004` | `PURIFIED_ARCHITECTURE_ACCEPTED` |
| `+0.002 < mean Delta <= +0.004`, or clear seed sign-flip | `PURIFICATION_INCONCLUSIVE` (no further tuning; reference stays canonical) |
| `mean Delta > +0.004` | `PURIFICATION_REJECTED` (reference stays canonical) |

`PURIFICATION_PERFORMANCE_FAILURE` (10.3) is the seed-0 arm of the same rule set.

---

## 11. Mechanism audit (inference only, only on an accepted candidate)

No retraining.  The same interventions are run on the matched fresh reference
arm for pairing; the gates apply to the purified arm.

```text
alpha -> 0                         (constant channel, chemistry, size, global
                                    context, topology, reader all kept)
node assignment shuffle            (permute alpha <-> q inside each (root, shell);
                                    alpha multiset, q multiset, shell, constant
                                    channel preserved; 5 fixed permutations)
edge assignment shuffle            (permute structural edge role <-> b_uv inside
                                    each (root, shellpair); 5 fixed permutations)
combined shuffle                   (node + edge simultaneously)
```

Continuity thresholds (P1 standard):

```text
dictionary-zero degradation   >= 0.030
node assignment shuffle       >= 0.010
combined assignment shuffle   >= 0.015
edge shuffle                  reported, not a hard gate
```

If performance is preserved but these mechanisms no longer load the model, the
round must report `PERFORMANCE_PRESERVED_BUT_MECHANISM_CHANGED` and may **not**
claim the dictionary mechanism survived.

---

## 12. Constant-channel ablation (inference only, accepted candidate)

```text
node constant channel -> 0
edge constant channel -> 0
both -> 0
```

Report `MAE delta`, prediction-shift mean / median / p90.  This is **not** an
architecture-selection gate.  If the constant channel is barely used, that is a
first-class result (the old local chemistry anchor may have been redundant) and
the model is still not changed.

---

## 13. Dictionary health (both arms, matched comparison)

```text
active atoms, effective atom count, support entropy, max support share,
dictionary movement from initialization, train reconstruction,
valid reconstruction, task gradient to D, code variance,
environment effective rank
```

The cleanup must not buy its performance by weakening the dictionary and
leaning on raw chemistry.

---

## 14. Purity audit (both arms)

```text
number of independent local chemistry entry points
number of independent global chemistry entry points
number of local raw chemistry bypasses
trainable parameter count
module count
chemistry-specific parameter count
dictionary-specific parameter count
```

Reference: local raw chemistry bypass `present` (3 anchor blocks).
Purified candidate: `absent`.

---

## 15. Official split discipline

```text
official train  : training allowed
official valid  : pre-registered selection / evaluation allowed
official test   : NEVER LOAD in this round
```

There is no terminal-test authorization this round.

---

## 16. Forbidden in this round (regardless of outcome)

```text
remove global chemistry; remove global topology; remove pair composition;
change distance buckets; unify node and edge decoders; change fusion width;
change the reader; change dictionary capacity; change sparsity; change the
coder; add attention; add message passing; add recurrence; add a residual
bypass; add a new feature; add a ring feature; add a motif; change the radius;
optimizer sweep; lambda sweep; width sweep; seed hunting; official test;
MolHIV training; MolHIV test
```

These require separate preregistrations.

---

## 17. Durable artifacts

```text
notes/e2e_dictenv_purify_v0_architecture_audit.md
notes/e2e_dictenv_purify_v0_preregistration.md    (this file)
notes/e2e_dictenv_purify_v0_implementation.md
notes/e2e_dictenv_purify_v0_analysis.md

results/e2e_dictenv_purify_v0/
    reference_inventory.json
    information_flow.json
    semantic_refactor_equivalence.json
    parameter_ledger.json
    purity_audit.json
    smoke_reference.json  smoke_purified.json
    reference_seed0.json  purified_seed0.json
    reference_seed1.json  purified_seed1.json   (only if authorized by 10.3)
    paired_performance.json
    dictionary_health.json
    mechanism_interventions.json
    constant_channel_ablation.json
    decision.json  REPORT.md  DECISION.md
```

Files whose stage does not fire are **not** fabricated.  If seed 1 is not
authorized they are explicitly recorded as `NOT RUN` with the reason.

Records: `records/claims/`, `records/decisions/`, `STATE.yaml`.

If accepted, the claim is limited to:

> local zeroth-order chemistry can be consolidated into the attributed
> structural measure without material loss under the frozen H1 architecture.

It must not be widened to "all chemistry bypasses are unnecessary".

---

## 18. Stop rule

After this round is recorded:

```text
STOP
```

No automatic removal of global chemistry, pair relation, node/edge decoder
unification or global topology; no automatic next layer.  The user decides.
