# PROTOCOL — ECLC-v1 (Edit-Conditioned Local Contribution)

| field | value |
|---|---|
| round id | `zinc-edit-conditioned-local-contribution-v1` |
| track | `tracks/structural-sufficiency/` |
| status | `FROZEN_BEFORE_FORMAL_GPU` |
| data | official PyG ZINC `subset=True`: train 10,000 / valid 1,000 / test 1,000 |
| metric | raw-`y` MAE (lower is better); no target transform |
| primary selector | per-run single **best-valid** checkpoint (strict `<`, ties → earlier epoch) |
| test policy | **never constructed, loaded, predicted or scored** |
| budget | seed 0 + smoke ≤ 2.0 A100·h; seed 1 (only if promoted) +2.0; hard cap 4.0 A100·h |
| formal trajectories | at most four (REAL/NULL × seed 0/1) |

This round tests **one** structural hypothesis. Phase 0 has already passed its
support gate with the qualified verdict `PASS_COMPOSITION_SIGNAL_ONLY`; the
qualification is part of the frozen record and may not be rewritten as an
edit-class-identity result.

---

## 0. Boundary and relation to the historical record

Read-only historical inputs: LCRC-A (`zinc_local_relation_composition_v1`), DCC-L
(`zinc_dual_channel_composer_v1`), Additive Reader (`zinc_additive_reader_v1`),
Shared Factor (`zinc_shared_factor_v1`, supplement), RBO
(`zinc_rooted_branch_operator_v0`), Composition Transfer
(`zinc_composition_transfer_v1`), Support Transfer Probe
(`zinc_support_transfer_probe_v1`), HIS-v0 (`zinc_his_v0`), ZPR-v1
(`zinc_perf_recipe_pilot_v1`), and `NEXT_DIRECTION.md`.

The historical valid/test numbers are never recomputed as new evidence here.
Only the official valid split is used for model evaluation and checkpoint
selection. The official test split is not available to this round.

### 0.1 The one hypothesis

> **H_ECLC**: On the official ZINC 10k subset, with the frozen LCRC-A backbone
> and a fixed 600-epoch recipe, explicitly forcing the per-atom base
> contribution to depend only on the original atom type and the rooted
> radius-1 typed adjacency improves generalisation to unseen radius-2
> environments, relative to a parameter-identical control whose base head may
> read a same-dimensional representation that includes radius-2 content.

This is a **single controlled contrast**. It is not an architecture search and
does not license width, rank, pair-count, loss-weight, epoch, seed, or feature
grids.

### 0.2 Computational difference from the historical rounds

| round | forward computation | ECLC-v1 difference |
|---|---|---|
| DCC-L / P3 | learned junction encoder → junction state → local/global composer → 513-d reader | no junction object, no junction state, no wide reader |
| LCRC-A | `Σ_v f(h_v) + ρ(global33, mean h, mean h², count)`; `h_v` from atom108 + S1 joint + S2 typed 2-paths + scalars | same backbone, but the local readout is split into **two** additive per-atom heads: `f_base` (strict radius-1 only) + `f_context` (unchanged full LCRC content) |
| Additive Reader | DCC-L trunk on bonds, additive reader `wᵀΣh_b + ψ(G)` | different trunk and per-atom base/context split; no bond-state reader |
| Shared Factor | shared atom states + exact small-fragment contraction; factors shared across graph positions | no factor table, no fragment exact enumeration, no shared discrete factor |
| RBO | rooted branch operator on a minus-support split | no branch operator, no support-minus split, no exact key lookup |
| Composition Transfer | cross-group look-ahead update constraint on a source/query split | no update constraint, no group split, no message passing |
| Support Transfer Probe | additive error-correction head trained on held-out support | no residual/error head; the pair loss acts on end-to-end predictions, not on a residual target |
| HIS-v0 | explicit 2-hop ANOVA regulariser on LCRC-A's path channel | no high-order penalty and **no shrinkage of the high-order path**; `f_context` is unchanged and untied |
| LCRC relation arms R/N | one-shot atom-pair relation content with real-vs-mean-field ownership | no atom-pair relation head; the contrast is the input restriction of `f_base` |

The two-level numerical split "per-atom base + per-atom context + global" is
**not** the historical two-level statistical split: the historical rounds either
split a learned junction state (DCC-L), a bond state (AR), a factor table (SF),
a support set (RBO/STP), or a path channel (HIS). ECLC-v1's split is a
**same-backbone input restriction** on a new additive head, with a
parameter-identical radius-2-readable control.

### 0.3 Prior-art boundary

Matched-molecular-pair and edit-conditioned objectives are not new. SQRL
(NeurIPS ML4PS 2024) and related matched-pair / relative-property methods use
paired molecules to learn property differences. ECLC-v1 does **not** claim a
new relative-learning paradigm, a new pairwise-supervision principle, or a new
matched-pair construction. Its only claim under test is the structural
restriction on the base head, measured against the parameter-identical NULL
control. Any report must say this explicitly.

---

## 1. Phase 0 — frozen result

The frozen pair list is `analysis/edit_pairs.json`; the gate is
`analysis/phase0_gate.json`; the human-readable summary is
`analysis/phase0_report.md`.

### 1.1 Matching rule (label-free)

- Official train-10k structures only; `y`, valid and test are never read.
- RDKit reconstructs the PyG graph from the raw GVAE atom vocabulary
  (element, charge, explicit-H count) and bond orders. Reconstruction failure
  rejects the molecule (observed: 0).
- Bridges only. The larger side is the preserved **body**; the smaller side is
  one pendant **edit region**. The body is isomorphic with atom and bond
  labels, including the attachment port; the port is the unique dummy
  neighbour in the fragment graph.
- `min(|body|) ≥ 5`, `|sub| ≤ 12`, and
  `|sub_i| + |sub_j| < min(|body_i|, |body_j|)`.
- The two substituents must not be isomorphic and the two whole molecules must
  not be identical.
- For each molecule pair, keep the match that minimises
  `|sub_i| + |sub_j|`, then maximises `min(|body_i|, |body_j|)`, then uses the
  lexicographically smallest body key.
- Every accepted body is verified by an explicit NetworkX labelled
  isomorphism, not only by a canonical-SMILES string.
- Edit class = unordered pair of **rooted radius-1 typed signatures** at the
  attachment atom (cut bond type, root atom label, multiset of
  substituent-side neighbour labels and bond types). The strict
  full-substituent-pair class is reported separately.

### 1.2 Observed support

| criterion | required | observed | pass |
|---|---:|---:|:---:|
| strict edit pairs | ≥300 | **350** | yes |
| distinct preserved bodies | ≥100 | **317** | yes |
| edit classes in ≥3 bodies | ≥15 | **29** | yes |

Additional frozen observations:

- 621 independent molecules (6.21% of train); 190 edit classes; maximum 24
  bodies per class.
- Largest connected pair-graph component: 7 molecules; 285 components.
- Pair degree: mean 1.127, max 5, p90 2, p99 2, Gini 0.101; 700 pair endpoints
  (7.0% of train). Pair loss therefore mildly reweights a small subset.
- Ring composition: 37 pairs have two acyclic substituents; 313 pairs have at
  least one ring-containing substituent; both bodies contain a ring in all 350
  pairs.
- Edit-region size: median 13 (range 2–22); preserved-body size: median 16
  (range 9–28). These are one-pendant-region edits, not only tiny
  functional-group swaps.
- Strict full-substituent-pair classes: 350 total, **0** recurring in ≥3
  bodies. This is why the support gate class is the local radius-1 signature.

### 1.3 Differential-signal gate (train-only labels)

Features: signed low-order structural difference (heavy-atom count, 21 atom-type
counts, 3 bond-type counts, ring count, ring-atom count, aromatic-atom count).
For a strict single-edit pair the body cancels, so these are exactly the
edit-region differences. Ridge (`alpha=1.0`, standardized, intercept), 5-fold
GroupKFold by body, fold seed 0.

| model | held-out Δy MAE | rel. improvement vs zero |
|---|---:|---:|
| predict zero | 0.885882 | 0.0000 |
| **low-order structural difference (primary)** | **0.346950** | **+0.6084** |
| radius-1 edit class only | 0.966043 | −0.0905 |
| combined + true edit class (stress test) | 0.384284 | +0.5662 |

100 edit-class permutation controls: mean 0.376430, min 0.358892, 5th pct
0.361946. The primary low-order model beats every control. The combined
true-class model is worse than 81/100 controls (one-sided empirical p = 0.812).

**Verdict: `PASS_COMPOSITION_SIGNAL_ONLY`.** There is a strong cross-body
transfer signal, but it is additive-composition signal, not a categorical
edit-class identity effect. The GPU pilot is licensed because an additive local
base head can consume that signal. The pilot may **not** be described as
evidence for edit-class identity transfer.

---

## 2. Phase 1 — the two arms

The backbone is the frozen LCRC-A exactly: atom/bond encoders with the frozen
train-only masked-SSL initialisation, `LocalContent`, `f_context` (the original
`f`), `global33`, and the original `ρ`. No relation head, no junction module,
no extra message passing, no teacher, no ensemble.

Let, per atom `v`:

```
base_v = [ onehot(atom_type_v, 21),
           log1p(R1_counts_v, 63),          # (neighbour type, bond type)
           log1p(R2_path_counts_v, 189),    # (middle type, b1, b2)
           log1p(R2_end_counts_v, 63),      # (end type, b2)
           log1p(R2_closure_count_v, 1) ]   # 337 dims
```

- **REAL**: `f_base` receives `[onehot(atom_type), log1p(R1_counts)]` (84 dims)
  and the 253 radius-2 dims are identically zero. It cannot read radius-2,
  P3 atom108, S2 content, ring membership or the graph-level vector.
- **NULL**: `f_base` receives the full 337-dim representation, which contains
  radius-1 **and** radius-2 content. This is the parameter-identical
  "base contribution is not forced to be shared across contexts" control.
- `f_base`: `337 → 32 (SiLU) → 1`; the last Linear weight **and** bias are
  zero-initialised.
- `f_context`: the original LCRC `f: 64 → 64 → 1`, unchanged; the high-order
  interaction is neither deleted nor uniformly shrunk.
- `ρ`: the original LCRC `ρ(global33, mean h, mean h², log(1+|V|))`, unchanged.

Final prediction:

```
y = Σ_v f_base(base_v) + Σ_v f_context(h_v) + ρ(global33, mean h, mean h², count)
```

Both arms:

- have exactly the same parameter count: LCRC-A 237,519 + `f_base` 10,849 =
  **248,368** (≤500k);
- are built from the **same** frozen LCRC-A backbone state and the **same**
  `f_base` initial tensors; the full initial state hash is checked to be equal;
- share `f_context`, `ρ`, the atom/bond encoders and `LocalContent` tensor by
  tensor;
- have the same prediction interface and the same training recipe.

At initialisation `f_base ≡ 0`, so both arms output exactly the same prediction
as the frozen LCRC-A at the same SSL start. This is verified, not assumed.

### 2.1 Frozen pair loss

Two arms use exactly the same train-only pair list, the same pair sampling
stream, and the same loss:

```
loss = L1(molecule prediction, molecule y)
     + 0.2 * L1(pred_i - pred_j, y_i - y_j)
```

- Per update: the normal 128-molecule training batch (unchanged) plus
  **8** edit pairs = 16 additional molecules, forward-propagated as a separate
  batch. The optimizer performs one update per step.
- Pair sampling: `torch.randint(0, 350, (8,))` from a dedicated CPU generator
  seeded `seed + 202`, once per update. The stream is identical for REAL and
  NULL.
- The pair loss is a mean over the 8 pairs; the molecule loss is a mean over
  the 128 molecules.
- The pair loss uses only `y_i` and `y_j`, both already present in the molecule
  L1. It is an additional **inductive bias**, not a new label and not
  independent supervision. The report must state the algebraic facts:
  `pred_i - pred_j` is by construction the difference of the additive terms;
  for L1 the pair loss is a function of residual differences, not identical to
  the molecule L1; it is invariant to a common additive offset and therefore
  cannot replace the molecule loss. The CPU gate verifies these statements.
- `0.2` and `8` are fixed; no weight/coefficient/draw-count search.

### 2.2 Frozen recipe

Identical to LCRC-A/ZPR-v1:

- AdamW, peak LR `1e-3`, min LR `1e-5`, 500-update linear warm-up, per-step
  cosine thereafter, weight decay `1e-5`;
- batch 128, 600 epochs = 79 steps/epoch = **47,400 updates**;
- raw-`y` L1; gradient clip 5; dropout 0;
- no EMA, no soup, no seed averaging, no early stopping;
- full official valid every epoch; strict minimum valid MAE single checkpoint
  (ties → earlier epoch); `final` and fixed-window diagnostics are reported but
  are never used to select the primary number.

Data order: `torch.randperm(10000, Generator().manual_seed(seed+101))`,
reshuffled once per epoch; the last partial batch is kept. Identical for REAL
and NULL.

---

## 3. Phase 2 — CPU correctness gates (all must pass before GPU)

1. **Independent local-statistics reference.** For a sample of real train
   molecules, rebuild `atom_type`, radius-1 typed counts, radius-2 path/end/
   closure counts with a naive loop from the raw graph and compare element-wise
   to the cache-derived `base_v` used by the model. Exact integer equality.
2. **Synthetic radius-1 invariance / radius-2 response.** Construct two graphs
   with an identical radius-1 environment at a target atom and a different
   radius-2 environment. REAL `f_base` at the target atom must be **exactly**
   identical; NULL must be allowed to differ; `f_context` at the target atom
   must differ. The r2 block of REAL must be identically zero.
3. **Invariance.** Atom renumbering, bond direction reversal, molecule order
   and batch split must not change predictions beyond the execution tolerance
   (`≤1e-5` for float32 CPU; record the actual maximum).
4. **Parameter identity and cap.** REAL and NULL have the same parameter count,
   the same per-module breakdown, and ≤500k deployment parameters.
5. **Initial equivalence.** Both arms have the same full initial state hash;
   `f_base` last layer is zero; both arms' initial full predictions equal the
   frozen LCRC-A prediction at the same SSL start (max abs diff 0).
6. **Gradients and path liveness.** One molecule+pair step: all gradients are
   finite; `f_base`, `f_context` and `ρ` receive nonzero gradient; in REAL the
   253 radius-2 input columns of `f_base` receive exactly zero gradient; in
   NULL they receive nonzero gradient for at least one column; the pair loss
   contributes a nonzero gradient to `f_base`.
7. **Checkpoint round-trip.** Save/restore the full wrapper state; predictions
   reproduce exactly (or ≤1e-6 float32) on a fixed batch.
8. **Split isolation.** The pair list contains only train rows `0..9999`; no
   valid/test row; no test cache exists or is read; the training code asserts
   `official_test_loaded=False`; the valid labels are used only for evaluation
   and checkpoint selection.
9. **Pair-loss algebra.** Verify: (a) `pred_i - pred_j` equals the sum of the
   differences of the three additive terms; (b) adding a constant offset to all
   predictions leaves the pair loss unchanged but changes molecule L1; (c) the
   pair-loss gradient is not proportional to the molecule-L1 gradient;
   (d) for squared loss the pair term is a quadratic form in the same
   residuals, i.e. no new label information. The report must not call the
   output difference "additional structural information".
10. **Mechanism difference is real.** On a batch with varied radius-2
    environments, REAL and NULL `f_base` outputs differ; their `f_context`
    outputs are identical (same weights, same input). If the difference
    disappears, stop before GPU.
11. **Forward non-equivalence with LCRC-A after a non-zero base step.** After
    one optimizer step with the pair loss, zeroing `f_base` changes the
    prediction by a nonzero amount; the new model is not forward-equivalent to
    LCRC-A. The *initial* zero-base equivalence is intentional and is reported
    as such.

If any gate fails, record `STOP_CPU_GATE_FAILED` and do not start GPU.

### 3.1 Phase 2 outcome (appended before GPU; thresholds unchanged)

Recorded before any formal GPU launch.  All **12** automated checks in
`code/eclc_cpu_checks.py` passed (`analysis/cpu_checks.json`, SHA-256
`34492c1a956f372240218f2ea4d3e3202b73e9330a392e97702d1e62facf6c06`), and the
10 fast unit tests passed.  Key values:

- independent naive-loop reference: 14 train molecules / 332 atoms, max abs
  diff **0.0**;
- REAL/NULL params **248,368** each (LCRC-A 237,519 + `f_base` 10,849),
  identical per-module breakdown and full initial state hash;
- initial REAL/NULL prediction equals frozen LCRC-A with max abs diff **0.0**;
- invariance maxima: atom renumbering `4.77e-7`, bond reversal `2.38e-7`,
  batch split `5.96e-8`, molecule order `0.0`;
- REAL radius-2 input block identically zero; REAL radius-2 first-layer
  gradient exactly zero after the zero-init last layer becomes nonzero, while
  NULL receives nonzero radius-2 gradient (`1.15e-2` max abs);
- pair loss offset-invariant (delta 0.0), molecule L1 changes by `3.70` under
  the same offset; pair-vs-molecule gradient cosine `-0.216`; squared-pair
  quadratic identity `0.0`;
- REAL/NULL `f_base` outputs differ (`0.378` max abs) while `f_context`
  outputs are identical (`0.0`);
- after one step, zeroing `f_base` changes predictions (`0.045` max abs),
  so the trained model is not forward-equivalent to LCRC-A;
- pair list: 350 pairs, all rows in `0..9999`, no valid/test row.

The historical post-hoc `lc_test.pt` file exists in the frozen LCRC cache
directory, but no ECLC production module references it and it was not opened;
the official test split was never constructed, loaded, predicted or scored.

---

## 4. Phase 3 — GPU pilot

### 4.1 Seed 0 (only after Phases 0 and 2 pass)

- GPU0 (`res-gpu0`): **NULL**, seed 0.
- GPU1 (`res-gpu1`): **REAL**, seed 0.
- Exactly the frozen config; no changes after launch.

Valid slices (frozen from the historical anchor, official valid only):

- `full` (1000);
- `ordinary` = `y > -3` (935);
- `r2_unseen_ordinary` = any radius-2-unseen atom & `y > -3` (225);
- `seen_ordinary` = no radius-2-unseen atom & `y > -3` (710);
- `minus_row172` (999);
- `minus_fixed_worst10` (990, frozen P3B worst-10 ids);
- `tail` = `y ≤ -3` (65).

Report for each run: full valid MAE, all slices, seen/unseen gap, best epoch,
train@best, W20/T100/final, curve, wall time, GPU·h, parameter/update counts,
checkpoint SHA, run id.

### 4.2 Seed-0 promotion rule (all must hold)

1. REAL full valid ≤ **0.075**.
2. REAL − NULL full valid ≤ **−0.003**.
3. REAL − NULL on `r2_unseen_ordinary` ≤ **−0.005**.
4. REAL is not worse than NULL on `ordinary` and `minus_row172`, where
   "not worse" means `REAL ≤ NULL + 0.001` (1e-3 execution-drift allowance).
5. The smooth-window direction is not reversed: REAL W20 < NULL W20 and REAL
   T100 < NULL T100; and the new base pathway is active: zeroing `f_base` at
   the best REAL checkpoint changes valid MAE by at least `1e-4`, and the
   per-molecule base contribution has nonzero standard deviation.

If any condition fails: `STOP_NO_COMPOSITION_GAIN`. Do not change width,
parameter count, pair count, loss coefficient, epochs, data order or structure;
do not add seeds.

### 4.3 Seed 1 (only after full seed-0 promotion)

- Run REAL and NULL at seed 1 in parallel with the identical config.
- The two-seed direction must be the same for full valid and
  `r2_unseen_ordinary`; report per-seed and mean effects with paired bootstrap
  CIs and the execution-drift band.
- Only then may the result be called `COMPOSITION_TRANSFER_CANDIDATE`.
- A single seed, even if all seed-0 conditions pass, is never a confirmed
  mechanism.

### 4.4 Budget and stop rules

- Phase 0 and all CPU gates: **0 GPU**.
- Seed 0 (two arms, including the 2-epoch smoke): hard cap **2.0 A100·h**.
- Seed 1 (only if promoted): additional hard cap **2.0 A100·h**.
- Whole task hard cap **4.0 A100·h**, including failed/smoke/pull overhead.
- Before formal seed 0, one 2-epoch smoke on one GPU measures steady epoch time
  and the pair-loss path. If the projected two-arm seed-0 cost exceeds 2.0
  A100·h, stop immediately (`STOP_BUDGET`), do not launch formal runs.
- At most four formal trajectories. No grid search, no extra seeds, no
  re-runs for a better number.

### 4.5 Prohibitions

- No official test construction, loading, prediction or scoring.
- No teacher predictions, ensembles, soups, EMA, seed averaging, external
  property labels, label-formula decomposition, test tuning, or additional
  message-passing layers.
- No use of valid labels to select pairs, edit classes, features, loss weight,
  pair count, width or epoch.
- No claim that the pairwise difference is new structural information, new
  supervision, or an original relative-learning paradigm.
- No post-hoc change of the frozen slices, gates, or primary model.
- No automatic new experiment after a failed gate.

---

## 5. Decision labels

| label | meaning |
|---|---|
| `STOP_INSUFFICIENT_EDIT_SUPPORT` | Phase 0 support gate failed |
| `STOP_NO_TRANSFER_SIGNAL` | Phase 0 differential-signal gate failed |
| `PASS_COMPOSITION_SIGNAL_ONLY` | Phase 0 passed on the low-order composition signal; edit-class identity not established |
| `PASS_TRANSFER_SIGNAL` | Phase 0 passed with edit-class identity also supported |
| `STOP_CPU_GATE_FAILED` | Phase 2 failed |
| `STOP_BUDGET` | smoke projection exceeds the cap |
| `STOP_NO_COMPOSITION_GAIN` | seed-0 promotion rule failed |
| `COMPOSITION_TRANSFER_CANDIDATE` | seed-0 and seed-1 both pass in the same direction |

The final report answers only the five questions in the task brief and records
commit SHA, checkpoint SHA, real run ids, parameter count, update count and
cumulative GPU·h.
