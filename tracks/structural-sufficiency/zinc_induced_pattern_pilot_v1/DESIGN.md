# DESIGN — zinc-induced-pattern-pilot-v1 (pre-registration)

**Status:** declared before the first formal GPU run. This is a bounded
representation pilot, not a GRIT reproduction and not a reopening of the
M/S/T readout search.

**Track:** `tracks/structural-sufficiency/zinc_induced_pattern_pilot_v1/`
**Base commit:** `46db232` (`zinc_recipe_trunk_v2` closed)
**Budget:** 1.7558 A100-hours remain for this task (both GPUs summed, including
timing, smoke and formal training). The cap is a hard boundary; no purchase
outside it.

## 1. Proposition

Fix the P3 raw descriptors and the P3 Null composer/reader. Does a learned
higher-order, **typed rooted induced-path** object improve generalization over
the current P3, before the fixed global pooling?

This is a new performance hypothesis, not a confirmed bottleneck. It is tested
only in the representation/interface described below. A negative result is
limited to this representation, interface and recipe; it is not evidence that
"higher-order patterns are useless" and it cannot support a SOTA claim.

### Existing evidence used as context

- P3 x recipe A seed 0/1 best-valid: 0.079105 / 0.076668.
- **Primary reference for this pilot:** P3 x recipe B seed 0 best-valid
  **0.076224** (`runs/xBssl_P3N_s0`, epoch 377), the best P3 cell measured in
  the preceding round. Its SSL initialisation, recipe B, checkpoint and data
  path are audited before this pilot (`analysis/cpu_checks.json:
  reference_audit`). P3 x recipe A is listed only as historical background; it
  is not the reference used to judge J.
- P3 scratch x B = 0.093734 versus SSL x B = 0.076224 shows that the
  train-only SSL initialisation matters. J reuses the seed-0 train-only SSL
  encoder and fine-tunes it normally.
- R0/R1 compress neighborhoods into statistics; P0/P1 use sampled walks with
  older readout/closes semantics; M/S still use atom/bond/junction objects.
  None directly tests full-enumeration short paths with the complete typed
  induced adjacency, a shared continuous pattern encoder, and an early entry
  into the P3 encoding chain.
- **Correction carried into this design.** P3's Null grouping is *not*
  "each bond receives the junctions that touch it". In code,
  `group_to_bonds(NULL)` computes the graph-wide junction means
  `mean(w)` and `mean(w^2)` and returns `n_b * mean(w)` and
  `n_b * mean(w^2)`, where `n_b` is this bond's junction count. Only the count
  `n_b` is local; the junction content is graph-global. J deliberately uses
  this exact P3 Null composer, not a local-junction variant. The old
  `zinc_recipe_trunk_v2/REPORT.md` interpretation is corrected separately
  without changing any experiment number.

## 2. Static objects: all rooted simple induced paths

For every molecule and every root atom `v`, enumerate **all simple paths**
starting at `v` with `l` edges, `l in {2,3,4,5}` (3–6 atoms). Nodes in a path
are distinct. The root is fixed at position 0; reversing a path changes the
root and is a different object.

Each path retains:

- raw atom category at each of the up to 6 positions (21-way one-hot category
  from the raw atom type, not a raw atom index);
- a learned position embedding, so position 0 is the root role;
- the complete typed **induced** adjacency on the visited nodes: the `l`
  consecutive path edges plus every non-consecutive edge present in the raw
  graph. Pair type is `0 = no edge` or raw bond type `1..3`;
- explicit node mask, explicit pair mask and length. A padded pair and a real
  no-edge pair are distinct (mask 1 with type 0 versus mask 0).

Important object semantics:

- A simple path never revisits a node. Ring-closure information is a real extra
  edge between two visited nodes, not "the same atom type at two positions".
  For a length-5 path on a six-membered ring, the first/last edge appears as a
  non-consecutive pair in J's induced matrix.
- This is **all rooted path traces of the specified lengths**, not all rooted
  subgraphs, not a complete radius neighborhood, and not claimed to be
  lossless or WL-complete.
- There is no node-number truncation, no random sampling, and no split-dependent
  count. The enumeration is exhaustive by construction; the cache records the
  exact occurrence profile (train: 3,970,062 path occurrences; valid:
  394,064; see `analysis/cpu_checks.json`).
- The root stays at position 0. Reversal is not forced to be invariant under a
  fixed root. Whole-graph renumbering must leave the prediction unchanged
  (checked).
- A raw typed trace may be compressed to one content pattern with a true
  per-root multiplicity. Unique pattern count is never substituted for
  occurrence count. Multiplicity is part of the model definition.

For the optional control Q, the same rooted path objects, atom categories,
consecutive bond types, lengths and multiplicities are used; every
non-consecutive pair type is replaced by `0` (no edge). Q distinguishes the
value of the complete induced topology over the joint typed sequence. J vs Q
is not a controlled test of "all higher-order information", because both add a
new typed sequence encoder.

## 3. Shared pattern encoder φ and root statistics

The pattern content is encoded by a shared MLP, never by a pattern-id table:

```
input = [atom-category embedding at 6 positions,
         position embedding at 6 positions,
         pair-type embedding for 15 unordered pairs,
         pair-position embedding for 15 pairs,
         node mask (6), pair mask (15), normalized length (1)]
        (268 dimensions)
φ = MLP(268 -> 80 -> 32), shared across all patterns and all molecules
```

Padding embeddings are multiplied by the explicit mask; padding is zero.
No `nn.Embedding(pattern_id)` exists. Pattern ids in the cache are only a
reversible content address (byte-sorted content); valid patterns absent from
train are still encoded from their raw content, with no OOV table and no
frequency filtering. During a mini-batch, each unique content pattern is
encoded once and occurrence counts scatter it to roots; the φ output is
recomputed every forward and is never cached detached.

For every root `v` and every length `l`:

```
n_vl   = sum over occurrences of the true multiplicity
mu_vl  = sum(c * φ(pattern)) / n_vl
sq_vl  = sum(c * φ(pattern)^2) / n_vl
stats  = [mu_vl, sq_vl, log1p(n_vl), 1(n_vl>0)]        (66 dims)
```

If `n_vl = 0`, the first three entries are zero and the existence mask is 0.
The four lengths are concatenated (264 dims) and passed through a shared
projection `264 -> 128 -> 96`; the last linear layer is zero-initialised.
Roots with no paths of any length receive exactly zero increment.

## 4. P3 interface and exact parameter size

Build order is fixed:

1. instantiate the original P3 modules (`P3Encoders`, `group_composer`,
   `reader`) with the same `torch.manual_seed(seed)` as the reference builder;
2. load the seed-0 train-only SSL encoder state
   (`pretrain_encoder_s0.pt`, sha256 `006fabc1...`);
3. create the new path branch.

Then:

```
a_v = atom_encoder(raw108_v) + d_v
```

Recompute the original P3 bond encoder from `a`, the original junction encoder
from `a` and `b`, then the original P3 **Null** composer and reader. No
lookup residual on the final prediction, no frozen encoder, no encoder LR
scaling, no auxiliary loss; all parameters train together under the fixed
recipe.

**Final audited size (CPU, before any GPU run):**

| component | parameters |
|---|---:|
| P3 atom encoder | 26,336 |
| P3 bond encoder | 62,816 |
| P3 junction encoder | 112,896 |
| P3 group composer | 62,976 |
| P3 reader | 148,481 |
| **P3 subtotal** | **413,505** |
| path branch (embeddings + φ + projection) | **70,776 |
| **J total** | **484,281** |
| cap | 500,000 |
| headroom | 15,719 |

Q uses the same parameter layout and seed initialisation; only its pair content
table differs. Every parameter is in the optimizer; every parameter requires
grad. The last projection is zero at initialisation, so the initial forward is
bitwise equal to the same SSL-start P3 forward (checked on real batches; max
absolute difference 0.0). φ's first-step gradient is expected to be zero; after
one update the projection opens and φ receives finite non-zero gradients
(checked over six CPU steps; first non-zero φ gradient at step 2).

## 5. Data, splits and no-test rule

- Official PyG ZINC subset, train 10,000 / valid 1,000. Raw-`y` L1.
- **This round never reads the official test cache/labels/predictions and never
  invokes a terminal-test script.** No test file is constructed. `path_test.pt`
  does not exist.
- All preprocessing thresholds/standardization use train only. The static raw
  graph enumeration itself uses no labels. Valid labels are used only for the
  fixed per-epoch validation and best-checkpoint selection.
- The existing v1 data cache, P3 objects and seed-0 SSL encoder are reused
  unchanged. The new path cache is built from the frozen P3 objects (themselves
  a label-free function of the raw graph); it is not built from `y`.
- Data fingerprints and reference state are recorded in
  `analysis/cpu_checks.json` and each run's `train.json`.

## 6. Fixed recipe B

Identical to the existing P3 x B implementation (`ad_train.RECIPE_VARIANTS["B"]`):

- AdamW; batch size 128; `ceil(10000/128) = 79` steps/epoch, last partial batch
  kept; 600 epochs; 500-update linear warm-up; per-step cosine 1e-3 -> 1e-5;
  weight decay 1e-5; gradient clip 5; raw-`y` L1.
- Full 1,000-molecule valid every epoch; strict `<` best-valid, earlier epoch
  wins a tie. Keep `best.pt`, `final.pt`, full train/valid curves and
  `predictions/valid_best.npz`.
- Online train MAE and fixed eval-mode train-subset MAE are recorded
  separately. No warm start from any supervised checkpoint; no SSL
  re-pretraining.

## 7. Budget branches (decided before the formal run)

Total remaining: **1.7558 A100-hours**, including timing and smoke. GPU-hours
are the sum over both GPUs; running two jobs in parallel does not halve the
accounting.

1. Run CPU enumeration, batching and parameter checks first.
2. Run a **2–3 epoch GPU timing job** for J and Q (target <= 0.04 A100-h total
   for timing/smoke). Measure actual seconds/epoch including full valid.
3. Decide and record the branch **before** any 600-epoch launch:

- **a.** If `2 × projected_J_full + timing/smoke + 15% margin <= 1.7558 h`,
  run J seed 0 and Q seed 0 in parallel on the two GPUs.
- **b.** If only `1 × projected_J_full + timing/smoke + 15% margin` fits, run J
  seed 0 only; Q is deferred.
- **c.** If even one full run does not fit, stop after CPU checks and timing;
  report the real required budget and do not silently extend the horizon.

No horizon compression, no cosine reset, no half-run extrapolation. A run that
does not complete the full 600 epochs is not reported as a P3-comparable
comparison.

## 8. Checks required before the formal run

CPU (all in `ip_cpu_checks.py`, 12/12 pass before GPU timing):

- small-graph enumeration/multiplicity hand check; same-type different nodes
  remain distinct occurrences;
- six-ring vs chain: J distinguishes the extra `(0,5)` edge, Q removes it;
- whole-graph random renumbering: raw pattern multiset, root aggregation and
  eval prediction unchanged (float tolerance);
- cross-graph batch isolation: perturbing one molecule does not change another
  molecule's root increment or prediction;
- expanding/compressing the same multiplicity multiset leaves weighted
  mean/second/count and model output unchanged;
- zero-increment J vs P3 forward bitwise equal on the same batch;
- SSL encoder state bitwise equal to the checkpoint; train/eval switches;
- every parameter in the optimizer; φ gradient opens after the zero projection;
- finite output for roots with no paths; no non-finite short training values.

A 2-epoch GPU timing run is a cost measurement, not a generalization result.
No single-batch overfit is used as evidence of generalization.

## 9. Read-out and stop rules

- Decision is on **valid only**. Primary reference: P3 x B seed 0 = 0.076224.
  Historical P3 x A seed 0/1 are background only.
- J/Q are compared at seed 0 with the same recipe and initialization. If either
  shows a clear valid gain (roughly 0.003–0.005 is a confirmation-worthy
  signal, not a significance threshold), the next purchase would be seed 1
  plus an initialization-matched P3 x B seed 1; no SOTA claim from one seed.
- If J/Q show no meaningful signal, stop this pilot. Do not open rank, depth,
  path-length, pooling, auxiliary, optimizer or recipe grids, and do not rerun
  old P0/P1 to rescue the story.
- Do not treat valid-to-test family offsets as calibration constants. The
  official test is not read in this round and cannot be used to select a
  family, recipe or epoch.
- Any negative conclusion is bounded to this representation, interface and
  recipe.

## 10. Deliverables

- This DESIGN.md (pre-registration, written before the first formal GPU run).
- `code/ip_data.py`, `code/ip_models.py`, `code/ip_runner.py`,
  `code/ip_cpu_checks.py`, `code/tests/test_ip_cpu.py`.
- `REPORT.md` with real costs, raw curves, best-valid/epoch, parameter counts,
  J/Q comparison, and explicit Observed / Interpretation / Next choice.
- Machine-readable config, budget ledger, checkpoint hashes, valid predictions
  (large artifacts git-ignored per repo rules).
- Correction of the previous P3 local-incidence interpretation without changing
  any original number.
