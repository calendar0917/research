# DESIGN — zinc-recipe-trunk-v2 (pre-registration)

**Status:** declared before the first formal GPU run. Continues
`zinc_arch_discovery_v1` (closed at `936cdf1`) without reopening its results.
P3 remains the performance reference; T does not become a baseline by ranking
first among the v1 candidates.

Two questions, in this order:

1. **Recipe cross (Phase I).** v1 compared T against a *historical* P3 that ran
   a different training recipe. Fill in the two missing cells of the 2x2 so the
   architecture comparison is not confounded with the recipe.
2. **New composition trunks (Phase II).** Rebuild the "per-object nonlinear
   composition driven by whole-graph context, then pool" idea of P3 as a
   three-object-type trunk, in two variants (global moments vs learned context
   slots), each with and without a small masked structural auxiliary task.

Not in scope: full GRIT, new input representations, new vocabularies/VQ, larger
path GRUs, R0/R1/B reruns, or a rank sweep of T.

## 0. Fixed boundary

- Data: official PyG ZINC subset, train 10,000 / valid 1,000. Raw-`y` L1.
  Valid is the only development and selection split. **No test read in this
  round at all** — not for selection, not for description.
- Selector: per-seed single best-valid epoch-end checkpoint, strict `<` (earlier
  epoch wins a tie). Full 1,000-molecule valid evaluation every epoch.
- Cap: <= 500,000 trainable parameters per model, including SSL encoders and
  any auxiliary head; every run records the count and optimizer coverage.
- No local recursive message passing: no hidden-state update along the local
  adjacency. Global context and slot cross-attention are allowed and are
  one-shot.
- The seed-0 train-only masked-SSL encoders
  (`.../bjoc_full10k_pretraining_transfer_v1/pretrain_encoder_s0.pt`) are loaded
  and **fine-tuned normally** (no frozen encoder, no reduced encoder LR).
- Reused as-is: the v1 data cache (`zinc_arch_discovery_v1/cache/ad_{train,
  valid}.pt`), the v1 trainer/selection/evaluation code, and the frozen P3
  masked-SSL mask construction (`bjoc_masked_pretraining_v1`).
- Budget: **5.65 A100-hours total** for this task (both GPUs summed), of which
  v1 already spent 4.35 of the earlier 10.0. Phase I ~0.55 h, Phase II <= 3.5 h,
  remainder for short runs and at most seed-1 confirmation of a candidate that
  is close to or better than P3.

## 1. Phase 0 — cheap CPU facts (done before any GPU run)

1. **`closes` semantics fixed.** `PathModel.encode_paths` used
   `closes = (wa == wa[:, :1]) & (pos > 0) & valid`, i.e. "current atom type
   equals start atom type". That is not return-to-start: on propane a walk
   visiting three *different* atoms of the same element was flagged at every
   step after the first. It now uses `(wanon == 0) & (pos > 0) & valid`
   (anonymous pattern index 0, excluding the start/padding), factored into
   `ad_models.path_token_scalars`. Check on two 3-token walks:

   | walk | old flags | corrected flags |
   |---|---|---|
   | propane `0-1-2` (three distinct atoms) | `F T T` | `F F F` |
   | ethane `0-1-0` (one genuine return) | `F T T` | `F F T` |

   Old P0/P1 runs and their reported numbers are kept and are not renamed; they
   were not retrained. Anonymous-mode tokens still carry identity structure, so
   this bug is not offered as an explanation of the whole P0/P1 gap.

2. **Corrected P0/P1 checkpoints, 32 vs 128 sampled walks** (`analysis/
   cpu_checks.json`). Same 8 valid molecules, train-style 32-walk sample vs the
   fixed 128-walk bank, plus a duplicate-every-token invariance test:

   | model | mean abs prediction shift 32→128 (valid, n=8) | duplicate-token max abs diff |
   |---|---:|---:|
   | P0 corrected | 0.100 | 2.4e-7 |
   | P1 corrected | 0.077 | 4.8e-7 |

   The old 0.338-vs-0.691 split-scale blow-up is gone; what remains is sampling
   noise / distribution shift at the level of a few hundredths, not a reader
   scale collapse. This does not trigger a GPU rerun of P0/P1.

3. **P3N equivalence.** The recipe cross needs P3 to run through the v1 trainer.
   `ad2_models.P3NullModel` reimplements P3's computation graph (encoders ->
   Null composer -> 321-d reader) in the v1 batch layout. Loading the frozen
   `zpp` P3 seed-0 best checkpoint (`epoch 272`) into both and scoring 16 valid
   molecules gives **max |difference| = 0.0** with 413,505 parameters in both.
   The recipe cross therefore compares recipes, not a reimplementation.

## 2. Phase I — 2 x 2 architecture x recipe

|  | recipe A (P3/R3 mature) | recipe B (actual v1 T) |
|---|---|---|
| **P3** | existing `zinc_perf_protocol_v1/runs/p3_s0` (0.079105 @ 272, not rerun) | **new**: `xB_P3N_s0` |
| **T (rank 16)** | **new**: `xA_T_s0` | existing `zinc_arch_discovery_v1/runs/T_s0` (0.091375 @ 426, not rerun) |

- **A** = Adam, batch 128, 360 epochs, **no warm-up**, per-step cosine
  1e-3 -> 1e-5, weight decay 1e-5, clip 5, raw-y L1.
- **B** = AdamW, batch 128, 600 epochs, 500-step linear warm-up, per-step
  cosine 1e-3 -> 1e-5, weight decay 1e-5, clip 5, raw-y L1.
- Both new cells: seed 0, initialised from the seed-0 train-only SSL encoders,
  same cached inputs/standardisation, `steps/epoch = ceil(10000/128) = 79`,
  last partial batch kept. No hot start from any supervised checkpoint. Only
  the recipe dict differs; optimizer implementation, schedule function, shuffle
  seed (101), collate, best-valid rule and evaluation are shared.
- Declared deviation: the v1 trainer sets the LR of update `k` from step `k`;
  the historical `zpp_p3` loop used `k-1` (a one-step schedule index offset,
  1 / 28,440 of the curve). Disclosed, not corrected.
- This compares **recipes as wholes**. It does not identify separate optimizer,
  epoch or warm-up effects, and a win/loss here is not an architecture verdict.

Phase I read-out (declared rules):

- If **T under A improves clearly** (>= 0.005 valid), T stays a live model
  candidate for a later seed-1.
- If **P3 under B also degrades clearly** (>= 0.005 worse than 0.079105), the
  shared recipe is a major factor and the v1 gap must not be attributed to the
  architecture alone.
- If **P3 is strong under both and T weak under both**, the next performance
  work should rebuild the composer rather than scan T's rank.

## 3. Phase II — two new composition trunks

Shared input and encoders: the frozen P3 atom/bond/junction objects, the
P3-compatible encoders (atom 108->128->96, bond 295->160->96,
junction 608->160->96), all loaded from the seed-0 train-only SSL checkpoint and
fine-tuned. Typed local embeddings `a_v, b_e, w_j` (96) + a per-type embedding,
plus the existing `global33` and explicit per-type object counts. Every context
comes from the current graph only; nothing is mixed across molecules and `y` is
never used to build context. No P3 prediction shortcut, no association tensor,
no local recursive update.

**M — global moment context + conditional per-object composition**

```
s_t, m2_t, n_t = sum h, mean h^2, log1p N          (t = atom, bond, junction)
c_G   = ContextMLP([s_t, m2_t, n_t]_t, global33)    (612 -> 128 -> 64)
t_o   = h_o + type_embedding
u_o   = t_o + Phi([t_o, c_G[graph(o)], t_o * W c_G[graph(o)]])   (256 -> 128 -> 96)
z_G   = [sum u, mean u^2, log1p N]_t, global33      (612)
y     = Reader(z_G)                                 (612 -> 128 -> 128 -> 1)
```

**S — K=8 learned context slots + per-object conditional composition**

```
object set = all atoms, bonds, junctions of the graph, tokens t_o as above
slots   = CrossAttend(8 learned queries, objects) + pre-LN FFN   (4 heads, d=96)
ctx_o   = CrossAttend(objects, slots) -> project to 64
u_o     = t_o + Phi([t_o, ctx_o, t_o * W ctx_o])     (same shared Phi as M)
z_G, y  = same typed sum/second/count readout as M
```

Both keep unnormalised `sum` and the object counts; the slot softmax only builds
a context and never normalises the graph readout. Attention is masked to the
molecule (block-diagonal); no object-object attention layer and no N^2 graph.

Parameter counts (CPU-audited, head included):
M 435,681; Maux 448,385; S 468,161; Saux 480,865. All <= 500k.

**Auxiliary task (joint variants only).** The masked view is produced by the
frozen P3 SSL construction (15% atoms / 15% physical bonds, dedicated mask
tokens, every feature tensor rebuilt from the masked raw graph so no field leaks
the hidden type; hidden raw codes are the CE targets outside the input).
`L = L1(clean pred, y) + lambda * (mean CE_atom + mean CE_bond)` with a single
declared `lambda = 0.05`; no grid, no automatic balancing. Linear CE heads
(28 atom classes, 4 bond classes) are freshly initialised and counted in the
500k budget. The masked branch never receives `y` and the clean branch predicts
`y` alone. To stay inside the budget, four mask snapshots (visit = 0..3) are
precomputed once on CPU and cycled one per epoch instead of rebuilding masks on
every step; mask ratio, mask source, rebuild pipeline, targets and CE definition
are the published ones.

**Recipe for M/S (fixed before the first M/S trajectory, same for both
architectures and both training variants):** batch 128, **360 epochs**,
500-step linear warm-up, per-step cosine 1e-3 -> 1e-5, weight decay 1e-5,
clip 5, raw-y L1. Optimizer: **Adam** unless P3 under recipe B beats P3 under
recipe A by >= 0.002 valid, in which case **AdamW** for both. 360 epochs is
chosen because P3's best epoch was 272/267 of 360 and T's A-recipe curve is
decided here; the cosine horizon is fixed before launch and is never compressed
mid-run. The warm-up is added because the composition/readout modules are
randomly initialised (and S contains attention); this is stated as a deviation
from P3's no-warm-up recipe, applied equally to M and S.

### 3b. Pre-launch amendments (made before the first M/S trajectory)

- The trunk recipe is declared as the named variant **C** in
  `ad_train.RECIPE_VARIANTS`: Adam (Phase I kept P3's recipe, so the declared
  optimizer rule selects Adam), batch 128, 360 epochs, 500-step warm-up,
  per-step cosine 1e-3 -> 1e-5, weight decay 1e-5, clip 5. C = A's
  optimizer/horizon plus the declared warm-up; both M and S and both training
  variants use it unchanged.
- Phase I result that triggered this choice: P3 under B (0.093734) is worse than
  P3 under A (0.079105), so the rule "Adam unless P3-B beats P3-A by >= 0.002"
  selects Adam.

## 4. Checks before formal runs

- CPU: parameter counts, real forward/backward, every parameter tensor receives
  a finite non-zero gradient, optimizer covers all parameters.
- S: cross-graph isolation (perturbing one molecule leaves the others bitwise
  unchanged: measured 0.0), batch-vs-single invariance (~1e-6).
- M/S: masked-view leak check (changing the hidden truth at a masked position
  leaves every masked feature array unchanged: measured pass).
- P3N equivalence to BJOC (max diff 0.0 on a frozen checkpoint).
- One short 2-epoch timing run per new configuration before its formal run, to
  size the declared horizon against the remaining budget. Timing runs are
  billed and use tag `timing`.

## 5. Stop rules and budget

- Phase I may not exceed ~0.8 A100-h; Phase II <= 3.5 A100-h; total <= 5.65.
- If Phase I already produces a T cell clearly better than P3, Phase II may be
  reduced to the two supervised trunks and the saved budget moved to T seed 1.
- A new trunk gets a seed-1 repeat only if it is close to or better than P3
  (<= ~0.080 valid at seed 0) or shows a clear, cost-effective gain. `~0.075`
  valid is a development target, not a promise and not a significance test.
- If neither the recipe cross nor the new trunks show a signal, stop the round
  and return the remaining uncertainty and its cost. Do not open a rank grid, a
  new vocabulary, or an eighth family to fill the budget.
- Observed / interpreted / next-choice stay separate in the report. A T test
  gap cannot be used to declare the new trunks a success.

## 6. Amendment log (append-only; deviations are dated and justified)

- **A1 (after the first P3 x B launch).** The P3 x B cell was first launched
  through a builder that did not load the seed-0 SSL encoders, so it compared an
  SSL-initialised P3 x A against a scratch-initialised P3 x B. Detected by
  auditing `model_info.ssl` in `runs/P3N_s0/train.json`. Correction:
  `ad2_models.build_model` now loads the seed-matched encoder for `P3N` as well,
  and a CPU check asserts that the initial state is bitwise identical to the
  historical P3 builder (`ad2_cpu_checks.check_p3n_initial_state`, 24/24 tensors
  equal). The corrected cell is `runs/xBssl_P3N_s0` (0.301 h). The defective run
  is kept, labelled "P3 scratch x B", and billed (0.343 h); it is excluded from
  the 2x2 but reported as an SSL-initialisation replicate at 600 epochs
  (+0.017510).
- **A2 (after the corrected P3 x B cell).** The corrected cell shows that
  recipe B *improves* P3 (0.076224 vs 0.079105). Phase II's trunk recipe had
  already been fixed to C on the defective input, and the pre-declared rule
  ("Adam unless P3 x B beats P3 x A by >= 0.002") is now met, i.e. the rule
  would have chosen AdamW. To avoid leaving the four declared candidates
  compared to P3 under an arbitrary recipe, two **post-hoc** seed-0 cells (M and
  S under recipe B, 1.005 h total) were run to test whether the rule transfers
  across architectures. Result: it does not (M +0.0073 worse under B, S −0.0003
  noise), so recipe C was the better or equal choice for both trunks. Declared
  as post-hoc and reported as such. No other post-hoc GPU purchase was made; the
  remaining 1.756 h were not spent.
- **A3.** Budget ledger updated to the realised costs (3.894 h of 5.65).
- **A4 (terminal test read).** At the user's request, and only after every
  checkpoint had been frozen and the round decision written, the official PyG
  ZINC test split (1,000 molecules) was read once with
  `code/ad2_test.py` (CPU, ~3 s, no GPU time). Strictly descriptive: no model,
  recipe, epoch or checkpoint was selected or changed from it, and the §7
  decision stands. The harness reproduces the published P3 test numbers
  (0.069471/0.070664) and v1's T x B (0.075628) exactly, which validates the
  test objects and labels. Reported in §5b with paired bootstrap intervals.
