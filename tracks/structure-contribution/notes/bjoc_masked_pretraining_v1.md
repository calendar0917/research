# cscl-bjoc-masked-pretraining-v1 — design / pre-registration note

**Status (at freeze):** local CPU phases complete (split, object caches, dependency
audit, target-swap audit, full-rebuild audit, feasibility/throughput audit, 22/22
correctness gates, 18/18 fast tests); no GPU trajectory has run.  Protocol
`configs/bjoc_masked_pretraining_v1.yaml` frozen with the split/cache hashes before
any GPU.  **Official valid/test are never loaded in any form.**

## 1. Why this round exists

The predictive-bottleneck round established that at a fixed optimizer-update budget the
BJOC-Null plateau is dominated by labelled-sample quantity / utilisation
(`N8 − N4-stepmatched = −0.0373` locked, both seeds negative, top-10-removed
`−0.0365`), not by junction ownership and not by an optimization-budget deficit.  It
also left one obvious, pre-registered next question: can the **same 8k labelled
molecules** be used better through a *label-free* structural self-supervised
initialization?  This round answers exactly that, with one frozen pretext task and one
downstream contrast.

## 2. Single scientific question

> Without extra y labels, without extra molecules and without changing the downstream
> BJOC-Null architecture/recipe, does strictly leak-free masked atom/bond structural
> pretraining of the three existing BJOC encoders improve held-out raw-y MAE?

The historical reference remains the full-10k official-test two-seed BJOC-Null MAE
`0.092233`; this round runs on a new internal split of official train 10000 only and
makes no extrapolation to the official test.  The round cannot reach 0.06–0.07 and does
not claim to.

## 3. Frozen split `cscl-bjoc-maskpretrain-v1-A`

Canonical-SMILES grouped greedy split of the official train 10000 (salt
`cscl-bjoc-maskpretrain-v1-A`): Fit8k 8000 / Monitor 1000 / Locked-dev 1000, duplicate
molecules never cross sides, no y used.  Sizes are exactly 8000/1000/1000.
Frozen row hashes (also in the config `frozen:` block):

| set | rows | sha256 |
|---|---:|---|
| Fit8k | 8000 | `22cd10eb51b0811b…` |
| Monitor | 1000 | `414b7f6168b80f2d…` |
| Locked | 1000 | `9ff89564ff2c409e…` |

Structure-only object caches (built from raw `x`/`edge_index`/`edge_attr`, labels never
touched) and the Fit8k-only global standardiser are hash-frozen in the config.

## 4. The one pretext task

* Mask **15%** of raw atoms and **15%** of raw *physical* bonds per Fit8k molecule, at
  least one of each per molecule; masks are a deterministic function of
  `SeedSequence([seed, molecule_id, visit_index])`, independent of the data-shuffle
  generator.
* Mask tokens: atom `27`, bond `0`; both are verified unused in Fit8k (`MASK_TOKEN_SLOTS_FREE`).
* The masked raw codes are written into a copy of the raw PyG arrays (both directed
  records of every masked physical bond) and **every** BJOC feature is rebuilt from
  the masked raw graph (`build_mol_objects`).  No incremental patch, no surviving
  derived feature.
* Encoders trained: `atom_encoder`, `bond_encoder`, `junction_encoder` (shared MLP
  context heads, discarded afterwards).  `group_composer` / `reader` are never touched.
* `L_SSL = CE(masked atoms) + CE(masked physical bonds)`, 1.0/1.0, 7560 updates,
  batch 128, Adam 1e-3 / wd 1e-5 / clip 5, last step only.
* No y, no teacher, no external data, no second pretext task, no ratio/length search.

## 5. Frozen downstream arms

All arms: 413,505 params, raw-y L1, Adam 1e-3 / wd 1e-5 / batch 128 / clip 5, FP32,
no scheduler, no dropout, last-5 soup.

| arm | init | updates |
|---|---|---:|
| B | scratch | 15120 (240 epochs) |
| P | B init + three encoders overwritten by the frozen pretrained state | 15120 |
| C | scratch (conditional) | 22680 (360 epochs) |

Per seed, B and P share the exact init state, composer/reader init, fit rows, loss and
batch order (`torch.Generator(seed+101)`); the only primary difference is the encoder
initialization.  C is an **extra-supervision control** (not FLOP-matched): encoder
updates, supervised updates and GPU seconds are all reported.

## 6. Pre-GPU gates (local, all pass)

`results/bjoc_masked_pretraining_v1/`:

* `masked_feature_dependency_audit.json` — full leak-path map + Fit8k mask-token scan
  (`MASK_TOKEN_SLOTS_FREE`);
* `mask_target_swap_audit.json` — deterministic minimal witness + 200/200 atom and
  200/200 bond random hidden-truth swaps on real Fit8k molecules with all BJOC inputs
  (and pretext-head inputs) bit-identical, plus a visible-change positive control
  confined to the dependency cells (`MASK_SWAP_INVARIANT`);
* `masked_object_rebuild_audit.json` — 100/100 cases: deterministic rebuild, changes
  confined to the dependency map, own slots hold the mask token, both directed records
  masked, hidden truth absent from the own slot (`MASKED_REBUILD_LEAK_FREE`);
* `pretraining_feasibility_audit.json` — class priors (atom entropy 1.112 nats,
  majority fraction 0.702; bond 0.586 nats, 0.742), masked-position priors, throughput
  (0.88 ms/molecule masked rebuild; projected 15.1 min/seed data prep), memory and
  model sizes (backbone 413,505 / pretext 557,217) (`MASKED_DATA_PATH_FEASIBLE`);
* `pretext_leakage_gates.json` — 22/22 gates: swap invariance, hidden-truth absence,
  rebuild consistency, double-edge sync, label correspondence, finite CE, non-zero
  gradients in all three encoders, head isolation, 413,505 downstream params,
  permutation invariance, endpoint symmetry, batch consistency, Fit8k-only pretraining,
  no-y pretraining process, save/load replay, mask RNG reproducibility, no official
  split access, no GNN/attention/iteration.

## 7. Pre-registered monitor purchase gate

C is purchased only if **all** hold (monitor split, seed-mean paired over molecules,
`Δ = MAE(P) − MAE(B)`):

1. seed-mean monitor Δ ≤ −0.005;
2. both per-seed monitor Δ < 0;
3. ordinary molecules (`y > −3`) mean Δ ≤ 0;
4. top-10-error-removed Δ < 0;
5. paired molecule bootstrap CI95 upper < 0;
6. no leakage/protocol deviation (all gates pass);
7. pretraining actually changed the encoders and fine-tuning started from the pretrained state.

Otherwise `MASK_PRETRAIN_NO_COMPUTE_PURCHASE`, C is never trained (`NOT_PURCHASED`), and
the frozen decision uses the locked B/P read.

## 8. Locked read discipline

All training and label-blind predictions run first; every prediction file is hashed
into `locked_prediction_manifest.json`; the locked-dev labels are read exactly once in
`score` (single `locked_read.lock`), then B/P (and C if purchased) are scored with the
fixed 5000-resample paired molecule bootstrap.  No further training or model selection.

## 9. Budget

At most 8 formal stages, one GPU, ≤2 GPU-hours, total wall ≤7 h 45 min, ≥35 min reserve
for analysis/report/commit.  No full-10k final training.

## Results (completed 2026-10-13)

**Frozen decision: `SELF_SUPERVISED_PRETRAINING_USEFUL`.**  Locked-dev seed-mean raw-y MAE:
B 0.104023, P 0.092039, C 0.099752; `P − B = -0.011984` (CI95 `[-0.017530, -0.005974]`,
per-seed `-0.007380 / -0.016588`), ordinary molecules `-0.013472`, top-10-removed `-0.013204`;
the frozen monitor gate fired 7/7 (`P − B` monitor `-0.010851`) so C was purchased, and P still
beats C by `-0.007713` (CI95 `[-0.013252, -0.001911]`) — not merely optimization acceleration.
Pretext: atom acc `0.9268/0.9289` (CE `0.2104/0.1990` vs prior `1.1181`), bond acc
`0.9952/0.9946` (CE `0.0154/0.0186` vs prior `0.6051`).  Leak audits: 22/22 gates, 400/400
randomized hidden-truth swaps, 100/100 rebuild cases.  Caveat: the `y ≤ -3` tail is not improved
(locked `+0.0075`, seed-mixed) and P's locked RMSE (`0.1842`) is worse than B's (`0.1663`) while
MAE/median/p90 improve; the junction encoder moves most (relative `2.07×`) and its effective rank
rises `5.0 → 18–21`.  Mechanism and subgroup tables are in the report; the recommended (not yet
authorised) next route is the data-efficiency / external-unlabelled-structure scale-up of this
exact frozen protocol.

Full record: `results/bjoc_masked_pretraining_v1/REPORT.md`.
