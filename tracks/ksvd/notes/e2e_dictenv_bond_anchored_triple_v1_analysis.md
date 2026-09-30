# BondAnchoredTriple-v1 — analysis

Round `e2e_dictenv_bond_anchored_triple_v1` (study `zinc-context-gap`), executed
locally on CPU on 2026-09-30.  Frozen specification:
`notes/e2e_dictenv_bond_anchored_triple_v1_preregistration.md` (commit `623ef09`,
implementation `c02ae61`, report fix `9c858e3`, metrics alias `e742a9e`).  Control plane: `zinc_bond_anchored_triple_v1_mainline`,
promoted run `20260930-143250-4edd72a2` (the training run is
`20260930-143004-e4df458c`; the promoted run is its cache-hit re-execution
after the metrics-alias fix, same artifacts, no retraining).

## Question

Does a frozen Sem108+C6 parent plus a *new, only-trainable* three-environment
static composition (bond-anchored triples) reach `M_soup <= 0.120` on official
ZINC valid within a fixed 80-epoch seed-0 budget?  This is a single-candidate
mainline performance screen, explicitly **not** a matched control and **not** a
mechanism test.

## Frozen pieces (unchanged during the round)

| piece | value |
|---|---|
| parent checkpoint | `results/e2e_dictenv_sem108_v1/checkpoints/SEM108-seed0_soup_state.pt` sha256 `6ec0fdef…4f89` |
| canonical parent state | `7a721984c5579dd76f9bcaff13a2055b8efba71da553eefb3645d94207f5050a` (49 keys, 97 709 params) |
| parent replay (C6, valid, CPU) | `0.123704927947314` (historical `0.123704927947314`, abs diff `0.0`) |
| cached features | `p_ij [P,16]` (pair_value), `z_old [G,302]` (Reader input `unified`), targets, pair pointers/buckets |
| cache key | full parent checkpoint sha256 + canonical state sha256 + C6 mask signature + module sha256 + cache format + split + graph order |
| triple objects | per real bond `(i<j)` × every third node `k`, `m*(n-2)` per graph, zero 64-D summary when `n<3` or `m==0` |
| new path | `F: 48→64→32` (5 216 params), `t=0.5(F([p_ij,p_ik,p_jk])+F([p_ij,p_jk,p_ik]))`, `z3=[mean(t); mean(t*t)]` (64-D), Reader `366→13→13→1` (4 967 params, +832 vs parent) |
| trainable / full model | 10 183 / 103 757 params |
| protocol | seed 0, 80 epochs, batch 128, Adam lr 1e-3, coupled wd 1e-5, global clip 5 over F+Reader only, graph-level L1, no dropout/gates/scheduler/early stop |
| soup | Top-5 trainable checkpoints from epochs 41–80 by valid MAE (ties → earlier), weights averaged |
| gate | `M_soup <= 0.120` → `FROZEN_TRIPLE_SCREEN_PROMISING`, else `FROZEN_TRIPLE_SCREEN_NO_STRONG_SIGNAL` |

Counts (from `cache_report.json`): train 10 000 graphs / 2 668 346 pairs /
5 511 568 triples; valid 1 000 / 264 776 / 546 830.  Train-only fixed
standardizers for `p`, `z_old` and the initial `z3`; `z_old` has 6 near-constant
channels (the C6 count blocks) with `scale = 1` recorded; no valid statistics.

## Result

| metric | value |
|---|---|
| `M_soup` (Top-5 soup valid MAE) | **0.12300291641423246** |
| `M_parent` (replayed frozen parent soup) | `0.123704927947314` |
| `Delta = M_soup − M_parent` | `-0.0007020115330815396` (unmatched screen difference) |
| best valid | `0.1239305234811618` @ epoch 30 |
| last-20-epoch mean | `0.1276113124455675` |
| final train MAE | `0.0601091604590416` |
| soup members (window 41–80) | `[50, 57, 43, 49, 71]`, member valid `[0.124576, 0.125264, 0.125437, 0.125595, 0.125666]` |
| epochs / wall / peak RSS | 80 / 330.9 s (4.14 s/epoch) / 3 517 MB |
| verdict | **`FROZEN_TRIPLE_SCREEN_NO_STRONG_SIGNAL`** (0.1230 > 0.120) |

Curve landmarks (full curve in `results/…/curve_seed0.csv`): epoch 1
train 0.6732 / valid 0.3594; epoch 10 0.0839 / 0.1336; epoch 20 0.0728 /
0.12498; epoch 30 0.0666 / **0.12393**; epoch 80 0.0601 / 0.12733.  Valid MAE
enters the parent band (0.124–0.135) by epoch ~20, stops improving at epoch 30
and drifts upward afterwards while train keeps falling: the 64-D static triple
composition buys no measurable valid accuracy over the frozen parent features.

## Correctness (all pre-registered checks passed before training)

- `parent_replay`: 0.123704927947314, abs diff `0.0` (≤ 1e-7).
- `cache_alignment` (valid, 16 graphs, fresh per-graph parent forward vs cache):
  pair tokens max diff `4.25e-07`, `z_old` max diff `3.81e-06`, cached `z_old`
  through the original Reader vs direct parent prediction `2.98e-07` (≤ 1e-6),
  targets bit-identical.  (Batched vs per-graph float32 batch-composition
  rounding; each path is bitwise self-consistent.)
- `triple_audit_train` / `_valid`: `5 511 568` / `546 830` triples, brute-force
  anchor and `m*(n-2)` counts by graph, no OOB row, no cross-graph offsets,
  0 degenerate-summary graphs on valid.
- `endpoint_swap`: the `(i,j)`/`(j,i)` construction is exactly symmetric
  (max abs diff `0.0`); `hand_pooling` and `batch_pooling` (vs per-graph)
  max abs diff `0.0` including the empty-triple graph.
- `one_batch_gradient`: finite L1 loss `1.4602`, all 10 F/Reader tensors have
  non-zero gradients (min grad norm `0.0669`).
- `parent_freeze`: every parent parameter `.grad is None`, optimizer/clip
  parameter set exactly F+Reader, `parent_state_unchanged: true` (state sha and
  checkpoint file sha identical before/after training).
- `parameter_accounting`: F 5 216 + Reader 4 967 = 10 183 trainable, full model
  103 757, reader increment +832.

## Reading

1. **No strong signal.**  The candidate reproduces the parent band and only
   nominally undercuts the unmatched parent soup (`-0.00070`); it is far above
   the pre-registered `0.120` boundary.  For the frozen question the answer is
   negative/at-parity, and no further spend on this candidate is justified.
2. **The delta is not attributable to the triple operator.**  The Reader is
   re-initialised, `p`/`z_old`/`z3` are standardised, the parent is not in the
   training graph, and the budget differs from the parent's 320 epochs.  The
   `-0.00070` is an unmatched screen difference, not a treatment effect.
3. **This is an accuracy verdict, not an implementation failure.**  All
   plumbing, ordering, freeze and gradient checks pass; the parent state is
   bit-identical before/after; only the frozen official-valid metric is read and
   the official test split was never instantiated (`test_access: blocked`).
4. **Curve shape.**  The plateau beginning at epoch 30 while train MAE keeps
   dropping (0.0666 → 0.0601) shows the extra composition capacity is being used
   for train-only structure, consistent with the static, query-free composition
   adding no generalising information over the parent's cached pair/global
   features under this budget.

## Provenance notes

- Device: local CPU, 8 threads.  The remote A100 host was unavailable
  (`nvidia-smi`: `Unable to determine the device handle for GPU0000:4B:00.0:
  Unknown Error`); per user instruction no GPU was used.
- The first formal attempt (`20260930-141312-a9db89b7`) trained identically but
  failed while rendering the report (mask-key `KeyError`); fixed in `9c858e3`
  and re-executed from cache (`20260930-143004-e4df458c`), then re-executed once
  more with the `valid_mae` metrics alias (`20260930-143250-4edd72a2`,
  promoted).  No hyper-parameter or data change occurred in any re-execution.
- All scientific constants are in the algorithm module; the runner stages only
  orchestrate and record.  The preregistration was committed before the caches
  were built and before training.

## Recorded next shapes (not scheduled, not authorised here)

- A *mechanism-bearing* composition operator computed from the frozen parent
  inside the forward pass with liveness gates, plus a matched-control design at
  equal parameters, under a NEW preregistration.
- A separate optimizer/horizon round that isolates the 80-vs-320-epoch and
  re-initialised-Reader confounds before any accuracy claim about the operator.
