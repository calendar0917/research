# PLAN — zinc-performance-search-v1

Performance-first search on the official PyG ZINC subset (`subset=True`,
train 10,000 / valid 1,000 / test 1,000), raw-`y` MAE, single deployment model
≤ 500,000 parameters per run. This round is explicitly a *search*, not a
single-mechanism confirmation.

## Time / budget ledger (declared at start)

- wall start 2026-10-23 16:06 UTC; hard deadline 2026-10-24 00:06 UTC (8 h).
- 2 × A100-40GB on `res` (process backend, pools `res-gpu0`, `res-gpu1`).
- GPU-h cap 16.0; soft 15.0. Every scheduler-occupancy hour counts (failures,
  timing, data prep, idle).
- Reserved: reconfirmation seeds 2/3 for the frozen winner + matched A-R0
  baseline (~5 GPU-h), terminal test read (CPU unless late), ~1 GPU-h margin.
- Discovery closes ≈ 20:00–20:30 UTC (≈4–4.5 h in); freeze and start
  confirmation by ≈ 21:00–21:30 UTC; ≥30 min for the terminal test table.

## Candidate pool (max three bodies)

| arm | body | initialisation | source |
|---|---|---|---|
| A | `ad2_models.P3NullModel`, 413,505 par | seed-matched train-only masked SSL encoders, fine-tuned | `zinc_recipe_trunk_v2` |
| B | `dcc_models` L arm (organized dual-channel), 494,193 par | same SSL encoders + zero-init local columns | `zinc_dual_channel_composer_v1` |
| C | GRIT (`GritTransformer`, RRWP I..P^20), target ≈473k par | scratch, native recipe | vendored upstream `zinc_grit_reference_preparation_v1` |

C is tried only inside a 60-minute dense-engineering window, after A/B are
running. If the block/whole-net dense path cannot be aligned to the sparse
reference, fall back to one declared DBT run (batch 128, 1500 epochs); if that
is also not completable, the round reports A/B only. No fourth body.

## Recipes

`R0` = AdamW lr 1e-3 → 1e-5, wd 1e-5, dropout 0, L1, batch 128, 600 ep,
500-step warm-up, clip 5. This is *identical* to the historical recipe B, so
A-R0 (`xBssl_P3N_s0`, 0.076224 @377) and B-R0 (`L_s0`, 0.078537 @489) are
reused with their sources recorded rather than retrained.

`R1` = R0 with lr 5e-4 → 5e-6, wd 1e-4.
`R2` = R0 with wd 1e-3 and post-encoder dropout 0.05.
`R3` = R1 with SmoothL1(β=0.02) in raw-y units.

New A/B runs carry the declared EMA policy (decay 0.999, initialised after 25 %
of updates) *in addition to* the online weights; online and EMA are separate
checkpoint policies, both fully evaluated, never averaged at prediction time.
C/DBT (BN) get no EMA.

## Stages

1. **S0 (now)**: A-R1 and B-R1 seed 0, one per GPU, complete 600-epoch cosine
   (no early stop). In parallel, CPU checks + dense GRIT engineering.
2. **S1**: R2/R3 for the body/recipe that is informative; C0 on one GPU.
   At most two extra targeted configs beyond the table for A/B.
3. **S2**: rank by seed-0/1 mean best-valid (ties: W20, then cost). Freeze one
   model+recipe+checkpoint policy and run seeds 2/3; matched A-R0 seeds 2/3.
4. **S3**: freeze the shortlist, then one terminal test read. No post-test
   changes.

`W20` = minimum over all 20-epoch contiguous windows of the window's median
valid MAE. `final` = last-epoch valid MAE. Ranking metric is best-valid MAE;
W20/final are diagnostics for isolated low points, not selectors.

## Exclusions

No new raw-structure vocabulary, no target formula, no row-specific gate, no
test-time ensemble / seed-average prediction / checkpoint soup in this round,
no external labels, no recursive local hidden-state MP. Historical runs are
never overwritten.

## Outcome (appended at close, 2026-10-23 21:40 UTC)

- Discovery closed at **21:29 UTC** (5.4 h in), 4 min later than the planned
  20:00–20:30 because arm C's cost measurements and two seed-1/2 runs were
  still in flight. Confirmation seeds and the terminal test still finished
  2.5 h before the hard deadline.
- Winner by seed0/1 mean: **B-R0**; confirmation seeds 2/3 plus the matched
  A-R0 baseline were completed as declared.
- Arm C: GRIT was built and parity-verified but measured at 30.97 s/epoch
  (batch 32) / 18.65 (batch 64) / 14.84 (batch 128) dense and ~9.5 s/epoch
  sparse, i.e. ≥4 h for the native horizon, so the declared DBT fallback ran
  instead (600 epochs, 94,200 updates, 1.48 GPU·h).
- EMA was dropped after the copy-direction bug was found (see DECISIONS D4).
- Final cost 8.39 GPU·h including 0.97 wasted on failures/cancel; no run
  exceeded 1.55 GPU·h and no GPU was left occupied.
