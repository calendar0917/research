# AMENDMENT-001 — two-seed execution on `res`

**Date:** 2026-10-31 (Asia/Shanghai)
**Status:** researcher-directed execution amendment; no change to the frozen
model, recipe grid, fold split, normalization, SSL method or decision
thresholds.

## Instruction

> "用 res 的 GPU 跑吧，然后跑 2 个 seed 就行了，不用三个了"
> ("Use the `res` GPUs, and run 2 seeds; three are not needed.")

## Scope

1. All remaining planned GPU runs use `res` (2×A100-SXM4-40GB, driver
   550.163.01) with the `res-gpu0` / `res-gpu1` process pools. The `res-2`
   Slurm queue was abandoned while the first job remained pending.
2. Primary seeds are `0,1` for every arm, fold, SSL pretrain and full-Fit
   retrain. The registry no longer plans seed 2.
3. The following planned seed-2 jobs are dropped:
   `F0_A_R0_s2`, `F0_C_R0_s2`, `F0_E_R0_s2` (already completed before this
   amendment), `F0_SSLpre_s2`, `F0_A_SSL_sel_s2`, `F0_A_sel_s2`, and the five
   `full_*_s2` runs.
4. `seeds` remains `[0,1,2]` in the configuration so the already-completed
   pre-amendment seed-2 Fold-0 R0 artifacts remain valid and auditable.
   `primary_seeds: [0,1]` is the new analysis/registry scope. The three
   completed seed-2 Fold-0 R0 runs are reported only as exploratory extras and
   are excluded from H1/H2/H3 selection and verdicts.
5. Fold-0 recipe selection still uses seeds 0/1, so this amendment does not
   change the selected recipe rule. H1/H3 use seeds 0/1 on every fold. Full-Fit
   Monitor predictions use seeds 0/1.
6. The hard cap remains 3.5 A100 GPU-hours. The pre-amendment jobs and the one
   duplicated `F0_A_R0_s1` run are charged to the ledger; the cap is not reset.
7. No optional PhysChem-augmented run is added. The CPU-vs-A gap condition was
   met (CPU 0.72357 vs A-R0 Fold-0 mean ~0.801), but that diagnostic was outside
   the frozen registry and is left for a separate round.

## Disclosure

This amendment was issued after Fold-0 R0 (three seeds) and part of the A
R1/R2 recipe screen had completed, but before the recipe lock, SSL gate, Fold
1/2 runs, full retrains or Monitor prediction freeze. It reduces compute; it
does not use outcome data to change a decision threshold. The report must show
the 2-seed primary result and, separately, the available 3-seed Fold-0 R0
sensitivity result.
