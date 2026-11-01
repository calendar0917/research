# P2 hypothesis — pre-registered before launching the P2 runs

**Decision: Candidate A, `ESOL-DROP20` (dropout 0.10 → 0.20). One direction only, 3 seeds.**

## Which P1 observation forces this choice

`ESOL-BASE`, 3 seeds, `analysis/runs/base_s*/train.json` + `analysis/p1/p1_summary.json`:

| Seed | Best epoch | Train RMSE @best | Valid RMSE @best | Train RMSE @360 | Valid RMSE @360 | Valid RMSE @best+25 | Epochs after best | Improving epochs after best |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 172 | 0.432 | 0.720 | 0.341 | 0.776 | 0.785 | 188 | 0 |
| 1 | 157 | 0.431 | 0.767 | 0.339 | 0.900 | 0.920 | 203 | 0 |
| 2 | 159 | 0.440 | 0.661 | 0.351 | 0.719 | 0.700 | 201 | 0 |

The decision rule in the task book is: *choose A if the three seeds repeatedly show
Train RMSE still falling while Validation RMSE clearly degrades, supporting
small-sample overfitting.* That is exactly what happened, in **3/3 seeds**:

- the best epoch is at 43–48% of the fixed 360-epoch budget in every seed;
- after that epoch, **zero** of the 188/203/201 remaining epochs improves Validation;
- Train RMSE still falls by 0.089–0.092 while Validation RMSE rises by
  +0.055/+0.133/+0.058 (and rises already within 25 epochs of the best epoch);
- mean Train 0.417 vs mean Validation 0.716, i.e. the Validation/Train gap is 0.30
  on a target whose Train standard deviation is 2.08 — the model is not
  underfitting the Train set, it stops transferring.

There is no evidence pointing the other way (no plateau at both curves, no low
Train / low Validation regime that would suggest an information gap instead).

## What changes

**Only** `p` in the frozen dropout-at-`h` hook: 0.10 → 0.20 (`ESOL-DROP20`).
Unchanged: every module, every input, the S1/S2 statistics, the local readout,
the global `rho`, AdamW/LR/cosine/WD/clip/batch, the 360-epoch budget, the
Train-only standardisation, the single best-Validation-RMSE epoch checkpoint
rule, the seeds, the split, and the data. `ESOL-DROP20` is built from the same
`torch.manual_seed(seed)` stream as `ESOL-BASE`, so the two arms share their
initialisation exactly; the comparison is paired within seed.

## What result would refute this explanation

Primarily: **no Validation improvement in ≥2/3 seeds** (paired ΔRMSE ≥ 0), i.e.
increasing regularisation does not buy generalisation, so "small-sample
overfitting" is not the dominant, *fixable-by-regularisation* bottleneck — the
next candidate would then be information (which is candidate B, not run here).

Secondary refutations, recorded now so they cannot be reinterpreted later:

- the best epoch moves *earlier* and Train RMSE stays high (≥0.55) while
  Validation also worsens → the model is capacity-limited/underfitting, not
  overfitting, and the 0.20 change merely slows fitting;
- Validation improves but Test does not (direction disagreement between
  Validation and Test) → the change is split-specific;
- the improvement appears in only one seed and is smaller than the seed spread
  (sd of Validation RMSE across seeds ≈ 0.053) → noise.

## Explicit non-actions

- No dropout sweep afterwards (0.15 / 0.25 / 0.30 are not tried).
- No candidate B in this round, no third candidate, no extra seeds, no extra
  split, no S2 ablation.
- Test is still not read at this point; it is read once, after both P1 and P2
  checkpoints are frozen (`analysis/manifest.json` + `MANIFEST_FROZEN`).
