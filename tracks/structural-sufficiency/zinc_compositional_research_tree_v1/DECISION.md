# DECISION — SSRC Research Tree v1 (CMA / SPC)

Every decision below is the mechanical application of the frozen rules in
`PROTOCOL.md` §7 to numbers computed by `shared/xs_analyze.py` (which reads only
the frozen Monitor/valid/Locked rows). No decision was taken after looking at a
result that the rules did not pre-specify, and official test was never read.

---

## D1 — Stage II: SPC is not promoted (H2 rejected)

Frozen promotion conditions (all five must hold for the family's seed-0 pair):

| # | condition | SPC s0 | verdict |
|---|---|---|---|
| 1 | REAL − NULL on full Monitor ≤ −0.004 | **+0.00131** [−0.00371, +0.00613] | ✗ |
| 2 | ordinary-slice delta ≤ 0 | **+0.00063** [−0.00359, +0.00478] | ✗ |
| 3 | unseen-ordinary delta ≤ −0.004 | −0.00364 [−0.01354, +0.00577] | ✗ |
| 4 | REAL full Monitor ≤ DCC-L + 0.010 = 0.133151 | **0.14511** | ✗ |
| 5 | last-100 / roll-20 smoothing agrees in sign | smoothing flat, REAL worse | ✗ |

**Decision:** SPC stops at Stage II. No seed 1, no Locked read, no further SPC
runs. The family's full results are kept in `analysis/` (`stage2-spc-*`) as
required by §7. Reason in one line: the separator-port composition produces no
matched REAL/NULL increment and is also the weakest arm in absolute terms
(0.1451, i.e. +0.0219 vs DCC-L), so there is nothing to confirm.

## D2 — Stage III: CMA is promoted, and the Locked read was spent once

| # | condition | CMA s0 | verdict |
|---|---|---|---|
| 1 | REAL − NULL full ≤ −0.004 | **−0.01094** [−0.01623, −0.00594] | ✓ |
| 2 | ordinary delta ≤ 0 | **−0.00825** [−0.01261, −0.00384] | ✓ |
| 3 | unseen-ordinary delta ≤ −0.004 | −0.00691 [−0.01903, +0.00533] (CI crosses 0) | ✓ point estimate |
| 4 | REAL full Monitor ≤ 0.133151 | **0.12492** | ✓ |
| 5 | smoothing agrees in sign | roll-20 0.12532 vs 0.13636 | ✓ |

Both checkpoints were frozen *first* (`analysis/locked_freeze.json`, SHA-256 in
the file; the reader refuses to run without it), then Locked1k was read once:

| arm | Locked1k MAE |
|---|---|
| REAL | **0.090529** |
| NULL | 0.097547 |
| matched delta | **−0.00702** [−0.01182, −0.00235], win-rate 55.1 % |

**Decision:** Locked confirms the Monitor direction (same sign, CI excludes 0,
ordinary subset −0.0058) → buy the seed-1 pair, per §7. The Locked read is spent:
no further Locked evaluation is permitted in this round, and none was done.

## D3 — Stage III result: two seeds agree → Stage IV authorised

| | seed 0 | seed 1 |
|---|---|---|
| full Monitor delta | −0.01094 | **−0.00449** [−0.00931, +0.00036] |
| ordinary delta | −0.00825 | **−0.00475** [−0.00929, −0.00008] |
| unseen-ordinary delta | −0.00691 | −0.00437 |
| REAL full Monitor | 0.12492 | 0.132780 |
| NULL full Monitor | 0.13587 | 0.137275 |
| tail-only pattern | no | no |
| ordinary degradation | none (improves) | none (improves) |

**Decision:** signs agree, no ordinary-molecule degradation → Stage IV is
authorised (§7). Magnitude is seed-dependent (seed 1 is ~2.4× smaller), which is
recorded honestly: the effect is real but not large, and a single seed would have
overstated it.

## D4 — Stage IV: full train10k with official valid1k selection

@@STAGE4_DECISION@@

## D5 — What was deliberately not done

* **Official test was never opened.** No post-hoc test read was requested and none
  was performed; the round has no test number to report. The 16 GPU·h hard cap and
  the 8 h wall-clock budget were not the binding constraint — the frozen data
  boundary is.
* **No extra seeds, no seed averaging, no ensembles, no distillation, no lookup.**
  Each reported number is one single model.
* **No reallocation of unused Stage II budget** to CMA/Stage IV: Stage IV stayed
  inside its own 3.0 GPU·h cap (see `analysis/budget.json`).
* **No re-tuning after seeing results**: recipe, slices, promotion thresholds and
  the worst-10 / row172 row sets were frozen in `analysis/frozen_baseline.json`
  before Stage I started.
* **No slice was added after the fact.** The `tail20` curve statistic is a
  reporting detail (§6 asks for last-100 and last-20 curves); it was added to the
  analysis script and is computed for every run, including the ones already
  finished, without changing any selection rule.

## D6 — Final position of the tree

1. **CMA (composition of jointly typed directed path environments): supported.**
   Matched REAL/NULL gain on Monitor for both seeds, reproduced once on Locked,
   broad across ordinary/unseen/tail slices, not carried by the fixed worst rows,
   present in 90.8 % of training epochs, and with the NULL arm fitting the training
   set just as well. Absolute level 0.1249 does **not** beat DCC-L 0.1232.
2. **SPC (cut-vertex / separator-port composition): rejected.** No matched
   increment, and the worst absolute level of the four arms.
3. **The tree's frontier** is therefore: "joint local typing matters, and the
   directed-path coupling is the right way to see it; separator algebra is not".
   The remaining gap to DCC-L (+0.0018 Monitor, +0.0069 Locked) is not closed, so
   no claim of superiority is made.
