# Pre-registration — ZINC overnight bottleneck v1 (2026-10-02/03)

**Status:** frozen before the formal GPU runs · protocol `zinc-overnight-bottleneck-v1`
· runner `zinc_overnight_bottleneck_v1` · branch `task/zinc-overnight-bottleneck-v1`
· official ZINC **test** is never instantiated, loaded or evaluated.

This round has two independent directions, both from the canonical fresh Full
initialisation, plus a conditional combination and a single second-seed pair.
Only official **valid** is used, as an exploration set; no significance, no
independence or SOTA claim.

## 0. Starting point and bottleneck

Latest 240-epoch, last-5 soup result (`zinc_node_binding_residual_pair_v1`):
product control calibrated valid MAE **0.109042**; the fixed-amplitude additive
residual candidate 0.114261.  Under the same fresh init the product node path
collapses: aggregated slot RMS `2.42e-4 -> 1.16e-10` (ep40) `-> 0` (ep80), and
`node_out` becomes columnwise constant by ep40.  Stage-A audit (this round):
`rms(node_out) = 7.26e-2` but `node_encoder` pre-activation signal RMS
`= 1.37e-4` vs bias RMS `= 4.86e-2` (~355x).  The reader/global/interface
blocks carry the signal; the node branch is a per-molecule constant.

Error budget of the 0.109042 control (recomputed from the per-graph CSV):
G0 965 rows contribution 0.084897 (77.9%), G1 34 rows 0.007172, G172 1 row
0.016973.  The 0.019042 gap to 0.09 is spread over G0 (0.0568 median abs error;
302 rows above 0.1); the single id172 row alone is 0.01697.

## 1. Direction N — fixed 2x2 node-collapse diagnosis

All arms use the **original product binding** and the same canonical fresh
seed-0 initialisation, the same 408,651 trainable parameters and the same
recipe (Adam lr=1e-3, batch 128, clip 5.0, L1 + H1_LAMBDA*rec, 240 epochs,
fixed last-5 soup 236..240, train-median bias).

| factor | level 0 | level 1 |
|---|---|---|
| A amplitude | original product | fixed unit scale `kappa = 1 / r_init` |
| B node weight decay | original `1e-5` | `0` on `W_A_S`, `W_A_C`, `node_encoder.*` |

`r_init` = per-entry uncentred RMS of the aggregated 3-shell product slots over
the fixed seed-0 1024-graph official-train sample (graph-id sha
`6b884fdf…`); for the canonical seed-0 init it is `2.469e-4`, so
`kappa = 4050.2396551095` (frozen manifest
`results/zinc_overnight_bottleneck_v1/frozen_node_scale_seed0.json`).
`kappa` is a **non-trainable buffer** multiplying the same product; by linearity
of the shell aggregation this is exactly a uniform slot scaling (`kappa=1`
reduces to the frozen product — checked).  No extra parameter, gate, activation
or loss.  The WD factor touches only the three named node parameter groups,
including their biases; **fusion node columns are excluded** and keep `1e-5`.
The optimizer is the original `torch.optim.Adam` with exactly two groups; at
equal WD it is mathematically the original single-group update (checked).

Arms: `N0` = (0,0) shared control; `N1` = (1,0); `N2` = (0,1); `N3` = (1,1).

### 80-epoch prefix

Four jobs, one per arm; **the prefix never reads official valid** (it only
records train health).  Frozen health gate, checked at epochs 40 and 80 on two
fixed diverse train sentinel batches (rows 0..127 and 2048..2175, fixed eval
order):

* `node_out` alive: `max column std > max(1e-8, 1e-3 * that arm's initial max column std)`;
* `W_A_S` / `W_A_C` / `node_encoder.0.weight` task gradients finite and at
  least one norm `> 1e-10`;
* the product term is not identically zero.

The gate is a numeric screen, not a proof of task value; a dependency response
`<= 1e-4` is recorded as `WEAK_TASK_USE`.

### Selection (frozen, no valid used)

`N1` if healthy, else `N2`, else `N3`, else close the node direction.
`N0` always runs 240 epochs (shared control).  At most **one** healthy winner
continues 80 -> 240 (same trajectory: model + Adam + step + generator/RNG;
no fresh optimizer).  Non-selected prefixes stop at 80 and are not published
as 240-epoch performance.

### Performance gate (only a complete 240-epoch arm with the fixed soup)

calibrated total gain `>= 0.003`, `gain_without172 > 0`, G0 contribution
worsening `<= 0.001`.  Health is reported separately from performance.

## 2. Direction T — one additive topology readout (T0)

`pred = GenericReader_other(R[:, :806]) + Linear(8, 1)(R[:, 806:814])`.

The body/reader layers are copied from the canonical parent fresh state (other
first layer = the first 806 columns + bias; the two hidden layers and output
layer are identical modules); the 8->1 head is initialised from a fixed private
seed (`head_seed=0`) that does not consume the shared global RNG.  T0 keeps the
**original product binding, node WD and amplitude** (no node winner is folded
in), so the two directions stay independent.  Parameters 408,348 vs 408,651
(303 fewer); no capacity is bought in the head.  Loss, optimizer, 240 epochs,
seed and last-5 soup are identical to N0.

Frozen purchase rule (bought when the shared control identity holds and no
`<=0.09` model has been confirmed): calibrated total gain `>= 0.003`, G0
contribution worsening `<= 0.001`, `gain_without172 >= -0.0005`.  Grading:
`gain_without172 > 0` is `BROAD_PAIRED_SIGNAL`; overall improvement with
near-zero/slightly-negative `gain_without172` is `OUTLIER_DOMINATED_SIGNAL`
(needs confirmation, not a generalisation claim).

## 3. Conditional combination (C)

Bought **only if** both directions pass their full performance gates and the
per-group accounting is complementary: node winner G0 contribution gain
`>= 0.001`, T0 (G1+G172) contribution gain `>= 0.003`, both single-direction
total gains `>= 0.003`, no obvious group regression beyond the gates.  The
combination is one fresh 240-epoch run from the canonical init (frozen node
winner + fixed AdditiveReader), not a splice of two trained checkpoints.
Skipped if a single candidate is already `<= 0.09` (confirm that instead).

## 4. Single second-seed confirmation

Exactly one frozen candidate is confirmed: the lowest calibrated valid among
those passing their purchase gate (ties `< 0.0005` prefer fewer changes;
combination preferred only if it beats the best single by `>= 0.001`).
A fresh paired seed-1 `candidate` vs `N0`, both 240 epochs, two independent GPU
jobs.  Upstream inputs / shared dictionary are unchanged; `kappa` for seed 1 is
recomputed once from the seed-1 fresh init under the same frozen rule.

Confirmation is recorded as a more credible purchase signal only when the
two-seed mean gain `>= 0.003`, both seed gains `> 0`, and the group bounds still
hold.  **A training seed is not an independent validation set.**  A single seed
`<= 0.09` with a two-seed mean above it is reported as "single-seed exploratory
0.09", never as a stable 0.09.

## 5. What this round does not do

No message passing/transformer; no label-derived inputs; no target-formula
shortcut; no manual outlier-row handling; no new hand-crafted ring features;
no valid-residual / isotonic / group calibration; no best-by-valid model
selection; no third seed; no ensemble.  `id172` is reported but never special
cased.  Test access is `blocked` for every non-terminal run.