# COC-v1 Decision

**Decision: `STOP_NO_MATERIAL_GAIN`.**

Date: 2026-10-27  
Round: `zinc_counterfactual_ownership_composer_v1`  
Protocol: `ssrc-zinc-counterfactual-ownership-composer-v1`  
Formal training revision: `331fab569441`  
Monitor analysis revision: `86bec2fad3da`

---

## 1. What was tested

Under the frozen DCC-L Fit8k recipe, with all existing local structural paths
unchanged, the production `DualChannel` replaced the local ownership contribution
by the shared-parameter nonlinear difference

```text
c_loc = local_composer(x_real) - local_composer(x_null),
```

without detaching the null branch. The reader stayed 513-dimensional, the
parameter count stayed 494,193, and no new loss, gate, scaling coefficient, norm
layer, attention, feature, or support mechanism was added.

The comparison is the two historical frozen DCC-L checkpoints on the same Fit8k
rows, same seed-matched SSL initialisation, same batch order, same 47,400 updates,
and same Monitor1k evaluation.

---

## 2. Why the round stops

The pre-registered Monitor gates failed. Δ = MAE(COC) − MAE(L); negative would be
improvement.

| gate | result | observed |
|---|---|---|
| G1 both seeds overall Δ negative | fail | +0.006019 (s0), +0.008475 (s1) |
| G2 seed-mean overall ≤ −0.005 | fail | **+0.007247** (95% CI [+0.002829, +0.012760]) |
| G3 no-penalty any-unseen both seeds negative, mean ≤ −0.008 | fail | s0 +0.004586, s1 −0.003224, mean +0.000681 |
| G4 no-penalty overall improves | fail | +0.003453 (s0), +0.004922 (s1) |
| G5 W20/T100 support | fail | W20 +0.007401, T100 +0.007411 seed mean |
| G6 no fitting/instability failure | fail | seed-1 best-Monitor checkpoint underfits Fit8k (ratio 2.158 > 2.0); not a divergence |

The best COC run (s0 best Monitor 0.129170) is worse than the worse frozen L run
(s1 0.127772). Every COC run is worse than both L runs at best-Monitor and final.
The sign is opposite to the required improvement. W20, T100 and final agree; this
is not a single-checkpoint artefact.

The difference representation is not degenerate and the gradient path is not
broken:

- C8 initial median bond RMS `c_loc` 0.353/0.284, nonzero fraction 1.0, ratio to
  `c_abs` 0.766/0.808.
- Post-training median bond RMS `c_loc` 0.090/0.096, still nonzero (fraction 1.0),
  ratio to `c_abs` 0.371/0.503, within-molecule variance 0.014/0.017, participation
  ratio ~8–9.
- C4 junction correspondence changes `c_loc` by 5.413 with fixed `w`.
- C5 step-1 local-composer gradient is zero only because the reader local columns
  start at zero; after one optimiser step the shared composer gradient is 4.725,
  and the formal runs record 0.676/0.953. All gradients and losses are finite.
- 47,400 updates complete in both seeds; final Fit8k MAE 0.000506/0.000407.

The causally supported failure category is **learned-but-non-generalising**, not
a degenerate representation and not an unlearnable path. We do not claim a more
specific mechanism (such as molecule-specific noise fitting) because no
intervention isolating it was run.

G6's failure is a checkpoint-selection/underfit difference, not instability:
all curves are finite, Monitor tail100/best is 1.004/1.009, and the final Fit
ratios are 1.160/0.813. No Monitor PASS is possible regardless because G1–G5 also
fail.

---

## 3. Access barrier and stopping discipline

- **Locked1k was not read.** `coc_locked.py` was not run. There is no Locked
  prediction file and no authorization sentinel.
- **Official valid/test were never loaded, predicted or scored.** The static scan
  and `torch.load` audit report `official_valid_loaded: false`,
  `official_test_loaded: false`, `locked_labels_read: false`.
- **No extra seeds.** Exactly the two pre-registered seeds were run.
- **No architecture changes** after the freeze. The training code and recipe that
  produced the checkpoints are the frozen revision `331fab569441`.
- **No full10k proposal.** The Locked/full10k thresholds were not evaluated.
- Historical L checkpoints were reused read-only; no L retraining and no L GPU
  cost.

Locked1k had been used in earlier rounds, so it would not have been a pristine
blind test even if the Monitor gate had passed. That limitation is not used to
rescue this result; it is recorded for future rounds.

---

## 4. Compute accounting

| item | A100·h |
|---|---:|
| failed profile (cache-hash assertion, 8 s, no optimiser step) | 0.0022 |
| successful profile (19 s) | 0.0053 |
| `COC_fit8k_s0` formal run | 0.3625 |
| `COC_fit8k_s1` formal run | 0.4122 |
| **total job occupancy** | **0.7878** |
| round cap | 3.0 |

The profile projected 0.4801 A100·h per formal run (< 1.0 threshold). Actual
formal training was 0.7803 A100·h. The failed job is included in the total. Full
ledger: `analysis/run_ledger.json`.

---

## 5. What is closed

For this exact hypothesis, recipe, data split and two independent seeds:

> A shared-parameter nonlinear function-space centering of local ownership,
> `f(x_real) - f(x_null)`, with no added capacity or regularisation, does not
> improve cross-molecule generalisation under the finite Fit8k supervision
> protocol. It is learned and non-degenerate, but it is worse than the frozen
> DCC-L baseline on Monitor1k.

This closes the specific mechanism claim. It does **not** close "all centering
methods", "all ownership-conditioned composers", or "all support-conditioned
methods". SCSB used a different input-space shrinkage operator; RCC-v0 used a
different graph-level object; the frozen transfer audit used a fixed reader; block
ownership used a different object and was unresolved due to variance.

---

## 6. What would be a genuinely new hypothesis

Not a re-run or a cosmetic change of this operator. A future round would need a
new pre-registered mechanism, for example:

- an explicit regulariser or capacity constraint on the difference channel;
- a different target/estimator for the counterfactual ownership contrast;
- a different data regime (e.g. full10k) only after a mechanism passes its own
  Monitor gate;
- a different composition or support mechanism that is mathematically distinct
  from shared-parameter `f(real) - f(null)`.

Any such round must re-run its own CPU gates, freeze its own config, and respect
the same Locked/official-test barrier. It must not cite this negative result as
evidence for a different operator.

---

## 7. Bottom line

`STOP_NO_MATERIAL_GAIN`. The local composer is sensitive to ownership difference
(answer A: yes), but it does not improve unseen-structure generalisation (B: no),
it does not reach the required −0.006 gain (C: no; observed +0.007247), and the
supported cause is learned-but-non-generalising rather than degeneracy or an
unlearnable path (D). No Locked read. No full10k proposal. No extra seeds.
