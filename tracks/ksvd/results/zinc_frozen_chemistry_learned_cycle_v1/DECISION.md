# DECISION — zinc-frozen-chemistry-learned-cycle-v1

Date: 2026-10-03.  Frozen protocol:
`tracks/ksvd/protocols/zinc-frozen-chemistry-learned-cycle-v1.yaml`
(committed before the two formal heads, `e391550`).

## The four questions

### 1. How much does real inference `P` improve?

Two seeds, y units, positive = better:

| | seed 0 | seed 1 | mean |
|---|---:|---:|---:|
| `Y_cal` dev MAE | 0.125680 | 0.120339 | 0.123010 |
| `Y_raw` dev MAE | 0.127943 | 0.122033 | 0.124988 |
| `P_cal` dev MAE | 0.113180 | 0.111120 | 0.112150 |
| `P_raw` dev MAE | 0.113610 | 0.111760 | 0.112685 |
| `O_cal` dev MAE | 0.101626 | 0.098999 | 0.100312 |
| `K_cal` dev MAE | 0.263890 | 0.259970 | 0.261930 |
| `gain(Y→P)` cal | +0.012500 | +0.009220 | **+0.010860** |
| `gain(Y→P)` raw | +0.014330 | +0.010270 | +0.012300 |
| `gain(K→P)` cal | +0.150706 | +0.148853 | +0.149779 |
| oracle retention | 0.5197 | 0.4320 | **0.4758** |
| `G0` MAE(P)-MAE(Y) | -0.001839 | -0.005356 | -0.003597 |
| `G0` MAE(P)-MAE(O) | +0.0000236 | -0.0000213 | +0.0000011 |

`P` improves the real inference objective on the reused dev by about
`+0.0109` mean (raw `+0.0123`), i.e. it retains about **half** of the frozen
oracle gain, and it loses only `+0.0118` mean relative to the oracle.  Both seeds
improve; raw and calibrated directions agree.  `G0` is not harmed — it improves
(`G0` is fully preserved, within `2.5e-5` of the oracle).  `P` beating the
constant-offset control `K` by `~0.15` shows the gain is genuine learned-cycle
content, not a constant shift or the de-cycled branch.

### 2. What did the head learn, fit vs dev (especially severe)?

* fit overall `q` vs `c` MAE `0.01371 / 0.01449` vs constant `0.15261`
  (**90.5–91.0 % drop**); dev overall `0.01263 / 0.01403` vs `0.16648`.
* The discrete cycle levels are learned: fit `k=-1` (`c=-3.468`) `0.046/0.064`,
  fit `k=-2` (`c=-6.936`) `0.603/0.625`, dev `k=-1` `0.012/0.038`.
* fit severe (`k<=-2`, n=38) `2.395/2.418` vs constant `8.397`; the aggregate is
  dominated by the 5 `k<=-3` rows (`14.2` MAE) and two `k=-2` misses.
* dev severe (n=9) `2.617/2.717` vs constant `11.947`; 7 of 9 `k=-2` rows are
  recovered to `~0.02–0.08`, while the `k=-5` and `k=-12` rows are missed.

So fit failure and transfer failure are **distinguished**: the fixed finite
network learns the common cycle classes and transfers them; the residual is the
rare extreme tail, which is neither fit nor transferred.  The fixed descriptive
criteria are overall-drop `>= 50 %` **met** and fit-severe `<= 1.0` **not met**
(`2.395`), so the extreme-tail readout is not complete.

### 3. Is true `c` completely absent from inference?

Yes.

* `deploy_wrapper.py` `forward(batch, topo25)` produces
  `frozen_h(x) + q(T) + b_P` and takes no label table.
* Wrapper vs cached `P`: `8.5e-7` / `1.4e-6` (`<= 2e-6`); repeated calls
  bit-identical; permuting dev `c`/`g`/`k`/`y` labels changes the output by
  exactly `0`.
* Head soup re-exported and replayed over all 10000 `T25` rows: `max_abs = 0`.
* Exactly one fit-only calibration per seed (`b_P`); `q` is never separately
  calibrated; no dev offset, coefficient, threshold or gating is used.
* `identity_checks.json`: `T25` is the actual frozen Full input, `h_raw`
  reproduces the raw-soup forward, and `h_raw + b_O + c` reproduces the released
  oracle dev MAE exactly.

### 4. What is the single next action?

**Buy one formal full decomposition-model training / confirmation round with the
already-frozen interface, and do not add more local audits first.**  What
supports it (fixed conditions, no new evidence needed):

* the pre-registered inference gate passed on both seeds and on the mean
  (`gain(Y→P)` mean `+0.01086 >= 0.003`; both seeds `> 0`);
* it is not a constant/de-cycle artefact (`gain(K→P)` mean `+0.1498`, both seeds
  `> 0`);
* `G0` is preserved (both seeds improve; secondary `G0` marker passed);
* identity, no-true-c inference, single calibration and replay all pass.

Because `q` would reproduce only this round's oracle level (`~0.1003`) if it were
exact, the next round must be a real training attempt on the chemistry branch,
not another local head audit.  The fit-severe boundary means the extreme tail is
still open and must be part of that training/confirmation's question — this is a
caveat on scope, not a reason to keep probing locally.

If a stricter reading of the mixed-case rule is applied (gate met but the
fit-severe descriptive reference `<= 1.0` not met), then this round is a boundary
case and the missing discriminator is *whether the rare extreme `c` rows
(`k<=-3`, `c` down to `-41.6`) are recoverable from this input at all, or only the
discrete `k=-1`/`k=-2` levels are*.  That discriminator is not tested here and
must not be bought with another local head audit.

## What this does NOT license

* Not a claim that topology25 is information-free, nor that the cycle term is
  unlearnable.
* Not a claim that this fixed head is optimal, or that the input is unambiguous.
* Not a claim of reaching the official-valid `/ 0.09` target, nor dictionary
  specificity.
* Not a new architecture / width / `lambda` / WD / loss / optimizer search, no
  third seed, no Full fine-tune.

## Route closure

The round does **not** close the decomposition route.  It establishes a positive,
deployable inference signal and localises the remaining gap to the rare extreme
cycle tail.  Only the specific *frozen head configuration* is bounded: no
additional local head-only audits are authorised; the next compute goes to the
full-training/confirmation branch above.