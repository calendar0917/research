# ERRATA — corrected evidence scope (ZINC topology cross-fit diagnostic v1)

This file corrects the calibration scope and the mechanism scope of earlier
ZINC reports, using only read-only historical artifacts.  It does **not**
rewrite history; the original files are untouched.  A machine-readable copy is
`corrected_historical_metrics.json` and `mechanism_check.json` in this
directory.

Source of the historical numbers:
`tracks/ksvd/results/zinc_joint_dictionary_decision_v1/{F,B,D}_seed0.json`
(executed version `ac8ef5b`, working tree unchanged).  The historical
`DECISION.md` / `EXECUTION.md` / `gate_seed0.json` are left as-is.

## E0.1 — double calibration in `run_arm`

`zinc_joint_dictionary_decision_v1.py::run_arm` does, in order:

```python
dev_raw, dev_y = collect_predictions(fresh, dev_data, device)   # bias ALREADY folded
dev_cal = dev_raw + delta                                       # adds delta a 2nd time
```

`fresh.reader.net[4].bias.add_(delta)` happens **before** `collect_predictions`,
and `reader.net[4]` is the final `Linear(..., 1)` of `GenericReader`, so the fold
shifts every prediction by exactly `delta`.  Consequences:

* historical `dev_predictions_raw` is already **single**-calibrated;
* historical `dev_predictions_calibrated` is **double**-calibrated;
* the true unfolded raw prediction is `saved_dev_raw - delta`.

**No soup checkpoint was saved** — only `soup_state_sha256` in each JSON and the
soup member epoch list.  Therefore this is a **cached-prediction recomputation**,
not a checkpoint replay.  The minimal algebra check (fold-once) is reproduced in
`corrected_historical_metrics.json` (`saved_cal == saved_raw + delta` to 1e-9),
and the reader structure (`net[4]` = output linear) was verified by code read.

### Corrected historical dev MAE (2000-row inner dev, official train only)

| arm | true unfolded raw | correct single-calibrated | historical double-calibrated |
|---|---|---|---|
| F | 0.11637254 | **0.11636470** | 0.11801785 |
| B | 0.12875943 | **0.12733959** | 0.12888260 |
| D | 0.12471639 | **0.12445951** | 0.12450468 |

Paired gains on the corrected single-calibrated predictions (1000× row bootstrap,
seed 20261003, descriptive only — does not cover seed noise):

| comparison | corrected gain | 95 % row-bootstrap | historical (double) gain |
|---|---|---|---|
| D − F (MAE_F − MAE_D) | **−0.00809481** | [−0.012871, −0.003122] | −0.00648683 |
| D − B (MAE_B − MAE_D) | **+0.00288008** | [−0.002517, +0.008306] | +0.00437791 |

The old “+0.0044” for `D − B` and the old “−0.006487” for `D − F` were both
double-calibrated.  After correction **the old D purchase gate still fails and
fails harder** on the total clause (`D − F = −0.008095 ≤ 0.003`); the D/original-
config line stays closed, no continuation.  The `D − B` edge shrinks and its
bootstrap interval still contains zero.

### Corrected group contributions (severity strata, single calibration)

| arm | penalty 0 (n=1926) MAE / contrib | penalty −1 (n=65) MAE / contrib | penalty ≤ −2 (n=9) MAE / contrib | overall |
|---|---|---|---|---|
| F | 0.0988268 / 0.0951702 | 0.1672995 / 0.0054372 | 3.5016086 / 0.0157572 | 0.1163647 |
| B | 0.1086012 / 0.1045830 | 0.1652313 / 0.0053700 | 3.8636926 / 0.0173866 | 0.1273396 |
| D | 0.1065430 / 0.1026009 | 0.1935398 / 0.0062900 | 3.4596852 / 0.0155686 | 0.1244595 |

Contribution identity `Σ contrib == overall MAE` holds for every arm.  Note
that under the corrected scope D is **not** worse on the ≤ −2 contribution
(D 0.015569 vs F 0.015757) — D's deficit vs F is entirely in the penalty-0 bulk
and the penalty −1 group; the old “D severe worsens” wording came from the
double-calibrated split.  This does not rescue D: the bulk deficit dominates.

## E0.2 — mechanism-scope corrections

Fact table (fact → supporting file/code → what still cannot be concluded):

| # | fact | supporting evidence | still cannot conclude |
|---|---|---|---|
| 1 | Full’s structural dictionary `D` (65×32) is an `nn.Parameter`, in the Adam param group, and receives **both** task and reconstruction gradients | `e2e_dictenv_p1.py:348`; `mechanism_check.json` (task grad 6.37e-5, rec grad 5.27e-3) | says nothing about sparsity or transferability |
| 2 | Full’s task dictionary `D_L`/`V_L` (144×288 / 288×144) are `nn.Parameter`s and receive **task** gradients only (through the unrolled tied-ISTA bridge; no reconstruction gradient) | `e2e_dictenv_latent_bridge_v1.py:153-154`; `mechanism_check.json` (task grads 0.560 / 0.589, rec grad `None`) | per-parameter attribution to a mechanism is not claimed |
| 3 | arm D’s `local.dictionary` (175×256) and `local.value.weight` are end-to-end trained: dictionary gets task + relative-recon gradients, value gets task gradient | `zinc_joint_dictionary_decision_v1.py::JointLocalEncoder`; `mechanism_check.json` | end-to-end ≠ the dictionary learned sparse/transferable codes |
| 4 | the X175/D availability 1.26 % reconstruction error is measured on the **post-prep initial** model, not the final soup | `availability_checks.json`; no soup `.pt` exists (`soup_checkpoints_present: []`) | final-soup reconstruction error is **missing**, not verified |
| 5 | X175/D simultaneously changed the input correspondence, the local function family, the capacity, and the loss — all at once | `zinc_joint_dictionary_decision_v1.py` F vs B vs D construction | its negative result cannot be attributed to dictionary, capacity, or information separately |
| 6 | exact topology-25 class lookup recovers `valid:0172` (−6) and `valid:753` (−2); of the 5 severe valid rows, 3 (214, 249, 935) have **no** exact train class; the ExtraTrees severe error is dominated by 0172 (ET −2.99 vs true −6) | `cycle_probe_predictions.csv`; `cycle_input_decision.json::A3` | the `min_samples_leaf=2` singleton shrinkage is only a competing explanation; it is not re-litigated here |
| 7 | raw and standardised T25 keys induce the **identical** partition of the 10000 train rows (355 classes, affine residual 6.5e-7); train penalties are pure in 350/355 classes | `cycle_input_decision.json::A2` | “no exact match / one tree failure” does **not** show T25 carries no generalisable information |
| 8 | T25’s permutation-invariant longest-cycle / min-cycle-basis statistics are **not** the same quantity as the label’s particular `cycle_basis` penalty formula | `zinc_topology_features.py` module docstring; `zinc_long_cycle_audit` label | T25 is not a reconstruction of the label; the label formula is not used as a model input this round |
| 9 | the same 8-D topology block inside a nonlinear shared reader does not guarantee the same “topology contribution”; earlier four-forward evidence supports other blocks and conditional interactions also contributing to the gap | prior node 2×2 / additive-topology reports (historical, not re-run) | does not prove the T channel is innocent |

### E0.3 — what Phase 0 deliberately did not do

No GPU was started.  No historical model was retrained to recover a missing
checkpoint.  No official test was instantiated / loaded / evaluated.  No
official-valid re-evaluation was performed; all corrected numbers use the
existing 2000-row inner dev only.