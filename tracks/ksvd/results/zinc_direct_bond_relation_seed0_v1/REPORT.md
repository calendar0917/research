# REPORT — `zinc_direct_bond_relation_seed0_v1`

## Status at a glance

The completed O/T pair shows a **small positive calibrated point estimate**, but the frozen calibrated-gain gate is not met. The row-bootstrap result is **`DIRECTIONAL_NOT_CONFIRMED`**, with 0/3 candidate-gate conditions passed; the confidence intervals are too wide to establish practical equivalence. Separately, execution did **not** comply with the frozen one-pair / minute-90 rules: two partial training jobs were cancelled, the O/T pair was then relaunched from initialization after the cutoff, and the cancellation rationale was mistaken. Under the method contract this round's strict status is therefore **`NOT_COMPLETED`**. Treat the numerical pair below as descriptive evidence, not a fully protocol-compliant formal verdict.

No official valid/test data were loaded. No additional compute was run after the two completed jobs. No marginal-input diagnostic was run because T did not pass the gate.

## Question and intervention

Does giving the frozen fresh-fold M model explicit direct-bond type at each existing environment pair improve its internal chemical residual `g = y − c`?

* **O:** source relation + four zeros.
* **T:** the same source relation + primitive, unstandardized 4-D direct-bond beta.

The source C6 mask is retained (including path-count and global atom/bond histogram masking); raw `path_bond_mean` columns 14:18 are excluded. Beta is `pair_relation[:,19:23]`, independently derived from raw **train-only** edges and exactly matched across all 10,000 training graphs. Of 249,279 adjacent pairs, category mass is `[0, 185060, 63548, 671]` (three classes observed). The interface adds no scaler, loss, message passing, or environment write-back.

Both completed arms use 297,883 parameters; source M initialization is expanded 15→19 by copying the original columns and bias and zeroing the new four columns. Fit/dev, target, schedule and global train-GID stream match the frozen source. The unchanged path and O/T initialization were verified before training; O's new-column task gradient was zero and T's nonzero.

## Primary dev results

Internal dev has 2,000 reused train-partition examples; G0 has 1,935. Each arm's sole calibration offset is the fit-set median residual. Positive O−T gains favor T.

| arm | fit G0 raw / cal | fit overall raw / cal | dev G0 raw / cal | dev overall raw / cal | fit bias | dev−fit cal gap (G0 / overall) | dev raw→cal MAE change (G0 / overall) |
|---|---:|---:|---:|---:|---:|---:|---:|
| O | .029138 / .028401 | .029347 / .028637 | .095642 / .095018 | .097518 / .096880 | −.007809 | .066617 / .068243 | −.000624 / −.000638 |
| T | .029789 / .026442 | .030014 / .026685 | .095740 / .094603 | .097356 / .096337 | +.016293 | .068161 / .069652 | −.001137 / −.001019 |

T's dev calibrated point estimates are lower by .000415 on G0 and .000543 overall; raw gains are −.000097 on G0 and +.000162 overall. T's calibration offset is +.016293 versus O's −.007809. Calibration-accounting difference (overall cal gain minus raw gain) is +.000381; this is an accounting identity, not a causal decomposition. Fit-to-dev gaps remain large (~.066–.070).

### Frozen paired bootstrap and gate

1,000 paired dev-row bootstrap draws, seed `20261007`:

| endpoint | O−T gain | 95% CI | ≥ .003? |
|---|---:|---:|---|
| G0 calibrated (primary) | +.000415 | [−.002728, +.003617] | no |
| overall calibrated | +.000543 | [−.002623, +.003932] | no |
| G0 raw | −.000097 | [−.003308, +.003372] | — |
| overall raw | +.000162 | [−.003074, +.003480] | — |

The primary CI lower bound is not above zero. Thus the three gate conditions (G0 calibrated gain ≥.003, overall calibrated gain ≥.003, G0 CI lower >0) all fail. Both calibrated intervals extend outside ±.003, so **practical equivalence is not established either**. The witnesses pass: identical-arm gain/CI are exactly zero and arm swapping mirrors the point and CI. Deleting the largest O calibrated-error row (dev position 1403, global train ID 6842) changes the G0 gain to +.000566; this does not alter the gate conclusion.

**Observed statistical label:** `DIRECTIONAL_NOT_CONFIRMED`. **Strict execution status:** `NOT_COMPLETED` because of the deviations below.

## Required group accounting

Calibrated dev MAE and contribution (`sum absolute error / 2000`); contribution rows sum to each arm's overall MAE.

| group | n | O MAE (contribution) | T MAE (contribution) | O−T contribution gain |
|---|---:|---:|---:|---:|
| k=0 | 1,935 | .095018 (.091930) | .094603 (.091528) | +.000402 |
| k=−1 | 56 | .144715 (.004052) | .143725 (.004024) | +.000028 |
| k=−2 | 7 | .163545 (.000572) | .121513 (.000425) | +.000147 |
| k≤−3 | 2 | .325799 (.000326) | .359431 (.000359) | −.000034 |
| **overall** | **2,000** | **.096880** | **.096337** | **+.000543** |

The small net point gain comes mostly from the dominant k=0 contribution, with tiny gains in k=−1/−2 and a small loss for k≤−3 (only two rows). This describes where the observed error mass moved; it does not identify a mechanism. Fit-side calibrated k-group values, sums, and contributions are in `fit_group_table.csv`; full raw/cal, per-arm shifts, gaps, and per-graph rows are also retained in the CSV artifacts.

## Identity, replay, and mechanism boundary

* Completed O/T init hashes match (`9678be159fdc…`); each has 297,883 parameters, exactly 15,120 optimizer steps, and the frozen RNG, schedule, and GID-stream hashes. Raw-soup prediction replay on 128 fit/dev rows per arm has maximum absolute error O fit `9.54e−7`, O dev `1.43e−6`, T fit `1.67e−6`, T dev `1.91e−6`, all below `1e−5`.
* At epoch 240, the O new-column gradient is exactly zero; T's is `0.004351`. This supports implementation/gradient-path functionality, **not** a generalization claim.
* Source M on this same fold has dev calibrated G0/overall MAE `.097383 / .099304`; the completed T pair is numerically below those point values, but the frozen gate compares T against its matched O arm. Do not treat this one-seed/source-soup comparison as a purchased gain.
* The conditional beta-to-graph-marginal forward intervention was **not run**; it was permitted only after a gate pass. No architecture, dictionary, pooling, hyperparameter, seed, fold, or rescue attempt was added.

## Execution deviation — material to status

Two initial remote jobs began training and were cancelled: O job `55999` reached epoch 140 and T job `56000` epoch 40. Neither completed; their dev scores were not accessed or used. The cancellation assumed that same node `c05` plus `CUDA_VISIBLE_DEVICES=0` meant the same physical GPU. That assumption was wrong: each job had a one-GPU Slurm allocation, and local CUDA index 0 is allocation-relative. Physical UUIDs were not captured for those cancelled jobs; they may have occupied distinct cards. They were partial training attempts, not clean fit-only smoke jobs.

The completed jobs were subsequently launched from initialization rather than resumed from a saved checkpoint/RNG state: O `56001` (UUID `GPU-c7067be6…`, PCI `1D:00.0`) and T `56002` (UUID `GPU-0c6f2aa8…`, PCI `25:00.0`). Both completed 240 epochs and 15,120 steps with exit 0 on the same A100/CUDA/Torch regime; the two completed jobs ran sequentially on distinct physical cards without oversubscription. However, the relaunches began after the frozen minute-90 stop (measured from the 08:03 first-work/branch time; cutoff about 09:33; completed jobs began 09:39/09:42). The partial attempts plus fresh relaunch also violate the fixed-two-trajectories-only rule. Exact GPU allocation use for the completed pair is 0.3501 GPU-hours; including cancelled allocations is ≤0.5876 GPU-hours by elapsed-allocation upper bound, below 0.8. The work/reporting extended beyond the 120-minute round window.

The final reporting-only analysis change (fit-side G0, fit-group, gap, and raw→cal output) was made after the frozen training commit; it does not change either completed trajectory or the dev gate, and is separately committed. No new compute followed the completed pair. Full job IDs, times, device records, resource bounds, and chronology are in `EXECUTION.md` and `budget.json`.

## Scope and conclusion

The completed matched pair gives a small positive calibrated direction but does not establish the preregistered gain or practical equivalence. With the execution deviations, it cannot be promoted as a protocol-compliant negative experiment either. This result does **not** resolve a general bottleneck, show that direct-bond information is useless, or distinguish an input limitation from an optimization/interface issue. It is one seed on a reused internal partition, and the row bootstrap does not measure training randomness. Official ZINC `y`, valid, and test performance are not evaluated.

No confirmation run is warranted by this result. Any future purchase requires a **new**, independently motivated and preregistered question and a clean execution plan; this round's dev rows are not a fresh confirmation set. See `DECISION.md` and `NO_CONFIRMATION_NOTE.md`.
