# DECISION — `zinc_direct_bond_relation_seed0_v1`

## Decision

**Do not buy a confirmation run.** No additional compute, seed, fold, arm, architecture, or rescue experiment is authorized by this result. The relevant internal-dev data have been reused, the frozen calibrated gate is not met, and the round's execution deviated from the two-trajectory / minute-90 contract.

## What the completed pair says (descriptive only)

The completed matched O/T pair gives T small positive calibrated point gains on internal `g=y−c`: G0 `+0.000415` (95% paired row-bootstrap CI `[−0.002728,+0.003617]`) and overall `+0.000543` (`[−0.002623,+0.003932]`). Both fall below the frozen `+0.003` bar, and the G0 lower bound is not above zero. Thus **0/3 candidate conditions pass**; the stored performance-only label is `DIRECTIONAL_NOT_CONFIRMED`. Both CIs extend beyond ±0.003, so practical equivalence is also not established. Raw results are mixed and effectively near zero (G0 `−0.000097`, overall `+0.000162`). Bootstrap implementation witnesses and replay checks pass.

This is not a general bottleneck verdict. It does not establish that direct-bond information is useless, that the model lacks information, or that a particular module is the residual bottleneck. One seed and a reused internal partition do not measure training-randomness uncertainty.

## Contract status and limitation

**Strict experiment status: `NOT_COMPLETED`.** Two partial remote training jobs were cancelled after they had begun (O job 55999 through epoch 140; T job 56000 through epoch 40). The cancellation was based on the incorrect assumption that identical `CUDA_VISIBLE_DEVICES=0` on c05 meant the same physical card; both held separate one-GPU Slurm allocations, and the first jobs' UUIDs were not captured. The later complete O/T trajectories were started fresh, not resumed from saved state/RNG, and started after the compute cutoff measured from first work. These deviations violate the exactly-two-trajectories and minute-90 conditions. The partial runs' dev scores were not used, but their existence still matters for the strict classification.

The completed pair itself is healthy on the checked implementation path: identical init hashes, expected 15,120 steps each, frozen schedule/RNG/GID hashes, T's new-column task gradient nonzero, and CPU replay errors ≤`1.91e−6`. Those checks do not repair the process deviation or convert the descriptive gate result into a protocol-compliant formal conclusion.

## Action / reopening threshold

* Close this round with no additional analysis requiring new labels and no more compute.
* Do not run the conditional beta-to-graph-marginal diagnostic; its frozen prerequisite (T passing the gate) was not met.
* Any future purchase must be a **newly motivated, separately preregistered** study—not a confirmation bought from this point estimate—and needs a clean run plan with two explicit Slurm GPU allocations, allocation-resolved UUID/PCI evidence before any cancellation decision, checkpoint/RNG-aware recovery rules, and a deadline schedule that places both formal trajectories before the cutoff. It must specify how it obtains independent evidence rather than treating these reused dev rows as fresh confirmation.
* No official-valid/test data were loaded; no official `y` result is claimed.

See `REPORT.md`, `EXECUTION.md`, `budget.json`, and `NO_CONFIRMATION_NOTE.md` for the full numeric table, chronology, and design-only reopening conditions.
