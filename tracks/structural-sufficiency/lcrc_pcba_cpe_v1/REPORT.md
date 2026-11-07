# LCRC PCBA Cross-atom Predictive Evidence (CPE) v1 — report

**Status: protocol and code frozen; formal runs pending.** This file is created
with the freeze commit so the deliverable exists, and is rewritten with the
measured results after the two 40-epoch runs. Nothing below is a result.

Frozen design (see [`PROTOCOL.md`](PROTOCOL.md) for the binding text):

| arm | pair statistic | new params | total params |
|---|---|---:|---:|
| `P` | cross-atom `(sum_A·sum_B − sum_AB)/(n(n−1))` | 33,408 | 1,027,732 |
| `S` | same-atom `sum_AB/n` | 33,408 | 1,027,732 |

Frozen reference: historical `W2` seed-0 epoch-40 Validation AP **0.2618669181**
(994,324 parameters, not retrained in this round).

Pre-registered decision labels: `PROMISING_CROSS_ATOM_EVIDENCE`,
`CAPACITY_WITHOUT_CROSS_ATOM_ADVANTAGE`, `WEAK_OR_MIXED`, `NO_CLEAR_GAIN`,
`REGRESSION` (thresholds in PROTOCOL.md §13).

Every number in the final version of this file will be read back from
`analysis/*.json` and the git-ignored `runs/P`, `runs/S`; the official Test split
is never used for selection, prediction or scoring.

Required disclaimer that will appear in the final report:

```
single-seed architecture exploration
historical W2 baseline
same-round paired P/S mechanism control
not independently confirmed
```
