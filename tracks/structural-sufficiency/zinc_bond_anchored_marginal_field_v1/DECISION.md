# DECISION — BAMF-v1

**Decision:** **STOP_NO_SIGNAL**  
**Date:** 2026-10-27  
**Protocol:** `zinc-bond-anchored-marginal-field-v1`  
**Frozen commit:** `f0d05a55e42edbde6291cb2c558b60ce56314939`  
**Formal run commit:** `a04eb5591ec42e0cde571c4999bb9b4044dff2bf`

## Ruling

BAMF-v1 does not proceed to seed 1, does not access or score any Locked1k value, and does not build
or score an official-test submission. The pre-registered seed-0 gate failed on
both required comparisons, with the wrong sign:

| gate | required | observed | outcome |
|---|---:|---:|---|
| G_s0 − I_s0 | ≤ −0.004 | **+0.003149** | fail |
| G_s0 − L_s0 | ≤ +0.002 | **+0.004509** | fail |
| any-unseen/no-penalty vs L (suggested) | ≥ +0.005 improvement | **−0.008552** | fail |

G_s0 Monitor MAE = 0.127660; I_s0 = 0.124510; matched DCC-L = 0.123151. The
independent-feature control I is also +0.001359 worse than L (95% CI
[−0.001778, +0.004571]), so the 12-feature channel itself is not a gain. The
paired-bootstrap intervals for both required contrasts exclude the required
direction. G is also worse after dropping the ten most influential rows
(+0.003890) and in the no-penalty subgroup (+0.003453). The result is not a
0.001–0.003 positive fluctuation that needs more evidence; it is a wrong-sign
result against both controls.

## Why this is a stop, not a rescue

The protocol fixed the operator, C=4, S=2, amplitudes, width, optimizer,
47,400 updates, and the seed-0 gate before the formal runs. The gate was
designed so that a mechanism-only difference would not be enough: G had to beat
both the independent-edge control I and the matched DCC-L baseline. It beat
neither. Changing the potential scale, state count, width, LR, schedule, or
feature set after seeing this would be a new experiment selected on the
evaluation split, and is prohibited by the frozen protocol. The Phase-0
redundancy diagnostic (G features largely predictable from DCC local input)
was already known before the GPU run and did not license a retune.

The special verdict `GLOBAL_MECHANISM_BUT_NO_PERFORMANCE` is **not** used: it
would require G to beat I, which it did not. `FACTOR_FEATURE_GAIN_NOT_GLOBAL`
is also not used: G did not beat L. The applicable verdict is the plain
`STOP_NO_SIGNAL`.

## Scope of the claim

This decision closes **this exact bond-anchored exact-marginal operator, at
497,093 parameters, with this frozen DCC-L skeleton and recipe, on this Fit8k/
Monitor protocol**. It does not prove that every possible global inference
method is useless, and it does not prove the elimination code is wrong. Phase 0
showed the elimination is exact, differentiable, and non-local; the performance
claim is what failed.

## Compute and audit

- Formal seed-0 cost: G_s0 1.564 A100·h + I_s0 0.469 A100·h = 2.033 A100·h.
- All BAMF jobs including smoke, timing, and three instrumentation failures:
  **2.073 A100·h** of the 12 A100·h cap.
- The G per-run cost was 1.564 A100·h, below the 2.5 A100·h gate; the timing
  projection was 1.884 A100·h.
- Failure cost: 0.0072 A100·h, all in the timing wrapper, no formal update lost.
- Locked1k values accessed/scored: **no**. Official valid values accessed/scored:
  **no**. Official test loaded: **no**. Seed 1 launched: **no**. An
  incidental load of the historical L prediction file (which also contains
  unused `locked_pred`/`official_valid_pred` arrays) is documented in
  `REPORT.md` §6; no such value was inspected or used.
- Run IDs and per-job records: `analysis/run_records.json`.
- Full evidence and required answers A/B/C: `REPORT.md`.
- Monitor gate artifact: `analysis/monitor_gates.json`.

## Final instruction

No follow-up experiment is authorized or proposed by this decision. Do not
launch seed 1, do not access or score Locked1k, do not create an official-test submission,
and do not retune this operator on Monitor. If this direction is ever reopened,
it must be under a new pre-registration with a materially different mechanism;
this round supplies no menu of variations to try.
