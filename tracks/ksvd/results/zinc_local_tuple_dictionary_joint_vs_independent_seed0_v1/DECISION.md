# DECISION — `zinc_local_tuple_dictionary_joint_vs_independent_seed0_v1`

## Final class

**Primary gate (J vs I): `INCONCLUSIVE`** (with a directional J advantage).
**Performance class vs B: `EXPLORATORY_PERFORMANCE_SIGNAL`** (calibrated Gate A threshold not met).
**Label/access lineage: `ACCESS_LINEAGE_RESOLVED`** (see `ERRATA.md`; no held-out labels fitted,
no official-valid/test prediction or scoring in this round).
Round complete: two arms (J, I), one seed, one fold, 240 epochs each; no third arm, no second seed,
no rescue configuration, no ring head, no official split, no 10k confirmation.

## Direct answers to the six closing questions

**1. Frozen setup and checks.** Everything (operator index, protocol, method contract, evidence
scope, errata) was frozen and committed before any dev score. All operator/contract checks passed:
raw-vs-cache incidence exact, marginal identity `1e-10`/`1e-5`, batch/shuffle/offset equivalence
`0.0`, same-phi-different-`C` witnesses, synthetic star witness, B reproduction max Δ `3.4e-7`,
GPU smoke `all_ok=true`, replay `1.9e-6`.

**2. J or I vs B (Gate A).** J is directionally better than B on dev G0 cal (`0.100416` vs
`0.101875`) and overall cal (`0.101907` vs `0.103717`), and raw-significantly better (G0 raw gain
`+0.00457`, CI `[+0.00020, +0.00901]`; overall raw `+0.00486`, CI `[+0.00080, +0.00927]`).
But the calibrated points are below the pre-registered `0.003` threshold with CIs crossing zero
(G0 `+0.00146` `[-0.00289, +0.00601]`; overall `+0.00181` `[-0.00207, +0.00596]`), so Gate A does
not fire: **`EXPLORATORY_PERFORMANCE_SIGNAL`**, not a performance claim. I is not better than B
(raw CI crosses zero; calibrated negative).

**3. Joint vs independent (Gate B).** J beats I directionally: G0 cal `+0.003575`
CI `[-0.000230, +0.007270]`; G0 raw `+0.005784` CI `[+0.001904, +0.009678]`; overall cal
`+0.002775` CI `[-0.000947, +0.006911]`; overall raw `+0.004846` CI `[+0.000905, +0.008889]`.
`JOINT_SUPPORT` fails only on the calibrated CI crossing zero (`CI_low = −0.00023`);
`LOCAL_EQUIVALENCE` fails because the G0 cal CI upper bound `+0.00727` exceeds `+0.003`;
`INDEPENDENT_SUPPORT` fails. **`INCONCLUSIVE`**: real correspondence access is favoured but not
established by one seed at this budget.

**4. Mechanism validity.** The local tuple path is the dominant functional term after training:
zeroing it at init changes nothing (exact 0), after training it degrades dev cal MAE to `0.5485`
(J) / `0.4655` (I). The operator switch J↔I changes predictions (mean |Δ| `0.0034`/`0.0039`) and
moves the error in the expected direction (+`0.00008` for J→I, `−0.00013` for I→J). The tied IHT
dictionary is exactly 8-sparse per tuple, uses 60–61/64 atoms, has 3–4 dead root-code dims, and its
`D_loc` norm moves `88.8→33.3` (drift `0.75`) while `W_loc` reaches norm `9.2`; task gradients to
both exist by epoch 40. No `MECHANISM_FAILED`, `MECHANISM_INIT_BLOCKED`, or `NO_OPERATOR_CONTRAST`
condition applies.

**5. Sensitivity and location of the effect.** The J > I ordering survives dropping the single
largest-error dev row (J `0.10026` / I `0.10216` overall cal) and is already present in sample
(fit J `0.028155` vs I `0.031167`). It is driven by G0 (97.5% of dev MAE): J `0.1004` vs I `0.1040`;
on the rare cycle groups I is directionally better (`k=-1`: `0.1082` vs `0.1227`; `k=-2`:
`0.1759` vs `0.1877`; `k<=-3`: `0.2819` vs `0.3431`), on 12 dev rows total, so no ring claim is
made. Unattributable at one seed/fold/soup: optimisation/dropout noise vs a genuine G0-specific
effect, capacity/`s` sensitivity, and the role of the fit-median calibration bias difference
(J `−0.0164` vs B `−0.0314`).

**6. Single next design (design only).** Incidence-permuted marginal control with ≥3 seeds: keep
J's architecture/capacity/recipe but permute each root's `(t,a)` incidence entries so the marginals
`(n_t, n_a, d_v)` are preserved while the joint pairing is destroyed; compare J / I / perm with the
same paired bootstrap and the same `0.003` calibrated G0 gate plus a `±0.003` equivalence band.
Outcomes would be `JOINT_SUPPORT` (J > I and J > perm), `MARGINAL_ONLY` (perm ≈ J > I), or
`LOCAL_EQUIVALENCE`. Not executed in this round.

## Boundaries

* This round tests a **static, one-hop, root-local** correspondence in a skeleton with **no message
  passing**; no global-structure or propagation claim follows either way.
* Reference B is one historical soup, read-only; `EXPLORATORY_PERFORMANCE_SIGNAL` means "not
  established", not "no effect".
* `EVIDENCE_SCOPE.md` corrections stand: prior D_g/M_g evidence inconclusive, ring relief
  unconfirmed, old fit/dev mix-ups, old binding ≠ new tuple operator, `X175` ≠ new tuple, no MPG.
* All remote jobs (3 Slurm allocations) completed with exit 0; nothing left running. No push/merge;
  no historical file overwritten.
