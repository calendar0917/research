# Phase 1b decision log — GPU purchase record

Protocol: `ssrc-phase1b-v1` (frozen pre-GPU at commit `43313de`).
Question template asked before every purchase:

1. **Q1** — Which decision does this purchase inform?
2. **Q2** — What is the pre-registered outcome mapping (which result buys what next)?
3. **Q3** — Is there a cheaper CPU-only / existing-data test?
4. **Q4** — Cost and remaining budget?
5. **Q5** — Stop rule: what will *not* be bought regardless of the result?

No purchase was made after seeing a result unless the result mapped to a
pre-registered branch; the one judgment call (P2) is flagged explicitly.

---

## P0 — prep + GPU smoke (2 jobs, 55 s occupancy)

| Q | Answer |
|---|---|
| Q1 | Does the frozen code/data pipeline execute on the A100 node, with finite loss and measured step time/memory, before buying the matrix? |
| Q2 | If any arm fails or produces non-finite MAE, fix implementation and do not interpret mechanism. If all pass, proceed to P1. |
| Q3 | CPU smoke already passed; the GPU smoke uniquely checks CUDA kernels, TF32/AMP-free determinism path, and real step time. |
| Q4 | 0.002 GPU-h run-level (0.015 h job occupancy). Budget after: 7.998 h. |
| Q5 | No model/architecture change from the smoke; smoke MAE is not a result. |

Outcome: 10/10 arms exit 0; finite MAE; peak memory 40–55 MB; step time
12–34 ms. Proceeded.

---

## P1 — Phase A matrix: 5 arms × 2 tasks, seed 0 (10 runs, 1.076 GPU-h)

| Q | Answer |
|---|---|
| Q1 | The core descriptive question: absolute performance of S0/R1/M1/M2/G_ref and the sign/size of `Gain_relation`, `Gain_update`, `Gain_iteration`, `Gap_vs_Gref` on two tasks. |
| Q2 | Pre-registered bands: Lipo negligible <0.02, meaningful ≥0.04; ZINC negligible <0.01, meaningful ≥0.02. Relation and iteration signals trigger at most one control each; all-close triggers B3; all-worse-than-G_ref triggers B4; task disagreement triggers B5. |
| Q3 | No: the endpoints are defined on trained models; CPU cannot estimate them. |
| Q4 | 1.076 GPU-h. Budget after: 6.922 h (Phase A cap 2.0, used 1.08). |
| Q5 | No extra seeds, no hyperparameter sweep, no new arms in Phase A. If any sanity check fails, stop and debug. |

Outcome: all 10 runs sane (all beat the constant predictor). Both the relation
and iteration signals appeared on ZINC (CIs exclude zero), with the same signs
on Lipo. This triggered P2 and P3.

---

## P2 — B1 control: R1 with degree-preserving broken root adjacency (1 run, 0.100 GPU-h)

| Q | Answer |
|---|---|
| Q1 | Is the R1 gain over S0 due to true relational correspondence, or would any extra edge channel of the same capacity do? |
| Q2 | If broken-R1 ≈ true-R1, downgrade the claim to "generic edge channel". If true-R1 is better, the correspondence carries information. |
| Q3 | No cheaper test: the contrast is defined on the trained R1 architecture; the rewiring itself was CPU-gated (degree/type/count preserved, patches unchanged). |
| Q4 | 0.100 GPU-h. Budget after: 6.822 h (Phase B cap 2.0, used 0.10 so far). |
| Q5 | One rewiring realization only; no second realization unless the result is ambiguous (it was not). No architecture change. |

**Trigger note (honest deviation).** B1's literal pre-registered clause included
"while M2 does not beat M1 by that band". That clause was false (M2 did beat M1
on ZINC), so B1 was not literally satisfied. The control was still bought
because the relation signal was the larger and cleaner of the two, and the only
way to distinguish information access from a generic edge channel is this
contrast. The purchase is within Phase B's cap and its outcome is interpretable
regardless of the clause. This is recorded as a judgment call, not a
post-hoc change of the decision rule.

Outcome: broken-R1 = 0.3307 vs true-R1 = 0.2602 (delta +0.0706, CI95
[+0.0480, +0.0948]), and broken-R1 is even worse than S0 (0.3173). The
correspondence is real; the gain is not a generic edge channel.

---

## P3 — B2 control: M1 trained for the matched budget (1 run, 0.198 GPU-h)

| Q | Answer |
|---|---|
| Q1 | Is the M2 > M1 gain a mechanism effect of the second tied round, or just the extra computation/schedule? |
| Q2 | If a longer-scheduled M1 closes the gap, iteration is not separated from extra computation → report the null and do not pursue iterative relations. |
| Q3 | No: the control is a training-budget change on the same architecture; it must be trained. |
| Q4 | 0.198 GPU-h. Budget after: 6.624 h (Phase B used 0.30). |
| Q5 | One control only (600-epoch M1, same seed); no M2-long, no new round count. |

Outcome: M1_long = 0.2495 vs M2 = 0.2522 (delta −0.0027, CI95 [−0.0209,
+0.0152]). M1_long early-stopped at epoch 285/600, so it converged; it used
285 one-round applications vs M2's 390 two-round applications, at matched GPU
seconds (713.8 vs 705.6 s). The iteration advantage collapses under both
matchings. This is the pre-registered "iteration is not separated from extra
computation" outcome.

---

## P4 — Phase D: second seed for the gate-passing relation candidate (2 runs, 0.224 GPU-h)

| Q | Answer |
|---|---|
| Q1 | Is the S0→R1 relation gain robust to run-to-run noise, or comparable to it? |
| Q2 | If the sign replicates and the gain exceeds S0's run-to-run spread, confirm; if noise is comparable to the effect, report `INSUFFICIENT_EVIDENCE`. |
| Q3 | No: the estimate needs a second trained seed. |
| Q4 | 0.224 GPU-h. Budget after: 6.400 h (Phase D cap 1.5, used 0.22). |
| Q5 | Only the relation candidate gets a second seed; S0 seed 1 is the paired baseline, not a separate candidate. M2 does **not** get one: the B2 gate failed, so a seed-1 M2 cannot change the route. No 5-seed paper claim. |

Outcome: seed-1 S0 = 0.3563, R1 = 0.2571 (gain +0.0992). Seed-0 gain was
+0.0571; both signs positive; R1 run-to-run spread 0.0031, S0 spread 0.0390.
The effect is larger than S0's spread in both seeds. Relation gain confirmed.

---

## Totals

| Phase | Runs | GPU-hours (run-level) | Cap | Used / cap |
|---|---:|---:|---:|---:|
| A (matrix) | 10 | 1.076 | 2.0 | 54% |
| B (controls) | 2 | 0.298 | 2.0 | 15% |
| C (improvement) | 0 | 0.000 | 2.0 | 0% (no evidence-backed fix identified) |
| D (seed) | 2 | 0.224 | 1.5 | 15% |
| smoke/reserve | 10 short runs | 0.002 | 0.5 | <1% |
| **Total** | **24** | **1.601** | **8.0** | **20%** |

Job occupancy (start→completion, including startup/eval) = 1.64 GPU-h.
Wall clock from first GPU launch to last completion = 2 h 18 m.
All 10 GPU jobs exited 0; no failed-job GPU cost. All execution, transfer,
and formal runs used `rr` on clean commits; one read-only `ssh res ls` was used
only to check run-file presence during monitoring (no result depends on it).
Under-spend is intentional:
the pre-registered evidence was obtained without Phase C, and no purchase was
made that could not change the decision.
