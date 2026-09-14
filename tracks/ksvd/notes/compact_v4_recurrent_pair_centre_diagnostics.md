# compact-v4 recurrent pair–centre — signed-q & T=3 diagnostics (2026-09-13)

> Two low-cost, seed0-only validation diagnostics on the frozen best model
> (compact-v4-smallhead 2-round weight-tied recurrent pair–centre, 82,115
> params). No official test was loaded. No activation sweep, no T-depth sweep,
> no pair-to-pair continuation.

Base reference (frozen): T=2 recurrent seed0 valid = **0.138376**, seed1 valid
= **0.133440**, test mean = 0.114951 (not re-accessed).

Code: `tracks/ksvd/experiments/luyin16/zinc_compact_v4_recurrent_pair_centre_diagnostics.py`.
Results: `tracks/ksvd/results/compact_v4_recurrent_pair_centre_diagnostics/`.

---

## Task A — signed relation state (remove final `pair_encoder` ReLU)

### A.1 Pre-activation audit (original ReLU T=2 model, trained seed0 checkpoint)

First 128 official-valid molecules, 34,876 pair rows. Hooked the input of
`pair_encoder.layers[-1]` (the final ReLU).

| quantity | q0 | q1 |
|---|---:|---:|
| pre-activation negative fraction | **0.8505** | **0.8605** |
| pre-activation zero fraction after ReLU | 0.8505 | 0.8605 |
| pre-activation mean | −0.5829 | −0.6186 |
| pre-activation std | 0.6332 | 0.6333 |
| post-ReLU mean | 0.0478 | 0.0421 |
| post-ReLU std | 0.1453 | 0.1303 |

q0 → q1: sign/on–off flip fraction **0.0318**, cosine mean 0.9870 (median
0.9924), relative delta norm mean 0.1628 (median 0.1377).

**Reading.** ~85–86% of relation pre-activations are negative and are discarded
by the final ReLU, so the relation state *is* strongly rectified. The round
drift is small but non-trivial. This makes the signed-q intervention a
legitimate single-change hypothesis.

### A.2 Signed-q variant and training

`pair_encoder.layers[-1]: ReLU -> Identity`, nothing else changed.

* parameters = **82,115** (unchanged; Identity/ReLU have no parameters)
* head = 4,135; state-dict shapes identical to the ReLU recurrent model
* only the final pair ReLU removed (patch/relation/global encoders still end in
  ReLU; call counts unchanged: `pair_projection` 4, `pair_encoder` 2,
  `center_update` 2); forward/backward finite, non-zero grads
* protocol inherited verbatim (Adam 1e-3, wd 1e-5, batch 128, max 240,
  patience 40, no scheduler, L1, best valid), seed0 only

### A.3 Result

| model | seed0 valid | best epoch | params |
|---|---:|---:|---:|
| T=2 recurrent (ReLU) | 0.138376 | 199 | 82,115 |
| signed-q | **0.140051** | 171 | 82,115 |
| **Δ (T2 − signed)** | **−0.001675** | | |

Verdict: **DEGRADED / no signal**. Δ is outside the ±0.001 no-signal band on
the wrong side. **seed1 not run.** The audit's heavy truncation does *not*
translate into a useful signed relation state.

---

## Task B — T=3 recurrent depth diagnostic

### B.1 Actual flow (validated by `sanity_t3.json`, 23/23 checks)

```
h0 -> q0 = Q(P(h0)) -> A0 -> h1 = h0 + U([h0,A0])
   -> q1 = Q(P(h1)) -> A1 -> h2 = h1 + U([h1,A1])
   -> q2 = Q(P(h2)) -> A2 -> h3 = h2 + U([h2,A2])
readout = unary_moments(h3) + pair_moments(q2) + global + topology
```

`P = pair_projection`, `Q = pair_encoder`, `U = center_update` are the same
tensor objects in all three rounds. Sanity: `pair_encoder` called 3×,
`center_update` called 3×, centre aggregation 3×; `q1` depends on `h1` and `q2`
on `h2` (autograd max |grad| 18.3 / 17.4); the readout consumes exactly
`h3 = h2 + Δ2` and `q2`; per-round incidence count block bit-identical;
manual pair-count reconstruction matches; params 82,115 (identical state-dict
keys/shapes to T=2); at init the T=3 function is bit-identical to baseline
(zero-init centre update collapses all rounds).

### B.2 Result

Horizon: max_epochs 320, patience 40 (extended cap so a late-converging depth-3
model cannot be a false negative). Best epoch 173, run stopped at 213 →
**no horizon warning** (warning threshold 288).

| model | seed0 valid | best epoch | epochs run | params |
|---|---:|---:|---:|---:|
| T=2 recurrent | 0.138376 | 199 | 239 | 82,115 |
| T=3 recurrent | **0.139022** | 173 | 213 | 82,115 |
| **Δ (T2 − T3)** | **−0.000647** | | | |

Verdict: **SATURATED** (|Δ| ≤ 0.001), no depth scaling. **No T4/T5.**

---

## Combined conclusion

* signed relation state: **no value** (degraded by 0.0017 on seed0);
* deeper recurrent computation: **no value** (T=3 within ±0.001 of T=2, no
  horizon warning);
* ⇒ **neither is the current bottleneck.** The next step should change the
  relation *primitive* (e.g. ordered shortest-path encoding), not the relation
  activation or the recurrence depth.

Official test: never loaded. Files:
`taskA_pre_activation_audit.json`, `sanity_signed.json`, `sanity_t3.json`,
`signed_seed0.json`, `t3_seed0.json`, `decision_signed_seed0.json`,
`decision_t3_seed0.json`.
