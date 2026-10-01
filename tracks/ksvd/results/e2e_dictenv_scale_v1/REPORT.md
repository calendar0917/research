# e2e_dictenv_scale_v1 — unified Small/Full task-dictionary scaling

CPU only · official test never loaded · single Full seed-0 trajectory.

## A. Provenance

- formal-run commit `5335301d0e8f36c67b1acd8fb192d92c79769e3e`
- analysis / report commit `638d6a3f9017b15ba1af2127fda59a3f2fed46ae`
- device `cpu` · threads `8` · seed `0`
- `official_test_loaded = false`
- vendored reference package `tracks/ksvd/experiments/luyin16/dictionary_scaling/v1_20261001`

## B. Historical references (read-only; never rerun; unmatched)

- CSSD-q1-seed0: `0.130028`
- FINAL-CLEAN-C6-seed0: `0.128499`
- P1-sparse-seed0: `0.131975`
- Sem108-seed0-background: `0.123705`
- T1-tuned-seed0: `0.125765`
- latent-bridge-seed0-background: `0.121058`

## C. Exact architecture

```
Sem108: static local object -> fusion (446 -> 342 -> 144) -> h (144)
candidate: h -> shared task dictionary [D_L 144x288, V_L 288x144]
           -> rho-normalised 16-step unrolled ISTA codes alpha (288)
           -> E = rho * alpha @ V_L (144) -> unary + pair (16m -> 48) + reader (814 -> 39 -> 39 -> 1)
no graph index, no message passing, no residual bypass, no new raw features
```

## D. Parameter budget

- Small `106925` (projection reference) · Full `408651` (task dictionary `82944`, body `325707`)

## E. Training

- epochs `320` · wall `6274.4 s` · seed `0`
- best valid `0.123307` @ 299 · Top-5 members [268, 276, 289, 299, 317] · soup `0.119154`
- same-protocol soup train MAE `0.045397` · train→valid gap `+0.073757`

## F. Performance interpretation

- `M_S = 0.119154` → band **SCALE_LIMITED_SIGNAL_CLOSE_CAPACITY_ROUTE**
- latent_bridge background `0.121058` (difference `+0.001904`) — unmatched background only
- sem108 background `0.123705` (difference `+0.004551`) — unmatched background only

## G. Task-dictionary mechanism (frozen soup state, inference only)

- zero task-dictionary code: `delta_mae = 1.436962`, `delta_pred_rms = 1.940718`
- within-molecule code permutation (5 seeds): mean `delta_mae = 0.394184`, mean `delta_pred_rms = 0.607477`
- reset D_L / V_L to init (encoder and head kept): `delta_mae = 1.329684`, `delta_pred_rms = 1.746171`
- MAE-only task gradients: `D_L 9.626e-02`, `V_L 2.431e-01`; movement Frobenius `D_L 16.4566`, `V_L 12.8033`
- code density: mean non-zero `250.89/288` (p50 `255.0`, p95 `275.0`) — variable density
- diagnostic low-dimensional reconstruction: mean `0.0588` (diagnostic only, never an outer loss)
- learned `True` · load bearing `True`

## H. Verdict

**Case C — SCALE_LIMITED_SIGNAL_CLOSE_CAPACITY_ROUTE**

> D_L / V_L moved from the frozen frame and receive MAE-only task gradient; the zero-code and permutation probes show the channel carries information, not that dictionary learning beats a matched trained control

## I. Evidence discipline

- single seed, no second seed, no matched control arm, no official-test read; every historical comparison is unmatched context
- the zero-code / permutation / reset probes establish channel information only; they do not establish that dictionary learning beats a matched trained control
- the band is a resource-decision threshold, not a significance test

