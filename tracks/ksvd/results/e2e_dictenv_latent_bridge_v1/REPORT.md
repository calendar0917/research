# e2e_dictenv_latent_bridge_v1 — SEM108 + local task dictionary bridge

CPU only · official test never loaded · single seed-0 trajectory.

## A. Provenance

- formal-run commit `9d4cbedf69adf6e15d4dbfa2cdefab65306b8f61`
- analysis / report commit `9d4cbedf69adf6e15d4dbfa2cdefab65306b8f61`
- device `cpu` · threads `8` · seed `0`
- `official_test_loaded = false`
- vendored reference package `tracks/ksvd/experiments/luyin16/latent_bridge_reference/v1_20261001`

## B. Historical references (read-only; never rerun; unmatched)

- CSSD-q1-seed0: `0.130028`
- FINAL-CLEAN-C6-seed0: `0.128499`
- P1-sparse-seed0: `0.131975`
- Sem108-seed0-background: `0.123705`
- T1-tuned-seed0: `0.125765`

## C. Exact architecture

```
Sem108: static local object -> fusion -> h (48)
candidate: h -> shared task dictionary [D_L 48x96, V_L 96x48]
           -> rho-normalised 16-step unrolled ISTA codes alpha (96)
           -> E = rho * alpha @ V_L (48) -> unchanged unary + pair + reader
no graph index, no message passing, no residual bypass, no new raw features
```

## D. Parameter budget

- Sem108 body `97709`, bridge `9216`, candidate `106925` (D_L / V_L only)

## E. Training

- epochs `320` · wall `2434.8 s` · seed `0`
- best valid `0.129975` @ 309 · Top-5 members [285, 296, 303, 307, 309] · soup `0.121058`
- same-protocol soup train MAE `0.055133` · train→valid gap `+0.065926`

## F. Performance interpretation

- `M_S = 0.121058` → band **LATENT_BRIDGE_BORDERLINE**
- background Sem108 soup `0.123705` (difference `+0.002647`) — unmatched background only, not a causal increment

## G. Bridge mechanism (frozen soup state, inference only)

- zero bridge code: `delta_mae = 1.185274`, `delta_pred_rms = 1.696537`
- within-molecule code permutation (5 seeds): mean `delta_mae = 0.288035`, mean `delta_pred_rms = 0.473635`
- reset D_L / V_L to init (encoder and head kept): `delta_mae = 1.231997`, `delta_pred_rms = 1.809348`
- MAE-only task gradients: `D_L 2.694e-01`, `V_L 2.401e-01`; movement Frobenius `D_L 5.7622`, `V_L 5.6371`
- code density: mean non-zero `89.10/96` (p50 `89.0`, p95 `94.0`) — variable density, never a fixed l0
- diagnostic low-dimensional reconstruction: mean `0.0048` (diagnostic only, never an outer loss)
- learned `True` · load bearing `True`

## H. Verdict

**Case C — LATENT_BRIDGE_BORDERLINE**

> D_L / V_L moved from the frozen initial frame and both receive a non-zero MAE-only task gradient. The zero-code and within-molecule permutation probes are live, but they only establish that the channel carries information; they do not establish that dictionary learning beats a matched trained control.

## I. Evidence discipline

- single seed, no second seed, no matched control arm, no official-test read; every historical comparison is unmatched context
- the zero-code / permutation / reset probes establish channel information only; they do not establish that dictionary learning beats a matched trained control

## J. Dev evidence (scratch only, outside the formal chain)

- `dev/acceptance.json`
- `dev/short_trainability.json`

