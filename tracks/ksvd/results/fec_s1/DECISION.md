# FEC-S1 — decision

```
FEC_S1_SHARED_REPLACEMENT_STRONG
```

* primary metric (fixed Top-5 soup official-valid MAE): **0.130422**
* band: case **A** (strong ≤ 0.1408, viable ≤ 0.145, borderline ≤ 0.148, else failed)
* best-checkpoint valid MAE: **0.136783** @ epoch 238
* `seed1_authorized`: **true**
* seed 1 executed: False
* official test loaded: False

## Stop reason

Shared local-environment replacement retains the strict-static S0 performance band; a paired seed 1 is authorised (not auto-executed).
