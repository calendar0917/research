# E2E-DictEnv-A2-Confirm — DECISION

* protocol: `e2e_dictenv_a2_confirm`
* parent round: `E2E-DictEnv-A2` (frozen protocol `e2e_dictenv_a2`, prereg `1813f53`)
* predecessor: `E2E-DictEnv-A2-Lite` / `e2e_dictenv_a2_lite`
* GPU: physical GPU1 only; user-authorised parallel schedule (arm schedules: {'REAL': None, 'INDEP': 'single-arm job (user-authorised parallel GPU1 schedule)'})
* seed: 0; horizon: 320; arms: `REAL`, `INDEP` (TOPO not retrained)
* continuation regime: `fresh_matched_320` (mixing prohibited)

## Verdict: `ATTRIBUTED_PAIRING_SUPPORTED_BUT_NOT_COMPETITIVE` (case C)

* reason: G_pair_320 0.005567 >= 0.003 but Delta_vs_TOPO 0.010169 > 0.003: the pairing carries task information yet the attributed dictionary/code formation is not competitive with the topology-only dictionary; no attributed E2E/IHT follows
* G_pair_320 = MAE(INDEP-OMP-320) - MAE(REAL-OMP-320) = 0.005566724489443009
* material bar = 0.003 (parent A2 frozen bar)
* Delta_vs_TOPO = MAE(REAL-OMP-320) - MAE(TOPO-OMP-320) = 0.010168869751389115
* practical tolerance = 0.003; TOPO soup = 0.12681294702464949

## Frozen-arm results (320 epochs, matched protocol)

| arm | soup valid MAE | best valid MAE | best epoch | soup members |
|---|---:|---:|---:|---|
| `ATTR-REAL-OMP-320` | 0.1369818167760386 | 0.14135768095491222 | 290 | [266, 287, 288, 290, 303] |
| `ATTR-INDEP-OMP-320` | 0.1425485412654816 | 0.14588125765224685 | 319 | [304, 311, 315, 317, 319] |
| `ATTR-TOPO-OMP-320` (parent, not retrained) | 0.12681294702464949 | 0.13136472144449363 | 314 | — |

## Diagnostics (never gates)

* paired late window [241, 320]: mean 0.006538768048032944, median 0.006299895591568197, positive fraction 0.7625, first 0.0018247949490323712, last 0.021618235073052328
* 160 -> 320: REAL 0.15437116196932038 -> 0.1369818167760386; INDEP 0.15993232336913935 -> 0.1425485412654816; G_pair 0.005561161399818965 -> 0.005566724489443009 (KEPT, sign_reversed=False)

## Not run (deferred; not cancelled)

* seed 1 replication — DEFERRED_PENDING_USER_AUTHORIZATION
* PCA32 dense control — DEFERRED_PENDING_USER_AUTHORIZATION
* continuity-v2 diagnostic — DEFERRED_PENDING_USER_AUTHORIZATION
* IHT-10/30/100/200 coder qualification — DEFERRED_PENDING_USER_AUTHORIZATION
* task-coupled E2E sparse dictionary (E0/E1/E2) — DEFERRED_PENDING_USER_AUTHORIZATION
* REAL->INDEP code-pairing-removal mechanism — DEFERRED_PENDING_USER_AUTHORIZATION
* node-only / edge-only pairing interventions — DEFERRED_PENDING_USER_AUTHORIZATION
* DenseTied specificity control — DEFERRED_PENDING_USER_AUTHORIZATION
* TOPO-OMP retraining — DEFERRED_PENDING_USER_AUTHORIZATION
* K64 dictionary — DEFERRED_PENDING_USER_AUTHORIZATION
* s12 dictionary — DEFERRED_PENDING_USER_AUTHORIZATION
* official test split — DEFERRED_PENDING_USER_AUTHORIZATION
* any architecture / hyper-parameter sweep — DEFERRED_PENDING_USER_AUTHORIZATION

## Stop

This round stops here in every case.  No IHT qualification, task-coupled E2E,
mechanism intervention, DenseTied specificity, second seed, capacity study or
official-test access follows without explicit user authorisation.

Official test loaded: `false`.
