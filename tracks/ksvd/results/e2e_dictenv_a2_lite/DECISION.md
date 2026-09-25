# E2E-DictEnv-A2-Lite — DECISION

* protocol: `e2e_dictenv_a2_lite` (compute-budget amendment)
* parent round: `E2E-DictEnv-A2` — `COMPUTE-BUDGET-TRUNCATED`
* git commit: `4f184dd3a16da4b1b3f35ba3d690f99438bf3a0a`
* GPU: physical GPU1 only (`CUDA_VISIBLE_DEVICES=1`)
* screen: horizon 160, arms `REAL, INDEP`, seed 0

## Verdict: `NO_ATTRIBUTED_CODE_FORMATION_SIGNAL_AT_32D`

* reason: G_pair_screen 0.005561 < 0.006 (direction_stable=False); cheap dense rank-32 diagnostic authorised
* G_pair_screen = MAE(INDEP-OMP) - MAE(REAL-OMP) = 0.005561161399818965
* G_pair_PCA = MAE(INDEP-PCA32) - MAE(REAL-PCA32) = -0.0014047007597982886 (rank 32)
* dense reason: G_pair_PCA -0.001405 < 0.003: no attributed code-formation signal at rank 32

## Not run (deferred by the amendment; not cancelled)

* TOPO-OMP predictor baseline — DEFERRED_PENDING_USER_AUTHORIZATION
* IHT-10/30/100/200 coder qualification — DEFERRED_PENDING_USER_AUTHORIZATION
* task-coupled E2E sparse dictionary (E0/E1/E2) — DEFERRED_PENDING_USER_AUTHORIZATION
* REAL->INDEP code-pairing-removal mechanism — DEFERRED_PENDING_USER_AUTHORIZATION
* node-only / edge-only pairing interventions — DEFERRED_PENDING_USER_AUTHORIZATION
* DenseTied specificity control — DEFERRED_PENDING_USER_AUTHORIZATION
* seed 1 replication — DEFERRED_PENDING_USER_AUTHORIZATION
* official test split — DEFERRED_PENDING_USER_AUTHORIZATION
* K64 dictionary — DEFERRED_PENDING_USER_AUTHORIZATION
* s12 dictionary — DEFERRED_PENDING_USER_AUTHORIZATION
* any architecture / hyper-parameter sweep — DEFERRED_PENDING_USER_AUTHORIZATION

No official test data was loaded.

