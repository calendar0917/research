# E2E-DictEnv-A2-Lite — REPORT

* amendment: `tracks/ksvd/notes/e2e_dictenv_a2_compute_amendment.md`
* amendment commit: `4f184dd`
* git commit: `4f184dd3a16da4b1b3f35ba3d690f99438bf3a0a`; branch `exp/e2e-dictenv-a2-code-value-compression`
* frozen artifacts reused by reference: `artifact_identity.json` (all_passed=True)

## Screen (frozen exact-OMP, horizon 160, seed 0, matched init/batch order)

* MAE(REAL-OMP) = 0.15437116196932038
* MAE(INDEP-OMP) = 0.15993232336913935
* G_pair_screen = 0.005561161399818965 (strong bar 0.006)
* late direction: {"best_delta": 0.005376214825548231, "delta_first": 0.006360038555227238, "delta_last": 0.0064425996718928125, "mean_delta": 0.0038183560276782375, "min_positive_fraction": 0.75, "n_epochs": 40, "positive_fraction": 0.65, "stable": false, "window": [121, 160]}
* curves finite: True

## Dense control (train-only PCA32, horizon 160)

* MAE(REAL-PCA32) = 0.15774420121958246
* MAE(INDEP-PCA32) = 0.15633950045978418
* G_pair_PCA = -0.0014047007597982886 (bar 0.003)

**Verdict: `NO_ATTRIBUTED_CODE_FORMATION_SIGNAL_AT_32D`**

Deferred (not cancelled): TOPO-OMP predictor baseline = DEFERRED_PENDING_USER_AUTHORIZATION; IHT-10/30/100/200 coder qualification = DEFERRED_PENDING_USER_AUTHORIZATION; task-coupled E2E sparse dictionary (E0/E1/E2) = DEFERRED_PENDING_USER_AUTHORIZATION; REAL->INDEP code-pairing-removal mechanism = DEFERRED_PENDING_USER_AUTHORIZATION; node-only / edge-only pairing interventions = DEFERRED_PENDING_USER_AUTHORIZATION; DenseTied specificity control = DEFERRED_PENDING_USER_AUTHORIZATION; seed 1 replication = DEFERRED_PENDING_USER_AUTHORIZATION; official test split = DEFERRED_PENDING_USER_AUTHORIZATION; K64 dictionary = DEFERRED_PENDING_USER_AUTHORIZATION; s12 dictionary = DEFERRED_PENDING_USER_AUTHORIZATION; any architecture / hyper-parameter sweep = DEFERRED_PENDING_USER_AUTHORIZATION. Official test never loaded.

