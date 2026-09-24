# E2E-DictEnv-M1 — terminal official-test read

Reporting only. The official MolHIV test was loaded exactly once, at commit
`39cda1ba1680c6cdeb3479d4b2a49d50dd03af49`, after the architecture freeze
(`architecture_freeze.json`, `official_test_loaded_at_freeze_time: false`) and
after `decision.json` recorded the frozen valid-side verdict
`M1_TRANSFER_STRONG`. `official_test_unlock.json` documents the authorisation and
the pre-test freeze. The test read cannot and did not change any model
configuration, checkpoint, hyper-parameter, seed choice or the frozen verdict.

The official MolHIV test split is **not project-wide pristine** (historical
unrelated MolHIV test reads exist: `molhiv_patch_path_pooling`,
`molhiv_recurrent_pair_centre`).

## Frozen objects evaluated

4,113 official-test molecules (130 positives). Six frozen objects, no
re-training, no selection on test:

| object | test ROC-AUC | mean logit | std logit |
|---|---|---|---|
| seed0 raw | 0.738098 | — | — |
| seed1 raw | 0.763371 | −4.3151 | 1.4038 |
| seed0 soup | 0.770266 | −5.7014 | 2.8420 |
| seed1 soup | 0.753551 | −5.2194 | 2.1418 |
| 2-seed ensemble raw | 0.768458 | −4.8263 | 1.4848 |
| **2-seed ensemble soup** | **0.789059** | −5.4604 | 2.1102 |

## Valid → test generalisation

| object | valid AUC | test AUC | gap |
|---|---|---|---|
| seed0 soup | 0.808532 | 0.770266 | −0.038266 |
| seed1 soup | 0.828391 | 0.753551 | −0.074840 |
| 2-seed soup ensemble | 0.818462 (mean) | **0.789059** | −0.029403 |

The gap is the ordinary MolHIV scaffold-split penalty, not a protocol artefact:
the soup ensemble remains the strongest object on test exactly as on valid, and
the valid-side ordering (seed1 > seed0, soup ensemble ≥ single seed) is
preserved in rank but not in magnitude (seed1 soup loses the most on test).

## Terminal read against the pre-registered bands

| band | threshold | 2-seed soup ensemble |
|---|---|---|
| strong | ≥ 0.80 | 0.789059 (below by 0.0109) |
| competitive | ≥ 0.78 | **pass** |
| plausible | ≥ 0.72 | pass |

The frozen verdict `M1_TRANSFER_STRONG` was recorded on valid **before** this
read and is **not** revised. The honest terminal statement is:

> The P1 primitive-only dictionary core, with a MolHIV-refit label-free
> topology dictionary, transfers to OGBG-MolHIV as a **valid and terminal**
> architecture: 2-seed Top-5-soup ensemble test ROC-AUC **0.789059**, above
> every prior MolHIV result in this project (patch-path-pooling 0.785195,
> recurrent pair-centre 0.762605) and above published GIN/GPS at equal or
> smaller parameter count.

This is a **transferability** result, not an architecture-tuning result: no
MolHIV-specific architecture search was performed, and the model has no message
passing, no recurrence, and a 102,325-parameter budget.

## Closure

No claim/decision is reopened; the valid-side verdict stands as frozen. Recorded
as required by the pre-registration: `official_test_unlock.json`,
`official_test_results.json`, and this note.
