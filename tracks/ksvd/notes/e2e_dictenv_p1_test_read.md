# E2E-DictEnv-P1 — terminal official-test read

Reporting only. The official ZINC test was loaded exactly once, at commit
`e16ac15`, after the architecture freeze (`architecture_freeze.json`,
`official_test_loaded_at_freeze_time: false`) and after `decision.json` recorded
the frozen valid-side verdict. `official_test_unlock.json` documents the
authorisation and the pre-test freeze. The test read cannot and did not change
any model configuration, checkpoint, hyper-parameter, seed choice or the frozen
verdict `P1_CLEAN_ENVIRONMENT_VIABLE_BUT_DICTIONARY_NOT_SPECIFIC`.

This round's test read is permissible under the pre-registered order because the
round completed past the seed-0 GO gate and froze its configuration before the
read; seed 1 was not authorised (specificity gate failed on valid), so no seed-1
test row exists. The official test is not project-wide pristine (historical
unrelated ZINC test reads exist).

## Frozen objects evaluated

1000 official-test molecules; the Sparse and DenseTied seed-0 Top-5 soups,
parameter-identical (97,865), were evaluated on the same loader.

| object | test MAE | mean pred | std pred |
|---|---|---|---|
| Sparse seed0 soup | 0.107593 | 0.024383 | 1.992072 |
| DenseTied seed0 soup | **0.101661** | 0.009399 | 2.009836 |
| Sparse seed1 soup | not run (seed 1 not authorised) | — | — |
| DenseTied seed1 soup | not run | — | — |

Valid-side contrast for the same objects: Sparse 0.131975, DenseTied 0.134534
(`G_specific` +0.002559). On test the sign reverses: DenseTied is 0.0059 MAE
better. The valid-side "sparse marginally better" reading therefore does not
survive the split change, reinforcing (and not reopening) the frozen
not-specific decision.

## Mechanism generalisation on the Sparse seed0 soup (report only)

| intervention | test MAE | test gap | valid gap |
|---|---|---|---|
| clean | 0.107593 | — | — |
| zero dictionary coordinate | 0.456346 | `G_zero_test` 0.348753 | 0.347116 |
| node-assignment shuffle (5 perms) | 0.118886 | `G_node_test` 0.011293 | 0.013124 |
| edge-binding shuffle (5 perms) | 0.163968 | `G_edge_test` 0.056375 | 0.059528 |
| combined shuffle (5 perms) | 0.178907 | `G_all_test` 0.071314 | 0.072999 |

Mechanistic generalisation is essentially exact between valid and test: the
dictionary is strongly load-bearing (`G_zero` ≈ 0.35 on both), node-assignment
mediation clears the gate on test (0.0113), and the edge-binding and combined
interventions are large on both splits. Accuracy generalises well
(0.131975 valid → 0.107593 test for Sparse), i.e. the model is not overfitting
the valid soup.

## Closure

No claim/decision is reopened. Recorded as required by the pre-registration:
`official_test_unlock.json`, `official_test_results.json`, and this note.
