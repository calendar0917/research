# MolHIV B-Null frozen small-head probe — authorized official-test read

Protocol: `molhiv_bnull_small_head_probe_test_closure_v1`
Pre-registration (valid-only probe): `molhiv_bnull_small_head_probe_preregistration.md`

## Authorization

The valid-only pre-registration lists "no official test" as a non-goal. After the
probe returned `HEAD_PROBE_INVALID` (see final report), the user explicitly
requested the MolHIV **official test** read because MolHIV generalization is
difficult. This note records that the read was **user-authorised, one-shot, and
terminal**, and that no test number was used for selection.

## Discipline

* `freeze` materialised the already-valid-selected head states and wrote
  `architecture_freeze.json` with `test_loaded_at_freeze_time: false`, asserting:
  `R_dim == 423`, the probe never extracted a test representation, the probe
  never loaded the test split, the frozen head reproduced the backbone logits
  exactly (max abs 0.0), and both heads' valid AUCs reproduced bit-exactly from
  the rebuilt raw/Top-5-soup states (`passed: true`).
* `test` wrote `official_test_unlock.json`, then loaded the official OGBG-MolHIV
  test split exactly once, fit transforms on official train only, and extracted
  the test `R` from the **frozen** B-Null soup backbone in a single pass. Test
  labels were used only to score frozen predictions.
* Re-running `test` is refused (unlock file present).

## Result (official test, n=4113, 130 positives)

| head | params | valid soup | test raw | test soup | valid→test Δ soup |
|---|---:|---:|---:|---:|---:|
| B-Null backbone (joint) | 237,133 | 0.8408 | 0.743411 | 0.760501 | −0.0803 |
| H_refit (100k) | 100,417 | 0.8303 | 0.752365 | 0.749177 | −0.0811 |
| H_small32 (14k) | 14,177 | 0.8302 | 0.762960 | 0.762549 | −0.0677 |

The backbone row reproduces the earlier `molhiv_local_token_null_test_closure`
one-shot read (0.743411 / 0.760501) exactly, confirming provenance.

## Reading

* Every head loses ≈0.07–0.08 AUC valid→test, confirming MolHIV generalization
  is hard and that valid-only numbers overstate test performance.
* The 14k `H_small32` does **not** lose on test; it is the best soup (0.7625),
  while the 100k `H_refit` soup (0.7492) falls below the backbone (0.7605).
* Caveat: 130 test positives ⇒ ±0.01–0.015 is within seed/split noise. This is
  directional evidence that the 100k graph head is not needed, not a decisive
  architecture claim. It does not override the valid-only `HEAD_PROBE_INVALID`
  gate; the single recommended next action is a paired seed-1 confirmation
  (not run here).
