# Local-token-null: authorised post-hoc official-test read (ZINC + MolHIV)

The user explicitly authorised opening the official test for the two
non-promoted local-token-null screens, to be reported as a **post-hoc terminal
read**: both `raw` (best-valid checkpoint) and fixed Top-5 `soup` were frozen
in `architecture_freeze.json` **before** the test was loaded
(`test_loaded_at_freeze_time: false`), the test was loaded exactly once per
closure (`official_test_unlock.json` written), and the numbers may **not** be
used for promotion or any further architecture selection.

This supersedes, for reporting only, the "do NOT open the official test" clause
of `decision-local-token-null-20260919` and
`decision-molhiv-local-token-null-20260918`.

Code: `zinc_local_token_null_test_closure.py`,
`molhiv_local_token_null_test_closure.py`. Revision `4818df6`.

## Results

### ZINC official test (n = 1000, MAE, lower better)

| model | params | valid raw | **test raw** | valid soup | **test soup** |
|---|---|---|---|---|---|
| B-Null | 49,343 | 0.126873 | **0.102759** | 0.123028 | **0.097731** |
| Constant-16 | 49,359 | 0.126260 | 0.109475 | 0.120515 | 0.100665 |
| cell A typed (2-seed mean) | 85,763 | — | 0.109361 | — | 0.106717 |

### MolHIV official test (n = 4113, ROC-AUC, higher better)

| model | params | valid raw | **test raw** | valid soup | **test soup** |
|---|---|---|---|---|---|
| B-Null | 237,133 | 0.843168 | 0.743411 | 0.840847 | **0.760501** |
| Constant-32 | 237,165 | 0.856445 | 0.716694 | 0.860768 | 0.744255 |
| typed lookup (2-seed mean) | 1,076,589 | — | 0.762605 | — | 0.762095 |

## Reading

**1. Channel necessity is confirmed on test, on both datasets.**
Deleting the molecule-dependent local-token generator costs nothing on test.
ZINC: B-Null (49,343 params, no generator) soup test **0.097731**, i.e.
−0.008986 vs the cell-A typed reference and, for the first time in this track,
a **single model below the 0.10 target** (cell A raw 0.109361 / soup 0.106717;
the previous best single model was the H96 soup 0.104990, and only the labelled
2-seed prediction ensemble reached 0.098641). MolHIV: B-Null soup 0.760501 vs
typed 0.762095 = −0.001594, a tie inside the typed 2-seed std 0.008464, at
237k instead of 1.08M params.

**2. The shared-offset ("Pattern 2") reading does NOT survive on test.**
On valid, Constant was the best of the family on both datasets
(ZINC 0.120515 < Null 0.123028; MolHIV 0.860768 > Null 0.840847). On test the
ranking flips: ZINC Null 0.097731 < Constant 0.100665, MolHIV Null 0.760501 >
Constant 0.744255. So the surviving, defensible claim is the weaker one:
**the local-token channel is redundant**; "a 16/32-param constant recovers its
value" was a valid-set artefact.

**3. Valid ranking is not predictive of test ranking here.** MolHIV valid soup
ordering (Constant 0.861 > Null 0.841 > typed 0.810 by seed0) does not transfer
(typed 0.762 ≈ Null 0.761 > Constant 0.744). The valid→test drop is large and
model-dependent (B-Null −0.080 AUC soup, Constant −0.117, typed −0.047 for
seed0). This is exactly why the soup-only valid gate had to be reported as a
screen.

## Limits

* Candidates are **single seed** (0); the typed references are 2-seed means
  (ZINC cell A: raw 0.109361 ± 0.001731, soup 0.106717 ± 0.002698; MolHIV:
  raw 0.762605 ± 0.009688, soup 0.762095 ± 0.008464). Single-seed vs 2-seed
  means is not a paired comparison.
* Post-hoc: these test numbers were not available when the ZINC/MolHIV
  decisions were made and cannot change their status.
* ZINC test was evaluated on CPU deterministic (`vd._predict_state`), matching
  the existing cell-A test closure; the training regime was deterministic A100.
* The closures are one-shot: `official_test_unlock.json` now blocks re-runs.

## Artefacts

`results/local_token_null_test_closure/` and
`results/molhiv_local_token_null_test_closure/`: `architecture_freeze.json`,
`official_test_unlock.json`, `official_test_results.json`, `report.json`.
