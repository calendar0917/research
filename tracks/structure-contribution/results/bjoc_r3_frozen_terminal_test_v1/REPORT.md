# P3 frozen exposed-test terminal evaluation — REPORT

Protocol `cscl-bjoc-r3-frozen-terminal-test-v1` · commit `a41194153e46` · driver
`code/run_bjoc_r3_frozen_terminal_test_v1.py` (`sha256 cad877647e6f…`) ·
audit `checkpoint_audit.json` · freeze `test_prediction_freeze.json` ·
preregistration `test_protocol.json`.

**Scope.** Authorized terminal evaluation of the frozen full-10k **P3** model
(masked-SSL encoder + R3 360-epoch cosine last-5 soup, BJOC-Null, 413,505
params) on the official `ZINC(subset=True, split="test")` 1,000-molecule split,
against the historical full-10k **P0** control. Inference/audit/scoring only; no
training, tuning or model selection. All predictions were frozen and hashed
before the single test-label read of this round. The previous round's frozen
artifacts were opened read-only and were not modified.

**Exposure disclosure.** The official test instrument was already read in this
project (KSVD track 2026-10-03, then `cscl-bjoc-pretraining-mechanism-terminal-test-v1`).
This is an **exposed-test terminal evaluation**, not a first blind test. It may
not be used to select models, seeds, recipes or post-processing, and no
test-driven follow-up was run.

---

## A. One-line conclusion

**P3 official-test MAE (seed-mean) = 0.069822** (seed 0 0.069304,
seed 1 0.070340); P0 replay confirmed at 0.073957
(0.074825 / 0.073088). P3−P0 = **-0.004135**
(CI95 [-0.007342, -0.001042], per-seed
-0.005521 / -0.002748),
decision `P3_TEST_IMPROVES`. Target band 0.06–0.07: **REACHED** — P3 seed-mean official-test MAE 0.069822 is inside the 0.06-0.07 band.

---

## B. Replay gate (P0 control)

| model | expected MAE (previous round) | replayed MAE | diff | npz max abs diff | ok |
|---|---:|---:|---:|---:|---|
| P0 seed 0 | 0.074825 | 0.074825 | +4.60e-07 | 0.00e+00 | True |
| P0 seed 1 | 0.073088 | 0.073088 | -3.19e-07 | 0.00e+00 | True |
| P0 seed-mean | 0.073957 | 0.073957 | — | — | True |

Checkpoint hashes, 413,505-parameter audits, forbidden-module scans, train-only
normaliser and the historical test structure fingerprint all pass
(`checkpoint_audit.json`).

## C. Official-test metrics

| key | arm | seed | MAE | RMSE | median abs | p90 abs |
|---|---|---:|---:|---:|---:|---:|
| P00 | P0 | 0 | 0.074825 | 0.160208 | 0.039154 | 0.151002 |
| P01 | P0 | 1 | 0.073088 | 0.138969 | 0.045494 | 0.142503 |
| P30 | P3 | 0 | 0.069304 | 0.146028 | 0.036150 | 0.137556 |
| P31 | P3 | 1 | 0.070340 | 0.137783 | 0.041737 | 0.149839 |

| seed-mean | MAE | RMSE | median | p90 |
|---|---:|---:|---:|---:|
| P0 | 0.073957 | 0.149588 | 0.042324 | 0.146752 |
| P3 | 0.069822 | 0.141905 | 0.038943 | 0.143697 |

Primarily requested contrast (paired per molecule, 5000 bootstrap resamples):

| statistic | value |
|---|---:|
| P3−P0 seed-mean Δ MAE | -0.004135 |
| 95% CI | [-0.007342, -0.001042] |
| per-seed Δ (seed 0 / seed 1) | -0.005521 / -0.002748 |
| both seeds improve | True |
| better fraction | 0.5690 |
| ordinary (y > −3, n=923) Δ | -0.002450 |
| tail (y ≤ −3, n=77) Δ | -0.024329 |
| top-10 removed Δ | -0.004018 |
| context: P3−B0 seed-mean Δ | -0.024804 |

Decision: **`P3_TEST_IMPROVES`**. Target band judgement (P3 seed-mean):
**reached** (P3 seed-mean official-test MAE 0.069822 is inside the 0.06-0.07 band).

## D. What this does and does not show

* Shows: the exact exposed-test MAE of the frozen P3 checkpoints and the exact
  same-round paired difference to the frozen P0 control, with the previous
  round's P0 numbers reproduced to <1e-05 MAE.
* Does not show: a first blind test result; a statement about arbitrary new
  molecule distributions; any justification for changing the model, seeds,
  recipe, checkpoint choice or error post-processing after seeing the test.
* P3 was chosen before this evaluation by the recipe round on the exposed
  official-valid split; the test numbers here are descriptive terminal
  measurements, not a new selection.

## E. Artifact hashes

Checkpoints (read-only inputs; hashes also match their original training records):

| key | file | sha256 |
|---|---|---|
| P00 | `tracks/structure-contribution/results/bjoc_full10k_pretraining_transfer_v1/soup_P_seed0.pt` | `73118e9be466c62a252eacb83d9c86d1cb4bf0ce01478db2f278b7c27a5fab2a` |
| P01 | `tracks/structure-contribution/results/bjoc_full10k_pretraining_transfer_v1/soup_P_seed1.pt` | `bf640cd8c11a9f6e0d11ceccd318185e6067ef2ba5a950b4774d5185a9ab053c` |
| P30 | `tracks/structure-contribution/results/bjoc_recipe_tuning_v1/soup_R3_seed0.pt` | `a021d36e3c07039cf191895f8037d09bf2e734101863dccf8993a69de737f782` |
| P31 | `tracks/structure-contribution/results/bjoc_recipe_tuning_v1/soup_R3_seed1.pt` | `b622824a8eb08043777391d7ed617da2639658b413dd83a05c57f81df53d03d9` |

Frozen label-blind test predictions (written before the label read this round):

| key | file | sha256 |
|---|---|---|
| P00 | `tracks/structure-contribution/results/bjoc_r3_frozen_terminal_test_v1/predictions/test_P00.npz` | `e81c88ca428103f8fa0d0adde88dfbc9c8dea0b3df64e71a4007430ca8e704bc` |
| P01 | `tracks/structure-contribution/results/bjoc_r3_frozen_terminal_test_v1/predictions/test_P01.npz` | `ccba8bc4bdccce83c6c6bb605e8fb3248a4aa38e6ad6eec13524687cda35e427` |
| P30 | `tracks/structure-contribution/results/bjoc_r3_frozen_terminal_test_v1/predictions/test_P30.npz` | `dd1dc3c47203bc86b13564776f8dd590c750ff6499a9a1c46b07d0052d0a75dd` |
| P31 | `tracks/structure-contribution/results/bjoc_r3_frozen_terminal_test_v1/predictions/test_P31.npz` | `b8699768feca42e499de60dd260cfe314e628bec0186b400b70bd8698b25004f` |

* combined prediction sha256: `eae356d5372b0bf147ce46fe992cba92ae1b6f4f762b2756121421daaf375961`
* test objects combined sha256: `6b0fd9e0d6bf2fa1212144c97b58a6156802934da985f13593c8e7d6a0167b2b`; signature `{'n_molecules': 1000, 'n_atoms': 23117, 'n_bonds': 24860, 'n_junctions': 34595}`; matches the previous round's objects: `True`
* test structure fingerprint: processed `bd7a1ceb5b4ec126160bc095fb2bf28c0019a736c5575afa3ed11fb844a3aced`, structures `75668e7838f6efcde85e789b7064b1ce9fb99e6561c40ec63a6cf790ad5010ce`
* test labels (single read this round): `3afb4be3e3634fec4178dec810dd0f71ef7a1519fa24cef4cb11206b809e2d0e`; identical to the previous round's stored copy: `True` (max abs diff 0.0)

Code hashes recorded in the prediction freeze:

| component | sha256 |
|---|---|
| driver | `cad877647e6f992a20c891b13b1048cbe6347b924485bc375adc3468a25ab52e` |
| terminal_test_driver | `42c6c0b483ead0aae1f931da1d3fa14fb4253009acc2805ffa98989d361fd2f0` |
| previous_round_driver | `40ec29443bec5f83e20b1dab2dcbdc5290191360c78944408f67f81626c3a90e` |
| model | `1c83a97ece60a151c6fa0c1ad5d2022b470ede9356b23d8d7b1e6b62daa71f4f` |
| objects | `8f4ba44dc3c047d674a1a6832f7ed8265eaa2457b15e820b3973a4cbd3a0861c` |
| masked_module | `f27dbfe36ac5a9a0a1915320dba7c955115ccfa36e6b4755f2d2d62d8efb1489` |

Execution environment: `Linux-6.18.54-1-lts-x86_64-with-glibc2.44`, python `3.12.14`, torch `2.5.1+cu124` (threads 8), device `cpu` (no CUDA used).
