# E2E-DictEnv-M1 — valid-side analysis

Round **E2E-DictEnv-M1** · protocol `e2e_dictenv_m1` · commit
`39cda1ba1680c6cdeb3479d4b2a49d50dd03af49`.
Pre-registration: [`e2e_dictenv_m1_preregistration.md`](e2e_dictenv_m1_preregistration.md).
Implementation: [`e2e_dictenv_m1_implementation.md`](e2e_dictenv_m1_implementation.md).

Official MolHIV **test was not read** at the time this analysis was written;
the verdict below is frozen on official-valid only.

## 1. Question

Does the P1 primitive-only dictionary core — a frozen-shape, label-free
topology dictionary with tied-IHT coding, no message passing and no recurrence —
transfer to OGBG-MolHIV as a **valid and terminal** architecture, after the P1
dictionary is re-fit label-free on MolHIV official-train topology?

## 2. Frozen configuration

| item | value |
|---|---|
| φ65 | `fsar_r2_ar0.build_phi` (P1 coordinate, unchanged) |
| dictionary | K-SVD `K=32, s=8, IHT=10, epochs=10, seed=20260924`, 500,000-node deterministic sample of 830,936 official-train nodes |
| `D` sha256 | `0aefb1702d6836339f7c4d33e7123d07e15434bee372b57780d894c50b72c0eb` |
| fit mse | 0.0008324875 (1096 s) |
| anchor | 62-D, official-train per-coordinate scaler |
| `lambda_M` | 61.9921982356686 (`= 0.25 × 247.9687929426744`) |
| params | 102,325 (80k–130k budget) |
| seeds | 0, 1 |
| horizon / patience | 240 / 40 |
| soup | Top-5 valid-AUC checkpoints, equal weight |

Chemistry enters only through OGB categorical embeddings (atom 24-D, bond 12-D,
per-field embedding sums); the pair relation is pure topology with P1's 5
distance buckets, and 52.2 % of official-train node pairs clip into the `5+`
bucket (MolHIV molecules are much larger than ZINC's).

## 3. Official-valid results

| seed | raw best AUC | epoch | soup AUC | soup members | epochs run | wall |
|---|---|---|---|---|---|---|
| 0 | 0.814328 | 35 | **0.808532** | 28/35/45/59/66 | 75 | 1634 s |
| 1 | 0.836603 | 20 | **0.828391** | 16/20/21/41/43 | 60 | 1287 s |

* mean soup AUC **0.818462 ± 0.009930** (population std over 2 seeds)
* best soup AUC **0.828391** (seed 1)
* verdict band `M1_TRANSFER_STRONG` (≥ 0.80)

Both seeds early-stopped well inside the horizon (75 and 60 epochs), so the
horizon/patience schedule is not the binding constraint.

## 4. Context (not directly comparable protocols)

| reference | split | score |
|---|---|---|
| this round, 2-seed soup ensemble | valid | 0.8184 mean (best 0.8284) |
| `molhiv_patch_path_pooling` | test | 0.785195 |
| `molhiv_recurrent_pair_centre` (mean) | test | 0.762605 |
| published GIN | test | 0.756 |
| published GPS | test | 0.788 |
| published CIN-small | test | 0.801 |

Valid-side numbers sit above the earlier MolHIV test numbers, and MolHIV's
scaffold split typically costs ~0.03–0.05 AUC valid→test. The terminal read is
required before any comparison is claim-worthy.

## 5. Mechanism notes (frozen, report only)

* Dictionary movement `‖D̄_soup − D̄_init‖ / ‖D̄_init‖` = 0.9999 (seed 0) and
  0.8935 (seed 1) — the task phase re-learns the dictionary almost entirely, so
  the reported result belongs to the *learned* dictionary, not the K-SVD
  initialisation. This mirrors the P1 finding that the dictionary is load-bearing
  and task-shaped.
* Exact top-8 sparsity held throughout (gate G2, smoke `max_l0 = 8`).
* Relabel invariance `max|Δlogit| = 6.6e-7` (gate G12).

## 6. Status

* all 15 correctness gates G0–G14 passed (`correctness.json`)
* GPU smoke passed (`smoke.json`), peak 170.7 MB
* `freeze` recorded (`architecture_freeze.json`), `analyze` recorded
  (`decision.json`, verdict `M1_TRANSFER_STRONG`)
* terminal official-test read: see [`e2e_dictenv_m1_test_read.md`](e2e_dictenv_m1_test_read.md)
