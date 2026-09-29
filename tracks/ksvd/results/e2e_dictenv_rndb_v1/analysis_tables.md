# `e2e_dictenv_rndb_v1` — analysis tables (durable, tracked)

Round `E2E-DictEnv-RNDB-v1` (Rolewise Nonlinear Dictionary Binding).
CPU only · one seed-0 trajectory · official ZINC test never loaded.
Machine-readable evidence lives in this directory as git-ignored local JSON
(`summary.json`, `run_seed0.json`, `correctness.json`, `smoke.json`,
`dictionary_health.json`, `psi_disable.json`, `dictionary_zero.json`,
`assignment_shuffle_*.json`, `historical_references.json`,
`parameter_audit.json`). This file is the durable numeric snapshot.

Verdict: **Case F — `RNDB_NO_GO_TASK_LEVEL_MECHANISM_STRONGLY_SUPPORTED`**
(boundary; task-level no-go, mechanism very strongly supported).

## 1. Provenance

| item | value |
|---|---|
| starting HEAD | `cfe0716f3c4ac3c72df9521c9e075092b5213ac3` |
| implementation commit | `e233beb` |
| preregistration sha256 prefix | `5d6bb0d87305` |
| device | cpu (8 threads) |
| official test loaded | false |
| parent | CSSD-q1, commit `5f0f284` |
| parent params / RNDB params / new | 97727 / 105407 / 7680 |
| relative param increase | 0.07858 (cap 1.10) |

## 2. Correctness gates (all PASS)

| gate | check | result |
|---|---|---|
| G0 | decomposition `sum_k p_k == p_parent`, node synthetic / real | 4.3e-19 / 4.3e-19 (float64) |
| G0 | decomposition, edge synthetic / real | 3.5e-18 / 3.5e-18 (float64) |
| G1 | psi-off prediction / environment vs parent | 0.0 / 0.0 (bit-identical) |
| G2 | `psi(0)=0`, inactive role strict zero | exact |
| G3 | role permutation equivariance | ≤ 1e-5 (obs ~1e-9) |
| G4 | environment invariant to global/relation/topology; new params exactly 4 psi tensors | true |
| G5 | `RNDB_MASK is cm.C6_MASK`, `c6_equivalence_check()`, backend reuse | true |
| G6 | official-test blocker raises | true |

Focused tests: `tracks/ksvd/tests/test_e2e_dictenv_rndb_v1.py`, 12/12 pass.

## 3. Smoke (mechanism trainability only)

| item | value |
|---|---|
| epochs / train / valid | 8 / 2048 / 512 |
| finite loss + predictions | true |
| `grad(D)`, `grad(psi_A.W1/W2)`, `grad(psi_E.W1/W2)` | all non-zero |
| best valid MAE | 0.664893 |

## 4. Training diagnostics

| epoch | valid MAE | `grad(D)` | `grad psi_A W1` | `grad psi_A W2` | `grad psi_E W1` | `grad psi_E W2` | node ratio | edge ratio | dictionary active / N_eff |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.908482 | 3.14e-2 | 7.8e-9 | 1.5e-7 | 2.4e-5 | 1.8e-5 | 0.0001 | 0.0046 | 31 / 20.81 |
| 20 | 0.257444 | 8.20e-2 | 8.3e-3 | 1.2e-3 | 1.6e-2 | 2.3e-3 | 3.056 | 2.873 | 30 / 22.49 |
| 40 | 0.236941 | 1.46e-1 | 1.0e-2 | 5.3e-3 | 1.8e-2 | 1.2e-3 | 3.066 | 2.250 | 31 / 22.06 |
| 80 | 0.173122 | 5.81e-2 | 1.3e-2 | 1.1e-3 | 7.9e-3 | 8.6e-3 | 3.076 | 1.717 | 31 / 21.67 |
| 160 | 0.160710 | 6.50e-2 | 1.6e-2 | 1.8e-2 | 6.6e-3 | 3.4e-3 | 2.698 | 1.459 | 32 / 21.44 |
| 240 | 0.146161 | 6.93e-2 | 2.1e-2 | 1.3e-2 | 6.1e-3 | 5.1e-3 | 2.501 | 1.365 | 32 / 22.61 |
| 320 | 0.169487 | 1.19e-1 | 1.9e-2 | 1.4e-2 | 1.9e-2 | 2.0e-2 | 2.341 | 1.327 | 31 / 21.43 |

(node/edge ratio = mean `||psi(p)|| / ||p||` on a fixed train batch.)

## 5. Task result

| item | value |
|---|---|
| best valid MAE | 0.1395986 @ epoch 256 |
| soup members | [248, 256, 258, 298, 314] |
| soup member MAEs | [0.141603, 0.139599, 0.141693, 0.140422, 0.141401] |
| **`M_R` (soup)** | **0.1331175** |
| band | P3 `RNDB_NO_USEFUL_TASK_GAIN` |
| historical CSSD-q1 soup | 0.1300280 (unmatched, read-only) |
| `G_hist` | -0.003089 (unmatched) |

## 6. Frozen interventions

| intervention | valid MAE | delta vs `M_R` | gate |
|---|---|---|---|
| `M_R` (soup) | 0.1331175 | — | — |
| `M_psi0` (psi_A = psi_E = 0) | 0.3237969 | **+0.190679** | 0.003 |
| node-only psi off | 0.2738873 | +0.140770 | — |
| edge-only psi off | 0.1961412 | +0.063024 | — |
| dictionary `alpha -> 0` | 0.4219441 | **+0.288827** | 0.010 |

Assignment shuffle (frozen parent semantics):

| seed | node delta | edge delta |
|---|---|---|
| 101 | +0.14194 | +0.14383 |
| 202 | +0.13623 | +0.15142 |
| 303 | +0.10372 | +0.16518 |
| 404 | +0.15336 | +0.17727 |
| 505 | +0.15170 | +0.15126 |
| mean | **+0.137391** | **+0.157791** |

## 7. Final model health

| item | value |
|---|---|
| `psi_A` param / W1 / W2 norms | 6.201 / 5.111 / 3.512 |
| `psi_E` param / W1 / W2 norms | 3.183 / 2.418 / 2.070 |
| `psi_A` fraction < 1e-12 | 0.0208 |
| psi alive | true |
| dictionary active atoms | 32/32 |
| dictionary effective atoms | 22.16 |
| dictionary top-1 activation rate | 0.658 |
| residual reconstruction relative | 0.97696 (frozen full-descriptor diagnostic) |
| dictionary movement (Frobenius) | 5.37 |
