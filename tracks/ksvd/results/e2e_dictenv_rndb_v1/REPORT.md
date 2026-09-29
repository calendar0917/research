# e2e_dictenv_rndb_v1 — Rolewise Nonlinear Dictionary Binding

CPU only · official test never loaded · single seed-0 trajectory.

## A. Provenance

- starting HEAD `cfe0716f3c4ac3c72df9521c9e075092b5213ac3`
- formal/analysis commit `e233beba98407491439c3763291ccc16b365aa3b`
- device `cpu` · threads `8`
- `official_test_loaded = false`

## B. Historical references (read-only; never rerun; unmatched)

- CSSD-Q1-seed0: `0.130028` — not_rerun_unmatched_context `tracks/ksvd/results/e2e_dictenv_common_subspace_dictionary_v1/training/final.json`
- CSSD-Q2: not trained (context only) — never_trained_context_only `tracks/ksvd/results/e2e_dictenv_common_subspace_dictionary_v1/zero_training/selection.json`
- FINAL-CLEAN-seed0: `0.128499` — not_rerun_unmatched_context `tracks/ksvd/results/e2e_dictenv_clean_mechanism_v1/stage_c_independence/frozen_C6_seed0.json`
- T1-tuned-seed0: `0.125765` — not_rerun_unmatched_context `tracks/ksvd/results/e2e_dictenv_t1/tuning_decision.json`

The prompt's approximate CSSD-q1/q2 values do not match the local artifacts; the exact artifacts above are the source of truth. No matched baseline was rerun.

## C. Exact RNDB architecture

```
p_{v,k}  = z_{v,k} (W_A_S[k,:] odot c_v),   c_v = q_v W_A_C / sqrt(D_A)
u_v      = p_v^common + sum_{k in K_dict} [ p_{v,k} + psi_A(p_{v,k}) ]
p^E_{uv,k} = r_{uv,k} odot c^E_{uv},  r from [+, |.|, odot] blocks
u^E_uv   = p^E,common_uv + sum_{k in K_dict} [ p^E_{uv,k} + psi_E(p^E_{uv,k}) ]
psi_A: 96->32->96 (SiLU), bias=False;  psi_E: 48->16->48 (SiLU), bias=False
K_common = {0}; K_dict = {1..32}; W1 Kaiming, W2 ~ N(0, 0.01)
```

## D. Why RNDB is new

- not F (multi-rank bilinear): F is affine in the coordinate with new heads; RNDB applies a nonlinearity per role before the sum, with no new heads.
- not FEC-D1: no 2688-D covariance/joint statistic is formed.
- not FSAB/BCE: no new binding stream or residual MLP is added.
- not T1: no coarse146 / handcrafted bypass; no raw phi inside psi.

## E. Correctness gates

- all passed: `True`

## F. Parameter budget

- parent `97727`, RNDB `105407`, new `7680`

## G. Training

- epochs `320` · wall `5727.5 s` · seed `0`
- best valid `0.139599` @ 256 · Top-5 members [248, 256, 258, 298, 314] · soup `0.133117`

## H. Performance interpretation

- `M_R = 0.133117` → band **RNDB_NO_USEFUL_TASK_GAIN**
- `G_hist = -0.003089` vs historical CSSD-q1 `0.130028` — **historical, unmatched comparison**

## I. psi mechanism

- `M_clean=0.133117` `M_psi0=0.323797` `G_psi=0.190679` (gate 0.003)
- `M_node_off=0.273887` (`G_psi_node=0.140770`), `M_edge_off=0.196141` (`G_psi_edge=0.063024`)

## J. psi health (final soup)

| operator | param norm | frac<1e-12 | W1 norm | W2 norm |
|---|---|---|---|---|
| psi_A | 6.2014 | 0.0208 | 5.1110 | 3.5120 |
| psi_E | 3.1828 | 0.0000 | 2.4176 | 2.0700 |

| role | mean ratio ||psi||/||p|| | mean ||psi|| | eff. rank |
|---|---|---|---|
| node | 2.3409 | 0.004367 | 1.003 |
| edge | 1.3267 | 0.003814 | 1.001 |

## K. Dictionary mechanism

- `M_dict0=0.421944` `G_dict=0.288827` (gate 0.010) → `load_bearing`
- health: active 32/32, effective 22.16, top1 share 0.658, recon 0.97696

## L. Assignment correspondence

- node shuffle mean delta `0.137391` (seeds [101, 202, 303, 404, 505])
- edge shuffle mean delta `0.157791` (seeds [101, 202, 303, 404, 505])

## M. Final verdict

**Case F — RNDB_NO_GO_TASK_LEVEL_MECHANISM_STRONGLY_SUPPORTED**

> Boundary outcome. The literal Case F condition is `M_R > 0.126` AND `no strong new mechanism evidence`; the mechanism evidence here is very strong (G_psi=0.190679 >= 0.003, G_dict=0.288827 >= 0.010, psi alive), so the conjunction does not hold. Case C would require `M_R <= 0.126`. No pre-registered case enumerates `M_R > 0.126` WITH strong new mechanism evidence. The dominant condition (`M_R > 0.126`, band P3, no useful task gain) forces the task-level no-go; the mechanism result is recorded rather than folded into a case. This boundary was not enumerated in the pre-registered table.

## N. Next step

Task-level no-go: RNDB does **not** improve ZINC valid MAE (band P3), and the historical CSSD-q1 comparison is worse, not an improvement. No seed 1, no rescue, no width/init/gate/lr/lambda sweep is executed this round. The rolewise nonlinearity is, however, strongly load-bearing (`G_psi`, `G_dict` far above their gates), so the mechanism finding is recorded durably; converting it into a task gain is a future hypothesis only and requires a new pre-registered round.
