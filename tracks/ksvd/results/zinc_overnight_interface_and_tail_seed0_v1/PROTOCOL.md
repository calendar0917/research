# PROTOCOL — zinc-overnight-interface-and-tail-seed0-v1

Single seed (seed 0).  Train-only exploration on the frozen 8000 fit / 2000
internal dev fold; at most one final frozen official-valid read; official-test
is never loaded.  Every number in this round is a single-seed exploratory
signal on a reused internal dev, not an independent confirmation.

## 0. Sources (verified before fitting)

| object | source | verification |
|---|---|---|
| S_M raw soup (structural `D_star`, `U`, `common_rms`, `kappa`) | `tracks/ksvd/results/zinc_structure_semantic_factorial_seed0_v1/S_M_raw_soup_state.pt` (branch `task/zinc-structure-semantic-factorial-seed0-v1`, HEAD `4083ee38d8785c5a44d1f9628a5baeed568d4c34`, trained at `118e2481362f`) | state hash `2def4cb64c1350a60348fb6fc7ce88a07bffe563367ac7e9666ee6f7dff5c5b8` |
| compressed `S_M` body `B` (reference) | `tracks/ksvd/results/zinc_zero_binding_baseline_seed0_v1/S_M_deploy_state.pt` (exec `dc96fac`, result `53ab987`) | 267,611 params; full dev replay vs `phaseA_sm_replay.npz` `max|diff| = 1.9e-06` |
| fold objects / prep | `tracks/ksvd/results/zinc_joint_dictionary_decision_v1/prep/fold_objects.npz` | fit hash `165e87ef…`, dev hash `fb8b7806…`; dev strata `1926/65/9` |
| targets | `tracks/ksvd/results/zinc_full_cycle_target_decomposition_v1/target_decomposition.npz` | y (positional), c, g, k, gid; 10,000 rows |
| CPU tail inputs | `zinc_frozen_chemistry_learned_cycle_v1/T25_all.npz` (sha `dc2e1516…`), `zinc_full_cycle_target_decomposition_v1/O_seed0_predictions.npz` (`h_raw`) | P_U reproduces the released `P_seed0` `q_fit`/`q_dev` exactly (`max|diff| = 0.0`) |

`load_split("train")` is not used; only `encoded_train.pt` + `env_train.pt`
are loaded.  Official-valid is not touched until the single frozen evaluation
(`final_eval.py --mode freeze` then `--mode heldout`), and only if a candidate
passes a pre-registered gate.

## 1. Main question and interface

On the compressed reference `B` (fusion input `[Sem108; size2] = 110-D`,
task dictionary `D_L/V_L`, static relations, topology25, C6 mask; no slot
binding), insert a zero-initialised residual adapter **after the fusion output
(width 144) and before the shared task dictionary bridge**:

```
h = h0 + delta(F)
delta = Linear(5102, 24) -> SiLU -> Linear(24, 144)
F = [Sem110 (110), flatten(K_A) (2688), flatten(K_E) (2304)]
```

* structural code: `S_M`'s `D_star [65,32]`, `U`, `common_rms`; sparse =
  tied-IHT (s=8, 10 steps) on the residual `r = phi - (phi U) U^T`; dense =
  `kappa * (r Dbar)` with `kappa` re-estimated once on the 8000 fit rows
  (`0.380682`), matching the sparse alpha RMS.
* `K_A^J = sum_v a_v (x) t_v` per root/shell (3x32x28); `K_A^M =
  (sum a)(x)(sum t)/n`, `n=0 -> 0`.
* `K_E^J = sum_uv g_uv (x) bond_onehot`, `g_uv=[a_u+a_v, |a_u-a_v|, a_u*a_v]`
  per root/shellpair (6x96x4); `K_E^M = (sum g)(x)(sum bond)/n`.
* `tau_A = 0.0252396`, `tau_E = 0.061480` (float64 RMS over all 8000 fit rows
  and all block elements of the **Sparse+Joint** features), frozen for all arms.
* last adapter layer weight/bias zero, so every arm starts exactly at `B`
  (verified `max|diff| <= 1.0e-05`; measured `9.5e-07` fit, `1.9e-06` dev).

Parameter accounting (measured): body 267,611 + `D` 2,080 + adapter 126,072 =
**395,763** (below the original Full 408,651).

## 2. Fixed training matrix

| stage | arms | body/`D` | epochs | loss | soup |
|---|---|---:|---:|---|---|
| Phase 1 | `R_CS` (Sem only), `R_SM` (sparse marginal), `R_SJ` (sparse joint), `R_DM` (dense marginal), `R_DJ` (dense joint) | frozen / frozen | 160 | mean `L1(y)` | 156–160 |
| Phase 2 | `A0` (delta x0), `C` (Sem-only delta), `T` (selected Phase-1 spec) | trainable / trainable | 240 | `L1(y) + 33.95873017865987 * rec` | 236–240 |
| Phase 3 (gated) | `A0`, selected `C/T` | trainable / trainable | 240 on all 10k | same | 236–240 |
| CPU tail | `P_U` (unweighted) and `P_B` (severity-balanced) 25->64->32->1 heads | frozen `h_raw` (`O_seed0`) | 300 | weighted/unweighted mean `L1(c)` | 296–300 |

* Adam coupled L2, lr `1e-3`, wd `1e-5`, global clip 5.0, batch 128,
  no scheduler.  Data schedule seeded `0 + 101` (per-epoch permutations only);
  diagnostics/eval use separate RNG state save/restore.  Shared init hash,
  stream hash and step counts are recorded per arm.
* Phase 1 optimiser holds the adapter only; the parent is eval mode and its
  state hash is checked byte-identical after smoke training.
* Phase 2/3 encoding mode (sparse vs dense) is fixed to `T`'s Phase-1 mode for
  all three arms so the reconstruction/clipping paths match (`A0` and `C` use
  the same reconstruction term and the same `D`).
* CPU `q_B` group weights are fixed on the fit side: groups `k=0, -1, -2,
  <=-3`; `w_g = N_fit/(G n_g)`; max weight 400, mean 1, effective sample size
  68.29 (8k fit has only 5 rows with `k<=-3`).

## 3. Endpoints, selection and gates

* Primary endpoint: internal-dev G0 (`k=0`, 1926 rows) calibrated MAE.
  Secondary: dev overall calibrated MAE.  `gain = reference MAE - candidate
  MAE` (positive = candidate better); raw and one train-median calibration
  (eval-mode fit) reported side by side.
* Phase-1 `T` selection (exploratory, fixed before Phase 2): among
  `R_SM/R_SJ/R_DM/R_DJ`, if any has dev G0 cal better than `B`, take the
  lowest; ties and the no-better case fall back to `R_SJ`.
* Phase-2 gate (per arm vs `A0`): dev G0 cal gain `>= 0.003` with paired
  bootstrap 95% lower bound `> 0`; dev overall cal gain `>= 0.003`; overall
  raw gain `> 0`; replay/identity/calibration checks pass.  `C` passing is a
  valid performance candidate even if it does not support the dictionary
  claim; `T` must additionally be reported against `C`.
* CPU tail gate (`P_B` vs `P_U`): dev overall cal gain `>= 0.003`, overall raw
  gain `> 0`, G0 cal worsening `<= 0.001`, identity/replay checks pass;
  paired CI must be reported and a CI lower bound `<= 0` marks a weak
  candidate that does not advance.
* Phase 3 is bought only if `C`/`T` or `P_B` passes; at most one family; the
  decision file is written before the official-valid read.  No posterior
  combination, no ensemble, no test tuning.

## 4. Statistics

Paired bootstrap 1000x, seed `20261003`, one shared index set per replicate
across arms (G0 stratum 1926, overall stratified `1926/65/9`).  Group
contributions use `sum(|error|)/N_total` and must sum to the overall value.
Witnesses: identical predictions -> gain 0; swapped arms -> sign mirror;
constant output shift 0.5 -> gain bounded by 0.5.  Extreme errors are listed
row-by-row with deletion sensitivity, but no gate is revised after inspection.

## 5. Compute and boundaries

* GPU: `res-2` (Slurm, pool `res2-cu124`, c05), A100-PCIE-40GB,
  driver `525.85.12`, torch `2.5.1+cu124`, FP32, no AMP/DDP.  Max 2
  concurrent GPUs, <= 6 GPU-hours in total, <= 10 formal GPU trajectories.
* Local CPU: `torch.set_num_threads(4)` for the CPU tail runner and
  `set_num_threads(8)` for local phase-0/analysis; no other heavy local jobs.
* New-training cutoff 420 min after the first tool call; delivery by 460 min.
* Not in scope: amplitude/WD/lambda/width grids, reader widening, node
  revival, old-trajectory extension, optimiser/loss swaps, new message
  passing/transformers, posterior combination, test tuning.

## 6. Deviations recorded at freeze time

1. The first Phase-1 submissions failed before any training (missing
   `interface_stats.json` on the remote because `tracks/*/results/**/*.json`
   is git-ignored).  Fixed by force-adding Phase-0 evidence; the failed jobs
   never trained a step and are not trajectories.
2. The first CPU-tail implementation drew a fresh permutation per batch
   instead of per epoch; it was corrected **before** any CPU result was used,
   and `P_U` then reproduced the released `P_seed0` exactly (`0.0`).  (The
   buggy run is not a reported result.)
