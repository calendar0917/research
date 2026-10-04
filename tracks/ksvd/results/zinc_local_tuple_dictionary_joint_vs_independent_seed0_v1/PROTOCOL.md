# PROTOCOL — `zinc_local_tuple_dictionary_joint_vs_independent_seed0_v1`

Frozen **before any dev/target score is computed or read** for this round.
Slug: `zinc_local_tuple_dictionary_joint_vs_independent_seed0_v1`
Branch: `task/zinc-local-tuple-dictionary-joint-vs-independent-seed0-v1`
Seed: `20261004` (single seed; no seed search).
All artifacts under `tracks/ksvd/results/zinc_local_tuple_dictionary_joint_vs_independent_seed0_v1/`.

## 1. Scientific question

Under the *same compressed skeleton and the same capacity*, does a **shared local dictionary that is
given direct access to the correspondence** `(root's local structure descriptor, root atom type,
neighbour atom type, connecting bond type)` generalise the chemical target `g` better than
**combining the same raw attributes by their marginals alone**?

The correspondence is an *explicit* per-root incidence operator, not a learnable attention map:

```
C_v[t, a]      = number of neighbours u of v with bond type t and atom class a          (real incidence)
C_ind_v[t, a]  = n_t(v) * n_a(v) / d_v   with d_v = sum_{t,a} C_v[t,a]                  (marginal product)
```

Both arms see the same tuple encoder; only the scalars multiplying each tuple differ.

## 2. Operator (frozen)

For root row `v` with enumerated neighbours `(u, t)` (bond type `t ∈ {1,2,3}`, raw category
`a ∈ {0..20}` mapped into 28 atom categories):

* tuple input `x(v,a,t) = [phi_v (65, root's local structure descriptor);
  onehot28(a_v); onehot28(a); onehot4(t)]` → **125-D**.
* Union-support pairs `T(v) = { (t,a) : C_v[t,a] > 0 or C_ind_v[t,a] > 0 }`.
* J weights: `w_J(v,t,a) = C_v[t,a] / d_v`.
* I weights: `w_I(v,t,a) = C_ind_v[t,a] / d_v`.
* Per-root pooled code `e_arm(v) = sum_{(t,a)∈T(v)} alpha(v,t,a) · w_arm(v,t,a) ∈ R^64`,
  with `alpha = f_D(x) ∈ R^64` produced by the shared local dictionary.

A Python-level reconstruction of `C`, `d_v`, `n_t`, `n_a` from the **raw** `train.pt`
(`edge_index` deduplicated by canonical `(min,max)` key, first occurrence wins, symmetric
double-edge consistency checked) must agree with the frozen env cache (`env_train.pt`
`bond_root/bond_u/bond_v/bond_type/atom`) and satisfy
`C_ind == marginal product ÷ d_v` to `1e-10` in float64 and `1e-5` in float32.
This is required before training.

## 3. Arms (exactly J and I, seed 0, 240 epochs / 15120 steps each)

| arm | weight source |
|-----|----------------|
| J   | real incidence `C_v[t,a]/d_v` |
| I   | marginal product `C_ind_v[t,a]/d_v` |

* Shared encoder `f_D`: **tied IHT dictionary**, `D_loc ∈ R^{125×64}`, `s = 8`, **10 IHT steps**,
  step size `eta = 1/(1.05 · λ_max(D_bar^T D_bar))` (`D_bar` = column-normalised `D_loc`).
* New parameters: `125×64` (`D_loc`) + `342×64` (`W_loc`) = **29,888**.
  Expected total **297,499** = 184,667 body + 82,944 bridge + 29,888 local.
* `D_loc` shared init (generator seed 20261004) identical in J and I; `W_loc` exactly 0;
  body = M_g **untrained** init; never warm-started from any trained soup.
* `preact = fusion0(Sem110) + W_loc @ (kappa · e_arm)`; `kappa = clip(pi/2 · RMS(D_bar), max=kappa_ref)`
  fitted on fit-root frame statistics only; `kappa` stored in the tuple index.

## 4. Reference B (read-only)

Historical `zinc_chemistry_dictionary_vs_mlp_seed0_v1` **M_g raw soup**:
`M_raw_predictions.npz` + `M_meta.json`. Fit-median bias recomputed once from
`g_fit − raw_soup_fit`; must reproduce published fit cal `0.029280`, dev overall cal `0.103717`,
G0 cal `0.101875` within `1e-5`. No B re-training, no B re-evaluation on valid/test.

## 5. Training regime (identical to the paired g round, `zw` constants)

* Adam, `lr = zw.LR`, `weight_decay = zw.WEIGHT_DECAY`, `grad_clip = zw.GRAD_CLIP`,
  batch size `zw.BATCH_SIZE`, 240 epochs, soup epochs `zw.SOUP_EPOCHS`, seed `20261004`.
* Data: frozen fold `zw.build_fold()` — 8000 fit / 2000 dev, hashes
  `7bf1cfb8…` (fit) and `a61c8010…` (dev); target `g = y − c` from
  `zcdm.TARGETS_NPZ` (fit-only constants, sha256 `e2adf5f2…`).
* Schedule/data stream hashes must match `zw`/`zcdm` frozen hashes
  (schedule `7b11a529…`); arms must produce the *same* dropout mask stream
  (both built and trained from the same seed; verified in smoke).
* No dropout at evaluation; dev probes in eval mode.

## 6. Required checks before training

1. `--build-tuple`: operator build, fit-only `phi` scaler, kappa sample, marginal
   consistency, double-edge audit, masked-self checks recorded in `tuple_index_meta.json`.
2. `--operator-checks`: raw-vs-cache incidence reference, float64/32 marginal identity,
   single-vs-batch / two-graph offset / shuffled-order equivalence on the production model,
   `J`/`I` same-code (same-weights) identity, 4-term synthetic star witness
   (`e_I` identical, `e_J` differs), fit contrast statistics, exact-duplicate-`phi`
   different-`C` witness. All must pass.
3. `--phase-a`: historical anchor reproduction, init identity (J vs I; J/I vs fresh M init in
   eval mode), parameter audit, environment checks, input manifest.
4. `--smoke` (CPU): init identity, target-never-read / label-permutation invariance,
   loss-is-L1-on-`g`, first-step `W_loc` gradient nonzero, `D_loc` path established after a
   nonzero `W_loc`, ablation changes the function, endpoint-offset equivalence,
   dropout-stream identity, tiny schedule match.

## 7. Analysis (after both arms finish; dev only)

Paired bootstrap: `N_BOOT = 1000`, seed `20261004`, **shared resample indices** across B/J/I.
Gains (positive = J/I better than B):
`gain_BJ = MAE_B − MAE_J`, `gain_BI`, `gain_IJ = MAE_I − MAE_J` (oracle raw minus candidate),
computed for **dev G0 (k=0)** and **dev overall**, on calibrated (`+fit-median bias`) and raw scales.
Identity check: gains recomputed from the bootstrap replicates must equal the point definitions.

* **Gate A (per arm vs B, exploratory)**: G0 cal ≥ +0.003 **and** CI_low > 0; overall g-cal ≥ +0.003
  **and** CI_low > 0; G0/overall raw gains > 0; contracts valid. Otherwise
  `EXPLORATORY_PERFORMANCE_SIGNAL` with reasons.
* **Gate B (J vs I, primary)**: `JOINT_SUPPORT` (G0 cal ≥ +0.003 & CI_low>0, G0 raw>0,
  overall cal ≥ −0.001); `INDEPENDENT_SUPPORT` (G0 cal ≤ −0.003 & CI_high<0, G0 raw<0,
  overall cal ≤ +0.001); `LOCAL_EQUIVALENCE` (G0 and overall cal 95% CIs inside [−0.003,+0.003]
  and no raw≥0.003 opposite); else `INCONCLUSIVE`; `INVALID` on contract violation.
* **Sensitivity**: drop the single dev row with the largest `|err_B|+|err_J|+|err_I|` and repeat
  the J/I point estimates; report stability.
* **Contributions**: per `k`-group (`zw.GROUP_NAMES`) MAE and contribution to overall MAE,
  B/J/I, fit and dev.
* **Mechanism health**: `D_loc`/`W_loc` gradient norms at logged epochs, alpha sparsity (expected 8)
  and activity, root-code dead dims, `D_loc`/`W_loc` norms and drifts, trained J/I weight deltas.
* **Three frozen forward interventions** (dev sample, eval mode, same seed):
  (a) original, (b) `W_loc` zeroed, (c) `J↔I` operator switch; report prediction-change and
  error-change separately.
* **Replay**: re-run a fixed dev subset through the saved soup state; max |Δ| ≤ `1e-5`.
* **Budget**: `budget.json` with wall clock, GPU-hours, device, pool (≤180 min wall from first
  tool call, compute stopped at 150 min; ≤1.2 GPU-h; ≤2 concurrent GPUs; CPU ≤8 threads).

## 8. Prohibitions and stopping rules

* No official valid/test reading or re-evaluation; no ring head; no 10k confirmation; no new seeds,
  folds, or configuration search; no B re-training.
* Stop all own Slurm jobs when the round ends. Do not overwrite historical files; no push/merge.
* If an operator/contract check fails: stop, classify `INVALID` where applicable, and report.
