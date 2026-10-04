# METHOD_CONTRACT — `zinc_local_tuple_dictionary_joint_vs_independent_seed0_v1`

What is actually computed, where, and what is held fixed. Frozen before the dev run.

## 1. Data and frames

* Raw molecules: `data/ZINC/subset/processed/train.pt` (10,000 molecules, PyG collated dict).
  Used **only** locally to (a) build the tuple index, (b) cross-check the incidence operator,
  (c) write manifest hashes. The GPU training host reads the frozen env cache and the committed
  tuple index, never the raw file for new label access.
* Fold: `zw.build_fold()` → fit 8000 (`7bf1cfb8…`), dev 2000 (`a61c8010…`).
* Target: `g = y − c`, `zcdm.TARGETS_NPZ` sha256 `e2adf5f2…`, fit-only constants (see `ERRATA.md`).
* `phi65` scaler: mean/std over **fit roots only** (337,586 rows), std floored at `1e-3`;
  stored in the tuple index. No dev/valid/test statistics enter the scaler or kappa.

## 2. Tuple index (committed artifact `local_tuple_index.npz`)

Built by `--build-tuple` from raw graphs:

* `root_base` (10001): molecule → first global root.
* `pair_ptr` (231665): global root → slice in the union-support pair table.
* `pair_t`, `pair_a`, `pair_wJ`, `pair_wI`: per-pair bond type, neighbour atom class,
  and the two scalars. Total pairs **508,032** (mean 2.19/root; 100% of realised incidence pairs
  included; union support by construction).
* `root_atom` (231664), `phi_mean`/`phi_std` (65), `phi_scale` (scalar), `kappa` (scalar),
  `kappa_sample` (seed-20261004 root sample used for the frame fit).
* Marginal identity: `pair_wI` sums to 1 per root (`1e-10` float64); `pair_wJ` sums to 1 per root;
  both use the same union support.

Recorded build statistics: marginal max delta float64 `0.0`, float32 `5.96e-8`;
roots with any `C ≠ C_ind` contrast on fit = 38,722 / 185,204 (20.9%);
realised fit pairs with `wJ ≠ wI` = 173,878 / 337,586 (51.5% of realised; 42.8% of union support).

## 3. Model

`LocalTupleFull(zbn.DeployFull)`:

* Body/bridge: exactly the `zbn`/`zw` 110-D-interface skeleton; `local_dictionary_bridge`
  replaced by `zw.MLPBridge(D_L, V_L)` (bias-free 144→288→144 SiLU, 82,944 params) to match the
  M_g family.
* `LocalTupleEncoder`: `D_loc` (125×64, IHT-tied), `W_loc` (342×64, init exactly 0),
  `kappa` scalar; `forward` returns `kappa · e_arm`; `environments_masked` adds
  `F.linear(kappa·e_arm, W_loc)` to `fusion0(Sem110)`.
* Codes are computed on the *root rows of the concatenated batch*:
  `global_root = root_base[local_mol_id][batch] + (row − ptr[batch])`; tuple rows are gathered
  via `pair_ptr` repeat-interleave; pooled by `index_add_`.
* `ablate` flag (mechanism probes only) removes the additive term; default False.
* Parameter audit is hard-asserted at build: total **297,499**; body 184,667; bridge 82,944;
  local 29,888.

## 4. IHT encoder

For tuple matrix `X ∈ R^{T×125}` and `D = D_loc`:

1. `D_bar = D / max(||D_col||, eps)`.
2. `eta = 1/(1.05 · λ_max(D_bar^T D_bar))`, `λ_max` by deterministic power iteration
   (`tccd_v0.power_iter_sigma`, frozen start vector).
3. `a_0 = D_bar^T X`; repeat `IHT_STEPS = 10`: `a ← hard_threshold_s(a + eta·(X − a D^T) D)`
   where `hard_threshold_s` keeps the top-`s` coordinates by magnitude (mask, gradient flows).
4. Return `alpha = a` (T×64), detached statistics.

The sparsity mask and step count are fixed; no per-step or per-arm tuning.

## 5. Training

* Identical to the paired g round: Adam, 240 epochs, batch 32, `zw.LR/WEIGHT_DECAY/GRAD_CLIP`,
  soup epochs `zw.SOUP_EPOCHS`, schedule seed `SEED + TRAIN_SHUFFLE_OFFSET`, data-stream hash logged.
* Arms are seeded identically and consume the same dropout draws (verified in smoke);
  they differ only through `weight_key ∈ {joint, independent}`.
* Soup = mean of epoch-end states at `SOUP_EPOCHS` (raw soup, no search); the soup state is the
  analysis object, as for B.
* Per-epoch curve and epoch-1/40/120/240 probe (`W_loc`/`D_loc` grad norms, alpha stats,
  code RMS, dead dims) are logged for mechanism health.
* No early stopping, no dev-based model selection (dev is analysis-only; the round is a
  pre-registered single comparison).

## 6. Outputs (per arm, in `results/<slug>/`)

`{arm}_raw_soup_state.pt`, `{arm}_last_state.pt`, `{arm}_init_state.pt`, `{arm}_train.json`,
`{arm}_meta.json`, plus analysis artifacts: `predictions.npz` (fit/dev raw), `bootstrap.json`,
`gates.json`, `gate.json`, `contributions.csv`, `mechanism.json`, `replay.json`,
`interventions.json`, `figures/*.png`, `budget.json`, `manifest.json`,
`EXECUTION.md`, `REPORT.md`, `DECISION.md`.

## 7. Reproduction contract

* Rerunning `--phase-a/--smoke` on the same commit must reproduce the stored JSONs (hash-compared).
* `--replay` reloads the soup state on a fixed dev subset and must reproduce predictions to `1e-5`.
* `--manifest` records sha256 of every input and output; all gates read only files listed there.
* `--analyze` refuses to run if any arm artifact is missing or if contract errors are present
  (then yields `INVALID` for the primary gate, but still writes the report).
