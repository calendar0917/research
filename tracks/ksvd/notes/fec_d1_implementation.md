# FEC-D1 — implementation

Round: **FEC-D1** (*Localized Sparse Structural Dictionary Binding*).
Protocol `fec_d1`. Pre-registration: [`fec_d1_preregistration.md`](fec_d1_preregistration.md).
Prior-artifact audit: [`fec_d1_prior_artifact_audit.md`](fec_d1_prior_artifact_audit.md).

## 1. Files

| role | path |
|---|---|
| core primitives | `tracks/ksvd/experiments/luyin16/fec_d1.py` |
| runner | `tracks/ksvd/experiments/luyin16/zinc_fec_d1.py` |
| focused CPU tests | `tracks/ksvd/tests/test_fec_d1.py` (11 pass) |
| results | `tracks/ksvd/results/fec_d1/` |

No shared file is modified. `zinc_patch_path_pooling.py`,
`zinc_static_dictionary_pair.py`, `sdb_v0.py`, `fsar_r2_ar0.py`,
`fec_s1_shared_local_env.py` are imported and called read-only.

## 2. Frozen artifacts read (never refit)

* `results/sdb_v0/dictionary.pt` → `D_SDB (65×32)`, `pca_mean`, `pca_components (32×65)`.
* `results/fsar_r2_ar0/cache/{train,valid}.pkl.gz` → `phi_v^65`, `atom_idx`.
* `results/zinc_static_dictionary_pair/cache/encoded_{train,valid}.pt` →
  FEC-S1 patch order / `patch_cont`.
* `results/fec_s1/states/fec_s1_seed0_selection_state.pt` → frozen FEC-S1 best
  checkpoint. **Compatibility note:** the FEC-S1 checkpoint keys are
  `local_env_adapter.net.*`; the branch wrapper adds a `base.` prefix, so the
  checkpoint is loaded into the unwrapped FEC-S1 model *before* wrapping
  (`load_fec_d1_best`).

Sparse coding calls `sdb_v0.omp_codes` → `tccd_v0.omp_codes` (exact top-`s`).
`K = 32`, `s = 8`, `DICT_SEED = 20260924`, normalization and coding semantics
are untouched.

## 3. Localized binding statistic (`fec_d1.per_root_binding`)

For one molecule: `phi[n,65]`, `atom_idx[n]`, untyped `graph`, and one 32-D
structural coordinate per node (`alpha` for Dict, `z = PCA32(phi)` for PCA):

```
for root i in 0..n-1:
    shells = BFS distances from i, radius 2     # ego_shells
    for shell k in {0,1,2}:
        S = shells[k];  |S| < 2  =>  C_ik = 0
        C_ik = Σ_{v∈S} (a_v - ā)(q_v - q̄)^T ∈ R^{32×28}
C_i = [ C_i0 ; C_i1 ; C_i2 ] ∈ R^{2688}         # shell-major flatten
```

`q_v = one_hot(atom_v)`. The root/patch order equals `graph.nodes` (0..n-1),
which was verified equal to the FSAR/encoded patch order (probe: raw
`_data_to_graph` ↔ FSAR `phi` bit-identical, `atom_idx` identical, `y`
identical; `encoded[m].num_nodes == FSAR n_nodes == graph node count`).

`per_root_binding(..., shuffle_seed=...)` performs the evaluation-only
within-root/shell permutation of the `alpha ↔ q` correspondence
(`np.random.default_rng(seed).permutation` consumed in a fixed root/shell
order), keeping both multisets.

## 4. Train-only scaling

Two streaming passes per arm: pass 1 accumulates
`Σ C_j^2` over official **train** only; `mean_sq = sumsq / n_patches`;
`mask = 1[sqrt(mean_sq) > 1e-9]`; `scale = sqrt(mean_sq + 1e-12)` where
masked, else `1`. Pass 2 writes `(C / scale) * mask` float32 into a
preallocated `[N, 2688]` buffer. No mean subtraction. Valid is never used to
fit. Effective/zero coordinates are recorded in `binding_scalers.json`.

Observed (local build): Dict arm 1254 / 2688 effective coordinates (46.7 %),
raw RMS range `[1.9e-06, 1.436]`; PCA arm 1344 / 2688 effective (50.0 %), raw
RMS range `[6.1e-08, 0.471]`. Cache build ≈ 64 s (Dict) / 20 s (PCA) on CPU.

## 5. Rank-8 branch (`fec_d1.BindingLocalEnvAdapter`)

```
e_new = base_adapter(patch_cont) + (C~ @ W1^T) @ W2^T
W1 : (8, 2688)   kaiming_uniform_(a=√5), torch.Generator().manual_seed(0)
W2 : (24, 8)     exact zeros
```

No bias, no nonlinearity. `21696` trainable parameters per arm, identical for
Dict and PCA. The statistic is installed with `set_stat(C~)` immediately before
each forward; `C~` is attached per `Data` as `d1_stat [n_patches, 2688]` and
PyG collation concatenates it in patch order, so it aligns with
`data.patch_cont` exactly.

`freeze_base_train_branch` sets `requires_grad=False` on all `66170` FEC-S1
parameters and `True` only on `local_env_adapter.W1/W2`; it re-checks that the
trainable set is exactly those two tensors.

## 6. Injection point

`zinc_patch_path_pooling.PatchPathModel.encode` calls
`self.local_env_adapter(data.patch_cont)` and uses the 24-D output as
`[e_patch(16); parent(8)]`. The wrapper adds `Δe` to that 24-D output; every
downstream computation (patch encoder, pair system, reader) is the frozen
FEC-S1 path. There is no graph residual, no pair-kernel or centre path, no
recurrence, no context writeback. G6 (`_g6_no_bypass`) confirms that with
`W2 = 0` the statistic has **zero** effect on the prediction and that a nonzero
`W2` changes the prediction only through the adapter.

## 7. Hard gates (`correctness_stage`, all pass in 22 s on local CPU)

* **G0** `phi65` bit-identical between a fresh `build_phi(graph)` (float32) and
  the FSAR cache; `alpha` deterministic; `max l0 = 8`.
* **G1** chemistry relabelling on fixed topology leaves `phi`/`alpha` unchanged.
* **G2** `alpha_v` is node-centric (identical when recoded per node); the same
  shared node changes shell across roots.
* **G3** `C_ik` matches an independent float64 loop (max abs diff `9.9e-09`).
* **G4** within-root/shell assignment shuffle changes the statistic.
* **G5** `W2 = 0` ⇒ base adapter output and prediction bit-identical to FEC-S1;
  `W2` exactly zero.
* **G6** no readout bypass (see §6).
* **G7** `W2` grad `0.00897` at step 0; `W1` grad `0.0` at step 0 and
  `0.00111` after one optimizer step.
* **G8** `static_contract_checks` passes; the pre-pair local 24-D output is
  bit-identical under `pair_relation` mutation.
* **G9** official-test blocker: `_load_zinc(..., "test")` raises; the encoded
  cache records `official_test_loaded = false`.

## 8. Training protocol

Inherited from FEC-S1: Adam, `lr=1e-3`, `weight_decay=1e-5`, batch 128, L1/MAE,
grad clip 5, no scheduler, `max_epochs=240`, fixed Top-5 soup over branch
checkpoints, seed 0. Only `W1/W2` are optimized (frozen base). Loader shuffle
seed offsets `91011/91012` are inherited from `OPTIMIZED_PROTOCOL`, so Dict and
PCA see the same batch order.

**Recorded deviation:** pre-registration §11 says *no early termination*.
`zinc_fec_d1.PATIENCE = MAX_EPOCHS = 240`, so the run always completes 240
epochs; `early_stopped` will be `false`. This only disables an early stop that
FEC-S1's `patience=40` could in principle trigger; it does not change the data,
loss, optimizer or checkpoint selection.

Both arms are trained from the *same* frozen FEC-S1 best checkpoint with the
same branch init seed; the branch state is the only thing that changes. The
matched baseline `M_B` is the read-only replay of that checkpoint
(`baseline_guard`, tolerance `1e-5`); historical soup `0.130422` is external
context only.

## 9. Mechanism (evaluation only)

* **Branch neutralization**: `W2 ← 0` on the trained Dict soup; the prediction
  must return to the frozen base (`neutralization_max_abs_pred_shift ≤ 1e-6`).
* **Assignment shuffle**: five fixed within-root/shell permutations
  (`SHUFFLE_SEEDS = (101, 202, 303, 404, 505)`), re-normalized with the same
  train-only Dict scaler, evaluated with the trained Dict soup. The FEC-S1 base
  input is never modified.

## 10. Verdict

`analyze_stage` computes `M_B, M_D, M_P, M_shuffle`, the three gains, the
pre-registered gates A/B/C, and the frozen verdict
(`fec_d1.VERDICTS`), then writes `decision.json`, `REPORT.md`, `DECISION.md`.

`official_test_loaded = false` everywhere.
