# Mentor artifact typed-patch + K-SVD MolHIV reproduction

Date: 2026-09-06

## Conclusion

The recovered `artifacts/` do document a MolHIV route built around

```text
composition[69] + recon_typed[624] + context_mass[5*K+5]
```

with `K=64` and sparse coding level `T=8`. The missing upstream payload and helper scripts prevent a byte-for-byte reproduction, so this work implements an auditable, artifact-aligned proxy rather than claiming an exact recovery.

On the official MolHIV split, using only `seed=0` for the newly requested frozen test evaluation:

| frozen view | meaning | valid ROC-AUC | test, train only | test, train+valid refit |
|---|---|---:|---:|---:|
| `st_raw` | 69-D composition + 624-D pooled raw typed patches | **0.833662** | 0.783351 | 0.776451 |
| `st_final` | 69-D composition + 624-D pooled K-SVD reconstructions | 0.810710 | **0.785434** | **0.785766** |
| `sta_final` | `st_final` + 325-D K-SVD atom-use/ring-context mass | 0.813037 | 0.809639 | **0.815000** |
| `sta_cross_cov` | `sta_final` + 320-D atom-use/context covariance | **0.817936** | 0.804504 | 0.803445 |

Adding the A/context block produces a result above the mentor's reported `0.80` threshold in this controlled seed-0 evaluation. The additional covariance block raises the official-valid score in this one run, but does not improve the held-out test. Therefore:

- strict valid-selected refit result across the predeclared `st_raw`, `st_final`, and `sta_final` views: **0.776451 test ROC-AUC** (`st_raw`);
- best observed result among the predeclared candidates: **0.815000** from `sta_final` after train+valid refit;
- if `sta_cross_cov` is included in view selection, it wins official valid (`0.817936`) but reaches only `0.803445` after train+valid refit on test;
- the A-augmented proxy therefore reaches `>0.80`, but this does not prove exact mentor-code reproduction.

The repository had already exposed the MolHIV official test in older routes. This evaluation is consequently a controlled frozen evaluation, not a claim of an untouched test set.

## Exact-reproduction boundary

## What Is Confirmed vs Inferred

The following pieces are directly visible in the recovered artifacts:

- the `composition[69]` block;
- a structural K-SVD route using the `K=64`, sparsity-`8` convention;
- decoding original and reconstructed structural patches into a `208`-D typed descriptor;
- mean/std/max pooling into a `624`-D typed graph feature;
- explicit ring-context features, including the six-context `6*K+6=390` form and the no-`ring_any` `5*K+5=325` form;
- view assembly that combines composition, typed reconstruction, and context blocks.

The following pieces are not recoverable exactly from the supplied artifacts and are therefore implementation choices in this repository:

- the exact 208-D coordinate schema and ordering;
- the original structural patch-row construction and normalization;
- exact K-SVD initialization, update, and OMP conventions;
- exact ring-cycle implementation and context pooling details;
- exact downstream classifier/reference hyperparameters.

So the current implementation is dimension/mechanism-aligned and leakage-audited. It is not justified to call it a byte-for-byte or line-for-line reproduction of the mentor's code.

There is one important implementation-level difference that should not be
hidden behind the shared dimensions. The visible artifact source describes:

```text
graph -> fixed-coordinate structural patch x
      -> K-SVD / OMP on x
      -> reconstructed structural patch D*c
      -> decode with raw shell/mask layout metadata
      -> 208-D typed descriptor -> mean/std/max pool
```

This proxy instead uses:

```text
graph -> invariant 208-D typed descriptor
      -> K-SVD / OMP directly on that descriptor
      -> reconstructed 208-D descriptor -> mean/std/max pool
```

The first route is visible in `artifacts/molhiv_online_structural_ksvd_full.py`
and `artifacts/run_molhiv_ksvd_three_diagnostics.py`; the second is the
deliberate fallback implemented here because the original structural payload,
capacity/cache, and classifier inputs are absent. This difference is material
and is the main reason the current result must be reported as a proxy.

The artifact source itself expects files that are absent from the supplied workspace:

- the original `composition[69]` and `recon_typed[624]` feature payloads;
- `results/molhiv_ksvd_three_diagnostics_K64_s8/`;
- `results/molhiv_r2_atom_ring_context_K64_s8/`;
- `results/molhiv_best/summary.json`, containing the reference classifier parameters;
- `run_molhiv_shared_atom_classification_old_protocol.py`;
- `run_molhiv_explicit_ring_oracle.py`;
- the optional historical `components_molhiv_direct_typed_patch_pool...npz` cache.

`artifacts/run_molhiv_auc_objective_study.py` says that its original `nora` XGBoost control should reproduce approximately `0.8059` test ROC-AUC, but the required payload, ring builder, classifier, and reference summary are not present. That number can therefore be treated only as recovered provenance, not as a result reproduced here.

## Implemented artifact-aligned proxy

Implementation:

- `tracks/ksvd/experiments/luyin16/mentor_artifact_typed_ksvd.py`
- `tracks/ksvd/experiments/luyin16/mentor_artifact_optuna.py`
- `tracks/ksvd/configs/luyin16/mentor_artifact_typed_ksvd_k64_s8.yaml`
- `tracks/ksvd/tests/test_mentor_artifact_typed_ksvd.py`

The proxy preserves the dimensions and mechanism visible in the artifacts:

1. Every atom is used as the centre of a complete radius-2 induced ego patch.
2. Each patch has a permutation-invariant 208-D typed descriptor:
   - root, shell-1, and shell-2 atom semantics: `3 * 48 = 144`;
   - four shell-pair bond summaries: `4 * 13 = 52`;
   - topology statistics: `10`;
   - shell-size statistics: `2`.
3. Mean/std/max pooling gives `3 * 208 = 624` graph-level coordinates.
4. The graph composition block is `48` atom + `13` bond + `8` graph statistics = `69` coordinates.
5. A shared train-only `K=64`, `T=8` K-SVD dictionary reconstructs the proxy's
   invariant typed descriptors (the mentor artifact factorizes a lower-level
   fixed-coordinate structural row instead).
6. Five ring contexts (`ring5`, `ring6`, `aromatic_ring`, `multi_ring`, and `ring_boundary`) give `5*64+5 = 325` activation-mass coordinates, matching the artifact's `nora` layout.

For the A-augmented view, the feature blocks are:

```text
S: composition                         69D
T: final K-SVD typed pool             624D
A: five-context atom-use mass         325D
-------------------------------------------
sta_final                             1018D
```

The A block uses the normalized absolute OMP atom-use distribution for each
root patch, sums atom mass inside each context, and appends context coverage.
It is a ring-conditioned sparse-code summary, not a new neural attention module.

The exploratory `sta_cross_cov` view appends a `64*5=320`-D population
covariance between normalized atom-use mass and the five context indicators:

```text
sta_cross_cov = S + T_final + A + cross_cov_final  (1338D)
```

This is an additional exploratory statistic, not part of the recovered
mentor layout. It was searched with the same 12-trial, single-seed protocol.

Compared with the older 52-D proxy, this implementation removes three known correctness problems: no radius-2 patch is truncated to eight nodes, no node-ID tie break enters a coordinate, and categorical atom values are not reduced modulo a smaller vocabulary.

## Full feature-build audit

| item | value |
|---|---:|
| graphs | 41,127 |
| official train / valid / test | 32,901 / 4,113 / 4,113 |
| atom-centred radius-2 patches | 1,049,163 |
| sampled dictionary-training patches | 24,000, official train only |
| patch dimension | 208 |
| dictionary atoms / sparsity | 64 / 8 |
| K-SVD update iterations | 5 |
| feature-build runtime | 276.75 s |

K-SVD reconstruction:

| scope | INIT relative error | FINAL relative error |
|---|---:|---:|
| 24,000 training patches | 0.272338 | 0.214749 |
| all encoded patches | 0.270064 | 0.215234 |

The manifest records that coordinate scaling and dictionary fitting use official-train patches only, and that valid/test labels are not used to construct the representation.

## Optuna search

The objective used three official-train-only Bemis-Murcko scaffold folds, XGBoost early stopping, at most 1,000 trees, and a persistent SQLite Optuna study. The main two candidates each completed 12 trials:

| view | dimension | trials | best train-only CV | seed=0 official valid | frozen trees |
|---|---:|---:|---:|---:|---:|
| `st_raw` | 693 | 12 | 0.792775 | **0.833662** | 73 |
| `st_final` | 693 | 12 | **0.803920** | 0.810710 | 301 |

Additional feature combinations received one queued/reference trial rather than a full search:

| view | blocks | train-only CV | seed=0 valid |
|---|---|---:|---:|
| `t_raw` | raw typed pool | 0.782300 | 0.806721 |
| `t_final` | K-SVD typed pool | 0.793925 | 0.815851 |
| `sta_init` | composition + INIT reconstruction + context | 0.797236 | 0.815407 |
| `sta_final` | composition + FINAL reconstruction + context | 0.804862 | 0.814115 |
| `st_raw_final` | composition + raw + FINAL reconstruction | 0.800918 | 0.812704 |
| `sta_raw_final` | composition + raw + FINAL reconstruction + context | **0.806996** | **0.818261** |

The completed search predates the request to stop running five model seeds, so `optuna_search.json` retains its earlier five-seed valid audit. No new five-seed job was run. The frozen official-test continuation used only `seed=0`, and the tables above use seed 0 wherever a seed-dependent score is shown.

The A-augmented search was run separately with one model seed as requested. It
used the same three official-train-only Bemis-Murcko scaffold folds, search
space, 12 trials, and early-stopping protocol. Its best trial was trial 7:

| view | dimension | trials | best train-only CV | seed=0 official valid | frozen trees |
|---|---:|---:|---:|---:|---:|
| `sta_final` | 1018 | 12 | **0.806822** | **0.813037** | 265 |

The corresponding frozen seed-0 test scores are `0.809639` when fitting on
official train only and **`0.815000`** after refitting on official train+valid.

The earlier `sta_final` row in the additional-view table is the old queued
reference trial (`0.804862` CV, `0.814115` valid); it is retained for provenance
and should not be confused with this new 12-trial A search.

The follow-up covariance search included both `sta_final` and `sta_cross_cov`
under the same single-seed protocol:

| view | dimension | trials | best train-only CV | seed=0 official valid | frozen trees |
|---|---:|---:|---:|---:|---:|
| `sta_final` | 1018 | 12 | 0.806822 | 0.813037 | 265 |
| `sta_cross_cov` | 1338 | 12 | 0.804428 | **0.817936** | 358 |

The frozen test continuation is:

| view | test, train only | test, train+valid refit |
|---|---:|---:|
| `sta_final` | **0.809639** | **0.815000** |
| `sta_cross_cov` | 0.804504 | 0.803445 |

Consequently `cross_cov` is a validation-set improvement in this seed-0 run,
not evidence of a robust test improvement.

## Interpretation

The experiment reproduces the visible representation sizes, train-only dictionary discipline, sparse coding, reconstructed typed pooling, and five-context mass block. It does not establish that this inferred 208-D schema is the mentor's hidden schema.

The current result suggests three separate issues:

1. Adding A is useful in this run: relative to `st_final`, train+valid refit test AUC rises from `0.785766` to `0.815000`.
2. The recovered dimensions alone are insufficient to establish exact mentor reproduction. The missing exact typed-patch schema, payload, ring implementation, and reference XGBoost configuration remain material.
3. There is substantial scaffold-shift/selection instability: `st_raw` scores 0.833662 on valid but falls to 0.776451 after train+valid refit on test. MolHIV has only 81 positive valid examples, so a single high validation score is not strong evidence of generalization.

The K-SVD reconstruction itself is functioning: reconstruction error falls materially after five updates. However, the downstream result does not justify claiming that the reconstructed representation recovers the mentor model.

## Reproduction commands

Build the full features:

```bash
env UV_CACHE_DIR=/tmp/codex-uv-cache uv run --offline --no-sync python \
  -m tracks.ksvd.experiments.luyin16.mentor_artifact_typed_ksvd \
  --config tracks/ksvd/configs/luyin16/mentor_artifact_typed_ksvd_k64_s8.yaml \
  --result-dir tracks/ksvd/results/luyin16/mentor_artifact_typed_ksvd_k64_s8
```

For a fresh search that follows the new single-seed preference:

```bash
env UV_CACHE_DIR=/tmp/codex-uv-cache uv run --offline --no-sync python \
  -m tracks.ksvd.experiments.luyin16.mentor_artifact_optuna tune \
  --features tracks/ksvd/results/luyin16/mentor_artifact_typed_ksvd_k64_s8/features_full.npz \
  --result tracks/ksvd/results/luyin16/mentor_artifact_typed_ksvd_k64_s8/optuna_search_seed0.json \
  --storage tracks/ksvd/results/luyin16/mentor_artifact_typed_ksvd_k64_s8/optuna_seed0.sqlite3 \
  --folds-file tracks/ksvd/results/molhiv/molhiv_n41127_scaffold_folds3_seed20260726.npz \
  --views st_raw,st_final --trials 12 --seeds 0 --n-jobs -1
```

The frozen test command actually used in this continuation:

```bash
env UV_CACHE_DIR=/tmp/codex-uv-cache uv run --offline --no-sync python \
  -m tracks.ksvd.experiments.luyin16.mentor_artifact_optuna test \
  --features tracks/ksvd/results/luyin16/mentor_artifact_typed_ksvd_k64_s8/features_full.npz \
  --tuning tracks/ksvd/results/luyin16/mentor_artifact_typed_ksvd_k64_s8/optuna_search.json \
  --result tracks/ksvd/results/luyin16/mentor_artifact_typed_ksvd_k64_s8/frozen_test_seed0.json \
  --views st_raw,st_final --seeds 0 --n-jobs -1
```

The A-only single-seed search:

```bash
env UV_CACHE_DIR=/tmp/codex-uv-cache uv run --offline --no-sync python \
  -m tracks.ksvd.experiments.luyin16.mentor_artifact_optuna tune \
  --features tracks/ksvd/results/luyin16/mentor_artifact_typed_ksvd_k64_s8/features_full.npz \
  --result tracks/ksvd/results/luyin16/mentor_artifact_typed_ksvd_k64_s8/optuna_search_a_seed0.json \
  --storage tracks/ksvd/results/luyin16/mentor_artifact_typed_ksvd_k64_s8/optuna_a_seed0.sqlite3 \
  --folds-file tracks/ksvd/results/molhiv/molhiv_n41127_scaffold_folds3_seed20260726.npz \
  --views sta_final --trials 12 --seeds 0 --n-jobs -1 \
  --study-prefix mentor_artifact_typed208_k64_s8_seed0_a \
  --protocol-id luyin16-mentor-artifact-typed-ksvd-optuna-a-seed0-v1
```

The corresponding frozen A test:

```bash
env UV_CACHE_DIR=/tmp/codex-uv-cache uv run --offline --no-sync python \
  -m tracks.ksvd.experiments.luyin16.mentor_artifact_optuna test \
  --features tracks/ksvd/results/luyin16/mentor_artifact_typed_ksvd_k64_s8/features_full.npz \
  --tuning tracks/ksvd/results/luyin16/mentor_artifact_typed_ksvd_k64_s8/optuna_search_a_seed0.json \
  --result tracks/ksvd/results/luyin16/mentor_artifact_typed_ksvd_k64_s8/frozen_test_a_seed0.json \
  --views sta_final --seeds 0 --n-jobs -1 \
  --protocol-id luyin16-mentor-artifact-typed-ksvd-frozen-test-a-seed0-v1
```

The follow-up covariance search and frozen test were:

```bash
env UV_CACHE_DIR=/tmp/codex-uv-cache uv run --offline --no-sync python \
  -m tracks.ksvd.experiments.luyin16.mentor_artifact_optuna tune \
  --features tracks/ksvd/results/luyin16/mentor_artifact_typed_ksvd_k64_s8/features_full.npz \
  --result tracks/ksvd/results/luyin16/mentor_artifact_typed_ksvd_k64_s8/optuna_search_cross_seed0.json \
  --storage tracks/ksvd/results/luyin16/mentor_artifact_typed_ksvd_k64_s8/optuna_a_seed0.sqlite3 \
  --folds-file tracks/ksvd/results/molhiv/molhiv_n41127_scaffold_folds3_seed20260726.npz \
  --views sta_final,sta_cross_cov --trials 12 --seeds 0 --n-jobs -1 \
  --study-prefix mentor_artifact_typed208_k64_s8_seed0_a \
  --protocol-id luyin16-mentor-artifact-typed-ksvd-optuna-cross-seed0-v1
```

```bash
env UV_CACHE_DIR=/tmp/codex-uv-cache uv run --offline --no-sync python \
  -m tracks.ksvd.experiments.luyin16.mentor_artifact_optuna test \
  --features tracks/ksvd/results/luyin16/mentor_artifact_typed_ksvd_k64_s8/features_full.npz \
  --tuning tracks/ksvd/results/luyin16/mentor_artifact_typed_ksvd_k64_s8/optuna_search_cross_seed0.json \
  --result tracks/ksvd/results/luyin16/mentor_artifact_typed_ksvd_k64_s8/frozen_test_cross_seed0.json \
  --views sta_final,sta_cross_cov --seeds 0 --n-jobs -1 \
  --protocol-id luyin16-mentor-artifact-typed-ksvd-frozen-test-cross-seed0-v1
```

## Outputs and checksums

| file | SHA-256 |
|---|---|
| `features_full.npz` | `102981bf591e64bcd518a5f13f0723ed6d2e7b588643385f8a2ee27ef0ed4c3d` |
| `manifest.json` | `da8b6b2eaa558e29bd2c5afd308bf0a03ca96e2c027078a2e18e10e661fbd431` |
| `optuna_search.json` | `e032fbb1fb79084fe7bf861b891531e5cad97ae4241d9b416cbf04b81b160cfc` |
| `frozen_test_seed0.json` | `0d0eb090964a12e7d0412840eefd1de541c7a54ff1e4bf55fb4b0a8d1ec2f55a` |
| `optuna_search_a_seed0.json` | `5647150c5c3e87dc751d209c86ae49d3814535f22c2f4ad3ff371da15e8bcd9b` |
| `frozen_test_a_seed0.json` | `6d939e185593e4bf62209628e7dad0272596017ca3cad8cd5d281ab7301aef3a` |
| `optuna_search_cross_seed0.json` | `97e5719ba149e9ce4fd1c844f817163c98db2e716d0a2a4115836164789a5100` |
| `frozen_test_cross_seed0.json` | `797bf03c1673bd7e87eb9868d4e4eecedbc3f4c8a227c310820af9bff7b8c1d8` |

The test predictions are saved in `frozen_test_seed0.predictions.npz`; Optuna state remains resumable in `optuna.sqlite3`.

## Verification

- Ruff passed for both new experiment modules and their dedicated test module.
- `11` focused pytest cases passed (`test_mentor_artifact_typed_ksvd.py` and `test_role_attribute_binding_screen.py`).
- The recovered structural K-SVD, three-diagnostics, and atom-ring-context artifact self-tests all passed.
- Python compilation passed for the new experiment and test modules.
- All expected feature blocks have the recorded full-dataset shapes and finite values.
- The original frozen test result has four prediction arrays of shape `[1, 4113]`; the A-only frozen test has one such array, confirming that no new multi-seed test job was run.
- The covariance frozen test has four prediction arrays of shape `[1, 4113]`; both covariance outputs are single-seed evaluations.
