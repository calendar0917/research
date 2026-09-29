# REPORT — `e2e_dictenv_common_subspace_dictionary_v1`

Common-Subspace-Separated Structural Dictionary (CSSD) on the closed ZINC
dictionary-environment line (Workstream Z).  CPU only, no GPU, no SSH, no
architecture / coder / feature / relation / split change; the official ZINC
test was never loaded (`official_test_loaded = false` in every payload).

* Pre-registration:
  `notes/e2e_dictenv_common_subspace_dictionary_v1_preregistration.md`
  (frozen before Stage A; sha256 `8e722afbc757…`, implementation freeze
  commit `5f0f284`).
* Full discussion:
  `notes/e2e_dictenv_common_subspace_dictionary_v1_analysis.md`.
* Evidence: `results/e2e_dictenv_common_subspace_dictionary_v1/` —
  `preflight.json`, `preregistration_snapshot.json`, `stage_a.json`,
  `common_subspace.json`, `zero_training/{raw,mean_center,q1,q2}.csv`,
  `zero_training/{selection,comparison,probe}.json`,
  `training/{curve.csv,epoch40_gate.json,final.json,checkpoints/}`,
  `reusable_structure/{atom_profiles,graph_coverage,common_coordinates,sparse_dense_recoverability,graph_reuse,argmax_atom_hist,dictionary_geometry,reverse_recoverability}.csv`,
  `reusable_structure/{summary,extra_summary}.json`, `analysis_tables.md`,
  `summary.json`.

## A. Bottom line

1. **The common-subspace separation fires and is worth training.**  In a
   zero-training probe on all three reused C6 dictionaries (same `D`, same
   `iht10`, only the descriptor changes), the q1 residual variant satisfies
   the frozen A∧B∧C conditions on **3/3** seeds: the universal triplet
   6/24/27 drops from 1.000 to 0.03–0.40, top-5 usage drops 1.06–1.87, and
   the residual keeps 78.5 % of the centred structural variation.  q2 also
   passes A∧B and demotes atom 23 (ratio 0.10–0.38) but keeps only 34.0 % of
   the centred variation, so the frozen rule selects **q1**.
2. **The epoch-40 gate passes and the trajectory completes.**  At epoch 40
   the residual dictionary has 0 atoms above 0.95 activation (reference 3),
   top-5 2.746 vs 4.796, `N_eff` 22.73 vs 14.52, weighted Spec ratio 1.320,
   `||∂L/∂D|| = 6.6e-2`, projected column min 0.317, and
   `max |Uᵀ D̄⊥| = 4.2e-16` (hard constraint).  Verdict `CSSD_CONTINUE`;
   one trajectory to epoch 320 (39.5 min, 97 727 = 97 487 + 240 params,
   0 dead-column fallbacks).
3. **Task-neutral, representation-supported.**  Soup valid MAE
   **0.130028** vs FINAL-CLEAN sparse seed-0 soup `0.128499` →
   `Δ = +0.001529`, inside the frozen neutral band `|Δ| ≤ 0.003`
   (`CSSD_REPRESENTATION_SUPPORTED / TASK_NEUTRAL_SINGLE_SEED`).  The
   "no representation gain" clause does not fire.
4. **The dictionary is structurally healthier on every frozen metric.**
   32/32 atoms active (RAW 27), `N_eff` 21.39 vs 14.52, top-5 3.01 vs 4.80,
   max rate 0.796 vs 1.000, atoms > 0.95: 0 vs 4, coherence max 0.780 vs
   0.987, weighted Spec 0.258 vs 0.207, min graph coverage 126 graphs vs 5
   never-active atoms.
5. **The concentration was the common direction.**  The RAW universal
   triplet was exactly the set of atoms aligned with the train mean
   (`|cos(atom, u1)|` = 0.965 / 0.941 / 0.819; only 3 atoms > 0.5).  After
   separation, atom 6 becomes the *most specialized* atom (rate 1.000 →
   0.010, Spec 0.000 → 0.965) and atom 19 becomes the broad generalist
   (rate 0.796, Spec 0.080).  The CSSD dictionary contains no
   common-direction component at all (`max |cos(atom, u1)| = 8e-8`).
6. **Each graph recruits ~2.7× more atoms.**  Post-hoc descriptive
   `|α|`-mass statistics: effective atoms per graph 6.14 → **16.69** (train)
   and 6.15 → 16.66 (valid), atoms ≥ 5 % mass 4.01 → 7.67, max atom share
   0.278 → 0.133.  Honest caveat: the graph argmax remains concentrated —
   atom 6 in 100 % of RAW graphs, atom 19 in 75.5 % (atom 9 in 16.5 %) of
   CSSD graphs.
7. **`c1` is a patch-scale coordinate, not a nuisance.**  Valid Spearman:
   `patch_nodes` 0.960, `patch_edges` 0.952, root degree / shell-2 / walk-2
   0.891, `shell_pop1` 0.882.  `|cos(u1, PC1)| = 0.524` — the mean direction
   is not PC1 (PC1 = 57.7 % of centred variance).
8. **Sparse/dense equivalence is preserved, not changed.**
   dense residual → IHT-10 `R² 0.9709`, IHT-10 → dense `R² 0.9993`
   (RAW audit reference 0.970825 / 0.998845); mutual RAW↔CSSD code
   recoverability 0.9704 / 0.9813.  Separation neither destroys nor
   manufactures linear information.
9. **Claim decision.**  **Claim C refuted**, **Claim A supported on
   usage/reuse structure (one seed)**, **Claim B not excluded** — top-5 SMD
   profile diversity is unchanged (Jaccard 0.136 over 32 CSSD atoms vs 0.125
   over 27 active RAW atoms; 28 vs 29 of 65 descriptors), and the two sparse
   codes stay near-linear re-parameterisations.  The round demonstrates a
   reusability-shaped dictionary at equal accuracy, not a new information
   channel.
10. **No DenseTied full run, no seed 1/2, no q2 training.**  The historical
    DenseTied `0.125563` is context only and is **not** a matched control.

## B. Design (frozen, unchanged)

```text
FINAL-CLEAN: C6 + paired node/edge structure-semantic binding
             + sparse tied-IHT (K=32, s=8, IHT-10) + full relation
CSSD change: z = [c~ = U^T x / s ; alpha_res = IHT10(Dbar_perp, r)]
             Dbar_perp = colnorm((I - U U^T) D_raw),  U from train only
             loss     = L1(y_hat, y) + H1_LAMBDA * mean ||r - r_hat||^2 / (||x||^2 + EPS)
```

Everything else — optimizer, lr, weight decay, batch, clip, epochs 1 → 40 →
320, node/edge binding forms, decoder — is the frozen FINAL-CLEAN recipe.
`D_A = 96`; the widened rows are zero-initialised, so the common coordinate
must be learned.  One new seed-0 trajectory; existing baselines were reused
without retraining.

## C. Stages A–C

* Feature space: train 231 664 × 65, valid 23 083 × 65, float32, cache slice
  == `data.dict_phi` bit-identical, no normalization.  Nine identically-zero
  columns `[2,3,5,7,8,33,35,48,50]` retained (65-D frozen).
* Common subspace (train only): `||μ|| = 5.0662`, `|cos(u1, PC1)| = 0.524`;
  `E_common(q1)` 0.9772 / `E_centered_residual(q1)` 0.7811 train (0.9769 /
  0.7851 valid); q2: 0.9903 / 0.3336.
* Zero-training probe (valid; `analysis_tables.md` §2 has all seeds and both
  splits): RAW top-5 4.80, `N_eff` 14.5, triplet 1.000; MC 2.69, 21.1,
  0.39–0.56; Q1 2.92, 19.9, 0.03–0.13; Q2 3.21, 19.0, ~0.  Selection
  `valid` primary, `train_valid_agree = true`; q2 stored as diagnostic.

## D. Training and the epoch-40 gate

| gate item | reference (RAW seed 0) | epoch 40 | verdict |
|---|---|---|---|
| atoms 6/24/27 rate > 0.95 | 3 | **0** | PASS |
| top-5 usage | 4.796 | **2.746** (drop 2.050 ≥ 0.4) | PASS |
| `N_eff` | 14.520 | **22.725** (gain 8.205 ≥ 1.0) | PASS |
| usage-weighted Spec ratio | 1.000 | **1.320** (≥ 1.2) | PASS |
| `||∂L/∂D||` / projected column min | — | 6.65e-2 / 0.317 | PASS |
| `max |Uᵀ D̄⊥|` | — | 4.2e-16 | hard constraint |
| catastrophic (MAE / rec) | — | 0.1930 / 0.2108, rec 6.1e-5 | false |

Trajectory: best valid 0.136913 @ 307; soup members epochs 282/299/307/312/313
(individual 0.1369–0.1393), **soup 0.130028**; 97 727 params; 0 fallbacks.
`curve.csv` carries both `train_rec` (frozen full-descriptor diagnostic,
≈ 0.976 = common fraction) and `train_rec_term` (the optimised residual term,
4.9e-5 at epoch 320).

## E. Stage G — reusable-structure audit (CSSD soup vs RAW seed-0)

Usage (valid): active 32/32 vs 27, `N_eff` 21.39 vs 14.52, top-5 3.007 vs
4.796, max rate 0.796 vs 1.000, > 0.95: 0 vs 4, weighted Spec 0.258 vs 0.207,
median Spec 0.373 vs 0.564, coherence mean/max 0.206/0.780 vs 0.182/0.987.

Fate of the old universal atoms (valid rate, Spec; `|cos(RAW atom, u1)|`):
atom 6 1.000/0.000 (0.965) → **0.010/0.965**; atom 24 1.000 → 0.495/0.180
(0.941); atom 27 1.000 → 0.366/0.336 (0.819); atom 23 0.977 → 0.409/0.356
(0.324).  New rare specialists: atoms 11 / 16 (Spec 3.85 / 4.19, rates
0.001–0.011), atom 12 (4.19, 0.011).  `analysis_tables.md` §7 has all 32
atoms.

Graph reuse (post-hoc, `|α|` mass; train | valid): effective atoms per graph
6.14 | 6.15 (RAW) → 16.69 | 16.66 (CSSD); atoms ≥ 5 % mass 4.01 | 4.01 →
7.67 | 7.71; max share 0.278 | 0.278 → 0.133 | 0.133.  Argmax atom (train):
RAW `a6:1.000`; CSSD `a19:0.755, a9:0.165, a24:0.026, a16:0.023`.

Common coordinates (valid Spearman; full `common_coordinates.csv`):
`patch_nodes` 0.960, `patch_edges` 0.952, `root_induced_degree` /
`root_neighbour_shell2` / `root_walk2` 0.891, `shell_pop1` 0.882.

Recoverability (train-only OLS, valid): dense→IHT10 0.9709 (RAW 0.970825),
IHT10→dense 0.9993 (RAW 0.998845), RAW IHT10→CSSD IHT10 0.9704,
CSSD IHT10→RAW IHT10 0.9813, [c1;CSSD]→RAW 0.9815.  Vocabulary diversity
(post-hoc): mean pairwise top-5 |SMD| Jaccard 0.136 (CSSD, 32 atoms) vs 0.125
(RAW, 27 active atoms).

## F. Claim decision and task band (frozen)

```text
MAE band : |0.130028 - 0.128499| = 0.001529 <= 0.003
           -> CSSD_REPRESENTATION_SUPPORTED / TASK_NEUTRAL_SINGLE_SEED
no-gain clause: not triggered (0 universal residual atoms; top-5 3.007 vs
           4.796; weighted Spec +0.051)
Claim C  : refuted   (concentration resolved on every frozen metric)
Claim A  : supported at the usage/reuse level, single seed
Claim B  : not excluded (unchanged top-profile diversity, unchanged MAE,
           near-linear RAW<->CSSD code recoverability)
```

## G. Guardrails

* CPU only (`CUDA_VISIBLE_DEVICES=""`, `device = "cpu"` guards in the tests
  and at every stage entry).
* Official ZINC test never loaded; every payload carries
  `official_test_loaded = false`.
* Forbidden list respected: no IHT30/100/OMP training, no K/s/λ sweep, no
  coherence/usage/entropy/orthogonality penalty, no new feature/relation/
  fusion, no q3+, no width/depth/LR search, no DenseTied full run, no seed
  1/2 run, no `docs/luyin/luyin19.txt` modification.
* Frozen preregistration untouched after the first Stage-A process; the only
  post-freeze implementation commits are a runner import fix (`34fb311`) and
  a logging-only fix (`5f0f284`), both before the formal training process.
* Transparency on the single trajectory: the first launch of `stage-train`
  was killed at ~epoch 20 to apply the logging-only fix (the frozen-loop
  `train_rec` diagnostic was being reported where the optimised residual term
  belonged).  That attempt wrote **no checkpoint and no artifact** (the
  `training/` directory was emptied before the restart); the formal run is the
  restarted process, and it is the only trajectory whose artifacts exist.
* Focused tests 24/24 passing, including bit-equivalence of the training loop
  with the frozen audit loop and the `Uᵀ D̄⊥ = 0` hard constraint.

## H. Reproduce

```bash
CUDA_VISIBLE_DEVICES="" uv run pytest -q tracks/ksvd/tests/test_e2e_dictenv_common_subspace_dictionary_v1.py
CUDA_VISIBLE_DEVICES="" uv run python -m tracks.ksvd.experiments.luyin16.zinc_e2e_dictenv_common_subspace_dictionary_v1 chain --threads 4
CUDA_VISIBLE_DEVICES="" uv run python -m tracks.ksvd.experiments.luyin16.e2e_dictenv_common_subspace_dictionary_v1_extra
```

`run_chain.sh` wraps the chain.  JSON/CSV/PT artifacts are the evidence
snapshot; the checked-in copies are this file, `DECISION.md` and
`analysis_tables.md`.
