# PROTOCOL — LCRC Lipophilicity bond-object pairing pilot v1

- **Protocol id:** `lcrc-lipo-bond-object-pairing-v1`
- **Round:** `tracks/structural-sufficiency/lcrc_lipo_bond_object_pairing_v1/`
- **Config:** `configs/protocol.json`
- **Result label:** `EXPOSED_MONITOR_EXPLORATORY_ONLY`
- **Status at freeze:** `FROZEN_PRE_RUN`

## 1. Question and scope

Hold the LCRC-A atom 1/2-hop encoder and the original readout fixed. Add one
trainable bond-object residual that uses the real physical-bond pairing of two
atom local representations. Compare it with an identically parameterised
mean-field null that keeps the same per-(molecule, bond-type) stub multiset but
destroys which two atoms are actually bonded.

This is a three-seed exploratory pilot on the already-exposed Lipophilicity
Monitor504 set. It is **not** an independent blind test, not a SOTA attempt,
and not an architecture search. Reserved504 and the official validation/test
sets are never read, scored or used for structure.

## 2. Data boundary

| role | n | use |
|---|---:|---|
| Fit | 2352 | the only supervised training data; Fit-only target standardisation |
| Monitor | 504 | label-blind prediction, then one frozen evaluation |
| Reserved | 504 | never accessed |
| official valid/test | — | never accessed |

Frozen inputs (verified by SHA-256 and row hash at runtime):

- `tracks/structural-sufficiency/results/phase1a/split_phase1a.json`
- `tracks/structural-sufficiency/results/phase1a/labels_fit_monitor.json`
- `lcrc_cross_task_preparation_v1/analysis/feature_spec_lipo.json`
- `lcrc_cross_task_preparation_v1/analysis/norm_lipo.json`
- `lcrc_cross_task_preparation_v1/cache/lipo_fit.pt`
- `lcrc_cross_task_preparation_v1/cache/lipo_monitor.pt`
- `lcrc_lipo_screen_v1/analysis/norm_target.json`
- `lcrc_lipo_screen_v1/analysis/predictions/A_s{0,1,2}.npz` (read-only baseline)

If a cache is absent in this worktree it is rebuilt deterministically into this
round's `cache/` directory and must reproduce the frozen content fingerprint.
Historical caches are never written. The `FitBatchPlan` mapping is reused:
`shuffle_pos -> fit_cache_pos -> global_row_id` where the global row id is only
a provenance/label key and is never used as a cache position.

Target standardisation is the historical Fit-only mean `2.157857142857143` and
population std `1.2241982660328445`. All training, `y_base` and the new `delta`
are computed in that standardised space.

## 3. Model

The original LCRC-A trunk is built from the frozen `lcrc_model.LCRC` source and
is fully trainable. It produces `a_v` (96), `b_e` (96), `h_v` (64) and
`y_base = contrib + rho`.

New atom context: `z_v = silu(Linear(64, 32)(h_v))` (2,080 parameters).

For every physical bond `e=(u,v)`, once:

- **R (REAL_PAIRING):** `j_e = z_u * z_v` elementwise.
- **N (TYPE_CONDITIONED_MEANFIELD_NULL):** group all bonds by
  `(molecule, bond_type)`. Each bond contributes two stubs. For a group with
  `M` bonds and `S=2M` stubs,

  ```text
  sum_z  = Σ_stubs z
  sum_z2 = Σ_stubs z*z
  j_null = (sum_z*sum_z - sum_z2) / (S*(S-1))
  ```

  Every bond in the group uses the same `j_null`. This is the exact mean of
  `z_i * z_j` over all unordered pairs of distinct stubs drawn without
  replacement. It preserves the per-molecule/per-type stub multiset, each
  atom's bond-type incidence, bond counts, `b_e`, and all model parameters; it
  does not preserve which two atom contexts were actually bonded.

Shared bond-object network:

```text
edge_input = concat([b_e, j_e], dim=-1)          # 128
h_e        = silu(Linear(128, 64)(edge_input))  # 8,256
h_e        = silu(Linear(64, 32)(h_e))           # 2,080
score_e    = Linear(32, 1)(h_e)                  # 33, zero-initialised
delta_g    = Σ_{e in g} score_e / sqrt(max(1, n_bonds_g))
y_new_g    = y_base_g + delta_g
```

Total new parameters: **12,449**. Total expected: **259,555** for both R and N.
No hop enumeration, message passing, attention, junction token or extra
chemical descriptor is added.

## 4. Initialisation and pairing

Seeds `0,1,2`. For each seed: `torch.manual_seed(seed)`; build the original
LCRC-A trunk; verify its state hash against the historical A init hash; build
the new modules; zero the final `Linear(32,1)` weight and bias. R and N use
exactly the same initial state for a given seed. The historical A predictions
are reused read-only; A is not retrained.

## 5. Correctness gates

G1 data/permission, G2 trunk preservation, G3 bond-object invariance, G4
mean-field formula, G5 relation sensitivity, G6 optimisation path, G7
workspace/GPU isolation. Evidence is written to `analysis/gates.json`,
`analysis/resource_isolation.json`, `analysis/data_audit.json` and
`analysis/parameter_audit.json`. Formal training refuses unless every gate is
PASS.

## 6. Training recipe

Identical to the historical `lcrc_lipo_screen_v1` A run: Fit2352 only; scratch;
no SSL; AdamW; peak LR `1e-3`; min LR `1e-5`; per-step cosine; weight decay
`1e-5`; batch 128; 360 epochs; 19 updates/epoch; 6,840 updates/run; grad clip
5.0; dropout 0; MSE on the Fit-standardised target; soup = element-wise mean of
the last 5 epoch-end state dicts. No early stopping and no Monitor-based
selection. The six runs `R_s0, N_s0, R_s1, N_s1, R_s2, N_s2` all use the same
execution backend.

## 7. Prediction freeze and evaluation

After all six runs: label-blind Monitor inference, save six per-molecule
predictions plus `delta`, bond-count diagnostics, and the frozen-weight R->N
counterfactual for each R seed. Hash the six R/N predictions and the three
historical A predictions into `analysis/prediction_manifest.json` and lock it
with `.sha256`. Only then may Monitor504 labels be read, exactly once.

Primary metric: raw-target RMSE. Secondary: raw MAE. Report A/R/N per seed,
seed means and spreads, R-A, N-A, R-N per seed and mean, direction counts, and
paired molecule bootstrap intervals (2,000 resamples, seed 0). Seed uncertainty
and molecule-resampling uncertainty are reported separately and never combined
as if they were independent samples.

Materiality threshold: **0.03 raw RMSE**. Decision rules (three-seed mean RMSE):

1. `BOTH_SIGNALS`: R-A ≤ -0.03, R-N ≤ -0.03, at least 2/3 seeds with R better
   than both A and N, and R mean MAE not worse than A by more than 0.01.
2. `RELATION_SIGNAL_NOT_COMPETITIVE`: not 1, but R-N ≤ -0.03.
3. `BOND_BRANCH_GAIN_NOT_ATTRIBUTABLE_TO_PAIRING`: not 1/2, but R-A ≤ -0.03.
4. `NO_CLEAR_VALIDATION_SIGNAL`: none of the above.
5. `INCOMPLETE`: any run/data/manifest/resource failure.

The R->N frozen-weight swap is a mechanism diagnostic, not a causal estimate of
pairing benefit. Bond-count groups and top-error removals are descriptive and
do not change the decision.

## 8. Limitations

Monitor504 is a development split already used by SSRC Phase 1a. Three seeds do
not turn this into a blind or population-level study. Reserved and official
valid/test data remain untouched. No result from this round authorises a second
round, a new architecture, or a hyperparameter search.
