# Graph-head function family audit — output manifest

**Representation-frozen, downstream-function-family diagnostic** on the frozen
compact-v4 graph representation (official TRAIN molecules only; official valid/test
never loaded). It asks a single question:

> On the *same* frozen graph representation `R` (302D) and with the *same* L1/MAE
> objective and data, is an explicit **low-rank cross-coordinate interaction** head
> (rank-4 Factorization Machine) a more sample-efficient downstream regression
> function family than a **same-budget** generic ReLU MLP?

No backbone is trained; patch representation, tokenizer, pair encoder, centre update,
pooling, topology branch and loss are unchanged. No new structural feature is added.
Unlike the earlier frozen adapter audits, every head is a **direct** predictor
`yhat = f(R)` (no residual `yhat_0`), and each fold uses the true nested OOF structure
(7200 head-fit / 800 head-selection / 2000 untouched outer-heldout).

Full scientific note: `tracks/ksvd/notes/graph_head_function_family_audit.md`.
Module: `tracks/ksvd/experiments/luyin16/zinc_graph_head_function_family.py`
(stages `inventory → spec → cache → integrity → splits → preprocessing → convergence →
stage1 → stage1b → stage2 → mechanism → catboost → figures → decision → all`).
Tests: `tracks/ksvd/tests/test_graph_head_function_family.py` (13 tests, all pass).

CSV/JSON/NPZ/PNG artifacts below are generated outputs (git-ignored by policy); this
README, the note, and the claim/decision YAML records are the tracked scientific record.

## Heads and parameter accounting

| head | function | params |
|---|---|---:|
| **H0** | frozen original jointly-trained graph head `302→64→32→1` (reference) | 0 |
| Hlinear | direct `302 → 1`, descriptive (not in the GO gate) | 303 |
| **H1** | direct generic ReLU MLP `302 → 5 → 2 → 1` | **1530** |
| **FM (E)** | direct rank-4 Factorization Machine: `b + w^T z + Σ_{i<j}<v_i,v_j>z_i z_j` | **1511** |
| H2 | direct stronger generic ReLU MLP `302 → 13 → 13 → 1` | 4135 |

FM = bias 1 + linear 302 + interaction 302×4 = 1208 = **1511**. H1/FM mismatch
**+1.257%** ≤ pre-registered 3%. All heads share fit-only coordinate standardisation
(7200 head-fit only), L1/MAE, Adam(lr=1e-3, wd=0), deterministic mini-batch 512,
best-selection checkpointing and fixed horizon 800 (chosen by a fold0/seed0
convergence trace).

## Files

| File | Contents |
|---|---|
| `checkpoint_inventory.json` | frozen OOF checkpoint inventory (fold × seed, state `.pt` SHA-256, params, oof_mae); seeds [0,1] complete |
| `representation_integrity.json` | integrity/reconstruction gates on 5/5 folds for both seeds; `all_passed: true` |
| `split_manifest.json` | per-fold 7200/800/2000 nested OOF membership + sha256 |
| `preprocessing_stats.json` | per-fold fit-only standardisation stats and degenerate-coordinate counts |
| `head_spec.json` | representation, preprocessing, objective, head family, parameter accounting, CatBoost status |
| `parameter_counts.json` | exact parameter counts and the H1/FM match gate |
| `convergence_trace.csv` / `.json` | fold0/seed0 selection MAE at 100/200/400/800 and the fixed-horizon rule |
| `stage1_fold_results.csv` | per-fold `H0 / Hlinear / H1 / H2 / FM / FM-no-interaction` MAE, `Δsmall`, `Δstrong`, `Δorig`, bulk, selection MAE, FM diagnostics |
| `stage1_bootstrap.json` | stratified molecule-level paired bootstrap for `Δsmall`, `Δstrong`, `Δorig`, interaction ablation + aggregate |
| `stage2_fold_results.csv` / `stage2_bootstrap.json` | second frozen backbone seed (seed 1) under the identical protocol |
| `pooled_backbone_results.csv` | both backbones × 5 folds pooled per-fold table |
| `final_bootstrap.json` | stage1/stage2 `Δsmall` CIs and the pooled per-fold `Δsmall` |
| `fm_interaction_contributions.csv` | per-molecule bias / linear / interaction / prediction decomposition (backbone 0) |
| `fm_interaction_ablation.csv` | per-fold MAE with V=0 (interaction off, no retrain) |
| `fm_mechanism_summary.json` | interaction-contribution scales and ablation summary |
| `catboost_secondary_status.json` | pre-registered CatBoost-MAE secondary reference recorded **unavailable** |
| `final_frozen_decision.json` | final verdict + criteria + case + scope caveat |
| `e2e_status.json` | end-to-end FM replacement **not authorized** |
| `state_exports/graph_head_R_cache_v1_fold{0-4}_seed{0,1}.npz` | compact per-molecule cache `R / yhat_0 / target / subset_index / role / fingerprint` (all 10000 molecules per fold) |
| `figures/figure1_stage1.png` | per-fold `Δsmall`/`Δstrong` and direct-head MAE (backbone 0) |

`stage1b_init_results.csv`, `e2e_seed_results.csv` and `shared_initialization_audit.json`
are **not** produced: Stage 1 was an EFFICIENCY ADVANCE (not borderline), so Stage 1b was
not triggered, and the frozen final gate was not met, so the end-to-end replacement was
not authorized. `catboost_secondary_results.csv` is absent because CatBoost is not
installed.

## Key numbers

Representation integrity (5/5 folds, both seeds, all exact): stored target == label;
role membership == 7200/800/2000; re-fed `R` through the frozen head == `yhat_0`
(max 0.0); holdout `yhat_0` == frozen OOF prediction (max 0.0); holdout `R` ==
centre-incidence export `R` (max 0.0, seed 0).

Evaluation MAE means (untouched outer-heldout 2000):

| stage | H0 | Hlinear | H1 | H2 | FM | FM no-interaction |
|---|---:|---:|---:|---:|---:|---:|
| backbone seed 0 | 0.175809 | 0.187802 | 0.175539 | **0.170051** | 0.172165 | 0.286558 |
| backbone seed 1 | 0.174253 | 0.186197 | 0.176349 | **0.170508** | 0.175931 | 0.283952 |
| pooled | 0.175031 | 0.186999 | 0.175944 | **0.170280** | 0.174048 | — |

- Stage 1 (seed 0): `mean Δsmall = +0.003374` (4/5 folds positive), `mean Δstrong =
  −0.002114`, `mean Δorig = +0.003643` → **EFFICIENCY ADVANCE**, so the second
  backbone seed was spent.
- Stage 2 (seed 1): `mean Δsmall = +0.000417` (only 2/5 positive), `mean Δstrong =
  −0.005424` → **did not replicate**.
- Pooled: `Δsmall = +0.001896` (< 0.0025 GO threshold), `Δstrong = −0.003769`,
  fold-averaged `Δsmall` positive in 3/5 folds, paired bootstrap 95% CIs both include 0.
- FM interaction is alive: `std(interaction) = 0.905` vs `std(linear) = 1.852`;
  forcing `V=0` raises MAE by `+0.1144` (5/5 folds).
- Decision: **INCONCLUSIVE (Case E)** — `final_frozen_decision.json`. No rank sweep, no
  end-to-end FM, no hybrid.
- CatBoost-MAE secondary reference: **unavailable** (package not installed).

## Gate 0 / protocol

`G1` stored target == official-train label `y_stored` (max 0.0); `G2` role membership ==
(7200 fit, 800 selection, 2000 holdout); `G3` re-feeding stored `R` through the frozen
head == stored `yhat_0` (max 0.0); `G4` holdout `yhat_0` == frozen OOF fold prediction
(max 0.0); `G5` holdout `R` == corrected centre-incidence export `R` (max 0.0, seed 0).
Degenerate (fit-constant) coordinates are set to `z = 0`.
