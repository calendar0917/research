# Graph head refit & capacity decomposition audit — output manifest

**Representation-frozen, downstream-head decomposition diagnostic** on the frozen
compact-v4 graph representation (official TRAIN molecules only; official valid/test
never loaded). It answers one question:

> Why does a small post-hoc MLP head (`302→13→13→1`, H2) beat the original
> jointly-trained graph head (`302→64→32→1`, H0) on the *same* frozen `R` by
> ≈ +0.00475 — is it **capacity**, **frozen-representation refit**,
> **late readout adaptation**, **initialisation**, or **standardisation /
> optimisation protocol**?

No backbone is trained; tokenizer, patch representation, pair encoder, centre update,
pooling, topology, `R`, loss and backbone are unchanged. The two frozen OOF backbone
seeds × 5 folds are reused from
`../graph_head_function_family/state_exports/graph_head_R_cache_v1_fold*_seed*.npz`
(read-only). Every refit head is a direct predictor `yhat = f(R)` and each fold uses the
true nested OOF structure (7200 head-fit / 800 head-selection / 2000 untouched
outer-heldout).

Full scientific note: `tracks/ksvd/notes/graph_head_refit_capacity_decomposition.md`.
Module: `tracks/ksvd/experiments/luyin16/zinc_graph_head_refit_capacity_decomposition.py`
(stages `inventory → parameters → spec → integrity → splits → standardization → refit →
summary → bootstrap → raw → decision → figures → all`).
Tests: `tracks/ksvd/tests/test_graph_head_refit_capacity_decomposition.py` (13 tests, all
pass).

JSON/CSV/NPZ/PNG artifacts below are generated outputs (git-ignored by policy); this
README, the note, and the claim/decision YAML records are the tracked scientific record.

## Heads and parameter accounting (exact, from the real checkpoints)

| head | function | init | input | params |
|---|---|---|---|---:|
| **H0** | original jointly-trained head `302→64→32→1` + LayerNorm(64) + Dropout(0.05) | checkpoint | raw `R` | **21633** (21505 linear + 128 LN) |
| **S-scratch** | pure ReLU MLP `302→13→13→1` | scratch seed0 | `z` | **4135** |
| **L-scratch** | pure ReLU MLP `302→64→32→1` | scratch seed0 | `z` | **21505** |
| **L-warm** | original head module, exact first-layer reparameterisation | checkpoint | `z` | 21633 |
| **S-raw** (control) | pure ReLU MLP `302→13→13→1` | scratch seed0 | raw `R` | 4135 |
| **L-scratch-raw** (conditional) | pure ReLU MLP `302→64→32→1` | scratch seed0 | raw `R` | 21505 |

`z = (R−μ)/max(σ,ε)` uses fit-only statistics from the 7200 head-fit molecules;
`z=0` on fit-degenerate coordinates. L-warm reads `z` but is initialised with
`W' = W diag(scale)`, `b' = b + W μ`, so its optimisation step 0 is function-identical to
H0 on raw `R` (float64 identity max `6.2e-15`; float32 deployment max `4.4e-5 < 1e-4`).

Real-model parameter accounting (fold0/seed0): total **96141**, head 21633, non-head
74508. Replacing the head with the small one gives **78643** (mean **78732.6** over the
10 fold-seeds), a saving of **17498** per model.

## Files

| File | Contents |
|---|---|
| `checkpoint_inventory.json` | frozen OOF checkpoint inventory (fold × seed, `.pt` SHA-256, params, oof_mae); seeds [0,1] complete |
| `representation_integrity.json` | Stage-0 integrity gates on 10/10 fold-seed instances; `all_passed: true` |
| `standardization_equivalence.json` | per-fold raw-head vs transformed-standardised-head equivalence (float64 identity + float32 deployment) |
| `standardization_stats.json` | per-fold fit-only standardisation stats and degenerate-coordinate counts |
| `split_manifest.json` | per-fold 7200/800/2000 nested OOF membership + sha256 |
| `head_specs.json` | representation, preprocessing, standardisation transform, objective, head matrix, parameter counts, forbidden search space |
| `parameter_counts.json` | exact head/total parameter counts and the small-head replacement totals |
| `fold_results.csv` | per (backbone, fold) `MAE_H0/S/Lscratch/Lwarm/Sraw`, `ΔS/ΔL/ΔW/Δcap/Δinit`, selection/train MAE per head |
| `backbone_summary.csv` | per-backbone means/medians/positive-fold counts |
| `pooled_summary.json` | pooled means/medians/per-fold deltas/positive-fold counts |
| `paired_bootstrap.json` | primary molecule-level stratified bootstrap + hierarchical bootstrap for `ΔS/ΔL/ΔW/Δcap/Δinit` |
| `learning_curves.csv` | fit/selection MAE at epochs 100/200/400/800 for S / L-scratch / L-warm |
| `raw_vs_standardized.csv` / `.json` | S-std vs S-raw comparison and the conditional-control trigger |
| `large_raw_control.csv` / `.json` | conditional L-scratch-raw run plus `Δcap_raw` (capacity sign under raw inputs) |
| `final_decision.json` | final verdict + criteria + case + raw-control summary |
| `second_init_status.json` | second head init **not run** (`|Δcap| = 0.00025 < 0.001`) |
| `molecule_errors.npz` | per-molecule outer-heldout errors for every head (bootstrap input) |
| `molecule_deltas.npz` | molecule-keyed paired deltas (cached bootstrap input) |
| `figures/figure1_refit_gains.png` | per-fold `ΔS/ΔL/ΔW` |
| `figures/figure2_delta_cap.png` | per-fold `Δcap` per backbone seed |
| `figures/figure3_learning_curves.png` | mean fit / selection MAE vs epoch |

## Key numbers

Stage-0 integrity: all gates exact on 10/10 fold-seed instances (target == label; role
membership 7200/800/2000; re-fed `R` through the frozen head == `yhat_0` max 0.0; forward
gate max 0.0). Standardisation equivalence: float64 identity max `6.2e-15`; float32
deployment max `4.4e-5 < 1e-4`.

Pooled evaluation MAE (untouched outer-heldout 2000):

| head | pooled mean MAE |
|---|---:|
| H0 | 0.175031 |
| S-scratch | 0.170280 |
| L-scratch | 0.170534 |
| **L-warm** | **0.169661** |
| S-raw | 0.168045 |
| L-scratch-raw | 0.164781 |

`S-scratch` reproduces the prior H2 pooled MAE to `4.9e-7` and `H0` to `2.4e-7`.

| delta | pooled mean | 95% CI | P(>0) | fold-averaged + |
|---|---:|---|---:|---:|
| ΔS = H0 − S | **+0.004751** | [+0.00150, +0.00952] | 0.9997 | 5/5 |
| ΔL = H0 − L-scratch | **+0.004496** | [+0.00108, +0.00946] | 0.9983 | 5/5 |
| ΔW = H0 − L-warm | **+0.005369** | [+0.00414, +0.00660] | 1.0000 | 5/5 |
| **Δcap = L-scratch − S** | **+0.000255** | [−0.00097, +0.00150] | 0.658 | 2/5 |
| Δinit = L-scratch − L-warm | +0.000873 | [−0.00390, +0.00406] | 0.695 | 3/5 |
| S-raw − S | −0.002234 | [−0.00356, −0.00090] | 0.0004 | 3/5 |
| Δcap_raw = L-scratch-raw − S-raw | −0.003264 | — | — | 1/5 |

Learning curves: L-scratch overfits (fit MAE `0.0371` vs S `0.0740` at epoch 800 while
selection worsens `0.1789→0.1871`) but that overfit does not survive on outer-heldout;
L-warm shows the under-adaptation pattern (starts at H0 and improves).

## Decision

**LATE-READOUT-ADAPTATION GO (Case C; the Case B post-hoc-refit conditions also hold).**
The original architecture is sufficient; at the end of joint training the head is
under-adapted to the final frozen `R`, and any reasonable head refit — including a warm
continuation of the original head — recovers the gap.

- **Not** CAPACITY GO: `Δcap = +0.000255`, CI includes 0, 2/5 fold-averaged, and the
  sign reverses on raw inputs.
- **Not** MIXED: requires `Δcap ≥ +0.002`.
- Standardisation is **not** the mechanism (raw is significantly better).
- No end-to-end small-head replacement, no width sweep, no second init, no new head
  family, no horizon extension.
- Next: leakage-safe `canonical training → freeze backbone → head-only MAE adaptation`
  on the official train/valid protocol.
