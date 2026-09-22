# Preregistration — TCCD-GLOBAL62-v0: Frozen TCCD + B-Full Global Context Residual Screen

Round name: **TCCD-GLOBAL62-v0** (*B-Full Global Context Residual Screen*).
Study: `zinc-context-gap`.
Status: frozen before the single formal seed-0 probe on 2026-09-22.

## 0. This is a new, single-variable round — not a rescue and not a retrain

This round does **not** reopen TCCD-v2/v5/v6, CENTER-COMP, CHEM-CONT,
GRAD-CONT, B-Full or B-Null, and it does not re-run any existing baseline.
Every existing number is read from its frozen formal artifact; the only new
trained object is one small residual head on `R^62`.

It is **not** a TCCD-v8 rescue. It is a targeted diagnostic that asks whether
the frozen TCCD route is simply missing a very plain block of graph-level
information.

## 1. Question

> With the TCCD representation/prediction **completely frozen**, does the
> exact 62-D `global_context` used by B-Full
> (`global_feature_views(...)["global_all"]`) contain a large amount of
> task-relevant graph-level information that TCCD currently leaves on the
> table?

Concretely: can

```text
frozen TCCD prediction  +  global_62
```

materially break the existing `~0.2514` frozen ceiling, beyond a mere
intercept recalibration, and only if the molecule <-> global-statistics
binding is correct?

## 2. Frozen lineage and execution discipline

* Base prediction: **exact existing TCCD-v5 PRE Top-5 soup**.
  * Formal artifact:
    `tracks/ksvd/results/tccd_v5/stageA_seed0.json` (`results.PRE.soup_valid_mae
    = 0.2514181435108185`).
  * Frozen soup state:
    `tracks/ksvd/results/tccd_v5/stageA_pre_seed0_soup.pt`, produced by the
    formal v5 run (commit `1ec96ffd2abf0e9e2690d4fa7faf0ddb734bb1bb`) on the
    frozen TCCD-v2 PrototypeREL checkpoint
    `tracks/ksvd/results/tccd_v2/prototype_rel_seed0_best.pt` (SHA-256
    `093e0e46d4d5dab7e5558b807d993d1136df58d8f531d6b9b3621e8ab05c0f42`).
  * The base prediction is obtained by a **pure forward pass** of the frozen
    soup over the exact reused TCCD-v5 pair cache
    (`tracks/ksvd/results/tccd_v5/cache/pair_all_*.npy`,
    `official_test_loaded: false`). No TCCD parameter is ever trained,
    modified, or instantiated as trainable.
* Split: the exact fixed internal split, seed `20260922`, `8000 train / 2000
  dev` (`run_tccd_v0.internal_split`). Official valid and official test are
  never loaded.
* `global_context`: the exact B-Full `global_feature_views(...)["global_all"]`
  from `tracks/ksvd/experiments/luyin16/zinc_long_range_proxy.py`, reused
  unchanged. It is label-free and built from the official PyG ZINC
  `subset=True` train dataset, whose iteration order is exactly the TCCD record
  order (`run_aiom_representation_audit.repo_loader` uses the same
  `_load_zinc` / `_data_to_graph`). Width is 62:
  `global_structure_short` (15) + `global_structure_long` (15) +
  `global_attributes` (atom histogram 28 + bond histogram 4).

**DEV reproduction gate.** The frozen forward must reproduce internal-dev
`MAE(yhat_base, y)` equal to `0.2514181435108185` within `1e-6`. If the
prediction cannot be reproduced exactly, the round **STOPs and reports the
artifact gap**; it does not fall back to a best checkpoint, POST, or a rerun.

## 3. The single new variable — and the exact list of what is NOT included

Only `global_context ∈ R^62` is tested. Specifically excluded:

* `patch_cont[146]`, `pair_relation[23]`, `parent_token`,
* B-Full local state, center context, `topology_features[25]`,
* any new cycle / structural descriptor,
* any local-state bridge, center composition, or end-to-end joint training.

No new statistic is designed. The 62 columns are the existing B-Full views.

Rationale for not adding the compact-v4 topology 25-D: prior topology work
already established a real but small graph-level topology increment (~0.01
MAE), which cannot explain `0.25 -> 0.12`. The unknown tested here is the
**mixed 62-D global context**, not topology.

## 4. Feature preprocessing

1. Read the **raw** 62-D `global_all` for all 10,000 internal graphs.
2. Fit a standardizer on the **internal 8000 train rows only** (mean/std,
   `std -> 1` for constant columns); transform train and dev.
3. Never read official valid/test.
4. Record: raw width 62, finite rate, train mean/std, constant columns, and
   exact function/version provenance.

## 5. Residual target and model

Frozen base prediction `yhat_base`. Residual target, computed on internal
train only:

```text
r_G = y_G - yhat_base,G
```

Final prediction:

```text
yhat_G = yhat_base,G + f(g_G),   g_G ∈ R^62
```

The **only new trained object** is the GLOBAL62 residual head, architecture
fixed once:

```text
62 -> Linear(62,32) -> LayerNorm(32) -> ReLU -> Linear(32,32) -> ReLU -> Linear(32,1)
```

This is exactly the B-Full `global_encoder` hidden transformation
(`zinc_ksvd_patch_path_pooling.GLOBAL_WIDTH = 62`,
`_MLPBlock(62, 32, 32, dropout=0.0)`, with `patch_hidden = 64 ->
global_encoder_hidden = max(64//2, 32) = 32`) with a single scalar residual
output appended. No width/depth/dropout/optimizer sweep; seed 0 only.

## 6. Loss and training protocol

* Loss: `mean | y - (yhat_base + f(g)) |` (L1 on the final prediction).
* seed 0; internal train/dev `8000/2000`.
* Adam `lr = 1e-3`, weight decay `1e-5`; batch `128`; max `240` epochs;
  patience `40`; Top-5 equal-weight soup; no gradient clipping.
* Primary metric: **Top-5 soup internal-dev MAE**. Best-checkpoint dev MAE is
  secondary.
* No hyperparameter adaptation to results.

## 7. Bias-only control (no second network)

On train residuals only: `b* = median(y - yhat_base)`; `yhat_bias = yhat_base
+ b*`. Report dev `MAE_BIAS`. No optimizer is used.

## 8. Evaluation-only GLOBAL-SHUFFLE

After training the single REAL head, apply a deterministic row **derangement**
`g_i -> g_{pi(i)}` inside internal dev only, with fixed seed `20260923`:

* frozen TCCD prediction unchanged,
* residual-head weights unchanged,
* dev global-feature multiset exactly preserved,
* labels unchanged,
* no fixed points.

Report `MAE_SHUFFLE` and `G_bind = MAE_SHUFFLE - MAE_REAL`. The shuffled model
is never retrained.

## 9. Registered quantities and verdict

```text
G_total  = MAE_BASE  - MAE_REAL
G_global = MAE_BIAS  - MAE_REAL
G_bind   = MAE_SHUFFLE - MAE_REAL
gap_closed_fraction = (MAE_BASE - MAE_REAL) / (MAE_BASE - 0.119818)
```

B-Full (`0.119818`) is used as a **scale reference only**, never as a strict
causal comparator.

Preregistered cases (frozen before training):

* **Case A — GLOBAL CONTEXT IS A MAJOR MISSING CHANNEL**:
  `G_global >= 0.030` and `G_bind >= 0.020`. Next: research a unified
  dictionary representation with explicit global context.
* **Case B — MATERIAL BUT NOT DOMINANT**:
  `0.015 <= G_global < 0.030` and `G_bind >= 0.010`.
* **Case C — SMALL SIGNAL**: `0.005 <= G_global < 0.015`.
* **Case D — NO MATERIAL GLOBAL SIGNAL**: `G_global < 0.005`. Close this
  explanation; no global-context architecture rescue.

**Shuffle precedence.** If REAL looks improved but `G_bind < 0.005`, the gain
must be interpreted as residual-head calibration / extra capacity /
optimization effect, and the conclusion is downgraded regardless of
`G_total`.

## 10. Secondary analyses (read-only, no retraining)

* Size stratification on dev (small / medium / large tertiles by atom count):
  report `MAE_BASE`, `MAE_REAL`, improvement per stratum. No model is retrained
  on the stratification.
* Per-column Pearson/Spearman correlation between each **raw** global feature
  and the frozen residual `r = y - yhat_base` on train and dev. Explanatory
  only; no feature selection or feature deletion.

## 11. Gate 0 (any critical failure -> STOP)

1. v5 PRE dev prediction reproduction PASS (`1e-6`).
2. TCCD weights fully frozen / no TCCD training.
3. Exact 62-D B-Full `global_all` definition, width 62, finite.
4. Train-only standardization.
5. No labels enter feature construction.
6. No official valid.
7. No official test.
8. Residual gradient reaches only the new head.
9. Shuffle preserves the dev global-feature multiset exactly.
10. Shuffled mapping has no fixed points.
11. Bias-only value computed from train residuals only.

## 12. Forbidden in this round

TCCD / B-Full retraining; official valid; official test; topology 25-D;
`patch_cont 146`; `pair_relation`; local-state bridge; center composition;
end-to-end TCCD+global training; seed 1; MLP width/depth sweep;
linear-vs-MLP selection; feature subset selection; global feature redesign;
RDKit descriptors; target-derived statistics; residual target engineering.

## 13. Stop rule

Even a very strong result (e.g. `0.251 -> 0.19`) ends the round at the record.
Do not chain topology / `patch_cont` / pair features into the same round.

## 14. What this round answers

1. Does `0.2514` reproduce exactly?
2. `G_global` — how much does the 62-D context add beyond bias correction?
3. `G_bind` — does the gain depend on correct molecule <-> global binding?
4. What fraction of the `0.2514 -> 0.1198` gap is closed, and which
   preregistered case fires?
