# Structure-route resolution, 8h sprint (v1)

This file freezes the execution protocol for
`tracks/structural-sufficiency/structure_route_resolution_8h_v1/`.  The study
design is the user-supplied 2026-11-23 prompt; this document records the
implementation choices that make it executable and the queue that will be run.
No result below was used to change a model, feature, threshold, or metric.

## 0. Clock and budget

- T0 = `2026-11-23T15:23:01Z` (project clock, UTC).
- Wall target 7–7.5 h, hard stop T0+8 h.
- Two A100-40GB GPUs, one training process per GPU, no DDP.
- Target ≤14 GPUh, hard cap 16 GPUh.  GPUh is allocated-GPU wall time,
  including setup, evaluation, failures, and recovery.
- No optional new training after T0+6:30; all training stopped by T0+7:15.
- GPUh ledger and per-run estimates: `analysis/budget.json`.
- Queue transitions and trigger arithmetic: `analysis/branch_log.jsonl`.

## 1. Data and labels

Only the 3,360 official Train molecules are used.

| split | identity sha256 | Fit | Dev |
|---|---|---|---|
| D0 | `60ac13c04fb0dab484fc8e7cc28f2a85450794ed3ebeeb0a87d1b41c3532559e` | 2856 | 504 |
| D1 | `c94e2d4bd92e51ab91119923d61e955f5a2f686fc1751883e858fce88dab6bee` | 2856 | 504 |

`sr_data.load_bundle` re-verifies the pinned `r2_sentence_replication_v1`
loader, the split identity and row counts, finite labels on Fit∪Dev, and that
every official Valid label is NaN.  Each run recomputes its target z-score,
graph/atom scalars, categorical support, and unseen-category masking from its
own Fit rows only.

## 2. Route A — energy response × Möbius (2×2)

Arms (D0/s0 first): `E_RAW`, `E_MOB`, `F_RAW`, `F_MOB`.

- Support: for every atom root, every rooted connected induced subgraph of size
  1–4 inside the root's radius-2 support, de-duplicated by the exact full
  9-atom/3-bond coloured rooted type, with all real internal bonds.
- `E_*`: pinned RICE-v0 model, CC off, Φ and the two-state symmetric potential
  unchanged; `E_MOB` is the RICE-v0 root-anchored Möbius transform.
- `F_*`: the same atom/bond embedding tables; `atom_projection` 144→16 SiLU,
  `bond_projection` 36→8 SiLU, masked absent slots exactly zero,
  `support_mlp` 122→96 SiLU→32.  All `(k-1)!` non-root permutations are
  averaged.  `F_MOB` applies the same Möbius recursion to ψ.
- Identical initial state within E_RAW/E_MOB and within F_RAW/F_MOB; shared
  heads identical across E and F.  Expected parameters: E 129,198,
  F 129,878.  `sr_train.py` hard-fails on the config ledger.
- Recipe: AdamW 3e-4→1e-5, wd 1e-4, effective batch 64 molecules, warmup 5,
  cosine ≤240 epochs, min 80, patience 45, clip 5, dropout 0.1, float32, no AMP.
- Formal entry: `code/sr_train.py`.

Conditional A deepening (only if 4.6 triggers): `E_MOB_BIG`, `F_MOB_BIG`,
`D0/s0`, both arms, widened sentence/readout/readout-mid 384/384/192, response
copied from the same-seed SMALL initial state.

## 3. Route B — message range

Arms (D0/s0): `G_FULL5`, `G_EGO2_5`; clone-identical initial state.

- 5 layers, hidden 128, MLP 256, per-node `torch.nn.LayerNorm`, residual,
  learnable ε, no trunk dropout, readout dropout 0.1, no root/distance/boundary
  features, no virtual node, no graph-mode norm.
- `FULL`: the original graph.  `EGO2`: one complete induced radius-2 ego per
  atom, packed as disconnected graphs, same weights, one root vector per atom.
- Shared readout: `[sum, mean, std, log1p(n_atoms), g8]` (393) →128→64→1,
  plus beta; g8 from the same safe loader and Fit-standardised.
- Recipe: AdamW 1e-3→1e-5, wd 1e-5, otherwise the same schedule/clip/batch.
- Formal entry: `code/sr_train.py`.

Conditional radius-3 (`G_EGO3_5`, D0/s0) only under the 5.5 trigger.

## 4. Strong reference R2_REF

Pinned `r2_sentence_replication_v1` SENTENCE_BASE, 416,177 parameters, original
Fit-only recipe, D0/s0 re-run once.  D0/s1 and D1/s0 each run once and are
shared by every comparison.  Historical D0/s0 (`0.5505589717057867`) is an
identity/replay check, never mixed into a new paired seed.

## 5. Route C — real-edit response kernels (CPU)

### 5.1 Catalogue

`sr_catalogue.py freeze` enumerates official Train only: connected, no explicit
H, non-ring single bridge, S 1–8 heavy atoms, C ≥10 and ≥75% of the parent,
full 9/3 attributes and port/cut columns, parent `verify_reassembly`, exact
coloured-incidence pynauty certificates, parent de-duplication.  Orientation is
`S_a < S_b` by exact key; `Δy = y_b − y_a`; `transform_id` includes the cut
bond triplet.  All pairs are written and hashed before labels are read:

- `analysis/catalogue/pairs_frozen.npz`
- `analysis/catalogue/catalogue_freeze.json` (`labels_read: false`)

Observed coordinate: 2,133 pairs (parent coordinate, not a target).

### 5.2 Features

`sr_wl.py`: 3 WL blocks × 512 counts (initial, two updates), each block
L2-normalised, plus `log1p(nodes)`, `log1p(internal bonds)` standardised on Fit
pairs with half pair weight per S endpoint (pair weight for the shared core).
Node tokens use all 9 raw atom columns, a port flag, `min(dist_to_port, 3)` and,
on the port, the 3 cut-bond columns.  Edge tokens use all 3 raw bond columns.
Hashing is fixed `blake2b`, never Python salted hash.  C_R2 is the radius-2
induced ball at the C port; C_FULL is the whole core.

### 5.3 Kernels and models

`x_p = φ(S_b) − φ(S_a)`, no target centering.  `χ_p` is the Fit-weighted centred
core feature; `K_T = x·x`, `K_C = χ·χ`, `K_TC = K_T ⊙ (1 + K_C)`.  `K_T` and
`K_C` are divided by their Fit weighted mean diagonal, then `K_TC` by its own;
all cross matrices use the training scales.  Degenerate diagonal blocks are
recorded, never jittered.

Required models: `ZERO`, `EXACT_MEAN` (component means, shrink `/(k+1)`,
coverage reported), `K_TRANSFORM`, `K_CONDITIONAL_R2`, `K_CONTEXT_ID`
(Fit-pair core vocabulary one-hot minus weighted mean; unseen core exactly 0).
`K_CONDITIONAL_FULL` is the single optional CPU extension, fixed as first to
drop.

### 5.4 Solver and CV

Training pairs are Fit–Fit only.  Pair weight is the inverse size of its Fit
pair-graph component, renormalised to mean 1.  Float64 weighted KRR with
`λ ∈ {1e-4, 1e-2, 1}`; 3-fold CV over connected components (fixed SHA-256
order, greedy pair-count balancing); component-equal MAE selects λ, exact ties
take the larger λ.  Each fold rebuilds vocabulary, centring, scales, weights,
and target sigma from fold-train only; OOF predictions are restored to label
units before scoring.  Final fit uses all Fit–Fit and the full-Fit label sigma.

Primary metric: component-equal MAE on Dev–Dev (`K_TRANSFORM −
K_CONDITIONAL_R2`, C5).  Also Fit–Dev, Fit–Fit diagnostic, pair-equal MAE/RMSE,
prediction std, ZERO, and the pre-registered strata:
transform seen/unseen; support 0/1/2+ training components; core seen/unseen in
Fit pairs; core seen in any legal Fit cut (strict-new is the complement); eval
component has Fit–Fit support.  `NA` is reported when a scope is empty.

Uncertainty: paired cluster bootstrap over all official-Train pair-graph
components, 5,000 draws, seed 1703, multiplicity preserved; primary metric is
within-component mean then equal-weight components.  A single component gives
`NA`.

Functional checks (`analysis/response_checks.json`): swap → sign flip, identical
endpoints → 0, PSD, weighted KRR = explicit tensor-feature ridge, and the fixed
32/32 synthetic conditional positive control (`MAE < 1e-4` and below `K_T`).

## 6. Queue and branch rules

Stage I (mandatory, D0/s0): `R2_REF`, `E_RAW`, `E_MOB`, `F_RAW`, `F_MOB`,
`G_EGO2_5`, `G_FULL5`; plus the D0 C kernels/support/positive controls.

Stage II triggers (config `branch_rules`, thresholds `δ=0.01`, `ε=0.015`):
BIG pair, radius-3, optional C_FULL, optional 50% Fit.

Stage III mandatory baseline package: D0/s1 `R2_REF`, `E_MOB`, `F_MOB`.
Then up to three fixed comparison packages in the order P_SCOPE, P_BASIS,
P_HEAD, each with its own candidate and reference, only under the written
formulas.

Stage IV: after D0 architecture, D0/s1 and C features are fixed, write
`analysis/queue_lock.json`; only then read/run D1.  D1/s0 baseline is
`E_MOB`, `F_MOB`, `R2_REF`; a package enters D1 only if both D0 seeds agree in
direction and mean |gain| ≥ δ/2, or a new model's two-seed mean ≤ R2 mean + ε.
C reruns the identical kernels/scalers/CV on D1 Fit.

Stage V: at most one 2-run conflict package, first matching priority only.

## 7. Reporting rules

Gain = error(reference) − error(candidate); positive favors the candidate.
Every split/seed is reported; aggregates are averages of model metrics, never
ensembled predictions.  Model-selection CIs are exploratory, not significance
tests.  Missing comparison cells are never filled with 0.  Route states come
from section 12 of the user prompt and must name the finite proposition,
unexcluded alternatives, and the evidence/support limitation.

Machine outputs: `analysis/main_table.csv`, `contrasts.csv`,
`response_support.csv`, `response_metrics.csv`, `decision.json`,
`analysis/figures/`, and `results/<run_id>/` with manifest, curves, best/final
checkpoints, and per-molecule Fit/Dev predictions.  `REPORT.md` uses the
one-page answer first.

## 8. Frozen implementation hashes

The parent source pins are in `sr_common.SRC_PINS` and re-checked on every
entry.  The catalogue freeze file records the pair arrays' SHA-256 and the D0
Fit/Dev row hashes.  Code commit used for formal runs is the commit containing
this file; `rr` refuses a dirty worktree or a stale deploy.