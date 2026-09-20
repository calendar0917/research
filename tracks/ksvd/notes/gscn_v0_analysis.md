# GSCN-v0 — analysis

Pre-registration: [gscn_v0_preregistration.md](gscn_v0_preregistration.md)
(frozen before any run). Code: `code/run_gscn_v0.py`,
`code/analyze_gscn_v0.py`; tests `tests/test_gscn_v0.py` (9 pass).
Regime: deterministic A100, canonical PyG ZINC `subset=True` official
**train 10000 / valid 1000**; **official test never loaded**
(`official_test_loaded: false` in every summary). Run commit `e7f5e02`, seed 0,
canonical `OPTIMIZED_PROTOCOL` (Adam lr 1e-3, wd 1e-5, batch 128, ≤240 epochs,
patience 40, L1, clip 5, best-valid checkpoint, equal-weight Top-5 soup).

## 0. B-Null / B-Full clarification (carried from the pre-registration)

Reading the repo, the canonical objects are **rooted-patch** models that share a
49,343-param backbone and differ only by the local 16-D token generator:
`B-Null = "null"` (49,343; soup 0.123028), `B-Full = "shared_structural"`
(84,495; soup 0.119818). Both already consume handcrafted structural statistics
(`patch_cont`, `pair_relation`, `global_context`, `topology_features`,
`parent_token`); the canonical handcrafted gap `B-Null − B-Full = 0.0032 < 0.02`.
The round's premise (B-Null = raw learner, B-Full = +handcrafted statistics) is
**false in this repo**; per §0 the GSCN input discipline was frozen to
**raw atom category + raw bond category + adjacency only** and the primary
baseline became a raw-input graph model, `B-Null-Raw`, at the frozen capacity
`d=64, L=4` using the repo's own raw reader primitive. Canonical B-Null/B-Full
are reported as handcrafted-statistics references.

## 1. Stage 0 / Stage 1 (implementation integrity)

* Stage 0 (data-free) **all passed** on CPU and GPU: permutation equivariance
  max err ≤ 6.3e-6, batching invariance ≤ 1.4e-6, graph-prediction invariance
  ≤ 1.6e-6, `‖∇_D L‖ > 0` all layers, `Y:N×d / A:N×K / AD:N×d`, `A ≥ 0`,
  no hidden bypass (perturbing `D` changes the block output).
* Stage 1 train-only smoke (**valid not used for tuning**) gate **passed**:
  active fraction 0.077–0.088 (∈ (0.02,0.50)); pooled atom usage 0.590 ≥ 0.50;
  max single-atom share ≤ 0.097; effective rank ≈ 42.5; train loss fell
  1.096 → 0.652 in 5 epochs; non-zero `D` gradients.

## 2. Primary result (seed 0, canonical protocol)

| arm | hidden update | params | best-valid | **Top-5 soup** |
|---|---|---|---|---|
| `bnull_raw` (B-Null-Raw) | `SiLU(W_self h + Σ(W_msg h_u + W_edge e))` | 55,681 | 0.2041 | **0.1950** |
| `generic` (param-matched control) | mixing + `Linear(64,64)→SiLU→Linear(64,64)` | 89,985 | 0.2111 | 0.1982 |
| `nothresh` (GSCN, λ=0) | mixing → projected ISTA (`ReLU`) `A D̄` | 89,473 | 0.2707 | 0.2605 |
| `sparse` (GSCN, calibrated λ) | mixing → **sparse** ISTA `A D̄` | 89,473 | 0.2271 | 0.2123 |
| *B-Full reference* | canonical handcrafted + learned encoder | 84,495 | — | *0.119818* |
| *B-Null canonical reference* | canonical handcrafted patch model | 49,343 | — | *0.123028* |

Deltas (soup): **Δ_Null = M_Null − M_Sparse = −0.0173** (sparse *worse*),
**Δ_Capacity = M_Generic − M_Sparse = −0.0141** (sparse worse),
**Δ_Sparsity = M_NoThresh − M_Sparse = +0.0482** (sparsity helps *within* the
dictionary-core family). Handcrafted gap `M_Null − M_Full = 0.0752 ≥ 0.02`, so
`ρ = Δ_Null/(M_Null−M_Full) = −0.230` — GSCN does **not** recover the gap; it
widens it (ρ well-defined but negative, no recovery claim).
Param ratio `P_Sparse/P_Null = 1.607×` (reported explicitly per §32); generic is
within `+0.57 %` of GSCN.

## 3. Mechanism, geometry, coupling audits (Sparse-GSCN)

* **Codes are genuinely sparse** (train and valid): active fraction 0.060–0.078
  per layer; median active atoms/node 7–10, p90 12–14; atoms used 75–84 % per
  layer (dead 21–32 / 128); max single-atom share ≤ 0.091 (no domination).
* **Dictionary non-collapsed**: effective rank 19.5–25.2 (of 64), coherence
  0.76–0.93, mean |pairwise cosine| 0.14–0.16, `‖D_final−D_init‖_F ≈ 10.9–11.4`
  → the task did move the dictionary. `L_task → D` is an active path
  (`‖∇_{D^ℓ}L‖` first-50 mean 2.6/4.3/6.4/15.8, last-50 mean 0.20/0.38/0.47/1.97;
  nonzero at all layers).
* **Inference ablations** (no retraining, intact valid 0.2271): atom-row
  permutation Δ = 0.0 (sanity invariance); random-normalised dictionary
  Δ = +40.71; within-batch code shuffle Δ = +2.58; layer-mean code Δ = +4.47 →
  the trained model *does* depend strongly on the learned atoms/codes.
* **Post-hoc structural semantics** (explanation only): top-100 activating nodes
  per atom show rooted radius-2 WL-key concentration ≈ 0.24–0.28 vs 0.044 for
  random nodes (radius-1 0.45–0.56 vs 0.199), i.e. the atoms do pick up repeated
  rooted structural contexts — but this does not translate into predictive value.

So the sparse-dictionary mechanism is **active and genuinely sparse**, not
collapsed, and task gradients reach the dictionary. The failure is **not**
mechanism collapse (§38) and not pure optimization failure (§39, train loss
descends: sparse train ≈ 0.17). It is that the dictionary-core hidden-state
update is **dominated by the plain raw baseline and by a parameter-matched
generic MLP control**.

## 4. Pre-registered gates

* strong-positive: **False** (`M_Sparse 0.2123 ≰ M_Null−0.02 = 0.1750`).
* capacity-only failure (§36): **True** (`M_Generic 0.1982 ≤ M_Sparse 0.2123`).
* sparsity-not-needed (§37): **False** (`M_NoThresh 0.2605 > M_Sparse 0.2123`;
  explicit λ is supported).
* ambiguous (requires seed-1): False (deltas not in the ambiguous bands).
* handcrafted-gap recovery: ρ negative → report absolute MAE only.

## 5. Verdict

> **The gain is explained by generic model capacity rather than sparse dictionary learning.**

Nuance: strictly there is *no* gain to explain — no GSCN variant beats the raw
baseline `B-Null-Raw` (0.1950); the pre-registered capacity-only gate fires
because the param-matched generic control (0.1982) is at least as good as
Sparse-GSCN (0.2123). The only mechanism-supported effect is
sparse-over-no-threshold (+0.048), which confirms that the explicit λ threshold
matters *within* the dictionary-core family but is not enough to make the family
competitive. Q1: the canonical handcrafted models are ~0.075 MAE better than a
raw graph model. Q2: Sparse-GSCN does **not** improve on the raw baseline. Q3:
the generic control explains at least as much. Q4: thresholded sparse coding is
necessary *internally* but not predictively sufficient. Q5: yes, `L_task`
updates `D`. Q6: yes, codes are sparse. Q7: atoms show above-random rooted
structural concentration post-hoc. Q8: no — the dictionary-core is not
competitive as an independent architecture.

## 6. Decision

**Reject dictionary-core architecture and reassess the research premise.** Even
the simplest raw SiLU message-passing learner beats both dictionary-core
variants, and extra generic capacity does not beat the raw learner either, so
there is no evidence here that a learned sparse code helps as the ZINC
hidden-state transition. Not authorized automatically: any further step needs a
new pre-registration.

## 7. What was not done / limitations

Official test never opened; single seed 0 (no seed-1, per the negative primary
gate); no K/λ/T/depth/width sweep; no residual/bypass; no reconstruction
auxiliary loss; no PSCD/motifs/ports/MDL/handcrafted-statistic inputs. The raw
baseline is a lifted repo raw reader (the repo has no canonical raw-graph
backbone); GSCN adds 2 LayerNorms and uses the same head/pooling. Results are
valid-only; deltas are at the seed-0 scale.

## 8. Computational reporting (no compression claim)

wall / peak GPU: `bnull_raw` 560 s / 37.2 MB, `generic` 719 s / 49.1 MB,
`nothresh` 1194 s / 73.0 MB, `sparse` 1270 s / 72.1 MB (240 epochs, 10000
train graphs). ISTA overhead ≈ +6 % wall over no-threshold at equal depth/width;
no computational-compression claim is made.

## Artefacts

`results/gscn_v0/`: `stage0_tests.json`, `smoke_gate.json`, `params.json`,
`summary_{bnull_raw,generic,nothresh,sparse}.json`,
`curves/history_*.json`, `states/*.pt`, `soup_states/*.pt`, `audit_sparse.json`,
`ablation_sparse.json`, `semantics_sparse.json`, `analysis.json`.
