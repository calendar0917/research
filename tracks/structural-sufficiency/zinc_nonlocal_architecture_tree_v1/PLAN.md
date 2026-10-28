# PLAN — ZINC non-local architecture tree v1

Status: frozen before any GPU run (2026-10-29).
Track: `structural-sufficiency`. Reference base: frozen LCRC-A (`zinc_local_relation_composition_v1`,
237,519 params, seed-0 valid 0.076582, test 0.066782 exposed post-hoc). Regime: `res`, A100-40GB,
driver 550.163.01, torch 2.5.1+cu124. Historical A/R/N/DCC-L numbers were produced on the same
machine/driver/torch regime, so they are the matched reference.

## Boundaries

- Official PyG ZINC `subset=True`: train 10k / valid 1k. **Official test is never constructed,
  loaded or scored in this round.** All model selection uses valid only.
- Reuse the frozen LCRC feature cache (`lc_train.pt`, `lc_valid.pt`), frozen train-only
  normalisation (`analysis/norm.json`) and the two frozen masked-SSL encoder checkpoints.
- Reuse the exact LCRC-A trunk and R0 recipe: AdamW, batch 128, 500-step warmup, cosine
  1e-3 -> 1e-5, wd 1e-5, grad clip 5, raw-y L1, 600 epochs = 47,400 updates, full valid every
  epoch, strict argmin checkpoint.
- New modules are initialised small-nonzero after the shared trunk (which is bit-identical to
  historical A seed 0) so the interaction terms neither destroy the trunk at step 0 nor start at
  exact zero with a dead first step.
- Single model, no ensemble/soup/EMA/seed averaging. <=500k deployment parameters.
- **No lambda / width / rank / layer sweep.** One fixed form per hypothesis.

## Hypotheses and arms

### H1 Separator-Conditioned Energy (SCE)

`y = sum_v f(h_v) + sum_bridge I_e / sqrt(1+n_bridge) + rho(G)`.

For each physical bridge bond `e=(a,b)`: delete `e`, take the two real connected sides
`A_e/B_e`. Shared projection `P: 64->32` of local states. Strictly endpoint-exchange-invariant
input `[m_A+m_B, |m_A-m_B|, m_A*m_B, h_a+h_b, |h_a-h_b|, b_e, log1p|A|, log1p|B|]` -> shared MLP
(322->64->1) -> scalar. `n_bridge=0` => term exactly 0.

- `SCE-Real`: real side contents.
- `SCE-Marginal`: identical params/structure/bridges/endpoints/sizes/training stream, but
  `m_A=m_B=` molecule-level content mean field. This is a **capacity/marginal control**, not a
  perfect causal control.

### H2 Graph Equilibrium Field (GEF)

Per atom source `z_v = P_z h_v` in R^12. Positive conduction
`w_e = softplus(MLP([h_u+h_v, |h_u-h_v|, b_e])) + 1e-3` on real bonds. `L_w` weighted Laplacian,
`K = (lambda I + L_w)^{-1}`, `lambda` fixed **before training** from a label-free propagation-range
analysis (no valid-MAE search). `c_v = sum_{u!=v} K[v,u]/sum_{u!=v}K[v,u] z_u - mean_{u!=v} z_u`.
`y = sum_v [f(h_v) + g(h_v,c_v) - g(h_v,0)] + rho(G)`, `g: 76->64->1`.
`c_v=0` when all sources are equal; one exact solve only; no iterative learned-state passing.

- `GEF-Real`: learned real-bond conduction.
- Mechanism controls are bought only if Real shows signal:
  - `GEF-Fixed`: identical field/reader but uniform conduction `w_e=1` (tests learned attribution).
  - `GEF-Local`: identical context reader `g` but `c_v := 0` (tests non-local field over extra
    nonlinear capacity). Only one of the two, chosen to match the observed signal.

### H3 Joint-Span Structural Pretraining (JSSP)

Only opened if SCE and GEF both fail to show practical promise. Reuse the frozen masked atom/bond
SSL code/structure, replace independent atom/bond masking by joint masking of contiguous 2-bond
spans, rebuild all derived inputs from the masked raw graph so ring/path/junction/global statistics
cannot leak the hidden targets. Same pretraining update count / visible structures / backbone /
downstream schedule; one fixed masking mechanism, no sweeps. Compare `old SSL -> A` vs
`JSSP -> A` at the same seed. Header deleted at deployment.

## Budget

- Phase 0 CPU only: implementation + math/rename/batch/save-reload/gradient/degeneracy gates.
- Soft 14 GPU-h, hard 16 GPU-h, wall target 7-8 h, max 2 concurrent formal runs, ~2 GPU-h reserved.
- Expected cost per 600-epoch run at the historical A rate: ~0.36-0.9 GPU-h. Throughput is measured
  before full runs; a truncated model is never reported as an architecture failure.

## Decision rule (frozen)

Phase 1: `SCE-Real-s0` on `res-gpu0`, `GEF-Real-s0` on `res-gpu1`, full 600 epochs. Inspect
150-250 epoch fit/valid/mechanism; early stop only if a candidate is >=0.010 worse than A at the
same epoch, has no faster slope, and the mechanism is inactive; such a run is tagged
`EARLY_STOPPED_EXPLORATION` and never ranked as a completed model.

GREEN: full or robust MAE improvement >= ~0.003-0.005 vs frozen A_s0, ordinary / r2-unseen
ordinary not reversed, improvement not carried by a single extreme row, curve sustained. Buy seed 1
of that direction plus one matched mechanism control on the other GPU.

AMBER: close to A but broad ordinary/unseen improvement, or the sign depends on one tail row, or a
clearly better generalisation trend. Buy at most one matched seed or mechanism test. No width/rank/
LR/layer search.

RED: clearly worse than A, or both Real and control weak with no performance sign, or persistent
degradation at matched progress, or a collapsed branch. Close the mechanism; keep the evidence.

Report every model on full raw-y MAE plus: drop valid:0172, drop the fixed historical A worst-10,
ordinary (y>-3), r2-unseen ordinary (225), seen ordinary (710), tail (65), W20/T100/final,
paired improvement rate and CI. Secondary metrics never replace full MAE.

No test read unless a candidate has two fully trained seeds, consistent same-seed valid improvement
>=0.003, is not driven by valid:0172/extremes, one mechanism control shows it exceeds capacity/mean
field, and all choices are frozen. Historical test is already spent; any future read is descriptive,
not blind confirmation.

## Non-goals

- No S2 feature injection, no six-category relation means, no graph-level quadratic contraction, no
  rare-environment weighting, no ring-penalty special correction.
- No ESA+/GRIT/DBT reproduction, no RDKit target decomposition inputs, no external labels.
- No "information exists" claim is promoted to "prediction improves" without a trained comparison.
