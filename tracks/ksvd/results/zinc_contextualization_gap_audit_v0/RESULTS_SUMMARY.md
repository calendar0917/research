# CGA-v0 — Contextualization Gap Audit — results summary

Protocol `zinc_contextualization_gap_audit_v0`; commit `c174ab36`.
`max_new_full_training_runs = 0`; `official_test_loaded = false`.

## Integrity

- seed0: MAE 0.13837561 vs recorded 0.13837560 (|Δ|=5.24e-09), A=True, C=True, B=True, D=True
- seed1: MAE 0.13344000 vs recorded 0.13343998 (|Δ|=2.42e-08), A=True, C=True, B=True, D=True

## Decision

`C_CONTEXTUAL_STATE_LOCALLY_COMPILABLE`

| seed | R_delta(H0) | ret.dDelta | mean J | P_local | P_context | context_gain | context_abs_gain | Spearman(C_h,gain) | Q4-Q1 gain |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.182 | 0.304 | 0.213 | 0.926 | 0.963 | 0.501 | 0.0370 | 0.026 | 0.0088 |
| 1 | 0.104 | 0.262 | 0.203 | 0.961 | 0.976 | 0.375 | 0.0145 | 0.065 | 0.0044 |

## Prototype context split (K=64)

| seed | prototypes used | weighted mean S_k | median S_k |
|---|---:|---:|---:|
| 0 | 64 | 0.185 | 0.174 |
| 1 | 64 | 0.128 | 0.112 |

Within-prototype (local-atom) delta variance is 13–18 % of the global delta
variance; 82–87 % of the recurrent shift is between local prototypes.

## Task link

| seed | Spearman(C_h,gain) | 95 % CI | Q4-Q1 | Q4-Q1 95 % CI | size-resid. Spearman |
|---|---:|---|---:|---|---:|
| 0 | 0.026 | [-0.036, 0.092] | +0.0088 | [-0.0119, 0.0322] | 0.018 [-0.047, 0.080] |
| 1 | 0.065 | [0.0005, 0.130] | +0.0044 | — | 0.042 [-0.024, 0.106] |

The task link is marginal at best and seed-unstable; seed 0's CI includes
zero. This is not the consistent, non-zero link a contextual-bottleneck claim
would require.

## Auxiliary B-Null seed0

Reproduced (0.12687264 vs recorded 0.12687270, |Δ|=5.6e-8). `R_delta` 0.257,
retention dDelta 0.397, 77.1 % exact h0 matches, `P_local` 0.864,
`P_context` 0.933, `context_abs_gain` 0.069. Same structure as the primary
pair: contextual splitting exists but is mostly locally predictable.

## Verdict

Case C — the contextual state is largely a locally determined transformation
with a small (1–4 % of shift variance) context-input-dependent remainder, and
that remainder is not consistently task-linked. This deprioritises further
local / pair / topology-cross dictionary designs. It does not authorize a
sixth dictionary; at most it authorizes one bounded, preregistered test of a
strictly-local two-update compiler, with a low ceiling because the context
residual is small and its task link is null.

## Figures

- `contextualization_vs_recurrent_gain_seed{0,1}.png`
- `gain_by_contextualization_quartile.png`
- `h0_distance_vs_delta_h_distance.png`
- `neighbour_retention_hist.png`
- `prototype_context_variance.png`

Analysis: `notes/zinc_contextualization_gap_audit_v0_analysis.md`.
Preregistration: `notes/zinc_contextualization_gap_audit_v0_preregistration.md`.

