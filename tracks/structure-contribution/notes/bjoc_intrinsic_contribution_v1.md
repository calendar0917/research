# bjoc-intrinsic-contribution-feasibility-v1 — design and pre-registration record

Round `cscl-bjoc-intrinsic-contribution-feasibility-v1` (track `structure-contribution`).
Pre-registration: `configs/bjoc_intrinsic_contribution_feasibility_v1.yaml` (committed before
any formal trajectory; split row hashes, recipe, gate thresholds, decision vocabulary and
trajectory plan are frozen there).
Audit: `code/audit_predictive_capacity_v1.py` (Phase A, CPU).
Model: `code/bjoc_intrinsic_contribution_v1.py` (IJCM). Synthetic audits: `code/ijcm_synthetic_v1.py`.
Driver: `code/run_bjoc_intrinsic_contribution_v1.py`. Tests:
`tests/test_bjoc_intrinsic_contribution_v1.py` (24 tests, all pass).

## Question

BJOC-Null is the strongest predictor in this track (official historical seed-mean 0.092233,
user target band 0.06–0.07 — still unreached). Its junction content is learned and pooled, so
it cannot answer "what did this junction contribute to this molecule's prediction?".

> On the same frozen 8000/1000/1000 canonical-SMILES grouped internal split, with the same
> encoders and the same last-5-soup recipe, what is the measured prediction cost of replacing
> the joint global reader by a strictly additive sum of shared per-junction scalar
> contributions, and is the resulting contribution vector identifiable enough to report as an
> intrinsic attribution?

This is a **feasibility** round about the predictive cost of an interpretable head, not a
performance round. Prediction loss is a hard constraint: if the additive head degrades
generalisation beyond the frozen stop-loss, the next round must go back to prediction research
and attribution is parked. No interaction terms, no optimiser changes, no extra supervision,
no teacher predictions may be used to close the gap.

## The two arms

Identical encoders, identical features, identical recipe, only the head differs:

* **B — BJOC-Null** (frozen reference): learned junction content `w`, static incidence
  grouping, bond composer, joint global reader. **413,505** parameters.
* **S — IJCM** (intrinsic junction contribution model): each atom / bond / junction emits one
  scalar; the molecule prediction is exactly

  ```text
  y_hat = bias + Σ_atoms α(type, context) + Σ_bonds β(type, context)
              + Σ_junctions ν(type, context) + Σ_features η_k
  ```

  where `α = α_type + α_ctx(own instance)`, etc. Every context head sees only features of the
  instance (or, for a junction, its centre atom and its two incident bonds). There is **no**
  operation combining two objects. Type baselines are zero-initialised; context heads are the
  only trainable correction; `η_k` is a scalar per global feature.

Parameter budget: S = 409,922 (0.9913 × B), within the frozen 5% tolerance. The expression
inventory contains no opaque reader / composer: the conservation identity is the guard.

## Conservation and gauge (Phase C, CPU gates)

`pred == bias + Σ α + Σ β + Σ ν + Σ η` is checked numerically with and without the fit-set
gauge (max gap < 1e-5; see `correctness_gates.json`). The gauge moves, per branch and type,
the fit-set mean of the context correction into the type baseline:

```text
α_type' = α_type + m_t,   α_ctx' = α_ctx − m_t,   m_t = mean_fit α_ctx over type t
```

so per-type fit-set canonical context means are 0 and per-instance totals are unchanged.
Reported contributions are canonical. Atom relabelling, batch recomposition, save/load replay,
builder determinism and the vocabulary/UNK backoff are gated numerically before any GPU run.

## Known answer from Phase A (predictive capacity audit)

On the already-exposed split-A locked rows (seed-mean): MAE 0.1433; ordinary molecules
(y > −3, n = 931) carry 59.8 % of the error mass at MAE 0.0920; the tail (n = 69) has MAE
0.8353; the top-10 rows carry 37.4 % of the error mass. Per-molecule seed-0/seed-1 errors
overlap only partly (top-100 overlap 60 %, Spearman 0.47), so paired seeds are required.
A label audit of the only duplicate-SMILES group with inconsistent targets found different
input tensors, so `IRREDUCIBLE_ERROR_NOT_ESTABLISHED`. The ordinary-molecule floor (~0.092)
means the 0.06–0.07 band needs broad improvement, not tail handling; attribution work cannot
be sold as a shortcut to that band.

## Synthetic known-truth audits (Phase D, CPU)

* **A — identifiability**: random chains/trees with a known additive rule. The type-count
  design block is full rank on the 1000-molecule fit set; type effects recover at r ≈ 0.98
  (atom/junction) and r ≈ 0.82–0.92 (bond); context recovery is weaker for bonds (branch
  transfer), which is exactly the gauge caveat.
* **B — collinearity trap**: two junction types that co-occur in *every* molecule (true A=1.5,
  B=0). The additive model necessarily splits the effect non-identifiably; the split is
  seed-unstable. The limitation is documented, not rescued.
* **C — necessary non-additivity**: target = 1.2 × (adjacent A/B junction pairs). The
  interaction is orthogonal to the additive count design; an additive head cannot represent
  it. The residual gap vs an interaction oracle is measured and reported; no interaction is
  added.

## Read protocol and stop-loss (frozen)

1. Splits A and B (canonical-SMILES groups, salted a priori, 8000/1000/1000).
2. Fit-only normalisation and type vocabularies (labels never read).
3. CPU gates must pass.
4. Freeze manifest (config + split/prep/gate artifact hashes + code hashes).
5. Train **pair-a-s0** = {B/A/s0, S/A/s0}; decide on **monitor** rows only.
6. Stop-loss: monitor seed-0 Δ(S−B) ≤ +0.005 buys seed 1; +0.005 < Δ ≤ +0.008 buys but
   flags as costly; Δ > +0.008 stops here.
7. Freeze predictions, then the **single** locked read of split A; split B is purchased only
   if the locked seed-mean Δ ≤ +0.005 (Δ ≥ +0.010 is clearly unacceptable).
8. `mechanism` diagnostics from the frozen checkpoints; `report`; decision label.

`decide` reads train JSONs only; `predict` hashes predictions before the locked read;
`score` is the only command that reads locked labels. Official valid/test are never loaded.

## Decision vocabulary

`INTRINSIC_CONTRIBUTION_FEASIBLE`, `ATTRIBUTION_WORKS_BUT_PREDICTIVE_COST_TOO_HIGH`,
`ADDITIVE_HEAD_UNDERFITS`, `CONTRIBUTIONS_NOT_IDENTIFIABLE`, `INCONCLUSIVE_OR_BLOCKED`.

## Budget

≤ 465 min wall clock (35 min reserved for pull/verify/report/git), ≤ 1.5 GPU-hours,
≤ 8 formal trajectories, concurrency 1. Bans are listed in the config (interaction terms,
GNN/attention, target-component supervision, teacher predictions, official splits, tuning
after the locked read, auto-training the full-10k final candidate).
