# Structure-route resolution, 8 h sprint

**One-page answer (details below).**

- **A. Energy constraint vs Möbius vs combination — `INTERNAL_GAIN_ONLY` / `SCREEN_ONLY`.**
  There is a large Möbius-vs-RAW gain only for the energy response and only at D0/s0:
  E_RAW 0.7177 → E_MOB 0.5988 (gain **+0.1189**, 95% CI [+0.0454, +0.2021]).
  The same decomposition **hurts** the free response on both seeds:
  F_RAW 0.6074/0.5555 vs F_MOB 0.6189/0.6009 (F_RAW−F_MOB s0 −0.0115, s1 −0.0454; mean gain **−0.0284**, CI [−0.0611, +0.0023]).
  E/F ranking flips across seeds (C3 −0.0030, CI includes 0), and every A arm is worse than
  R2 on the D0 two-seed mean (E_MOB −0.0525, F_MOB −0.0556, F_RAW −0.0271). The large C1
  contrast is single-seed (E_RAW was run only at s0), so it is an **energy-specific internal
  signal**, not a robust route to promote. BIG helps F (+0.0433 s0, +0.0066 s1) but hurts E
  (−0.0281 s0) and still does not beat R2.
- **B. Message range beyond radius 2 — `SEED_OR_SPLIT_SENSITIVE` / `SCREEN_ONLY`.**
  FULL vs radius-2 EGO2: 0.5147 vs 0.5161 at D0/s0 (gap **0.0014** << δ=0.01),
  0.5119 vs 0.5353 at D0/s1, 0.5803 vs 0.5563 at D1/s0, 0.5517 vs 0.5523 at the D1/s1
  conflict pair. No stable direction or useful magnitude. The fixed EGO3 trigger did not fire.
  Radius-2 egos carry the signal; there is no evidence that deeper message range is needed.
- **C. Transform × context response — `SCREEN_ONLY` / not promoted.**
  The transform kernel clearly beats lookup (K_T 0.5446/0.3604 vs EXACT_MEAN 0.6858/0.5063
  on D0/D1). Conditional context does not add robustly: C5 gain over K_T is **+0.0531**
  [−0.0070, +0.1180] on D0 and **−0.0161** [−0.0781, +0.0465] on D1; D0 strict-new-core gain
  is +0.0491 with CI [−0.0092, +0.1081]. The best whole-molecule neural difference is better
  on both splits (D0/D1 G_EGO2 0.3480/0.3441 vs K_CONDITIONAL_R2 0.4915/0.3769).
  Conditional response is an internal D0 gain only, not competitive with a simple model
  difference.

---

## 1. Scope, data and provenance

- Data: `ogbg-mollipo` **official Train only** (3,360 molecules). D0 and D1 are the frozen
  2,856 Fit / 504 Dev splits; each run recomputes its own Fit target z-score, graph/atom
  scalar normalisation and categorical support mask. Official Valid/Test labels were never
  read. Checkpoint selection is best Dev RMSE (earliest epoch on ties).
- D0 identity/fit/eval SHA-256: `60ac13c0…/7edd3c68…/eacce8ac…`.
  D1 identity/fit/eval: `60ac13c0…/b9f74030…/342c1de0…`.
- First commit with protocol/code/catalogue: `52f093d2`; formulation frozen in
  `PROTOCOL.md` (`sha256` recorded in `analysis/queue_lock.json`). D1 queue was locked in
  `analysis/queue_lock.json` (`fee69b44`) before any D1 run.
- 23/23 implementation checks pass (`analysis/checks.json`), including RICE checkpoint
  replay, Möbius/Φ/κ identities, 4-star counterexample, FreeResponse positive control,
  E/F initial-state equality, BIG-copy rule, relabelling invariance and G range controls.
- 21 best checkpoints were independently reloaded and replayed; all Dev/Fit RMSE and stored
  predictions agree within CPU/GPU numerical tolerance (`analysis/replay.json`, `all_ok=true`;
  max stored-prediction delta 6e-4 for the RICE model, <2.5e-5 for the rest).

## 2. Whole-molecule results (Dev RMSE, lower is better)

| arm | D0 s0 | D0 s1 | D0 mean | D1 s0 |
|---|---:|---:|---:|---:|
| R2_REF (416,177) | 0.54768 | 0.56103 | 0.55436 | 0.56380 |
| E_RAW (129,198) | 0.71770 | — | — | — |
| E_MOB (129,198) | 0.59884 | 0.61493 | 0.60688 | 0.64266 |
| F_RAW (129,878) | 0.60745 | 0.55552 | 0.58148 | 0.58104 |
| F_MOB (129,878) | 0.61893 | 0.60090 | 0.60992 | 0.56013 |
| E_MOB_BIG (395,438) | 0.62691 | — | — | — |
| F_MOB_BIG (396,118) | 0.57560 | 0.59432 | 0.58496 | 0.60450 |
| G_EGO2_5 (413,575) | 0.51611 | 0.53528 | 0.52569 | 0.55633 |
| G_FULL5 (413,575) | 0.51469 | 0.51190 | 0.51330 | 0.58027 |

R2 reproduction is numerically stable: D0/s0 0.54768 vs historical 0.55056; D1/s0 0.56380
vs historical 0.56440. E_MOB/s0 reproduces the historical RICE result (0.59884 vs 0.59937,
same best epoch 114). These are the only historical anchors used.

## 3. Five main contrasts

Gain is `error(reference) − error(candidate)`; positive means the candidate has lower error.
Intervals are 5,000 paired scaffold-cluster bootstrap draws (seed 1703), using the same
resample indices for all models within a split.

| # | contrast | split/seeds | gain | 95% CI | reading |
|---|---|---|---:|---|---|
| C1 | E_RAW − E_MOB | D0/s0 only | **+0.1189** | [+0.0454, +0.2021] | Möbius helps E strongly, single seed |
| C2 | F_RAW − F_MOB | D0/s0+s1 | **−0.0284** | [−0.0611, +0.0023] | Möbius hurts F on both seeds |
| C3 | E_MOB − F_MOB | D0/s0+s1 | −0.0030 | [−0.0428, +0.0380] | E/F ranking not stable |
| C4 | EGO2 − FULL | D0/s0+s1 | +0.0124 | [−0.0140, +0.0411] | FULL slightly better on D0 mean, not stable |
| C5 | K_T − K_TC | D0 / D1 | +0.0531 / −0.0161 | [−0.0070,+0.1180] / [−0.0781,+0.0465] | conditional context D0-only |
| C2−C1 | interaction | D0/s0 only | **−0.1304** | [−0.2174, −0.0482] | Möbius effect depends on parameterisation |

Supporting arms vs R2 (D0 two-seed mean): E_MOB **−0.0525** [−0.1018, −0.0059],
F_MOB **−0.0556** [−0.0964, −0.0138], F_RAW −0.0271 [−0.0726, +0.0168],
EGO2 +0.0287 [−0.0060, +0.0620], FULL **+0.0411** [+0.0089, +0.0730]. On D1/s0, E_MOB is
−0.0789, F_MOB +0.0037, F_RAW −0.0172, F_BIG −0.0407, EGO2 +0.0075, FULL −0.0165.
The F_MOB D1 sign change (+0.0037 vs D0 mean −0.0556) was a detected cross-split conflict;
because P_SCOPE had higher fixed priority, the single conflict package was spent on
EGO2/FULL at D1/s1, not on F_MOB/R2. That conflict therefore remains unresolved.

## 4. Route B detail

FULL and EGO2 have identical parameters, recipe and readout; the only difference is whether
the five-layer network runs on the whole molecule or on per-atom radius-2 ego graphs. D0/s0
was the pre-specified EGO3 gate: the gap 0.0014 is far below δ=0.01, so EGO3 was skipped.
The D0/s1 package (triggered because EGO2 was >δ better than R2) and the D1 pair show the
FULL−EGO2 difference changing sign. Conclusion: **no support for needing message range
beyond 2**; EGO2 is the simpler and equally good scope. The GNN arms being competitive with
R2 is an architecture/readout observation, not evidence for deeper range.

## 5. Route C detail

Support is limited: D0 Dev–Dev 276 pairs / 202 molecules / 62 pair-graph components;
D1 184 / 162 / 53. D0 has 68 unseen-transform pairs and 183 strict-new-core pairs
(49 components); D1 has 30 and 141 (45 components).

D0 Dev–Dev component-equal MAE: ZERO 0.7303, EXACT_MEAN 0.6858, K_TRANSFORM 0.5446,
K_CONDITIONAL_R2 **0.4915**, K_CONTEXT_ID 0.5524, K_CONDITIONAL_FULL 0.5177.
D1: 0.5387, 0.5063, 0.3604, 0.3769, 0.3601, 0.3674. The conditional kernel is best only
on D0; on D1 it is worse than both K_T and K_CONTEXT_ID. D0 unseen-transform K_TC 0.4814
vs K_T 0.5482 shows a real internal conditional signal, but it does not replicate on D1
(unseen: 0.555 vs 0.553; K_CONTEXT_ID 0.480).

Neural Δf on the same frozen pairs (component-equal MAE): D0 best G_EGO2_S0 0.3480,
R2_S0 0.3693, F_RAW_S1 0.3917; D1 best G_EGO2_S0 0.3441, F_MOB_S0 0.3740, R2_S0 0.3978.
K_TC (0.4915/0.3769) is not competitive with the best simple model difference on either
split. The conditional-response route should be deprioritised unless a larger, genuinely
held-out support is available.

## 6. Branch execution and budget

- BIG trigger **true** → E_MOB_BIG and F_MOB_BIG run at D0/s0. E_BIG is worse than E_SMALL;
  F_BIG is better than F_SMALL but still worse than R2. P_HEAD selected F and added
  F_MOB_BIG D0/s1; no E_BIG s1.
- EGO3 trigger **false** (0.0014 < 0.01) → skipped.
- P_SCOPE (EGO2 > R2 by 0.0316 at D0/s0) → EGO2/FULL D0/s1, then D1/s0.
- P_BASIS (|F_RAW−F_MOB| = 0.0115 at D0/s0) → F_RAW D0/s1, then D1/s0.
- D1 base (R2/E_MOB/F_MOB) + P_SCOPE + P_BASIS + P_HEAD; all 7 locked D1 runs completed.
- Stage V selected **P_SCOPE** as the single conflict package (FULL−EGO2 changed from
  +0.0124 on D0 to −0.0239 on D1/s0); the D1/s1 pair gives FULL 0.55172 vs EGO2 0.55233,
  i.e. no meaningful scope difference. No second conflict package.
- 25 formal GPU runs, **4.74 GPU-hours** (target ≤14, hard ≤16); wall-clock ~4.2 h at
  report time (target 7.5 h, hard 8 h). No training was started after T0+6:30; no training
  was running at T0+7:15. `analysis/budget.json` is the ledger.

## 7. Limitations / honest caveats

- Dev labels choose the best checkpoint; this is not an unbiased final test. D1 is a second
  split of the same official Train pool, not an independent blind set.
- C1 is single-seed by design (E_RAW was not in the seed-1 package), so the largest A effect
  is not a cross-seed estimate. The C2−C1 interaction is therefore D0/s0-only.
- The E_MOB/F_MOB seed spread is large (E_MOB 0.599→0.615; F_MOB 0.619→0.601), and the
  family ranking flips. Do not report a single-seed winner.
- Route C support is small (62/53 components) and the D0/D1 direction reverses; the
  bootstrap intervals are descriptive, not a claim of adequate power.
- The GNN diagnostic uses a different architecture/readout from R2; its competitiveness is
  not a controlled test of message range alone. FULL vs EGO2 is the controlled comparison.
- Early manifests recorded `git_commit`/`git_dirty` at manifest-write time; authoritative
  per-run rr metadata is in `results/*/rr_provenance.json` (all tracked-clean). Training code
  was unchanged across those commits (only analysis/scheduler/provenance changed).

## 8. Reproduce

```bash
cd tracks/structural-sufficiency/structure_route_resolution_8h_v1
uv run --project /home/calendar/code/research python code/sr_checks.py
uv run --project /home/calendar/code/research python code/sr_analyze.py --boot 5000
uv run --project /home/calendar/code/research python code/sr_gates.py --stage final
uv run --project /home/calendar/code/research python code/sr_replay.py
uv run --project /home/calendar/code/research python code/sr_figures.py
uv run --project /home/calendar/code/research python code/sr_budget.py
```

Formal runs were launched only through `rr`; scheduler and queue state are in
`code/sr_schedule.py`, `analysis/queue_state.json`, `analysis/queue_lock.json` and
`analysis/branch_log.jsonl`. Figures: `analysis/figures/route_{a,b,c}_*.png`.
