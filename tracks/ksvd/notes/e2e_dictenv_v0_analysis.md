# E2E-DictEnv-v0 — analysis

Round `e2e_dictenv_v0` · study `zinc-context-gap` · protocol `e2e_dictenv_v0`
Frozen preregistration: `notes/e2e_dictenv_v0_preregistration.md`
Implementation: `notes/e2e_dictenv_v0_implementation.md`
Prior-artifact audit: `notes/e2e_dictenv_v0_prior_artifact_audit.md`

```text
FROZEN VERDICT:  E2E_DICTENV_MECHANISM_COLLAPSED
(frozen verdict table, case S1: Stage-1 mechanism gate FAIL)
```

## 1. What was executed

| step | result |
|---|---|
| prior-artifact audit | done |
| preregistration | frozen before any run |
| implementation | `e2e_dictenv_v0.py` + runner + 15 CPU tests (15/15 pass) |
| Gate 0 G0..G12 | **all PASS** |
| λ_rec calibration | 135.834928 (frozen, both arms) |
| Stage-1 mechanism smoke | **FAIL** (`atoms_active` 23/32 < 24/32) |
| formal Sparse / DenseTied arms | **not run** |
| zero-code / shuffle interventions | **not run** |
| dictionary health | **not run** |
| performance band / `G_sparse` | **not evaluated** |

The single failed item is the preregistered dictionary-coverage threshold in the
Stage-1 train-only smoke. Everything else in Gate 0 passed, and the two
mechanism samples the smoke does report (no dominance, effective count, rank)
passed.

## 2. The Stage-1 failure, precisely

Fixed smoke = first 512 official-train molecules, 3 epochs, Sparse arm, seed 0,
official valid never read, λ_rec frozen. The `atoms-active` gate is a
dictionary-coverage property; consistent with SDB-v0's own `used >= 24` gate
(measured over all 231,664 train atoms), it is evaluated here over the official
train distribution, with the 512-molecule value retained as a diagnostic.

| point | official-train coverage | 512-subset coverage | exact `s=8` |
|---|---:|---:|---:|
| initialization (frozen K-SVD `D`) | **25/32** | **24/32** | yes |
| after 3-epoch smoke | **23/32** | **23/32** | yes (`max l0 = 8`) |

So the gate resolves the same way under **either** measurement scope: the
mechanism is present at initialization and the short train-only smoke prunes
rare-atom support by two atoms, leaving 23/32 — one below the frozen threshold.

The failure is **deterministic**: two independent runs give identical atom
supports and identical gate outcomes (only float-reduction noise differs,
≤1e-6). It is **not** a loss/reconstruction collapse: train MAE decreases
(1.34700 → 1.24918), the reconstruction term decreases (0.00844 → 0.00532),
task gradients reach `D` (4.00 / 2.72 / 1.89), codes stay exactly sparse
(`max l0 = 8`), the largest atom carries only 12.5 % of assignments, the
effective atom count is 12.83 (≥ 8) and the environment coordinate keeps
effective rank 2.30 (> 1).

The rare atoms involved are genuinely rare (e.g. one atom is selected for a
single-digit number of the 231,664 train atoms). Under the calibrated
λ_rec = 135.8 the tied reconstruction pressure is strong, and a 12-step smoke
is enough to drop those marginal directions.

## 3. Reading — mechanism vs threshold

The round's *intent* was to test whether a task-coupled sparse pure-topology
dictionary can be the only fine-grained learned structural coordinate and
whether its value is specific against a parameter-identical dense tied
coordinate. That question was **not answered**, because the frozen pipeline
stopped at its own coverage gate.

Two distinct things must not be conflated:

1. **Mechanism status — healthy.** The dictionary is live, diverse and
   load-bearing at every point measured: ≥24/32 atoms active at init under both
   scopes, exact top-8 support, no dominance, effective count far above the
   floor, non-trivial environment rank, non-zero task gradient to `D`.
2. **Frozen gate — failed.** The preregistered threshold is `≥ 24/32 active`;
   the trained smoke model yields 23/32 on both scopes.

The preregistration, written before any run, did not specify whether
`atoms-active` is measured on the smoke-trained model or on the frozen
initialisation, nor whether it is measured over the 512-molecule smoke subset
or the full official-train distribution. The smoke trains for only 3 epochs on
512 of 10 000 molecules, whereas the threshold's provenance (SDB-v0) is a
full-train coverage statistic on the fitted dictionary. The strictest reading —
trained model on the 512 subset — fails (23/32); the init readings pass
(24/32, 25/32). Because the frozen gate is binding and the frozen verdict table
makes any Stage-1 FAIL a STOP (case S1), the round is closed negatively. **No
rescue was applied**: no K / s / IHT-step / LISTA / soft-dictionary /
dictionary-count / decoder / attention / LayerNorm / λ-sweep / extra-epoch /
seed change.

## 4. Frozen report questions Q1..Q8

| # | question | answer |
|---|---|---|
| Q1 | architecture purity (no MP / recurrence / writeback / FEC-S1 bypass)? | **PASS** (G1, G5, G6, G7, G8, G9; 15/15 CPU tests) |
| Q2 | dictionary centrality (`E`'s fine structure only from `IHT(D,phi)`)? | **PASS** (G4, G5; `coord_zero` removes all fine structure, chemistry still active) |
| Q3 | end-to-end coupling (task gradient to `D`; movement from K-SVD init)? | **PASS at init/smoke** (G3; `grad_D` > 0); formal-arm movement **not run** |
| Q4 | structural anchoring (sparse / reusable / reconstructive / non-collapsed)? | **PASS** (G2; exact top-8, `max l0 = 8`, no dominance, effective 12.8, rank 2.3) |
| Q5 | absolute band of `M_S`? | **not reached** (no formal arms) |
| Q6 | `M_D − M_S ≥ 0.003`? | **not reached** |
| Q7 | zero-code Δ ≥ 0.010; shuffle Δ ≥ 0.010? | **not reached** |
| Q8 | verdict from the six allowed verdicts? | **`E2E_DICTENV_MECHANISM_COLLAPSED`** (case S1) |

## 5. Historical context (anchors only, never a matched comparison)

FEC-S1 seed-0 soup 0.130422 / best 0.136783; S0 seed-0 soup 0.140794. The
parameter budget is identical to FEC-S1 minus 12 (66,158 vs 66,170). No
performance comparison is possible this round.

## 6. Recommendation (new round only — no rescue of this one)

The stop is a **coverage-threshold miss on a statistically under-powered
measurement**, not evidence that the dictionary mechanism is broken. A future
round is justified only under a **new preregistration** that fixes the gate
definition, for example:

* define `atoms-active` as a property of the *frozen initialisation* measured
  over the **full official-train** distribution (the SDB-v0 precedent), with the
  trained-model coverage reported as a *diagnostic*, not a hard stop; or
* keep a trained-model gate but measure it over the full official-train split
  and set the floor with a stated power rationale (e.g. also require
  `effective_atom_count ≥ 8`, `top1 ≤ 0.5`, `max l0 = s` — all of which already
  pass), so a single rare atom cannot trigger a false `MECHANISM_COLLAPSED`.

The substantive question — sparse tied dictionary vs parameter-identical dense
tied coordinate, with zero-code and assignment-shuffle interventions — is
unchanged and remains open.

## 7. Evidence index

* `results/e2e_dictenv_v0/artifact_identity.json`, `parameter_accounting.json`
* `results/e2e_dictenv_v0/correctness.json`
* `results/e2e_dictenv_v0/lambda_calibration.json`
* `results/e2e_dictenv_v0/smoke_gate.json`, `states/smoke_seed0.pt`
* `results/e2e_dictenv_v0/decision.json`, `REPORT.md`, `DECISION.md`
* `experiments/luyin16/e2e_dictenv_v0.py`,
  `experiments/luyin16/zinc_e2e_dictenv_v0.py`,
  `tests/test_e2e_dictenv_v0.py`
