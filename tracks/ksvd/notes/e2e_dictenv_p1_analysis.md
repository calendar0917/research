# E2E-DictEnv-P1 — analysis

Round `E2E-DictEnv-P1`; protocol `e2e_dictenv_p1`; study `zinc-context-gap`.
Remote `res` = `hxy@a100-2`, repo `/home/hxy/cy/research`, NVIDIA A100-SXM4-40GB
(GPU0 occupied by a foreign ~37 GB / ~91% job, so every formal run used GPU1).
Implementation commit `684675f`; terminal-test commit `e16ac15`. Local-first
implementation, committed revision deployed, then remote preflight → env cache →
correctness → GPU smoke → formal run → freeze → terminal test.

## Frozen verdict

```
P1_CLEAN_ENVIRONMENT_VIABLE_BUT_DICTIONARY_NOT_SPECIFIC
```

Pre-registered case D (`M_S` absolute + mechanism OK, but the matched DenseTied
control is not worse by the frozen materiality margin of 0.003). STOP: no seed 1,
no sweep, no rescue.

## What the round asked

Whether a *primitive-only, chemistry-free* end-to-end dictionary environment —
62-D zeroth-order primitive anchor `p_i`, role↔chemistry binding, `z_i ∈ R^638`
SiLU decoder, 15-D pure-topology pair relation, the exact SDB-v0 K-SVD `D`
(65×32, sha256 `925d573a5808…`), K=32, s=8, 10 tied-IHT steps, `λ_rec`
33.95873017865987 — reaches the strong absolute band *and* keeps the dictionary
genuinely load-bearing, now that every coarse-chemistry / FEC-S0 bypass is
deleted. Total parameters 97,865 (local 82,762 + backend 15,103), inside the
pre-registered 90,000–105,000 budget.

Two mechanisms were the point: the anchor (`p_i`) has no chemistry lookup, so the
only place learned chemistry can enter is the dictionary code; and the
intervention gates were written as the pre-registered primary criterion, ahead of
the absolute band.

## Correctness and mechanism prerequisites

* Correctness G0–G14: **15/15 PASS** (identity φ65, dictionary identity, exact
  sparsity, gradient-to-`D`, chemistry-purity, forbidden-local-descriptors,
  primitive anchor, edge role symmetry, pair-relation chemistry blocker,
  environment freeze, no pair→centre, once-only composition, relabel
  invariance 2.4e-7, parameter budget 97,865, official-test blocker).
* Targeted CPU suites: P1 11/11 (`tests/test_e2e_dictenv_p1.py`); combined with
  the v0/T1 suites 41 passed.
* GPU smoke (512 molecules, 3 epochs): all 12 mechanic gates PASS — train MAE
  falls, `||dL/dD||` non-zero, node- and edge-binding gradients non-zero, exact
  top-8, active-atom retention 0.958, effective atoms 12.9, environment rank
  participation-ratio 3.62 ≥ 3.
* Environment cache audited bit-for-bit against v0 counts: train 10,000 mols /
  231,664 nodes / 1,418,500 node-occ / 1,232,844 bond-occ; valid 1,000 / 23,083
  / 141,287 / 122,934.

## Formal Sparse seed 0 (primary)

Frozen horizon 240 epochs, Adam lr 1e-3, wd 1e-5, batch 128, clip 5, no
scheduler, no early stop; seed 0; fixed Top-5 valid-MAE soup.

| quantity | value |
|---|---|
| best valid MAE | 0.136478 @ epoch 237 |
| **soup valid MAE `M_S`** | **0.131975** (band `strong`, ≤ 0.135) |
| soup members | [206, 217, 219, 236, 237] |
| soup valid reconstruction | 1.31e-4 |
| wall clock | 1555.7 s |
| peak GPU memory | 187.5 MB |
| parameters | 97,865 |

Absolute anchors (context only; different architectures/horizons):

| anchor | soup valid | `M_S` − anchor |
|---|---|---|
| E2E-DictEnv-v0 (2026-09-23) | 0.145508 | **−0.013533** |
| FEC-S1 static composition | 0.130422 | +0.001553 |
| E2E-DictEnv-T1 (tuned) | 0.125765 | +0.006210 |

So the primitive-only clean environment is **strong** (well inside ≤ 0.135),
materially better than the v0 dictionary environment (+0.0135) and essentially
level with the FEC-S1 composition anchor, but still 0.0062 behind the tuned T1
interface configuration.

## Mechanism (primary criterion) and health

| intervention | clean | intervened | gate | result |
|---|---|---|---|---|
| zero the dictionary coordinate | 0.131975 | `M_zero` 0.479091 | `G_zero ≥ 0.030` | **+0.347116 PASS** |
| node-assignment shuffle (5 perms) | 0.131975 | 0.145099 | `G_node ≥ 0.010` | **+0.013124 PASS** |
| edge-binding shuffle (5 perms) | 0.131975 | 0.191503 | report only | +0.059528 |
| node+edge combined shuffle | 0.131975 | 0.204974 | `G_all ≥ 0.015` | **+0.072999 PASS** |

This is the decisive difference from T1. Deleting the coarse-chemistry bypass and
exposing the `z_i` path only through the bound dictionary code restores genuine
load-bearing: zero-code now costs **+0.347** MAE (T1: +0.0179; v0: +0.8548) and
the node-assignment shuffle **+0.0131** clears the gate (T1 missed it at
+0.0076). Mean prediction shift on zero-code is 0.437 (max 6.65), i.e. the
dictionary is not merely decoratively present.

Dictionary health: **PASS on every gate** — 27/32 active atoms, effective atom
count 16.09, top-1 share 0.125, exact top-8 fraction 1.0, valid reconstruction
1.31e-4, `||dL/dD||` = 0.094, effective rank 10.30, Frobenius movement from the
K-SVD init 5.07, train/valid usage Spearman 0.999.

GO gate (all of absolute-viable, `G_zero`, `G_node`, `G_all`, health): **GO**.

## Matched DenseTied control and specificity

Same architecture and parameter count; the only change is the coding operator
(`phi @ Dbar` dense tied instead of tied-IHT top-8).

| seed 0 | soup valid | best valid | wall |
|---|---|---|---|
| Sparse `M_S` | 0.131975 | 0.136478 @237 | 1555.7 s |
| DenseTied `M_D` | 0.134534 | 0.137536 @224 | 1345.0 s |

`G_specific = M_D − M_S = 0.002559` against the frozen gate 0.003 → **FAIL**.
The sparse dictionary is *directionally* better than the matched dense tied
coordinate (unlike FEC-D1, where dense won), but the margin is below the
materiality threshold and is within the scale of GPU non-determinism.

Consequently the round stops at case D: **seed 1 was not authorised or run**, no
second seed exists for either arm.

## Terminal official-test read (reporting only)

Loaded exactly once, after the architecture freeze and after `decision.json` was
recorded; it cannot change any frozen decision. `official_test_unlock.json`
asserts the pre-test freeze.

| object | test MAE |
|---|---|
| Sparse seed0 soup | 0.107593 |
| DenseTied seed0 soup | **0.101661** |

The test **reverses** the valid-side direction: the dense tied coordinate is
0.0059 MAE *better* on test. Mechanistic generalisation holds strongly on test —
`G_zero_test` 0.3488, `G_node_test` 0.011293, `G_edge_test` 0.056375,
`G_all_test` 0.071314 — mirroring valid almost exactly.

## Reading

1. **Architectural conclusion (positive).** The P1 prototype works as designed as
   a *clean* environment: with every coarse-chemistry pathway deleted and the
   decoder input routed through the bound dictionary code, the dictionary is
   strongly load-bearing (`G_zero` +0.347, `G_node` +0.0131, `G_all` +0.0730) and
   the absolute band is strong (0.131975). The T1 mechanism loss is therefore
   attributable to the coarse-chemistry interface, not to the dictionary core —
   and a dictionary-core environment *can* be both accurate and load-bearing
   without message passing or recurrence.
2. **Specificity conclusion (negative).** The accuracy is **not** attributable to
   sparsity / dictionary specificity: the parameter-identical dense tied
   coordinate reaches 0.134534 on valid (0.0026 behind) and 0.101661 on test
   (0.0059 ahead). The bound *code* is load-bearing; its *sparse discrete*
   structure is not demonstrated to be. This is the same family of result as
   FEC-D1 ("local-binding-supported dictionary not specific"), now at the
   end-to-end environment level and with a stronger load-bearing signal.
3. **Honest precedence.** The pre-registered order (absolute + mechanism, then
   specificity) was followed exactly; the round must not be reported as a
   dictionary-specificity success, and the test reversal only strengthens the
   not-specific reading.

## Caveats

* Single tuning seed (0); seed 1 not purchased because the specificity gate
  failed. `G_specific` 0.00256 vs the 0.003 gate is close in absolute terms, but
  the test-side reversal makes the not-specific conclusion robust rather than
  marginal.
* GPU floating-point is not bit-deterministic; run-to-run valid differences are
  at the ~1e-3 scale, which is the order of the specific margin.
* The absolute comparison against T1/v0/FEC-S1 crosses architectures and
  horizons and is reported as anchor context only, never as a decision input.
* Official test is not project-wide pristine (historical unrelated reads); this
  round's read remains reporting-only and post-freeze.

## Artifacts (this round)

* results: `tracks/ksvd/results/e2e_dictenv_p1/` — `artifact_identity.json`,
  `parameter_accounting.json`, `env_cache.json`, `correctness.json`,
  `smoke_gate.json`, `sparse_seed0.json` + curve, `dense_seed0.json` + curve,
  `architecture_freeze.json`, `gate_sparse_seed0.json`,
  `mechanism_{zero,node_shuffle,edge_shuffle,all_shuffle}.json`,
  `dictionary_health.json`, `specificity_seed0.json`, `decision.json`,
  `official_test_unlock.json`, `official_test_results.json`, `REPORT.md`,
  `DECISION.md`.
* notes: `e2e_dictenv_p1_prior_artifact_audit.md`,
  `e2e_dictenv_p1_preregistration.md`, `e2e_dictenv_p1_implementation.md`,
  this file, `e2e_dictenv_p1_test_read.md`.
* records: `claim-e2e-dictenv-p1-clean-load-bearing-but-not-specific-20260924`,
  `decision-e2e-dictenv-p1-stop-not-specific-20260924`.
