# Optimized-Manifold Broad Frozen-State Sufficiency Screen

**Verdict: `INCONCLUSIVE_DO_NOT_BUILD_OOF`** (mandate Case C). No backbone was
trained; official test was never loaded.

Question: on the already fully-trained *optimized* compact-v4-hinge manifold,
do the frozen internal states (`h'_i`, `q_ij`) carry a low-complexity
predictive increment beyond the final 302D graph representation `R`?

Experiment: `experiments/luyin16/zinc_optimized_manifold_broad_state_screen.py`
Note: `notes/optimized_manifold_broad_state_screen.md`
Tests: `tests/test_optimized_manifold_broad_state_screen.py` (13 pass)

## Headline numbers (official valid, 1000 molecules)

| backbone | B0 frozen | B1 R-only (4135) | E broad state (4145) | Δstate = B1 − E |
|---:|---:|---:|---:|---:|
| seed0 | 0.146420 | 0.143495 | 0.143125 | **+0.000370** |
| seed1 | 0.149332 | 0.146420 | 0.145270 | **+0.001150** |

Init-0 mean Δstate **+0.00076** (both positive, far below the +0.003 advance
bar) → BORDERLINE. Mandated second adapter init: seed0 **+0.000179**, seed1
**−0.000513** (sign flip); combined mean **+0.00030 ≤ +0.0005**.

Common-input bulk safe (seed0 +0.00041, seed1 −0.00028; gate ≤ +0.002).
State branch alive (zero / within-molecule permutation sensitivity 0.026–0.121).
Gate 0 export integrity: all 28 gate instances pass on both seeds (13 per seed + 2 invariances) (true pre-head
`R` 302D; valid prediction anchor reproduced exactly 0.0).

**Decision:** do not build optimized OOF backbones, do not re-open the
optimized endpoint / covariance / triad witness tree. The next move is
paradigm-level, not another missing-local-statistic probe.

## Files

* `final_decision.json` — verdict, criteria, Q1–Q15, strategic implication
* `checkpoint_inventory.json`, `export_fingerprints.json` — reused optimized ckpts
* `export_integrity.json` — Gate 0 (all PASS)
* `adapter_protocol_lock.json` — frozen reader/protocol lock
* `split_manifest.json` — 7200 / 800 / 2000 official-train hash split
* `seed0_results.json`, `seed1_results.json` — init-0 per seed
* `valid_summary.csv`, `train_probe_summary.csv`, `second_init_results.csv`
* `common_input_bulk.json`, `adapter_dependency.json`
* `state_exports/optimized_frozen_state_export_v1_*.npz` (git-ignored)
* `figures/figure1_valid_mae.png`, `figures/figure2_delta_state.png`
