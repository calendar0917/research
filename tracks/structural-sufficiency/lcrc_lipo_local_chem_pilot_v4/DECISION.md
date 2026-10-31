# Decision — LCRC Lipophilicity v4 local-chemistry pilot

**Date:** 2026-10-31
**Protocol:** `lcrc-lipo-local-chem-pilot-v4`
**Status:** **COMPLETE**
**Freeze SHA-256:** `1085d817465c6515a85e0d7e22e44d2cd43c5132e83a891c29c434dff6b7bf2e`
**Verdict:** **`NO_CLEAR_LOCAL_CHEM_GAIN`**
**GPU:** 0.044191 h of 0.20 h; one unconditional A run; conditional broadcast control **not run**.

## Result

Fold 1, seed 0, recipe R3. `ΔRMSE = RMSE(A-R3) − RMSE(v4)`; positive would favour the atom-chemistry model.

| Model | Fold | Seed | Train RMSE | Dev RMSE | Dev MAE | ΔRMSE vs A-R3 |
|---|---:|---:|---:|---:|---:|---:|
| A-R3 (v2 historical) | 1 | 0 | 0.269630 | **0.740343** | 0.546694 | reference |
| P1 `A_AtomChem4` | 1 | 0 | 0.263057 | 0.747044 | 0.540148 | **−0.006701** |
| P2 `A_BroadcastChem4` | 1 | 0 | — | — | — | **not run** |

95 % paired bootstrap (2,000 resamples): ΔRMSE `[−0.031255, +0.016516]`, ΔMAE `[−0.007027, +0.020309]`; molecule win rate 53.2 %; mean squared improvement −0.009967.

## Decision rule applied

Pre-registered P1 gate (fixed before any v4 GPU run):

- `ΔRMSE ≥ 0.03` — observed **−0.006701** → **not met**
- `ΔMAE ≥ 0` — observed **+0.006547** → met
- gain not tail-dependent — structurally false, because the full-dev delta is negative (after dropping the top 2 % historical residuals the delta becomes +0.006229, i.e. only the bulk improves)

The rule maps `ΔRMSE < 0.03` to **`NO_CLEAR_LOCAL_CHEM_GAIN`** and forbids the conditional run. `F1_A_BroadcastChem4_R3_s0` was therefore **not executed**: with no material gain there is no atom-correspondence effect to attribute, and spending GPU on it would not test the pre-registered hypothesis.

Two of the three verdict labels are therefore unused in this round:

| Label | Status |
|---|---|
| `LOCAL_CHEM_ALIGNMENT_PROMISING` | not applicable (P1 gate not met) |
| `CHEMICAL_FEATURE_INFORMATION_PROMISING` | not applicable (P1 gate not met) |
| `NO_CLEAR_LOCAL_CHEM_GAIN` | **selected** |

## Interpretation

Adding atom-resolved Crippen logP, TPSA contribution and HBD/HBA flags to the frozen LCRC-A atom encoder — with every original weight bit-identical to the historical seed-0 init, exactly +512 parameters, and zero-extra forward equivalence — did **not** improve Lipophilicity generalisation on Fold 1.

Together with v3's `NO_GLOBAL_DESCRIPTOR_GAIN`, the two placements of the same chemistry families both fail:

| Round | Placement | ΔRMSE vs A-R3 |
|---|---|---:|
| v3 `A_Chem12` (12 global descriptors → `rho`) | molecule level | −0.009628 |
| v3 `A_Chem13` (+ MolLogP → `rho`) | molecule level | −0.017082 |
| v4 `A_AtomChem4` (4 atom features → atom encoder) | atom level | −0.006701 |

The subgroup pattern is the substantive observation: P1 improves low-to-mid polarity molecules (TPSA ≤ 58.4: +0.0459; TPSA 81.2–97.5: +0.0418; HBD 1–2: ≈ +0.022) and degrades the hardest, most polar ones (TPSA > 116.3: −0.1025; HBD ≥ 4: −0.2137). A chemistry prior that helps the easy molecules and hurts the hard ones cannot improve a tail-dominated regression. This subgroup table is post-hoc and did not drive the decision.

Plausible mechanism (not tested here): atom51 already encodes element, aromaticity, formal charge and H count, and Crippen/TPSA are deterministic functions of that same local environment; the four columns re-encode existing information and mainly add capacity (2,807 gradient-clip events vs 2,797 historically).

## What this closes and what it does not

**Closes** for frozen LCRC-A under this protocol: the route "supply the standard RDKit lipophilicity / polar-surface / hydrogen-bond quantities" — at molecule level (v3) **and** at atom level (v4).

**Does not close**: learned chemical representations (3-D, ESP, pharmacophores), supervision changes (SSL, auxiliary chemistry heads), chemistry outside this 4-column family, or architectural change including iterative message passing. Each requires a new hypothesis, a new protocol and a new freeze.

Also not answerable with this dataset: anything conditioned on formal charge. 2,351 of 2,352 Fit molecules are neutral and the single charged molecule is in the Fold-1 train split, so the dev charged subgroup has n = 0.

## Constraints honoured

- New directory only: `tracks/structural-sufficiency/lcrc_lipo_local_chem_pilot_v4/`. v1, v2 and v3 artifacts are byte-identical (re-verified at freeze time and at GPU launch).
- No SSL, no Monitor, no Reserved, no official valid/test labels. Diagnostics and Features use Fit-internal structure only; no dev label entered any feature, threshold or normalisation statistic.
- Atom order proven index-by-index against the frozen raw-graph artifact, not assumed: 0 mismatches over 2,352 molecules / 63,231 atoms.
- Training was authorized in-session by the researcher (`analysis/gpu_authorization_v4.json`, mode `pre_authorized_by_researcher_in_session`), pinned to the freeze SHA, capped at 0.20 A100 h; actual use 0.044191 h.
- All remote execution went through `rr` (host `res`, pool `res-gpu0`). No manual `sbatch`, no `nohup`.

## Artifacts

- `REPORT.md`, `DECISION.md`
- `analysis/table_v4.csv`, `results_v4.json`, `paired_v4.json`, `decision_v4.json`, `subgroup_results_v4.json`, `summary_v4.json`
- `analysis/protocol_freeze_v4.json`, `gpu_authorization_v4.json`, `cpu_gates_v4.json`
- `analysis/atomchem_audit_v4.json`, `fold_alignment_v4.json`, `atom_norm_v4.json`, `forward_equivalence_v4.json`, `broadcast_control_v4.json`, `denorm_check_v4.json`, `descriptive_v4.json`, `temp_training_v4.json`
- `analysis/frozen_extras_v4.json` / `.npz`, `run_registry_v4.json`, `budget_v4.json`, `gpu_ledger_v4.json`
- `analysis/runs/F1_A_AtomChem4_R3_s0/{train.json,soup.pt,predictions.npz}`

The GPU run used host `res`, pool `res-gpu0`, node `a100-2`, A100-SXM4-40GB, driver `550.163.01`, torch `2.5.1+cu124` — the same regime as the historical comparator.

**Round closed. No follow-up seed, fold, feature combination, architecture variant, full-Fit retrain or Monitor evaluation is authorized by this decision.**
