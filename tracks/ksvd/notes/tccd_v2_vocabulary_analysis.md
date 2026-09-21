# TCCD-v2 — Vocabulary Analysis

Pre-registration: `notes/tccd_v2_preregistration.md`.
Formal checkpoint / analysis commit: `69a985a4ea3eb184c96c8ddad3857c4e87ee1dda`.
Device: remote A100 **GPU1**. Seed `0`. Internal-dev: 2,000 graphs / 46,126
patches. Official test: never loaded. K-SVD refit: **NO**.

## Usage diagnostics

The analysis uses the trained Prototype-REL best checkpoint. All 64
prototypes are active under the preregistered dead threshold `1e-6`.

| diagnostic | value |
|---|---:|
| active prototypes | 64 / 64 |
| dead prototypes | 0 |
| effective prototype count | 62.5755 |
| top-1 usage mass | 0.020194 |
| top-8 usage mass | 0.152634 |
| mean local assignment entropy | 1.843895 |
| global usage entropy | 4.147136 |
| normalized global usage entropy | 0.997175 |
| learned temperature | 0.094572 |

The preregistered vocabulary gate required at least `48/64` active prototypes
and rejected top-8 mass above `0.80`. Both conditions pass comfortably. The
vocabulary is not collapsed or dominated by a small prototype subset.

## Semantics and continuity diagnostics

Top activating real patches for all 64 prototypes are saved in
`results/tccd_v2/prototype_semantics_seed0.json`. Each entry contains graph id,
root atom, assignment probability, canonical local-subgraph slot order, local
atom types, shell labels and local bond types.

Mean top-50 canonical-key concentration across prototypes was `0.504375` vs
random baseline `0.061563`. This is report-only evidence that prototypes latch
onto repeated local environments rather than arbitrary isolated assignments.

Structural continuity remains diagnostic only. The internal-dev diagnostic was:

* x-space AUC `0.501875`;
* code/assignment-space AUC `0.580000`;
* graded-tail assignment AUC `0.570625`.

This does not restore the smooth canonical raw-patch manifold rejected by
TCCD-v0, but it is consistent with a discrete, task-organized vocabulary.

## Verdict

Vocabulary quality **PASS**. The round is allowed to run the one preregistered
full-data train-to-official-valid experiment because Prototype-REL internal
best MAE was `0.286244 <= 0.30`.
