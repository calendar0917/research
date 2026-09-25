# E2E-DictEnv-A2-Confirm — preregistration (frozen 2026-09-25)

Round `E2E-DictEnv-A2-Confirm`, protocol `e2e_dictenv_a2_confirm`.
Written **before** any confirmation training ran.  Historical verdicts are not
modified by this round:

```
E2E-DictEnv-A2       COMPUTE-BUDGET-TRUNCATED
E2E-DictEnv-A2-Lite  NO_ATTRIBUTED_CODE_FORMATION_SIGNAL_AT_32D
```

## 0. Motivation and prior evidence (not a blind preregistration)

The lite compute-budget screen (160 epochs, amendment rules `457c276`, run
`4f184dd`) produced a **positive but sub-material and direction-unstable**
paired gap under the frozen exact-OMP code, and both arms were still improving
at the horizon:

| item | value |
|---|---|
| `ATTR-REAL-OMP` 160 soup | `0.15437116196932038` (best `0.16085957256139954` @158) |
| `ATTR-INDEP-OMP` 160 soup | `0.15993232336913935` (best `0.16623578738694778` @145) |
| `G_pair_screen` = INDEP − REAL | `+0.005561161399818965` |
| lite late window 121–160 | positive fraction `0.65`, mean `+0.003818`, best `+0.005376` |
| frozen lite strong bar | `≥ 0.006` **and** stable ⇒ not met ⇒ label `NO_ATTRIBUTED_CODE_FORMATION_SIGNAL_AT_32D` |
| dense rank-32 control (diagnostic only) | `G_pair_PCA = −0.0014047007597982886` ⇒ no compression bottleneck |

The in-flight parent `E2E-DictEnv-A2` run completed one full 320-epoch frozen-OMP
arm before truncation and is reused here as the finished practical baseline:

| item | value |
|---|---|
| `TOPO-OMP` 320 soup (frozen parent run) | `0.12681294702464949` |
| `TOPO-OMP` 320 best | `0.13136472144449363` @ epoch 314 |
| source | `tracks/ksvd/results/e2e_dictenv_a2/omp_screen_topo.json` (never retrained) |

This round answers exactly one question with the parent's own horizon:
**under the frozen exact-OMP representation, is `ATTR-REAL` better than
`ATTR-INDEP` at 320 epochs, and is REAL competitive with the completed TOPO
baseline?**  No representation redesign, no PCA, no IHT/E2E.

## 1. Frozen reuse (identity-verified, nothing re-derived)

Reused bit-identically from the identity-verified parent artifacts
(`results/e2e_dictenv_a2/artifact_identity.json`, `all_passed`, 26/26,
`git_commit 24d528635fab49c082115d90592cb7d5938eeb37`,
`preregistration_commit 1813f53`): 433-D REAL/INDEP objects, train-only scaler,
K-SVD `K=32`/`s=8` dictionaries (sha256-f32 REAL `c1cafb08…`, INDEP
`400821ee…`), frozen exact-OMP train/valid codes, matched H1 architecture and
trainer, matched initialisation.

Forbidden in this round: refitting K-SVD or the scaler, rebuilding the 433-D
objects, recomputing or altering the OMP codes, changing `K`/`s`/`λ`, any
architecture or hyper-parameter change, any new feature.  A hash/provenance
mismatch ⇒ `ARTIFACT_IDENTITY_FAILURE`, stop, no "close enough" rebuild.

## 2. Frozen protocol (identical for both arms)

```
seed = 0            horizon = 320        H1 model, unchanged
initialisation      = parent matched init (REAL reference state)
batch order         = SEED + 91011 (train) / SEED + 91012 (eval)
optimizer           = Adam(lr 0.001, weight_decay 1e-05), grad clip 5.0
lambda convention   = 33.95873017865987 (unchanged)
soup                = Top-5 valid MAE, unchanged
validation schedule = per epoch on official valid
code interface      = frozen exact-OMP codes (REAL alpha vs INDEP alpha only)
```

Only difference between arms: REAL vs INDEP code.  `TOPO` is **not** retrained.
CUDA on physical **GPU1 only**, one CUDA process at a time, no DDP, no
multi-GPU, GPU0 never referenced.  Official **test never loaded**.

## 3. Continuation regime (checked first, no mixing)

Before training, the round inspects whether the completed 160-epoch lite
segments can be continued exactly (model, optimizer, scheduler, epoch, all RNG
states, loader/batch-order state, and Top-5/soup bookkeeping for the full 1–320
trajectory).  Rule: exact resume is used **only if both arms can be proven
exactly resumable**; otherwise both arms restart from scratch as fresh matched
320-epoch runs.  Mixing (`REAL` resumed / `INDEP` fresh, or the reverse) is
prohibited.  If exact-resume infrastructure would require substantial new
machinery, fresh matched 320 is preferred.  The outcome and its evidence are
recorded in `continuation_mode.json`.

## 4. Primary quantity and threshold (frozen, single seed)

```
G_pair_320 = MAE(INDEP-OMP-320 soup) - MAE(REAL-OMP-320 soup)
material bar: G_pair_320 >= 0.003    (parent A2 frozen bar, unchanged)
```

Practical comparison with the completed TOPO baseline:

```
Delta_vs_TOPO = MAE(REAL-OMP-320 soup) - MAE(TOPO-OMP-320 soup)
positive = REAL worse than TOPO; negative = REAL better than TOPO
tolerance: 0.003
```

## 5. Frozen decision matrix

| case | condition | verdict |
|---|---|---|
| A | `G_pair_320 < 0.003` | `NO_CONFIRMED_ATTRIBUTED_CODE_FORMATION_SIGNAL_AT_32D` |
| B | `G_pair_320 ≥ 0.003` **and** `Delta_vs_TOPO ≤ 0.003` | `ATTRIBUTED_PAIRING_SUPPORTED_AND_COMPETITIVE` |
| C | `G_pair_320 ≥ 0.003` **and** `Delta_vs_TOPO > 0.003` | `ATTRIBUTED_PAIRING_SUPPORTED_BUT_NOT_COMPETITIVE` |
| — | provenance/hash mismatch | `ARTIFACT_IDENTITY_FAILURE` (stop) |

Case A closes the 32-D attributed-code-formation line.  Case C means the real
pairing carries task information but placing it inside the current attributed
dictionary/code formation is not better overall than the topology-only
dictionary: do **not** proceed to attributed E2E/IHT, keep the simpler
`topology sparse code → post-code chemistry binding` route, and record the
pairing-positive result as a mechanism finding.  In **every** case this round
stops after its own artifacts; the 320 confirmation outranks the 160 screen and
may not be overturned by it.

## 6. Diagnostics (reported, never gates)

* Paired late window `241–320` of `delta(e) = valid_MAE_INDEP(e) − valid_MAE_REAL(e)`:
  mean, median, positive fraction, first/last delta.  **No new gate.**
* `160 → 320` change: REAL soup, INDEP soup, and `G_pair` (kept / widened /
  shrunk / reversed), answering whether the lite `+0.00556` survived longer
  training.

## 7. Deferred — never auto-started, even if the result is beautiful

Second seed; PCA32; continuity-v2; IHT-10/30/100/200 qualification;
task-coupled E2E sparse dictionary; REAL→INDEP code-pairing-removal mechanism;
node-only / edge-only interventions; DenseTied specificity; K64; s12; official
test; any architecture or hyper-parameter sweep.  All are recorded as
`DEFERRED_PENDING_USER_AUTHORIZATION`.

## 8. Frozen artifacts of this round

```
tracks/ksvd/results/e2e_dictenv_a2_confirm/
    artifact_identity.json   continuation_mode.json
    real_320.json            indep_320.json
    paired_analysis.json     decision.json
    REPORT.md                DECISION.md     stage_status.json
tracks/ksvd/notes/e2e_dictenv_a2_confirm_preregistration.md   (this file)
tracks/ksvd/notes/e2e_dictenv_a2_confirm_analysis.md          (after the run)
records/claims/, records/decisions/, tracks/ksvd/STATE.yaml   (after the run)
```
