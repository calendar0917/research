# E2E-DictEnv-P2-ABS — pre-registration (frozen before any implementation run)

Round **E2E-DictEnv-P2-ABS** · protocol `e2e_dictenv_p2_abs` · study
`zinc-context-gap` (Workstream Z of the 2026-09-24 two-workstream round).

This is the frozen specification, written **before** any implementation, cache
build, training or GPU use.  Any deviation requires a written amendment here
before the affected run.

The paired Workstream M (MolHIV architecture transfer) is pre-registered
separately in `e2e_dictenv_m1_preregistration.md`.

---

## 1. Scientific objective

Only one objective:

```
min MAE_{official-valid}
```

while preserving the clean dictionary-core purity:

```
NO patch_cont ; NO atom_shell ; NO bond_shell ; NO typed/parent lookup ;
NO local chemistry-by-structure handcrafted bypass ; NO message passing ;
NO recurrence ; NO pair->centre
```

Local structural organisation must still pass through the dictionary code.

This round does **not** answer Sparse vs Dense, dictionary specificity,
TopK vs IHT, or shuffle-mechanism decomposition.  Those are deferred.

---

## 2. Frozen P1 reference (`Z0_REFERENCE`, not re-run)

```
P1 seed0 soup official-valid MAE = 0.13197501279687276
params                           = 97,865
K = 32, s = 8, IHT = 10
lambda_rec                       = 33.95873017865987
epochs                           = 240
```

---

## 3. Tuning discipline

Official-valid is an explicit **development / tuning set**.  Architecture,
`lambda` and horizon may be chosen directly on it.  Official ZINC test is
**not loaded in this round** (even though historical rounds opened it).

---

## 4. Common training protocol

Except where a candidate explicitly varies one named axis:

```
official train = 10,000 ; official valid = 1,000
seed = 0 ; Adam, lr 1e-3, weight_decay 1e-5, batch 128, grad clip 5
no scheduler ; fixed Top-5 soup by valid MAE
no seed 1 (stability is a later round)
```

---

## 5. Stage A — optimisation pressure (3 runs)

| id | architecture | lambda | horizon |
|---|---|---|---|
| Z0 | P1 exact | 33.958730 | 240 | (reference, not run) |
| Z1 | P1 exact | factor 0.25 → 33.958730 | 320 |
| Z2 | P1 exact | factor 0.125 → 16.979365 | 240 |
| Z3 | P1 exact | factor 0.0625 → 8.489683 | 240 |

Forbidden: `lambda = 0`, any other value, 400 epochs.

`C_A = argmin{Z0, Z1, Z2, Z3}` by official-valid Top-5 soup.

---

## 6. Stage B — hierarchical dictionary-slot decoder (2 runs)

Fix the Stage-A winner's `lambda factor`, horizon, K32/S8/IHT10.  Change only
the dictionary-mediated decoder.

### H1 — shared compact slot decoder

```
node raw slot   96D x 3      -> shared encoder 96 -> 64 -> 48  (SiLU)
edge raw slot   48D x 6      -> shared encoder 48 -> 48 -> 32  (SiLU)
primitive anchor 62D         -> 62 -> 32                       (SiLU)
fusion = [32 ; 3x48 ; 6x32] = 368 -> 128 -> 48                 (SiLU)
```

No residual bypass.

### H2 — shared wider slot decoder

```
node  96 -> 96 -> 64       -> 3x64
edge  48 -> 64 -> 48       -> 6x48
anchor 62 -> 48
fusion = 48 + 3x64 + 6x48 = 528 -> 96 -> 48
```

---

## 7. Parameter budget

Stage-B / Stage-C:

```
80_000 <= total trainable params <= 130_000      (target ~100k)
```

All added capacity must live inside dictionary-mediated slot processing or
environment fusion.  No handcrafted bypass may be re-created with parameters.

---

## 8. Stage-B selection

Compare `{Stage-A winner, H1, H2}` on official-valid Top-5 soup; keep the
minimum.  Each candidate needs only a cheap integrity gate: task gradient to `D`
nonzero, exact sparsity, dictionary not collapsed, no forbidden bypass, no
MP/recurrence, finite training.  No expensive shuffle/zero interventions in this
tuning phase.

---

## 9. Stage C — edge capacity (1 run)

Based on the Stage-B winner:

```
d_A = 96 unchanged ; d_E: 48 -> 64
structural role g_uv = [alpha_u+alpha_v ; |alpha_u-alpha_v| ; alpha_u o alpha_v]
```

Only the necessary `W_E^S`, `W_E^C`, shared edge-slot encoder input width and
fusion width are adjusted; total <= 130k.  Tag `E64`.

---

## 10. Stage D — dictionary capacity (1 run + conditional)

A `K64 / s8` candidate for an absolute-ceiling probe.  Re-fit on ZINC
official-train pure-topology `phi65` only (no valid):

```
K = 64, s = 8, same K-SVD family, epochs = 10
IHT steps = 10, exact top-8
```

Uses the current best interface, edge width, horizon and lambda factor.  The
reconstruction scale changes with `K`, so the absolute `lambda` is **not**
copied: recalibrate `lambda_base = L_task^init / L_rec^init` under the same rule
and multiply by the Stage-A chosen `lambda_factor`, then freeze.

Conditional `K64 / s12`: allowed only if `MAE_{K64,s8}` improves the best valid
soup before Stage D by `>= 0.001`.  `K64/s12` must re-fit a matched `s=12`
dictionary; re-using the `s8` fit with `s=12` inference is forbidden.

---

## 11. Search budget

```
Stage A              3 full runs
Stage B              2
Stage C              1
Stage D              1
conditional s12      1
------------------------------
MAX                  8
```

The P1 historical reference is not re-run.  No Cartesian grid.  Only the winner
of a stage passes to the next.

---

## 12. Selection and reporting

`C_Z* = lowest official-valid Top-5 soup MAE`.  Report architecture, params, K,
s, lambda factor, absolute lambda, horizon, best valid, soup valid, train MAE,
best epoch, reconstruction, dictionary health, wall time.

Descriptive bands (not gates):

```
<= 0.120       exceptional
0.120-0.125    target band
0.125-0.130    strong improvement
0.130-0.132    modest improvement
>= 0.132       no material absolute progress
```

---

## 13. Stop conditions

No reader / LR / weight-decay / activation / dropout / LayerNorm / attention /
MP / recurrence / patch_cont / atom_shell / bond_shell / multi-dictionary /
per-shell-dictionary sweep.  Stop when the 8-run budget is spent.

---

## 14. Final freeze

`results/e2e_dictenv_p2_abs/final_config.json` records the winning candidate,
commit, params, `D` init/fit hash, K/s/IHT, lambda calibration, horizon, Top-5
member epochs, Top-5 checkpoint hashes, valid MAE and
`official_test_loaded = false`.  **No ZINC test read this round.**

---

## 15. Durable artifacts

```text
notes/e2e_dictenv_p2_abs_preregistration.md
notes/e2e_dictenv_p2_abs_implementation.md
notes/e2e_dictenv_p2_abs_analysis.md

results/e2e_dictenv_p2_abs/
    correctness.json  curves/  states/
    stage_a_h320.json  stage_a_lambda0125.json  stage_a_lambda00625.json
    stage_b_h1.json  stage_b_h2.json
    stage_c_edge64.json
    stage_d_k64s8.json  stage_d_k64s12.json        (conditional)
    final_config.json  REPORT.md  DECISION.md
```
