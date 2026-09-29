# Analysis — `e2e_dictenv_capacity_localization_v2`

Repaired Capacity Localization round on the frozen ZINC dictionary line,
CPU only, official test never loaded.
Preregistration:
`tracks/ksvd/notes/e2e_dictenv_capacity_localization_v2_preregistration.md`
(sha256 `c5004e3d2aee…`, implementation freeze commit `dccc149`, harness fix
`cc11654`); runner and module frozen before the first training process; the
preregistration file itself was not edited afterwards.

## 1. Question

v1 (`e2e_dictenv_capacity_localization_v1`) returned
`NO_CLEAR_CAPACITY_LOCALIZATION` but its warm screen was not run in a stable
baseline neighbourhood: the M0 continuation control under fresh
`Adam(lr = 1e-3)` degraded from `0.130028` (step 0) to `0.135621` (best) and
`0.156383` (epoch 40).  v2 therefore asks the prior question first:

> Can a warm adaptation be run that keeps the trained CSSD-q1 representation
> intact, and, under that repaired protocol, does any of the **frozen** v1
> capacity directions F / R / G beat the absolute anchor `M_start = 0.130028`?

F/R/G architectures are unchanged (same names, shapes, parameter counts:
+29 568 / +34 400 / +30 336; totals 127 295 / 132 127 / 128 063; shape
fingerprints recorded in `architecture_fingerprints.json`).  Only the residual
projection *scale* changed (F 0.0008, R 0.0016, G unchanged 0.01), so the
augmentation is near-zero at initialisation but every new parameter still
receives finite non-zero gradient.  The optimizer is split into `base`
(lr `1e-4`) and `new capacity` (lr `1e-3`) groups.

## 2. Phase A — M0 calibration (20 epochs, fresh Adam lr 1e-4)

`WARM_ADAPTATION_PROTOCOL_VALIDATED`: all three frozen conditions pass.

| metric | value | threshold | pass |
|---|---|---|---|
| C1 soup (epochs 1–20) | `0.128984` | ≤ 0.1320 | yes |
| C2 mean valid (16–20) | `0.130331` | ≤ 0.1350 | yes |
| C3 epoch 20 | `0.129672` | ≤ 0.1370 | yes |
| C3 late slope (16–20) | `-1.55e-4` / epoch | ≤ +5e-4 | yes |

Deltas vs `M_start = 0.130028`: best `-0.000356` (epoch 20), soup `-0.001044`,
last-5 `+0.000303`.  The low-rate M0 continuation is not merely stable — it
slowly *improves* the frozen checkpoint: soup members `[8, 10, 11, 12, 20]`,
train MAE falls 0.0780 → 0.0755 while valid MAE stays in a 0.1297–0.1330 band.

This is the answer to the v1 diagnosis: at `lr = 1e-3` the same control jumped
to valid `0.1435` at epoch 1 and never recovered; at `lr = 1e-4` epoch 1 is
`0.130852` and the trajectory stays inside the soup basin.  The v1 failure was
the adapter (fresh full-parameter restart at 10x the rate), not a property of
the base representation, and not demonstrably of the capacities.

## 3. Initialisation audit (step 0, before any training)

| candidate | step0 MAE | Δ MAE | mean shift | median | max | residual-zero identity | new grads |
|---|---|---|---|---|---|---|---|
| M0 | 0.130028 | 0.000000 | 0.00000 | 0.00000 | 0.00000 | exact | — |
| F | 0.130073 | +0.000045 | 0.00080 | 0.00078 | 0.00389 | exact (0.0) | 18/18 > 0 |
| R | 0.129986 | −0.000042 | 0.00059 | 0.00050 | 0.00378 | exact (0.0) | 20/20 > 0 |
| G | 0.129941 | −0.000087 | 0.00113 | 0.00114 | 0.00300 | exact (0.0) | 9/9 > 0 |

All three candidates satisfy the hard bound (`mean shift ≤ 0.002`); F and R
also satisfy the preferred bound (`≤ 0.001`), G is at `0.00113` (unchanged
from v1, where it was already inside the hard bound).  F/R shifts are 12.5×
and 29× smaller than v1 (0.01053 / 0.01948).  Zeroing the residual projection
reproduces the CAP-BASE prediction **exactly** for all three candidates, which
is the frozen residual-augmentation contract.

## 4. Phase B — repaired differential-LR screen (40 epochs)

Frozen primary metric: Top-5-by-valid soup over epochs 21–40.  Absolute anchor
`M_start = 0.130028`; matched control = M0 of the same four-arm batch.

| candidate | params | soup (21–40) | Δ vs M0 | Δ vs start | last-10 | Δlast10 vs M0 | best valid | S1 | S2 | S3 | S4 | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| M0 | 97 727 | 0.128723 | 0.000000 | −0.001305 | 0.130721 | 0.000000 | 0.128668 @6 | — | — | — | — | CONTROL |
| F | 127 295 | 0.128613 | −0.000110 | −0.001415 | 0.130971 | +0.000250 | 0.129047 @16 | no | no | no | yes | NO_CAPACITY_SIGNAL |
| R | 132 127 | 0.129422 | +0.000699 | −0.000606 | 0.133506 | +0.002785 | 0.129948 @28 | no | no | no | yes | NO_CAPACITY_SIGNAL |
| G | 128 063 | 0.128879 | +0.000156 | −0.001149 | 0.130604 | −0.000116 | 0.128998 @6 | no | no | no | yes | NO_CAPACITY_SIGNAL |

Gate thresholds: S1 `Δsoup ≤ −0.003`, S2 `soup ≤ 0.1270`, S3
`Δlast10 ≤ −0.003`, S4 real branch usage.  Nobody passes S1–S3; every candidate
passes S4 (branch gradient and update are finite and non-zero throughout), so
the screen was *live*: the null is not dead-initialisation.

Interpretation of the three directions:

* **F (multi-rank structure-semantic fusion)** is neutral: `Δsoup = −0.00011`,
  `Δlast10 = +0.00025`, best valid 0.129047 (worse than the control's
  0.128668).  The 4-head branch was demonstrably trained (per-epoch new-group
  gradient norm 0.19 at the start, 0.011–0.024 through the window) and bought
  nothing.
* **R (two residual FiLM pair blocks)** is mildly *harmful* at this rate:
  `Δsoup = +0.00070`, `Δlast10 = +0.00279`, window mean valid `0.132906` vs
  M0 `0.130802`, with the largest new-branch gradient of the three (7.4 at the
  start, 1.2–1.4 mid-window).  A 10×-rate pair-state residual perturbs the
  frozen pair composition and slightly degrades it.
* **G (gated DeepSets summaries)** is neutral: `Δsoup = +0.00016`,
  `Δlast10 = −0.00012`, best valid 0.128998 (worse than the control).

All four arms improve relative to `M_start` (soup −0.0006…−0.0014) because the
low-rate base adaptation itself improves the checkpoint; the candidates do not
add to that improvement.  No candidate reaches the absolute anchor
`S2 = 0.1270`; the best is F at `0.128613`, still `+0.0026` above the
pre-registered absolute gate.

## 5. Protocol noise floor (measured, not assumed)

The M0 calibration (20 epochs, running alone) and the M0 screening arm
(40 epochs, running in the 3-process wave) are the *same* configuration,
seed and data order.  Their epochs 1–20 differ by `mean 0.00121 / median
0.00082 / max 0.00490` valid MAE, and the Top-5-mean-valid over epochs 1–20
differs by `-0.00060`.  This is CPU thread-scheduling floating-point
non-determinism (no dropout exists; RNG streams are identical by test), and it
gives the round its honest resolution: at the soup level the protocol's own
reproducibility floor is of order `6e-4`, the single-epoch spread inside the
screening window is `std ≈ 1.2e-3` (M0) to `2.0e-3` (R), and the frozen S1
effect size `3e-3` is ~5× the floor.

Consequences: F (−0.00011) and G (+0.00016) are indistinguishable from the
control at this resolution, while R (+0.00070 soup, +0.0028 last-10,
+0.0021 window mean) exceeds the floor by 2–5× and is a real, small negative
effect.  The measured floor is also why v1's `−0.0002…−0.0005` soup deltas were
never interpretable, and why the frozen v2 gate demands `−0.003`.

## 6. Decision

No candidate passes S1–S4 → frozen branch 1:
`LOCAL_CAPACITY_AUGMENTATION_NOT_SUPPORTED`.  `full/NOT_RUN.json` written; no
320-epoch run is bought; no mechanism probes (they require a frozen winner
soup); no combination candidate; no second LR protocol; no seeds 1/2.

Budget spent: 20 (calibration) + 160 (screen) = **180 training epochs**, 0
full-run epochs.  Training wall clock 3855 s (~64 min) at 4 threads in waves of
3 (M0/F/R) and 1 (G); R-bound.

## 7. What this round establishes and what it does not

Establishes (frozen decision + measured noise floor):

* a warm-adaptation protocol that keeps the CSSD-q1 basin: M0 at `Adam(lr =
  1e-4)` holds and slightly improves the checkpoint (`0.130028 → 0.128984`
  soup), so a discriminative local capacity screen is possible on this base;
* under that protocol, **none** of the three frozen single-block capacity
  directions produces an absolute improvement: `|Δsoup vs M0| ≤ 7e-4`
  (F/G ≤ 1.6e-4, i.e. at the protocol's own floor) and none reaches
  `0.1270`; the only deviation above the floor is R's small regression;
* therefore simple local widening — one extra bilinear rank block, two extra
  pair residual blocks, or one extra invariant readout summary — is **not**
  the missing ingredient that moves the model off the 0.128–0.130 plateau in a
  40-epoch adaptation, at the frozen sizes/rates;
* the v1 conclusion was adapter-limited: its "all candidates slow the drift"
  signal was a degradation-regime artefact, and no v1 candidate number is
  carried forward.

Does not establish:

* that F/R/G would fail under a 320-epoch from-scratch training regime; the
  screen gate (by design) never authorised that run, so the from-scratch
  capacity question for these three exact blocks remains unspent rather than
  answered;
* any statement about width/rank/depth beyond the single frozen size per
  direction (`H=4/d_h=24`; 2 blocks × 256; 192-dim summaries) — those are the
  only tested points, and no sweep was performed;
* anything about the mentor's ~0.06–0.062 reference band, the dictionary
  representation itself, or multi-seed stability (single seed 0);
* that a *sequence* of contextualisation stages fails — the tested objects are
  one-shot static compositions, which is exactly the direction the next round
  should interrogate.

## 8. Provenance

* preregistration sha256 `c5004e3d2aee…` (`preregistration_snapshot.json`),
  implementation freeze commit `dccc149`, harness fix `cc11654` (init-audit
  M0 guard, no decision change), recorded in `preflight.json` /
  `init_audit.json`;
* CAP-BASE checkpoint sha256
  `5fc41ab49eca10bf9cf31ef033f8fbe3dbb96ef0eb394116d7a9ca6bf3aa145d`,
  subspace sha256
  `36636ce92836bdb8d023cc91b3f532f8d4c46a57d457514c92d90c68028f6c24`;
  soup MAE recomputed `0.130028` (`|Δ| < 5e-7`);
* `architecture_fingerprints.json`, `parameter_budget.json` (counts equal v1;
  ratio 1.1634 ≤ 1.5), `init_audit.json`, `calibration/*`,
  `screening/*` + `screening_summary.csv` + `decision.json`,
  `full/NOT_RUN.json`, `mechanism/probes.json` (`run: false`),
  `analysis_tables.md`, `summary.json`;
* 59/59 focused tests
  (`tracks/ksvd/tests/test_e2e_dictenv_capacity_localization_v2.py`), 40/40 v1
  regression tests;
* `official_test_loaded = false` in every payload; `CUDA_VISIBLE_DEVICES=""`
  and `torch.device("cpu")` everywhere; `docs/luyin/luyin19.txt` untouched.

## 9. Next round (new preregistration only; nothing implemented here)

The two one-shot local-widening families this line has now tested — the v1
`lr = 1e-3` restart (no power) and the v2 stable screen (no effect) — leave the
plateau question at the *composition* level rather than the width level.  The
next preregistration should test whether the one-shot static composition itself
is the ceiling, by giving each environment a small number of controlled
iterative contextualisation stages:

```text
E_i^(0)  = dictionary-semantic environment (frozen CAP-BASE features)
m_i^(l)  = Agg_j psi(E_i^(l), E_j^(l), r_ij)
E_i^(l+1)= E_i^(l) + eta_l m_i^(l)          l = 0, 1  (at most two stages)
```

with the residual-zero contract of this round, a matched M0 control under the
validated `lr = 1e-4` / differential-rate adapter, and a gate effect size
stated relative to the measured floor.  That is the conceptual question this
round leaves open: whether the 0.13 plateau comes from one-shot static
composition itself rather than from any single block's width.
