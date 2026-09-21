# TCCD-v2 — Absolute Performance Analysis

Pre-registration: `notes/tccd_v2_preregistration.md`.
Formal corrected full-data commit: `030b3c5fdb65f311ed5e9882ad0b176fa6fd219c`.
Device: remote A100 **GPU1**. Seed `0`. Official train `10,000` to official
valid `1,000`. Official test: **never loaded**. K-SVD refit: **NO**.
OMP/IHT: **NO**. Raw reconstruction loss: **NO**.

## Authorization

The preregistered internal gates passed before this run:

* composition gain `0.552572 >= 0.010`;
* Prototype-REL vs Dense-REL gap `-0.124546`, strong pass;
* vocabulary usage `64/64` active, effective count `62.5755`, top-8 mass
  `0.152634`;
* internal Prototype-REL best MAE `0.286244 <= 0.30`.

## Corrected execution

The first full-data attempt at commit `8ad255c` was invalidated before analysis:
its runner passed official-valid indices into the training-record list, so the
reported validation metric was not an official-valid evaluation. That result
is not used anywhere below. The runner was corrected so train records and
official-valid records are separate objects, then redeployed and rerun on
GPU1. The corrected run loaded the cached official-valid records and is the
only full-data result used here.

## Corrected full-data result

| metric | value |
|---|---:|
| best valid MAE | **0.287337** |
| best epoch | 90 |
| Top-5 soup valid MAE | 0.261988 |
| soup epochs | 90, 123, 121, 80, 103 |
| learned temperature | 0.104300 |
| wall time | 851.7 s |
| peak GPU memory | 45.9 MB |

Canonical GPU1 baseline: `0.119818`.

Using the preregistered best-checkpoint decision metric:

    Delta_abs = 0.287337 - 0.119818 = 0.167519

This is above the `0.05` NOT-VIABLE threshold. The absolute gate is
**NOT VIABLE**. The diagnostic soup result is also `0.142170` above the
canonical baseline, so ensembling does not close the gap.

The corrected run is fully reported in `results/tccd_v2/absolute_seed0.json`:
`valid_records_cached=true`, `commit=030b3c5`, GPU1, official test false.

## Scientific interpretation

The local prototype vocabulary itself is not the main failure: it preserves
assignment-sensitive composition on the internal split, beats the matched
Dense-REL control there, and remains broad rather than collapsed. The failure
is the absolute capacity/generalization gap of the simple `714 -> 64` local
encoder plus second-order prototype composition when evaluated on the actual
official-valid split.

The round stops here. No temperature/K/reader/regularizer/relation sweep, no
stronger encoder rescue inside TCCD-v2, and no official-test read is allowed.
A stronger permutation-invariant local encoder would require a new
pre-registration.
