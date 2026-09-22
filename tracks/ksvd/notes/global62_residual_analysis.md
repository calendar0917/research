# TCCD-GLOBAL62-v0 — Frozen TCCD + B-Full Global Context Residual Screen

Preregistration: `notes/global62_residual_preregistration.md`.
Preregistration commit: `4d71f43`.
Formal implementation commit: `60f4b25`.
Formal result: `results/global62_residual/stageA_seed0.json`.
Gate 0: `results/global62_residual/gate0.json`.
Decision: `results/global62_residual/stageA_decision.json`.

## Verdict

**Case D — NO MATERIAL GLOBAL SIGNAL (with the shuffle precedence also
firing).** The exact 62-D B-Full `global_context` adds essentially nothing on
top of the frozen TCCD-v5 PRE prediction.

* Reproduced frozen base: `MAE_BASE = 0.251418160446` (formal reference
  `0.2514181435108185`, delta `1.69e-8`).
* `BASE + REAL GLOBAL62` (Top-5 soup): `0.251032248318`.
* `G_total = 0.000385912128`.
* Bias-only control: `b* = -0.008749365807`, `MAE_BIAS = 0.251554300576`
  (slightly *worse* than base), so `G_global = 0.000522052258`.
* Evaluation-only dev derangement: `MAE_SHUFFLE = 0.252333112178`, so
  `G_bind = 0.001300863860` — below the `0.005` precedence threshold.
* Fraction of the `0.2514 -> 0.1198` gap closed: `0.002932` (0.29%).

`G_global < 0.005` fires **Case D**, and `G_bind < 0.005` fires the
precedence rule, so even the `0.00039` total movement cannot be attributed to
a correct molecule <-> global-statistics binding. The honest reading is that
B-Full's plain graph-level statistics are **not** a hidden reservoir that the
frozen TCCD route fails to use.

## Execution integrity

* Local CPU execution (`device: cpu`), deterministic and fully resumable; no
  remote run was needed for a 3,169-parameter head.
* Seed `0`; fixed internal `8000/2000` split, split seed `20260922`.
* Base prediction: **pure forward pass** of the exact frozen TCCD-v5 PRE
  Top-5 soup state
  (`tracks/ksvd/results/tccd_v5/stageA_pre_seed0_soup.pt`, SHA-256
  `93325ac48f2b001899a307e3368db3efaa52a60247727ac02ce2368809a61441`)
  over the reused TCCD-v5 pair cache (`tccd_v5_occurrence_pair_nonlinearity_v1`,
  `official_test_loaded: false`). No TCCD parameter was trained, modified, or
  instantiated as trainable; the soup state is loaded with
  `requires_grad_(False)`.
* Frozen TCCD-v2 checkpoint SHA-256
  `093e0e46d4d5dab7e5558b807d993d1136df58d8f531d6b9b3621e8ab05c0f42`
  verified through the reused cache metadata.
* Global context: exact
  `zinc_long_range_proxy.global_feature_views(dataset)["global_all"]`,
  width `62 = short(15) + long(15) + attributes(28 atom + 4 bond)`, finite
  rate `1.0`, deterministic (recompute max abs diff `0.0`), label-free.
* Train-only standardizer fit on the internal 8000 train rows; constant
  columns `[17, 51, 52, 53, 54, 55, 56, 57, 58]` (9 columns), set to unit
  scale.
* Only new trained object: GLOBAL62 residual head, exact B-Full
  `global_encoder` sequence `Linear(62,32) -> LayerNorm(32) -> ReLU ->
  Linear(32,32) -> ReLU` plus `Linear(32,1)`; `3,169` parameters.
* Protocol: seed 0, Adam `lr 1e-3`, weight decay `1e-5`, L1 on the final
  prediction, batch `128`, max `240` epochs, patience `40`, Top-5 soup, no
  gradient clipping, no sweep.
* Initialization checksum
  `e155425a905dd4d186aaf91657622a3739c64173908b10849a0a7c2e211c4a4e`
  (deterministic; reruns reproduce the same numbers at commit `60f4b25`).
* Training: 53 epochs, best epoch `13`, soup members `[13, 17, 24, 32, 7]`;
  train L1 `0.151530 -> 0.113521`, so the head genuinely fits the train
  residual; the dev gain is nonetheless negligible. Wall `13.2 s`.
* Official ZINC valid and test: never loaded.
* No baseline was re-run; only the single GLOBAL62 seed-0 head was trained.

## Gate 0

**PASS** (`results/global62_residual/gate0.json`, commit `60f4b25`).

All eleven registered checks pass:

1. **Base reproduction**: `0.251418160446` vs `0.2514181435108185`,
   abs delta `1.69e-8 <= 1e-6`.
2. **TCCD frozen**: all soup parameters `requires_grad == False`; no frozen
   gradient after the head backward.
3. **Exact 62-D definition**: width 62, finite rate 1.0, deterministic,
   histogram blocks normalized (atom sum dev `<= 1e-5`, bond sum dev
   `<= 1e-5`), label alignment to the cached TCCD targets `<= 1e-6`.
4. **Train-only standardization**: mean/std reproduced from the internal 8000
   train rows; standardized train mean `9.9e-7`, std dev `1.8e-7`.
5. **Label-free**: perturbing `Data.y` on probe graphs leaves `global_all`
   bit-identical (max abs diff `0.0`).
6. **No official valid**: only the train split is loaded.
7. **No official test**: never loaded; every artifact certifies
   `official_test_loaded: false`.
8. **Gradient scope**: every head parameter receives gradient; frozen TCCD
   parameters receive none.
9. **Shuffle multiset**: dev global-feature multiset preserved exactly
   (max abs diff after per-column sort `0.0`).
10. **Shuffle derangement**: no fixed points, a valid permutation, and it
    changes rows (max abs diff `> 0`).
11. **Bias-only train-only**: `b*` is the median of the internal-train
    frozen residuals.

Parameter accounting `3169 = 2016 + 64 + 1056 + 33`; head shapes
`(32,62) -> (32,) -> (32,32) -> (1,32)`.

## Primary table

| arm | new training? | dev MAE | delta vs frozen BASE |
|---|---|---:|---:|
| Frozen TCCD-v5 PRE | NO | `0.251418160446` | — |
| BASE + bias-only | NO | `0.251554300576` | `-0.000136140130` |
| BASE + REAL GLOBAL62 | YES, tiny head only | `0.251032248318` | `+0.000385912128` |
| BASE + SHUFFLED GLOBAL62 | NO, eval-only | `0.252333112178` | `-0.000914951732` |
| B-Full | existing scale reference only | `0.119818026382` | — |

Registered quantities:

```text
G_total              = MAE_BASE   - MAE_REAL   = 0.000385912128
G_global             = MAE_BIAS   - MAE_REAL   = 0.000522052258
G_bind               = MAE_SHUFFLE - MAE_REAL  = 0.001300863860
gap_closed_fraction  = (MAE_BASE - MAE_REAL) / (MAE_BASE - 0.119818026382)
                     = 0.002932459990
remaining gap to B-Full                        = 0.131214
```

Best-checkpoint (secondary) values agree:

```text
MAE_REAL_best   = 0.251135724669
MAE_SHUFFLE_best= 0.252032670749
G_total_best    = 0.000282435777
G_global_best   = 0.000418575907
G_bind_best     = 0.000896946080
```

## The four questions

**Q1 — Is the frozen TCCD-v5 PRE `0.2514` reproduced exactly?**
Yes. A pure forward pass of the frozen soup state over the reused pair cache
gives `0.251418160446` versus the formal `0.2514181435108185` (abs delta
`1.69e-8`, at float32 reduction-order level). No re-training and no fallback
to another checkpoint was needed.

**Q2 — How much does the 62-D global context add beyond bias correction?**
`G_global = 0.000522`. Against a `0.005` small-signal floor this is an order
of magnitude too small. The bias-only control does not even help
(`MAE_BIAS = 0.251554` is worse than `MAE_BASE = 0.251418`), so there is no
meaningful intercept miscalibration to remove either.

**Q3 — Does the gain depend on correct molecule <-> global binding?**
No. `G_bind = 0.001301`, below the `0.005` precedence threshold. The shuffled
input is *worse* than real (`0.252333` vs `0.251032`), so there is a faint
direction consistent with real binding, but far too small to claim that the
global statistics are being used. By the preregistered shuffle precedence this
downgrades even the `0.00039` total movement to head capacity / optimization
noise.

**Q4 — What fraction of the `0.2514 -> 0.1198` gap is closed?**
`0.29%` (`gap_closed_fraction = 0.002932`). The remaining gap to the B-Full
scale reference is `0.131214`.

## Secondary analysis 1 — size stratification (read-only)

| stratum | n range (mean) | graphs | BASE MAE | REAL MAE | improvement | G_bind |
|---|---|---:|---:|---:|---:|---:|
| small | 9–21 (18.2) | 667 | `0.233485` | `0.233969` | `-0.000484` | `-0.000328` |
| medium | 21–25 (23.0) | 667 | `0.192746` | `0.192343` | `+0.000403` | `+0.002415` |
| large | 25–37 (28.0) | 666 | `0.328138` | `0.326899` | `+0.001239` | `+0.001816` |

The 62-D global statistics do **not** repair the large-molecule failure. The
large-graph improvement (`+0.00124`) is two orders of magnitude smaller than
the large-graph absolute error (`0.32814`), and the small-molecule arm is
actually slightly degraded. A hypothesis that the 62-D global channel is the
missing large-graph signal is not supported.

## Secondary analysis 2 — residual correlations (read-only)

Per raw global column versus the frozen residual `r = y - yhat_base`:

```text
train max |Pearson|  = 0.0588  (column 3)
dev   max |Pearson|  = 0.2246  (column 5; train there 0.0161)
train max |Spearman| = 0.0851  (column 34)
dev   max |Spearman| = 0.0626  (column 28; train there -0.0471)
```

Only one dev column (column 5 = `degree.min`) exceeds `|Pearson| 0.1`, and
its train correlation is `0.0161`: the relation is not stable across the
internal split. No single raw global column carries a stable, generalizable
linear or monotone relation to the frozen TCCD residual. These numbers are
explanatory only; no feature was selected or deleted from them.

## Scientific interpretation

1. The ~0.13 MAE gap between frozen TCCD and B-Full is **not** explained by
   the 62-D graph-level summary block that B-Full feeds to its
   `global_encoder`. With the head given exactly those 62 columns and a
   frozen, correct base prediction, the dev improvement is `0.00039` and it
   does not survive the binding control.
2. This closes the "we are simply missing plain graph-level statistics"
   explanation for the TCCD ceiling. It is a strong negative and it is
   consistent with the two preceding rounds: TCCD-v6 showed recent frozen
   gains are base-reconstructible coordinate exposure, and CENTER-COMP showed
   a single pair-to-center incidence layer is used but insufficient.
3. The head clearly fits the training residual (train L1
   `0.1515 -> 0.1135`) while dev stays flat, which is the signature of a
   capacity/optimization effect rather than transferable graph-level signal.
   This is exactly the pattern the preregistered shuffle precedence is meant
   to catch.
4. Because the base prediction is frozen and correct, this diagnostic cannot
   be rescued by a different reader, normalization, or head width: the
   information would have to be absent from the input, and it is.

Caveats, stated honestly:

* One seed; internal-dev only; no official-valid and no official-test read;
  absolute-level claims are out of scope.
* B-Full `0.119818` is used only as a scale reference, not as a strict causal
  comparator.
* The 62 columns are the *existing* B-Full views. This round does not test
  whether a different, richer global summary could help; it tests the exact
  block B-Full actually uses.
* A single dev column (`degree.min`) shows a dev-only `|Pearson| 0.225`; it is
  not stable on train and is reported only for completeness.

## Not run (frozen stop)

No second seed; no width/depth/dropout/optimizer sweep; no linear-vs-MLP
selection; no feature subset or feature selection; no topology 25-D, no
`patch_cont 146`, no `pair_relation`, no local-state bridge, no center
composition, no end-to-end TCCD+global training; no B-Full / TCCD baseline
re-run; no official valid; no official test. The round ends at the record,
even though the registered exploration is a negative.

## Next hypothesis (requires a new preregistration; not auto-authorized)

Do not build a global-context architecture rescue and do not add the excluded
feature blocks to this probe. The v6 + CENTER-COMP + GLOBAL62 sequence now
points away from "missing global statistics / incidence": the next honest
question is whether the frozen TCCD ceiling is a **local-object adequacy**
question (are the radius-2 canonical patch coordinates, and the prototype
vocabulary learned on them, expressive enough at all?) rather than a pooling
or context question. That requires a new preregistration with matched
controls, and it should not reopen any GLOBAL62 feature block.

Evidence:

* `tracks/ksvd/notes/global62_residual_preregistration.md`
* `tracks/ksvd/results/global62_residual/gate0.json`
* `tracks/ksvd/results/global62_residual/stageA_seed0.json`
* `tracks/ksvd/results/global62_residual/stageA_decision.json`
* `tracks/ksvd/code/global62_residual.py`
* `tracks/ksvd/code/run_global62_residual.py`
* `tracks/ksvd/tests/test_global62_residual.py`
