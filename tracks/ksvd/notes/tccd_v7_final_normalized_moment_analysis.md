# TCCD-v7 — Final Normalized-Moment Closure: Analysis

Round: **TCCD-v7** (*Final Normalized-Moment Closure*), study `zinc-context-gap`.
Preregistration: `notes/tccd_v7_final_normalized_moment_preregistration.md`
(commit `a3f531d`), implementation commit `34d931f`, remote A100 **GPU1 only**,
official ZINC test **never loaded**.

Frozen verdict (first line, per the preregistration):

> **Explicit normalized prototype / relation moments are not sufficient as a
> standalone TCCD representation: replacing the raw un-normalized coordinates
> with the normalized moment representation costs 0.133754 MAE under an
> identical linear reader, so the TCCD standalone performance route is CLOSED
> with no end-to-end run and no official-valid run.**

## 1. What was asked

TCCD-v5/v6 established that the local gain of the frozen TCCD-v2 representation
comes from **coordinate conditioning / readout geometry**, not from new
structural information:

```text
BASE (raw h_v2)              soup 0.2783639132976532
POST (raw h_v2 + 197-D pair) soup 0.2522333264350891    G_FULL 0.0261305868625641
RECON-NL (raw + 69 recon)    soup 0.252360463           rho 0.995135
RECON-LINFACT (no ReLU)      soup 0.252556831           gap    0.000196368
```

TCCD-v7 asked whether writing that conditioning **explicitly into the
representation itself** (size-normalized prototype mean and variance,
normalized relation contractions, explicit log scale/mass coordinates) is
enough on its own:

```text
h_N   = [ mu,      {vec_sym(Mhat_r)}, log(1+n), {log(1+s_r)} ]   (10470)
h_NM  = [ mu, v,   {vec_sym(Mhat_r)}, log(1+n), {log(1+s_r)} ]   (10534)
Mhat_r = (C^T R_r C) / max(s_r, eps),  s_r = sum_ij R_r(i,j)
```

with a **linear reader only** `Linear(h_G, 1)` and no other change.

## 2. Gate 0 — PASS

Data-free + real-cache checks (`results/tccd_v7/gate0.json`), locally and on
remote GPU1:

| check | result |
|---|---|
| A permutation invariance (`mu`, `v`, `s_r`, `Mhat_r`, final `h`) | PASS |
| B batching + padding invariance | PASS (`2.3e-10`, `0.0`) |
| C algebra `sum_k mu_k = 1`, `v = m2 - mu^2`, clamp path | PASS (`0.0`, `1.0e-09`) |
| D normalized composition mass on the full rebuilt symmetric matrix | PASS (`<=5.6e-08` synthetic, `1.8e-07` real) |
| E zero-mass relation exactly zero and finite | PASS (real cache: relation 2 has 127 zero-mass graphs, relation 3 has 9366, all exactly zero) |
| F no target leakage | PASS (label attribute access raises) |
| G official test blocked | PASS (`official_test_loaded: false`) |
| H independent float64 numpy reference == tensor path | PASS (rel `1.7e-08`) |
| I Stage-A cache path == Stage-B tensor path | PASS (`4.8e-07`) |
| J relation mass from `h_base` == relation mass from the frozen pair cache | PASS (rel `2.2e-07`) |
| K RAW features are exactly the frozen `h_base` | PASS |

Parameter accounting (Stage B, had it run): encoder 45,696 + prototypes 4,096 +
temperature 1 + reader 10,535 = **60,328**.

## 3. Stage A — frozen representation closure (decisive)

Frozen TCCD-v2 PrototypeREL checkpoint, frozen encoder / prototypes /
temperature, exact internal split 8000/2000 (`SPLIT_SEED 20260922`), identical
linear-reader protocol (seed 0, Adam `1e-3`, wd `1e-5`, clip 5, batch 32, 240
epochs, patience 40, Top-5 soup). All three arms ran in one session on GPU1.

| arm | dim | best dev MAE | best epoch | **Top-5 soup MAE** | early stop |
|---|---:|---:|---:|---:|---|
| RAW (frozen v5 reference) | 10464 | 0.28791576623916626 | 54 | **0.2783639132976532** | epoch 94 |
| RAW re-screen (same session) | 10464 | 0.28791576623916626 | 54 | **0.2783639132976532** | epoch 94 |
| NORM | 10470 | 0.4144890606403351 | 240 | **0.4146538972854614** | no (ran to 240) |
| NORM+MOM (primary) | 10534 | 0.4119139015674591 | 240 | **0.4121180474758148** | no (ran to 240) |

Protocol identity is exact: the same-session RAW re-screen reproduces the
frozen TCCD-v5 BASE **bit-for-bit** — best MAE, best epoch (54) and the whole
Top-5 soup member list `[54, 94, 50, 43, 79]` — so `protocol_drift = 0.0` and
the comparison is not confounded by execution-regime noise.

Registered decisions:

```text
delta_norm       = MAE_RAW          - MAE_NM = 0.2783639132976532 - 0.4121180474758148 = -0.13375413417816162
delta_norm_rerun = MAE_RAW_rescreen - MAE_NM = -0.13375413417816162
delta_mom        = MAE_NORM         - MAE_NM = 0.4146538972854614 - 0.4121180474758148 =  0.0025358498096466064
protocol_drift   = 0.0

MAE_NM <= 0.255                  -> FALSE (0.412118)
delta_norm       >= 0.015        -> FALSE (-0.133754)
delta_norm_rerun >= 0.015        -> FALSE (-0.133754)
protocol_drift   <= 0.005        -> TRUE

STAGE A VERDICT: FAIL   ->  STOP TCCD standalone immediately
```

Per the frozen preregistration this blocks Stage B (end-to-end), the
official-valid run and the official test. **None of them was run.**

## 4. Why it failed: representational underfitting, not optimization noise

The failure is not a training artifact. The normalized arms never early-stopped
(patience 40 never triggered within 240 epochs) and were still improving when
the budget ended, while their **train** loss stayed roughly twice the RAW train
loss:

| arm | train loss @ epoch 200 | valid @ epoch 200 | train loss @ epoch 240 | valid @ epoch 240 |
|---|---:|---:|---:|---:|
| RAW (stopped @ 94) | — (train `~0.180` @ 80) | — | — | — |
| NORM | 0.401679 | 0.418548 | 0.397303 | 0.414489 |
| NORM+MOM | 0.397673 | 0.416005 | 0.393731 | 0.411914 |

A linear reader over the normalized representation cannot even fit the
*training* split as well as it fits the raw representation (`~0.39` vs
`~0.18`). The evaluator is identical (same seed, same device, same protocol,
same split); only the features differ.

Mechanism, stated exactly:

* `RAW -> (NORM+MOM)` is a deterministic map: `mu = BAG/n`,
  `Mhat_r = M_r / max(s_r, eps)`, plus `v`, `log(1+n)`, `log(1+s_r)`.
* Recovering the raw coordinates back requires multiplying by the per-graph
  scale (`BAG = n * mu`, `M_r = s_r * Mhat_r`). A **linear** reader cannot
  perform that multiplication: `n` and `s_r` enter only as separate additive
  coordinates.
* Therefore the size/mass scale information that the raw coordinates expose
  directly is no longer linearly available, and ZINC's target has a strong
  additive size/mass component that the raw reader demonstrably exploits
  (RAW train MAE `~0.18` vs NORM+MOM `~0.39`).

This does **not** contradict TCCD-v6. v6's RECON/RECON-LINFACT arms were
`h_base` **plus** normalized coordinates (a union representation, 10,480 dims):
the raw coordinates were still present, so the linear reader could use both.
v7 removes them and the gain disappears. The v6 conclusion must therefore be
sharpened to:

```text
the v5/v6 gain is an ADDITIVE coordinate-exposure effect inside the union
representation (raw coordinates remain necessary); it is NOT reproducible by
substituting an explicitly normalized moment representation for the raw one.
```

## 5. Variance diagnostic (report only)

```text
delta_mom = 0.0025358498096466064  <  0.005  ->  NEGLIGIBLE
```

The prototype variance `v` adds no independent increment. This is consistent
with the v6 attribution (second moments / nonlinearity are not the active
ingredient) and is now confirmed under end-to-end-free frozen screening with a
matched reader.

## 6. Vocabulary / Stage B / official-valid

Not run. Stage A FAIL forbids them by the frozen preregistration, so there is
no end-to-end vocabulary-health report, no official-valid number and no
`Case S/P/G` classification. This round produces no standalone performance
candidate at all.

## 7. Scientific conclusion (frozen boundary)

Supported by this round plus the frozen prior rounds:

* **Prototype vocabulary supported: YES.** Task-learned K=64 prototypes beat
  matched DenseREL (v2: REL 0.286244 vs Dense 0.410789) and show healthy
  usage (64/64 active, effective count 62.58, top-8 mass 0.1526, tau 0.0946).
* **Assignment-sensitive composition supported: YES.** v2 REL-SHUFFLE 0.838816
  vs REL 0.286244.
* **Normalization / coordinate conditioning supported: YES, as an additive
  readout effect only.** v5 `G_FULL = 0.026131`, v6 `rho_RECON = 0.995135` /
  nonlinearity gap `0.000196`. v7 shows this effect cannot be converted into a
  standalone normalized representation.
* **Prototype-moment standalone predictor competitive: NO.**
  `MAE_NM = 0.412118` versus RAW `0.278364` (frozen v2 checkpoint;
  v2 official-valid soup `0.261988`; canonical strong GPU1 reference
  `0.119818`).
* **TCCD standalone performance route: CLOSED.**
  `No TCCD-v8 rescue is authorized.`

Explicitly **not** concluded: "prototype learning failed". The correct boundary is

```text
prototype vocabulary works;
prototype-moment standalone predictor is insufficient.
```

## 8. Recommended future role of the prototype vocabulary

Not a standalone predictor, but an **interpretable / reusable local coordinate
system inside a stronger structural backbone**. Any such hybrid must carry a
matched control (same backbone without prototypes; same backbone with
prototypes; prototype shuffle / zero intervention; vocabulary health). No
hybrid was implemented or authorized in this round.

## 9. Evidence

* `results/tccd_v7/gate0.json`
* `results/tccd_v7/stageA_seed0.json`
* `results/tccd_v7/stageA_decision_seed0.json`
* `results/tccd_v7/stageA_raw_rescreen_seed0.{json,pt}`
* `results/tccd_v7/stageA_norm_seed0.{json,pt}`
* `results/tccd_v7/stageA_norm_mom_seed0.{json,pt}`
* `code/tccd_v7.py`, `code/run_tccd_v7.py`, `tests/test_tccd_v7.py` (24 tests)
* remote run log `tccd-v7-stageA` (GPU1, commit `34d931f`)
