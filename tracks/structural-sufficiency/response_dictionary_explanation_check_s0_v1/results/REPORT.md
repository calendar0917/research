# Response dictionary explanation check — report

**Round** `response_dictionary_explanation_check_s0_v1`  
**Base commit** `4fc17e33b6cc7fd2f301e05d705183787162e233`  
**Status** complete, frozen replay + CPU linear algebra / SciPy SLSQP  
**Test policy** `DEVELOPMENT_TEST`; Test is evaluated only, never used for any choice  
**Budget** 0 GPU, 0 A100·h; full pipeline wall clock ≈ 9 s on CPU; no OOM

This round reuses, read-only, the frozen RESP best-Valid checkpoint (epoch 63), the K=16/K=64
dictionaries, the fragment cache, the official splits, and the 823-pair edit catalogue. It does not
retrain RESP/ADD/JOINT/C1/GINE, add a seed, enlarge data, sweep K, change the prototype panel, D0,
the splits, or the model. The protocol and frozen settings are in `PROTOCOL.md` and
`configs/explanation_check_s0.json`; the machine record is `results/run_manifest.json`.

## 1. Replay identity

| check | result |
|---|---|
| checkpoint sha256 | `c42fc25f…aac77c` (same as parent) |
| native RESP vs saved predictions, max abs raw | 1.16e-6 Train / 1.16e-6 Valid / 8.10e-7 Test |
| replayed FW64 vs saved stress predictions, max abs raw | 0.0 for all 9 dictionary×split cells |
| K=16 | 136 words / 702 Train candidates / 16 classes; 13 exact-storage classes |
| K=16 vs K=64 | prototypes are the exact K=64 prefix for all compressed classes; other classes identical |
| panels / D0 | identical |
| 823-pair mapping | 791 Train–Train / 5 Train–Valid / 27 Train–Test / 0 both-non-Train; 231 components; 0 duplicates |
| catalogue identity vs follow-up | 396 compared, 0 missing; max \|Δ\| of \|error_RESP\| = 6.21e-7, mean 1.63e-7 (float32 storage precision, not forced) |

The edit-pair replay orients **A = catalogue `row_a`, B = `row_b`**, so
`delta_full = F_D(G_B) - F_D(G_A)` and `delta_y = y_B - y_A`. The follow-up stored the opposite
orientation for some pairs; only absolute errors were compared.

## 2. Projection certificate and high precision

For every observed non-prototype fragment in the three compressed classes, the frozen problem is

\[
\min_{\alpha\ge0,\ \sum\alpha=1} f(\alpha)=\frac1m\|P^\top\alpha-y\|_2^2,
\]

with `P` the K×16 prototype responses on the unchanged Train panel, `y` the fragment response, and
`alpha` the simplex weight. The FW64 alpha is the old 64-step solution. The first-order certificate
is `gap = g·alpha - min(g)`, which upper-bounds `f(alpha)-f*`. The fixed pass rule is
`gap <= 1e-8*max(1, mean(y^2))`, `min(alpha) >= -1e-10`, `|sum(alpha)-1| <= 1e-9`.

There are **2,004 rows** (3 dictionaries × 668 observed non-prototype compressed fragments).
Simplex feasibility is clean for every row (minimum alpha 0; max sum error 9.2e-14); the strict gap
certificate is the binding constraint.

| dictionary | class τ | rows | FW64 f median | FW64 gap median | FW64 gap p95 | FW64 pass | SLSQP FW pass | SLSQP uniform pass | certified / unresolved |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| D_RESPONSE | Z=5, no arom | 496 | 3.15e-3 | 2.26e-3 | 8.69e-3 | 0.8% | 11.7% | 15.1% | 116 / 380 |
| D_RESPONSE | Z=6, no arom | 91 | 2.98e-3 | 1.23e-3 | 6.53e-3 | 3.3% | 17.6% | 23.1% | 29 / 62 |
| D_RESPONSE | Z=5, aromatic | 81 | 1.14e-3 | 1.01e-3 | 6.87e-3 | 4.9% | 29.6% | 37.0% | 37 / 44 |
| D_LATENT | Z=5, no arom | 496 | 5.28e-3 | 1.04e-3 | 5.36e-3 | 19.0% | 36.1% | 33.5% | 218 / 278 |
| D_LATENT | Z=6, no arom | 91 | 4.34e-3 | 2.88e-3 | 1.21e-2 | 11.0% | 23.1% | 23.1% | 27 / 64 |
| D_LATENT | Z=5, aromatic | 81 | 1.09e-3 | 1.01e-3 | 7.09e-3 | 7.4% | 40.7% | 44.4% | 45 / 36 |
| D_FREQ | Z=5, no arom | 496 | 7.52e-3 | 2.87e-3 | 2.30e-2 | 6.2% | 32.9% | 29.2% | 225 / 271 |
| D_FREQ | Z=6, no arom | 91 | 8.93e-3 | 1.36e-3 | 1.71e-2 | 15.4% | 45.1% | 48.4% | 56 / 35 |
| D_FREQ | Z=5, aromatic | 81 | 2.51e-3 | 1.49e-4 | 3.69e-3 | 33.3% | 69.1% | 65.4% | 66 / 15 |

Summary over all 2,004 rows: **193 FW64 passes (9.6%)**, **591 SLSQP-from-FW passes (29.5%)**,
**591 SLSQP-from-uniform passes**, **363 rows where both starts pass**, and **1,185 primary
`NUMERICALLY_UNRESOLVED` rows (59.1%)**. SLSQP reduces the median gap from ~1e-3 to ~1e-8–7e-8
(one class reaches ~1e-16) and the max gap to 2.8e-6, but the 1e-8 rule still rejects many rows. On certified rows the f
improvement over FW64 has median 9.7e-6–5.1e-4 (depending on class), p90 0.8e-3–7.4e-3, and max
1.8e-3–1.1e-2; 625 of 819 certified rows improve, the rest were already FW64-certified.

Two high-precision versions are reported, fixed before results:

* **HP_CERTIFIED** (primary): rows that pass the certificate use the smaller-f certified SLSQP
  solution (tie within numerical tolerance prefers the FW start); unresolved rows stay at FW64.
* **HP_SLSQP_ALL** (sensitivity): unresolved rows use the smaller-f feasible SLSQP start even
  though it missed the 1e-8 certificate. This is not a certified optimum and is not the primary
  claim.

### Full-F metrics

`F_D` is the complete molecule prediction from all fixed views (or the existing fallback); `fidelity`
is `sqrt(mean((F_D - F_RESP)^2))` in raw label units, i.e. distance to the frozen native RESP
function, not to the label. The full table is `results/projection_full_f_metrics.csv`.

| dictionary / version | split | RMSE | MAE | fidelity to RESP |
|---|---|---:|---:|---:|
| D_RESPONSE FW64 | Train | 0.35331 | 0.26970 | 0.03232 |
| D_RESPONSE FW64 | Valid | 0.62480 | 0.48077 | 0.03937 |
| D_RESPONSE FW64 | Test | 0.67848 | 0.52625 | 0.04011 |
| D_RESPONSE HP_CERTIFIED | Train | 0.35317 | 0.26954 | 0.03184 |
| D_RESPONSE HP_CERTIFIED | Valid | 0.62502 | 0.48094 | 0.03866 |
| D_RESPONSE HP_CERTIFIED | Test | 0.67842 | 0.52619 | 0.03989 |
| D_RESPONSE HP_SLSQP_ALL | Train | 0.35234 | 0.26878 | 0.02900 |
| D_RESPONSE HP_SLSQP_ALL | Valid | 0.62476 | 0.48077 | 0.03654 |
| D_RESPONSE HP_SLSQP_ALL | Test | 0.67823 | 0.52579 | 0.03800 |

Baseline fidelity (Valid / Test): D_LATENT 0.09649/0.07653 (FW), 0.09626/0.07613 (HP_CERTIFIED),
0.09538/0.07478 (HP_SLSQP_ALL); D_FREQ 0.07691/0.06952 (FW), 0.07592/0.06692 (HP_CERTIFIED),
0.07554/0.06695 (HP_SLSQP_ALL).

D_RESPONSE is the best dictionary on both Valid and Test in every version. Relative fidelity
reduction against the better baseline:

| version | Valid | Test |
|---|---:|---:|
| FW64 | 48.8% | 42.3% |
| HP_CERTIFIED (primary) | 49.1% | 40.4% |
| HP_SLSQP_ALL (sensitivity) | 51.6% | 43.2% |

**Label: `RESPONSE_ADVANTAGE_SURVIVES_HIGH_PRECISION`.** No `STEP_LIMITED`, no `REVERSED`. The
20% reference is the frozen follow-up investment criterion, not a general effect threshold.

### Panel ties

363 rows have both SLSQP starts passing. 54 of them have `||alpha_FW - alpha_uniform||_inf > 1e-8`;
129 have Val/Test views. Holding all other fragments fixed, the largest actual Val/Test readout
difference between the two alphas is **2.95e-7 raw** (4 views above 1e-7; none above 1e-6).
**Label: `PANEL_TIES_DO_NOT_MATERIALLY_CHANGE_READOUT`**; no `PANEL_UNDERDETERMINES_PREDICTION` was
triggered. The panel still does not identify a unique alpha, but in this frozen setup the
prediction difference is below the fixed 1e-6 materiality threshold. The sub-1e-6 tail is reported
in `panel_underdetermination.csv`, not hidden.

## 3. Explanation coverage

Every one of the 823 real edit pairs was mapped and checked. The existing fixed ≤4-view system is
used as-is; a pair is interpretable only if the two endpoints have a selected view with the **same
C identity and the same cut-bond features**. If several candidates exist, the C with the most atoms
wins, then the smallest canonical key. No pair was re-mined, no view or cut was changed.

| cell | pairs | components | interpretable pairs | interpretable components | fraction | NO_SHARED_SELECTED_CONTEXT |
|---|---:|---:|---:|---:|---:|---:|
| Train–Train | 791 | 228 | 375 | 151 | 47.4% | 416 |
| Train–Valid | 5 | 4 | 4 | 3 | 80.0% | 1 |
| Train–Test | 27 | 7 | 17 | 5 | 63.0% | 10 |
| **All** | **823** | **231** | **396** | **153** | **48.1%** | **427** |

There are **0 `SHARED_CONTEXT_UNSUPPORTED`** pairs: whenever a shared selected C exists, both S
fragments have a supported dictionary encoding. The 427 uncovered pairs are the explanation
boundary: 379 are both viewed but have no common selected C, 33 have one fallback endpoint, 15
have both endpoints in the fallback path. Thus the dominant limiter is the shared-C definition,
not dictionary support.

Held-out coverage is much smaller than the raw 823-pair support. The full support has 7 Test
molecules / 7 Train–Test components and 4 Valid molecules / 4 Train–Valid components. The shared-C
subset has **5 Test molecules / 5 components / 17 pairs** and **3 Valid molecules / 3 components /
4 pairs**. The other 2 Test and 1 Valid molecules are lost to `NO_SHARED_SELECTED_CONTEXT`. These
are `LOW_SUPPORT` cells.

## 4. Full-change versus one-view word account

For each interpretable pair and each dictionary/version:

\[
\Delta_{\text{full}}=F_D(G_B)-F_D(G_A),\quad
\Delta_{\text{local}}=s_D(C,S_B)-s_D(C,S_A),\quad
r=\Delta_{\text{full}}-\Delta_{\text{local}}.
\]

The local term is one existing view; the full term is the complete molecule prediction, including
the other views and the fixed view set. All numbers below are raw label units.

For **D_RESPONSE HP_CERTIFIED**:

| cell | pairs / components | Δfull RMSE | Δlocal RMSE | remainder RMSE | remainder MAE | remainder median | remainder p90 | remainder / Δfull RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Train–Train | 375 / 151 | 0.5879 | 0.5595 | 0.1792 | 0.1318 | 0.1007 | 0.2956 | 30.5% |
| Train–Valid | 4 / 3 | 0.3499 | 0.3771 | 0.0900 | 0.0817 | 0.0659 | 0.1247 | 25.7% |
| Train–Test | 17 / 5 | 0.4061 | 0.4467 | 0.1023 | 0.0827 | 0.0673 | 0.1595 | 25.2% |
| **All** | **396 / 153** | **0.5793** | **0.5535** | **0.1759** | **0.1292** | **0.0991** | **0.2887** | **30.4%** |

The remainder is 14.5% of the Train label standard deviation (1.2115), but 30.4% of the full-change
RMSE. The held-out cells have smaller remainders, but they also have small n and small full-change
magnitudes; this is not enough to claim a task law.

**Model fidelity versus label accuracy must be kept separate.** For the same 396 pairs:

| quantity (D_RESPONSE HP_CERTIFIED) | RMSE | MAE |
|---|---:|---:|
| dictionary Δfull vs native RESP Δfull | 0.0326 | — |
| dictionary Δlocal vs native RESP Δlocal | 0.0863 | — |
| Δfull vs real Δy | 0.4341 | 0.3389 |
| Δlocal vs real Δy | 0.4812 | 0.3684 |
| native RESP Δfull vs real Δy | 0.4315 | 0.3378 |
| zero-change baseline vs real Δy | — | 0.5962 |

The dictionary reproduces the native model's **change** well (0.033 full), but the native model's
change itself is far from the real label (0.434). The word account is an explanation of the frozen
model, not a label-accurate predictor. The other dictionaries are similar or worse on these
diagnostics; D_FREQ has remainder 0.1725 and D_LATENT 0.2304 (HP_CERTIFIED, all cells).

## 5. Word account

For a selected pair, with `alpha(S_B)` over the prototypes of `tau(S_B)`:

\[
\text{contribution}_k=\alpha_k(S_B)\,[s_D(C,D_k)-s_D(C,S_A)],\qquad
\sum_k \text{contribution}_k=\Delta_{\text{local}},\qquad
\Delta_{\text{full}}=\sum_k \text{contribution}_k+r.
\]

There are 3,564 equations (396 pairs × 3 dictionaries × 3 versions). The maximum absolute closure
error is **2.50e-15**; label `WORD_ACCOUNT_CLOSES`. This is an algebraic identity from the linear
`a(S), q(S), b(C)` and bilinear view score. It does **not** establish explanatory correctness,
chemical causality, or an implementable replacement. Each `C ⊕ D_k` is a model readout probe;
no chemical validity check was performed. `alpha` is a simplex weight, not a probability or a
causal share.

The closure is exact, but the local account explains only the local part. The remainder above is
the part attributable to other views / the view set; it is ~30% of the full-change RMSE and is not
resolved by adding more words to the same one-view definition.

## 6. Auditable cases

`CASES.md` contains the fixed 12 cases: the first 6 interpretable pairs with a non-Train endpoint,
then the first 6 interpretable Train–Train pairs, by catalogue `pair_id`, with no effect-based
selection. `cases.csv` contains the same rows as a table.

Each case gives both molecules and SMILES, the common C fragment and cut bond, the S_A/S_B
fragment IDs, SMILES and ports, the main prototype structures/ports, alpha and word contributions,
Δlocal/remainder/Δfull/Δy, FW64/HP_CERTIFIED/HP_SLSQP_ALL values, split, train-word/prototype
flags, and the component ID. RDKit is used only to display cached structures; no structure was
rebuilt for scoring, and no chemical validity is claimed.

Limits of this case set: cases 1–6 use only **2 distinct Test molecules (rows 37 and 758) / 2
components**, and cases 7–12 use component 28 (5 cases) and component 53 (1 case). They are
clustered audit examples, not 12 independent experiments. In several cases `S_B` is a stored
prototype or its row is unresolved under HP_CERTIFIED, so the primary alpha does not change; the
HP_SLSQP_ALL column shows the uncertified change (for example, case 2 changes `alpha` by 0.137 and
Δlocal by +0.0039). This is expected and is why the global projection table, not the case page, is
the evidence for the high-precision comparison.

## 7. What the 136-word dictionary actually is

The 702→136 compression is a reduction of the **S-word table only**:

* 702 Train word candidates → 136 stored words (80.6% fewer words).
* 16 classes: 13 exact-storage classes (all candidates stored), 3 compressed classes (16
  prototypes each: 48 words instead of 442+91+81 candidates).
* The whole model remains **198,262 parameters** (169,187 in the encoder, 29,075 outside it).

Inference path that remains:

1. Molecule → fixed ≤4 views by bridge cuts and the existing selection; fallback head for
   molecules without a cut (6.3% Train, 9.3% Valid, 8.6% Test).
2. For each view: `b(C)`, `k(C)` and `a(S)`, `q(S)`. A stored S word is a lookup with one-hot
   alpha. An unseen S still needs the encoder `z_S` and the `a/q` heads, then the 16-environment
   panel response and the simplex projection to alpha. An unseen C still needs the encoder for
   `b(C)`, `k(C)`.
3. Per view score `b(C)+a(S)+<q(S),k(C),w>/sqrt(16)`; molecule F is the mean over the selected
   views, or the fallback head.
4. Test/Valid are mostly **not** pure lookup: 63% of their views have a stored S, but only
   6.5–7.4% have a stored C (C is the large, molecule-specific side). A view with both stored is
   only 0.5–0.7% of Test/Valid views, versus 66.6% of Train views.

If an application materialises the tables, the storage is approximately: word `a/q` 2,312 floats;
prototype response matrix P 2,176 floats; panel IDs 256; Train `C` `b/k` 156,247 floats; optional
alpha cache for observed S 14,658 floats. This is comparable to the 198k model weights, not a
replacement for them. The fragment cache itself holds 12,369×16 `z` values.

**Therefore: 702→136 is not a smaller or faster whole network.** It is a smaller S-word vocabulary
with the same encoder, the same C side, the same view decomposition, and the same fallback path.
A new S still requires the encoder; a new C still requires the encoder; a molecule with no shared
selected C is outside the explanation definition.

## 8. Answers to the four questions

### Q1. Is the old RESPONSE word-selection advantage larger than the fixed 64-step solver effect?

**Yes, in the sense that matters for the frozen claim.** The D_RESPONSE fidelity advantage is
present at FW64 (Valid 48.8%, Test 42.3% reduction versus the better baseline) and remains after
high-precision projection (HP_CERTIFIED 49.1% / 40.4%; HP_SLSQP_ALL 51.6% / 43.2%). The advantage
is not created by stopping FW at 64 steps; high precision does not erase or reverse it. The label
is `RESPONSE_ADVANTAGE_SURVIVES_HIGH_PRECISION`.

**But it is not a proof of global optimality for every fragment.** Only 9.6% of FW64 rows pass the
1e-8 certificate, 29.5% pass after SLSQP, and 1,185/2,004 rows remain unresolved in the primary
version. The absolute fidelity change from FW64 to HP_CERTIFIED is small (Valid 0.03937→0.03866,
Test 0.04011→0.03989); the large advantage over the baselines is not a high-precision artifact, but
the per-row projection is still partially unresolved.

### Q2. How much of the full prediction change can the word account explain, and with what error?

It explains the **selected one-view local change exactly** (closure 2.5e-15), and the full change
only up to a remainder. For D_RESPONSE HP_CERTIFIED over all 396 interpretable pairs: Δfull RMSE
0.579, Δlocal RMSE 0.554, remainder RMSE 0.176 (MAE 0.129, median 0.099, p90 0.289). The remainder
is 30.4% of the Δfull RMSE and 14.5% of the Train label std. Against real Δy, Δfull error is 0.434
and Δlocal error 0.481, while zero-change is 0.596; the dictionary's approximation to the native
model's change is much tighter (0.033 full, 0.086 local). So: **local account exact, full account
partial, label accuracy limited, model fidelity good.**

### Q3. Which conclusions are only model-internal, and which have real held-out label support?

**Model-internal:** the full-F fidelity advantage; the projection certificates and alpha values;
the word-account closure and contributions; the panel-tie/readout analysis; all Train–Train
descriptions. These are properties of the frozen function and its dictionary, not of chemistry or
the task.

**Held-out label support:** only 5 Test molecules / 5 Train–Test components (17 interpretable
pairs) and 3 Valid molecules / 3 Train–Valid components (4 pairs), all `LOW_SUPPORT`. On the
Train–Test subset the model's Δfull error is 0.669 and the native RESP error 0.671; on Train–Valid
0.292/0.299. These are not enough to establish a task law, and the Test labels were used only for
`DEVELOPMENT_TEST` evaluation. The response advantage is a model-fidelity advantage, not a
demonstrated label advantage: full-F label RMSE/MAE are essentially the native RESP values.

### Q4. Can this be located as a “structure-anchored response summary/dictionary model”, or does the explanation definition still need to change?

**It can be located as a model-internal, structure-anchored response summary/dictionary model.**
The object is: fixed C/S views + 16 prototype responses per compressed class + simplex alpha +
a linear word account. It is a compact summary of how the frozen network responds to local edits,
not a chemical mechanism and not an implementable replacement.

**The explanation definition still needs extension for stronger claims.** Only 48.1% of the 823
pairs have a shared selected C; 379 of the 427 uncovered pairs are both viewed but have no common
C. The word account covers one view, leaving a ~30% full-change remainder. Alpha is not unique
under the panel for 363 ties (54 with alpha differences >1e-8), although the readout effect is
<1e-6 here. This round did not change the definition, because the task froze it. A stronger
full-change explanation would require a new, pre-registered definition (multi-view/all-view
accounting, pairs without a common C, or a richer panel), and more independent held-out edit
components would be needed for label-level validation. Those are research decisions for the
researcher, not changes to make inside this round.

## 9. Labels and limits

* `PROJECTION_CERTIFIED` per row where the fixed rule passes; `NUMERICALLY_UNRESOLVED` otherwise.
* `RESPONSE_ADVANTAGE_SURVIVES_HIGH_PRECISION` — primary and sensitivity versions agree in direction.
* `PANEL_TIES_DO_NOT_MATERIALLY_CHANGE_READOUT` — max actual readout difference 2.95e-7 < 1e-6.
* `WORD_ACCOUNT_CLOSES` — max closure error 2.50e-15.
* `HELDOUT_EDIT_SUPPORT_LOW`, `TASK_LAW_NOT_ESTABLISHED` — 5 Test / 3 Valid molecules in the
  interpretable subset; low support.

Not claimed: global optimality of every row (1,185 unresolved); chemical causality or valid
`C⊕D_k` replacements; a task-level law from Train–Train pairs; a smaller/faster whole network from
702→136; a unique alpha from the panel; any new training, seed, data, K, prototype, panel, D0, or
split. Test was not used for selection.

## 10. Artifacts and reproduction

Main files in `results/`:

* `projection_certificate.csv` — 2,004-row FW64/SLSQP certificate, f/gap/simplex/pass fields.
* `projection_summary.csv`, `projection_summary.json` — per dictionary/class aggregates.
* `projection_full_f_metrics.csv` — full Train/Valid/Test RMSE, MAE, fidelity for 3 dictionaries ×
  3 versions.
* `panel_underdetermination.csv`, `panel_underdetermination.json` — 363 both-qualified rows and
  actual readout differences.
* `explanation_coverage.csv`, `explanation_coverage_by_endpoint.csv`,
  `explanation_uncovered_pairs.csv` — 823-pair coverage and the 427 uncovered pairs.
* `explanation_residuals.csv` (all 396×3×3 pair-level terms), `explanation_metrics.csv`
  (aggregates by dictionary/version/cell), `explanation_summary.json`,
  `explanation_checks.json`, `explanation_mapping.json`.
* `word_accounting.csv` (10,611 prototype contribution rows), `word_accounting_closure.csv`
  (3,564 pair×dictionary×version closure equations).
* `CASES.md`, `cases.csv` — the fixed 12 cases.
* `compression.json`, `summary.json`, `run_manifest.json`.

Reproduce with:

```bash
uv run python tracks/structural-sufficiency/response_dictionary_explanation_check_s0_v1/code/ec_run.py
```

The run is deterministic, CPU-only, and takes about 9 s; `run_manifest.json` records source hashes,
checkpoint/cache/catalogue hashes, the executed commit, environment, artifacts, and the 0 GPU budget.
