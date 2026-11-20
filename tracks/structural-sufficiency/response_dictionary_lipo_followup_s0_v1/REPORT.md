# Lipo follow-up, seed 0: JOINT fusion control + frozen dictionary stress test

**Status:** completed. One new formal training unit (`Lipo-JOINT`, 360 epochs), one
frozen dictionary stress test, one read-only edit replay. Parent round
`response_dictionary_small_s0_v1` (commit `185db246`) is read-only; nothing in it
was overwritten or re-run.
**Design:** [`PROTOCOL.md`](PROTOCOL.md) (frozen before the formal run),
`protocol_hash=347fefc5…` (see `summary.json`; it covers the parent config plus
the follow-up decisions).
**Test use:** `test_policy: always`; every Test number is **`DEVELOPMENT_TEST`**
and never selected a checkpoint, a dictionary, a K or a recipe.
**Machine-readable:** `summary.json`, `main_table.csv`, `joint_gain.csv`,
`bootstrap_joint.csv`, `influence_joint.json`, `corrections.csv/json`,
`results/lipo_JOINT/` (curve, predictions, preflight, status, summary),
`results/lipo_RESP/dictionary_stress_k16/` (replay, tables, alpha, response
dimensions, intervention), `analysis/edit_*`.

## 0. Three interpretation corrections (S0 numbers are not rewritten)

**(a) "RESP fits worse than ADD" is not supported by the epoch-63 vs epoch-91
comparison.** Those are two different points on two different curves. At the
*same* epoch, from the committed S0 `curve.csv` ([`corrections.csv`](corrections.csv)):

| epoch | RESP Train RMSE | ADD Train RMSE | difference |
|---:|---:|---:|---:|
| 63 (RESP's best-Val) | 0.34751 | 0.37879 | +0.03128 |
| 91 (ADD's best-Val) | 0.27972 | 0.30178 | +0.02206 |
| 360 | 0.08354 | 0.10019 | +0.01665 |

RESP also has the lower Train RMSE at the same epoch, so a fitting/optimisation
component cannot be excluded from the S0 Lipo RESP-vs-ADD difference.

**(b) "13/16 classes stored exactly" is a class count, not a statement that the
compressed structure is small.** The three compressed `tau` classes hold
442+81+91 = **614 of 702** Train candidates (87.46%) and **76% of all actual
views** (Train 75.9%, Val 77.0%, Test 76.8%). Class count, candidate count,
observed view frequency and held-out structure coverage are four different
quantities; this round reports all four
(`results/lipo_RESP/dictionary_stress_k16/stress_counts.csv`). The fourth one is
small: only 73/1224 Val views (6.0%, 65 distinct fragments) and 69/1182 Test
views (5.8%, 63 distinct fragments) use an `S` fragment that is not a Train word,
so any dictionary comparison is dominated by re-storing Train words — §3.3
therefore reports seen and unseen separately instead of calling the aggregate a
generalisation number.

**(c) D_LATENT text vs code.** The S0 report described `D_LATENT` in a latent
`k(S)` space; the parent code (`rd_dict_run.py`) builds the vectors as
`[a(S), q(S)]`. This round uses the **implementation** as fact and does not re-run
the parent dictionary experiment for a wording difference. The K=64 replay below
reproduces the parent dictionary predictions exactly, so the implemented
definition is the one being stress-tested.

## 1. The JOINT arm

Frozen reuse: parent fragment caches, fixed ≤4 views, encoder, `a_head`,
`b_head`, `U`, `V`, `w`, `fallback_head`, loss, optimiser, 360 epochs, batch 128,
canonical seed-0 init state `ccc8171d…`, Train order `361af4df…`. Only the
operator changed:

```
RESP : T = sum_j w_j tanh(U z_S)_j tanh(V z_C)_j / sqrt(16)
JOINT: h = tanh((U z_S + V z_C) / sqrt(2));  T = sum_j w_j h_j / sqrt(16)
both : s = b(C) + a(S) + T;  F = mean over the same fixed view set
```

Bias-free `concat(z_S,z_C) → 16 (tanh) → 1` joint MLP with the shared `w` readout;
198,262 parameters; no extra module, bias, warm start or dropout change.

Pre-flight executed inside the formal job ([`results/lipo_JOINT/preflight.json`](results/lipo_JOINT/preflight.json)),
all passed:

| check | result |
|---|---|
| parameter layout / count | RESP = ADD = JOINT, 198,262, identical state hash `ccc8171d…` |
| w=0 predictions | JOINT = RESP = ADD on 420 Val and 32 Train molecules, max diff `1.4e-7` |
| data / views / order | fragment cache `ba6e9a41…`, official cache `fe002287…`, order `361af4df…`, 3360/420/420, max 4 views |
| single vs batch | 8 Val molecules, max diff `1.2e-7` |
| 3 optimiser steps | finite; `w` moves (`‖Δw‖=5.4e-4` after 3 steps from 0) |
| loop reproduction | the copied loop as RESP for 1 epoch reproduces S0 `curve.csv` epoch 1 within `2.3e-8` (train_loss, all RMSE/MAE, lr) |

## 2. RESP vs JOINT vs the parent ADD control

Best-Val state per arm; Train/Val/Test are full-split raw-unit numbers
([`main_table.csv`](main_table.csv)):

| arm | best-Val epoch | Train RMSE | Val RMSE | Test RMSE | Val MAE | Test MAE | view dispersion Val |
|---|---:|---:|---:|---:|---:|---:|---:|
| RESP (S0) | 63 | 0.34751 | **0.62874** | **0.67603** | **0.48208** | **0.52352** | 0.187 |
| JOINT (new) | 85 | 0.32122 | 0.66418 | 0.69986 | 0.48900 | 0.53947 | 0.207 |
| ADD (S0) | 91 | 0.30178 | 0.67625 | 0.71208 | 0.50102 | 0.54927 | 0.223 |

Final and last-10 states (descriptive, same ordering):

| arm | final Val | final Test | last-10 Val | last-10 Test |
|---|---:|---:|---:|---:|
| RESP | 0.65182 | 0.69361 | 0.65179 | 0.69314 |
| JOINT | 0.68814 | 0.70746 | 0.68848 | 0.70721 |
| ADD | 0.71055 | 0.72977 | 0.71043 | 0.72961 |

`Gain_product = RMSE_JOINT − RMSE_RESP` (positive = RESP better), best-Val state:

| split | gain | 95% paired molecule bootstrap (2000, seed 1703) | share of draws > 0 |
|---|---:|---|---:|
| Val | **+0.03543** | [+0.00748, +0.06151] | 99.55% |
| Test (follows) | **+0.02383** | — (not a selection endpoint) | — |

Paired influence (appendix, does not change the official metric):
dropping the single most favourable Val molecule (official row 1760, contribution
3.02, 15.7% of the summed molecule gain) leaves **+0.03012**; dropping the top
five leaves **+0.01652**; the Test gain after dropping its top-1 is **+0.01954**.
The interval is sample uncertainty at the two fixed training states, not a
training-seed interval.

Matched-epoch check (the three selected states are different epochs; from the
committed per-epoch curves, [`matched_epochs.csv`](matched_epochs.csv); each cell
is Train / Val / Test raw RMSE):

| epoch | RESP | JOINT | ADD |
|---:|---|---|---|
| 63 (RESP best-Val) | 0.3475 / 0.6287 / 0.6760 | 0.3737 / 0.6763 / 0.6955 | 0.3788 / 0.6822 / 0.7070 |
| 85 (JOINT best-Val) | 0.3007 / 0.6555 / 0.7095 | 0.3212 / 0.6642 / 0.6999 | 0.3157 / 0.6783 / 0.7153 |
| 91 (ADD best-Val) | 0.2797 / 0.6403 / 0.6890 | 0.2970 / 0.6731 / 0.6964 | 0.3018 / 0.6762 / 0.7121 |
| 120 | 0.2088 / 0.6478 / 0.6910 | 0.2337 / 0.6773 / 0.7058 | 0.2553 / 0.7091 / 0.7494 |
| 200 | 0.1298 / 0.6503 / 0.6865 | 0.1550 / 0.6915 / 0.7058 | 0.1527 / 0.7085 / 0.7296 |
| 360 | 0.0835 / 0.6518 / 0.6936 | 0.0908 / 0.6881 / 0.7075 | 0.1002 / 0.7106 / 0.7298 |

Across all 360 epochs RESP has the lower Train RMSE than JOINT in **330** epochs
and the lower Val RMSE in **342** (and lower Val than ADD in 345). So the Val
ordering is not a selection-epoch artefact: RESP both fits Train better and
generalises better at matched epochs, and JOINT is in turn better than ADD on Val
at matched epochs (e.g. 0.6731 vs 0.6762 at epoch 91; 0.6881 vs 0.7106 at 360).
The **Test** ordering is weaker: at epoch 85 (JOINT's own best-Val epoch) JOINT
has the better Test RMSE (0.6999 vs 0.7095) while still being worse on Val
(0.6642 vs 0.6555). The selected-state Test gain of +0.0238 is the pre-declared
endpoint, not an epoch-invariant margin — the main reason the label is
"this config" rather than "product operators".
Curves: [`rmse_curves.png`](rmse_curves.png).

### Decision 1

`PRODUCT_SPECIFIC_SIGNAL_THIS_CONFIG`, `SAMPLE_CONSISTENT_AT_THIS_STATE`.
RESP beats a parameter- and init-matched ordinary joint non-linear fusion by
0.0354 Val (≥ the pre-declared 0.02), Test follows (+0.0238), the interval does
not cross zero, and the sign survives removing the most favourable Val molecule.
So in **this configuration** the product form is not interchangeable with a
general joint MLP, and continuing to study the product induction bias is
justified. The honest limits: the Val margin is matched-epoch robust (342/360
epochs), the Test margin is not (JOINT is ahead on Test at its own best-Val
epoch), the parent round's ESOL result is neither re-run nor re-interpreted here,
and this is one seed, one dataset, one recipe. It says nothing about product operators in
general or about a chemical mechanism.

## 3. Frozen dictionary stress test, K=64 → K=16

Source: S0 Lipo best-Val RESP (epoch 63), checkpoint SHA `c42fc25f…`, all weights
frozen. Replay before any change ([`replay.json`](results/lipo_RESP/dictionary_stress_k16/replay.json)):
native `F(G)` matches the saved S0 predictions within `1.2e-6` (float32 storage),
and rebuilding the three parent K=64 dictionaries with the parent code
reproduces the saved dictionary predictions **bitwise** (`max |Δ| = 0.0`, all
9 cells) with the same RMSE/MAE/fidelity.

Only change: per `tau` class `K = 16` if Train candidates > 64, else
`min(64, candidates)`. The three classes 442/81/91 → 16; the other 13 unchanged.
Words 280 → **136** (= 88 + 3×16), candidates 702 unchanged; panels, `D0`,
`tau` definition, selectors, tie rules and the 64-step Frank–Wolfe projection are
identical. Identity checks
([`identity_checks.json`](results/lipo_RESP/dictionary_stress_k16/identity_checks.json)):
the K=16 selection is exactly the **first 16 of the parent K=64 sequence** for all
three compressed classes and all three selectors, the other 13 classes are
identical to the parent dictionaries, panels and `D0` unchanged.

### 3.1 Full-split metrics ([`stress_table.csv`](results/lipo_RESP/dictionary_stress_k16/stress_table.csv))

Fidelity is `sqrt(mean((F_D − F_RESP)²))` in raw units, not a panel
reconstruction error.

| dictionary | split | RMSE | MAE | ΔRMSE vs RESP | fidelity RMSE | exact-word view share |
|---|---|---:|---:|---:|---:|---:|
| D_RESPONSE | Train | 0.35331 | 0.26970 | +0.00580 | 0.03232 | 0.666 |
| D_RESPONSE | Val | **0.62480** | **0.48077** | **−0.00394** | **0.03937** | 0.632 |
| D_RESPONSE | Test | 0.67848 | 0.52625 | +0.00245 | **0.04011** | 0.630 |
| D_LATENT | Train | 0.38255 | 0.28900 | +0.03504 | 0.14747 | 0.649 |
| D_LATENT | Val | 0.62875 | 0.48221 | +0.00000 | 0.09649 | 0.631 |
| D_LATENT | Test | 0.67939 | 0.52600 | +0.00336 | 0.07653 | 0.630 |
| D_FREQ | Train | 0.35701 | 0.27187 | +0.00950 | 0.05643 | 0.830 |
| D_FREQ | Val | 0.63154 | 0.48489 | +0.00279 | 0.07691 | 0.801 |
| D_FREQ | Test | 0.68038 | 0.52745 | +0.00435 | 0.06952 | 0.802 |

(RESP reference: Train 0.34751, Val 0.62874, Test 0.67603. Supported view
fraction is 1.000 for all three dictionaries; fallback molecule fraction is
0.063/0.093/0.086, unchanged by construction.)

Parent K=64 for the same three dictionaries, replayed
([`parent_k64_table.csv`](results/lipo_RESP/dictionary_stress_k16/parent_k64_table.csv)):
D_RESPONSE fidelity 0.01766/0.02913/0.02547, D_LATENT 0.07744/0.05526/0.04487,
D_FREQ 0.01448/0.02548/0.02266 (Train/Val/Test). Under K=16 the *ordering*
changes: `D_FREQ` was (marginally) the most faithful at K=64 and is now clearly
behind `D_RESPONSE`; `D_RESPONSE` loses the least fidelity going 280 → 136 words
(×1.35 Val) while `D_LATENT` (×1.75) and `D_FREQ` (×3.02) degrade much more.

### 3.2 The three-compressed-class molecule subset, fixed by structure

Molecules with at least one view in the three compressed classes (identical set
for the three dictionaries): Train 2996/3360, Val 368/420, Test 359/420
([`stress_subset_table.csv`](results/lipo_RESP/dictionary_stress_k16/stress_subset_table.csv)).

| dictionary | split | n | RMSE | ΔRMSE vs RESP | fidelity RMSE |
|---|---|---:|---:|---:|---:|
| D_RESPONSE | Val | 368 | 0.61469 | −0.00457 | 0.04206 |
| D_RESPONSE | Test | 359 | 0.66438 | +0.00293 | 0.04338 |
| D_LATENT | Val | 368 | 0.61926 | +0.00000 | 0.10308 |
| D_LATENT | Test | 359 | 0.66547 | +0.00402 | 0.08278 |
| D_FREQ | Val | 368 | 0.62249 | +0.00323 | 0.08216 |
| D_FREQ | Test | 359 | 0.66665 | +0.00519 | 0.07520 |

### 3.3 Where the fidelity difference comes from

By actual view frequency per `tau` class
([`stress_class_fidelity.csv`](results/lipo_RESP/dictionary_stress_k16/stress_class_fidelity.csv)),
Val fidelity of the dominant class (`Z=5, chg=5, arom=0`, 442 candidates, 767 Val
views) is 0.0706 for `D_RESPONSE`, 0.1683 for `D_LATENT`, 0.0980 for `D_FREQ`;
that class carries 64%/91%/42% of each dictionary's squared fidelity change. The
aromatic carbon class (81 candidates, 124 Val views) is 0.0412/0.0493/0.2401;
`D_FREQ` misses it badly at K=16, and that is consistent with its frozen
projection barely improving the panel objective in that class (§3.4) — the
frequent words do not span the response directions there.

Train-seen vs Train-unseen `S` fragments (per view; "unseen" is *not* new-structure
generalisation — the fragment exists, it just was not a Train word)
([`stress_view_groups.csv`](results/lipo_RESP/dictionary_stress_k16/stress_view_groups.csv)):

| dictionary | split | seen views | seen fidelity | unseen views / distinct | unseen fidelity |
|---|---|---:|---:|---:|---:|
| D_RESPONSE | Val | 1151 | 0.0559 | 73 / 65 | 0.1790 |
| D_RESPONSE | Test | 1113 | 0.0588 | 69 / 63 | 0.1840 |
| D_LATENT | Val | 1151 | 0.1360 | 73 / 65 | 0.1888 |
| D_LATENT | Test | 1113 | 0.1275 | 69 / 63 | 0.1960 |
| D_FREQ | Val | 1151 | 0.0729 | 73 / 65 | 0.3988 |
| D_FREQ | Test | 1113 | 0.0877 | 69 / 63 | 0.3159 |

`D_RESPONSE`'s advantage is carried by the **seen** words (Val 0.056 vs 0.136
`D_LATENT` and 0.073 `D_FREQ`); for unseen fragments all three are worse
(0.179–0.399) and frequency selection is the worst (0.399 Val / 0.316 Test).
Only ~6% of Val/Test views are unseen, so every aggregate above is dominated by
Train words — which is why "seen" is not a generalisation claim.

### 3.4 Projection diagnostics and worst samples

`alpha` stays in the simplex (max violation 7e-15); prototypes are one-hot;
supported fragment count 702/702 (0 unsupported).
All three compressed classes × 3 dictionaries exhaust the frozen **64** Frank–Wolfe
steps, i.e. they are *not* converged to the panel optimum
([`alpha_diagnostics.json`](results/lipo_RESP/dictionary_stress_k16/alpha_diagnostics.json)).
Panel residual ratios (after/before): `D_RESPONSE` 0.121/0.126/0.130,
`D_LATENT` 0.549/0.150/0.121, `D_FREQ` 0.110/**0.936**/0.545 for the
442/81/91-candidate classes. `D_FREQ`'s aromatic-carbon projection barely
improves the panel objective at K=16. Per the protocol the iteration budget was
**not** increased to rescue the numbers.

Worst fidelity samples ([`stress_worst_samples.csv`](results/lipo_RESP/dictionary_stress_k16/stress_worst_samples.csv)):
the largest Val changes are three molecules (rows 1647 |Δ|=0.41, 1653 |Δ|=0.20,
1689 |Δ|=0.14) for `D_RESPONSE`; row 1647 is the same molecule the parent stage
already flagged. Largest Test change is row 132 (|Δ|=0.33). These are a handful
of molecules, not a broad shift: excluding the worst Val molecule (row 1647) the
`D_RESPONSE` Val RMSE is 0.61646 vs RESP 0.62874, and excluding Test row 132 it is
0.67320 vs 0.67603.

### 3.5 Response-dimension diagnostics of the three compressed classes

Frozen `D0` reference and Train panel only; `f_S` = Train molecule frequency
([`response_dimensions.json`](results/lipo_RESP/dictionary_stress_k16/response_dimensions.json)):

| class (candidates) | `rho` within/total | rank for 95% energy, freq-weighted `R` | same, row-demeaned | top-1 energy share |
|---|---:|---:|---:|---:|
| `Z=5,arom=0` (442) | 0.124 | 2 | 3 | 0.914 |
| `Z=5,arom=1` (81) | 0.067 | 1 | 3 | 0.960 |
| `Z=6,arom=0` (91) | 0.332 | 2 | 3 | 0.825 |

The response matrix in this reference/panel is nearly rank-1–2 with a large
common component (`m_S` mean |·| = 0.54/0.49/0.47); the class-specific variation
is `rho` = 7–33% of total energy. This describes the frozen reference and the
16-environment panel, not the full data.

### 3.6 Does the trained function use environment variation? (frozen intervention)

Replacing each view's `k(C)` with the Train panel mean `k` of its class, keeping
`a`, `b`, `q`, `w` and every fallback path
([`panel_intervention.json`](results/lipo_RESP/dictionary_stress_k16/panel_intervention.json)):

| split | RMSE as trained | RMSE with panel-mean `k` | Δ | prediction change RMSE |
|---|---:|---:|---:|---:|
| Train | 0.34751 | 0.52499 | +0.17748 | 0.33575 |
| Val | 0.62874 | 0.71326 | +0.08451 | 0.32393 |
| Test | 0.67603 | 0.74656 | +0.07053 | 0.31820 |

All 16 classes have a Train panel, 0 views keep the original `k`. The trained
prediction path therefore depends substantially on *which* environment `C` is
present, not only on the environment's `b(C)`. This is a dependence check, not
evidence that the dependence generalises.

### Decision 2

`RESPONSE_SELECTOR_ADVANTAGE_AT_THIS_RATE`.
At K=16, `D_RESPONSE`'s full-`F` fidelity is 49% (Val) and 42% (Test) below the
better of `D_LATENT`/`D_FREQ` — far beyond the 20% investment reference — and the
actual-property Val RMSE change is −0.0039 (an *improvement*), inside the allowed
+0.0121 (`0.01 × Train_std`). The advantage is not a single-molecule artefact
(seen/unseen split, subset table and per-class table agree in direction), and it
is **not** explained by the fallback path (fallback fraction and fallback scores
are identical across dictionaries). It is, however, a self-fidelity statement on
one frozen state: the property-level differences among the three dictionaries
are small (Val 0.6248 / 0.6287 / 0.6315), and the 64-step projection is
unconverged for exactly the classes under pressure. The honest reading:
**response selection preserves the frozen model's own output best under this
compression**, which is a necessary condition for "response words are better
words" — not yet a task-level advantage. The earlier K=64 conclusion (no
selector advantage) does not carry over to K=16; both are reported.

## 4. Real edit evidence: 823 pairs on official splits

Read-only replay of `lcrc_lipo_edit_supervision_public122_s0_v1/analysis/pairs.csv`
([`analysis/edit_support.json`](analysis/edit_support.json)). This is the 823-pair
public122 catalogue; the parent round used a different, 74-pair amide-only
catalogue, so the two edit sets are not comparable and the parent numbers are not
re-used. Mapping by RDKit
canonical SMILES to the official rows; labels and split re-derived from the
official row id. The catalogue's stored `row_a/row_b` agree with the canonical
mapping for 823/823 (so the rows were verified, not trusted), but its stored
`split_a/split_b` disagree with the official scaffold split for **331/823**
pairs — the independent replay reproduces the repair round's 331 exactly. Its
distribution (Train–Train 501, Train–Test 147, Train–Valid 139, Test–Test 12,
Valid–Test 19, Valid–Valid 5) would have invented 36 non-Train pairs; it is not
used.

Coverage (0 duplicate pairs, 0 self pairs, 704 endpoints, 231 components):

| cell | pairs | endpoints | non-Train endpoints | components | support |
|---|---:|---:|---:|---:|---|
| Train–Train | 791 | 690 | 0 | 228 | sufficient for description |
| Train–Valid | 5 | 9 | 4 Valid | 4 | `LOW_SUPPORT` |
| Train–Test | 27 | 34 | 7 Test | 7 | `LOW_SUPPORT` |
| both non-Train | 0 | — | — | — | — |

Metrics on saved **full F(G)** best-Val predictions; `CE-MAE` averages within
each component and then equally over components with a pair
([`analysis/edit_metrics_by_cell.csv`](analysis/edit_metrics_by_cell.csv)):

| model | cell | pairs | components | pair-MAE | CE-MAE | zero-change CE-MAE |
|---|---|---:|---:|---:|---:|---:|
| RESP | Train–Train | 791 | 228 | 0.3339 | 0.3450 | 0.6420 |
| ADD | Train–Train | 791 | 228 | 0.3064 | 0.3114 | 0.6420 |
| JOINT | Train–Train | 791 | 228 | 0.3120 | 0.3166 | 0.6420 |
| RESP | Train–Test | 27 | 7 | 0.4367 | 0.3568 | 0.7052 |
| ADD | Train–Test | 27 | 7 | 0.4147 | 0.3949 | 0.7052 |
| JOINT | Train–Test | 27 | 7 | 0.4097 | 0.3619 | 0.7052 |
| RESP | Train–Valid | 5 | 4 | 0.2261 | 0.2755 | 0.4675 |
| ADD | Train–Valid | 5 | 4 | 0.2876 | 0.3232 | 0.4675 |
| JOINT | Train–Valid | 5 | 4 | 0.2474 | 0.2860 | 0.4675 |

(Sum over cells counts 239 component entries vs 231 distinct components: 8
components appear in more than one cell — 6 Train–Train+Train–Test and 2
Train–Train+Train–Valid — so cells are not disjoint component sets.)

### Decision 3

Actual independent held-out edit support is **27 pairs / 7 components / 7 Test
molecules** (plus 5/4/4 on Valid) — `LOW_SUPPORT`, described without intervals.
The parent round's statement "no usable held-out edit endpoints" was wrong; the
correct statement is that held-out edit endpoints exist but are too few to
support any claim. On those 7 components the three arms are within 0.04 CE-MAE of
each other in both directions (RESP 0.3568, JOINT 0.3619, ADD 0.3949). This
diagnostic cannot establish a chemical task law, cannot adjudicate the arms, and
did not select anything.

## 5. Labels

| scope | label | basis |
|---|---|---|
| Lipo, product form | `PRODUCT_SPECIFIC_SIGNAL_THIS_CONFIG` | Val +0.0354 ≥ 0.02, Test +0.0238 > 0, top-1 removal +0.0301 > 0 |
| Lipo, sample | `SAMPLE_CONSISTENT_AT_THIS_STATE` | bootstrap [+0.0075, +0.0615] does not cross 0 (one fixed state, not seeds) |
| Dictionary K=16 | `RESPONSE_SELECTOR_ADVANTAGE_AT_THIS_RATE` | fidelity −49% Val / −42% Test vs better baseline; Val RMSE −0.0039 ≤ +0.0121 |
| Dictionary K=16, caveat | `PROJECTION_STEP_LIMITED` | all 3 compressed classes × 3 dictionaries used all 64 frozen Frank–Wolfe steps |
| Edits | `HELDOUT_EDIT_SUPPORT_LOW` | 27 pairs / 7 components / 7 Test molecules; 5/4/4 on Valid |
| Generalisation | `PRODUCT_OPERATOR_GENERALISATION_NOT_SHOWN` | one seed, one dataset, one recipe; no second initialisation |

Not claimed: product operators are universally better; three-way equivalence of
the fusion operators; chemical meaning for the "structural words"; that the
K=16 fidelity advantage is a task-level gain; that edit pairs measure a real
task law.

## 6. Resource use and provenance

| item | value |
|---|---|
| formal units | 1 (`lipo_JOINT`, 360 epochs) |
| execution | `res`, pool `res-gpu1`, process backend, node `a100-2`, driver 550.163.01, torch 2.5.1+cu124, TF32 off |
| run id | `rd-lipo-joint-s0-20261120-150314-08f16566`, commit `15c338e9a014` |
| JOINT training | 597.6 s = **0.166 A100·h** (includes data load and the ~4 s pre-flight) |
| frozen analyses | local CPU: dictionary stress ~2 s, edits ~3 s, aggregation ~2 s (0 A100·h) |
| **round total** | **0.166 A100·h** (hard limit 1.0); wall ≈ 45 min from protocol freeze to report, no GPU retry (hard limit 2 h) |
| identities | init state `ccc8171d…`, Train order `361af4df…`, fragment cache `ba6e9a41…`, official cache `fe002287…`, parent checkpoint `c42fc25f…`, protocol `347fefc5…` |
| parent artefacts | read-only; `results/lipo_RESP`, `lipo_ADD` and the parent report are untouched |

The dictionary stage and the edit replay run on CPU from the committed revision;
their own identity proof is the bitwise replay of the parent saved predictions
(`max |Δ| = 0.0`), so the frozen-weight regime matches the parent unit.

## 7. What this round changes for the next decision

1. **Product form:** a matched ordinary joint MLP does *not* reproduce the RESP
   gain (ΔVal 0.035, ΔTest 0.024, sign survives top-1 removal). The product
   induction bias remains the most plausible source, and worth the next
   controlled step — but the current evidence is one seed, so a second
   initialisation is the cheapest discriminating test, and no such run was
   authorised or started here.
2. **Response words:** under genuine compression pressure the response selector
   is the best-preserving one by a wide fidelity margin, and the K=64
   "no advantage" conclusion is specific to a regime where frequency selection
   already covers most views. Before promoting the claim, the projection budget
   (currently unconverged at 64 steps for the compressed classes) and a task-level
   (not just self-fidelity) endpoint would need to be fixed *a priori*.
3. **Edits:** the held-out edit support exists but is 7 components. Any
   `TASK_RELEVANCE` claim still needs a split-respecting, larger catalogue; the
   Train–Train 791 pairs remain training-endpoint dominated and are not held-out
   evidence.

No extra seed, dataset, K sweep or configuration scan was run; the round stops
here and the next research decision returns to the researcher.
