# REPORT — ZINC topology cross-fit diagnostic v1

**Question (frozen).** Keeping Full's original local environment, dictionaries
and relation channels unchanged, does the existing topology25 channel provide a
*generalisable* error-correction signal beyond the base prediction itself?

**Answer.** No, not with this fixed readout.  The pre-registered gate **FAILS**
in both folds.  The topology channel does carry a held-out conditional signal
`TP > P` (direction-consistent in both folds), but the fixed `ExtraTrees`
residual readout generalises so badly that both `P` and `TP` are worse than the
untouched base.  Per the frozen interpretation, this **closes the fixed
ExtraTrees prompt-value readout only**; it does **not** show that T25 carries no
information, and it does **not** touch the dictionary route.

All numbers below are on the frozen 2000-row outer dev of official-train
molecules (strata 1926 / 65 / 9).  Official test was never instantiated,
loaded or evaluated.  No official-valid re-evaluation was performed.

## 1. Main comparison table (outer dev, equal-weight average before MAE)

| arm | input | dev MAE | Δ vs base |
|---|---|---|---|
| historical F (8000-fit, corrected single-cal, 2000-row inner dev) | — | 0.116365 | *reference only* |
| **base** | `(p_A + p_B)/2` calibrated | **0.150606** | — |
| **P** | `base + q_P(p_base)` | 0.196708 | **+0.046102 worse** |
| **TP** | `base + q_TP(T25, p_base)` | 0.176965 | **+0.026359 worse** |
| const control | `base + median(r_meta)` | 0.150507 | −0.000099 (nothing) |

The historical `F` row uses a different base training size (8000 vs 4000) and is
listed for scale only — it is **not** used to compute any topological causal
gain.  The two 4000-row bases are expectedly weaker than the 8000-row Full.

Frozen gate (`gain_base = MAE(base) − MAE(TP)`, `gain_topo = MAE(P) − MAE(TP)`):

| clause | value | threshold | outcome |
|---|---|---|---|
| `gain_base ≥ 0.003` | **−0.026359** | ≥ 0.003 | FAIL |
| `gain_topo ≥ 0.003` | +0.019743 | ≥ 0.003 | pass |
| TP vs base G0 worsening ≤ 0.001 | **+0.031459** | ≤ 0.001 | FAIL |
| per-fold TP−base gain non-negative | A **−0.045248**, B **−0.039601** | ≥ 0 | FAIL |
| contribution identity / identity checks | true | true | pass |
| **overall** | | | **FAIL** |

Row bootstrap (1000×, seed 20261003, descriptive; does not cover seed noise):
`gain_base` point −0.0264, 95 % [−0.0637, +0.0100]; `gain_topo` point +0.0197,
95 % [−0.0177, +0.0593].  Deleting the frozen base-max-error row (train index
2210, penalty −12) leaves `gain_base = −0.026944`, `gain_topo = +0.018698` —
the failure is not one outlier.

## 2. Group contribution table (`n/N × MAE_group`; identity verified)

| group | n | base MAE (contrib) | P MAE (contrib) | TP MAE (contrib) | TP−base contrib |
|---|---|---|---|---|---|
| penalty 0 | 1926 | 0.123065 (0.118511) | 0.168592 (0.162354) | 0.154523 (0.148806) | **+0.030295 worse** |
| penalty −1 | 65 | 0.210561 (0.006843) | 0.255623 (0.008308) | 0.206868 (0.006723) | −0.000120 better |
| penalty ≤ −2 | 9 | 5.611470 (0.025252) | 5.787997 (0.026046) | 4.763466 (0.021436) | **−0.003816 better** |
| overall | 2000 | 0.150606 | 0.196708 | 0.176965 | +0.026359 worse |

`Σ contrib == overall MAE` to 1e-9 for all three arms.  The picture is clean:
**TP partially helps the ring/severe pools but wrecks the bulk G0 pool**, and the
bulk is ~79 % of the error mass.  This matches interpretation #3's shape, except
the G0 regression is far larger than any severe-row gain, so it is not a
buy signal.

### Ring + severe rows (all listed, no cherry-picking)

Severe (penalty ≤ −2) rows, base → TP absolute error:

| dev row | train idx | penalty | base err | P err | TP err |
|---|---|---|---|---|---|
| 260 | 1238 | −2 | 0.288 | 0.378 | 1.454 (worse) |
| 435 | 2052 | −2 | 6.209 | 6.232 | 2.552 (better) |
| 464 | 2210 | −12 | 22.700 | 23.666 | 21.556 (better) |
| 874 | 4344 | −2 | 2.016 | 2.039 | 1.051 (better) |
| 898 | 4485 | −2 | 0.062 | 0.236 | 0.304 (worse) |
| 1004 | 5050 | −5 | 13.821 | 13.657 | 12.651 (better) |
| 1009 | 5093 | −2 | 0.580 | 0.375 | 0.021 (better) |
| 1499 | 7507 | −2 | 4.665 | 4.557 | 2.591 (better) |
| 1535 | 7659 | −2 | 0.162 | 0.952 | 0.690 (worse) |

The severe-row improvement is real but comes with a much larger, diffuse G0
regression.  All 74 ring rows are in `DECISION.json::ring_and_severe_rows`.

## 3. Coverage / residual conflict (cheap, no new fits)

| diagnostic | fold A (meta = B) | fold B (meta = A) |
|---|---|---|
| exact T25 train-class coverage of dev | 97.55 % (1951/2000) | 98.00 % (1960/2000) |
| mean \|dev_resid − class-median meta_resid\| on covered rows | 0.1409 | 0.1440 |
| median of the same | 0.0982 | 0.0998 |
| nearest-neighbour penalty agreement (all dev) | 0.9955 | 0.9990 |
| nearest-neighbour penalty agreement (severe) | 0.5556 | 0.7778 |

The last severe row (penalty −12, index 2210) has **no** exact train class and
its nearest meta neighbour is 30–35 std-units away with penalty −1.

Reading: coverage is **not** the failure mode — ~98 % of dev rows land in an
exact train T25 class.  But within an exact class the residual varies by
≈0.10–0.14, the same order as the base G0 error, so `r` is **not a function of
the T25 class**.  This is consistent with the residual carrying chemistry/base
error that T25 does not encode.  Per the frozen boundary, *coverage sufficient
but residual conflict* does not mean T25 is useless, and *coverage insufficient*
is not the explanation here.

## 4. Why `P` is worse than `base` (readout-overfitting diagnosis)

`P` is a single-feature, `min_samples_leaf=1` ExtraTrees fit to the meta-fold
residual `r = y − p_base_cal`.  With one feature and leaf=1 the fit is a
piecewise-constant nearest-neighbour interpolation of a heavy-tailed residual
(median ≈ 0.002, but severe rows up to ~22).  It has near-zero bias on the meta
fold and maximum variance off it: `P` loses 0.046 MAE against the untouched
base, i.e. the readout's variance dominates its signal.  `TP` has 26 features
and does better than `P` in both folds (`+0.028`, `+0.016`), which is the only
positive, direction-consistent signal in this round — but it is still not enough
to beat `base`.

## 5. Direct answers

**1. Did severe-cycle error actually improve?  What did topology add over the
prediction value itself?  Where, and did it hurt G0?**
Yes, partially and locally: the severe-group contribution fell from 0.025252
(base) to 0.021436 (TP), with real per-row gains (rows 435, 464, 874, 1004,
1009, 1499) and some regressions (260, 898, 1535).  Over the prediction value
itself, TP beat P by +0.019743 on dev, +0.0280 (fold A) and +0.0162 (fold B) —
i.e. the topology features carry a held-out residual signal that the scalar
prediction does not.  But G0 worsened by +0.031459, which is ~8× the severe
contribution gain, so the net is a loss.

**2. Why are “topology already carries information” and “the current Full
already uses it” two different propositions?  What new evidence did this round
add?**
A feature can be predictive of a target/ residual while the trained network
never extracts it: `TP > P` is measured out-of-fold, on data neither the readout
nor the base trained on, and the Full's own topology encoder is *shared and
training-size-limited* (4000 rows), so nothing forces it to realise the signal.
The new evidence is exactly that cross-fit conditional increment (`TP > P` in
both folds) plus the coverage table showing the information is present and the
residual is not a pure function of it.  This is a property of the **feature**,
not a claim that the **network** uses it.

**3. Are the Full's two-layer dictionaries and the new D end-to-end trained?
Driven by which losses?**
Yes.  `mechanism_check.json` (post-prep **initial** state, one real backward):
Full structural `D` (65×32) is an `nn.Parameter` in the Adam group and receives
both the L1 task gradient (6.37e-5) and the structural-reconstruction gradient
(5.27e-3); Full task dictionary `D_L`/`V_L` (144×288 / 288×144) are
`nn.Parameter`s receiving task gradients (0.560 / 0.589) through the unrolled
tied-ISTA bridge and **no** reconstruction gradient; arm-D `local.dictionary`
receives task (0.082) + relative-recon (0.0062) gradients and `local.value`
receives the task gradient (0.432).  This is end-to-end training, **not** a
claim about sparsity or transferability, and it is an initial-state check — no
final-soup checkpoint existed for the historical D (the 1.26 % reconstruction
number is the post-prep initial model).

**4. Next single action worth buying, and what remains open?**
See `DECISION.md`.  In short: the only next purchase is a **regularised
topology-class residual correction** (shrink class-level residual means toward
zero, no leaf-1 memorisation) under this same cross-fit protocol; everything
else (larger dictionary, extra topology features, third base, extra seeds) stays
closed.  Open competing explanations: readout variance; residual genuinely not a
function of T25; 4000-row base residuals being too heteroscedastic; and T25
being useful as an internal representation rather than an external residual
feature.

## 6. Evidence boundaries (must not be overstated)

* This is a **fixed-readout** negative result.  It closes the ExtraTrees
  leaf-1 prompt-value readout, not the topology channel, the dictionary route,
  or the overall architecture direction.
* No official-valid or official-test score is produced or converted.
* The ring/severe improvement is genuine but not generalised to G0.
* `TP > P` is direction-consistent but its bootstrap interval covers zero; it is
  a hint for the next design, not a bought gain.
* The two 4000-row bases are weaker than the 8000-row Full; the residual
  distribution differs from the historical 8000-row one, which is exactly why
  the cross-fit is kept base-matched (no OOF head is attached to the historical
  F).