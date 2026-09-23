# PEC-C1 — analysis (Pure Environment Composition Confirmatory)

Round **PEC-C1** · study `zinc-context-gap` · protocol `pec_c1`.
Pre-registration: [`pec_c1_preregistration.md`](pec_c1_preregistration.md)
(frozen before the run). Implementation note:
[`pec_c1_implementation.md`](pec_c1_implementation.md).

> **PEC-v0 Gate 1 remains a historical frozen FAIL. PEC-C1 does not
> retroactively pass or recalibrate it.**

Remote compute: `hxy@a100-2`, repo `/home/hxy/cy/research`, commit
`bf8b5a628d99eedaf52f0b6e95e79cf583c4ba7b` (clean; `git_dirty = false`),
2 × NVIDIA A100-SXM4-40GB, CUDA 12.4, torch 2.5.1+cu124, Python 3.12.14.
**`official_test_loaded = false`** in every artifact; only splits
`["train", "val"]` were ever read. No historical baseline was retrained.

---

## 1. Frozen verdict

```
PURE_ENV_COMPOSITION_ABSOLUTE_WEAK
seed1_authorized = false
```

| quantity | value |
|---|---|
| `M_CK` (SparseDict, frozen dictionary, Top-5 soup) | **0.160971** |
| `M_CD` (DenseRole, trainable, Top-5 soup) | **0.151767** |
| `Δ_dict = M_CD − M_CK` (positive ⇒ sparse better) | **−0.009204** |
| `min(M_CK, M_CD)` | **0.151767** → band `> 0.145` |

Frozen cases: `A_absolute_weak = true`, `B_dense_dominates_sparse = true`,
`C/D/E = false`. Precedence is `A > B > D > C > E`, so **Case A** is the
verdict: the pure no-MP environment→static-composition architecture does not
reach the pre-registered viability band even at full data, and seed 1 is not
purchased.

## 2. Arms and provenance

| | CK — SparseDict | CD — DenseRole |
|---|---|---|
| role coordinate | `tied-IHT_{K16,s4}(D)`, `D` **frozen** K-SVD | learned dense `Linear(11→16)`/`Linear(15→16)`, `Dᵀ`-initialized |
| params total / trainable | 94,049 / **93,633** | 94,049 / **94,049** |
| role params (total / trainable) | 416 / **0** | 416 / 416 |
| dictionary drift `max|D − D₀|` after training and soup | **0.0** | — |
| dense-map drift `max|M − M₀|` | — | 0.375 / 0.613 |
| best official-valid MAE | 0.169599 @ epoch 210 | 0.159782 @ epoch 235 |
| **Top-5 soup official-valid MAE** | **0.160971** | **0.151767** |
| soup members | [191, 198, 210, 220, 238] | [178, 199, 219, 235, 239] |
| soup member MAEs | 0.169599 … 0.172235 | 0.159782 … 0.162802 |
| train MAE at best epoch | 0.090447 | 0.081518 |
| train–valid gap at best | 0.079151 | 0.078264 |
| train MAE min / final | 0.080633 / 0.084890 | 0.073690 / 0.078045 |
| wall | 832.4 s (3.47 s/epoch) | 765.5 s (3.19 s/epoch) |
| peak GPU memory | 84.9 MB | 89.1 MB |
| GPU | A100-SXM4-40GB | A100-SXM4-40GB |
| seed / epochs | 0 / 240 fixed, no early stop | 0 / 240 fixed, no early stop |

The 240-epoch fixed schedule was respected exactly; CK's minimum is at epoch
210 and CD's at 235, and the record below shows the horizon does not affect the
verdict.

## 3. Q1 — Absolute viability: **band `> 0.145`**

`min(M_CK, M_CD) = 0.151767 > 0.145`. The pure environment→static-composition
class is **not viable** at the pre-registered absolute level, even after the
full official-train → official-valid regime.

Historical context (orientation only, **not** gates — different computation
class; never re-run):

| reference | soup |
|---|---:|
| strict-static S0 seed 0 | 0.140794 |
| strict-static S0 seed 1 | 0.136423 |
| B-Null | ≈ 0.123 |
| B-Full | ≈ 0.119 – 0.118 |

Deficits against the same-family strict-static S0 seed-0 reference:

| | soup | Δ soup vs S0 soup |
|---|---:|---:|
| CK | 0.160971 | **+0.020177** |
| CD | 0.151767 | **+0.010973** |

### 3.1 Where the deficit comes from (descriptive)

Within the same data / selection protocol, comparing each arm's best epoch with
S0 seed-0's best epoch (S0: best 0.145674 @163, train-at-best 0.075434, gap
0.070241):

| arm | total deficit | fit component | generalization component |
|---|---:|---:|---:|
| CK | +0.023924 | +0.015014 (63 %) | +0.008911 (37 %) |
| CD | +0.014108 | +0.006084 (43 %) | +0.008023 (57 %) |

Both components are real. The pure class **cannot even fit the training set as
well** as the strict-static backbone (train minimum 0.0806 / 0.0737 versus
0.0676), *and* it generalizes slightly worse. So this is a
representation/optimization ceiling, not a pure overfitting story. The
decomposition is descriptive — S0 and PEC are different architectures with
different parameter counts — and is not a causal attribution.

### 3.2 The fixed horizon does not change the verdict

* CK: block minima over the last five 24-epoch blocks are
  `0.1797, 0.1739, 0.1722, 0.1696, 0.1706` — the minimum is at epoch 210 and the
  tail is flat-to-worse.
* CD: block minima `0.1665, 0.1658, 0.1602, 0.1628, 0.1598` — the minimum is at
  epoch 235 and the whole last 48 epochs move within `0.003`.
* To leave the weak band CD would need `−0.0148` (0.159782 → ≤0.145), i.e.
  several times the entire observed tail movement. Extending training is
  forbidden by the pre-registration **and** could not flip Case A.

## 4. Q2 — Sparse dictionary vs parameter-matched DenseRole: **worse**

```
M_CK − M_CD = +0.009204   (SparseDict is 0.0092 MAE worse)
```

i.e. `Δ_dict = −0.009204`, past the `0.003` materiality threshold in the
direction of the dense control. Both `dense dominating` and `absolute weak`
fire; Case A takes precedence.

Because CK is deliberately handicapped (pre-registration D1/D2: CK's 416 role
parameters are **frozen** — `requires_grad = False`, excluded from the
optimizer, drift exactly `0.0` — while CD's 416 role parameters are trained),
the correct statement is:

> **A frozen K-SVD sparse structural-role dictionary is materially worse than a
> parameter-matched trainable dense role map inside this pure composition
> class.** The result says nothing about *task-coupled* dictionaries: a
> trainable dictionary is a different hypothesis, reserved for a future
> `PEC-C2`, and this round does not estimate it.

This is exactly why the `PURE_ENV_COMPOSITION_ABSOLUTE_WEAK` verdict (Case A)
outranks `ENV_COMPOSITION_SUPPORTED_DENSE_NOT_DICT` (Case B): the architecture
as a whole fails the absolute band, so closing the dictionary route is
secondary — and the dictionary route is only closed *in its frozen form*.

## 5. Q3 — Mechanism integrity: **all four checks pass, decisively**

Evaluation-only interventions on the CK Top-5 soup (never used for selection):

| intervention | valid MAE | degradation | mean abs prediction shift | required |
|---|---:|---:|---:|---|
| chemistry-placement shuffle | 1.737075 | **+1.576104** | 1.710692 | > 0 |
| neutral dictionary | 2.128386 | **+1.967415** | 2.093377 | > 0 |
| composition relation shuffle | 1.553879 | **+1.392907** | 1.507191 | > 0 |

(CD for reference: chem-shuffle degradation `+1.433714`, relation shuffle
`+0.874643`.)

* **chemistry placement is load-bearing** — destroying only the
  structure↔chemistry assignment with topology and the atom/bond multisets held
  fixed destroys the model (`+1.576`). Stronger at full scale than in the PEC-v0
  Gate-2 screen (`+0.835`).
* **the dictionary branch is alive** — replacing `D` with a matched random
  unit-normalized dictionary shifts predictions by 2.09 and costs `+1.967` MAE.
  The CK model genuinely depends on the sparse code; the dictionary is *used*,
  it just is not *better* than the dense map.
* **composition is not relation-invariant** — permuting only `ρ_ij` within each
  graph, with environments and pair endpoints fixed, costs `+1.393`. This check
  is new in PEC-C1 (Gate 2 had BAG/SHUFFLE) and it passes decisively.
* **no message passing / no recurrence / no mixed bypass** — the frozen PEC-v0
  contract is unchanged (`pec_v0.py` is byte-identical to PEC-v0), and all
  Gate-0 correctness checks still hold under PEC-C1's import.

So: **the mechanism is real, the capacity is not enough.** The failure mode is
not inertness, collapse, or a dead branch.

## 6. Label-free dictionary diagnostics (report-only, no gate)

K-SVD on the full official train (1,418,500 node / 1,232,844 edge occurrences,
1930 s, `DICT_SEED = 20260924`), `K = 16`, `s = 4`, tied-IHT 10 steps:

| role | `E_rec` train | `E_rec` valid (diagnostic) | used atoms | dead atoms | max l0 | exact `s` |
|---|---:|---:|---:|---:|---:|---|
| node | 0.024673 | 0.024777 | 13 / 16 | 3 | 4 | true |
| edge | 0.074966 | 0.075086 | **8 / 16** | **8** | 4 | true |

`E_rec train ≈ E_rec valid` for both roles: the dictionary generalizes across
the official split, so it is not overfit. The **edge dictionary degenerates to 8
effective atoms at `K = 16`** on the full corpus (the same overcompleteness
effect that produced PEC-v0 Gate 1's `edge_used = 11 < 12` on 8,000 molecules).
This is reported for the record; `K` may not be changed, and no rescue is
performed.

## 7. Q4 — Next decision

```
verdict            = PURE_ENV_COMPOSITION_ABSOLUTE_WEAK
seed1_authorized   = false
```

Consequences, exactly as pre-registered:

* The environment / static-composition **mechanisms** remain supported
  (chemistry placement, dictionary dependence and relation sensitivity all
  load-bearing; no MP / recurrence / bypass).
* The **current pure no-MP architecture lacks absolute capacity** in the full
  ZINC regime.
* **STOP.** No seed 1, no more seeds, no `K`/`s` rescue, no reader or
  recurrence widening, no threshold change, no task-coupled dictionary.
* A future `PEC-C2 task-coupling` round is **not** authorized by this result:
  the pre-registration makes task coupling conditional on a stable absolute pure
  architecture, which we do not have.

## 8. Purity / discipline audit

* `pec_v0.py`, `pec_v0_gate0.py`, `zinc_pec_v0.py` are **byte-identical** to
  PEC-v0 (empty `git diff`); the architecture, feature definition, `K`/`s`,
  tied-IHT, static composition and reader are untouched.
* The K-SVD dictionary is fit on official train only and is **never** refit,
  never fit on valid, and never part of the optimizer.
* Official valid was used only for checkpoint selection and the Top-5 soup,
  exactly as pre-registered. No valid-informed architecture, threshold, `K`/`s`
  or reader choice was made.
* Official test was never loaded; the loader raises for `"test"` and a test
  pins that.
* 34 targeted tests pass (`test_pec_c1.py`), 41 together with `test_pec_v0.py`.
* No detached run was left unattended: every launched job was followed to its
  `.exit` file in-session.

## 9. Limitations

* One seed (0), as pre-registered. The `M_CK − M_CD = 0.0092` gap is large
  relative to the documented seed spread (≈0.004–0.007), but a single seed
  cannot establish "stably worse"; the verdict rests on Case A (absolute band),
  which is a far larger margin (0.1518 vs 0.145).
* CK's dictionary is frozen while CD's role map is trainable (D1/D2), so the
  dictionary comparison is conservative **against CK** and, if anything,
  understates a task-adapted dictionary's potential.
* `K_V = K_E = 16` is overcomplete for the 11-D / 15-D bases; the edge
  dictionary uses only 8 atoms. This is inherited from PEC-v0 and deliberately
  not repaired.
* The absolute comparison to strict-static S0 / B-Null / B-Full is
  orientation only: different computation classes, not gates.
* The fit/gap decomposition is descriptive, not causal.
* Single device regime (no DDP), one A100 per arm, run in parallel; both arms
  are small enough (peak < 90 MB) that contention is a non-issue.
