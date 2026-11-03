# DECISION — LCRC 远距离化学角色绑定瓶颈验证 v1

Round: `lcrc-nonlocal-binding-bottleneck-v1`
Decision authority: the pre-registered rule in `PROTOCOL.md` §4.7 (frozen before the run,
never relaxed). All values below come from `analysis/metrics.json`,
`analysis/bootstrap.json`, `analysis/structural_witness.json` and
`analysis/feature_margin_checks.json`.

```
DECISION:            NO_OBSERVED_INCREMENT
PHASE A (expressivity):  POSITIVE — constructive, reproducible witness
PHASE B (task relevance): NEGATIVE — no observed predictive increment over the controls
ROUND STOPS HERE. No next-phase experiment is authorised or started.
```

---

## 1. Stop / continue gate: Phase A

| # | Pre-registered check | Result | Verdict |
| --- | --- | --- | --- |
| A1.1 | both molecules sanitise | 18 atoms each, formula `C16H33NO` | PASS |
| A1.2 | identical formula | `C16H33NO` = `C16H33NO` | PASS |
| A1.3 | non-isomorphic under element/bond labels | labelled backtracking: **not** isomorphic (1 221 nodes); independent cyclic-distance invariant 6 ≠ 8 | PASS |
| A1.4 | identical radius-2 labelled environment multiset | 18 roots, 7 root signatures, identical counts | PASS |
| A1.5 | identical 2048-bit Morgan radius-2 fingerprint | both 15 on-bits, bit-identical | PASS |
| A1.6 | RDKit N–O shortest distance 6 / 8 | 6 / 8 | PASS |
| A1.7 | production `lipo_v1` `global` 42-d identical | 0/42 columns differ, max abs diff 0.0 | PASS |
| A1.8 | real LCRC model checks local multiset / global readout / output | see A2 rows | PASS |
| A2 | `h` row multiset (Hungarian, min cost) | cost 0.0 for A0/C1 × seeds 0,1,2 | PASS |
| A2 | `contrib`, `rho`, `y` equivalence ≤ 1e-5 | max \|Δy\| = 9.54e-7, max \|Δcontrib\| = 9.54e-7, max \|Δrho\| = 6.0e-8 | PASS |
| A2 | random atom-reordering invariance | max \|Δy\| = 1.90e-6, `h` matching cost 0.0 | PASS |
| A3 | REAL differs for A/B; N/O at bin 6 vs 8; NULL identical | REAL L1 = 0.0988; N/O at `6` vs `8`; NULL max abs diff 0.0 | PASS |

Additional observation (not a gate): the production per-atom local **input** rows are
identical as a multiset (Hungarian cost 0.0, 6/6 model instances). The collision is
therefore already present at the model's input, not only at its output.

**Stopping rule was checked and not triggered** → Phase B was allowed to run.

---

## 2. Item-by-item verdict against the six pre-registered conditions (Phase B)

| # | Condition | Threshold | Observed | Verdict |
| --- | --- | --- | --- | --- |
| 1 | mean `REAL − NULL` ROC-AUC (3 folds) | ≥ **+0.015** | **−0.005216** | **FAIL** |
| 2 | all three folds `REAL − NULL` ROC-AUC | strictly > 0 | [−0.010544, −0.000337, −0.004768] | **FAIL** |
| 3 | mean `REAL − BASE` ROC-AUC (3 folds) | ≥ **+0.010** | **−0.002637** | **FAIL** |
| 4 | mean `REAL − NULL` AP difference | ≥ 0 | +0.001699 | PASS |
| 5 | paired-bootstrap 95% CI lower bound of `REAL − NULL` | > 0 | −0.009875 | **FAIL** |
| 6 | all data / feature correctness gates | all pass | 15/15 checks pass (train rows sha256, frozen cache fingerprint, label discipline, margin conservation, folds, no scaffold leakage, ECFP4 purity, …) | PASS |

**2 of 6 passed.** The pre-registered classification then depends only on the sign of the
mean `REAL − NULL` AUC difference:

* not all six passed → cannot be `PROMOTE_INFORMATION_SIGNAL`;
* correctness gates passed → not `INFEASIBLE`;
* mean `REAL − NULL` ROC-AUC = −0.005216 **≤ 0** → **`NO_OBSERVED_INCREMENT`**
  (not `WEAK_OR_INCONSISTENT_SIGNAL`).

---

## 3. Supporting statistics

| Quantity | Value |
| --- | --- |
| Mean ROC-AUC `BASE` / `BASE_NULL` / `BASE_REAL` | 0.794281 / 0.796861 / 0.791644 |
| Mean AP `BASE` / `BASE_NULL` / `BASE_REAL` | 0.353671 / 0.346835 / 0.348534 |
| `REAL − NULL` AUC mean, bootstrap 95% CI (1 000 valid replicates, 0 rejections) | −0.005211, **[−0.009875, −0.000548]**, 1.3% of replicates > 0 |
| `REAL − BASE` AUC mean, bootstrap 95% CI | −0.002488, [−0.008072, +0.003325], 19.1% > 0 |
| Selected `C` (BASE-only inner selection) | 0.1 in all three outer folds |
| Official Train | 32 901 rows, 1 232 positives (3.74%) |
| Feature coverage | 99.06% of molecules have a distance-≥5 pair; all 21 type pairs and all 6 bins carry mass |

Interval semantics: sampling uncertainty **at the fixed fitted models and fixed folds**;
re-splitting and re-training variance is **not** included.

---

## 4. What this decision does and does not assert

* **Asserts:** (a) a reproducible constructive blind spot of the current LCRC-A input/output
  for a pair of valid, non-isomorphic molecules that the type-distance statistics separate;
  (b) inside the official MolHIV Train split, no observed positive increment from the
  type-distance correspondence over both the marginal-matched NULL control and the
  ECFP4+g42 baseline, under the pre-registered estimator, splits and thresholds.
* **Does not assert:** that the information is useless in general; that REAL is harmful by
  mechanism; that LCRC cannot benefit; that the task would not benefit at a larger positive
  count or with a different estimator family. None of these was tested, and this round is
  forbidden from testing them after seeing the result.
* All numbers are **exploratory Train-only evidence**, not independent confirmation. The
  official Valid/Test labels, predictions and metrics were never read.

---

## 5. Actions explicitly not taken (pre-registered prohibitions)

* No threshold, distance bin, element class, feature, model, hyper-parameter, seed or split
  change after seeing any result. No extra seeds. No fourth model.
* No GPU work (0 GPU·h used), no gradient training, no new LCRC/GNN/Transformer variant.
* No atom-pair attention, distance embedding, global two-body module, atom-role search, or
  additional binning.
* No official Valid/Test scoring of any kind.
* No modification of any historical track's code, configuration, data or report.

**The round is closed.** Any further architecture design or experiment plan is the
researcher's decision and requires a new pre-registration.
