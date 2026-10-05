# REPORT — `zinc_cycle_prototype_transfer_cpu_v1`

Train-only cycle **prototype lookup** on the exact T25 classes the frozen `Q` sees, compared to
the frozen COMP+Q reference, on the reused exposed official-valid (1000 rows). CPU only, no
training, no GPU, no remote job, official-test never read.

---

## Direct answers

### 1. Is the COMP+Q source and published result reproduced? What else changed?

**Yes, reproduced exactly.** The frozen source round
(`zinc_component_supervision_fulltrain_confirmation_seed0_v1`, execution commit
`947d2837c4936ef1276fcbbf0e9189e7d44ace23`) replays:

| check | reproduced | published |
|---|---|---|
| train COMP `y_cal` MAE | 0.03859298209673725 | 0.03859298209673725 |
| train COMP `g_cal` MAE | 0.028076743795956326 | 0.028076743795956326 |
| valid COMP `y_cal` MAE | 0.11740618350630393 | 0.11740618350630393 |
| valid COMP `g_cal` MAE | 0.08930046045603672 | 0.08930046045603672 |
| valid SUM `y_cal` MAE | 0.12265322754724184 | 0.12265322754724184 |

`y = g + c` (≤ 1.4e−17) and `g = ell + s` (≤ 4.4e−16) hold; the frozen `Q_raw_soup` state
replayed on the freshly built current T25 reproduces the cached `q_raw` with max abs diff `0.0`
on both train and valid. Source SHA-256 matches `frozen_eval_manifest.json`.

**What changed this round:** only (i) the cycle route — on train-consistent exact-key hits the
train class median `c` replaces the frozen `Q` output — and (ii) the bias produced by the same
fixed train-only calibration rule (`b_H = median(y_train − h_raw_train − q_H_raw_train)`).
The COMP body `h`, the `Q` weights, prep, constants and targets are untouched. The `g`
prediction is identical and only replayed; no new chemical `g` gain is claimed.

### 2. How many train-consistent classes did Q miss? Valid coverage? New conflicts? Which route for severe errors?

* Train: **355** exact T25 classes, **350 consistent** (all members one integer `k`), **5
  conflict** classes, **151 singletons**. Consistent-class `c` span is exactly 0; `c` equals
  `(k − mu_cycle)/sigma_cycle` to full precision. `Q`'s train `q`-vs-`c` MAE is **0.011506**,
  i.e. it does miss train-consistent class signal (the prototype reaches 0.005207 on train).
* Valid coverage: **988 / 1000 `CONSISTENT_HIT`**, **0 `TRAIN_CONFLICT_FALLBACK`**, **12
  `UNSEEN_FALLBACK`**. Among the 988 hits the train-class-`k` vs valid-true-`k` match rate is
  **988/988 = 1.0**; **0 new conflicts**.
* Severe errors: B's largest valid error (`valid:0172`, k=−6, |err| = 20.81) is a
  `CONSISTENT_HIT` (singleton train class, same T25). The other large errors
  (`valid:0935` k=−2 |err| = 5.27, plus 11 more) are **`UNSEEN_FALLBACK`** — the rule has no
  coverage there and makes no change.

### 3. H's whole-valid y raw/cal, gain/CI/gate? G0 harm? Bias-only or single-row?

| quantity | B (COMP+Q) | H (prototype) |
|---|---|---|
| valid `y_cal` MAE | 0.1174061835 | **0.0966943750** |
| valid `y_raw` MAE | 0.1166162887 | **0.0958850152** |
| `b_y` | −0.0116300024 | −0.0114467995 |

* overall cal gain (point) = **+0.0207118085**; 95% CI **[−0.0002260, +0.0623398]** (lower < 0);
* overall raw gain = **+0.0207312735** (> 0);
* G0 cal worsening = **+0.0000116891** (≤ 0.001, no harm);
* gate: `1_ge_0.003` ✓, `2_ci_lower_gt_0` ✗, `3_raw_gt_0` ✓, `4_G0_worsening_le_0.001` ✓ →
  **`PROTOTYPE_DEPLOY_SUPPORT = false`**.

It is **not bias-only** (fixed-`b_B` H MAE 0.096719 vs B 0.117406, raw gain still positive), but
it **is** single-row dominated: `valid:0172` contributes +20.74 of the +21.17 positive sum
(share 0.980); excluding B's worst row the gain is **−0.0000288**. Markers:
`SINGLE_ROW_DOMINATED = true`, `TARGETED_REPAIR_ONLY = true`.

### 4. Does q→c improvement convert to y improvement? Cancellation? Is 0.09 observed?

Conversion is **almost entirely lost to cancellation**:

* on the 987 `CONSISTENT_HIT` rows excluding 0172: `q` MAE vs `c` falls **0.001139 → 8.6e−10**,
  but `y_cal` MAE moves **0.087806 → 0.087835** (slightly worse);
* the body error `e_g = h − g` has **MAE 0.087004** and dominates; `e_c_B = q_B − c` is only
  **0.001139**; the triangle gap is **−0.001149** for B (Q was cancelling a little of `e_g`) and
  **−6.9e−10** for H (cancellation removed). `Σ(|e_g| + |e_c_B|) = 0.088143` versus the true
  `|e_g + e_c_B| = 0.086994` — component MAEs are not a budget.

`H`'s whole-valid `y_cal` is **0.09669**, so `y < 0.09` is **not observed**
(`BENCHMARK_TARGET_OBSERVED = false`). The B oracle `h + c + b_B` is 0.089325, still above 0.09.

### 5. Function-fit miss, input conflict, uncovered, or still unlocalized — and next responsibility?

All three, on their evidence-corresponding subsets:

* **Function-fit miss (covered, resolved rows):** on train-consistent, valid-label-consistent
  exact-hit classes, the prototype lowers `q` error to ~0 while the original `Q` errs (0.0011):
  the fixed small head missed a class signal already present in train. This is *not* an input
  information deficit for those class inputs. But it does **not** convert to `y` gain because the
  body error dominates and the small cycle cancellation is lost.
* **Input conflict (train conflict):** 5 train classes have non-unanimous `k` at the same T25
  (e.g. class 270: k∈{−2, 0}; class 280: k∈{−6, 0}); the current T25 cannot simultaneously fit
  those train `c`. A same-input capacity increase cannot remove their class-constant L1 floor.
* **Uncovered:** the 12 `UNSEEN_FALLBACK` valid rows (largest Q errors among them) get **no**
  candidate information. This closes only the exact-lookup rule's action there; it does **not**
  falsify a learnable cross-class cycle regularity, and a miss is not attributed to missing
  input information.

**Next responsibility (one question, not executed):** since most relevant classes are covered and
cycle prediction is already accurate while `y` is still poor, the next question returns to the
**chemistry body error `h − g` and its cancellation** with the cycle term. Explicitly stopped:
scanning coefficients/seeds, changing the coverage gate/key/threshold, nearest-neighbour or
smoothing rescue, re-confirming COMP, reopening dictionary/binding/reader side branches, and
opening official-test.

---

## A. Train table and the unidentifiable part

* 10,000 train rows → **355** exact float32 T25 classes (collision-free; `-0` absent, no NaN/Inf).
* **350 consistent**, **151 singletons**, **5 conflict** classes; support by `k` group
  (a conflict class is counted under its minimum `k`):

| group (by class k_min) | classes | rows | conflict classes | singletons |
|---|---|---|---|---|
| k=0 | 245 | 9620 | 0 | 92 |
| k=−1 | 82 | 327 | 2 | 41 |
| k=−2 | 21 | 41 | 1 | 13 |
| k≤−3 | 7 | 12 | 2 | 5 |

* Train class-constant L1 floor (per-class median then mean |err|): **0.005200** overall
  (this is the train empirical fit floor for this exact input, not a population error floor).
* Conflict classes (T25 shared by different `k`):

| class | n | k values | members |
|---|---|---|---|
| 73 | 2 | {−1, 0} | train:0189, train:8419 |
| 88 | 2 | {−1, 0} | train:3741, train:7491 |
| 249 | 5 | {−5, 0} | train:0593, 2447, 2472, 5368, 9913 |
| 270 | 2 | {−2, 0} | train:2232, train:3626 |
| 280 | 2 | {−6, 0} | train:1270, train:1424 |

  These show the current T25 does not uniquely determine the train cycle label — not that `Q`
  lacks capacity. No valid row routes to `TRAIN_CONFLICT_FALLBACK` (0), so this does not move the
  valid result.

* LOO coverage (diagnostic only): 151 singletons lose support when the row is removed; 9845 rows
  stay in a consistent class; 4 rows become newly conflicting. Train self-match `q_H` error is
  near 0 by construction and is not generalization.

## B. Valid coverage and new conflicts

Routing × group (`q` errors vs `c`; `n`):

| route | group | n | q_B MAE | q_H MAE | y gain (cal) |
|---|---|---|---|---|---|
| CONSISTENT_HIT | overall | 988 | 0.022130 | 1.7e−09 | +0.020963 |
| CONSISTENT_HIT | k=0 | 960 | 0.000397 | 2.2e−11 | −0.000012 |
| CONSISTENT_HIT | k=−1 | 26 | 0.022840 | 3.0e−08 | +0.001067 |
| CONSISTENT_HIT | k=−2 | 1 | 0.148827 | 5.3e−08 | −0.044722 |
| CONSISTENT_HIT | k≤−3 | 1 | 20.740807 | 8.1e−07 | +20.740623 |
| UNSEEN_FALLBACK | overall | 12 | 0.802062 | 0.802062 | +1.1e−16 |
| TRAIN_CONFLICT_FALLBACK | overall | 0 | — | — | — |

The `k≤−3` CONSISTENT_HIT is `valid:0172`. Excluding it, `k=0` and `k=−2` hits are **negative**
(slight worsening) and `k=−1` is marginally positive. Route contributions add back exactly to
the overall (`addback.json`, `addback_ok = true`). For raw fallback routes the prediction
difference is exactly 0; for cal it is the common `Δb = +0.0001832`.

No new conflicts. `CONSISTENT_HIT` train-`k` vs valid-`k` match rate = 1.0.

## C. The true budget and cancellation

overall valid: B `y_raw` 0.116616 / `y_cal` 0.117406; H `y_raw` 0.095885 / `y_cal` 0.096694;
`Δb = +0.0001832`. The shared COMP `g_cal` MAE is unchanged at 0.089300 (H changes only the
cycle route/bias).

* `e_g = h − g` MAE **0.087004** (bulk covered rows); `e_c_B = q_B − c` MAE **0.001139**;
  `e_c_H = q_H − c` MAE **8.6e−10**.
* opposite-sign rate (B) **0.4792**, (H) **0.4985**; `triangle_gap` (B) **−0.001149**,
  (H) **−6.9e−10**.
* `Σ(|e_g| + |e_c|)` (B) **0.088143** vs true `|e_g + e_c|` **0.086994**; component MAEs are
  never added as a budget.
* fixed oracle (`h + c + b_B`) MAE **0.089325**; it is a label-reading diagnostic only, not
  reachable deployable performance and not part of the gate.

## D. Tail rows and single-point sensitivity

* `valid:0172` detail (`valid_0172_detail.json`): k=−6, c=−20.8010, y=−20.3405, g=0.4605,
  h=0.5460, q_B=−0.0602, q_H=−20.8010; B `y_cal` err −20.8146, H `y_cal` err −0.0740. It is a
  `CONSISTENT_HIT` on train singleton class 265 (train:3776, same T25, k=−6).
* B's fixed top-10 valid cal errors: 6 are `CONSISTENT_HIT` (0172 k=−6, 0423/0189/0615/0221/0016 k=0)
  and 4 are `UNSEEN_FALLBACK` (0935/0214/0249/0917). The top-10 contributes 1.001× the total
  gain — i.e. the gain is essentially the top row.
* sensitivity: excluding `valid:0172` → gain **−0.0000288** (cal) / **−0.0000095** (raw).
  Excluding B's worst row is the same row. The main gate always uses all 1000 rows.
* positive rows 512, negative rows 488; positive sum +21.169, negative sum −0.457; max positive
  row share of positive sum 0.980.

## Engineering / deployable evidence

* Single-file reloadable wrapper (`deploy_wrapper.py` / `H_model_package.pt` +
  `prototype_table.npz`): standard forward returns `(n,) y`; diagnostic returns `q` and route.
  Unknown keys, non-finite keys and conflict-class keys route to the frozen `Q`.
* Replay: wrapper vs cached H valid `y_cal` max abs **0.0**; train wrapper `y_cal` MAE
  **0.03325384** matching the frozen train table. Label-permutation leaves `q`, route and `y`
  unchanged. Key serialization deterministic and reloadable; order restoration exact; mismatched
  lengths raise. All identity checks pass (`replay_checks.json`, `checks.json`).
* No optimizer/backward, no GPU, no remote job, no official-test read.

## Boundary disclosure

Single seed; reused exposed official-valid; extra train-side supervision label `c` (as in source
`Q`); no dictionary claim; one rule with one bias; not `luyin19` fulfillment; no test or new
training opened.

## Deliverables in this directory

`PROTOCOL.md`, `METHOD_CONTRACT.md`, `REPORT.md`, `DECISION.md`, `EXECUTION.md`,
`EVIDENCE_SCOPE.md`; `source_identity.json`; `train_table.{csv,json}`,
`train_conflict_classes.csv`, `per_row_train.csv`, `train_bias.json`, `train_metrics.json`,
`train_routing.json`, `loo_coverage.json`; `frozen_eval_manifest.json`, `H_model_package.pt`,
`prototype_table.npz`, `T25_cache.npz`; `main_table.json`, `group_table.csv`,
`routing_table.csv`, `coverage.json`, `per_row_valid.csv`, `tail_top10.csv`, `tail_summary.json`,
`sensitivity.json`, `cancellation.json`, `valid_0172_detail.json`, `addback.json`; `gate.json`,
`bootstrap.json`, `benchmark_marker.json`, `checks.json`, `replay_checks.json`, `budget.json`,
`heldout_access.json`, `analysis.json`, `manifest.json`; scripts `source_identity.py`,
`freeze_prototype.py`, `evaluate_valid.py`, `deploy_wrapper.py`, `replay_checks.py`,
`analyze.py`, `make_manifest.py`.
