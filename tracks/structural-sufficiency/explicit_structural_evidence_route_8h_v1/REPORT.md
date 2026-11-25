# REPORT — explicit structural evidence route (8 h round)

Round `explicit-structural-evidence-route-8h-v1` · T0 `2026-11-25T16:04:18Z`
Design frozen in [`PROTOCOL.md`](PROTOCOL.md) + `analysis/protocol_freeze.json`
(SHA-256 of configs, schedule and every code module).

Every number below is Validation-only unless it explicitly carries the
`EXPOSED_TEST_ENGINEERING_COMPARISON_SET` label. Test was read once, after the
freeze, by `code/esr_score_test.py`; PCBA Test is not scoreable at all (the
frozen label artifact is NaN on every Test row).

---

## First page: ten questions

**1. What was actually run?**
<!--RESULTS_SUMMARY-->

**2. Is the explicit route competitive on PCBA?**
<!--Q2-->

**3. Which aggregation wins?**
<!--Q3-->

**4. Does the task projection position matter?**
<!--Q4-->

**5. Does the relation channel earn its place?**
<!--Q5-->

**6. Small datasets.**
<!--Q6-->

**7. Capacity control.**
<!--Q7-->

**8. Is the accounting exact?**
<!--Q8-->

**9. What do the structure rules look like?**
<!--Q9-->

**10. What is *not* supported?**
<!--Q10-->

---

## 1. Question and design

Can a **non-recursive** prediction function that is computed directly on explicit
structural objects — rooted local structures plus their shortest-path relations —
be predictive on par with the historical LCRC backbones, while keeping a
prediction account that can be checked to the last decimal?

The historical reference points are the frozen PCBA numbers of the previous
rounds (official scaffold split, macro AP): W2 Validation `0.265273`
(Test `0.253616`), FIXED1 `0.275846` (`0.257002`), DYNAMIC `0.275067`
(`0.262226`), plus the 25-epoch prefix finals W2 `0.2630573` and
FIXED1 `0.2758455`. Historical checkpoints were never re-trained, never fine-tuned
and never used as a teacher.

Frozen objects of this round:

* relation instances = **all** unordered atom pairs with shortest-path distance
  1–2 (no sampling, no truncation, no atom-id selection), 19-d relation vector
  (5 distance one-hots + `log1p` common neighbours + the cached raw 13-d bond
  features of the direct bond);
* tokens `u_i`, `v_ij` from the *frozen* `h0` of the inherited trunk (one call
  per forward, `h0` never updated);
* evidence aggregation `A(s) = sqrt(m) · [π_mean·Mean + π_pos·LME − π_neg·LME(−s)]`,
  `τ = 1`, `π` initialised `[0.5, 0.25, 0.25]`, learned per task (EVID) or per
  channel (BANK);
* arms `SREN-ADD`, `SREN-EVID`, `SREN-BANK`, `SREN-COMPOUND`, `HIDDEN`
  (capacity control), `UNARY` (content-free SELF tokens), and the conditional
  `SREN-BANK-FLEX`, `REL-BLIND`, `BEST_EXPLICIT_RANGE4`;
* `BASE` = the unmodified historical `BSModel` trained with the same recipe on
  the same data, the only arm that keeps `f`/`ρ`/attention readout.

## 2. Data and isolation

| dataset | split | Train / Valid / Test | metric |
|---|---|---|---|
| Lipo | official OGB scaffold | 3360 / 420 / 420 | raw RMSE |
| ESOL | official OGB scaffold | 902 / 113 / 113 | raw RMSE |
| Tox21 | frozen ChemBench split | 6264 / 783 / frozen | observed-label macro ROC-AUC |
| PCBA | official OGB scaffold | 350343 / 43793 / 43793 | official macro AP |

All caches and labels are inherited read-only. Structural normalisation, the
target scaler and the two auxiliary global statistics (`log1p n_atoms`,
`log1p n_relation_pairs`) are fitted on **Train rows only**; the training views
expose `train`/`valid` and a static scan verifies that no training code path can
index `test` (`analysis/checks.json`, check 18).

## 3. Verification before any formal GPU step (Phase A)

`code/esr_checks.py` ran 18 checks on all four datasets (`analysis/checks.json`);
all pass. Highlights and the independent witnesses they use:

| check | witness |
|---|---|
| data identity / split disjointness | row SHA-256 per split, official sizes, no overlap |
| Train-only normalisation | refit from the Train cache reproduces the recorded normaliser (PCBA: inherited subsample fit, Train refit closer to the stored values than a Valid refit) |
| pair index and uniqueness | shortest-path distances re-derived with an independent `scipy` BFS on 64 molecules per dataset (48 for PCBA), exact set equality |
| degenerate graphs | single atom, no-bond molecule, isolated triangle, two-component molecule: empty set → 0, single token → identity |
| aggregation vs reference | float64 reference implementation of `Mean`, `LME`, `LME(−s)`, `A(s)`; max error `~1e-7`, gap non-negativity `D_pos, D_neg ≥ 0` verified |
| shared initialisation | `U`, `V`, global-linear tensors bitwise identical across arms (private generator, fixed module order) |
| `h0` once, heads deleted | forward hooks count exactly one trunk call; `f`/`ρ`/`readout` absent outside `BASE` |
| ADd commutes / BANK does not | with a task-linear head, `ADD` commutes with the projection; a heavily asymmetric `BANK` token projection is witnessed to not commute, and a pure-mean aggregation is shown to commute (so the witness is real) |
| renumbering / edge order | permuting atom ids and edge order leaves logits unchanged (`≤1e-6`) |
| gradients | every trainable new tensor receives a non-zero gradient |
| accounting | predicted logits reconstructed from the decomposition for `ADD`, `BANK`, `COMPOUND` and the EVID entropy decomposition (`max |Δ| = 0.0`) |
| checkpoint replay | 2 epochs from scratch vs 1 epoch + resume: parameter max |Δ| `1.7e-6`, metric |Δ| `2.5e-6` (pre-registered tolerances 1e-5 / 1e-5 relative, set after the first observation of remote FP32 reduction order noise) |
| Test isolation | static scan of every training module for `test` indexing |

Throughput preflight (`analysis/preflight_timing_*.json`) measured per-arm step
times on real batches on both A100s, the peak allocated memory (0.13–0.23 GiB),
the pair distribution (Lipo mean 70.4 pairs/molecule, PCBA mean 67.3, max 791)
and the projected 25/40/360-epoch cost used for scheduling.

`analysis/protocol_freeze.json` records the SHA-256 of `PROTOCOL.md`, both config
files, the schedule and every code module at freeze time.

## 4. What was trained

<!--TRAINING_TABLE-->

## 5. PCBA results

<!--PCBA_RESULTS-->

## 6. Small-dataset results

<!--SMALL_RESULTS-->

## 7. Exact prediction accounting

<!--ACCOUNTING-->

## 8. Mechanism diagnostics

<!--MECHANISM-->

## 9. Structure rules and case studies

<!--RULES-->

## 10. Efficiency

<!--EFFICIENCY-->

## 11. Conditional branches and decisions

<!--DECISIONS-->

## 12. Deviations, errata and limitations

<!--LIMITATIONS-->
