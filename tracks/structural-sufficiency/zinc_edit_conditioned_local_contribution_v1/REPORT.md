# REPORT — ECLC-v1

**Decision:** `STOP_NO_COMPOSITION_GAIN`

**Execution commit:** `2b0208c088d097726e184ffea21051bb0c29cde2`
**Formal trajectories:** 2 (seed 0, REAL and NULL). Seed 1 was not launched because the frozen seed-0 promotion rule failed.
**Official test:** never constructed, loaded, predicted or scored. All numbers below are official-valid, `VALID_AS_DEV_DESCRIPTIVE`.

The five questions are answered in order. “REAL − NULL” is the paired difference in MAE; negative favours REAL. Intervals are paired bootstrap 95% CIs over per-molecule absolute errors (10,000 resamples).

---

## Q1. Was the edit-pair support sufficient?

**Yes, on the pre-registered support thresholds.**

| support item | frozen threshold | observed |
|---|---:|---:|
| strict matched edit pairs | ≥300 | **350** |
| distinct bodies | ≥100 | **317** |
| edit classes with ≥3 bodies | ≥15 | **29** |
| distinct molecules covered | — | 621 / 10,000 (6.21%) |
| total edit classes | — | 190 |
| largest class | — | 24 bodies |
| ring / acyclic pairs | — | 313 / 37 |

Pairs were selected from official train-10k structures only, without labels, using bridge cuts, a larger body, a smaller pendant edit, strict locality `|sub_i| + |sub_j| < min(|body_i|, |body_j|)`, labelled-body isomorphism verification, and RDKit reconstruction checks. The pair graph has 285 components, largest 7, mean degree 1.127, max degree 5; the pair loss covers 6.2% of train molecules. Median edit region is 13 atoms and median body 16 atoms, so these are not tiny functional-group swaps. The support gate therefore passed without lowering thresholds.

---

## Q2. Was there a transferable differential signal?

**Yes for the low-order composition signal; no for edit-class identity.** The frozen verdict is `PASS_COMPOSITION_SIGNAL_ONLY`.

Held-out Δy prediction on the official train structures:

| model | MAE | relative improvement vs predict-zero |
|---|---:|---:|
| predict zero | 0.885882 | — |
| low-order structural difference (primary) | **0.346950** | **+60.8%** |
| edit-class one-hot only | 0.966043 | −9.1% |
| low-order + edit-class | 0.384284 | +56.6% |

The combined model was worse than 81 of 100 edit-class permutation controls; the permutation mean was 0.376430, the one-sided empirical p was 0.8119, and `class_identity_supported = false`. The pilot was therefore licensed by the **composition-additive low-order signal**, not by a categorical edit-class effect. This report does not call the result edit-class identity transfer.

---

## Q3. Did the CPU correctness gates pass?

**Yes. All 12 automated Phase 2 checks and all 10 fast unit tests passed** (`analysis/cpu_checks.json`, SHA-256 `34492c1a956f372240218f2ea4d3e3202b73e9330a392e97702d1e62facf6c06`). No GPU was used for Phase 0 or Phase 2.

Key verified facts:

- REAL and NULL each have **248,368** parameters (LCRC-A 237,519 + `f_base` 10,849), identical per-module breakdown, identical full initial state hash, and ≤500k deployment parameters.
- At initialisation, `f_base ≡ 0`, and both arms reproduce the frozen LCRC-A prediction with **max abs diff 0.0**. This is intentional; the trained model is not forward-equivalent to LCRC-A.
- REAL’s 253 radius-2 base-input columns are identically zero. After the zero-initialised last layer becomes nonzero, REAL’s radius-2 first-layer gradient is exactly **0.0**, while NULL’s is **1.15e-2** (max abs). The pair loss alone contributes nonzero gradient to `f_base`, `f_context` and `rho`.
- The independent naive-loop local-statistics reference agrees exactly (`max abs diff 0.0`, 332 atoms). Invariance maxima are `4.77e-7` (atom renumbering), `2.38e-7` (bond reversal), `5.96e-8` (batch split), `0.0` (molecule order).
- Pair-loss algebra: the additive identity holds to `1.19e-7`; a common prediction offset changes the pair loss by **0.0** but changes molecule L1 by `3.70`; the pair-vs-molecule gradient cosine is `−0.216`; the squared-pair quadratic identity is exact. The pair loss uses only `y_i, y_j` and is not new supervision.
- The pair list contains only rows `0..9999`; no valid/test row. The historical post-hoc `lc_test.pt` exists in the frozen LCRC cache directory, but no ECLC production module references or opens it.

---

## Q4. Did the restricted REAL base beat the parameter-identical NULL?

**No at the pre-registered effect size.** REAL was directionally better on full valid and on unseen radius-2 ordinary molecules, but the effects were smaller than required, the CIs include zero, and the per-molecule win rate was below 0.5.

Seed 0, best-valid checkpoints:

| slice | n | REAL | NULL | REAL − NULL | 95% CI | REAL win rate |
|---|---:|---:|---:|---:|---|---:|
| full | 1000 | **0.077558** | 0.078881 | **−0.001324** | [−0.005707, +0.002686] | 0.491 |
| ordinary (`y > -3`) | 935 | 0.062548 | 0.062458 | +0.000090 | [−0.004119, +0.003899] | 0.489 |
| `r2_unseen_ordinary` | 225 | **0.131118** | 0.133636 | **−0.002518** | [−0.018558, +0.011045] | 0.467 |
| `seen_ordinary` | 710 | 0.040818 | 0.039901 | +0.000917 | [−0.001663, +0.003471] | 0.496 |
| `minus_row172` | 999 | 0.069343 | 0.070904 | −0.001561 | [−0.005872, +0.002389] | 0.491 |
| `minus_fixed_worst10` | 990 | 0.062254 | 0.061655 | +0.000598 | [−0.002731, +0.003830] | 0.488 |
| tail (`y ≤ -3`) | 65 | 0.293469 | 0.315130 | −0.021661 | [−0.049088, +0.003138] | 0.523 |

Best epochs and windows:

| run | best valid | best epoch | final | W20 | T100 |
|---|---:|---:|---:|---:|---:|
| REAL | 0.077558 | 514 | 0.078682 | 0.078498 | 0.078688 |
| NULL | 0.078881 | 520 | 0.079233 | 0.079184 | 0.079226 |

The base pathway was active: zeroing `f_base` at the best REAL checkpoint changed valid MAE from `0.077558` to `0.476070` (|Δ| = **0.3985**), and the per-molecule base contribution had std **0.5698**. REAL also had lower W20 and T100 than NULL. Those checks confirm activity and window direction, not transfer: the REAL−NULL full effect is only `−0.00132`, below the frozen `−0.003` requirement; the unseen-radius-2 effect is `−0.00252`, below the frozen `−0.005` requirement. Ordinary and seen molecules were slightly worse for REAL (both within the 1e-3 “not worse” slack on ordinary), and the tail drives much of the full-valid mean. The full-valid absolute level (`0.07756`) is also above the frozen `0.075` gate; both arms are about `0.003` above the historical LCRC-A 4-seed valid `0.075990`, close to the known `~0.0026` execution-drift band.

---

## Q5. What is the decision, and what are the provenance and budget facts?

**Decision: `STOP_NO_COMPOSITION_GAIN`.** The frozen seed-0 promotion rule requires all five conditions:

| condition | result | observed vs threshold |
|---|---|---|
| 1. REAL full valid ≤ 0.075 | **FAIL** | 0.077558 > 0.075 |
| 2. REAL − NULL full ≤ −0.003 | **FAIL** | −0.001324 > −0.003 |
| 3. REAL − NULL `r2_unseen_ordinary` ≤ −0.005 | **FAIL** | −0.002518 > −0.005 |
| 4. not worse on ordinary and `minus_row172` (≤ +0.001) | PASS | +0.000090 and −0.001561 |
| 5. REAL W20/T100 lower and base path active | PASS | windows lower; ablation 0.3985; base std 0.5698 |

Because conditions 1–3 failed, seed 1 was **not** run, no architecture/width/pair/loss changes were made, and no replacement experiment was started. Under this frozen pilot, forcing the base contribution to atom type + rooted radius-1 typed adjacency did not produce a REAL-vs-NULL gain of the pre-registered size on unseen radius-2 environments. This is a negative pilot, not proof that radius-1 information is useless: the NULL was not worse, and REAL has 253 dead `f_base` input columns, which biases the comparison toward NULL.

### Provenance

| item | value |
|---|---|
| execution commit | `2b0208c088d097726e184ffea21051bb0c29cde2` |
| REAL run id | `eclc-real-s0-20261029-225105-822a37b8` |
| NULL run id | `eclc-null-s0-20261029-225028-3910e6d9` |
| REAL best checkpoint SHA-256 | `84ac7bec6547cd0a8830239cd8ec5f226f22465da001378573debe40aebddcb1` |
| REAL final checkpoint SHA-256 | `ced30e9732e6785cb2d5e6b928c84ba9329e6f03ceab3671dcf482a2332aff6c` |
| NULL best checkpoint SHA-256 | `b37042f872c2077f5a02ad0fad4758a5177843a6409da28bcdbc7de055980756` |
| NULL final checkpoint SHA-256 | `274ae25851b7213b3a63e8d3c41c2853dad73a7013aa1346d479ec2e1f109983` |
| pair list SHA-256 | `d9ddf89cb1edf577e56ec15c82f333d4d44baa0ace1a73e0fac6b824d870a476` |
| train/valid content fingerprints | `ded004e8a0c03af31413b0f8d0e8223d80c6ba424a2cbd34db42e16c9b85459b` / `f28a0ceed34444c39523e1d301a02350a67087c15b3de725850e45976212aef9` |
| parameters / updates per arm | 248,368 / 47,400 (600 epochs, 79 steps/epoch) |
| GPU·h seed 0 | REAL 0.775 + NULL 0.683 + smoke 0.002 = **1.459** |
| seed-0 cap / cumulative cap | 2.0 / 4.0 A100·h; respected |
| seed 1 | **not run** |
| official test loaded | **false** |

The remote cache files are byte-reserialized copies (`lc_train.pt` file SHA `1aef7f21…`, `lc_valid.pt` `514e8c72…`) whose content fingerprints and label hashes are identical to the frozen local artifacts. The frozen config records the local file SHAs; the content, not the zip container bytes, is what the runs used.

**Bottom line:** Phase 0 and Phase 2 passed; the seed-0 pilot failed the pre-registered transfer-effect thresholds and stopped. No claim of composition transfer, edit-class identity, or a new relative-learning paradigm is made.
