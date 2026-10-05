# REPORT — `zinc_component_supervision_fulltrain_confirmation_seed0_v1`

## Direct answers

### 1. Is COMP's `g` gain over SUM confirmed on the full-train valid?

Yes. COMP's calibrated `g`-MAE gain over SUM is **+0.005354** (G0) and **+0.004563** (overall),
with 95% paired CIs `[+0.000765, +0.010104]` (G0) and `[-0.000583, +0.009310]` (overall). Raw
gains are also positive (G0 +0.006388, overall +0.005635). All four CHEM_CONFIRMED conditions
hold.

The internal-dev gain of `+0.00987` (overall calibrated `g`) is **degraded roughly by half**
under full-train confirmation. This is expected: the old internal dev2000 fold is now part of
the training set, and the valid set is a separate (harder) evaluation. The signal survives but
is materially smaller.

### 2. What are the deployable `y` errors, and is `y < 0.09`?

SUM: y_cal = **0.1169**, COMP: y_cal = **0.1122**. **Neither is below 0.09.** The benchmark marker
is not met.

The `y`-MAE gain (overall calibrated) for COMP over SUM is **+0.005247**, CI `[+0.000499,
+0.010275]` — positive and significant (CI excludes 0). All four DEPLOY_CONFIRMED conditions hold.
The absolute `y` bar of 0.09 is **below** both arms and is a separate marker, not a purchase
for this round; the incremental gate passes independently.

### 3. Sources of the `g`-vs-`y` difference

The raw `g`-gain and the raw `y`-gain are nearly equal in magnitude because the shared cycle head
Q has low train/valid error (Q soup train L1 = 0.011506). The `b_y` calibration bias adjusts
for Q's train/valid distribution shift. The deployable `y` MAE is higher than `g` MAE overall
because `y = g + c`, and `c` contributes its own irreducible error.

G0 (k=0, 965/1000 valid rows) captures the bulk of the gain: both `g` and `y` G0 cal gains are
≈ +0.0053. The five non-G0 rows (k=-1:30, k=-2:4, k<=-3:1) are small in number but the cycle
head is weakest there (Q's per-k error is larger for extreme k), so the y-g difference is
largest in those rows.

The row-wise identity `y_cal - y = (hat_ell - ell) + (hat_s - s) + (q - c) + b_y` checks out to
3.8e-7 (SUM) and 4.1e-7 (COMP) — float64, confirming the wrapper composition is correct.

### 4. Is `s` still the main gap inside COMP?

COMP's component MAEs on valid: L_ell = 0.079, L_s = 0.086 (soup). On train: L_ell = 0.052,
L_s = 0.057. The `s` component error is consistently larger than `ell`, and `s` is the residual
after subtracting `ell` from `g` — no separate `s` head was trained. The `s_SA` decomposition
shows `epsilon = s - s_SA` is non-zero (mean signed error 0.01–0.02), indicating that `s`
contains structure beyond pure SA-normalised residual. The `s` residual remains the main
modeling gap inside COMP.

Cancellation: the COMP `e_ell`/`e_s` opposite-sign fraction is 0.14, and the triangle gap
(`|e_ell| + |e_s| - |e_g|`) is 0.015 — meaning the component errors partially cancel in `g`, which
is why COMP's `g` MAE (0.0374) is less than the sum of its component MAEs would suggest.

### 5. Conclusion and next responsibility

**CHEM_PASS_DEPLOY_PASS.** Both gates are confirmed on the full-train valid. The chemistry
signal is real and survives, the deployable `y` gain is real and significant, and the deploy
wrapper identity is verified.

**However**, the `y`-MAE benchmark (0.09) is not met. This is a marker, not a gate, and the
round's decision is based on the incremental gates. The next research responsibility is to
**close the gap to `y < 0.09`** — this round confirms that adding component supervision helps
but does not close the overall error bar. The `s` residual is the largest remaining component
gap and is the natural next target.

## Boundaries

* Single-seed (seed 0), single full-train range, single valid confirmation. No seed replication.
* official-valid is reused (used in prior research rounds); disclosed. official-test never loaded.
* Full-train standardizers and kappa were refit on 10,000 rows (not the old 8,000-fold artifacts).
* Q was retrained (not reused from any prior round) using the frozen 10k prep/constants.
* The `0.5` component-loss coefficient is fixed; no hyperparameter scan or model selection.
