# PROTOCOL — `zinc_local_tuple_dictionary_vs_mlp_seed0_v1`

Frozen **before** the single formal training run and before any new dev score. One question, one
new arm, one seed, one fold.

## 1. Question

Keeping the real local tuple structure `(phi_v, a_v, a, t)`, the real correspondence aggregation
`e(v) = sum_{t,a} w_J(v,t,a) f(x(v,a,t))`, the injection location and the original skeleton
unchanged: does the current local IHT dictionary `f_D` (tied 10-step, hard-top-8, 125→64) bring a
generalisation gain on the chemical target `g = y − c` relative to a **same-parameter-count local
nonlinear projection** `f_M` (row-normalised Linear 125→64 + SiLU), within this fixed interface?

Both endpoints are valid: a gate-passing performance candidate, or a clear statement of the
evidence and its boundary for this encoder pair. A positive result is not required.

## 2. Fixed inputs

* Fold fit 8000 / dev 2000, same fold as the previous round; fit_idx sha256 `7bf1cfb8…`,
  dev_idx sha256 `a61c8010…`.
* Target `g = y − c` from the frozen `fit_only_targets.npz` sha256 `e2adf5f2…`; fit-median
  calibration bias is recomputed per arm from its own raw fit predictions (`b = median(g_fit − p_fit_raw)`),
  once per arm, never on dev.
* Tuple index `local_tuple_index.npz` sha256 `4facb6ec…`, read-only: complete 125-D tuple features,
  fit-only `phi` scaler, `pair_ptr`/`pair_t`/`pair_a`/`pair_wJ` (real correspondence), `root_base`,
  `root_atom`, `kappa_sample`.
* Schedule/data stream sha256 `7b11a529…`; D_J raw soup state sha256 `95122a3f…`;
  D_J init state hash `185db5ed…`; D_J init `D_loc` sha256 `807d0247…`.
* Reference B = `zinc_chemistry_dictionary_vs_mlp_seed0_v1` M_g raw soup, read-only.
* Official valid/test are never loaded or scored.

## 3. Design

* The only change relative to D_J: the local encoder becomes `f_M(x) = SiLU(x @ A_bar.T)` with
  `A_raw[64,125]` (8,000 parameters), `A_bar = row_l2_normalise(A_raw)`, no bias/gain/norm/dropout/
  second layer. Per-tuple nonlinearity first, then real `pair_wJ` aggregation by `index_add`.
* `A_raw_init = D_loc_raw_init.T` element-wise; `W_loc` remains 342×64 and exactly zero; shared
  body/bridge/W_loc stay byte-identical to the previous round's init.
* One-shot label-free scale match: `kappa_M = RMS(kappa_D · e_D_init) / RMS(e_M_init)` over the
  frozen fit-only `kappa_sample` (≤8192 roots). No label, no dev, no per-dimension or dynamic
  normalisation, no multiplier search.
* Training recipe copied from the actual previous run: seed 0, Adam lr 1e-3, coupled WD 1e-5,
  global grad clip 5.0, batch 128, 240 epochs, 15,120 steps, L1 on `g`, fixed last-5-epoch
  (236–240) parameter soup, FP32, no AMP/DDP, C6 mask, body and static relations unchanged.
* The only new formal trajectory is `M_J`; D_J is reused read-only under the name D_J.

## 4. Classification (frozen)

`gain_MJ = MAE(D_J) − MAE(M_J)`; positive = M_J better. 1000 paired per-row bootstrap draws,
seed 20261004, one shared resample index set per draw per metric; G0 (k=0, n=1915) resampled inside
G0, overall (n=2000) over all dev rows.

| category | frozen condition |
|---|---|
| INVALID / INCOMPLETE | input/lineage/init/stream/replay fail, or the fixed formal trajectory did not complete |
| MLP_MECHANISM_FAILED | M local path to final state not executing, zero change or dead channels |
| MLP_LOCAL_SUPPORT | G0 cal gain ≥ +0.003, CI lower > 0, G0 raw gain > 0, overall cal gain ≥ −0.001 |
| DICT_LOCAL_SUPPORT | G0 cal gain ≤ −0.003, CI upper < 0, G0 raw gain < 0, overall cal gain ≤ +0.001 |
| LOCAL_EQUIVALENCE | G0 and overall cal CIs fully inside [−0.003, +0.003] and both raw point values inside |
| INCONCLUSIVE | otherwise |

Separate PERFORMANCE_SIGNAL vs B: G0 cal gain ≥ 0.003 **and** overall cal gain ≥ 0.003 **and**
G0 cal CI lower > 0 **and** G0 raw gain > 0 **and** overall raw gain > 0. Every condition is
reported item by item; the mechanism and performance classes do not replace each other.

## 5. Evaluation and mechanism

* Primary endpoint dev G0 (k=0) calibrated g-MAE; secondary overall calibrated g-MAE; fit/dev,
  raw/cal, per-k-group MAE and `contribution = sum|err|/N`, with group-contribution sum and
  group-gain sum identities checked.
* One pre-fixed sensitivity: drop the unique dev row with maximal mean calibrated absolute error
  across D_J and M_J; no other deletion rule is tried.
* Mechanism (fit-only bias, no retraining): one operator J→I switch for M_J compared with the
  already-stored D_J switch, and one fit-mean local-code replacement for D_J and M_J each
  (fit mean of `kappa·e(v)` over fit roots; roots without neighbours stay zero; no bias refit).
* The calibrated four-grid J/I decomposition from the previous round is re-read and checked
  (`G = O + W`), as a descriptive decomposition only.
* No new loss, head, ridge/tree, coverage match, kappa scan, post-hoc weighting or capacity probe.
* If dev g-MAE < 0.09, that is only a chemical-component diagnostic; it is not an official-valid
  y-MAE result, and no official-valid/test or 10k confirmation is bought.

## 6. Stop rule

After the single M_J trajectory, the frozen analysis and the small mechanism set above, stop
regardless of sign. No seed/fold/config/encoder-family search; no rescue run; on failure only an
exact restart of the same trajectory from saved state/RNG within budget, otherwise INCOMPLETE.
