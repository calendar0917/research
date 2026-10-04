# REPORT — `zinc_pooling_scale_count_seed0_v1`

**Question.** With the same information, same parameter count and same training recipe, does
replacing the unary/pair sum moments by count-normalised mean moments improve the internal
chemical component `g = y − c`? Secondary: does merely exposing the six existing count
coordinates already help? Exactly two seed-0 fresh trajectories (`C` sum/count, `N` mean/count);
the fresh-fold `M` soup is a read-only reference. Official-valid/test never loaded.

## Opening table — the five closing questions

| # | question | answer |
|---|---|---|
| 1 | Does `N` beat `C`, and does it truly exceed the source `M`? raw/cal consistent? | **No on both, and worse rather than merely flat.** `N` vs `C`: G0 cal `−0.011631` CI `[−0.015971,−0.007671]`, overall cal `−0.011797` CI `[−0.016011,−0.007776]`, G0 raw `−0.007861`, overall raw `−0.007643` — all four CI-separated negative. `N` vs `M`: all four endpoints negative (`−0.0079…−0.0092`), CIs all below zero. raw and cal agree in sign for `N`: the mean/count parametrisation is clearly worse in this skeleton/recipe. |
| 2 | Does count-channel visibility alone explain a gain? | **Not confirmed.** `C` vs `M` (count coordinates restored on top of the source skeleton) is better on the calibrated endpoints: G0 cal `+0.003722` CI `[+0.000460,+0.007317]`, overall cal `+0.003560` CI `[+0.000292,+0.006764]`. But both raw endpoints are slightly **negative** (G0 raw `−0.001053` CI `[−0.004362,+0.002621]`; overall raw `−0.001525` CI `[−0.005079,+0.002023]`). Frozen gate 3/5. The cal gain is created by the fit-median bias (C bias `−0.031017` vs M `+0.013944`), not by better raw predictions; under the frozen rule this is not a buy. Additionally, C was trained on local CPU while M was trained on A100 (res-2 unavailable), so this secondary comparison is regime-confounded. |
| 3 | Did the intervention really change block scale; do scale-bin errors co-move; which competing explanations remain? | **Yes, strongly.** Init pre-reader unary RMS: `C 1.5424` vs `N 0.0685` (≈22× smaller); epoch-240 4.514 vs 0.305. `N` is worse in **every** fit-quartile dev bin (cal gains −0.0029…−0.0173) with no size trend; `C` vs `M` cal gains are positive in 3/4 bins (+0.0079, −0.0015, +0.0027, +0.0052). `N` fits better (fit cal 0.025735 vs C 0.028201) but generalizes worse (dev cal 0.107541 vs 0.095743) — a fit/dev divergence/optimisation-path explanation is not separated from a "mean pooling is a worse inductive bias" explanation. Since C and N moments are recoverable from each other, this is a scale/function-accessibility result, not an information result. |
| 4 | Close to 0.09? | `C` reaches the best calibrated numbers seen in this track (`G0 cal 0.093661`, overall cal `0.095743`, gaps `+0.003661` / `+0.005743` to 0.09) but **raw is `0.099319` / `0.101624`**, and the calibrated improvement comes from the bias shift, not raw accuracy. This is the internal `g = y − c` dev diagnostic, not an official `y` or official-valid score; no 0.09 crossing is claimed. |
| 5 | What closes this round; what would buy the next? | Frozen rule 5 fires: **`NO_CANDIDATE`**. The fixed aggregation-scale/count variant is closed. `N` (mean/count) is rejected with a CI-separated degradation (`N`-vs-`C` G0 cal interval entirely below −0.0077); `C` (count visibility) fails the raw conditions and is regime-confounded, so no confirmation round is bought. Next investment would need a regime-matched (same host/device) run showing `C` raw > `M` raw, or a new discriminative hypothesis with its own pre-registered gate. |

## 1. Frozen setup and identity witnesses

Protocol/contract/evidence-scope frozen in commit `ddd5bd394fe2` before any training; the two
formal trajectories also executed at that commit. `pre_checks.json` all_ok=true:

* Frozen inputs copied byte-identically from the source round and re-hashed
  (`frozen_input_copy.json`): fit `2a21cb8771f6…`, dev `270ab4126b0f…`.
* Init identity: C and N state dicts byte-identical to source `M_init_state.pt` (all keys/shapes,
  max abs 0.0), 297,499 parameters each.
* Schedule `7b11a529…`; build RNG `a2e8a8ab…`; train RNG `1ccf1725…`; both arms' actual global
  graph-ID stream `69187f13…` (identical to the source round and to each other).
* C's forward under the round mask is bit-identical to the inherited audit forward
  (`forward_equivalence: 0.0`), so only the pooling routing changed.
* Pooling checks: 24 independent float64 reference comparisons (two graph sizes, empty pair
  buckets 1/2, per-graph vs concatenated batch, shuffle/restore, `sum = mean·n` recovery,
  count `= log1p(n)`, sum-mode bit-identity with existing masked pooling) all pass.
* Count witness (runtime, not flags): six count coordinates match independently recomputed
  per-graph counts exactly, are identical between C and N, and are non-constant where `n>0`;
  N's two moments equal the per-graph/per-bucket sums divided by the actual `n` (rtol 1e-4);
  snapshot saved (`count_snapshot.npz`, `count_witness.json`).
* Smoke (local CPU, 3 steps each, states discarded): finite losses/grads, forward finite,
  save/reload exact.

## 2. Performance

Dev calibrated `g`-MAE (one fit-median bias per arm) and raw:

| arm | fit overall cal | dev overall raw | dev overall cal | dev G0 raw | dev G0 cal | bias |
|---|---:|---:|---:|---:|---:|---:|
| C (sum/count) | 0.028201 | 0.101624 | **0.095743** | 0.099319 | **0.093661** | −0.031017 |
| N (mean/count) | **0.025735** | 0.109267 | 0.107541 | 0.107180 | 0.105292 | +0.022794 |
| M (source, read-only) | 0.030535 | **0.100098** | 0.099304 | **0.098265** | 0.097383 | +0.013944 |
| B (source, context) | 0.031167 | 0.103191 | 0.100192 | 0.100786 | 0.097932 | −0.022919 |

Paired bootstrap (1000 draws, seed `20261006`, shared dev-row indices; positive = candidate
better):

| gain | point | 95% CI |
|---|---:|---|
| N vs C, G0 cal | −0.011631 | [−0.015971, −0.007671] |
| N vs C, overall cal | −0.011797 | [−0.016011, −0.007776] |
| N vs C, G0 raw | −0.007861 | [−0.012480, −0.003576] |
| N vs C, overall raw | −0.007643 | [−0.012502, −0.003494] |
| N vs M, G0 cal | −0.007910 | [−0.012171, −0.003646] |
| N vs M, overall cal | −0.008237 | [−0.012426, −0.004034] |
| C vs M, G0 cal | +0.003722 | [+0.000460, +0.007317] |
| C vs M, overall cal | +0.003560 | [+0.000292, +0.006764] |
| C vs M, G0 raw | −0.001053 | [−0.004362, +0.002621] |
| C vs M, overall raw | −0.001525 | [−0.005079, +0.002023] |

Frozen gate (`pass_count`): N vs C **0/5**, N vs M **0/5**, C vs M **3/5** (the two cal gain
conditions and the cal CI condition pass; both raw conditions fail). Classification by the
frozen order: rule 1 no, rule 2 no (`N` vs M fails), rule 3 no (`C` vs M fails on raw), rule 4
no (`N` vs C fails) → **`NO_CANDIDATE`**. Witnesses: identical predictions give gain 0 with CI
[0,0]; swapping arms mirrors point and CI exactly. Drop-worst-row sensitivity (dev pos 1403,
gid 6839, k=0): N vs C `−0.011295`, N vs M `−0.007968`, C vs M `+0.003327` (signs unchanged).

Per-group calibrated MAE and contribution `Σ|err|/N_dev` (n=2000; identities hold to machine
precision):

| comparison | k=0 (1935) | k=−1 (56) | k=−2 (7) | k≤−3 (2) |
|---|---:|---:|---:|---:|
| C MAE (contribution) | 0.093661 (.090617) | 0.161226 (.004514) | 0.100853 (.000353) | 0.259057 (.000259) |
| N MAE (contribution) | 0.105292 (.101870) | 0.179937 (.005038) | 0.123014 (.000431) | 0.201571 (.000202) |
| M MAE (contribution) | 0.097383 (.094218) | 0.155907 (.004365) | 0.135217 (.000473) | 0.247386 (.000247) |
| N−C cal gain | −0.011631 | −0.018711 | −0.022160 | +0.057486 |
| C−M cal gain | +0.003722 | −0.005319 | +0.034364 | −0.011671 |

`N`'s loss is concentrated in the k=0 bulk and k=−1; `C`'s calibrated edge over `M` is almost
entirely k=0 (+0.003601 contribution) with a small k=−1 loss.

## 3. Scale / mechanism evidence (descriptive, not a gate)

Fixed fit node-count quartiles (edges `9–20 / 20–23 / 23–26 / 26–37`; n_dev
`427/460/545/568`): `N` is worse than `C` in every bin (cal gains −0.0173, −0.0153, −0.0029,
−0.0134), so this is not a graph-size effect; `C` vs `M` cal gains are `+0.0079, −0.0015,
+0.0027, +0.0052` with no monotone size trend. Pre-reader block RMS at the fixed first-128 fit
batch: unary first block `C 1.5424 → 3.7704 → 4.5144` (epoch0/40/240), `N 0.0685 → 0.2523 →
0.3048`; the intervention is real and large. Reader-first-layer gradient norms from normal
training: `C` unary/relation `0.18/3.66` at epoch 1 rising to `1.63/0.56` at epoch 240; `N`
`0.11/0.33` to `1.09/0.45`; N's global grad norms are much smaller (2.3–3.2 vs 5.8–9.4) and
clip fraction near zero. Pair-count/moment scale at init: mean pairs per graph per bucket
`24.9/34.3/34.0/30.8/141.1`; `rms(mean)/rms(sum)` `0.039/0.028/0.028/0.031/0.006`; unary mean
environment rows per graph `23.1`. The five buckets are not equally sized, and dividing by `n`
rescales them by very different factors. `N` fits the fit set better but dev worse — an
optimisation/scale-path or inductive-bias explanation is not separated from a genuine
"sum parameters are better here" effect; neither is claimed as mechanism.

## 4. Replay / identity

* Source `M` soup replay with the original C6 path: max abs fit/dev (128 rows each) `9.5e-7`;
  source `M` main metrics reproduced exactly (diff 0.0 on all four endpoints).
* C and N soup states reload from their single files with max abs `0.0` on 128 fit + 128 dev
  rows (CPU replay).
* Both arms: 15,120/15,120 steps, init identity max abs 0.0 vs source M, RNG streams frozen,
  global gid stream `69187f13…` equal to the source round.

## 5. Boundaries

One seed, one soup per arm, no warm start, no search. The **primary N-vs-C comparison is
same-regime** (both local CPU, identical init/schedule/RNG); the **C-vs-M comparison is
cross-regime** (local CPU vs source A100 on res-2) because no GPU was available: res-2 jobs
55976/55977 stayed pending on `(Resources)` and were cancelled, and `res` GPUs are unavailable
(`torch.cuda.is_available()` false, NVML device-handle error). CPU-only means the required
CPU/GPU replay equality check was not performable; CPU self-replay is exact. Therefore even a
5/5 C-vs-M result would not have been a clean count-accessibility confirmation; as run it is
3/5 with negative raw, and it is not purchased. `NO_CANDIDATE` closes this fixed
aggregation variant; it does not show that all pooling forms are useless, that the model lacks
input information, or where the residual bottleneck is. No official-valid/test read; no 0.09
crossing claim.

## 6. Direct answers

1. `N` loses to `C` on all four endpoints with CI-separated negative point estimates
   (`−0.0116/−0.0118` cal), and loses to `M` likewise. raw and cal agree in sign for `N`.
2. Count visibility alone (`C` vs `M`) produces a calibrated-only gain (`+0.00372/+0.00356`,
   CIs just above 0) but negative raw; gate 3/5 → no buy, and the comparison is
   regime-confounded.
3. The intervention changed block scales by ~22× at init; errors do not co-move with graph-size
   bins for `N` (worse everywhere); C-vs-M cal gains are mixed across bins. Unseparated:
   scale/optimisation path vs inductive bias for `N`; platform confound for C-vs-M.
4. `C` reaches `0.093661` (G0 cal) / `0.095743` (overall cal), still above 0.09; internal
   `g`-MAE only; raw is worse than `M`.
5. Round closes with `NO_CANDIDATE` (frozen rule 5). Nothing is bought; a future buy would need
   a same-regime raw improvement over the working reference under a new pre-registered gate.
