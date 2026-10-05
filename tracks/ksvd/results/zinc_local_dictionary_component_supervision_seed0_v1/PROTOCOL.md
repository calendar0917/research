# PROTOCOL — zinc_local_dictionary_component_supervision_seed0_v1

**One question.** At the same local 125-D tuple input, the same real J incidence
aggregation, the same M_g skeleton and the same two-component (COMP)
supervision, does the shared sparse IHT local tuple dictionary (`D_COMP`) beat
the parameter-matched SiLU local MLP (`M_COMP`) with a discernible chemical
generalisation gain on the internal dev g-MAE?

**Two formal arms only** (`D_COMP`, `M_COMP`), seed 0, 240 epochs, 15,120
optimizer steps each, fresh init, identical recipe. No SUM arm, no marginal
control, no other encoders, no width/loss/sparsity/IHT-step/init/WD/optimizer/
fold/seed search. The result classifies the *local encoder under COMP
supervision in this fixed interface*; it cannot claim component supervision
revived dictionaries, a supervision×encoder interaction, or all-dictionary >
all-MLP (needs controls this round does not buy).

## Data / fold / targets

* 10,000 official-train rows only (dedicated train-only loader; official
  valid/test never loaded, deserialised, predicted or scored; no old test
  predictions or groupings used for any decision).
* New fold, fixed once: `perm = np.random.default_rng(20261006).permutation(10000)`;
  `fit_idx = np.sort(perm[:8000])` (sha `734d27f2…`), `dev_idx = np.sort(perm[8000:])`
  (sha `cb5f49dc…`). Not re-drawn for any reason. The dev labels are evaluated
  only after both arms complete and the eval roster is frozen; this fold is NOT
  claimed to be unseen or an independent research test set.
* All fitted statistics refit on the new 8000 fit rows only: body input
  standardizers (invert frozen all-train cache constants, refit per column),
  tuple phi scaler + `phi_scale`, `kappa_D`, `kappa_M`, target constants
  (`sigma_logP, sigma_SA, mu_SA, sigma_cycle, mu_cycle`; `MU_LOGP` fixed
  2.4570953396190123). Raw per-graph tuple incidence (structure only) reused
  from the verified production index.
* Targets: `c = (k − mu_cycle)/sigma_cycle`, `g = y − c`,
  `ell = (logP − MU_LOGP)/sigma_logP`, `s = g − ell`; float64 identities
  `y = g + c`, `g = ell + s` verified; `s` named "residual chemistry component".
* No warm start: no COMP/SUM full-train soup, body, Q, prototype or discrete
  cycle head enters as weights or outputs of this round.

## Arms

Shared: current M_g skeleton (Sem108+size2, C6 mask, sum/squared-sum, static
relations, topology25, posterior 144→288→144 **MLP bridge** in both arms),
ComponentReader 39→2 (half-split init; initial sum function equals the original
39→1 head, verified ≤1e-5), `W_loc[342,64]` zero init.

| | D_COMP | M_COMP |
|---|---|---|
| local tensor | `D_loc_raw[125,64]` (8,000) | `A_raw[64,125]` (8,000) |
| normalisation | columns L2 | rows L2 |
| per-tuple code | tied IHT, 10 steps, hard top-8/64, `eta = 1/(1.05·σ(Dbar)²+1e-12)` (detached, frozen power-iteration start) | `SiLU(x @ A_bar.T)` |
| init | canonical `init_d_loc` (frame seed 20261004) | `A_init = D_init.T` element-wise |
| injection | `fusion0(Sem110) + W_loc @ (kappa·e)` | same |
| parameters | 297,539 | 297,539 |

Both arms use the real J incidence weights (`pair_wJ`) end-to-end; root codes
aggregate per tuple then by incidence (never average-then-encode). One-shot
label-free scale match on ≤8192 fit roots (seed 20261004, same sample both
arms): `r_D = RMS(kappa_D·e_D_init)`, `r_M = RMS(e_M_init)`, `kappa_M = r_D/r_M`
(r_D not assumed 1). kappa_D per the historical J/I merged RMS rule
(`1/RMS(e_J,e_I)`), fit-only, not an I-model training.

## Supervision (identical)

```
[hat_ell, hat_s] = ComponentReader(body_features)   # (n,) each, one body pass
hat_g = hat_ell + hat_s
L = MAE(hat_g, g) + 0.5·(MAE(hat_ell, ell) + MAE(hat_s, s))
```

Shapes asserted `(n,)`; no broadcasting/flattening tricks; coefficient 0.5
fixed; no reconstruction term, rare-row weighting, gradient balancing or extra
auxiliary losses.

## Training recipe (identical, frozen before launch)

seed 0; Adam lr 1e-3, coupled weight_decay 1e-5, global clip 5.0; batch 128;
240 epochs = 15,120 steps; position schedule `build_schedule(8000, 240, 0+101)`
(hash `7b11a529…`, identical for both arms, actual global gid stream recorded
and required equal); soup epochs 236–240; checkpoints at 1/40/120/240.
Construction consumes the historical seed-0 stream (RNG-equal builds,
`fork_rng` for the reader split; spectral estimate and eval loader RNG-neutral,
verified). No epoch/last-vs-soup selection by dev.

## Pre-training checks (all pass, fit-only)

fold hashes; dev-label shuffle leaves the fit constants unchanged (exact 0);
shared initial tensors identical (max abs 0.0); `A = D_init.T`; `W_loc = 0`;
reader split sum = original head (3e-8 float rounding); forward label
independence (0.0); root codes vs production reference operators (0.0);
IHT per-tuple nnz ≤ 8; posterior bridge is the MLP in both arms; smoke (≤2
runs, ≤4 steps/arm, states discarded): W_loc task gradient at step 1 > 0,
D_loc/A task gradient 0 at step 1 (expected at W_loc=0) and > 0 by step 4.

## Dev evaluation (after both arms complete + roster frozen)

`b_g = median(g_fit − hat_g_raw_fit)` per arm (fit-only; never dev-calibrated);
`hat_g_cal = raw + b_g`. Main endpoint: dev **G0 (k=0) calibrated g-MAE**;
overall cal = common gate; raw reported in parallel. Paired bootstrap 1000×,
seed 20261006, the same resample indices per graph for both arms; G0/overall
use their own row sets. Bootstrap checks: same predictions → 0, swap → mirror.
CI does not cover training-seed uncertainty.

**Dictionary candidate gate (all five):** G0 cal gain ≥ 0.003 AND overall cal
gain ≥ 0.003 AND G0 cal CI lower > 0 AND G0 raw gain > 0 AND overall raw gain >
0, with no dictionary mechanism failure. Gain = MAE(M_COMP) − MAE(D_COMP);
positive = dictionary improves.

Classifications fixed ahead: `DICTIONARY_CANDIDATE` (above); `MLP_SUPPORTED_
IN_THIS_INTERFACE` (same five with M/D roles swapped); `LOCAL_EQUIVALENCE` (G0
and overall cal CIs both inside [−0.003, +0.003], no winning gate);
`DIRECTIONAL_NOT_CONFIRMED` (same-direction points, gate not passed); `INCONCLUSIVE`
(conflicting/indistinguishable; report which conditions failed — "not
significant" is never written as "useless/equivalent"); `MECHANISM_FAILED` /
`NOT_EXECUTED` / `RESOURCE_BLOCKED` recorded separately with the available
numbers. Table: fit/dev × G0/overall × raw/cal, bias, gap; per-k (0/−1/−2/≤−3)
n, MAE, Σ|err|/N contribution, gain; contributions sum = overall (verified).
One sensitivity run: drop M_COMP's dev max-|err| row (never a new main result).
If internal dev g-MAE < 0.09 it is only an internal-fold chemistry-component
score, never official-valid y-MAE, deployment or SOTA. No cross-fold/regime
gain splicing.

## Mechanism checks (no new search)

Fit-only health at init/ep40/ep120/ep240 from saved checkpoints: per-tuple nnz,
atoms used/dead dims, dictionary/A drift, task gradients (probe log), root-code
variance/effective rank, `W_loc` norm, injection RMS. 8-sparse is per tuple;
pooled root codes may be dense (pool nnz is never a failure verdict). For D at
minimum: the unrolled path actually receives COMP task gradients; codes and
injection not all-zero/per-row constant. Survival is necessary, not a gain.

Three frozen operator checks per arm on dev with the native fit bias (labels
never enter the intervention definition; predictions not recalibrated):
(1) zero local injection; (2) root code → that arm's fit-roots mean code (same
kappa/W_loc); (3) J aggregation switched to the existing marginal-product I
incidence (no new features, no I-model training). Report mean/max |Δpred|, g
raw/cal MAE and both component outputs. These are channel-sensitivity only:
a large zeroing loss includes synergy destruction, not an information share;
the J→I swap does not by itself prove a training-time inductive-bias claim.

Component error and cancellation per arm (fit/dev × G0/overall, ell and s raw
MAEs, never separately calibrated): `e_ell`, `e_s`, `e_g_raw = e_ell + e_s`,
`e_g_cal = e_ell + e_s − b_g`, `triangle_gap_raw = mean|e_ell| + mean|e_s| −
mean|e_ell+e_s|`, opposite-sign fraction, D−M component changes. One bias for
the total g only; component MAEs never added into a cal-g budget.

## Resources / execution

Wall target ≤120 min (hard 150; stop compute by 130). New training launched as
early as possible, at the latest 70 min; no new training/diagnostics after 100
min. GPU ≤1.0 allocation-GPU-hour total (incl. smoke/failure fragments),
concurrency ≤2; local CPU ≤8 threads; remote ≤4 threads/arm. Two arms on two
allocated physical GPUs in parallel via `rr run res-2 --pool res2-cu124`
(frozen commit, same regime/driver/torch, FP32, no AMP/DDP); GPU UUID/PCI
recorded per job (both showing `CUDA_VISIBLE_DEVICES=0` is normal). If only one
card is free, serial in the same regime if budget allows; never mix
CPU/GPU-fragment main comparisons; `RESOURCE_BLOCKED` instead of redesign.
Smoke: fit-only, ≤2 runs, ≤4 steps/arm, states discarded.

## Delivery / bookkeeping

New dir `tracks/ksvd/results/zinc_local_dictionary_component_supervision_
seed0_v1/`: PROTOCOL, METHOD_CONTRACT, REPORT (opens with the five answers),
DECISION, EXECUTION; run records; split/targets/prep/kappa/schedule/input
manifest; init identity; smoke checks; both arms' init/last/raw-soup +
1/40/120/240 checkpoints, curves, health, meta, allocation proof, budget;
fit/dev per-graph ell/s/g raw and g-cal predictions with stable IDs; main
table, group contributions, paired CI/gate, sensitivity, component
cancellation, frozen interventions, replay (`research verify`-compatible
manifest; no repo-wide new tests). Engineering failures: recovery preferred
(checkpoint + RNG state; torch RNG tensors moved to CPU before restore); at
most one from-scratch engineering restart per arm, only if no dev score was
seen, data/recipe unchanged and budget allows; discarded fragments never
select models; a bad loss/score is never a restart reason. Protocol, METHOD_
CONTRACT, code, input/prep/scale manifests, schedule and gates are committed
and frozen BEFORE the two formal runs. Isolated branch/worktree; no push, no
merge, no rewriting old artifacts, no touching others' work. All jobs terminal
before delivery.
