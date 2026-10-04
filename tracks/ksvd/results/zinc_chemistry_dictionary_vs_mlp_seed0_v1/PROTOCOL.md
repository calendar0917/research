# PROTOCOL — zinc-chemistry-dictionary-vs-mlp-seed0-v1

Single seed (seed 0). Exactly two formal GPU arms (`D_g`, `M_g`), one frozen
new internal fold, no official-valid/test.  This is a **matched current-
implementation** comparison, not a claim about all dictionaries or all MLPs.
Frozen before any new score is read.  Runner:
`tracks/ksvd/experiments/luyin16/zinc_chemistry_dictionary_vs_mlp_seed0_v1.py`.

## 1. Question and completion criteria

**Only main question.** With identical inputs, parameter counts, initialization
and training protocol, after removing the ring/cycle supervision (`y -> g`,
`g = y - c`), does the current shared task dictionary ``D_g`` generalise the
remaining chemical target `g` better than the matched MLP ``M_g``?

Completion criterion: a verifiable `D_g`/`M_g` comparison with an explicit
interpretation boundary.  A negative, locally equivalent or inconclusive result
is a valid completion; it is not repaired with extra arms.  If target
provenance or engineering cannot be guaranteed, the round delivers blocking
evidence and does not buy substitute experiments.

## 2. Budget and execution

| item | fixed requirement |
| --- | --- |
| wall clock | <= 150 min from first tool call; stop compute at 120 min, deliver in the rest |
| formal training | exactly `D_g`, `M_g`, seed 0, 240 epochs each; no third arm |
| GPU | <= 1.0 GPU-hour total (smoke, failures, repeats included); <= 2 GPUs at once |
| expected | paired y round used ~0.42 GPU-h; expected training 15-25 min parallel |
| local CPU | every compute subprocess explicitly <= 8 threads |
| server | `res-2`, pool `res2-cu124`, FP32, no AMP/DDP; both arms same regime |
| data | the frozen 8000 fit / 2000 dev fold of the 10000 official-train rows |
| search | no new seed/fold, no width/ISTA/λ/WD/loss scan, no early stopping |

If a full 240-epoch trajectory cannot run inside the remaining budget, formal
training is not started.  Short trajectories are never reported as formal.

## 3. Frozen fold, inputs and initialization

New fold generated once (same as the paired y round, not redrawn):

```python
perm = np.random.default_rng(20261004).permutation(10000)
fit_idx = np.sort(perm[:8000]); dev_idx = np.sort(perm[8000:])
```

* fit hash `7bf1cfb856a79617baf83c4ba95290d08248b4633609fa24725a5c028c9096d9`,
  dev hash `a61c801073fbda237a46e4f8fe6c8e4f8f24e867e7a8b452f7aaffc86a9cceff`;
* fit k counts 7713/252/30/5, dev 1915/73/10/2 (k=0/-1/-2/<=-3);
* schedule/data-stream hash `7b11a529584a8bdaa8e2f17a8c8413cdc6677714ae855339b5984e9fe6938e65`,
  15120 Adam steps per arm.

Inputs are the same fit-only refit standardizers of the paired y round
(`zw.load_and_prepare`), built only from `encoded_train.pt` + the train env
cache.  `load_split("train")` and any composite table that deserialises
official-valid are forbidden.  Initial states are the fresh seed-0
`build_arm('D'/'M')` states, verified item-for-item against the saved untrained
`D_init_state.pt` / `M_init_state.pt` of the paired y round; no trained
`D_y`/`M_y`/`O`/`H`/`S_M` state is ever loaded.

## 4. Target definition (fit-only, frozen)

`y = g + c` is a training-side decomposition of the label:

* `c = (k - mu_cycle) / sigma_cycle`, `k = clip(round(effective_cycle), -20, 0)`,
  `effective_cycle = eff_norm * sigma_cycle + mu_cycle`,
  `eff_norm = y_stored - [(logP - mu_logP)/sigma_logP + (SA - mu_SA)/sigma_SA]`;
* `mu_logP` is the fixed label-generation community constant
  `2.4570953396190123`; `sigma_logP/sigma_SA/mu_SA` are the OLS coefficients of
  `y_stored ~ logP + SA` on the cycle-free bulk, and `sigma_cycle/mu_cycle` are
  the Nelder-Mead snap constants.

The paired y round reused constants fitted on the whole 12000-row audit table
(which includes the current dev rows).  This round **refits every constant on
the 8000 fit rows only** and freezes them into `fit_only_targets.npz`
(committed).  Dev labels `k/c` are used only for grouped scoring; they never
enter training, input standardization, sampling or model selection.  The
prediction path reads no label.  Phase A reports the numerical difference from
the old decomposition; the fit-only definition is frozen regardless.

The frozen k labels turned out identical to the old ones (0 rows differ); `c`
differs by the constant refit (`max|c_new - c_old| = 0.03365`), so `g` differs
by the same per-group constant.  `y = g + c` holds exactly.

`G0` (`k = 0`) has constant `c0 = 0.0005790572`; therefore on G0 the new
`g`-MAE is exactly the y-unit MAE of `pred_g + c0` and is comparable to the old
y arms' G0 y-MAE.  Away from G0, old y-MAE and new g-MAE are **not** the same
deployment task and are never compared as one.

## 5. Arms

Common body: Sem108 + size2 (110-d), fusion `110 -> 342 -> 144`, static
unary/pair aggregation, existing global C6, topology25, reader.  Bridge
inserted once after fusion, before every consumer.

| arm | coding | parameters |
| --- | --- | --- |
| `D_g` | `rho = sqrt(mean(h^2)+1e-12)`, `x = h/rho`, `Dbar = column_normalize(D_L)`, 16-step ISTA (`lambda1=0.05`, `lambda2=0.01`), `E = rho * alpha @ V_L` | 82,944 |
| `M_g` | same `rho/x`, `E = rho * W2 SiLU(W1 x)`, biases off, `W1 = D_L_init.T`, `W2 = V_L_init.T` | 82,944 |

Body 184,667, total 267,611 per arm.  Non-bridge tensors are item-for-item
identical (SHA-256).  Initial D/M functions are not identical and this is not
claimed away.

## 6. Training protocol

* seed 0, 240 epochs, batch 128, Adam (coupled L2), `lr=1e-3`, `wd=1e-5`,
  global clip 5.0, no scheduler.
* loss exactly `mean L1(pred, g)`; no reconstruction term, auxiliary loss,
  severity weight, special sampler, stop-gradient or dropout change.
* shared batch schedule generated from `torch.Generator(seed=101)` over the 8000
  fit rows before training; 15,120 steps/arm; epoch 236-240 parameter soup with
  no dev-based member selection; init/last/raw-soup states saved.
* one calibration per arm: `b_g = median(g_fit - pred_raw_fit)` on the soup
  state in eval mode; `pred_cal = pred_raw + b_g`; raw states never contain
  `b_g`; old `b_y/b_O/b_P` are never added.

## 7. Endpoints, gains and gates

Gain is always `err(control) - err(candidate)`; positive means the candidate
is better (here `MAE(M) - MAE(D)`).

Primary endpoint: new-dev G0 calibrated `g`-MAE; primary gain
`G_g = MAE_G0(M_g) - MAE_G0(D_g)`.  All gains are reported raw and calibrated,
fit and dev, overall and per k group.

Four-arm G0 comparison (valid because `y = g + c0` on G0):

* `G_y = MAE_G0(M_y) - MAE_G0(D_y)`,
* `B_D = MAE_G0(D_y) - MAE_G0(D_g)`, `B_M = MAE_G0(M_y) - MAE_G0(M_g)`,
* interaction `I = B_D - B_M = G_g - G_y` (per-row identity enforced).

Statistics: 1000 paired bootstrap resamples, seed 20261004; the four arms share
the same G0 row resample indices; the new-arm overall g comparison uses shared
full-dev indices.  Intervals describe only the fixed seed / current fold.
Self-tests: same predictions -> zero gain/CI; swapped arms -> mirrored gain and
CI; constant shift -> bounded.

Sensitivity (pre-registered, no gate change): delete the single new-dev row
with the largest `|err_D_g_cal| + |err_M_g_cal|` and re-report overall and G0
gains on the rest.

Pre-registered dictionary category (evaluated top-down, `INVALID` first):

| category | definition |
| --- | --- |
| `INVALID` | label/init/parameter/input/pairing/loss/steps/replay contract failure |
| `D_CHEM_SUPPORT` | G_g cal >= +0.003 and CI lower > 0; G0 raw gain > 0; overall g-cal gain >= -0.001 |
| `M_CHEM_SUPPORT` | G_g cal <= -0.003 and CI upper < 0; G0 raw gain < 0; overall g-cal gain <= +0.001 |
| `LOCAL_EQUIVALENCE` | both G0 and overall g-cal gain CIs fully inside [-0.003, +0.003]; no raw >= 0.003 signal in the opposite direction |
| `TRADEOFF` | G0 cal >= 0.003, CI separated, raw same direction, but the G0-winning arm's overall g-cal worsens by > 0.001 |
| `INCONCLUSIVE` | everything else (a CI crossing zero is **not** equivalence) |

Independent target-interference label (descriptive):

| label | definition |
| --- | --- |
| `BULK_RELIEF_SUPPORTED` | B_D and B_M cal both >= 0.003, CI lower > 0, raw same direction |
| `D_SPECIFIC_RELIEF` | I cal >= 0.003 and CI lower > 0, raw same direction |
| `M_SPECIFIC_RELIEF` | I cal <= -0.003 and CI upper < 0, raw same direction |
| `RELIEF_UNCONFIRMED` | otherwise; reported with point values and intervals |

Both bridges consume dropout-free modules; the only training randomness is the
fixed schedule plus the three shared-body Dropout(0.05) draws, which are
function of the shared forward path.  Therefore each y/g arm pair (D_y vs D_g,
M_y vs M_g) shares its dropout mask sequence; the y->g change is target-only.
This is verified as a code-path statement and recorded as a descriptive
qualifier, not as a causal proof of every historical execution condition.

## 8. Stop / failure rules

Engineering failures (shape, device, replay, contract) may be fixed and
re-frozen; they are never an excuse to rerun a different configuration after
seeing scores.  No dev-based rescue, no extra seed, no third arm, no official
split, no write-back into historical result files (errata only).  If the server
is unavailable, the round is delivered as `NOT_EXECUTED` with the local Phase A
evidence and no substitute training.
