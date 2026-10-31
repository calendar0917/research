# REPORT — `lcrc-lipo-context-mechanism-check-v2`

Pre-registered, two-seed, exposed-Test mechanism check on the frozen
random42 split (`Train3360` / `Validation420` / `Test420`, raw logD).

**Test scope: every Test420 number in this report is
`EXPOSED_TEST_EXPLORATORY`.** Test labels and predictions were used
historically and these results feed the next decision. There is no blind-test
claim, no confirmatory claim and no new model selection after the formal runs.

## 0. Decisions

| dimension | decision |
| --- | --- |
| `PERFORMANCE_SIGNAL` | **`ABSENT`** |
| `ROLE_SPECIFICITY_SIGNAL` | **`ABSENT`** |

The two dimensions are independent. The first asks whether the BASE→CONTEXT
RMSE gain repeats across the two training seeds on both Validation420 and
Test420. It does not: 3 of the 4 required cells are positive, but seed 1
reverses sign between Validation and Test. The second asks whether the
role-specific two-mean context beats the parameter-matched all-atom
`ROLEBLIND` control. It does not: `ROLEBLIND_s0` is better on both splits, so
no part of any CONTEXT gain can be attributed to chemical-role grouping by
this experiment.

The formal runs were launched at commit `b0bef900282a`; analysis artifacts are
at `610ff6c3`. The protocol freeze and all 11 required CPU gates predate every
formal run. A single post-freeze, evaluation-only orchestration fix is
documented in §6 and `analysis/post_freeze_changes.json`; no training, model or
data code changed after freeze.

---

## 1. Frozen seed-0 checkpoints (Part A)

The historical checkpoints were loaded read-only and evaluated by independent
code in this round. No retraining, fine-tuning, checkpoint selection or v1
artifact modification was performed.

| checkpoint | SHA-256 | parameters |
| --- | --- | ---: |
| `base_s0` | `1a2820e376e6a426cc470b80bdf3f28a9f6bb9044d0028792463aeddabadc60d` | 247,106 |
| `context_s0` | `38bf6926c32db02f4c10f72da5c84abf2194c8b954681e0334aca88c91eccaee` | 255,458 |

Evaluation used the frozen Train3360 input structure normalisation, the
Train3360-only target scaler (mean `2.2081011904761905`, scale
`1.1906281212426577`), and raw-logD denormalisation before every metric.
Validation predictions reproduce v1 to `5.2e-9` RMSE (`base_s0`) and
`2.5e-8` RMSE (`context_s0`); per-molecule maximum absolute differences versus
the stored v1 predictions are `1.70e-6` and `1.49e-6`, i.e. float32
reduction-order noise from CPU-versus-GPU evaluation.

### 1.1 Seed-0 arm metrics

| split | arm | RMSE | MAE | R² | SSE |
| --- | --- | ---: | ---: | ---: | ---: |
| Validation420 | BASE | 0.580416 | 0.427854 | 0.777097 | 141.490871 |
| Validation420 | CONTEXT | 0.568587 | 0.411573 | 0.786090 | 135.782504 |
| Test420 `EXPOSED_TEST_EXPLORATORY` | BASE | 0.576987 | 0.403963 | 0.791450 | 139.824015 |
| Test420 `EXPOSED_TEST_EXPLORATORY` | CONTEXT | 0.544731 | 0.384851 | 0.814116 | 124.627563 |

Paired seed-0 comparison (`gain = RMSE(BASE) − RMSE(CONTEXT)`, positive =
CONTEXT better; interval is the 95 % percentile paired-bootstrap interval from
10,000 molecule resamples, seed `20261101`):

| split | gain | MAE change | R² change | SSE gain | CONTEXT-better | mean / max \|Δpred\| | gain interval |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Validation420 | +0.011829 | −0.016282 | +0.008993 | +5.708368 | 213 / 420 | 0.203474 / 1.190982 | [−0.016638, +0.012054, +0.039801] |
| Test420 `EXPOSED_TEST_EXPLORATORY` | +0.032256 | −0.019113 | +0.022666 | +15.196451 | 225 / 420 | 0.198331 / 1.199656 | [+0.001666, +0.031715, +0.065042] |

The seed-0 Test interval excludes zero, but the Validation interval does not.
Molecule resampling describes within-model sampling variability only; it does
not estimate training-seed variance. The second seed in §2 is the training-seed
replicate. For completeness, the seed-0 MAE-change intervals are
[−0.038881, −0.016113, +0.006734] (Validation) and
[−0.039821, −0.019154, +0.001667] (Test); the mean-|Δpred| intervals are
[0.186279, 0.203329, 0.221742] and [0.181563, 0.198396, 0.215910].

### 1.2 Test identity and labels

Test row IDs are exactly the frozen rows, in the frozen order:

* rows SHA-256 `4bc449106a64833973b8e8d3d16097bf5d2906c788d7dfcdf02a4b3a30ffa49b`;
* 420 rows; all five evaluated models aligned to the same IDs;
* labels read from the hash-verified raw CSV
  `data/lipophilicity/Lipophilicity.csv`
  (`aed41590cb30609d51d8e08ad3ff06495a76e80e211358801f596b10da69bacd`),
  Test label vector SHA-256
  `a0724b4c1ca6f7bb6b69b720ac2d020825685d9666544c7312e0f24d7e3a4021`.

### 1.3 Descriptive historical reference

The historical PNA-native Test RMSE is `0.528874` (MAE `0.373771`). It is a
descriptive reference on the same frozen split only. PNA was **not** retrained
in this round, and the recipe/architecture is not matched to the LCRC arms.
`CONTEXT_s0` Test RMSE `0.544731` is 0.015857 worse than that PNA reference;
`BASE_s0` is 0.048113 worse. The historical `LCRC-A-R3` Test RMSE was
`0.583985`, also descriptive and not a matched recipe.

---

## 2. Second training seed (Part B)

`BASE_s1` and `CONTEXT_s1` reused the frozen v1 architecture definitions and the
identical recipe: AdamW, batch 128, 27 steps/epoch, 360 epochs (9,720 updates),
peak LR 3e-4 → minimum 1e-5 with per-step cosine, weight decay 1e-3, gradient
clip 5.0, MSE on standardised Train3360 labels, dropout 0.10 on the local atom
representation, and a soup of the last five epoch-end state dicts. No early
stopping and no Validation-based checkpoint selection. Training seed 1 and
split seed 42.

The two seed-1 arms share bit-identical public initialisation (35 shared state
entries), the same batch order
(`29beb3028f5d459bbc5e67b5d6b08ea497e2825b02ecd6d65688a38c5cc154d1`), the
same dropout RNG reseed and the same optimiser settings. `BASE_s1` has 247,106
parameters; `CONTEXT_s1` has 255,458 (8,352 new). Both completed 9,720 steps.

| arm | checkpoint SHA-256 | soup state SHA-256 | train RMSE | train wall |
| --- | --- | --- | ---: | ---: |
| `BASE_s1` | `6cf238117189e1edf0a7be8a7e72b9749f3b32ec800efa1ff89920c4bd354092` | `cc359eedd851b959fa63199644d5a0705821a8d3d8af9e34aef569a981e26d67` | 0.268944 | 275.1 s |
| `CONTEXT_s1` | `2197d27cffb2532058652b396814ffd39efab202633a9d0b0232925339c8f202` | `43d505ba83c5a1bf567c96f34de0930bf6c80cd560ca635cf6b705e3741b0bbb` | 0.249902 | 410.5 s |

### 2.1 All four seed/split cells

| seed | split | BASE RMSE | CONTEXT RMSE | gain | BASE MAE | CONTEXT MAE | MAE change | R² BASE → CONTEXT |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | Validation420 | 0.580416 | 0.568587 | +0.011829 | 0.427854 | 0.411573 | −0.016282 | 0.777097 → 0.786090 |
| 0 | Test420 `EXPOSED_TEST_EXPLORATORY` | 0.576987 | 0.544731 | +0.032256 | 0.403963 | 0.384851 | −0.019113 | 0.791450 → 0.814116 |
| 1 | Validation420 | 0.576780 | 0.570922 | +0.005858 | 0.422146 | 0.422536 | +0.000390 | 0.779881 → 0.784329 |
| 1 | Test420 `EXPOSED_TEST_EXPLORATORY` | 0.554118 | 0.556859 | −0.002741 | 0.404287 | 0.403427 | −0.000860 | 0.807654 → 0.805747 |

Paired seed-1 detail (same conventions as §1.1):

| split | gain | SSE gain | CONTEXT-better | mean / max \|Δpred\| | gain interval |
| --- | ---: | ---: | ---: | ---: | --- |
| Validation420 | +0.005858 | +2.823626 | 206 / 420 | 0.203105 / 1.452406 | [−0.025003, +0.005304, +0.037265] |
| Test420 `EXPOSED_TEST_EXPLORATORY` | −0.002741 | −1.278949 | 217 / 420 | 0.180688 / 1.185065 | [−0.028259, −0.002463, +0.021262] |

The seed-1 Validation gain and Test loss both have intervals containing zero.
The sign reverses between splits: `+0.005858` on Validation, `−0.002741` on
Test. The seed-to-seed change in the Test gain is `−0.034997`, larger in
magnitude than seed 0's Test gain (`+0.032256`). With two seeds this is a
descriptive sensitivity check, not a variance estimate. The seed-1 MAE-change
intervals are [−0.022813, +0.000684, +0.023358] (Validation) and
[−0.020577, −0.000866, +0.020047] (Test); mean-|Δpred| intervals are
[0.186337, 0.203025, 0.220530] and [0.164810, 0.180733, 0.197151].

The seed-1 Validation gain is also concentrated in the five known tail rows:
without them the remaining 415 molecules go from RMSE `0.543590` to `0.545203`
(gain `−0.001613`), while the five tails contribute `+3.552562` SSE of the
`+2.823626` total. See §5.

### 2.2 Answer to Proposition 1

**Not established: there is no repeatable BASE→CONTEXT predictive value under
the pre-registered rule.**

* Seed 0 agrees in direction on both splits, but only the Test interval
  excludes zero.
* Seed 1 is positive on Validation (`+0.0059`, interval includes zero) and
  negative on Test (`−0.0027`, interval includes zero): a sign reversal.
* The pre-registered rule requires all four gains to be strictly positive.
  Only 3 of 4 are, so `PERFORMANCE_SIGNAL = ABSENT`.
* The single positive-with-interval result is seed 0 on an exposed Test set;
  seed 1 reverses it, and the seed-to-seed change is of the same order as the
  apparent effect. That is evidence of training-seed sensitivity, not
  replication.
* The seed-1 Validation result is tail-carried and does not survive removal of
  the five fixed extreme rows.

Molecule-level bootstrap intervals and training-seed variability are different
uncertainties. The intervals above answer "how stable is this trained model's
paired comparison over molecules?"; they do not answer "how much does the gain
move if the training seed changes?". The two-seed table is the only evidence
on the latter, and it shows a reversal.

---

## 3. ROLEBLIND control (Part C)

`ROLEBLIND_s0` keeps the exact CONTEXT parameter structure, names, shapes,
initialisation and count (255,458; 8,352 new; 43 state entries; 39 parameter
tensors). Only the context aggregation changes:

* `m_all = mean(h_i over all atoms of the molecule)`;
* `m_het = m_ch = m_all`;
* `a_i = tanh(W_atom h_i)`, `p_g = tanh(W_het m_het)`,
  `c_g = tanh(W_ch m_ch)`, `change_i = W_out(a_i · p_g · c_g)`;
* the correction is applied only to hetero atoms; `f` and `rho` still read the
  original `h`.

It was trained once with seed 0 on `res-gpu1`, same recipe, same batch order as
the historical seed-0 CONTEXT run
(`b10278f960300fef8fc9a4438947e9decee2ac62978427f2eefb7aa4bbe8f268`), no
search.

### 3.1 Initialisation and aggregation checks

| check | result |
| --- | --- |
| fresh CONTEXT_s0 vs fresh ROLEBLIND_s0, same-name values | bit-identical, 43/43 state entries, 39/39 parameters, 255,458 each |
| fresh CONTEXT_s0 init vs historical CONTEXT_s0 init | match (`a45caeea…`) |
| fresh BASE_s0 init vs historical BASE_s0 init | match (`743c0ea5…`) |
| `m_het` vs `m_ch` | bitwise identical |
| `m_all` vs manual per-molecule all-atom mean | max diff `3.58e-7` (float32 reduction order) |
| all-atom mean vs hetero-only mean | differs for **420/420** Validation molecules; max diff `2.51` |
| non-hetero atom correction | exactly `0.0` |
| batch composition / order invariance | max prediction diff `4.77e-7` CPU, `1.19e-6` GPU (tol `1e-6` / `1e-5`) |
| atom renumbering invariance | max diff `9.54e-7` |
| cross-molecule isolation | other molecules' context and predictions unchanged; `0.0` |

The GPU batch-invariance differences are float32 reduction-order noise, not
batch-dependent aggregation. Predictions for the same molecule are stable
across single, full, reversed, mixed and unrelated-graph batches to within
`1.2e-6` logD.

`ROLEBLIND_s0` checkpoint SHA-256:
`7f7bc2e7e40fb25f97f1982db37453ad50c952029ec5249424d9385268b79942`;
soup state SHA-256:
`9c3b900842b3a7a5fd7e56a7d28a07f5671e9411c28a4f46e600acede072e3dc`.

### 3.2 Seed-0 CONTEXT vs ROLEBLIND

`ROLEBLIND advantage = RMSE(CONTEXT) − RMSE(ROLEBLIND)`; positive means
ROLEBLIND is better. MAE advantage is `MAE(CONTEXT) − MAE(ROLEBLIND)`.

| split | CONTEXT RMSE | ROLEBLIND RMSE | ROLEBLIND advantage | CONTEXT MAE | ROLEBLIND MAE | MAE advantage | R² CONTEXT → ROLEBLIND |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Validation420 | 0.568587 | 0.534232 | **+0.034355** | 0.411573 | 0.386601 | +0.024972 | 0.786090 → 0.811159 |
| Test420 `EXPOSED_TEST_EXPLORATORY` | 0.544731 | 0.537335 | **+0.007396** | 0.384851 | 0.375737 | +0.009114 | 0.814116 → 0.819129 |

| split | ROLEBLIND-better molecules | mean / max \|Δpred\| | ROLEBLIND-advantage interval |
| --- | ---: | ---: | --- |
| Validation420 | 233 / 420 | 0.173838 / 1.397926 | [+0.006146, +0.034563, +0.062389] |
| Test420 `EXPOSED_TEST_EXPLORATORY` | 226 / 420 | 0.166733 / 1.059050 | [−0.015334, +0.007752, +0.029403] |

The paired bootstrap interval on Validation is entirely above zero in
ROLEBLIND's favour. The Test interval contains zero, but its point estimate and
median also favour ROLEBLIND; there is no split on which CONTEXT is better in
point estimate. In CONTEXT-minus-ROLEBLIND sign convention the intervals are
`[−0.062389, −0.034563, −0.006146]` and
`[−0.029403, −0.007752, +0.015334]`; a role-specific CONTEXT win would require
these to be entirely positive, which they are not. The ROLEBLIND MAE-advantage
intervals are [+0.003849, +0.025187, +0.046247] (Validation) and
[−0.009328, +0.009097, +0.027415] (Test); mean-|Δpred| intervals are
[0.158092, 0.173794, 0.190582] and [0.152080, 0.166638, 0.182251].

### 3.3 Answer to Proposition 2

**No: chemical-role grouping does not beat whole-graph context in this control.
`ROLE_SPECIFICITY_SIGNAL = ABSENT`.**

The parameter-matched all-atom control is better on both splits. The
role-specific CONTEXT gain therefore cannot be attributed to the chemical-role
split; if anything, the role-specific two-mean parameterisation is the weaker
summary here. This conclusion is independent of the `PERFORMANCE_SIGNAL`
decision: even if BASE→CONTEXT had repeated, the ROLEBLIND result would still
deny a role-specific mechanism attribution.

Caveat: `m_all` is not an information-matched "less specific" copy of the two
role means. It weights the role means by their atom counts and therefore also
encodes the hetero-atom fraction/composition, which may itself be predictive.
This experiment shows that the **role-specific grouping is not necessary** for
the context gain; it does not identify whether the all-atom mean wins because
of composition information, lower variance, parameter sharing, or another
effect. No post-hoc model was trained to separate those explanations.

---

## 4. Training curves

All arms use the pre-registered last-five-epoch soup; no best epoch was
selected. The soup is worse than the best single epoch in every arm.

| arm | best epoch Validation RMSE | best epoch | soup Validation RMSE | last epoch | grad-clip events / 9,720 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `BASE_s0` | 0.564350 | 181 | 0.580416 | 0.581419 | 4,985 |
| `CONTEXT_s0` | 0.560576 | 227 | 0.568587 | 0.570557 | 4,478 |
| `BASE_s1` | 0.563294 | 176 | 0.576780 | 0.576804 | 4,672 |
| `CONTEXT_s1` | 0.544074 | 128 | 0.570922 | 0.573604 | 4,441 |
| `ROLEBLIND_s0` | 0.527955 | 282 | 0.534232 | 0.534138 | 4,603 |

Curves are in `analysis/runs/*/curve.csv`, merged in
`analysis/training_curves.csv` and plotted in `analysis/training_curves.png`.
Frequent clipping is part of the fixed recipe and was not tuned in response.

---

## 5. Tail and non-tail diagnostics (descriptive only)

The five row IDs `{1630, 2924, 822, 1153, 3634}` were fixed by the earlier
`lcrc_lipo_random42_gap_audit_v1`; no new tail search was performed. Positive
`Δ|error|` means the second model is better for that row.

| row | y logD | BASE pred | CONTEXT pred | \|e\| BASE | \|e\| CONTEXT | Δ\|e\| CONTEXT−BASE | ROLEBLIND pred | \|e\| ROLEBLIND | Δ\|e\| ROLEBLIND−CONTEXT |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1630 | 3.90 | 1.33601 | 1.35444 | 2.56399 | 2.54556 | +0.01844 | 1.39950 | 2.50050 | +0.04505 |
| 2924 | 3.08 | 0.96212 | 0.86180 | 2.11788 | 2.21820 | −0.10031 | 1.50894 | 1.57106 | +0.64714 |
| 822 | 3.44 | 2.11112 | 1.90887 | 1.32888 | 1.53113 | −0.20225 | 1.91226 | 1.52774 | +0.00339 |
| 1153 | 3.37 | 1.67161 | 1.88824 | 1.69839 | 1.48176 | +0.21662 | 1.82748 | 1.54252 | −0.06076 |
| 3634 | 1.81 | 3.38022 | 3.54836 | 1.57022 | 1.73836 | −0.16815 | 3.42785 | 1.61785 | +0.12051 |

Summary:

| comparison | split | tail net SSE | tails better | tail share of net SSE gain | non-tail RMSE (415) | non-tail gain |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| CONTEXT − BASE, seed 0 | Validation420 | −0.786656 | 2 / 5 | −0.1378 | 0.545111 → 0.530561 | +0.014550 |
| CONTEXT − BASE, seed 1 | Validation420 | +3.552562 | 3 / 5 | 1.2582 | 0.543590 → 0.545203 | −0.001613 |
| ROLEBLIND − CONTEXT, seed 0 | Validation420 | +2.910586 | 4 / 5 | 0.1829 | 0.530561 → 0.500164 | +0.030397 |

The seed-0 CONTEXT gain is **not** carried by the five tails; it is slightly
worse on them and positive on the remaining 415. The seed-1 Validation gain is
the opposite: entirely tail-carried, with the other 415 molecules slightly
worse. The ROLEBLIND advantage is broad and only partly tail-related. These
slices are descriptive; the decisions do not use them.

---

## 6. Verification, budget and provenance (Part D)

### 6.1 Gates

All **11/11 required CPU gates** passed before any formal run
(`analysis/cpu_gates.json`, `n_passed = 11`), plus supplementary checks:

1. frozen split/labels/Test-row hashes and sizes;
2. shared seed-1 initialisation, batch order and optimiser settings;
3. ROLEBLIND initialisation equals fresh CONTEXT_s0 bitwise;
4. ROLEBLIND all-atom aggregation (`m_het = m_ch = m_all`);
5. ROLEBLIND batch-composition/order invariance;
6. atom-renumbering invariance for all three families;
7. cross-molecule aggregation isolation;
8. finite non-zero gradients on all new tensors and parameter updates;
9. Test inference changes no weights, creates no gradients and stays aligned;
10. denormalisation and RMSE/MAE/R² recomputation;
11. CPU two-batch forward/backward/AdamW smoke for all three families.

Supplementary checks included dropout execution for `BASE_s1`,
`CONTEXT_s1` and `ROLEBLIND_s0` (dropout masks atoms in train mode, identity in
eval mode; hook counts 10,462 / 10,529 / 10,537), parameter counts, `W_out = 0`
equivalence, readout parity, ROLEBLIND usage on Validation420, and a scan of
training code for Test access.

### 6.2 Test inference

The freeze-weights audit ran on all five checkpoints for both Validation and
Test. In every case: zero changed state entries, no parameter received a
gradient, `model.training` stayed false, 420 finite predictions, and row order
remained aligned. Test labels were opened only in the evaluation process, after
all three new checkpoints were frozen.

### 6.3 GPU ledger

Charging rule: full `rr`-measured process lifetime of every CUDA process in
this round, both pools summed; historical v1 is not charged against this cap.

| run | pool | state | process window (local) | charged | note |
| --- | --- | --- | --- | ---: | --- |
| `m2-cuda-smoke` | `res-gpu0` | completed | 23:09:57–23:10:04 | 7 s (0.00194 h) | roleblind audits pass on GPU |
| `m2-base-s1` | `res-gpu0` | completed | 23:13:43–23:18:26 | 283 s (0.07861 h) | 9,720 steps |
| `m2-context-s1` | `res-gpu1` | completed | 23:13:43–23:20:45 | 422 s (0.11722 h) | 9,720 steps |
| `m2-roleblind-s0` | `res-gpu1` | completed | 23:24:32–23:31:25 | 413 s (0.11472 h) | 9,720 steps |

**Total: 1,125 s = 0.3125 / 0.45 A100 GPU·h (69.4 %); remaining 0.1375 h.**
No failures or retries. Host `a100-2`, 2 × NVIDIA A100-SXM4-40GB, driver
550.163.01, PyTorch 2.5.1+cu124, CUDA 12.4, fp32. Every run is recorded as
clean (`dirty: false`) at commit `b0bef900282a`.

The in-process budget guard reads the remote `analysis/rr_status/` directory,
which is not synced from the local pull area; consequently `train.json` records
`budget_before_gpu_hours = 0.0` for all three runs. The authoritative ledger is
the local `analysis/gpu_ledger.json`, checked after every run; before the
ROLEBLIND run it stood at 0.1978 h, and the final total is 0.3125 h. The cap was
never approached.

### 6.4 Post-freeze evaluation fix

One post-freeze change was made, after all formal training and seed-0
evaluation were complete, to `code/m2_evaluate.py`: `evaluate_all()` looked up
`ARM_SPECS` for the historical keys `base_s0`/`context_s0`, which are not in
`ARM_SPECS`. The precheck now uses `model_registry()`. This is an
evaluation-orchestration fix only. `m2_train.py`, `m2_models.py`, `m2_data.py`
and all other training/model code remain at the frozen hashes. The exact diff
and hashes are in `analysis/post_freeze_changes.json` and
`analysis/post_freeze_m2_evaluate.diff`. No checkpoint, prediction or metric
was altered by this fix; `eval-all` was run only after all three new
checkpoints existed.

### 6.5 Data and cache provenance

* split file SHA-256 `1834414a6da73e9d2c5942ad157a345f08653110acfc4b9fef83c3064bc233b4`;
* Train/Validation labels SHA-256 `1da269356e0e89974a7388d1e430d7e810a0ab1056be5cd747a1c8ed3d2009ca`;
* Test CSV SHA-256 `aed41590cb30609d51d8e08ad3ff06495a76e80e211358801f596b10da69bacd`;
* Test row hash `4bc44910…a49b`, Test label vector hash `a0724b4c…4021`;
* LCRC cache **content fingerprint**
  `23ec00be4fb7e3cf93967cd62911bd932159b1d7e0b43226b23cf7c6a3718e03`,
  `labels_present = false`.

The LCRC cache is not tracked. The local cache file SHA-256 is `450e7490…2bd8`
while the independently generated remote cache file SHA-256 is `0ff7d074…364e`;
the content fingerprint, molecule count, label-free certification and feature
identity are identical, and the remote training and local evaluation both
verify that fingerprint. This is `.pt` serialization nondeterminism, not a data
difference.

Before deploying the formal code, the remote checkout still had untracked v1
result copies that blocked the merge. Their SHA-256 values matched the local
committed v1 artefacts exactly for all nine files, so they were moved aside on
the host (not deleted or edited); no v1 checkpoint, metric or code was changed.

---

## 7. Limitations

1. **Test420 is exposed.** Every Test number is
   `EXPOSED_TEST_EXPLORATORY`; this is not a blind test and cannot support a
   confirmatory generalisation claim.
2. **Only two training seeds.** The seed-1 reversal is evidence of sensitivity,
   not a variance estimate. More seeds would be needed to quantify
   seed-to-seed variability; no such runs are proposed here.
3. **One frozen split.** Validation and Test are both 420 molecules from the
   same random42 split; the two splits are not independent samples of a new
   chemical distribution.
4. **ROLEBLIND is not information-matched.** The all-atom mean also encodes
   hetero-atom fraction/composition. The control establishes lack of
   role-specificity support, not the reason the all-atom summary performs
   better.
5. **Test metrics use CPU evaluation.** Historical seed-0 Validation was
   reproduced to float32 reduction-order noise (`≤1.7e-6` per molecule); the
   new checkpoints were trained on GPU and evaluated on CPU. Determinism within
   each device is checked by the gates.
6. **The budget guard's remote blind spot.** In-run `budget_before_gpu_hours`
   is `0.0` because remote `rr_status` is not synced; the local ledger is the
   authoritative record. No cap breach occurred.
7. **Frequent gradient clipping** (4,441–4,672 of 9,720 steps per new arm) is
   part of the fixed recipe and was not tuned.
8. **Post-freeze evaluation-only fix.** It changed no model, training or data
   code and no artifact; it is recorded rather than hidden.

---

## 8. Artefacts and commits

| item | path |
| --- | --- |
| protocol / freeze | `PROTOCOL.md`, `configs/context_mechanism_check_v2.yaml`, `analysis/protocol_freeze.json`, `analysis/protocol_sha256.json` |
| data audit / 11 gates | `analysis/data_audit.json`, `analysis/cpu_gates.json` |
| CPU / CUDA smokes | `analysis/smoke_cpu/`, `analysis/smoke_cuda/` |
| formal runs (checkpoints, curves, diagnostics, predictions) | `analysis/runs/base_s1/`, `analysis/runs/context_s1/`, `analysis/runs/roleblind_s0/` |
| rr status / GPU ledger | `analysis/rr_status/*.json`, `analysis/gpu_ledger.json` |
| seed-0 frozen evaluation | `analysis/eval_frozen_s0/` |
| all-model evaluation | `analysis/eval/metrics.json`, `analysis/eval/bootstrap.json`, `analysis/eval/per_molecule_valid420.csv`, `analysis/eval/per_molecule_test420.csv` |
| per-molecule prediction arrays | `analysis/predictions/*.npz` |
| comparison tables | `analysis/table_seed01.csv`, `analysis/table_roleblind.csv`, `analysis/table_tail.csv` |
| training curves | `analysis/runs/*/curve.csv`, `analysis/training_curves.csv`, `analysis/training_curves.png` |
| decisions | `DECISION.md`, `analysis/decision.json`, `analysis/comparison_seed01.json`, `analysis/roleblind_comparison.json`, `analysis/tail_audit.json` |
| post-freeze fix | `analysis/post_freeze_changes.json`, `analysis/post_freeze_m2_evaluate.diff` |

Formal deployment and runs: commit `b0bef900282a`. The analysis pipeline, the
evaluation-only fix and the first artifact pass: commit `610ff6c3`. The
evaluation metadata was regenerated at a clean checkout and this report and
`DECISION.md` are committed together; see `git log -1 -- REPORT.md` for the
exact report revision.

No follow-up model experiment, retraining, architecture/LR change,
Validation-based checkpoint selection or new model proposal was made after the
formal runs. The historical PNA checkpoint was not retrained, and the v1
one-shot Test lock was not modified.
