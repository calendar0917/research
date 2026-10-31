# PROTOCOL — `lcrc-lipo-context-mechanism-check-v2`

Frozen protocol for the open-Test re-evaluation, the second training seed, and
the capacity-matched chemical-role negative control.  Everything below is fixed
**before any formal training run** and enforced in code
(`code/m2_common.py:validate_config`, `code/m2_gates.py`, `tests/`).

| field | value |
| --- | --- |
| Protocol ID | `lcrc-lipo-context-mechanism-check-v2` |
| Track | `tracks/structural-sufficiency/` |
| Experiment root | `tracks/structural-sufficiency/lcrc_lipo_context_mechanism_check_v2/` |
| Data source (read-only) | `tracks/structural-sufficiency/lcrc_lipo_esa_random42_v1/` |
| Historical models (read-only) | `tracks/structural-sufficiency/lcrc_lipo_contextual_contrib_v1/` |
| Split seed | `42` (frozen random42: Train3360 / Validation420 / Test420) |
| Training seeds | `0` (ROLEBLIND) and `1` (second BASE/CONTEXT seed) |
| Test access | **`EXPOSED_TEST_EXPLORATORY`**: allowed for scoring, never for training or selection |
| GPU budget | **0.45 A100 GPU·h hard cap** for this round's three trainings + smoke + retries, both pools summed |
| Host / pools | `res`; BASE_s1 → `res-gpu0`, CONTEXT_s1 → `res-gpu1`, ROLEBLIND_s0 → `res-gpu1` |

---

## 0. Why Test is exposed

The frozen random42 Test420 split was already used by
`lcrc_lipo_esa_random42_v1` and by earlier exploratory rounds.  This round reads
Test labels and scores Test predictions deliberately.  Every Test number is
labelled `EXPOSED_TEST_EXPLORATORY`; it is not an independent blind test and
must not be described as one.  The historical one-shot Test lock, score function
and frozen artifacts in `lcrc_lipo_esa_random42_v1` are **not** modified.

---

## 1. Data (frozen, read-only)

Read (hash-verified, never written):

* `lcrc_lipo_esa_random42_v1/data/split_esa_random42_v1.json`
  SHA256 `1834414a6da73e9d2c5942ad157a345f08653110acfc4b9fef83c3064bc233b4`
* `lcrc_lipo_esa_random42_v1/data/labels_train_valid_v1.json`
  SHA256 `1da269356e0e89974a7388d1e430d7e810a0ab1056be5cd747a1c8ed3d2009ca`
* `data/lipophilicity/Lipophilicity.csv`
  SHA256 `aed41590cb30609d51d8e08ad3ff06495a76e80e211358801f596b10da69bacd`

| split | n | row-index SHA256 (`int64`) | first five rows |
| --- | ---: | --- | --- |
| Train | 3360 | `0df18c72…48ee` | `3942, 3415, 1426, 2365, 3462` |
| Validation | 420 | `8f9c3517…9b44` | `501, 1649, 3579, 2873, 106` |
| Test | 420 | `4bc44910…a49b` | `967, 3557, 3492, 462, 2258` |

Test labels are read only from the hash-verified raw CSV, in frozen Test row
order.  The resulting float64 vector must reproduce the historical one-shot
Test label hash
`a0724b4c1ca6f7bb6b69b720ac2d020825685d9666544c7312e0f24d7e3a4021`.

Target normalisation is the frozen Train3360-only StandardScaler
(`mean = 2.2081011904761905`, `scale = 1.1906281212426577`).  LCRC structure
normalisation is the frozen Train3360 fit
(fingerprint `4d6e5bf456ff9d2217eb11a2c4a29d3f7cbbcb95665bf56e62b93c3ae93d0e26`).
The label-free 4,200-molecule LCRC cache must match content fingerprint
`23ec00be4fb7e3cf93967cd62911bd932159b1d7e0b43226b23cf7c6a3718e03`.

Training receives only Train3360 and Validation420 caches and labels.  Test is
never loaded by `code/m2_train.py` or `code/m2_models.py`; a static scan is part
of the supplementary gates.

---

## 2. Part A — evaluate the frozen seed-0 checkpoints

Load read-only:

* BASE: `lcrc_lipo_contextual_contrib_v1/analysis/runs/base_s0/checkpoint.pt`
  SHA256 `1a2820e376e6a426cc470b80bdf3f28a9f6bb9044d0028792463aeddabadc60d`
* CONTEXT: `lcrc_lipo_contextual_contrib_v1/analysis/runs/context_s0/checkpoint.pt`
  SHA256 `38bf6926c32db02f4c10f72da5c84abf2194c8b954681e0334aca88c91eccaee`

No retraining, no fine-tuning, no checkpoint selection.  Both checkpoints are
the historical last-five-epoch soups.  Report Test420 RMSE, MAE, R², SSE,
per-molecule paired predictions/errors, and paired molecule-level bootstrap
intervals for Validation420 and Test420.

Historical descriptive reference on the same random42 split: PNA-native Test RMSE
`0.5288742856070585` (not retrained in this round).  The historical LCRC-A-R3
Test RMSE `0.5839847887469122` is reported only as an additional descriptive
anchor.

---

## 3. Part B — BASE_s1 and CONTEXT_s1

The BASE and CONTEXT classes are imported unchanged from
`lcrc_lipo_contextual_contrib_v1/code/x1_models.py`; the training recipe is
identical to the frozen v1 recipe.

* training seed **1**, split seed 42;
* 360 epochs, batch 128, 27 steps/epoch, 9,720 updates;
* AdamW, betas `(0.9, 0.999)`, eps `1e-8`, weight decay `1e-3`;
* peak LR `3e-4` → minimum LR `1e-5`, per-step cosine;
* gradient clip `5.0`;
* MSE on the Train3360-standardised target;
* dropout 0.10 on the final atom-level local `h` (`v2_models.DropoutAtLocal`),
  actually executed for both arms;
* checkpoint = average of the last 5 epoch-end `state_dict`s; no early stopping,
  no Validation-based selection.

Shared public initialisation: both models are built at seed 1; every state entry
that also exists in BASE is overwritten with BASE's value, so shared parameters
and buffers are bit-identical.  The global torch RNGs are reseeded after model
construction; the shuffle generator seed is `1 + 917000` for both arms, so the
batch order is identical and the dropout masks are identical.

---

## 4. Part C — ROLEBLIND_s0

ROLEBLIND keeps every CONTEXT module and every parameter; it has exactly the
same four `Linear` layers with the same shapes, the same total parameter count
(255,458), the same initialisation, and the same state-dict keys.

Original CONTEXT:

```
m_het = mean(h_i for hetero atoms)
m_ch  = mean(h_i for carbon/halogen atoms)
```

ROLEBLIND:

```
m_all = mean(h_i for all atoms in the same molecule)
m_het = m_all
m_ch  = m_all
```

Everything else is unchanged:

```
a_i      = tanh(W_atom h_i)
p_g      = tanh(W_het m_het)
c_g      = tanh(W_ch  m_ch)
change_i = W_out(a_i * p_g * c_g)
h_new_i  = h_i + change_i   if i is hetero
h_new_i  = h_i              otherwise
y        = sum_i f(h_new_i) + rho(G, original h statistics)
```

`f` and `rho` are untouched; `rho` still reads the original `h`.  No extra input,
no reduced parameter count, no extra module.

The freshly built CONTEXT_s0 and ROLEBLIND_s0 must have every same-name
parameter/state entry bitwise identical at seed 0 (gate).  ROLEBLIND_s0 is
trained once on `res-gpu1` with the same recipe, seed 0.

---

## 5. Pre-GPU correctness gates (11 required)

No CUDA work before all required gates pass on CPU:

1. `data_split_hashes` — all three row lists, sizes, order, hashes; labels;
   Test label vector; cache content fingerprint; normalisation fingerprints.
2. `shared_init_seed1` — BASE_s1/CONTEXT_s1 shared state entries bitwise
   identical, extra entries only `context.*`, parameter counts 247,106 / 255,458
   (+8,352), identical batch order and optimisation settings.
3. `roleblind_init_equals_context_s0` — full state/parameter parity, identical
   counts, and the fresh CONTEXT_s0 init hash equals the historical v1 hash.
4. `roleblind_all_atom_aggregation` — `m_het = m_ch = m_all` exactly; `m_all`
   equals the manual per-molecule all-atom mean; `p_g`, `c_g` use `m_all`;
   correction is zero on non-hetero atoms.
5. `roleblind_batch_invariance` — predictions unchanged under single-molecule,
   full-batch, reversed-order, even/odd subset and mixed-batch composition.
6. `atom_renumbering_invariance` — BASE, CONTEXT and ROLEBLIND.
7. `cross_molecule_isolation` — perturbing one molecule never changes another
   molecule's context or prediction.
8. `nonzero_gradients` — four new CONTEXT and four new ROLEBLIND tensors receive
   finite non-zero gradients and move after two real optimiser steps.
9. `test_inference_freezes_weights` — state entries bitwise unchanged and no
   gradients after Test inference; predictions row-aligned to the frozen split.
10. `denormalisation_and_metrics` — recomputation in raw logD.
11. `cpu_two_batch_smoke` — two real batches through forward/backward/AdamW for
    all three formal arms.

Supplementary: dropout execution, parameter counts, `W_out = 0` equivalence for
ROLEBLIND, readout parity with CONTEXT under identical context output, ROLEBLIND
all-atom-vs-hetero mean contrast, training-code Test-access scan.

---

## 6. Evaluation, bootstrap and fixed decision rules

Primary units: raw logD.  Primary pair: BASE vs CONTEXT; the same seed and split
are paired molecule by molecule.

Paired bootstrap (descriptive only): 10,000 resamples of the 420 molecules with
replacement, fixed seed `20261101`, percentile 2.5 / 50 / 97.5.  Reported
statistics: `gain_rmse = RMSE(ref) − RMSE(other)` (positive = `other` better),
`mae_change = MAE(other) − MAE(ref)` (negative = `other` better),
`sse_gain = SSE(ref) − SSE(other)` (positive = `other` better), and the mean
absolute prediction difference.  The unit is the molecule, and the same
resampled molecules are used for both arms (paired).

Two independent conclusion dimensions:

* **`PERFORMANCE_SIGNAL`** = `PRESENT` iff all four BASE→CONTEXT RMSE gains
  (seed 0 and seed 1, Validation420 and Test420) are positive.  Otherwise
  `ABSENT`.  A sign reversal between Validation and Test inside a seed is
  reported explicitly.  The paired bootstrap describes molecule-sampling
  variability inside one trained model; the seed-0 vs seed-1 difference is a
  separate second training draw.
* **`ROLE_SPECIFICITY_SIGNAL`** = `PRESENT` iff CONTEXT_s0 beats ROLEBLIND_s0 by
  RMSE on both Validation420 and Test420 and both paired-bootstrap 95%
  percentile intervals of the CONTEXT gain exclude 0.  Otherwise `ABSENT`.
  If ROLEBLIND is close to or better than CONTEXT, the report must state that
  chemical-role specificity is unsupported and must not attribute CONTEXT's
  gain to a chemical interaction mechanism.

These are pre-registered descriptive rules, not confirmatory hypothesis tests.

Tail description uses only the five row IDs fixed by
`lcrc_lipo_random42_gap_audit_v1`: `{1630, 2924, 822, 1153, 3634}` (all in
Validation420).  Report per-row error changes and the RMSE of the remaining 415
molecules for CONTEXT_s0 vs BASE_s0 and CONTEXT_s0 vs ROLEBLIND_s0.  No new tail
search and no model change.

---

## 7. GPU budget and run plan

Hard cap **0.45 A100 GPU·h**, charged as the full rr-measured process lifetime of
every CUDA process of this round (training, CUDA smoke, failures/retries).  Both
pools are summed; parallel pools are not wall-clocked.  Historical v1 usage is
recorded separately and is not charged here.

| pool | arm | command |
| --- | --- | --- |
| `res-gpu0` | BASE_s1 | `rr run res m2-base-s1 --pool res-gpu0 -- ... train --arm base_s1 --device cuda --pool res-gpu0` |
| `res-gpu1` | CONTEXT_s1 | `rr run res m2-context-s1 --pool res-gpu1 -- ... train --arm context_s1 --device cuda --pool res-gpu1` |
| `res-gpu1` | ROLEBLIND_s0 | `rr run res m2-roleblind-s0 --pool res-gpu1 -- ... train --arm roleblind_s0 --device cuda --pool res-gpu1` |

BASE_s1 and CONTEXT_s1 run in parallel, one per GPU.  ROLEBLIND_s0 follows on
`res-gpu1`, the same pool as the historical CONTEXT_s0.  One short CUDA smoke for
all three arms precedes the formal runs.  Each formal arm is trained once; no
configuration, duration or checkpoint rule may be changed after seeing
Validation or Test results.

---

## 8. Abort conditions

* any split/label/Test-label/cache/normalisation hash mismatch;
* any required CPU gate or unit test failure;
* a changed config, protocol or code hash after the freeze;
* an arm launched on the wrong pool or with the wrong seed;
* GPU ledger at or above the cap before a new run;
* any Test data entering training or checkpoint selection.

---

## 9. Deliverables

```
configs/context_mechanism_check_v2.yaml
code/  (m2_*.py, run_m2.py)
tests/
analysis/data_audit.json
analysis/cpu_gates.json
analysis/protocol_freeze.json (+ protocol_sha256.json)
analysis/eval_frozen_s0/{metrics.json,bootstrap.json,per_molecule_*.csv,metrics_by_model_split.csv}
analysis/smoke_cpu/, analysis/smoke_cuda/
analysis/runs/{base_s1,context_s1,roleblind_s0}/{checkpoint.pt,train.json,curve.csv,valid_predictions.npz,diagnostics.json,envelope.json}
analysis/rr_status/*.json
analysis/gpu_ledger.json
analysis/eval/{metrics.json,bootstrap.json,per_molecule_valid420.csv,per_molecule_test420.csv,metrics_by_model_split.csv}
analysis/comparison_seed01.json
analysis/roleblind_comparison.json
analysis/tail_audit.json
analysis/decision.json
analysis/table_seed01.csv, table_roleblind.csv, table_tail.csv
analysis/training_curves_summary.json
REPORT.md
DECISION.md
```

This document may not be edited after the protocol freeze.  Any post-hoc
deviation is listed in `REPORT.md`, not patched here.
