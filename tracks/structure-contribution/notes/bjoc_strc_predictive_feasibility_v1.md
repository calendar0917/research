# STRC predictive-feasibility v1 — note (pre-registration)

Protocol: `cscl-bjoc-strc-predictive-feasibility-v1`
Config: `configs/bjoc_strc_predictive_feasibility_v1.yaml`
Driver: `code/run_bjoc_strc_v1.py` · Model: `code/bjoc_strc_v1.py` · Tests: `tests/test_bjoc_strc_v1.py`

## Question

On the already-successful masked-pretrained BJOC-Null (P; official-valid
seed-mean MAE 0.083549, exposed-test 0.073957), can a *tiny*,
cross-molecule-shared, explicitly additive structural residual channel
improve held-out single-model raw-y MAE when trained jointly under the
identical recipe?

This round is a **predictive-mechanism experiment**, not the final
explanatory model.  It answers exactly one falsifiable proposition:

> Giving the model an explicit, cross-molecule-shared per-type /
> per-junction-context contribution channel — trained jointly under the same
> raw-y L1 objective — will lower official-valid MAE without degrading
> ordinary molecules.  If it does not, the shared additive residual route is
> stopped.

## Model (only one)

```
y_hat(G) = f_theta(G) + s_a * sum_i alpha[type_i]
                     + s_j * sum_j beta[context_j]
```

* `f_theta` = complete pretrained BJOC-Null forward (413,505 params,
  unchanged; same forward as P).
* `alpha` = 28 shared scalars over the raw PyG atom-type codes, zero-init.
* `beta` = 129 shared scalars over a frozen junction-context vocabulary
  (128 kept + OTHER), zero-init.
* context = `(endpoint_type_low, centre_type, endpoint_type_high)` with the
  endpoint types sorted and the centre explicit.
* `s_a = 1 / mean_train(n_atoms) = 0.0431659645...`,
  `s_j = 1 / max(mean_train(n_junctions), 1) = 0.0288718609...`, frozen,
  non-learnable, train-only structure statistics.
* New parameters: 28 + 129 = 157; total 413,662.  No MLP, no calibration, no
  `a(counts)*f+b(counts)`, no extra features, no GNN/attention/iteration.

Vocabulary rule (train10k structures only, label-blind): count each junction
context once, sort by `(-count, lexicographic context)`, keep the most
frequent 128, map everything else to `OTHER`.  Observed on official train10k:
201 distinct contexts; the kept 128 cover 99.95% of junctions
(`vocab_sha256 = 18303ee133b580107c9c626ed834f94a45c61320fcec82f123a6aa413973cc40`).

## Matching to P (the critical constraint)

S must start from the **same SSL-after, supervision-before** state as P, not
from the P soup:

* base = `f10k._make_downstream_model(seed)` (scratch init replaying the
  recorded P `init_full_sha256` / `init_group_state_sha256`), then the frozen
  `pretrain_encoder_s{seed}.pt` encoders, then the zero-init STRC wrapper;
* recipe identical: Adam lr 1e-3 / wd 1e-5 / batch 128 / clip 5 / fp32 /
  240 epochs / 18,960 updates / last-5 epoch parameter-mean soup;
* batch order: `torch.Generator(seed + 101)`, hash must equal P's recorded
  `batch_order_hash`.

## CPU pre-check (all passed, `cpu_precheck.json`)

* A1 init equality: STRC forward == P initialisation forward on 128 train
  molecules, max abs diff **0.0**; residual identically 0.
* A2 correctness: atom relabelling 0.0; batch composition 2.1e-7 (< 1e-6);
  checkpoint round-trip 0.0; one count per atom/junction; all params/grads
  finite; no forbidden modules.
* A3 gradients: alpha/beta receive non-zero gradient and move in the first
  optimizer step; optimizer contains every parameter; base gradients equal
  the zero-residual control **exactly** (max diff 0.0).  Note: the first-batch
  global grad norm is 11.98 > 5, so the shared clip is active (it is active
  for the base alone as well, i.e. P also clipped).
* A4 sharing: `d y_hat / d alpha[t] = s_a * n_t` and
  `d y_hat / d beta[k] = s_j * n_k` exactly (0.0).
* A6 design audit: rank 149/157; 7 zero-support atom types (raw ZINC classes
  21–27; 27 is the mask token — their alpha entries never move); one exactly
  collinear context pair; nonzero-spectrum condition number ~3.1e3.  The
  individual coefficients are therefore only partially identifiable; only
  fitted predictions are claimed.
* A5 synthetic: planted additive truth recovered (residual RMS 0.017 vs
  σ=0.05; identifiable cosine 0.998); gradient fit through the real forward
  R²=1.000 (OLS 1.000); the deliberate count(type1)==count(type2) collinear
  case is correctly reported as non-identifiable (X·(e1−e2)=0, shifting along
  the null direction changes predictions by 0.0) while predictions remain
  recoverable.

## Purchase gate (official valid only, frozen; see config)

`STRC_DEV_GAIN_SUPPORTED` requires all of: seed-mean `S−P ≤ −0.004`; both
seeds negative; ordinary (`y > −3`) not worse; tail (`y ≤ −3`) strictly
better; seed-mean RMSE ≤ P RMSE + 0.005; top-10-removed Δ < 0; improvement
not driven by `valid:0172`; all correctness gates.  Baseline P: seed0
0.084785663, seed1 0.082313002, seed-mean 0.083549333.

Failure classes: `STRC_GAIN_WEAK_OR_MIXED` (mean improvement but a gate
fails) and `STRC_NO_PREDICTIVE_GAIN` (no robust improvement).  No follow-up
vocabulary/scale/tail-weighting/seed search is permitted either way.

## Discipline

Official train 10000 for training, official valid 1000 as the exposed
development set; all prediction files hashed before the single valid label
read; official test never read in any form; no ensemble; no calibration; no
target transform; no seed selection.
