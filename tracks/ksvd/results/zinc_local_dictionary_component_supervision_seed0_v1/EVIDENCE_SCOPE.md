# EVIDENCE_SCOPE — zinc_local_dictionary_component_supervision_seed0_v1

## Supports

1. **In this fixed interface** (same 125-D local tuple input, same real J
   incidence aggregation, same M_g skeleton with matched posterior MLP bridge,
   same 39→2 ComponentReader and COMP loss, matched 8,000+21,888 local
   parameters, identical seed-0 recipe and identical position/gid streams), the
   shared tied-IHT local tuple dictionary (10 steps, hard top-8) does **not**
   beat the matched SiLU local MLP: dev G0 cal −0.002045 CI [−0.005525,
   +0.001502]; raw deficits −0.008939/−0.007946 with CIs entirely below zero.
   Fit-side agrees (D 0.031079 vs M 0.027889 fit cal). Direction survives the
   drop-M-worst-row sensitivity.
2. The dictionary mechanism was live (8-sparse codes, 61/64 atoms, drift 71.6,
   live task gradients from epoch 40, injection RMS 0.415, zero-injection
   collapse 0.090→0.494) — the negative is a performance result, not a
   mechanism failure.
3. The D−M gap is concentrated in the s (residual chemistry) component
   (−0.006832 dev raw; G0 −0.007253), slightly offset by ell (+0.000811); it is
   not a bias or cancellation artifact (comparable triangle gaps 0.032/0.034,
   opposite-sign 48.9%/51.1%; calibration moves both arms the same direction).

## Does NOT support

- Any claim that component supervision revived or failed the dictionary family
  (needs the g-supervised pair under the identical new fold — not run).
- All-dictionary vs all-MLP statements beyond this exact local encoder
  interface, seed 0, and the rng(20261006) fold.
- J vs marginal incidence aggregation (both arms use J; the J→I switch at the
  converged state is nearly inert in both — a frozen-model sensitivity
  statement only, not a training-time inductive-bias result).
- Any official-valid/test y-MAE (internal fold chemistry-component g only);
  any cross-fold or cross-regime gain splicing (the internal dev 0.088 is NOT
  comparable to historical valid/test y-MAE — different split, target and
  no cycle channel).
- Structure-attribute crossing / shared-structure dictionary (luyin19's own
  proposal) — untested here.
- Causal attribution of the raw→cal gap shrink (part of the raw deficit is the
  larger fit-side median offset, not pure signal quality).

## Frozen reuse (read-only, hash-verified)

Raw tuple incidence structure and env phi/atom caches (label-free, per-graph),
the M_g skeleton/bridge/reader construction, the train-only loader and audit
label path — all from the verified production modules; every fitted statistic
(body standardizers, phi scaler + phi_scale, kappa_D, kappa_M, target
constants) refit on the new rng(20261006) 8000 fit rows. No trained soup of any
kind entered as weights or outputs.
