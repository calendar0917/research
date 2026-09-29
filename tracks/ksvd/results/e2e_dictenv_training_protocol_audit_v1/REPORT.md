# REPORT — `e2e_dictenv_training_protocol_audit_v1`

CPU only; the official ZINC test split was never loaded. One architecture
(frozen CSSD-q1, 97727 parameters), one seed (0), one shared 280-epoch prefix
and one exact fork that changes only the learning rate of epochs 281–320.

## 0. Setup

- pre-registration: `notes/e2e_dictenv_training_protocol_audit_v1_preregistration.md`
- shared prefix: epochs 1–280, `Adam lr = 0.001`, wd 1e-05, batch 128, clip 5
- CONTROL continuation: epochs 281–320 at `lr = 0.001`
- LOW-LR continuation: epochs 281–320 at `lr = 0.0001`
- prefix epoch-280 valid MAE `0.139555` (frozen historical `0.141159`, tolerance +0.02) → `PREFIX_HEALTHY`
- fork integrity `FORK_INTEGRITY_OK`: prediction max |Δ| `0.0`, optimizer exp_avg/exp_avg_sq/step identical `True/True/True`, only lr differs `True`, batch order equal over 40 epochs `True`

## 1. Primary result (frozen Top-5 soup over epochs 1–320)

| arm | best valid | best epoch | Top-5 soup 1–320 | members | tail soup 281–320 | last-10 mean | epoch 320 | train MAE 320 |
|---|---|---|---|---|---|---|---|---|
| CONTROL `lr=0.001` | 0.133430 | 311 | 0.127428 | [258, 277, 297, 311, 313] | 0.128055 | 0.146977 | 0.148608 | 0.099271 |
| LOW-LR `lr=0.0001` | 0.127857 | 309 | 0.126818 | [294, 300, 309, 310, 313] | 0.126818 | 0.128832 | 0.128793 | 0.077021 |

```text
G_schedule = M_control - M_low_lr = +0.000610   (+1.0x the ~0.0006 CPU soup floor)
```

## 2. Dictionary health

| checkpoint | active | N_eff | top-5 | max rate | recon (mean row sq) |
|---|---|---|---|---|---|
| epoch 280 | 32 | 20.76 | 3.197 | 0.928 | 0.00272 |
| CONTROL 320 | 32 | 20.28 | 3.201 | 0.847 | 0.00249 |
| LOW-LR 320 | 32 | 20.28 | 3.157 | 0.928 | 0.00330 |

## 3. Answers to the frozen questions

### Q1 — Which late phase is better after the same 280-epoch prefix?

LOW-LR (`lr=1e-4`) is better by `+0.000610` soup MAE (+1.0x the measured floor).

### Q2 — Is the low-LR improvement above the ~6e-4 CPU soup floor?

`G_schedule / floor = +1.02`; frozen case A threshold is `G_schedule < 0.002`.

### Q3 — Is the historical `0.130028 -> 0.128723` warm improvement reproduced inside the formal schedule?

- LOW-LR soup `0.126818` vs historical CSSD soup `0.130028` (Δ `-0.003210`)
- LOW-LR soup vs v2 warm soup `0.128723` (Δ `-0.001905`)
- CONTROL soup vs historical CSSD soup `-0.002600`

### Q4 — Is fixed `lr=1e-3` a material contributor to the plateau?

Frozen verdict: **TRAINING_PROTOCOL_NOT_PRIMARY_BOTTLENECK** (case A `True`, directional `False`, case B `False`, case C `False`, case D `False`).

### Q5 — Minor polish, material factor, or major factor?

minor polish (not the plateau's main cause)

### Q6 — Should the canonical training protocol be updated?

`canonical_protocol = 320@1e-3` (adopt low-LR tail: `False`).

### Q7 — Is the 320-epoch horizon still possibly binding?

`horizon_may_still_be_binding = False` (LOW-LR best epoch 309, late slope 311–320 `+7.660e-05` per epoch). Reported only; no longer run was executed.

## 4. Budget

- epochs: prefix 280 + control 40 + low-LR 40 = 360 epoch-equivalents
- CPU wall clock 43.4 min
- no seed 1, no LR sweep, no extra epochs, no scheduler, no official test

