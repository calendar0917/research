# SUMMARY — `zinc_component_supervision_fulltrain_confirmation_seed0_v1`

## Verdict

**CHEM_CONFIRMED = True · DEPLOY_CONFIRMED = True**

Trained on the **full 10,000 official-train rows** (no fresh fold), two matched arms and one
shared cycle head Q, then evaluated **once** on the frozen official-valid (1000 rows).

## One-line answer to the two registered questions

1. On the full-train valid, COMP's calibrated `g` gain over SUM is **+0.005354** (G0) and
   **+0.004563** (overall), CIs `[+0.000765, +0.010104]` and `[-0.000583, +0.009310]`. Both are
   positive and both raw gains remain positive — the ~0.01 internal-dev chemistry signal
   **survived** the full-train / valid confirmation, though it is **degraded** (roughly halved
   in magnitude vs. the internal `+0.00987`).
2. With both arms connected to the same train-only Q, the complete deployable `y` is **not**
   below 0.09 (SUM y_cal = 0.1227, COMP y_cal = 0.1174). The valid `y`-MAE gain is
   **+0.005247** (overall), CI `[+0.000499, +0.010275]`. The DEPLOY gate passes (overall gain
   ≥ 0.003, CI lower > 0, raw gain > 0, G0 y-worsening = -0.005340). But the absolute
   `y`-MAE **benchmark marker (0.09) is not met**.

## Gains (valid, frozen soup)

| endpoint | arm | raw MAE | cal MAE |
|---|---|---|---|
| g | SUM | 0.0486 | 0.0428 |
| g | COMP | 0.0422 | 0.0374 |
| y | SUM | 0.1227 | 0.1169 |
| y | COMP | 0.1174 | 0.1122 |

| gain (SUM − COMP) | point | 95% CI |
|---|---|---|
| g G0 cal | +0.005354 | [+0.000765, +0.010104] |
| g overall cal | +0.004563 | [-0.000583, +0.009310] |
| y overall cal | +0.005247 | [+0.000499, +0.010275] |

## Gates

| gate | value |
|---|---|
| CHEM_CONFIRMED | True |
| DEPLOY_CONFIRMED | True |
| benchmark y_cal < 0.09 | False (SUM 0.1169, COMP 0.1122) |

## Resource budget

0.443 GPU-hours used of 1.2 limit. Wall clock 759 s + 835 s (parallel) + 13 s (Q, local CPU).

## Evidence scope

Full-train valid, single seed. official-valid reused (used in prior rounds). official-test never
loaded, predicted or scored. See PROTOCOL §1.3 / §1.8 for boundaries.
