# DECISION — zinc_full_reader_wd_pilot_v1

**FAIL — stop. Do not buy a second variant of this configuration.**

Concurrent paired warm-start pilot (H=20, Adam coupled L2, last-5 soup, seed 0), same parent
Full soup `17f5fcc3…574eb`:

| arm | reader wd | cal valid MAE | raw valid |
|---|---:|---:|---:|
| control | 1e-5 | 0.115023 | 0.115010 |
| candidate | 1e-4 (10x) | 0.115041 | 0.115000 |

Gate (all three required): total calibrated gain `-1.84e-5` (<0.003) **FAIL**;
gain_without_id172 `+1.82e-5` (positive, ~0); G0 contribution worsening `-6.63e-5` (improved).

Intervention executed (6 reader tensors / 33,385 params, reader group wd 1e-4, 1580 steps/arm) but
weak: `|data_grad| / |wd*p|` is `4.2e3–9.7e5` at 1x, still `4e2–1e5` at 10x; reader soup norms moved
0.3–6% from parent.

Four-forward correction: with a common topology input `T0`, the train:3776 / valid:0172 prediction gap
is still `D_ref = 17.90` (native `21.13`), so the gap is driven by the other R blocks and their
interaction with T, not by T alone. The prior round's strong attribution is corrected; no causal claim.

Scope of closure: this parent checkpoint, this 10x reader decay, this H=20 warm horizon.
It does NOT close regularization in general, from-scratch training, or other reader changes.
No official test access. Not promoted/committed.