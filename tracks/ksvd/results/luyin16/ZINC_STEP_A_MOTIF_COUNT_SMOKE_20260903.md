# ZINC Step A：collision-free typed motif count

Protocol: `luyin16-zinc-step-a-motif-count-smoke-v1`

## Representation

- radius: `3`; centres: `every atom`
- typed rooted-WL rounds: `0/1/2/3`
- vocabulary: `train-fold-only top-K exact nested tuple tokens; no proposed-feature hash bins`; K=`64` + OOV
- readout: `raw count + count / number of centres`
- Morgan reference: `RDKit-free count-ECFP/Morgan analogue`, bits=`128`

## XGBoost results

| view | dim | train CV MAE | valid MAE | valid ensemble | test MAE after train+valid | test ensemble |
|---|---:|---:|---:|---:|---:|---:|
| `s` | 62 | 1.021398 | 1.134209 ± 0.024938 | 1.134209 | 1.454355 ± 0.008246 | 1.454355 |
| `wl_count` | 520 | 1.015346 | 1.149099 ± 0.000189 | 1.143524 | 1.513876 ± 0.000507 | 1.513876 |
| `s_wl_count` | 582 | 0.995530 | 1.030822 ± 0.001939 | 1.015617 | 1.433815 ± 0.004194 | 1.433815 |
| `morgan_count` | 256 | 1.004219 | 1.132903 ± 0.031289 | 1.114486 | 1.444272 ± 0.065015 | 1.441178 |
| `s_morgan_count` | 318 | 0.972679 | 1.104680 ± 0.028746 | 1.096999 | 1.468529 ± 0.040172 | 1.467186 |

## Matched GINE

- valid MAE: `1.734689 ± 0.000000`; ensemble `1.734689`
- test after train+valid refit MAE: `1.562898 ± 0.000000`; ensemble `1.562898`

Selected XGBoost view by train-only CV: `s_morgan_count`.

Runtime: `2.5s`; token cache hit: `True`.
