# PEC-I1 — Static Composition Interface Audit: report

Round `pec_i1` · study `zinc-context-gap`.  Pre-registration:
`notes/pec_i1_preregistration.md` (frozen before the run).  Official test
never loaded.

Stage A: **LOCAL_INFORMATION_SUBSTANTIALLY_PRESENT**

Internal screen: **INTERFACE_SIGNAL_STRONG**

## Internal screen (PEC-v0 Gate-2 split, 2000 train / 500 dev)

| delta | value |
|---|---:|
| `0.467108 - M_CD-I1` (frozen historical) | 0.042375 |
| `M_CD_matched - M_CD-I1` (device-matched) | 0.041287 |
| gate (`min`) | 0.041287 |

Mechanism (eval-only on the CD-I1 screen soup):

* relation-shuffle degradation: 0.103946
* BAG degradation: 0.463472
* mechanism alive: True

Full-data run authorized: True

## Full-data seed 0 (official train 10 000 / official valid 1 000)

* `CD-I1` Top-5 soup valid MAE: 0.151657
* `CD-I1` best valid MAE: 0.159730
* PEC-C1 `CD` comparator soup: 0.151767
* improvement vs PEC-C1 CD: 0.000110
* case C -> **STATIC_POOLING_NOT_PRIMARY_GAP**
* matches/beats S0 orientation: False

Full-scale mechanism (eval-only on the `CD-I1` soup):

* relation-shuffle degradation: 0.822537
* BAG degradation: 0.816163

FINAL: **STATIC_POOLING_NOT_PRIMARY_GAP**

