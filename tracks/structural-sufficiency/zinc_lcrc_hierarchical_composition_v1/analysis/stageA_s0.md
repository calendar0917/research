# Stage A — Fit8k / Monitor, seed 0

Verdict: **STOP**

## Support groups (Fit8k vocabulary)

| group | n |
|---|---:|
| full | 1000 |
| ordinary | 926 |
| tail | 74 |
| any_unseen | 317 |
| unseen_ordinary | 271 |
| seen_ordinary | 655 |

## Best-checkpoint Monitor MAE

| slice | B | H | F | H−B [95% CI] | H−F [95% CI] |
|---|---:|---:|---:|---|---|
| full | 0.121258 | 0.120233 | 0.119856 | -0.001025 [-0.005601, +0.003401] | +0.000377 [-0.005164, +0.005500] |
| ordinary | 0.065893 | 0.064281 | 0.065112 | -0.001612 [-0.005997, +0.002276] | -0.000831 [-0.006411, +0.003539] |
| tail | 0.814065 | 0.820387 | 0.804900 | +0.006322 [-0.025105, +0.041709] | +0.015487 [-0.016363, +0.055692] |
| any_unseen | 0.186512 | 0.185517 | 0.185556 | -0.000995 [-0.014083, +0.011217] | -0.000039 [-0.016493, +0.014837] |
| unseen_ordinary | 0.121260 | 0.116529 | 0.121633 | -0.004731 [-0.018378, +0.006854] | -0.005104 [-0.022879, +0.008669] |
| seen_ordinary | 0.042985 | 0.042664 | 0.041727 | -0.000321 [-0.002842, +0.002146] | +0.000937 [-0.001052, +0.002948] |
| minus_fixed_worst10 | 0.070759 | 0.067434 | 0.069719 | -0.003326 [-0.007498, +0.000400] | -0.002285 [-0.007580, +0.001997] |
| minus_row172 | 0.121354 | 0.120334 | 0.119946 | -0.001020 [-0.005760, +0.003302] | +0.000388 [-0.005457, +0.005352] |

## Curves

| arm | best | best ep | W20 | T100 | final | GPU·h |
|---|---:|---:|---:|---:|---:|---:|
| B | 0.121258 | 486 | 0.121718 | 0.121746 | 0.121743 | 0.487 |
| H | 0.120233 | 626 | 0.120483 | 0.120516 | 0.120517 | 0.691 |
| F | 0.119856 | 581 | 0.120167 | 0.120194 | 0.120211 | 0.730 |

Penalty vs tail on Monitor (descriptive, never pooled):

| group | n | B | H | F | H−B |
|---|---:|---:|---:|---:|---|
| penalty/tail | 22 | 2.389286 | 2.433630 | 2.372507 | +0.044344 |
| penalty/ordinary | 14 | 0.164376 | 0.159841 | 0.140592 | -0.004535 |
| no_penalty/tail | 52 | 0.147625 | 0.137861 | 0.141682 | -0.009764 |
| no_penalty/ordinary | 912 | 0.064381 | 0.062814 | 0.063953 | -0.001567 |

Paired per-molecule wins (H better / ties / H worse):

| slice | H vs B | H vs F |
|---|---|---|
| full | 512/0/488 | 504/0/496 |
| ordinary | 472/0/454 | 464/0/462 |
| tail | 40/0/34 | 40/0/34 |
| any_unseen | 159/0/158 | 159/0/158 |
| unseen_ordinary | 140/0/131 | 136/0/135 |
| seen_ordinary | 332/0/323 | 328/0/327 |
| minus_fixed_worst10 | 510/0/480 | 502/0/488 |
| minus_row172 | 511/0/488 | 503/0/496 |

Largest |H−F| rows (negative favors H):

| row | y | pred H | pred F | d_HF | d_HB | group |
|---:|---:|---:|---:|---:|---:|---|
| 374 | 0.5753 | 0.4482 | -1.3359 | -1.7841 | -1.1781 | unseen_ord |
| 1760 | -21.9071 | -3.8719 | -4.8888 | +1.0170 | +0.6671 | penalty |
| 6367 | -2.9163 | -2.6593 | -2.2807 | -0.3785 | -0.2524 | penalty |
| 9469 | -1.7343 | -1.3694 | -1.0289 | -0.3404 | +0.2017 | unseen_ord |
| 8865 | -5.9611 | -6.4413 | -6.1042 | +0.3372 | -0.2976 | tail |
| 2210 | -42.0366 | -10.3968 | -10.7163 | +0.3196 | +0.5054 | penalty |
| 8395 | 0.2176 | 0.2370 | -0.1166 | -0.3149 | -0.0015 | unseen_ord |
| 4077 | -3.8043 | -3.7107 | -3.4056 | -0.3051 | -0.3567 | penalty |
| 4072 | -1.3279 | -0.7790 | -1.0701 | +0.2911 | +0.4809 | unseen_ord |
| 7255 | -2.5514 | -2.9465 | -3.2281 | -0.2816 | -0.1168 | unseen_ord |

One-row sensitivity (molecule 374, Monitor position 39; post-hoc, **not** a gate): full H−B without it +0.000154; full H−F without it +0.002163.

Smoothing deltas:

* H_minus_B: W20=-0.001235, T100=-0.001230, final=-0.001226, best=-0.001025
* H_minus_F: W20=+0.000316, T100=+0.000322, final=+0.000305, best=+0.000377

## Gates

* FAIL — full_H_minus_B
* FAIL — full_H_minus_F
* FAIL — ordinary_H_minus_B
* PASS — unseen_ordinary_H_minus_B
* PASS — minus_worst10_H_minus_B
* PASS — worst10_share
* FAIL — W20_H_minus_B
* FAIL — W20_H_minus_F
* PASS — T100_H_minus_B
* FAIL — T100_H_minus_F
* PASS — final_H_minus_B
* FAIL — final_H_minus_F

Fixed worst-10 share of net AE reduction: -2.213
