| model | best-valid | best epoch | train@best | params | steps | GPU-h | status | SSL | Δ vs P3 s0 |
|---|---:|---:|---:|---:|---:|---:|---|---|---:|
| P3 s0 | 0.079105 | 272 | 0.011261 | 413,505 | 28,440 | 0.201 | completed (historical reference; not rerun) | frozen train-only masked SSL | — |
| R0 | 0.140010 | 329 | 0.033097 | 307,809 | 47,400 | 0.203 | completed | none | +0.060905 |
| R1 | 0.134158 | 324 | 0.023706 | 458,961 | 47,400 | 0.901 | completed | none | +0.055052 |
| T | 0.091375 | 426 | 0.009922 | 462,081 | 47,400 | 0.355 | completed | yes (P3 seed0, fine-tuned) | +0.012270 |
| P0 | 0.625814 | 70 | 0.608447 | 367,361 | 47,400 | 0.380 | completed | none | +0.546709 |
| P1 | 0.729133 | 223 | 0.697384 | 442,145 | 47,400 | 0.636 | completed | none | +0.650027 |
| B | 0.165418 | 423 | 0.012708 | 338,805 | 47,400 | 0.475 | completed | none | +0.086313 |

### Pre-declared follow-ups

| model | best-valid | best epoch | train@best | params | steps | GPU-h | status | SSL | Δ vs P3 s0 |
|---|---:|---:|---:|---:|---:|---:|---|---|---:|
| P0 corrected | 0.332778 | 203 | 0.178623 | 281,121 | 47,400 | 0.375 | completed | none | +0.253673 |
| P1 corrected | 0.321900 | 312 | 0.095714 | 355,905 | 47,400 | 0.634 | completed | none | +0.242794 |
| T wd=1e-4 | 0.092437 | 432 | 0.009151 | 462,081 | 47,400 | 0.357 | completed | yes (P3 seed0, fine-tuned) | +0.013332 |

Training GPU-hours this round: 4.317; timing/smoke ~0.03; total ~4.35 of a 10.0 A100-h cap. P3 reference was not rerun.
