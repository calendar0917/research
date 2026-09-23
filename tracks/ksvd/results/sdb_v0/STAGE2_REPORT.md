# SDB-v0 Stage 2 — FSAR mechanism-preservation gate

Verdict: **PASS**

mean recovery (vs frozen phi65 oracle): 1.15928

mean assignment-shuffle MAE degradation: 0.245369

dict within dense slack: True; alive: True

phi65 protocol drift (max): 8.16584e-09

| seed | M0 | phi65 | dict32 | dict_dense32 | pca32 | dense32 | dense32_frozen | rand32 | recovery | shuffle_deg |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.55379 | 0.494538 | 0.486293 | 0.491329 | 0.48848 | 0.495732 | 0.501665 | 0.497199 | 1.13915 | 0.236825 |
| 1 | 0.53557 | 0.481283 | 0.47151 | 0.477209 | 0.475874 | 0.481869 | 0.483369 | 0.481945 | 1.18003 | 0.239532 |
| 2 | 0.546893 | 0.490486 | 0.481535 | 0.487915 | 0.484218 | 0.489971 | 0.492226 | 0.490163 | 1.15868 | 0.25975 |

*All arms are a frozen `M0` soup plus a single linear assignment residual. `dict32` total params = 65x32 (frozen D) + 32x28 (readout); `dense32` = 65x32 (trained W) + 32x28.*
