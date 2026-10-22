# Fixed four-group error tables

Signature grouping uses the corrected radius-2 content keys relative to train (the same keys as the audit's `r2_unseen_atom_frac`). The ordinary/tail label split is analysis-only and never enters the model.

### valid

| group | n | model | MAE | sum AE | share of total AE | ΔMAE vs P3B | ΔMAE vs L | ΔMAE vs DBT |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| covered_ordinary | 710 | P3B_s0 | 0.047468 | 33.702 | 0.442 |  |  |  |
| covered_ordinary | 710 | L_s0 | 0.044864 | 31.853 | 0.406 |  |  |  |
| covered_ordinary | 710 | DBT_s0 | 0.042996 | 30.527 | 0.393 | -0.004472 | -0.001867 |  |
| covered_ordinary | 710 | PA_s0 | 0.078503 | 55.737 | 0.397 | +0.031035 | +0.033639 | +0.035507 |
| covered_tail | 39 | P3B_s0 | 0.116284 | 4.535 | 0.059 |  |  |  |
| covered_tail | 39 | L_s0 | 0.225481 | 8.794 | 0.112 |  |  |  |
| covered_tail | 39 | DBT_s0 | 0.258636 | 10.087 | 0.130 | +0.142352 | +0.033155 |  |
| covered_tail | 39 | PA_s0 | 0.784161 | 30.582 | 0.218 | +0.667877 | +0.558680 | +0.525525 |
| unseen_ordinary | 225 | P3B_s0 | 0.133940 | 30.136 | 0.395 |  |  |  |
| unseen_ordinary | 225 | L_s0 | 0.130773 | 29.424 | 0.375 |  |  |  |
| unseen_ordinary | 225 | DBT_s0 | 0.141320 | 31.797 | 0.409 | +0.007380 | +0.010547 |  |
| unseen_ordinary | 225 | PA_s0 | 0.182115 | 40.976 | 0.292 | +0.048175 | +0.051341 | +0.040795 |
| unseen_tail | 26 | P3B_s0 | 0.301922 | 7.850 | 0.103 |  |  |  |
| unseen_tail | 26 | L_s0 | 0.325623 | 8.466 | 0.108 |  |  |  |
| unseen_tail | 26 | DBT_s0 | 0.206050 | 5.357 | 0.069 | -0.095872 | -0.119572 |  |
| unseen_tail | 26 | PA_s0 | 0.497270 | 12.929 | 0.092 | +0.195347 | +0.171647 | +0.291219 |
| all | 1000 | P3B_s0 | 0.076224 | 76.224 | 1.000 |  |  |  |
| all | 1000 | L_s0 | 0.078537 | 78.537 | 1.000 |  |  |  |
| all | 1000 | DBT_s0 | 0.077768 | 77.768 | 1.000 | +0.001545 | -0.000769 |  |
| all | 1000 | PA_s0 | 0.140224 | 140.224 | 1.000 | +0.064000 | +0.061687 | +0.062456 |

Sensitivity (descriptive only; row 172 stays in every headline):

| model | MAE without row 172 | Δ vs full |
|---|---:|---:|
| P3B_s0 | 0.074735 | -0.001488 |
| L_s0 | 0.072510 | -0.006027 |
| DBT_s0 | 0.070713 | -0.007056 |
| PA_s0 | 0.120082 | -0.020142 |

Row 172: y=-20.340498, unseen=0, tail=1, P3B_s0 AE=1.563011, L_s0 AE=6.099347, DBT_s0 AE=7.126242, PA_s0 AE=20.262345.

### test

| group | n | model | MAE | sum AE | share of total AE | ΔMAE vs P3B | ΔMAE vs L | ΔMAE vs DBT |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| covered_ordinary | 705 | P3B_s0 | 0.047252 | 33.313 | 0.477 |  |  |  |
| covered_ordinary | 705 | L_s0 | 0.043744 | 30.840 | 0.465 |  |  |  |
| covered_ordinary | 705 | DBT_s0 | 0.048998 | 34.543 | 0.498 | +0.001746 | +0.005254 |  |
| covered_ordinary | 705 | PA_s0 | 0.081736 | 57.624 | 0.535 | +0.034484 | +0.037992 | +0.032739 |
| covered_tail | 51 | P3B_s0 | 0.128495 | 6.553 | 0.094 |  |  |  |
| covered_tail | 51 | L_s0 | 0.076441 | 3.898 | 0.059 |  |  |  |
| covered_tail | 51 | DBT_s0 | 0.086733 | 4.423 | 0.064 | -0.041762 | +0.010292 |  |
| covered_tail | 51 | PA_s0 | 0.175218 | 8.936 | 0.083 | +0.046723 | +0.098777 | +0.088485 |
| unseen_ordinary | 218 | P3B_s0 | 0.114921 | 25.053 | 0.359 |  |  |  |
| unseen_ordinary | 218 | L_s0 | 0.114230 | 24.902 | 0.376 |  |  |  |
| unseen_ordinary | 218 | DBT_s0 | 0.118929 | 25.926 | 0.374 | +0.004007 | +0.004698 |  |
| unseen_ordinary | 218 | PA_s0 | 0.154758 | 33.737 | 0.313 | +0.039837 | +0.040528 | +0.035829 |
| unseen_tail | 26 | P3B_s0 | 0.190233 | 4.946 | 0.071 |  |  |  |
| unseen_tail | 26 | L_s0 | 0.254859 | 6.626 | 0.100 |  |  |  |
| unseen_tail | 26 | DBT_s0 | 0.173397 | 4.508 | 0.065 | -0.016836 | -0.081462 |  |
| unseen_tail | 26 | PA_s0 | 0.283117 | 7.361 | 0.068 | +0.092884 | +0.028258 | +0.109720 |
| all | 1000 | P3B_s0 | 0.069865 | 69.865 | 1.000 |  |  |  |
| all | 1000 | L_s0 | 0.066267 | 66.267 | 1.000 |  |  |  |
| all | 1000 | DBT_s0 | 0.069401 | 69.401 | 1.000 | -0.000463 | +0.003135 |  |
| all | 1000 | PA_s0 | 0.107659 | 107.659 | 1.000 | +0.037794 | +0.041392 | +0.038257 |

Sensitivity (descriptive only; row 172 stays in every headline):

| model | MAE without row 172 | Δ vs full |
|---|---:|---:|
| P3B_s0 | 0.069880 | +0.000015 |
| L_s0 | 0.066307 | +0.000040 |
| DBT_s0 | 0.069426 | +0.000024 |
| PA_s0 | 0.107635 | -0.000024 |

Row 172: y=-0.497084, unseen=0, tail=0, P3B_s0 AE=0.055188, L_s0 AE=0.026279, DBT_s0 AE=0.045148, PA_s0 AE=0.131497.

