# Valid four-group diagnostic (old models only)

| group | n | model | MAE | sum AE | share | DBT-P3B | DBT-L |
|---|---:|---|---:|---:|---:|---:|---:|
| covered_ordinary | 710 | P3B_s0 | 0.047468 | 33.702 | 0.442 |  |  |
| covered_ordinary | 710 | L_s0 | 0.044864 | 31.853 | 0.406 |  |  |
| covered_ordinary | 710 | DBT_s0 | 0.042996 | 30.527 | 0.393 | -0.004472 | -0.001867 |
| covered_tail | 39 | P3B_s0 | 0.116284 | 4.535 | 0.059 |  |  |
| covered_tail | 39 | L_s0 | 0.225481 | 8.794 | 0.112 |  |  |
| covered_tail | 39 | DBT_s0 | 0.258636 | 10.087 | 0.130 | +0.142352 | +0.033155 |
| unseen_ordinary | 225 | P3B_s0 | 0.133940 | 30.136 | 0.395 |  |  |
| unseen_ordinary | 225 | L_s0 | 0.130773 | 29.424 | 0.375 |  |  |
| unseen_ordinary | 225 | DBT_s0 | 0.141320 | 31.797 | 0.409 | +0.007380 | +0.010547 |
| unseen_tail | 26 | P3B_s0 | 0.301922 | 7.850 | 0.103 |  |  |
| unseen_tail | 26 | L_s0 | 0.325623 | 8.466 | 0.108 |  |  |
| unseen_tail | 26 | DBT_s0 | 0.206050 | 5.357 | 0.069 | -0.095872 | -0.119572 |
