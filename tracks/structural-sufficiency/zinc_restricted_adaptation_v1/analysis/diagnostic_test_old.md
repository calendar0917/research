# Test four-group diagnostic (old models only)

| group | n | model | MAE | sum AE | share | DBT-P3B | DBT-L |
|---|---:|---|---:|---:|---:|---:|---:|
| covered_ordinary | 705 | P3B_s0 | 0.047252 | 33.313 | 0.477 |  |  |
| covered_ordinary | 705 | L_s0 | 0.043744 | 30.840 | 0.465 |  |  |
| covered_ordinary | 705 | DBT_s0 | 0.048998 | 34.543 | 0.498 | +0.001746 | +0.005254 |
| covered_tail | 51 | P3B_s0 | 0.128495 | 6.553 | 0.094 |  |  |
| covered_tail | 51 | L_s0 | 0.076441 | 3.898 | 0.059 |  |  |
| covered_tail | 51 | DBT_s0 | 0.086733 | 4.423 | 0.064 | -0.041762 | +0.010292 |
| unseen_ordinary | 218 | P3B_s0 | 0.114921 | 25.053 | 0.359 |  |  |
| unseen_ordinary | 218 | L_s0 | 0.114230 | 24.902 | 0.376 |  |  |
| unseen_ordinary | 218 | DBT_s0 | 0.118929 | 25.926 | 0.374 | +0.004007 | +0.004698 |
| unseen_tail | 26 | P3B_s0 | 0.190233 | 4.946 | 0.071 |  |  |
| unseen_tail | 26 | L_s0 | 0.254859 | 6.626 | 0.100 |  |  |
| unseen_tail | 26 | DBT_s0 | 0.173397 | 4.508 | 0.065 | -0.016836 | -0.081462 |
