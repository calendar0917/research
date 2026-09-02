# Rooted-WL frozen terminal test audit

Historical disclosure: official test had already been viewed by older routes; this is a controlled frozen terminal evaluation.

## train_only

| candidate | five-seed mean AUC | seed-ensemble AUC |
|---|---:|---:|
| `t_a` | 0.764444 | 0.766266 |
| `f_centered` | 0.764398 | 0.766257 |
| `0.5*t_a+0.5*f_centered` | 0.778869 | 0.780378 |

## train_valid_refit

| candidate | five-seed mean AUC | seed-ensemble AUC |
|---|---:|---:|
| `t_a` | 0.761012 | 0.762781 |
| `f_centered` | 0.771633 | 0.773238 |
| `0.5*t_a+0.5*f_centered` | 0.782523 | 0.783939 |
