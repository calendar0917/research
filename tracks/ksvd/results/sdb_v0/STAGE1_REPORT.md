# SDB-v0 Stage 1 — label-free dictionary-domain gate

Verdict: **PASS**

## Structural reconstruction `E_phi`

| reference | train | dev |
|---|---:|---:|
| ksvd | 1.52622e-05 | 1.63678e-05 |
| rand | 0.6939 | 0.693799 |
| pca | 1.60065e-08 | 1.91002e-08 |

## Binding reconstruction `E_bind = ||C_phi - R C_code||^2 / ||C_phi||^2`

| reference | train agg | dev agg | dev per-mol median |
|---|---:|---:|---:|
| ksvd | 0.000288667 | 0.000345415 | 0.000150502 |
| ksvd_dense | 5.80178e-06 | 1.60177e-05 | 1.30484e-06 |
| rand | 1.73084 | 1.71345 | 1.73037 |
| pca | 3.53424e-07 | 5.30406e-07 | 1.15036e-07 |

*Sparse-vs-dense on the same learned `D`: `E_bind(sparse s=8) / E_bind(dense s=K)` = 21.5646. PCA reference ratio `E_bind_ksvd_dev / E_bind_pca_dev` = 651.228 (PCA is a near-lossless dense rank-32 affine projection; reported, not gated — Amendment A1).*

## Dictionary health (dev codes)

- used atoms: 32 / 32 (dead 0)
- row coverage: 1
- top-1 / top-8 coefficient mass: 0.803429 / 1
- support entropy (normalized): 0.210178
- exact sparsity: max l0 8 (within s=8: True)

## Dictionary semantic audit (interpretation only)

| atom | support freq | mean patch nodes | mean patch edges | mean cycle rank | root_basis | node_mean | node_std | edge_mean | edge_std |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 6 | 0.823636 | 9.12257 | 8.35978 | 2.67708 | 0.61219 | 0.432666 | 0.445922 | 0.603952 | 0.223172 |
| 15 | 0.673266 | 6.89329 | 6.82792 | 3.04348 | 0.497469 | 0.447398 | 0.406166 | 0.554299 | 0.214372 |
| 28 | 0.559893 | 7.64525 | 6.72378 | 2.38971 | 0.545897 | 0.41213 | 0.43283 | 0.610765 | 0.208173 |
| 14 | 0.512325 | 5.0583 | 4.0689 | 2.65863 | 0.484079 | 0.403344 | 0.429997 | 0.555191 | 0.181713 |
| 23 | 0.470996 | 8 | 7 | 2.59829 | 0.48144 | 0.390312 | 0.410707 | 0.630397 | 0.126527 |
| 30 | 0.461249 | 7 | 6 | 2.919 | 0.48144 | 0.397076 | 0.414187 | 0.591586 | 0.106811 |
| 4 | 0.458563 | 8.17969 | 7.31696 | 2.81481 | 0.581615 | 0.431075 | 0.427401 | 0.584969 | 0.176048 |
| 16 | 0.408482 | 4.21714 | 4.23192 | 3.12081 | 0.58342 | 0.539658 | 0.342185 | 0.524746 | 0.249589 |
| 11 | 0.320886 | 7.06498 | 7.28118 | 3.04237 | 0.561569 | 0.479087 | 0.412547 | 0.526521 | 0.266585 |
| 3 | 0.300784 | 5.90375 | 5.15706 | 2.30556 | 0.548425 | 0.440808 | 0.426699 | 0.524369 | 0.201574 |
| 26 | 0.252134 | 6.58391 | 5.58142 | 2.29333 | 0.620756 | 0.445409 | 0.452375 | 0.59453 | 0.203787 |
| 21 | 0.246242 | 10 | 9 | 3.33333 | 0.559898 | 0.405677 | 0.425393 | 0.592394 | 0.151521 |

*Top-12 dictionary atoms by support frequency; the semantic audit is report-only and never an input.*


`official_test_loaded = False`
