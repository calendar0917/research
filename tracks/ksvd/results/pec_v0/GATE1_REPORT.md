# PEC-v0 — Gate 1 (label-free dictionary + coarse-role recoverability)

Verdict: **FAIL**

Split: fit 8000 / monitor 2000 official-train molecules. Official valid and test not read.

| role | E_rec monitor (K-SVD) | E_rec fit | E_rec random | ratio | PCA16 | used atoms | max l0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| node | 0.0368659 | 0.0369128 | 0.330513 | 0.111541 | 5.49781e-31 | 12 | 4 |
| edge | 0.0542703 | 0.0543337 | 0.529933 | 0.10241 | 1.26481e-31 | 11 | 4 |

| probe | macro-F1 | majority macro-F1 | margin | accuracy |
|---|---:|---:|---:|---:|
| alpha^V -> shell | 1 | 0.217376 | 0.782624 | 1 |
| alpha^E -> shellpair | 1 | 0.179799 | 0.820201 | 1 |

Frozen criteria:

* `node_e_rec`: True
* `edge_e_rec`: True
* `node_random_ratio`: False
* `edge_random_ratio`: False
* `node_used`: True
* `edge_used`: False
* `node_exact_sparsity`: True
* `edge_exact_sparsity`: True
* `node_probe`: True
* `edge_probe`: True

`official_test_loaded = False`

## Reuse / non-collapse detail (durable copy of `gate1_label_free.json`)

| role | dict | E_rec monitor | E_rec fit | random ratio | PCA16 | used | argmax-used | dead | effective | top-1 mass | entropy (bits) | max l0 | row coverage |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| node | ksvd | 0.036866 | 0.036913 | 0.111541 | 5.5e-31 | 12 | 7 | 4 | 6.000 | 0.575 | 0.989 | 4 | 1.00 |
| node | random | 0.330513 | 0.330390 | — | — | 12 | 4 | 4 | 7.357 | 0.485 | 1.182 | 4 | 1.00 |
| edge | ksvd | 0.054270 | 0.054334 | 0.102410 | 1.26e-31 | 11 | 3 | 5 | 4.454 | 0.488 | 1.178 | 4 | 1.00 |
| edge | random | 0.529933 | 0.529747 | — | — | 14 | 5 | 2 | 5.357 | 0.385 | 1.280 | 4 | 1.00 |

## Frozen criteria (self-set, uncalibrated; reported verbatim)

```json
{
  "edge_e_rec": true,
  "edge_exact_sparsity": true,
  "edge_probe": true,
  "edge_random_ratio": false,
  "edge_used": false,
  "node_e_rec": true,
  "node_exact_sparsity": true,
  "node_probe": true,
  "node_random_ratio": false,
  "node_used": true
}
```

`official_test_loaded = False`; `official_valid_read = False`
