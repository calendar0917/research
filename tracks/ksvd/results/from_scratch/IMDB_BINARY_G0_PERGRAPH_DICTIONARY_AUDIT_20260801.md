# IMDB-BINARY G0 per-graph dictionary audit

> Frozen protocol: `KSVD_IMDB_PERGRAPH_DICTIONARY_PROTOCOL_20260801.md`

## 1. Method

Each raw IMDB graph independently factorizes its frozen 7-node WALK patch matrix. The primary descriptor is invariant to atom permutations; the legacy ordered D/X/Gram readout is secondary only.

```text
s=7, d=21, patches=min(n,24), K=8, T=2, updates=10
INIT=deterministic maximin, FINAL=same INIT + KSVD, PCA=rank 8
```

## 2. Label-free feature extraction audit

- graphs: `1000`;
- raw descriptor dimension: `58`;
- invariant factorization descriptor dimension: `90`;
- legacy descriptor dimension: `84`;
- mean reconstruction INIT / FINAL / PCA: `0.1370` / `0.0935` / `0.0663`;
- mean INIT→FINAL relative reduction: `0.1990`;
- FINAL-better graphs: `643/1000`;
- invariant atom-permutation max difference: `1.243e-14`;
- mean nnz INIT / FINAL: `1.2230` / `1.5716`.

## 3. Fold results

### stratified

| fold | STATS | STATS+RAW | +INIT | +FINAL | +PCA | FINAL−INIT | FINAL−PCA | aligned−shuffle |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.7400 | 0.7250 | 0.6850 | 0.7050 | 0.7150 | 0.0200 | -0.0100 | -0.0100 |
| 1 | 0.6750 | 0.6900 | 0.7200 | 0.7100 | 0.6850 | -0.0100 | 0.0250 | 0.0100 |
| 2 | 0.6700 | 0.6600 | 0.6350 | 0.5750 | 0.6700 | -0.0600 | -0.0950 | -0.1050 |
| 3 | 0.7000 | 0.6700 | 0.6750 | 0.6800 | 0.6600 | 0.0050 | 0.0200 | 0.0450 |
| 4 | 0.7250 | 0.7100 | 0.7500 | 0.7350 | 0.7150 | -0.0150 | 0.0200 | 0.0350 |

- primary update direction: `2/5`;
- mean primary update gain: `-0.0120`;
- mean beyond STATS+RAW: `-0.0100`;
- mean FINAL−PCA: `-0.0080`;
- reconstruction-positive folds: `5/5`;
- mean aligned−row-shuffle: `-0.0050`;
- mean legacy atom-order sensitivity: `0.0190`;
- mean label-shuffle BA: `0.4780`.

### exact_isomorphism_grouped

| fold | STATS | STATS+RAW | +INIT | +FINAL | +PCA | FINAL−INIT | FINAL−PCA | aligned−shuffle |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.5750 | 0.5400 | 0.7150 | 0.7150 | 0.4750 | 0.0000 | 0.2400 | 0.2250 |
| 1 | 0.6550 | 0.6700 | 0.6600 | 0.6650 | 0.6800 | 0.0050 | -0.0150 | -0.0100 |
| 2 | 0.6850 | 0.6450 | 0.6650 | 0.6500 | 0.6500 | -0.0150 | 0.0000 | 0.0300 |
| 3 | 0.6050 | 0.6250 | 0.5750 | 0.5650 | 0.6200 | -0.0100 | -0.0550 | -0.0500 |
| 4 | 0.7350 | 0.7550 | 0.7600 | 0.7400 | 0.7100 | -0.0200 | 0.0300 | -0.0050 |

- primary update direction: `1/5`;
- mean primary update gain: `-0.0080`;
- mean beyond STATS+RAW: `0.0200`;
- mean FINAL−PCA: `0.0400`;
- reconstruction-positive folds: `5/5`;
- mean aligned−row-shuffle: `0.0380`;
- mean legacy atom-order sensitivity: `0.0310`;
- mean label-shuffle BA: `0.5120`.

## 4. Registered decision

> **RECON_ONLY_PERGRAPH_DICTIONARY**

- [x] `invariant_permutation_difference_le_1e_8`
- [x] `reconstruction_positive_5_of_5`
- [ ] `primary_update_positive_at_least_4_of_5`
- [ ] `mean_primary_update_gain_ge_0_01`
- [ ] `aligned_not_below_row_shuffle`
- [x] `label_shuffle_in_0_45_0_55`

## 5. Interpretation boundary

A PASS would support per-graph sparse-factorization statistics as a graph descriptor. It would not establish a dataset-shared motif vocabulary or aligned atom identities. A reconstruction-only result means KSVD optimizes each graph's patch cloud but does not add stable task information beyond the same INIT and raw/statistical controls.
