# cscl-distance-pooling-bottleneck-v1 — Hard5 vs a frozen continuous-distance (Soft5) moment basis

Protocol: `configs/distance_pooling_bottleneck_v1.yaml` (frozen before the formal
runs). Track: `tracks/structure-contribution` (CSCL). Driver:
`code/run_distance_pooling_v1.py`. Operator: `code/distance_pooling_v1.py`.
Tests: `tests/test_distance_pooling_v1.py` (24 fast tests).

Status: **CPU audit complete — CPU_PURCHASE_GRANTED**. GPU arms and the frozen
official-validation evaluation are the only authorised follow-up (two fresh
seed-0 arms: Hard5, Soft5).

## 1. Question

Full-Y (the frozen historical strong non-GNN skeleton, 408,651 parameters)
summarises its 48-D pair state `h_ij` through **five hard shortest-path distance
buckets** (d = 1, 2, 3, 4, ≥5), each contributing `sum(h)`, `sum(h²)` and
`log1p(count)` to the 814-D reader input. Does replacing only that graph-level
hard grouping with one **pre-frozen continuous-distance basis** (Soft5) change
official-validation raw-y MAE, with everything else held fixed?

Authorised scope: CPU audit first; exactly two new seed-0 GPU arms if the
pre-frozen purchase gate passes; one frozen official-validation evaluation;
stop. No seed 1, no third arm, no kernel search, no test access.

## 2. Phase A — real code path (CPU, label-blind)

All checks in `results/distance_pooling_bottleneck_v1/source_path_audit.json`
pass (static source checks + live module inspection on the frozen checkpoint).

| item | finding | evidence |
|---|---|---|
| distance | unweighted BFS shortest path | `zinc_patch_path_pooling._shortest_path_summary` (line 387) |
| bucket | `bucket = min(max(int(distance),1),5) - 1` → 5+ truncation | `zinc_patch_path_pooling.py:620`, `DISTANCE_BUCKETS = 5` (line 112) |
| relation (raw 23-D) | `[5 one-hot \| log1p(d) col 5 \| 5 overlap \| 3 boundary \| 4 path-bond mean \| log path-count col 18 \| 4 adjacent bond]` | `_pair_relation` line 612, `np.log1p(float(distance))` line 656 |
| relation (consumed 15-D) | `P1_RELATION_INDICES = 0..13 + 18`; keeps **log1p(distance)**, keeps path count (C6 zeroes it) | `e2e_dictenv_p1.py:93` |
| distance_gate | `1 + tanh(Embedding(5, 48)[bucket])` — hard bucket only | `zinc_patch_path_pooling.py:2286,2778`; live weight shape `[5,48]` |
| pair_value | 192-D input `[left+right, \|left-right\|, product*gate, relation]` → 48-D | live `pair_encoder_in=192, out=48`; `pair_projection` 144→48 |
| pooling | per bucket `sum(h)`, `sum(h²)`, `log1p(count)`; C6_MASK zeroes the pair count block | `pool_pair_moments_masked` (clarity audit line 243) |
| reader | `[unary 289 \| pair moments 485 \| global 32 \| topology 8]` = **814** | live `reader_in=814`, params **408,651** |
| C6_MASK | zeroes unary count, pair count, relation path-count; topology **not** zeroed | `e2e_dictenv_clean_mechanism_v1.py:63` + live tuple check |

Frozen pair-state descriptor audit (`distance_descriptor_audit.json`, historical
soup checkpoint, 10,000 label-blind forward passes):

* the masked relation input still carries `log1p(distance)` (max abs gap < 1e-6)
  and the path-count column is exactly 0;
* the `distance_gate` output is exactly constant inside each hard bucket;
* the pair encoder output **does** vary with exact distance, including inside the
  last bucket (cosine distance of mean `pair_value` vs d=8: d=5 0.0236, d=6
  0.0103, d=7 0.0026, d=9 0.0014, d=10 0.0055, d=11 0.0100, d=12 0.0191).

This establishes that (a) the continuous distance coordinate is already
consumed by the relation encoder, so Soft5 introduces **no new information**,
and (b) the hard bucket discards the exact-distance organisation of the pooled
summary that the reader sees. Neither fact is predictive evidence.

## 3. Phase B — train-10k distance distribution (CPU, label-blind)

`distance_distribution.json`, 10,000 molecules / 2,668,346 pairs. Full all-pairs
BFS cross-check against the raw PyG graphs: **0 mismatches** on 2,668,346 pairs,
max |log1p round-trip| 8.6e-8 (float32).

| bucket | 1 | 2 | 3 | 4 | ≥5 |
|---|---|---|---|---|---|
| pairs | 249,279 | 344,139 | 340,991 | 309,126 | **1,424,811** |
| share | 9.34% | 12.90% | 12.78% | 11.58% | **53.40%** |

Exact-distance pair counts: d5 282,044 · d6 251,940 · d7 219,621 · d8 185,895 ·
d9 151,067 · d10 116,696 · d11 84,871 · d12+ 132,677. Diameter: mean 12.46,
p50 12, p90 16, max 22.

**The ≥5 bucket is not a single distance.** Per molecule (9,998 have bucket-4
pairs):

* mean 8.46 distinct exact distances inside the bucket (p50 8, p90 12, max 18);
* 99.80% of molecules mix ≥2 exact distances, 97.49% span ≥3;
* **99.25% of molecules hold both d=5 and d≥7 pairs**;
* mean distance 7.46, variance 4.43, conditional entropy 1.91 nats (p50 1.96);
* even the smallest molecules (≤20 atoms, n=2,793) mix distances 99.28% of the
  time and span ≥3 in 91.08%.

Correlations (descriptive): distinct distances vs atoms 0.804; entropy vs atoms
0.769; bucket-4 pair fraction vs atoms 0.882; bucket-4 fraction vs ring rank
0.543; mean distance vs atoms 0.765.

Conclusion: the last bucket mixes many real distances inside essentially every
molecule, not just "50% of pairs". A single pooled `sum(h)` per hard bucket is a
genuinely lossy description of the pair population.

## 4. Phase C — the one alternative operator (Soft5, frozen)

`code/distance_pooling_v1.py`:

```
s(d) = log1p(d)                                  # pair_relation[:, 5]
mu_k = log1p(centers[k]), centers = [1, 2, 3, 4, 8]
a_k  = exp(-(s - mu_k)^2 / (2 * 0.35^2))         # sigma fixed, never tuned
w_k  = a_k / sum_j a_j                           # non-negative, row sums 1

first_k  = sum_ij w_k(d_ij) * h_ij
second_k = sum_ij w_k(d_ij) * h_ij^2             # square h, then weight
count_k  = sum_ij w_k(d_ij)                      # stored log1p, C6-zeroed
```

Explicit non-variants: the second moment is **not** `(sum w h)²` and **not**
`sum (w h)²`; there is no kernel/center/width/parameter search. Everything else
is byte-for-byte the frozen Full-Y: K-SVD/CSSD coding, environment encoder,
15-D relation input, hard-bucket `distance_gate` (**not** swapped), pair
encoder, global/topology encoders, reader, raw-y L1 + label-free reconstruction,
C6_MASK. Output 5 × 97 = 485, reader 814, parameters 408,651, no new trainable
parameters, no new RNG consumption (`pair_pooling` is a plain attribute).

## 5. Phase D — correctness gates and purchase gate (CPU)

`correctness_gates.json` — all pass:

* Hard5 pooled features and prediction are **bit-identical** to the frozen
  `AuditModel.forward` path (max |Δ| = 0.0, readout and prediction);
* Soft5 weights: finite, non-negative, row sums 1 (max dev 1.19e-7);
* conservation vs a float64 reference: first 3.7e-8, second 3.5e-8, count
  2.0e-8 relative;
* endpoint-swap, pair-permutation, batch-composition and single-molecule
  invariants hold; the C6 pair-count block is exactly zero;
* 408,651 parameters for both arms; both build from the identical seed-0 init
  state (state-dict hashes equal, same as the parent builder);
* gradients reach `pair_encoder`, `relation_encoder`, `distance_gate`;
* forward output is unchanged when `batch.y` is randomised (y-independence).

`hard5_soft5_feature_audit.json` — on identical frozen pair states over all
10,000 train molecules: mean relative feature difference 0.459 (p50 0.458,
p90 0.538, max 0.695); **100%** of molecules differ by >1%. Loss of effective
rank inside the 480-D moment block: participation ratio 10.60 (Hard5) → 8.06
(Soft5); flattened correlation 0.888.

`cpu_purchase_gate.json`:

| gate | measured | threshold | pass |
|---|---|---|---|
| G1 long-distance heterogeneity (d=5 and d≥7 in bucket 4) | 0.9925 | ≥ 0.10 | ✅ |
| G2 Hard5/Soft5 feature difference >1% | 1.0000 | ≥ 0.25 | ✅ |
| G3 correctness gates | true | true | ✅ |

**CPU_PURCHASE_GRANTED.** These gates only show the operator is not a
near-identity reparameterisation acting on molecules with real long-distance
heterogeneity; they are not predictive evidence.

## 6. Phase E/F — frozen plan (executed only after purchase)

* two fresh matched seed-0 GPU arms on `res` pool `res-gpu1`: `Hard5` (frozen
  path) and `Soft5`, both from the identical untrained init, all official train
  10,000 rows, `all_train_prep.npz`, Adam lr 1e-3 / wd 1e-5 / clip 5 / batch
  128 / FP32, 240 epochs = 18,960 steps, raw-y L1 + label-free reconstruction,
  λ = 33.95873017865987, C6_MASK, shuffle seed 101, last-5 soup (epochs
  236–240);
* the historical full-10k Hard5 checkpoint (valid raw MAE 0.118618) is a
  read-only sanity reference only;
* GPU reduction-order nondeterminism is recorded; single-seed results are
  exploratory;
* after freezing both checkpoints: official-validation predictions written and
  hashed **without reading y**, then y read once, metrics computed, paired
  per-molecule bootstrap Δ = MAE(Soft5) − MAE(Hard5) (5000 resamples, seed 0),
  pre-frozen subgroups and the pre-frozen decision bands
  (A: Δ ≤ −0.010 / B: −0.010 < Δ ≤ −0.003 / C: Δ > −0.003; CI conflict →
  UNCERTAIN).

## 7. Evidence boundary

Phases A–D are label-blind. They establish a real representational loss in the
hard bucket and a non-trivial, exactly specified alternative; they say nothing
about prediction quality. The frozen Full-Y `distance_gate` remains hard-bucket;
Soft5 changes only the graph-level pooling of the pair states. Official
validation was already exposed in earlier rounds, so the upcoming evaluation is
a controlled comparison on an exposed instrument, not a fresh blind test.

<!-- RESULTS_SECTION -->

## 8. Final report — the 15 questions

<!-- FINAL_REPORT -->
