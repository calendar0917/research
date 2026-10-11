# cscl-distance-pooling-bottleneck-v1 — Hard5 vs a frozen continuous-distance (Soft5) moment basis

Protocol: `configs/distance_pooling_bottleneck_v1.yaml` (frozen before the formal
runs). Track: `tracks/structure-contribution` (CSCL). Driver:
`code/run_distance_pooling_v1.py`. Operator: `code/distance_pooling_v1.py`.
Tests: `tests/test_distance_pooling_v1.py` (24 fast tests).

Status: **COMPLETE**. CPU audit → `CPU_PURCHASE_GRANTED`; two matched seed-0
GPU arms (Hard5 0.118474 / Soft5 0.119218 on official validation) →
**`SOFT5_NO_USEFUL_GAIN`** (Δ = +0.000743, CI95 [−0.006523, +0.008980]). The
round stops here: no seed 1, no third arm, no kernel/center/width search, no
retuning, official test untouched.

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

Frozen pair-state descriptor audit (`distance_descriptor_audit.json`, the
historical full-10k soup checkpoint with SHA256 `7792c560…` = the frozen
0.118618 reference, 10,000 label-blind forward passes):

* the masked relation input still carries `log1p(distance)` (max abs gap < 1e-6)
  and the path-count column is exactly 0;
* the `distance_gate` output is exactly constant inside each hard bucket;
* the pair encoder output **does** vary with exact distance, including inside the
  last bucket (cosine distance of mean `pair_value` vs d=8: d=5 0.0052, d=6
  0.0027, d=7 0.0009, d=9 0.0016, d=10 0.0071, d=11 0.0163, d=12 0.0477).

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
* conservation vs a float64 reference: first 3.7e-8, second 3.9e-8, count
  2.0e-8 relative;
* endpoint-swap, pair-permutation, batch-composition and single-molecule
  invariants hold; the C6 pair-count block is exactly zero;
* 408,651 parameters for both arms; both build from the identical seed-0 init
  state (state-dict hashes equal, same as the parent builder);
* gradients reach `pair_encoder`, `relation_encoder`, `distance_gate`;
* forward output is unchanged when `batch.y` is randomised (y-independence).

`hard5_soft5_feature_audit.json` — on identical frozen pair states over all
10,000 train molecules: mean relative feature difference 0.495 (p50 0.496,
p90 0.573, max 0.702); **100%** of molecules differ by >1%. Loss of effective
rank inside the 480-D moment block: participation ratio 5.21 (Hard5) → 4.16
(Soft5); flattened correlation 0.872.

`cpu_purchase_gate.json`:

| gate | measured | threshold | pass |
|---|---|---|---|
| G1 long-distance heterogeneity (d=5 and d≥7 in bucket 4) | 0.9925 | ≥ 0.10 | ✅ |
| G2 Hard5/Soft5 feature difference >1% | 1.0000 | ≥ 0.25 | ✅ |
| G3 correctness gates | true | true | ✅ |

**CPU_PURCHASE_GRANTED.** These gates only show the operator is not a
near-identity reparameterisation acting on molecules with real long-distance
heterogeneity; they are not predictive evidence.

## 6. Phase E — matched GPU arms (seed 0, pool res-gpu1)

Two fresh arms, same commit `3a09c573`, sequentially on the same GPU (a100-2,
CUDA_VISIBLE_DEVICES=1, torch 2.5.1+cu124, driver 550.163.01):

| | Hard5 | Soft5 |
|---|---|---|
| run_id | `distance-pooling-hard5-20261011-222946-a3305c09` | `distance-pooling-soft5-20261011-230556-b65adb15` |
| init-state SHA256 | `93c2f23c39b1654d…` | `93c2f23c39b1654d…` (identical) |
| training-program SHA256 | `d5d08f36086409b2…` | `d5d08f36086409b2…` (identical) |
| RNG state at epochs 1/40/120/240 | identical | identical |
| steps / epochs | 18,960 / 240 | 18,960 / 240 |
| soup members | 236–240 | 236–240 |
| parameters | 408,651 | 408,651 |
| wall (240 epochs) | 1,683 s (7.0 s/epoch) | 1,607 s (6.7 s/epoch) |

The matching audit (`matching_audit.json`) passes every check: same git commit,
same training-program hash, same fit rows, prep and encoded-cache hashes, same
initial-state hash, same step count, same RNG-stream hashes. This proves the
two arms executed the identical op sequence (only the pooling arithmetic
differs); it is **not** a claim of bitwise training reproducibility (GPU
reduction order remains nondeterministic, as measured in the budget round's
seed-0 probe).

The new Hard5 arm is an independent seed-0 re-run of the historical full-10k
recipe and reaches **valid MAE 0.118474** vs the historical 0.118618 read-only
reference (Δ = −0.000144; the frozen historical soup itself reproduces the
historical CSV bit-identically on this CPU, max|d| = 0.0).

Internal loss values recorded by the arms (fit_all 0.0469 Hard5 / 0.0471 Soft5;
monitor 0.0464 / 0.0437; dev 0.0455 / 0.0409) are measured on rows that are part
of the all-10k training set, so they quantify fit, not generalisation — the
fit→valid gap is 0.0716 (Hard5) / 0.0721 (Soft5).

The two frozen checkpoints produce quite different official-validation
predictions: mean |Δpred| = 0.0920, max |Δpred| = 1.862. So the null result is
not an "operator barely did anything" artefact.

## 6b. Phase F — frozen official-validation evaluation (y read once)

`freeze_manifest.json` → `valid_predictions.npz` → single label read:

| metric | Hard5 | Soft5 |
|---|---|---|
| **MAE** | **0.118474** | **0.119218** |
| RMSE | 0.598707 | 0.658496 |
| median abs | 0.067640 | 0.065110 |
| p90 abs | 0.208375 | 0.214412 |
| p95 abs | 0.291278 | 0.297743 |
| max abs | 18.228 | 20.090 |
| fit→valid gap | 0.071611 | 0.072081 |

Paired per-molecule Δ = |e_Soft5| − |e_Hard5|: **Δ = +0.000743**,
bootstrap 95% CI **[−0.006523, +0.008980]** (5000 resamples, seed 0), 51.2% of
molecules better under Soft5. Subgroups (all pre-declared):

| subgroup | n | Δ (Soft5 − Hard5) | CI95 |
|---|---|---|---|
| y ≤ −3 | 65 | +0.05046 | [−0.02179, +0.14053] |
| y > −3 | 935 | −0.00271 | [−0.00840, +0.00296] |
| atoms 10–20 | 207 | +0.0092 | — |
| atoms 20–26 | 511 | +0.0006 | — |
| atoms 26–36 | 282 | −0.0051 | — |
| diameter 5–13 | 487 | +0.0063 | — |
| diameter 13–19 | 513 | −0.0045 | — |
| bucket-4 fraction low half | 485 | +0.0101 | — |
| bucket-4 fraction high half | 515 | −0.0081 | — |
| last-bucket diversity low half | 487 | +0.0063 | — |
| last-bucket diversity high half | 513 | −0.0045 | — |

(Quartile strata in `subgroup_analysis.json`; the coarse half-split deltas above
are computed from the same strata.) The `valid:0172` extreme row is 15.4% of
Hard5 error mass and 16.9% of Soft5 error mass; excluding it gives Δ = −0.00112,
i.e. the small positive sign is not row-driven but the overall magnitude stays
within ±0.01. Top-10 error mass share: 22.5% Hard5 / 24.7% Soft5.

**Pre-frozen decision: band C (Δ > −0.003), `SOFT5_NO_USEFUL_GAIN`.** CI
contains zero; no subgroup shows a CI-supported improvement; the y ≤ −3 tail is
slightly worse under Soft5 (+0.05) while ordinary molecules are slightly better
(−0.0027, CI crossing zero). Absolute level: Soft5 valid 0.119218 → +0.02922 to
0.09 and +0.03922 to 0.08 (Hard5: +0.02951 / +0.03951).

Evidence boundary: single seed per arm; GPU reduction order is nondeterministic,
so this is an arm-level contrast, not a bitwise controlled variable. Official
validation was already exposed in earlier rounds (and the historical Hard5
0.118618 is a read-only reference), so this is a controlled comparison, not a
fresh blind test. Official test was never loaded or scored.

## 7. Evidence boundary

Phases A–D are label-blind. They establish a real representational loss in the
hard bucket and a non-trivial, exactly specified alternative; they say nothing
about prediction quality. The frozen Full-Y `distance_gate` remains hard-bucket;
Soft5 changes only the graph-level pooling of the pair states. Official
validation was already exposed in earlier rounds, so the upcoming evaluation is
a controlled comparison on an exposed instrument, not a fresh blind test.

Provenance note: the audit's read-only Hard5 reference is the frozen
confirmation-dir soup, pinned by SHA256 `7792c560…` (valid raw MAE 0.118618) in
the driver; an earlier draft pointed at the older cycle-dir soup copy and was
caught by the hash pin before any formal run.


## 8. Final report — the 15 questions

1. **How is the distance actually computed?** Unweighted BFS shortest path over
the hydrogen-free bonded graph in `zinc_patch_path_pooling._shortest_path_summary`
(line 387); the cached `pair_relation[:, 5]` equals the raw-graph BFS on all
2,668,346 train pairs (0 mismatches).

2. **Does `pair_relation` contain `log1p(distance)`?** Yes, raw 23-D column 5
(`np.log1p(float(distance))`, line 656), and the consumed 15-D slice
`P1_RELATION_INDICES = 0..13 + 18` keeps it while dropping the path-bond-mean and
adjacent-bond blocks.

3. **Where does the bucket truncate to 5+?** `bucket = min(max(int(distance),1),
DISTANCE_BUCKETS) - 1` with `DISTANCE_BUCKETS = 5` (line 620); distances ≥5 all
map to one one-hot position and to the same `distance_gate` row.

4. **What is `distance_gate`, and was it swapped?** `Embedding(5,48)` applied as
`1 + tanh(...)` on the hard bucket only; it was **not** swapped — Soft5 changes
only the graph-level pooling, not the gate.

5. **What goes into the pair encoder and what is `pair_value`?** 192-D input
`[left+right, |left−right|, product*gate, relation(15)]` → 48-D `pair_value`;
`pair_projection` is 144→48 (live audit).

6. **How is pooling actually done, and what does C6_MASK do to counts?** Per hard
bucket `first=Σh`, `second=Σh²`, `count=log1p(n)`; C6_MASK zeroes the unary
count, pair count and relation path-count channels (topology untouched). Soft5
keeps the count zeroed.

7. **What is the real reader input and parameter count?**
`[unary 289 | pair moments 485 | global 32 | topology 8] = 814`; 408,651
trainable parameters; both unchanged by Soft5.

8. **What does the real train-10k distance distribution look like, and how much
distance diversity is hidden inside the ≥5 bucket?** 2,668,346 pairs; bucket
shares 9.34/12.90/12.78/11.58/53.40%; diameter mean 12.46 (max 22). Inside the
≥5 bucket: mean 8.46 distinct exact distances per molecule (p50 8, p90 12, max
18), 99.80% mix ≥2 distances, 97.49% span ≥3, **99.25% hold both d=5 and d≥7**,
conditional entropy 1.91 nats — the last bucket is not one distance.

9. **Mean/variance/entropy of the distances inside the last bucket?** Per
molecule mean distance 7.46, variance 4.43, conditional entropy 1.91 nats
(p50 1.96); diversity correlates with atom count (0.80 for distinct distances,
0.77 for entropy) and bucket-4 pair share correlates 0.88 with atom count / 0.54
with ring rank.

10. **Was Soft5 a genuine replacement, not a cosmetic one?** On identical frozen
pair states over all 10,000 train molecules, the 480-D moment block differs by
mean relative 0.495 (p90 0.573, max 0.702); 100% of molecules differ >1%;
effective participation ratio drops 5.21→4.16. On official valid, the two frozen
models' predictions differ by mean |Δ| 0.092 (max 1.86).

11. **Did every correctness gate pass?** Yes (`correctness_gates.json`): Hard5
bit-identical to the frozen path (prediction and readout, max|Δ| = 0.0); Soft5
weights finite/non-negative/sum-1; conservation first 3.7e-8, second 3.9e-8,
count 2.0e-8 relative; endpoint-swap, pair-permutation, batch-composition and
single-molecule invariants; C6 count block exactly zero; 408,651 parameters;
identical init for both arms; gradients flow; forward is y-independent.

12. **Did the pre-frozen purchase gate pass?** Yes — `CPU_PURCHASE_GRANTED`:
G1 0.9925 ≥ 0.10, G2 1.0000 ≥ 0.25, G3 all correctness gates. (The gate shows
the operator is non-trivial on genuinely heterogeneous molecules; it is not
predictive evidence.)

13. **Were the two GPU arms truly matched, and did Hard5 reproduce the historical
level?** Yes: same commit, same training-program hash, same init-state hash, same
rows/prep/cache hashes, same 18,960 steps and last-5 soup, and identical RNG-state
hashes at epochs 1/40/120/240. The new Hard5 arm scores valid 0.118474 vs the
historical read-only 0.118618 reference (Δ = −0.000144).

14. **Soft5 vs Hard5 on frozen official validation?** Hard5 0.118474, Soft5
0.119218, Δ = **+0.000743**, bootstrap CI95 [−0.006523, +0.008980], 51.2% of
molecules better under Soft5; the y ≤ −3 tail is +0.050 (CI [−0.022, +0.141]) and
y > −3 is −0.0027 (CI [−0.0084, +0.0030]); no pre-declared subgroup shows a
CI-supported improvement; excluding `valid:0172` gives Δ = −0.00112.

15. **Decision, absolute gap, and boundary?** `SOFT5_NO_USEFUL_GAIN` (band C,
CI contains zero). Soft5 valid 0.119218 = +0.0292 to 0.09 and +0.0392 to 0.08
(Hard5 +0.0295 / +0.0395). Single seed per arm, GPU nondeterminism, official
validation already exposed (historical reference 0.118618), official test never
loaded; the round stops here — no seed 1, no third arm, no kernel/center/width
search, no retuning.
