# Bond line-graph random-walk structure audit

- protocol `cscl-bond-random-walk-organization-v1` · commit `10496f29243c3a55ca7cb123d00c8322f1c55e4f`
- 2000 split-D Fit molecules (label-blind), first 2000 sorted Fit rows
- Part A decision: **PART_A_PASS_GPU_PURCHASE**

## 1. Kernel scales and mixing

| quantity | k=1 | k=2 | k=3 | k=4 |
|---|---:|---:|---:|---:|
| mean TV(P^k, uniform) | 0.842 | 0.745 | 0.688 | 0.651 |
| mean normalised entropy | 0.371 | – | 0.577 | – |
| mean reachable fraction | 0.158 | – | 0.442 | – |

- mean scale distinctness row-TV(P1,P3): **0.250** (raw matrix L1/(2N^2): 0.0104; see amendment-1)
- increment cosines: (P1-U,P2-U) 0.962; (P2-U,P3-U) 0.987
- row cosine P2 vs P3: 0.988
- K mean TV to uniform: **0.744**; effective rank: 8.97

Pre-frozen gate conditions: {"mean_tv_p3_ge_min": true, "mean_tv_K_ge_min": true, "scale_distinctness_ge_min": true}

By topology category:

| category | n | mean bonds | TV(P^3,U) | K TV(U) |
|---|---:|---:|---:|---:|
| acyclic_branch | 9 | 13.0 | 0.574 | 0.638 |
| ring_chain | 1990 | 25.1 | 0.689 | 0.745 |
| ring_only | 1 | 28.0 | 0.740 | 0.778 |

## 2. Invariance

- atom relabel max kernel diff: 0.000e+00
- bond permutation max conjugation diff: 1.110e-16
- bond permutation max statistic diff: 8.882e-15

## 3. Bond-class diffusion (k=3)

| class | n | mean TV to uniform |
|---|---:|---:|
| nonring_type1 | 16526 | 0.704 |
| nonring_type2 | 2991 | 0.741 |
| nonring_type3 | 139 | 0.794 |
| ring_type1 | 20723 | 0.688 |
| ring_type2 | 9703 | 0.693 |

## 4. Controlled organisation pairs

- marginal signature: `(n_bonds, sorted bond types, sorted line-graph degrees, sorted per-bond junction counts)`
- signature buckets: 1760; multi-molecule buckets: 176
- pairs with identical marginals but different K: **208**
- K sorted-entry max-abs difference among matched pairs: mean 0.023190870192307693 / max 0.050346999999999975
- loose signature `(n_bonds, sorted line-graph degrees)`: 365 multi-molecule buckets
- note: marginal signature is coarse (raw types/degrees/junction counts); K differs within the bucket

## 5. Communities on the bond line graph

- Walktrap (steps=4) mean communities/molecule: **3.63**; TASU 2ECC/bridge: 10.16; greedy modularity: 4.33
- mean ARI: walktrap_vs_tasu 0.506, walktrap_vs_greedy 0.629, walktrap_vs_cc 0.000, greedy_vs_tasu 0.467
- community connectivity: 0.931
- singleton community share: 0.034
- ring-like community share: 0.106
- Walktrap relabel ARI: min -0.024, mean 0.994, fraction >= 0.9 0.980
- every bond in exactly one community: True (partition covers all bonds: True)
- atoms with >1 incident bond: 38282; of these boundary atoms: 6713 (0.175)

**Boundary statement.** community counts/sizes are structural observations, not chemistry; bond communities guarantee unique bond membership but not unique atom membership
