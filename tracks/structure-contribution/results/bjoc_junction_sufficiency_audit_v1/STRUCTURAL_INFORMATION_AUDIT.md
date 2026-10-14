# Structural information audit — BJOC Junction sufficiency v1

Companion to `REPORT.md`. Protocol `cscl-bjoc-junction-sufficiency-audit-v1`, split D, one seed, label-blind except the single predeclared Monitor score (which Part B never touches). Everything here is deterministic and CPU-only; raw numbers are in `b1_algebraic.json`, `b5_synthetic.json`, `b2_layerwise.json`, `b3_collisions.json`, `decision.json`.

The audit deliberately separates three evidence classes:

1. **Algebraic / theoretically necessary blindness** — can be proved from the formulas, but is conditional on what is held fixed.
2. **Synthetic counterexamples with known truth** — show what a mechanism can and cannot distinguish under controlled construction.
3. **Real-molecule observations** — what actually happens on official-train structures. These are the weakest evidence for "impossible", because a finite search never proves non-existence.

## 1. The Null grouping and what it can see

Frozen system (A2 = BJOC-Null): `W_e = n_e·μ_G`, `V_e = n_e·ν_G`, `q_e = log1p(n_e)`, where `n_e` counts the junctions touching physical bond `e`, `μ_G`/`ν_G` are the graph means of the encoded junction vectors `w_j` and their squares, and the reader input is `[Σ_e c_e (96), Σ_i a_i (96), Σ_i a_i² (96), g_G (33)]`.

Grouping-level invariants: the per-graph quantities `Σ_e W_e = 2·n_junc·μ_G`, `Σ_e W_e² = μ_G²·Σ_e n_e²`, and the analogues for `V` depend only on the encoded `w` multiset and the multiset of per-bond counts. The per-bond vectors additionally depend on *which* bond carries which count.

## 2. B1 — conditional representation-level blindness (verified)

Statement: for fixed raw bond representations, fixed per-bond junction counts, fixed encoded junction multiset `{w_j}` and fixed other atom/global content, any reassignment of junction content to bonds leaves `W_e`, `V_e`, `q_e` unchanged for every bond (Null grouping), while the Organized grouping `W_e = Σ_{j∋e} w_j`, `V_e = Σ_{j∋e} w_j²` changes unless the reassignment is an incidence automorphism.

Worked example (historical 4-bond / 4-junction counterexample, reused):

| | bonds P1, P2, Q1, Q2 | junctions | assignment | per-bond Null `W` | per-bond Organized `W` |
|---|---|---|---|---|---|
| A | all have 2 junctions | X1, X2 (w=1); Y1, Y2 (w=2) | P1{X1,X2}, P2{Y1,Y2}, Q1{X1,Y1}, Q2{X1,Y1} | 3, 3, 3, 3 | 2, 4, 3, 3 |
| B | same | same | P1{X1,X2}, P2{Y1,Y2}, Q1{X1,Y2}, Q2{X1,Y2} | 3, 3, 3, 3 | 3, 3, 3, 3 |

Null cannot separate A and B; Organized can. Real-model check on the frozen P3 (`b4_test2_object_counterexample`): A0/A1/A2 mode max |ΔW| = 0.0 (exact), Organized max |ΔW| = 0.0586, max |ΔV| = 0.0977, and a frozen-prediction difference of 2.1e-5 (representation difference survives, diluted by the reader). **Caveat recorded: the switch is performed after encoding and does not have to correspond to a realisable molecule.**

Scaled family (`b5a_scaled_ownership_family`), m = 2, 3, 4, 5 bond pairs trained the same way, each bond touched exactly twice, all junctions seeing the same unordered bond-type pair, so the encoded `w` multiset is exactly preserved: Null `max|ΔW| = max|ΔV| = 0.0` and `max|Δpred| = 0.0` in all four cases; Organized `max|ΔW| ≥ 0.0586` in all four; the ownership-only synthetic target differs by 8, 8, 36, 22 while the marginal-only target is identical (`0.0`).

## 3. B5b — the conditional part is real: raw tokens do not fix `w`

A random family of 400 small synthetic instances with 6 bonds / 6 junctions and marginal signatures `(sorted bond tokens, sorted junction tokens, sorted per-bond counts)` contains 38 within-signature pairs with different incidences. For those pairs the Null per-graph summaries are **not** identical (max absolute difference 785.2), the ownership target differs by up to 23, and the marginal target never differs. Reason: the junction encoder consumes the two incident bond embeddings, so the *pairing* is visible inside `w` before pooling.

Consequence: "Null keeps the graph junction multiset but discards ownership" must be stated conditionally — Null discards the residual ownership information that is not already smuggled into `w` by the junction encoder's bond inputs. It is exactly blind *after* encoding, not *before*.

## 4. B2 — layer-wise representation audit (Fit first 2000 + all 1000 Monitor rows, label-blind)

Level 0 (raw structure summaries): atom-type histogram effective rank 8.7, 33-dim `global_feat` effective rank 8.6 — the fixed global vector is itself low-dimensional.

Level 1 (per-molecule embedding moments, frozen P3 weights, mode overrides):

| block (dim) | effective rank (A0 / A1 / A2) | PCA d90 (A0/A1/A2) |
|---|---|---|
| atom mean (96) | 21.8 / 21.8 / 21.8 | 9 / 9 / 9 |
| atom var (96) | 54.3 / 54.3 / 54.3 | 20 / 20 / 20 |
| bond mean (96) | 59.4 / 59.4 / 59.4 | 26 / 26 / 26 |
| bond var (96) | 74.2 / 74.2 / 74.2 | 41 / 41 / 41 |
| junction mean (96) | 58.1 / 58.1 / 58.1 | 23 / 23 / 23 |
| junction var (96) | 50.5 / 50.5 / 50.5 | 12 / 12 / 12 |
| composer sum (96) | 32.4 / 32.0 / 27.9 | 10 / 7 / 4 |

Level 2 (`Σ_b c_b`): Junction moments *reduce* the effective rank of the composer summary (32.4 → 27.9) while concentrating 90 % of variance in 4 principal directions — the downstream summary becomes more collinear, not richer.

Level 3 (321-dim reader input, Fit-only ridge R² of each block from the other three):

| block | ridge R² from the rest |
|---|---|
| composer sum (96) | 0.536 |
| atom sum (96) | 0.968 |
| atom squared sum (96) | 0.927 |
| global topology (33) | 0.938 |

Fit-standardised pairwise distances among distinct Fit structures (500 random pairs): composer-sum median 11.33 / p01 7.05; atom-sum median 12.88 / p01 6.56; atom-sq median 12.11 / p01 7.05; global median 7.12 / p01 2.71. These are the reference scales for the collision search.

## 5. B3 — real-molecule collision search (official train 10,000 structure fields, label-blind)

Rules predeclared before the scan: exact key = 33-dim `global_feat` float32 bytes + atom-type histogram + bond-type histogram + degree histogram + per-bond junction-count multiset; a collision counts only if canonical SMILES differ; a not-found verdict must report the nearest distinct-molecule pairs with their distances (≤ 200 candidates, ≤ 20 reported); the strict token additionally requires the Fit-standardised reader distance ≤ the 1st percentile of distinct-structure Fit distances.

Results (`b3_collisions.json`):

* `n_structures = 10000`; object rows asserted equal to official train row ids (`row_alignment_verified = true`); no labels read.
* **3 exact structural-statistics collision groups, all 3 pairs have equal canonical SMILES** (ZINC duplicate rows): `exact_collision_pairs_same_smiles = 3`, `exact_collision_pairs_different_smiles = 0`. NetworkX typed-graph isomorphism checks on the candidate pairs returned non-isomorphic only for the (non-existent) cross-molecule cases; every exact group is a same-molecule duplicate.
* Therefore **`REAL_COLLISION_NOT_FOUND_WITHIN_BUDGET`** (the descriptive B3 phrase `NO_REAL_EXACT_COLLISION_FOUND_UNDER_SEARCH_BUDGET` also applies).
* Nearest *distinct-molecule* pairs by signature distance (reader distance standardised by the 2000-row Fit reference, Fit p01 = 14.26, median = 22.59):

| official rows | n_atoms | n_bonds | n_junctions | signature distance | reader L2 | frozen pred |Δ| |
|---|---|---|---|---|---|---|---|
| 3676, 5288 | 22 | 24 | 32 | 0.183 | 7.197 | 0.561 |
| 4738, 7410 | 18 | 19 | 26 | 0.354 | 8.398 | 0.625 |
| 2490, 3339 | 23 | 24 | 35 | 0.454 | **3.352** | 0.074 |
| 36, 7416 | 22 | 23 | 30 | 0.484 | 5.348 | 0.107 |
| 7483, 8699 | 25 | 27 | 37 | 0.530 | 11.079 | 1.037 |

The minimum distinct-molecule reader distance found (3.35) is ~23 % of the Fit 1st-percentile reference and ~15 % of the median: the representation is genuinely compressive, so some different structures are much closer than typical, but no exact collision exists within this budget.

Search limitations (to be stated whenever the negative verdict is quoted): the exact key is a finite statistic vector, not the model's full input; the near-search used ≤ 200 candidate pairs ranked by the signature, so a physically closer pair outside that budget cannot be excluded; the canonical-SMILES table is the committed official-train table and alignment was verified against object rows; the comparison is against the frozen P3 model only.

## 6. B4 — intervention tests

| test | construction | result |
|---|---|---|
| 1: junction row permutation | reorder junction rows inside each molecule | A0 0.0, A1/A2 2.4e-7 max prediction change (known invariance, re-checked, not a discovery) |
| 2: assignment switch | post-encoding ownership switch with fixed junction multiset and per-bond counts | Null exactly invariant; Organized W/V change (0.059/0.098); frozen prediction changes 2e-5 |
| 3: real near-collision compare | the B3 nearest distinct pairs | reader distances 3.35–12.3 (standardised), predictions differ 0.024–1.04 → "the model distinguishes these pairs weakly", never "identical" |
| 4: block neutralisation | zero each reader-input block, frozen P3, 16 Fit molecules | mean |Δpred| global 0.174, composer 0.466, atom-sum 0.833, atom-sq 0.892; out-of-distribution sensitivity only, not attribution |

## 7. What this audit does and does not establish

Established:
* The Null grouping is exactly blind to ownership *conditional on the encoded junction multiset and per-bond counts* (algebra + two independent constructions).
* Part of the pairing information reaches the model through `w` before pooling; the blindness claim must be stated conditionally.
* On real train structures, exact cross-molecule structural-statistics collisions were not found within the declared budget; duplicates in ZINC produce the only exact matches.
* Neighbouring distinct structures can be much closer in the frozen reader input than typical distinct structures (3.35 vs 14.26 standardised), which is the practical form of "structure information loss" this round can measure.

Not established:
* That a real collision does not exist (finite budget, finite key).
* That the near-collisions cause property errors.
* That ownership information would help downstream (not trained this round; historical Organized/GLRC evidence was mixed).
* Any causal or attributive claim from block neutralisation or single-seed MAE contrasts.
