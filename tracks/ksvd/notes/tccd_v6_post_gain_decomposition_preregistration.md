# Preregistration — TCCD-v6: POST Gain Decomposition

Round name: **TCCD-v6** (*POST Gain Decomposition: Normalization vs Marginal
Moments vs Nonlinearity*).
Study: `zinc-context-gap`.
Status: frozen before formal Stage-A experiments on 2026-09-22.

## 1. Question

TCCD-v5 established a real, reproducible positive POST increment on the frozen
TCCD-v2 representation:

```text
BASE = 0.2783639132976532   (Top-5 soup, internal 8000/2000 dev)
POST = 0.2522333264350891   (Top-5 soup)
G_FULL = 0.0261305868625641
```

while the placement hypothesis failed (`PRE ~= POST`,
`PRE-SHUFFLE ~= PRE`).  This round is a **frozen representation diagnostic**:
it does not design a new graph architecture, does not retrain the prototype
vocabulary, and does not change the pair set.  It decomposes the POST mean
pair descriptor

```text
pbar_G = [A_G, B_G, C_prod_G, D_G] in R^197
```

into a **RECON** block (coordinates proven to be exactly reconstructible from
the frozen `h_base`) and a **NOVEL** block (coordinates not reconstructible),
and asks which block, and whether a ReLU, is responsible for `G_FULL`.

## 2. Frozen lineage and execution discipline

Base representation: exact TCCD-v2 PrototypeREL.

* TCCD-v2 commit: `69a985a4ea3eb184c96c8ddad3857c4e87ee1dda`.
* Checkpoint: `tracks/ksvd/results/tccd_v2/prototype_rel_seed0_best.pt`.
* Checkpoint SHA-256: `093e0e46d4d5dab7e5558b807d993d1136df58d8f531d6b9b3621e8ab05c0f42` (verified).
* Local encoder: exact `714 -> 64` linear encoder, K=64 prototypes, cosine
  assignment, temperature `0.05 + 0.95*sigmoid(a)`.

Pair cache: exact TCCD-v5 cache, reused byte-for-byte.

* Cache protocol: `tccd_v5_occurrence_pair_nonlinearity_v1`.
* Pair cache metadata certifies `official_test_loaded: false`.
* Pair set: all unordered distinct occurrence pairs `i<j`; no self-pairs; no sampling.
* 10,000 graphs, 2,668,346 pairs, 231,664 occurrences, 5 relations in exact
  TCCD-v2 order `[R_int, R_b0, R_b1, R_b2, R_geo]`.
* All audit code paths load only `base`, `C`, `c_offsets`, `pair_i0`, `pair_i1`,
  `pair_rel`, `pair_offsets`, `graph_index`, and never `pair_*_y.npy`.

TCCD-v5 references reused, provenance verified by re-evaluation:

* `tracks/ksvd/results/tccd_v5/stageA_seed0.json`
  SHA-256 `2ab232c74219529ef3ddbdf1c6175d664756eb0dc7c0a0a30325d62a94b8aefe`.
* `tracks/ksvd/results/tccd_v5/stageA_post_seed0_soup.pt`
  SHA-256 `5bf555be35d7db8358fa78a5359330a4af2cb82d22d528168d992be307c10866`.
* `tracks/ksvd/results/tccd_v5/stageA_post_seed0_best.pt`
  SHA-256 `c705176388bbb455c9e1261eab7a387690b637a55768872b9219628c6e753390`.
* Recorded: BASE best `0.28791576623916626`, soup `0.2783639132976532`;
  FULL best `0.2608269155025482`, soup `0.2522333264350891`.
* The v6 FULL-NL arm must reproduce the frozen v5 POST soup/best dev MAE to
  within `1e-5` before any arm is trained.

Internal split: exact `internal_split(10000)`, `SPLIT_SEED = 20260922`,
8000 train / 2000 dev.

Formal compute: remote A100 **GPU1 only**.  GPU0 is forbidden.  Official ZINC
test is blocked throughout and never loaded.

## 3. Descriptor definition (unchanged from v5)

For every graph with `n` occurrences and every unordered pair `i<j`:

```text
p_ij = [c_i + c_j, |c_i - c_j|, c_i * c_j, r_ij] in R^197
```

Blocks, with explicit names to avoid confusion with the assignment matrix `C`:

| block | dims | definition | meaning |
|---|---:|---|---|
| `A` | 64 | `E_{i<j}[c_i + c_j]` | normalized prototype mean |
| `B` | 64 | `E_{i<j}[|c_i - c_j|]` | absolute prototype dispersion |
| `C_prod` | 64 | `E_{i<j}[c_i * c_j]` | prototype product / second moment |
| `D` | 5 | `E_{i<j}[r_ij]` | raw relation marginals |

```text
pbar_G = [A_G, B_G, C_prod_G, D_G]
```

`FULL-NL` consumes the unmasked `pbar_G`.  No new relation, no occurrence
network, no center-preserving composition, no topology feature is added.

## 4. Label-free algebra audit (frozen before any target-dependent work)

Run on the complete 10,000-graph frozen cache, without reading `y`:

```json
{
  "official_test_loaded": false,
  "target_not_read": true,
  "n_graphs": 10000,
  "tolerance": 1e-06,
  "A": {"formula": "A_G = 2 * BAG_G / n_G, n_G = sum_k BAG_G[k]",
        "max_abs_diff": 4.6193599700927734e-07, "pass": true},
  "D": {"relation_names": ["R_int", "R_b0", "R_b1", "R_b2", "R_geo"],
        "diagonal_per_occurrence": [0.0, 0.0, 0.0, 0.0, 1.0],
        "max_abs_diff_per_relation": [2.384185791015625e-07, 2.9802322387695312e-08,
                                       7.450580596923828e-09, 3.725290298461914e-09,
                                       1.5273690223693848e-07],
        "reconstructible_mask": [true, true, true, true, true],
        "status": "BASE_RECONSTRUCTIBLE"},
  "S_all_identity_max_abs_diff": 0.00019227155280532315,
  "S_all_identity_rel_diff": 1.0801771956669848e-07,
  "recon_mask_nonzero": 69,
  "novel_mask_nonzero": 128,
  "frozen_mask_matches": true,
  "all_pass": true
}
```

### Block A proof

The first `K = 64` base coordinates are `b_G = sum_i c_i`, and every
assignment row sums to one, so `n_G = sum_k b_{G,k}`.  For `n >= 2`:

```text
A_G = E_{i<j}[c_i + c_j] = (n-1) * b_G / C(n,2) = 2 b_G / n_G
```

Numerically verified on all 10,000 graphs (max abs diff `4.62e-07`).

### Block D proof

With `M_r = C^T R_r C` and `sum_k c_{ik} = 1`:

```text
sum_{k,l} [M_r]_{k,l} = sum_{i,j} R_r[i,j]
```

`R_r` is symmetric for all five relations, so the stored upper triangle
(including the diagonal) gives the full sum:

```text
S_up[r]        = sum_{k<=l} [M_r]_{k,l}          (read from h_base)
diag_in_base[r]= sum_k [M_r]_{k,k}               (read from h_base)
S_all[r]       = 2*S_up[r] - diag_in_base[r] = sum_{i,j} R_r[i,j]
```

The occurrence-pair descriptor excludes the operator diagonal and averages
`i<j` only, with frozen diagonal conventions: `R_int` explicitly zeroed,
`R_b*` zero (no self loops), `R_geo[i,i] = exp(0) = 1`:

```text
sum_{i<j} R_r[i,j] = (S_all[r] - n_G * delta_{r, R_geo}) / 2
D_G[r]             = sum_{i<j} R_r[i,j] / C(n_G, 2)
```

Numerically verified per relation on all 10,000 graphs
(`<= 1.53e-07` per relation), plus the independent identity
`2 * C(n,2) * D_pair[r] + n * delta_{r,R_geo} == S_all[r]` at relative
`1.08e-07`.

### Blocks B and C_prod

`B` and `C_prod` are **not** reconstructible from `h_base`.  The base stores
only the occurrence sum `sum_i c_i` and graph-specific quadratic forms
`C^T R_r C`; it does not store the unary second moment `sum_i c_i * c_i`, the
absolute dispersion, or the pair product moment.  No exact or mechanical
formula exists, so they are placed in NOVEL.  This classification is by
algebra/code path only, never by prediction quality.

## 5. Frozen partition (fixed before any Stage-A training)

| block | coordinates | dims | partition |
|---|---|---:|---|
| `A` | `0..63` | 64 | RECON |
| `B` | `64..127` | 64 | NOVEL |
| `C_prod` | `128..191` | 64 | NOVEL |
| `D` | `192..196` | 5 | RECON (all five) |

* RECON nonzero coordinates: 69.
* NOVEL nonzero coordinates: 128.
* RECON descriptor zeroes `64..191`; NOVEL descriptor zeroes `0..63` and
  `192..196`.
* Both descriptors remain 197-D, so every arm has the identical input
  interface, tensor shapes, parameter count, and optimizer.
* `FULL = RECON + NOVEL` coordinatewise.

The partition may not be changed after observing any Stage-A dev MAE.  If the
audited mask differs from the frozen mask, Gate 0 fails and the round stops.

## 6. Arms

All arms share the exact frozen TCCD-v2 `h_base` (10,464-D) and the exact
TCCD-v5 lightweight Stage-A optimizer/stopping/soup protocol.  No sweep is
performed.

| arm | input | branch |
|---|---|---|
| `A0 BASE` | `h_base` only | reused linear reader (v5 result) |
| `A1 FULL-NL` | `pbar_G` | `Linear(197,64) -> ReLU -> Linear(64,16)` (reused v5 POST) |
| `A2 RECON-NL` | `pbar_G * RECON_MASK` | `Linear(197,64) -> ReLU -> Linear(64,16)` |
| `A3 RECON-LINFACT` | `pbar_G * RECON_MASK` | `Linear(197,64) -> Linear(64,16)` (no ReLU) |
| `A4 NOVEL-NL` | `pbar_G * NOVEL_MASK` | `Linear(197,64) -> ReLU -> Linear(64,16)` |

Final reader for A1–A4: `[h_base, q] -> Linear(10480, 1)`.

* Parameter counts: branch `13,712`, head `10,481`, total `24,193` for every
  new arm.  Parameter mismatch must be 0.
* Initialization: all four arms are constructed with the same seed
  (`torch.manual_seed(seed + 33001)`), the same construction order, and the
  same tensor shapes.  Same-named tensor max difference must be exactly 0; the
  checksum must equal the recorded v5 POST checksum
  `9fbedb92ced44449ade422b152684e3706f34fb943c32b5bfd64979290e8487f`.
  ReLU has no parameters, so NL and LINFACT share a bit-identical init state.
* FULL-NL is not retrained: the frozen v5 POST checkpoint is loaded into the
  v6 FULL-NL arm and re-evaluated.  Both soup and best dev MAE must match the
  recorded values within `1e-5`.
* PRE, PRE-SHUFFLE, pair occurrence placement, and any pair-to-centre update
  are explicitly forbidden in this round.

## 7. Stage A protocol

* Frozen cache only; no TCCD local encoder, prototype matrix, or temperature is updated.
* Adam, batch 32, lr `1e-3`, weight decay `1e-5`, gradient clip 5,
  max 240 epochs, patience 40, fixed equal-weight Top-5 soup.
* Primary metric: Top-5 soup MAE on the fixed internal 2000-graph dev split.
* Seed 0 for the three new arms.  Seed 1 only under §9's ambiguity rule.

## 8. Gate 0

Must pass before any Stage-A training; any failure stops the round.

1. **G0.1** exact FULL reconstruction from the frozen pair cache, including
   descriptor equivalence against the exact v5 tensor path and successful
   re-evaluation of the frozen v5 POST soup/best checkpoints.
2. **G0.2** `FULL = RECON + NOVEL` coordinatewise, max diff `< 1e-6`.
3. **G0.3** `A = 2*BAG/n` on all 10,000 graphs, max diff `< 1e-6`.
4. **G0.4** per-relation D reconstructibility on all graphs; the audited mask
   must equal the frozen mask; a failed relation may not enter RECON.
5. **G0.5** identical `197 -> 64 -> 16` shapes and `10480 -> 1` head in all arms.
6. **G0.6** identical trainable parameter count across A1–A4.
7. **G0.7** bit-identical initialization across A1–A4 (max diff 0) and equal to
   the recorded v5 POST init checksum.
8. **G0.8** permutation invariance of descriptors and predictions under joint
   node relabel and pair-order permutation.
9. **G0.9** batching invariance.
10. **G0.10** `official_test_loaded: false` everywhere.

## 9. Decision rules

```text
G_FULL  = 0.2783639132976532 - 0.2522333264350891 = 0.0261305868625641
G_X     = BASE_soup - MAE_X_soup
rho_X   = G_X / G_FULL
gap     = MAE_RECON-LINFACT - MAE_RECON-NL
```

* **Case R (RECON SUFFICIENT):** `G_RECON >= 0.015` and
  `MAE_RECON-NL <= MAE_FULL + 0.005`.
  * **R1 (NONLINEARITY NOT MATERIAL):** `gap < 0.005`.
  * **R2 (NONLINEAR DECODING MATERIAL):** `gap >= 0.010`.
  * **Ambiguity `0.005 <= gap < 0.010`:** run exactly one paired seed 1 for
    RECON-NL and RECON-LINFACT only; `mean_gap >= 0.0075` -> R2, otherwise
    R1 / inconclusive-small.
* **Case M (NOVEL MARGINAL MOMENTS SUFFICIENT):** `G_NOVEL >= 0.015` and
  `MAE_NOVEL-NL <= MAE_FULL + 0.005`.
* **Case J (JOINT GLOBAL MARGINAL INTERACTION):** `G_RECON >= 0.005`,
  `G_NOVEL >= 0.005`, `MAE_RECON-NL > MAE_FULL + 0.010`, and
  `MAE_NOVEL-NL > MAE_FULL + 0.010`.
* **Case N (DECOMPOSITION DOES NOT EXPLAIN FULL):** `G_RECON < 0.005`,
  `G_NOVEL < 0.005`, and `G_FULL >= 0.02` (J/N-strong-synergy).
* Precedence for the final label: R (R1/R2) > M > J > N.  If none matches
  exactly, report `INCONCLUSIVE_INTERMEDIATE` without inventing a new channel.

## 10. Explicitly forbidden work

No end-to-end prototype retraining, K-SVD, K sweep, temperature sweep, local
encoder change, new relation, pair occurrence network, center-preserving
model, GNN, topology feature, official-valid architecture selection, official
test, PRE branch, nonlinear 10,464-D reader, or moment-complete architecture.
This round ends at the frozen decomposition.  The next architecture is
determined by the observed Case and requires a new preregistration.

## 11. Durable records

* `tracks/ksvd/results/tccd_v6/label_free_audit.json` — label-free audit.
* `tracks/ksvd/results/tccd_v6/gate0.json` — Gate 0.
* `tracks/ksvd/results/tccd_v6/stageA_seed0.json` — Stage A and the Case.
* `tracks/ksvd/results/tccd_v6/stageA_recon_seed1.json` — only if needed.
* Per-arm checkpoints and summaries.
* Analysis note, claim, decision, and `tracks/ksvd/STATE.yaml` update.
* Every record states GPU1-only, `official_test_loaded: false`, checkpoint
  SHAs, masks, parameter counts, initialization checksums, gains, recovery
  fractions, and the final Case.
