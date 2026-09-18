# Structural-encoder mechanism decomposition (B-bag + rank-1 audit)

**Date:** 2026-09-15
**Branch:** `exp/structural-encoder-mechanism-decomposition`
**Commit (formal runs):** `5b19132890008e65c70a3a3392712ac092641f49`
**Starting revision:** `7372025` (`audit/identity-incremental-information`; contains
Agent A identity-capacity-control, Agent B shared-structural-patch-encoder and the
identity-incremental-information audit)
**Run tags:** `sbpe-s0` (GPU0), `sbpe-s1` (GPU1)
**Remote:** `res` = `hxy@a100-2`, repo `/home/hxy/cy/research`
**Verdict:** **connectivity axis: Case C1 (connectivity supported).**
**composition axis: inconclusive** (Delta_composition mean is negative and
sign-flips across seeds). **Representation axis: Case R1** — the frozen B-full
16-D `e_struct` is functionally a rank-1 scalar modulation.

Official ZINC test was **never loaded**.

---

## 1. Question

B-full (shared connectivity-aware encoder) improves the deterministic-A100 ZINC
Cell A 2-seed Top-5 soup valid MAE `0.126368 -> 0.118972`, but its 16-D
`e_struct` is nearly rank-1. Two questions:

- **Q1** Is the extra gain from *generic shared capacity* (A2), *primitive
  composition* (B-bag), or *real patch connectivity* (B-full)?
- **Q2** What is the near rank-1 `e_struct` actually encoding? Is the channel
  essentially a scalar structural modulation?

## 2. Reused (never retrained) baselines

| condition | 2-seed Top-5 soup valid | params | source |
|---|---:|---:|---|
| A2 (no typed identity + capacity-matched shared adapter) | **0.121914** | 85,740 | `results/compact_v4_identity_capacity_control` |
| B-full (shared connectivity-aware encoder) | **0.118972** | 84,495 | `results/shared_structural_patch_encoder` |

Provenance verified before running: cell-A baseline rebuilds to 85,763; the
B-full candidate rebuilds to 84,495; `torch.use_deterministic_algorithms(True)`;
seeds `[0, 1]`; identical fixed equal-weight Top-5 soup rule; identical `h=64`,
`q=16`, `T=2`, radius-2 patch split and preprocessing (B-bag **reuses B-full's
target-free structural graph cache verbatim**, schema
`shared_structural_patch_graphs_v2`). No baseline was retrained.

## 3. B-bag: exact architecture

```
node primitive: atom_embedding(28,48) + root_embedding(2,48) + distance_embedding(3,48)
              -> shared node MLP (48 -> 96 -> 48, ReLU)
              -> node mean / std / root pooling
bond primitive: bond_embedding(4,24)
              -> shared bond MLP (24 -> 48 -> 24, ReLU)
              -> bond mean / std pooling
concat [root(48); node_mean(48); node_std(48); bond_mean(24); bond_std(24)] = 192
              -> fusion MLP (192 -> 104 -> 16, ReLU) -> e_bag in R^16
```

`rounds = 0`: no message passing, no `struct_src` / `struct_dst` read, no edge
endpoint, no attention, no learned token dictionary. Bonds are grouped to their
patch only through the explicit `struct_edge_patch` index (a grouping index, not
adjacency). Same node/edge embedding widths as B-full (`node_dim=48`,
`edge_dim=24`); the node-MLP / bond-MLP / fusion widths were fixed **once**
(no sweep) to match B-full's released encoder budget.

### 3.1 Why it isolates connectivity from B-full

B-bag and B-full consume **exactly the same primitive multisets** (atom type,
root flag, root-distance, bond type), have the same output width (16), the same
parameter budget (within 0.02%) and byte-identical downstream backbone. The only
remaining degree of freedom is whether a patch is processed by 2 rounds of
edge-aware message passing over its **real adjacency** (B-full) or by a
permutation-invariant bag pooling that never sees an edge endpoint (B-bag).

The B-bag vs A2 comparison is **not** a clean information-matched comparison:
A2 reads the 146-D handcrafted shell descriptor (which itself encodes
connectivity-derived quantities), whereas B-bag reads only the raw primitives.
That confound is recorded explicitly (section 7).

## 4. Exact parameter accounting

| block | Cell A | B-full | **B-bag** |
|---|---:|---:|---:|
| typed token embedding (hybrid lookup) | 36,420 | 0 | 0 |
| shared structural patch encoder | 0 | 35,152 | **35,168** |
| parent / patch / pair / relation / centre / global / topology / head | 49,087 | 49,087 | 49,087 |
| **total** | **85,763** | **84,495** | **84,511** |

`candidate - bfull = +16` (`+0.019%`, within the pre-registered `±1%` budget).
Every one of the 15 shared blocks is identical to B-full; `h=64`, `q=16`, `T=2`,
`R=334` unchanged; no vocab-sized `typed_embedding`.

## 5. Integrity / adversarial tests (20/20 pass, local + remote)

The critical gate (brief Test 6/7): two rooted trees with **identical primitive
multisets** (atom all-0; root at node 0; distance labelling `[0,1,1,2,2]`; bond
multiset `{0,0,0,0}`) but **different actual connectivity**.

| | max abs delta |
|---|---:|
| B-bag (`e_bag` between the two patches) | **0.0** |
| B-full (`e_struct` between the two patches) | **1.787e-3** |

So B-bag is exactly invariant to connectivity while B-full is sensitive to it.
Also verified: node-relabel invariance; atom / bond-multiset / root-designation
/ root-distance changes all move the representation; B-bag forward runs on a
batch that carries **no endpoints at all**; no vocab-sized typed embedding;
output width 16; `q16/h64/T2`; forward/backward finite; parameter budget.
Repo suite: **888 passed**. Deterministic GPU repro bit-identical across GPU0 and
GPU1 in 6 epochs (`selection_state_sha256 =
2198026f796883bea1915068f37b0cc3129d47b98cb96606bbab6b555513b491`).

## 6. B-bag results (deterministic A100, fixed Top-5 soup)

| seed | raw best valid | best epoch | train@best | Top-5 soup | soup gain | wall | peak GPU |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.129574 | 199 | 0.093839 | **0.127382** | +0.002192 | 5553 s | 370 MB |
| 1 | 0.128219 | 174 | 0.096183 | **0.120229** | +0.007990 | 2164 s | 372 MB |

- **B-bag 2-seed soup mean = 0.123806** (raw mean 0.128896).
- Per-seed spread of the soup is large (0.0072) relative to the deltas.
- seed0 ran on a contended GPU0 (another user held ~34.9 GiB), only wall time
  affected; determinism is contention-independent.

## 7. Mechanism decomposition

```
A2     0.121914
B-bag  0.123806
B-full 0.118972   (frozen)

Delta_composition  = A2  - Bbag  = -0.001891   (seed0 -0.005688, seed1 +0.001905)
Delta_connectivity = Bbag - Bfull = +0.004833   (seed0 +0.007564, seed1 +0.002103)
Delta_total        = A2  - Bfull = +0.002942
```

2-seed means order **B-full < A2 < B-bag**.

- **Connectivity is the only robust increment**: `+0.004833` (>= 0.002) and both
  seeds agree in sign.
- **Composition is not a robust positive increment**: the per-seed
  `Delta_composition` **sign-flips** and the mean is **negative** (B-bag is on
  average *worse* than A2).

### 7.1 Decision case

- Connectivity axis: `|A2-Bbag| = 0.001891 < 0.002`, `Bbag-Bfull = 0.004833 >=
  0.002`, paired seeds same direction (both positive) -> **Case C1
  (connectivity supported)**.
- Composition axis: not direction-consistent and negative on average -> the
  composition increment is **inconclusive**, not "proven neutral". The `<0.002`
  magnitude only passes because the two per-seed composition deltas cancel
  (`-0.0057 / +0.0019`).
- Confound: A2 reads the 146-D handcrafted shell descriptor; B-bag reads only
  raw primitives. `Delta_composition` therefore mixes input information with
  the composition mechanism. The defensible composition statement is narrower:
  *a shared bag encoder built from raw primitives is not better than the generic
  shared adapter*; it is **not** evidence that compositional structure is
  worthless.

## 8. Workstream B — rank-1 mechanism audit (zero training)

Frozen B-full seed0/seed1 selection checkpoints; official train for fitting
(231,664 patch occurrences, 6,784 unique typed certificates); valid for
evaluation. Official test never loaded.

### 8.1 Spectrum (train)

| seed | PC1 EV (occ) | eff. rank (occ) | PR | PC1 EV (unique-structure) | eff. rank (unique) |
|---|---:|---:|---:|---:|---:|
| 0 | 0.9793 | 1.106 | 1.042 | 0.9937 | 1.039 |
| 1 | 0.9567 | 1.196 | 1.090 | 0.9504 | 1.220 |

PC1-PC4 cumulative variance = 1.0000 in every case; the occurrence-weighted
numbers reproduce the previous valid-side diagnostics (1.107 / 1.202).
Unique-structure weighting makes the collapse *stronger*, so it is not a
high-frequency-patch artifact.

### 8.2 What is PC1? (train-fit PC1 score vs deterministic descriptors)

Fixed Spearman correlation on official train, 224 descriptors (8 scalar
counts/degrees/cycle/boundary, 28+4+3 atom/bond/distance counts and
compositions, 146 shell-descriptor dims):

- seed0 strongest: `atom_composition_0` `rho=-0.483`,
  `patch_cont_108 (root_atom[0])` `-0.467`, `patch_cont_0 (root type-0/n)`
  `-0.445`, `atom_composition_2` `+0.398`.
- seed1 strongest: `patch_cont_1 (root type-1/n)` `+0.540`,
  `patch_cont_109 (root_atom[1])` `+0.529`, `patch_cont_57` `-0.318`.

No single descriptor dominates (max `|rho|` ~0.48 / ~0.54). The strongest
correlates are root-atom-type / atom-composition directions.

Fixed train -> valid ridge probe (alpha = 1.0, one setting, no sweep) predicting
the PC1 score from the descriptors:

| seed | valid R² |
|---|---:|
| 0 | 0.279 |
| 1 | 0.553 |

So PC1 is *partially* recoverable from existing handcrafted descriptors, but it
is not a trivial existing statistic, and the recoverability is seed-dependent.
This sits **between R3 and R4**.

### 8.3 Frozen rank-1 functional diagnostic

Replace the frozen model's `e_struct` by its train-fit rank-1 reconstruction
`mean + <e_struct - mean, v1> v1` (per seed, occurrence-weighted train basis);
all other tensors and the evaluation loader are untouched.

| | selection full -> rank1 | soup full -> rank1 | soup mean-only reference |
|---|---:|---:|---:|
| seed0 | 0.125324 -> 0.125337 (**+0.000013**) | 0.119818 -> 0.119838 (**+0.000020**) | +0.000269 |
| seed1 | 0.124394 -> 0.124383 (**-0.000011**) | 0.118126 -> 0.118138 (**+0.000011**) | +0.004989 |
| 2-seed soup mean | — | 0.11897219 -> 0.11898768 (**+0.0000155**) | 0.121601 |

- The local `full` evaluation reproduces the deterministic-A100 frozen B-full
  values exactly (`0.119818 / 0.118126`, mean `0.11897219`), so the intervention
  is measured on the real trained model.
- Relative reconstruction error on valid: 2.0% (seed0) / 4.3% (seed1);
  PC1 valid explained variance 98.0% / 95.7%.
- Replacing the whole 16-D channel by one scalar costs `~2e-5` MAE
  (`<< 0.001`) -> **Case R1: the 16-D structural channel is functionally
  sufficient as a scalar modulation.**
- Secondary: setting `e_struct := train mean` (no patch-specific modulation) is
  nearly neutral for seed0 (+0.0003 soup) but clearly harmful for seed1
  (+0.0050 soup) — even the scalar's functional necessity is seed-dependent.

## 9. Joint mechanism matrix

```
                        performance      representation rank   interpretation
A2                      0.121914         n/a                   generic shared capacity (reads 146D shell)
B-bag                   0.123806         not measured          shared bag over raw primitives, no connectivity
B-full                  0.118972         ~1.1-1.2              connectivity-aware, but functionally scalar
B-full rank1 frozen     0.118988         exactly 1             scalar modulation suffices
```

**Final methodological conclusion.** The `+0.00294` B-full gain over A2 is *not*
produced by shared low-dimensional composition: a shared bag encoder over the
same raw primitives is on average **worse** than the generic shared adapter
(composition axis inconclusive, sign-flipping). The reproducible component is
**explicit rooted-patch connectivity**: B-full beats B-bag by `+0.004833`
(same inputs, same budget, both seeds same direction). But the connectivity is
exploited through a **functionally rank-1 (scalar) channel**, not a rich
explicit structural code.

## 10. Limitations / honest scope

- 2 seeds only (no extra seeds purchased); B-bag soup per-seed spread (0.0072)
  is larger than the deltas; the composition axis cannot be resolved.
- A2 vs B-bag mixes input information with the composition mechanism (A2 reads
  the 146-D descriptor; B-bag reads raw primitives).
- B-bag `e_bag` effective rank was **not measured** (cheap follow-up, not run).
- Connectivity is entangled with "having two rounds of learned message
  passing" vs "having none"; that is exactly the pre-registered variable, but it
  is not "same encoder plus an edge flag".
- Frozen rank-1 reconstruction is a mechanism diagnostic, never a model
  selection device.

## 11. Stop rules honoured

No new seeds; no width/depth/rounds/pooling sweep; no B-bag width sweep (one
pre-registered architecture); no MolHIV training; no ZINC official test; no
attention/Transformer; no identity lookup / hash / KNN / SBCI; no h/q/T change;
no optimizer/seed sweep; no scalar-model design.

## 12. Provenance & artifacts

- Code: `structural_patch_encoder.py` (`SharedBagPatchEncoder`),
  `zinc_patch_path_pooling.py` (`patch_representation="shared_bag"`),
  `zinc_shared_bag_patch_encoder.py`, `zinc_structural_encoder_rank1_audit.py`.
- Tests: `tests/test_shared_bag_patch_encoder.py` (13) +
  `tests/test_structural_encoder_rank1_audit.py` (6); full ksvd suite 888 pass.
- Results: `results/shared_bag_patch_encoder/` (params, sanity, curves, runs,
  soup, diagnostics, decision, decomposition) +
  `results/structural_encoder_rank1_audit/` (collect, spectrum, correlations,
  functional, report).
- Determinism: `repro_smoke0/1_seed0.json` selection-state SHA-256 identical.
- `official_test_loaded = false` in every artifact.
