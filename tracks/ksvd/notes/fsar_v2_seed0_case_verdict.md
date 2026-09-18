# FSAR-v2 seed0 case verdict — exact marginals do NOT remove the A starvation (Case D)

Branch `exp/fsar-v2-rich-marginal-explicit-structure-zinc`.
Pre-registration: `notes/fsar_v2_preregistration.md`.
Training commit `81f70da` (architecture, runner, tests); eval-only commit
`56ae3c3` (added `--state {best,soup}` to the frozen diagnostics / witness /
interventions; no architecture or training change).
Official ZINC test never loaded (`official_test_loaded = false` in every JSON).

## 1. Branch / commits

* branch `exp/fsar-v2-rich-marginal-explicit-structure-zinc` from the FSAR-v1
  verdict HEAD `af8d5c2`;
* `81f70da` — `fsar_v2.py`, `zinc_fsar_v2.py`, `tests/test_fsar_v2.py`,
  pre-registration note (all formal training used this revision);
* `56ae3c3` — eval-only `--state` addition for the §16 frozen interventions.

## 2. GPU provenance and parallel schedule

| item | value |
|---|---|
| GPU model | NVIDIA A100-SXM4-40GB |
| CUDA / PyTorch | 12.4 / 2.5.1+cu124, Python 3.12.14 |
| determinism | `torch.use_deterministic_algorithms(True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8` |
| GPU 0 | occupied by an **unknown** task (35,803 MiB) for the whole session — never touched, never co-tenanted |
| GPU 1 | free (587 MiB) — all FSAR-v2 jobs |
| Wave 1 | `A seed0` + `SA seed0` in parallel on GPU 1 |
| Wave 2 | `SAB seed0` + `SAM seed0` in parallel on GPU 1 |
| Wave 3/4 | **not launched** (explicit gate closed, see §12) |

Two of our own independent jobs were run concurrently on GPU 1 under the user's
explicit authorization ("在1上并行"). No DDP; one process per job; no
architecture/protocol difference between the two parallel jobs; each job has its
own log / pid / metadata / result directory. Peak GPU memory: SAB 459.6 MB,
SAM 341.8 MB, A 258.8 MB, SA 338.8 MB.

Peak training provenance (commit `81f70da`):

| run | best valid | best ep | epochs | wall (s) | peak MB |
|---|---:|---:|---:|---:|---:|
| A | 0.168621 | 237 | 240 | 2393.6 | 258.8 |
| SA | 0.167918 | 201 | 240 | 2807.7 | 338.8 |
| SAB | 0.160203 | 236 | 240 | 3226.6 | 459.6 |
| SAM | 0.175238 | 165 | 205 | 2479.8 | 341.8 |

## 3. Exact ZINC attribute schema (audited, `schema_audit`)

`x` is a single categorical field (`x.shape = [n, 1]`), `edge_attr` a single
categorical field. Encoder schema (inherited) is padded to **28 atom / 4 bond**
categories; observed values are atom `0..20` (train) / `0..15` (valid) and bond
`1..3` (both). The runner refuses to run if a processed split file is missing,
because PyG `process()` would read the test split. No multi-field atom/bond
tuple exists, so the "complete attribute tuple multiset" equals the category
count vector.

## 4. Why `A_exact` is lossless for the defined marginal

```
raw_v = [ root_one_hot(x_v) (28) | context_atom_counts (28) | bond_counts (4) ]   -> 60-D
A_v   = MLP_A(raw_v): Linear(60,64) -> SiLU -> Linear(64,64)                      -> 64-D
```

`raw_v` is built from integer categories only. Two frames have identical
`raw_v` iff their root category and their context-atom / bond category multisets
are identical. `A_exact` reads only `struct_atom`, `struct_root`, `struct_patch`,
`struct_bond`, `struct_edge_patch`; it never reads `struct_dist`, `struct_src`,
`struct_dst`, degree, shell, role or any adjacency propagation
(`a_exact_is_blind_to_topology_fields`).

## 5. `A_exact` collision audit (`collision_audit`)

200 sampled train+valid molecules → 4,645 frames, **1,148 unique marginal keys,
1,148 unique raw vectors, 0 collisions, 0 same-key/different-raw mismatches**
→ `lossless_for_defined_marginal = true`.

## 6. Rich-A seed0 results (deterministic A100, Top-5 soup, valid MAE)

| mode | S | B | params | best valid | best ep | soup |
|---|---|---|---:|---:|---:|---:|
| A | – | – | 68,281 | 0.168621 | 237 | **0.165132** |
| SA | implicit | – | 89,273 | 0.167918 | 201 | **0.167141** |
| SAB | implicit | implicit | 106,873 | 0.160203 | 236 | **0.154115** |
| SAM | implicit | – (+M) | 106,867 | 0.175238 | 165 | **0.170986** |

`SAM - SAB = -6` params, `SAME - SABE = -20` (< 0.1 %).

## 7. Old-vs-rich A improvement (`ΔA = A_exact - A_weak`)

| mode | FSAR-v1 soup | FSAR-v2 soup | Δ (v2 − v1) |
|---|---:|---:|---:|
| A | 0.157917 | 0.165132 | **+0.007215** |
| SA | 0.151617 | 0.167141 | +0.015524 |
| SAB | 0.130913 | 0.154115 | +0.023202 |
| SAM | 0.134182 | 0.170986 | +0.036804 |

`ΔA = +0.007215` (improvement = **−0.007215**). Pre-registered §13 classes:
strong `<= 0.135`, partial `0.135 < A <= 0.145` **and** improvement `>= 0.010`,
otherwise failure. `A_exact = 0.165132 > 0.145` and the improvement is negative
→ **failure**. Indeed the entire nested family is *worse* than FSAR-v1, so the
`A_exact` substitution did not merely fail to fix the starvation — it degraded
the factorized backbone.

## 8. `SAB` vs `SAM` aligned residual

* `SA → SAB` = 0.013027 (bigger than FSAR-v1's 0.020705 in a worse regime)
* `SA → SAM` = 0.167141 − 0.170986 = **−0.003845** (the marginal-only capacity
  control is *worse* than SA)
* `SAM − SAB` = **0.016871** `>= 0.001` → `capacity_confound_excluded` for the
  seed0 aligned channel.

The aligned residual is larger than FSAR-v1's 0.00327, but the whole model is
far weaker, so this is a seed0 increment inside a degraded regime, not evidence
that the factorized rich model is competitive.

## 9. Fixed-centre shuffle witness (eval-only)

Same topology, same centre attribute, same context-atom and bond multisets,
only attribute↔structural-role assignment and bond-attribute↔edge-role
assignment change:

| state | mean abs Δ A | mean abs Δ S | mean abs Δ B | mean abs Δ pred |
|---|---:|---:|---:|---:|
| SAB best | 0.0 | 0.0 | 0.1562 | 0.5083 |
| SAB soup | 0.0 | 0.0 | 0.1560 | 0.5008 |

`A_stable` and `S_stable` are exact to float tolerance; `B_responds` is true.
This is the required clean binding witness.

## 10. Frozen no-B / shuffle-B / no-S / no-A interventions (full 1,000 valid)

| state | true | no_A | no_S | no_B | shuffle attrs |
|---|---:|---:|---:|---:|---:|
| SAB best (0.160203) | 0.160203 | 0.996270 (+0.8361) | 0.187569 (+0.0274) | 0.919977 (+0.7598) | 0.530450 (+0.3702) |
| SAB soup (0.154115) | 0.154115 | 0.965266 (+0.8112) | 0.185706 (+0.0316) | 0.956320 (+0.8022) | 0.495564 (+0.3414) |

B is functionally necessary; the shuffled-B / shuffled-assignment prediction is
much worse than true. S contributes only ~0.03 MAE.

## 11. Information-resolution confound — removed

`B` reads only `E_atom = atom_mlp(atom_embedding(struct_atom))` and
`E_bond = bond_mlp(bond_embedding(struct_bond))`; `A_exact`'s raw counts are
over exactly the categories indexing those embeddings, so the full multiset of
B's attribute-side entity embeddings is recoverable from `A_exact`
(`a_exact_reconstructs_full_attribute_multiset` + collision audit). The only
principled extra information B has is **which attribute belongs to which
structural role**. Verified: no raw chemical field reaches B outside `A_exact`.
**Pass** — the FSAR-v1 information-resolution confound is removed, and the
A starvation still does not disappear (indeed it worsens), which isolates the
problem to the relational interface / representation–optimization regime, not
to mean/std marginal information loss.

## 12. Explicit-S gate — CLOSED

Pre-registered gate: `A_exact <= 0.145` **and** (`SAB_exact <= 0.132` **or**
`SAM - SAB >= 0.001`). `A_exact = 0.165132 > 0.145` → `condition_A = false`,
`explicit_stage_open = false`. Per the pre-registration, no explicit-S formal
run (Wave 3/4) is launched; the explicit basis stays implemented and
unit-tested only.

## 13. Explicit structural basis — exact frozen definition

Node basis (11-D, chemistry-free, `log1p` on integer counts):
`1[u=v]`; shell one-hot for `d(v,u)=0,1,2`; `log1p(indeg_{W_v}(u))`;
`log1p(|N(u) ∩ shell_j(v)|)` for `j=0,1,2`; `log1p((A)_{vu})`,
`log1p((A^2)_{vu})`, `log1p((A^3)_{vu})`.

Edge basis (15-D, chemistry-free): one-hot unordered shell pair over 6 pairs;
`log1p(deg(u)+deg(w))`; `log1p|deg(u)-deg(w)|`; `log1p(common neighbours)`;
`log1p(n_j(u)+n_j(w))` and `log1p|n_j(u)-n_j(w)|` for `j=0,1,2`.

`S_explicit = F_S(φ_vv, mean_u φ_vu, std_u φ_vu, mean_e φ^E, std_e φ^E)`:
`Linear(63,32) → SiLU → Linear(32,32)`, no adjacency message passing.
`B_explicit` is the same centred aligned interaction with the explicit
structural side. Verified by `explicit_basis_chemistry_purity`,
`explicit_basis_relabel_equivariance`, `explicit_path_has_no_message_passing`.

## 14–16. Explicit seed0 / explicit SAM / implicit-vs-explicit cost

**Not run.** Wave 3 (`SAE`, `SABE`) was not authorized by the §12 gate, so
Wave 4 (`SAME`) was not triggered and the implicit-vs-explicit performance cost
is unmeasured in this round.

## 17. Seed1

**Not authorized.** `seed1_authorized_implicit = false` (A failure + explicit
gate closed). No seed1 runs.

## 18. Matched external references (frozen, never used as input)

| reference | seed0 soup |
|---|---:|
| B-Bag | 0.127382 |
| A2 / cell-A | 0.121694 |
| B-Full | 0.119818 |
| FSAR-v1 A / SA / SAB / SAM | 0.157917 / 0.151617 / 0.130913 / 0.134182 |

`SAB_exact = 0.154115` is +0.026733 / +0.032421 / +0.034297 above B-Bag / A2 /
B-Full: the rich-A factorized backbone is not competitive with the strong mixed
backbone.

## 19. Case verdict — **Case D** (strengthened)

> The A problem is **not** mean/std marginal information loss.
> `A_exact > 0.145` and the improvement is negative, so the pre-registered
> failure branch fires. The information-resolution confound is removed, the
> exact marginal is collision-free, the B channel is functionally used and even
> shows a larger seed0 aligned increment (0.016871) than FSAR-v1 (0.00327), yet
> every nested mode is *worse* than the weak-marginal FSAR-v1. Per the
> pre-registration: **STOP** adding attribute-encoder capacity (no
> DeepSets/attention/quantile/width sweep); re-audit the relational interface /
> root-centric formulation before any further attribute encoder work.

Do not read this as "the SAB ≈ 0.13 route is a failure": FSAR-v1 already
established that a bypass-free factorized backbone is trainable and that aligned
S–A access carries a seed0 signal beyond a matched marginal control. FSAR-v2
adds the negative result that completeness of the attribute marginal is **not**
the bottleneck, and that the aligned binding signal survives in the richer
representation.

## 20. Durable record

* this note;
* claim `records/claims/claim-fsar-v2-exact-marginal-no-starvation-fix-20260917.yaml`;
* decision `records/decisions/decision-fsar-v2-exact-marginal-stop-20260917.yaml`;
* `tracks/ksvd/STATE.yaml` `recent_decisions` entry.

## 21. `official_test_loaded = false`

Every produced JSON records `official_test_loaded = false`; no test split is
read by any FSAR-v2 stage (the runner hard-fails if a processed train/val file
is missing rather than letting PyG `process()` read the test split).

## Provenance / caveats

* Single seed (seed0). Per the pre-registered failure branch, no seed1 is
  bought.
* Two of our own jobs shared GPU 1 under explicit user authorization. Because
  all runs are deterministic and one process = one job, contention changes wall
  time, not the computed values.
* Training used commit `81f70da`; the frozen eval stages used the additive
  `56ae3c3`. No architecture or protocol changed between them.
