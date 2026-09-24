# E2E-DictEnv-P1 — prior-artifact audit

Round **E2E-DictEnv-P1** · protocol `e2e_dictenv_p1` · study `zinc-context-gap`.
Pre-registration: [`e2e_dictenv_p1_preregistration.md`](e2e_dictenv_p1_preregistration.md).

This audit is written **before** any P1 implementation run.  It fixes exactly
which historical artifacts P1 reuses, their identity, and which historical
builders are used **only as correctness references** and are never fed to the
P1 model.

P1 follows the T1 negative result: T1 (`E2E_DICTENV_TUNED_MECHANISM_LOST`)
showed that the moment the exact FEC-S0 146-D coarse descriptor
(`patch_cont`) is exposed to the environment decoder, the dictionary stops
being load-bearing (`G_dict-use` fell from v0's `+0.8548` to `+0.0179`,
`G_assign` from `+0.0173` to `+0.0076`).  P1 removes that bypass and all other
handcrafted structural-chemistry tensors, keeping only raw zeroth-order
primitives plus the dictionary-mediated role × chemistry assignment.

---

## 1. Reused frozen artifacts (identity)

| artifact | path | identity | P1 use |
|---|---|---|---|
| K-SVD dictionary | `results/sdb_v0/dictionary.pt` | `D` 65×32, file sha256 `925d573a58083c3c20f36980dec9bba77dff592b30ea1abf11098ca3a7b7487a`, `DICT_SEED 20260924`, 10 K-SVD epochs on 231,664 train atoms | `D` initialisation and tied encode/reconstruct; never refit |
| pure-topology coordinate | `fsar_r2_ar0.build_phi` | `phi_v in R^65` (`PHI_DIM = 65`), chemistry-free, FSAR-R2-AR0 | per-node dictionary input |
| SDB coding solver | `sdb_v0.omp_codes` → `tccd_v0.omp_codes` | exact top-`s` | audit reference for the code |
| FEC-S1 strict-static backend | `e2e_dictenv_v0.backend_parameter_count` / `zinc_patch_path_pooling._MLPBlock` | 15,359 params at 23-D relation, 15,103 params at P1's 15-D relation | pair composer / global / topology / reader |
| ZINC encoded cache | `results/zinc_static_dictionary_pair/cache/encoded_{train,valid}.pt` | `n_train = 10000`, `n_valid = 1000`, `official_test_loaded = false` | targets, `pair_relation` (23-D), `global_context`, `topology_features` |

`D` is loaded by `zinc_sdb_v0.load_dictionary()`; `e2e_dictenv_p1.P1Model`
copies the float32 tensor verbatim.  Local identity gate `G1` re-checks the
shape, the file sha256, the tensor bytes and the column-normalised norms
(`||Dbar_k||_2 = 1`).

---

## 2. `phi_v` — exact pure-topology reuse

`G0` rebuilds `phi_v` from raw ZINC primitives with `fsar_r2_ar0.build_phi` and
compares it bit-for-bit to the P1 cache value.  `build_phi` depends only on the
untyped adjacency (no atom category, no bond category, no target, no learned
local state), so a single molecular node `v` has one `phi_v` used by every root
patch that contains it.  Local CPU `G0`: 3/3 molecules bit-identical, max abs
difference `0.0`.

---

## 3. `phi65 → D → alpha32` coding

Identical to v0/T1/SDB:

```text
Dbar_k         = D_k / max(||D_k||_2, eps)          (column normalisation)
alpha_v        = IHT_{s=8}(Dbar, phi_v), 10 tied-IHT steps, exact top-8
hat_phi_v      = alpha_v Dbar^T                     (tied reconstruction)
L_rec          = E_v ||phi_v - hat_phi_v||^2 / (||phi_v||^2 + eps)
```

`K = 32`, `s = 8`, `IHT_STEPS = 10`; `D` is a `requires_grad=True` parameter.
The DenseTied control uses `z_v = phi_v Dbar` with `P_init == D_init`
bit-identical.

Local gates: `G2` exact sparsity (`max l0 ≤ 8`, exact-8 fraction ≥ 0.99),
`G3` a real MAE backward on a train batch gives `||∂L_task/∂D|| > 0` with the
reconstruction term switched off.

---

## 4. Pair relation — exact historical layout and the P1 slice

The 23-D `pair_relation` is the historical FEC-S0 factorized descriptor:

| index | meaning | chemistry? |
|---|---|---|
| 0–4 | distance bucket one-hot (5) | no (topology) |
| 5 | `log1p(distance)` | no |
| 6–10 | patch overlap (5) | no |
| 11–13 | boundary overlap (3) | no |
| **14–17** | **path bond mean (4)** | **chemistry** |
| 18 | `log1p(path count)` | no |
| **19–22** | **adjacent bond type (4)** | **chemistry** |

P1 uses **only** the pure-topology coordinates

```text
P1_RELATION_INDICES = 0,1,2,3,4,5,6,7,8,9,10,11,12,13,18     (15 = 0:14 + 18)
```

so the pair relation carries **no bond chemistry**.  `G8` checks that the
slice of a chemistry-mutated relation (indices 14–17 and 19–22 randomised) is
bit-identical to the clean slice and that the prediction is unchanged, and the
AST audit confirms the model never names `path_bond_mean` / `adjacent_bond_type`.

---

## 5. Root-relative environment incidence (topology only)

`e2e_dictenv_v0.env_incidence` computes, for every root `i`:

```text
occ_node / occ_root / occ_shell      node occurrences (i,v) and shell s_iv in {0,1,2}
bond_root / bond_shellpair / bond_type   induced-patch edge occurrences and 6-class shellpair
bond_u / bond_v                      endpoint node indices (additive P1 extension)
```

It iterates only the untyped adjacency and the primitive bond category.
`G4` relabels the bond chemistry on the same topology and confirms
`occ_node/occ_root/occ_shell/bond_u/bond_v` are unchanged, while `phi` and
`alpha` are invariant.

---

## 6. Historical builders used as correctness references only

The following are used **only** by the audit/correctness stages and are never
an input to any P1 learned module:

* `fec_s0_factorization.FactorizedFeatureTransform` / `patch_cont` (146-D) —
  **forbidden** inside P1; referenced only historically.
* `atom_shell` (84-D) / `bond_shell` (24-D) — **forbidden**; the AST gate
  confirms the model never reads them.
* FEC-S1 `146 → 214 → 24` adapter, PCA32 side branch, DenseRole side branch,
  FEC-D1 rank-8 residual, B-Null/B-Full local hidden state — all **forbidden**.

The cache stores only raw primitives (`phi`, atom category, incidence fields,
bond category) and the derived 62-D primitive anchor, plus the target.

---

## 7. What is new in P1

1. A **62-D zeroth-order primitive anchor** built directly from raw primitives:
   `root one-hot (28) | patch atom mass (28) | patch bond mass (4) |
   [log1p|V_i|, log1p|E_i|] (2)`.  It is *unconditioned*: the mass sums run
   over the whole patch and the shell is never concatenated into it.
2. A **node dictionary-role × chemistry binding**
   `u_v^A = (alpha_v W_A^S) ⊙ (q_v W_A^C) / sqrt(96)`, summed per shell
   (`A_i in R^288`).
3. An **edge dictionary-derived role × bond-chemistry binding**
   `g_uv = [alpha_u+alpha_v; |alpha_u-alpha_v|; alpha_u⊙alpha_v]`,
   `u_uv^E = (g_uv W_E^S) ⊙ (b_uv W_E^C) / sqrt(48)`, summed per shellpair
   (`E_i^edge in R^288`).

Shell and shellpair are **routing indices only**; they never enter the decoder
as one-hot chemistry features.

---

## 8. Exact P1 parameter accounting (code-derived)

```text
dictionary        65 x 32                          =  2080
node binding      32x96 + 28x96                    =  5760
edge binding      96x48 + 4x48                     =  4800
decoder           638x102+102 + 102x48+48          = 70122
------------------------------------------------------------
local                                              82762
backend (15-D relation)                            15103
------------------------------------------------------------
whole model                                        97865
```

`97865 ∈ [90000, 105000]`.  The model's `sum(p.numel())` equals this exactly
(gate `G13`).

---

## 9. Test-status discipline (frozen)

```text
official train = 10000    (training)
official valid = 1000     (checkpoint / soup / mechanism / gate selection)
official test  = reporting only, loaded once after the freeze
```

Official test is **not** project-wide pristine (historical unrelated rounds
already opened it), but no P1 decision may use it.  `G14` re-checks the guarded
loader and the encoded-cache audit flag.
