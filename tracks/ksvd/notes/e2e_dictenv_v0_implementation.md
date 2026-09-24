# E2E-DictEnv-v0 — implementation

Round `e2e_dictenv_v0` · protocol `e2e_dictenv_v0` · study `zinc-context-gap`
Frozen intent: `notes/e2e_dictenv_v0_preregistration.md`
Prior-artifact audit: `notes/e2e_dictenv_v0_prior_artifact_audit.md`

> **Outcome at a glance:** Gate 0 (G0..G12) **PASS**; Stage-1 mechanism smoke
> failed its `atoms_active` sub-gate (23/32 < 24/32) and was treated as passed
> under explicit user authorisation (Amendment A1); both formal arms, the
> dictionary-health and the zero/shuffle interventions ran. Frozen verdict
> **`E2E_DICTENV_ABSOLUTE_WEAK`** (`M_S` = 0.145508 > 0.145). See
> `notes/e2e_dictenv_v0_analysis.md`.

## 1. Files

| role | path |
|---|---|
| core primitives (model, IHT, incidence) | `tracks/ksvd/experiments/luyin16/e2e_dictenv_v0.py` |
| runner (stages, gates, training, analysis) | `tracks/ksvd/experiments/luyin16/zinc_e2e_dictenv_v0.py` |
| focused CPU tests | `tracks/ksvd/tests/test_e2e_dictenv_v0.py` |
| results | `tracks/ksvd/results/e2e_dictenv_v0/` |

## 2. Architecture (frozen)

Only coordinate: the shared pure-topology FSAR `phi65`. There is **no** message
passing, recurrence, pair→centre, relation refresh, attention, context
writeback, typed/parent token lookup, FEC-S1 `146→214→24` adapter, `patch_cont`
in the model, graph-level dictionary residual or pair-kernel dictionary side
channel.

Two arms, both **66,158** parameters, differing only in the tied coding operator
applied to the *same* `D ∈ R^{65×32}` parameter:

* **S — SparseDictEnv**: `alpha = IHT(Dbar, phi)`, exact top-8, 10 unrolled
  steps, `eta = 1/(sigma²+eps)`, `sigma` from the frozen deterministic power
  iteration (`tccd_v0.power_iter_sigma`). Row-equivariant ⇒ relabel invariant.
* **D — DenseTiedEnv**: `z = phi @ Dbar` (dense, same matrix).

`D` is initialised bit-exactly from the frozen SDB K-SVD artifact
(`results/sdb_v0/dictionary.pt`, sha256
`925d573a58083c3c20f36980dec9bba77dff592b30ea1abf11098ca3a7b7487a`, 65×32);
the same matrix initialises `D` and (dense arm) `P`. It is read-only with
respect to the artifact (never refit offline).

### Parameter accounting (re-derived by code, `parameter_accounting.json`)

```text
D = 65·32                                 2 080
W_R = (1+32)·64                           2 112
W_C = 28·64                                1 792
S   = 3·64                                  192
W_B = 4·16                                   64
P_shell = 6·16                               96
environment MLP 86→329→48                44 463
local total                              50 799
strict-static composition backend         15 359
TOTAL                                     66 158
```

FEC-S1 historical 66,170 → difference **−12** (0.018%). Matched by construction.

### Local environment (`E_i ∈ R^48`)

For each root node `i`, occurrences `v ∈ P_i` grouped by shell `s_iv ∈ {0,1,2}`
(BFS from `i`):

```text
r_v   = [1 ; coord_v]                        (33 = 1 + 32)
u_iv  = (r_v W_R) ⊙ (q_v W_C) ⊙ S_{s_iv} / sqrt(64)
m_V   = Σ_{v∈P_i} u_iv                       (SUM via index_add over env_occ_root)
w_ie  = (b_e W_B) ⊙ P_{p_ie} / sqrt(16)
m_E   = Σ_{e∈P_i} w_ie                       (SUM)
z_i   = [ m_V(64) ; m_E(16) ; t_i(6) ]       (86)
E_i   = MLP(z_i)                             86→329→48, SiLU
```

`t_i` = last-6 coordinates of the standardised `patch_cont[:, -6:]` (exact
historical FEC/S1 six-scalar semantics). Chemistry enters only through `q_v`
and the bond type; `phi65` and the incidence are topology-only.

### Static composition backend (exact FEC-S1 semantics, 15,359 params)

`u = W_P E` (48→16); pair input
`[u_src+u_tgt ; |u_src−u_tgt| ; u_src*u_tgt*gate(bucket) ; relation]`;
`gate = 1+tanh(distance_gate(bucket))`. Pooling is the FEC-S1 `_pool_values`
moment scheme: unary moments (97) + per-bucket pair moments (5 buckets × 33 =
165) + global encoder (62→32→32) + topology encoder (25→16→8) = 302 →
`GenericReader` 302→13→13→1. Dropout 0.05 (FEC-S1 default).

## 3. Environment incidence cache

Rooted, root-relative occurrence indices (BFS shells 0,1,2), built once from
the official-train/valid raw graphs (`zlr._data_to_graph` + `fec` shell logic),
aligned to the FSAR/encoded cache with per-molecule asserts
(`n_nodes`, atom order `data.y == molecule.y`, target match).

| split | molecules | nodes | node occurrences | bond occurrences |
|---|---:|---:|---:|---:|
| official train | 10 000 | 231 664 | 1 418 500 | 1 232 844 |
| official valid | 1 000 | 23 083 | 141 287 | 122 934 |

Official test never touched (env cache is built only from `train`/`val`).

## 4. λ_rec calibration (one shot, frozen)

First 512 official-train molecules, Sparse init (seed 0), no valid/test read:

```text
L_task_init = 1.3659378290176392     (mean per-molecule MAE)
L_rec_init  = 0.01005586596037212    (mean per-node normalised reconstruction)
λ_rec       = L_task_init / (L_rec_init + eps) = 135.834928
```

Frozen for **both** arms; `lambda_calibration.json`.

## 5. Gate 0 — correctness G0..G12 (`correctness.json`, all PASS)

| gate | result |
|---|---|
| G0 `phi65` bit-identity (fresh `build_phi` vs FSAR vs env cache) | PASS, max_abs = 0 |
| G1 chemistry purity (topology fixed, atom/bond categories relabelled) | PASS, Δphi = Δalpha = Δz = 0 |
| G2 exact sparsity `l0 ≤ 8`, exact-8 fraction ≈ 1 | PASS |
| G3 gradient to `D` from MAE alone (and from rec) | PASS, both > 0 |
| G4 tied reconstruction, perturb `D` ⇒ code & reconstruction change; no independent encoder | PASS |
| G5 no local dense bypass (`coord_zero` E independent of `phi`; chemistry still matters) | PASS |
| G6 environment freeze under pair-relation mutation | PASS, E bit-identical |
| G7 no pair→centre (monkeypatched to raise) | PASS, 0 calls |
| G8 once-only composition (pair/relation/env encoders exactly 1 call) | PASS |
| G9 relabel invariance (incidence permutation-consistent; pred Δ ≤ 1.2e-7) | PASS |
| G10 parameter identity Sparse = Dense = 66,158 | PASS |
| G11 init matching, all 40 tensors bit-identical, `D` sha shared | PASS |
| G12 official-test blocker + cache audit (10 000/1 000, test not loaded) | PASS |

Focused CPU test suite `tests/test_e2e_dictenv_v0.py`: **15/15 pass**.

## 6. Stage-1 mechanism smoke (`smoke_gate.json`) — **FAIL**

Fixed smoke: first 512 official-train molecules, 3 epochs, Sparse arm, seed 0,
official valid not read, λ_rec frozen. The dictionary-coverage gate is measured
on the official-train distribution (project precedent: SDB-v0's `used >= 24`
was taken over all 231,664 train atoms); the 512-molecule value is retained as a
diagnostic.

| check | value | verdict |
|---|---|---|
| loss finite | yes | PASS |
| train MAE decreases | 1.34700 → 1.24918 | PASS |
| `grad_D != 0` | 4.00 / 2.72 / 1.89 | PASS |
| alpha still sparse (`max l0 = 8`) | 8 | PASS |
| **atoms active ≥ 24/32** | **23/32 (train), 23/32 (512 subset)** | **FAIL** |
| no single atom > 50 % | top-1 0.125 | PASS |
| effective atom count ≥ 8 | 12.83 | PASS |
| environment effective rank > 1 | 2.297 | PASS |

Diagnosis (deterministic, reproduced twice; only float-reduction noise differs):

* at **initialization** the frozen K-SVD dictionary covers **25/32** atoms on
  official train (**24/32** on the 512-molecule subset) — the mechanism is
  intact;
* after the 3-epoch train-only smoke the rare-atom support is pruned to
  **23/32** under *both* measurement scopes; exact top-8 support is preserved
  (`max l0 = 8`), no atom dominates, `E` keeps rank > 1.

The failure is a genuine, deterministic miss of the frozen threshold. It is a
coverage artefact of the 512-molecule/3-epoch smoke under the calibrated
λ_rec = 135.8, **not** a loss/reconstruction collapse.

**Amendment A1** (user-authorised) treats this sub-gate as passed; the formal
run then proceeded, and the fully trained Sparse soup reaches **27/32** active
atoms (train and valid), confirming that the smoke value was a transient.

## 7. Formal arms (`sparse_seed0.json`, `dense_tied_seed0.json`)

Remote A100-SXM4-40GB GPU1, seed 0, 240 epochs, no early termination, fixed
Top-5 soup, commit `eeeb6b34f641260a0373a3daeda1286a855717d7`.

| | SparseDictEnv | DenseTiedEnv |
|---|---:|---:|
| best valid MAE | 0.150982 @ 228 | 0.317686 @ 146 |
| **Top-5 soup valid MAE** | **0.145508** | **0.313047** |
| soup members | 223, 228, 234, 235, 236 | 146, 156, 170, 185, 212 |
| train MAE at best | 0.115179 | 0.258287 |
| train minimum MAE | 0.111971 | 0.237859 |
| valid normalised reconstruction | 9.7e-5 | 8.0e-6 |
| `‖D̄_soup − D̄_init‖_F` | 5.941 | 4.342 |
| wall clock | 1586.7 s | 1272.1 s |
| peak GPU memory | 168.3 MB | 155.7 MB |

`G_sparse = M_D − M_S = +0.167539` (gate ≥ 0.003, PASS).

### Interventions and health

* zero-code (α → 0, DC/clχ/bond/scalars/backend kept): `M_zero` = 1.000347,
  `G_dict-use` = +0.854839 (gate ≥ 0.010, PASS); mean prediction shift 0.981.
* assignment shuffle (5 perms within (root, shell)): `M_shuffle` = 0.162797,
  `G_assign` = +0.017289 (gate ≥ 0.010, PASS); seeds 101/202/303/404/505 give
  0.16411/0.16314/0.15954/0.16458/0.16262.
* dictionary health: **PASS** — 27/32 active (train & valid), effective atom
  count 13.89, top-1 share 0.1250, exact top-8 fraction 1.0, usage Spearman
  0.998, effective rank 10.83, reconstruction 9.7e-5, task gradient to `D`
  0.0874.

Frozen verdict: **`E2E_DICTENV_ABSOLUTE_WEAK`** (case D, `M_S` > 0.145).

## 8. Provenance

* commit at run time: see `artifact_identity.json` / `decision.json`
  `git_commit`;
* `official_test_loaded = false` in every artifact;
* JSON/PT/CSV under `results/e2e_dictenv_v0/` follow the repo-wide gitignore
  convention (no force-add).
