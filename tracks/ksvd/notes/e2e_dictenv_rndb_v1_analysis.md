# Analysis — `e2e_dictenv_rndb_v1` (Rolewise Nonlinear Dictionary Binding)

Round: **E2E-DictEnv-RNDB-v1** (`e2e_dictenv_rndb_v1`), study `zinc-context-gap`.
Pre-registration: [`e2e_dictenv_rndb_v1_preregistration.md`](e2e_dictenv_rndb_v1_preregistration.md)
Prior-artifact audit: [`e2e_dictenv_rndb_v1_prior_artifact_audit.md`](e2e_dictenv_rndb_v1_prior_artifact_audit.md)

Verdict: **Case F — `RNDB_NO_GO_TASK_LEVEL_MECHANISM_STRONGLY_SUPPORTED`**
(boundary outcome; see §7).

Starting HEAD `cfe0716f3c4ac3c72df9521c9e075092b5213ac3`; implementation commit
`e233beb`. CPU only, `official_test_loaded = false` everywhere, exactly one
seed-0 RNDB trajectory.

---

## 0. Question

Does moving one small shared nonlinear operator between the individual residual
dictionary-role responses and their aggregation
(`rolewise product -> shared nonlinearity -> role sum`, instead of the parent's
`rolewise product -> role sum -> later nonlinearity`) improve ZINC valid MAE,
while keeping the sparse dictionary genuinely behaviourally load-bearing?

Single hypothesis, falsification test. No baseline rerun, no sweep, one seed.

---

## 1. What stayed frozen

Parent object `CSSD-q1` (`CSSDModel`, commit `5f0f284`, later only the additive
behaviour-neutral `model_factory` hook), `cm.H1_CONFIG`
(`d_e=48`, `K=32`, `s=8`, `lambda=33.95873017865987`, `h=320`, `sdb32`),
`cm.C6_MASK`, train-only common subspace q1 (`rms=[5.082852828320509]`),
residual dictionary `Dbar_perp`, IHT-10, node/edge shell definitions, pair
relation, distance bucket, anchor, global branch, topology branch, moment
pooling, reader, Adam `lr=1e-3`, `weight_decay=1e-5`, batch 128, grad clip 5.0,
Top-5 soup protocol. Nothing was retuned.

The parent parameters are preserved **bit-for-bit**: of the shared state-dict
keys, all are `torch.equal` to `cssd.build_cssd_model`; the only new keys are
`psi_A.0.weight`, `psi_A.2.weight`, `psi_E.0.weight`, `psi_E.2.weight`.

---

## 2. New architecture (exactly as pre-registered)

```
p_{v,k}    = z_{v,k} (W_A_S[k,:] odot c_v),   c_v = (q_v W_A_C)/sqrt(D_A)
u_v        = p_v^common + sum_{k in K_dict} [ p_{v,k} + psi_A(p_{v,k}) ]
p^E_{uv,k} = r_{uv,k} odot c^E_{uv},  r from [+, |.|, odot] blocks
u^E_{uv}   = p^E,common_uv + sum_{k in K_dict} [ p^E_{uv,k} + psi_E(p^E_{uv,k}) ]
psi_A      = Linear(96,32,bias=False) -> SiLU -> Linear(32,96,bias=False)
psi_E      = Linear(48,16,bias=False) -> SiLU -> Linear(16,48,bias=False)
K_common   = {0};  K_dict = {1..32}
```

`psi` is shared across all roles, nodes/bonds and shells/shellpairs. `W1` is
PyTorch default Kaiming-uniform; `W2 ~ N(0, 0.01)`; no bias (strict
`p_k = 0 => psi(p_k) = 0`). Implementation:
`tracks/ksvd/experiments/luyin16/e2e_dictenv_rndb_v1.py`; runner
`tracks/ksvd/experiments/luyin16/zinc_e2e_dictenv_rndb_v1.py`.

Not a re-typed copy: `RNDBModel` subclasses `CSSDModel` and calls the parent's
shell routing / index-add / encoders / C6 masking; only
`environments_masked` (and `code`-adjacent helpers) are overridden.

---

## 3. Parameter budget

| item | value |
|---|---|
| parent (CSSD-q1) | 97727 |
| RNDB total | 105407 |
| new | 7680 (`psi_A` 6144 + `psi_E` 1536) |
| relative increase | 0.07858 (cap 1.10) |

Recomputed from the code in `parameter_audit.json`. Within the frozen cap.

---

## 4. Correctness gates G0–G6 (all PASS)

Recorded in `correctness.json`:

* **G0** parent decomposition identity, node and edge, synthetic fixture and a
  real ZINC train batch: `sum_k p_k == p_parent`, `max_abs_diff` ≤ 4.3e-19
  (node) / 3.5e-18 (edge) in float64.
* **G1** psi-off exact parent equivalence: prediction and environment tensors
  are **bit-identical** to `CSSDModel` (`max_abs_diff = 0.0`).
* **G2** inactive-role strict zero: `psi_A(0) = psi_E(0) = 0` exactly;
  `dict_zero` reproduces the common-only occurrence exactly.
* **G3** role permutation equivariance: permuting residual coordinates together
  with the matching `W_A_S` / `W_E_S` dictionary rows (common coordinate fixed)
  leaves the prediction unchanged (`≤1e-5`, observed ~1e-9/1e-10).
* **G4** chemistry / topology purity: environment invariant to re-randomising
  global context, relation, pair bucket/index, topology; new parameters are
  exactly the four psi tensors.
* **G5** C6 / relation / backend frozen: `RNDB_MASK is cm.C6_MASK`,
  `c6_equivalence_check()` true, reader / node / edge / fusion / global /
  topology modules are the parent's classes (reused, not copied).
* **G6** official-test blocker raises on `official_test_loaded=True`; every
  payload carries `false`.

12 focused tests pass (`tracks/ksvd/tests/test_e2e_dictenv_rndb_v1.py`).

---

## 5. Smoke (mechanism trainability only)

`smoke.json`, 8 epochs / 2048 train / 512 valid, CPU:

* finite loss and finite predictions; `best_valid_mae` 0.664893.
* `grad(D)`, `grad(psi_A.W1/W2)`, `grad(psi_E.W1/W2)` all non-zero at
  initialisation and after the smoke run.
* node response ratio `||psi||/||p||` 0.0176 -> 5.3e-6 over 8 epochs, edge
  0.0117 -> 0.0101. (The node branch had a small-value transient, but the
  gradients stayed non-zero; the full run shows it is not a collapse — §6.)

No stop condition fired.

---

## 6. Training — the single seed-0 trajectory

`run_seed0.json`, `curve_seed0.csv`. 320 epochs, seed 0, 8 threads, wall
5727.5 s. No early stop.

| epoch | valid MAE | `grad(D)` | `grad psi_A W1` | node ratio | edge ratio |
|---|---|---|---|---|---|
| 1 | 0.908482 | 3.14e-2 | 7.8e-9 | 0.0001 | 0.0046 |
| 20 | 0.257444 | 8.20e-2 | 8.3e-3 | 3.056 | 2.873 |
| 40 | 0.236941 | 1.46e-1 | 1.0e-2 | 3.066 | 2.250 |
| 80 | 0.173122 | 5.81e-2 | 1.3e-2 | 3.076 | 1.717 |
| 160 | 0.160710 | 6.50e-2 | 1.6e-2 | 2.698 | 1.459 |
| 240 | 0.146161 | 6.93e-2 | 2.1e-2 | 2.501 | 1.365 |
| 320 | 0.169487 | 1.19e-1 | 1.9e-2 | 2.341 | 1.327 |

* best valid `0.139599` @ epoch 256.
* Top-5 soup members `[248, 256, 258, 298, 314]`, member MAEs
  `[0.141603, 0.139599, 0.141693, 0.140422, 0.141401]`.
* **soup `M_R = 0.1331174676119699`**.

**Trajectory note (observation, not a rescue).** The shared `psi` starts small
(`W2 ~ N(0,0.01)`), and Adam's normalised updates move it substantially within
the first ~20 epochs: the node response ratio is `1e-4` at epoch 1, then
`3.06` at epoch 20 and ~2.3-3.1 for the rest of training. This is a transient,
not a collapse: `psi` parameter norms grow monotonically
(`psi_A` `param_norm` 6.20, `W1` 5.11, `W2` 3.51 at the soup; `psi_E` 3.18 /
2.42 / 2.07), and `<2%` of `psi_A` parameters are `<1e-12`. The early
shrink-then-grow is an Adam-on-small-init-branch transient, recorded for a
future round; it was **not** corrected (no init/lr/scale rescue is authorised).

Final dictionary health (`dictionary_health.json`, full valid, adapted to slice
the common coordinate): 32/32 active atoms, effective atoms 22.16, top-1
activation rate 0.658, residual reconstruction relative 0.97696 (the frozen
full-descriptor diagnostic; the optimised residual term is ~4.9e-5), dictionary
movement 5.37.

Interim valid results were only recorded; they did not change any setting.

---

## 7. Frozen interventions (no retraining)

`psi_disable.json`, `psi_node_disable.json`, `psi_edge_disable.json`,
`dictionary_zero.json`, `assignment_shuffle_{node,edge}.json`. Evaluated on the
frozen soup state, parent weights / dictionary / common coordinate / relation /
backend / reader all retained.

| intervention | valid MAE | delta vs `M_R` |
|---|---|---|
| `M_R` (soup, psi on) | 0.133117 | — |
| `M_psi0` (psi_A = psi_E = 0) | 0.323797 | **G_psi = +0.190679** |
| node-only psi off | 0.273887 | G_psi_node = +0.140770 |
| edge-only psi off | 0.196141 | G_psi_edge = +0.063024 |
| residual dictionary `alpha -> 0` | 0.421944 | **G_dict = +0.288827** |

Both primary mechanism gates are passed by a **very large** margin
(`G_psi >= 0.003`, `G_dict >= 0.010`). Disabling `psi` or zeroing the residual
dictionary destroys the model, so both the rolewise nonlinearity and the sparse
dictionary are strongly load-bearing in the trained solution.

Assignment shuffle (frozen parent semantics, seeds `101/202/303/404/505`):

* node shuffle mean `+0.137391` (per-seed `+0.142, +0.136, +0.104, +0.153, +0.152`);
* edge shuffle mean `+0.157791` (per-seed `+0.144, +0.151, +0.165, +0.177, +0.151`).

The learned dictionary-role / shell assignment correspondence is load-bearing
for both nodes and edges.

---

## 8. Task interpretation

* `M_R = 0.133117` -> band **P3 `RNDB_NO_USEFUL_TASK_GAIN`** (`0.126 < M_R <= 0.135`).
* historical CSSD-q1 seed-0 soup `0.130028`; `G_hist = -0.003089`
  (RNDB is slightly **worse**). This is an **unmatched historical comparison
  only** — no baseline was rerun, so no architecture claim is made either way.

So the task answer to the pre-registered question is **no**: the rolewise
nonlinearity does not improve ZINC valid MAE; it lands just outside the
task-neutral band on the worse side. The mechanism answer is **yes and very
strongly**: the new operator is highly load-bearing, and so is the dictionary.

---

## 9. Case classification (pre-registered table)

Pre-registered cases (priority order A–G): A/B require `M_R <= 0.123`; C
requires `0.123 < M_R <= 0.126`; D requires `M_R <= 0.123` with `G_psi < 0.003`;
E requires `G_dict < 0.010`; F is literally
`M_R > 0.126` **and** `no strong new mechanism evidence`; G is branch collapse.

Here `M_R = 0.133117 > 0.126`, but the mechanism evidence is very strong
(`G_psi = 0.190679`, `G_dict = 0.288827`, `psi` alive), so the **conjunction in
Case F does not hold**, while Case C's range is exceeded. **No pre-registered
case enumerates `M_R > 0.126` with strong new mechanism evidence.** This is a
boundary outcome.

Reported as **Case F — `RNDB_NO_GO_TASK_LEVEL_MECHANISM_STRONGLY_SUPPORTED`**:
the dominant condition (`M_R > 0.126`, band P3, no useful task gain) forces the
task-level no-go; the strong mechanism result is recorded rather than folded
into an invented case. The boundary is flagged in `summary.json`
(`case_boundary_note`) and in `DECISION.md`.

---

## 10. Limitations

1. **One seed, no matched baseline.** All task comparisons are unmatched
   historical context; `|G_hist| = 0.0031` is at the edge of the round's
   descriptive resolution and cannot be read as a regression or a gain.
2. **Single trajectory.** No seed 1/2/3, no matched same-round CSSD run.
3. **Interventions are non-orthogonal.** `G_psi` is large partly because the
   linear parent path was free to atrophy during training once `psi` dominated
   (node ratio > 2). The gate measures end-to-end load-bearingness of the
   trained solution, which is what it was pre-registered to measure, but it is
   not a clean additive-decomposition estimate.
4. **`G_dict` zeroes both the linear residual and its `psi`.** It does not
   separate "linear residual dictionary" from "nonlinear residual dictionary".
5. The smoke node-ratio transient (1e-4 -> 3.06 in 20 epochs under Adam)
   suggests the small-`W2` init interacts strongly with Adam's normalised step
   size; the architecture's trainability is therefore init/optimizer-sensitive.
6. Historical CSSD-q2 was never trained (q1 selected), so any q2 comparison is
   context only.

---

## 11. Next-round shapes (recorded only; nothing authorised or implemented)

* **Task conversion, not rescue.** The mechanism is real; the open question is
  whether a *pre-registered* variant can make it help the task at matched
  capacity: e.g. an explicit residual scale / normalisation of the `psi`
  branch, or combining `psi` with a matched dense control, under a new
  pre-registration with ≥3 seeds and a same-round matched baseline. This is a
  future hypothesis only — the current round forbids it and no seed 1 is
  authorised.
* **Separate the dictionary mechanisms.** A future intervention that zeroes the
  linear residual coordinate while keeping `psi` (or vice versa) would separate
  "linear residual dictionary" from "nonlinear residual dictionary".
* **Init/optimizer sensitivity.** A controlled study of the small-`W2` init vs
  Adam step size on the early `psi` transient, without reading it as a rescue.

No post-hoc rescue, no width / init / gate / lr / lambda / horizon change, no
seed 1, no official-test read.
