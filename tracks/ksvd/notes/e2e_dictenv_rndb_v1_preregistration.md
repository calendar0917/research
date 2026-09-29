# E2E-DictEnv-RNDB-v1 — pre-registration (frozen)

Round: **E2E-DictEnv-RNDB-v1** (`e2e_dictenv_rndb_v1`).
Study: `zinc-context-gap`.  Method: **RNDB = Rolewise Nonlinear Dictionary Binding**.
Prior-artifact audit: [`e2e_dictenv_rndb_v1_prior_artifact_audit.md`](e2e_dictenv_rndb_v1_prior_artifact_audit.md).

This pre-registration is written **before any RNDB implementation is committed**.
It freezes the parent object, the single question, the architecture, the
initialisation, the parameter budget, the correctness gates, the training
budget, the mechanism gates, the performance interpretation bands, the stop
rules and the official-test firewall.  Nothing below may be changed after the
formal run starts; any change requires a new round id.

`official_test_loaded = false` is mandatory everywhere.

---

## 1. Scientific question

> The current CSSD / FINAL-CLEAN dictionary–chemistry binding sums the per-role
> structure x chemistry responses **linearly** and only then passes the summed
> vector into the environment's nonlinear decoder.  Does moving one small
> **shared nonlinear operator** between the individual residual dictionary-role
> responses and their aggregation — `rolewise product -> shared nonlinearity ->
> role sum` instead of `rolewise product -> role sum -> later nonlinearity` —
> improve ZINC valid MAE, while keeping the sparse dictionary genuinely
> behaviourally load-bearing?

This is a single-hypothesis falsification of one function-class change.  It is
**not** a sweep, not a new feature, not a capacity widening, and no baseline is
retrained this round.

---

## 2. Parent configuration (frozen; resolved from the repo, not assumed)

Resolved automatically at preflight and recorded in `parameter_audit.json` /
`correctness.json`:

| item | value | source |
|---|---|---|
| parent object | `CSSD-q1` (`CSSDModel`) | `e2e_dictenv_common_subspace_dictionary_v1.py` |
| parent commit | `5f0f284070cde1019e8c2b6bc762b8dd0223022b` (only later change: additive `model_factory` hook, behaviour-neutral) | git |
| config | `cm.H1_CONFIG` = P2Config(`decoder="h1"`, `d_e=48`, `K=32`, `s=8`, `lambda_factor=0.25`, `horizon=320`, `dict_kind="sdb32"`) | `e2e_dictenv_clean_mechanism_v1.py` |
| lambda | `cm.H1_LAMBDA = 33.95873017865987` | idem |
| mask | `cm.C6_MASK` (`cssd.CSSD_MASK is cm.C6_MASK`) | idem |
| common subspace | q1, `U` (65x1), `rms=[5.082852828320509]`, fit on official train only | `results/e2e_dictenv_common_subspace_dictionary_v1/common_subspace.json` |
| dictionary | `sdb32`, loaded by `p2run.load_dictionary` | `results/sdb_v0/dictionary.pt` |
| D_struct | `33 = 1 common + 32 residual dictionary roles` | `CSSDModel.code` |
| D_A, D_E | `96`, `48` | `e2e_dictenv_p1.py` |
| shells / shellpairs | `N_SHELLS=3`, `SHELLPAIR_CLASSES=6` | `e2e_dictenv_p1.py` |
| optimizer / protocol | Adam `lr=1e-3`, `weight_decay=1e-5`, batch `128`, grad clip `5.0`, Top-5 soup, 320 epochs | `zinc_e2e_dictenv_p1.py` / `p2_abs` |

Everything except the RNDB binding itself is frozen and reused, not re-typed:
`phi65`, train-only common direction, common scaling, residual sparse
dictionary, IHT-10, `K=32`, `s=8`, `lambda_rec`, C6 masks, node/edge shell
definitions, pair relation, distance bucket, anchor, global branch, topology
branch, moment pooling, reader, optimizer, batch size, data ordering, Top-5 soup
protocol, horizon.

---

## 3. Historical-reference policy (read-only; never rerun)

This round **trains only the new method**.  The following are read from local
durable artifacts and recorded in `results/e2e_dictenv_rndb_v1/historical_references.json`:

| reference | exact local value | artifact | use |
|---|---|---|---|
| CSSD-q1 seed0 soup valid | `0.13002798487985273` | `results/e2e_dictenv_common_subspace_dictionary_v1/training/final.json` | descriptive comparison |
| CSSD-q2 | not trained (q1 selected 3/3 zero-training seed rule) | `.../zero_training/selection.json` | context only |
| FINAL-CLEAN seed0 soup valid | `0.12849851670576026` | `results/e2e_dictenv_clean_mechanism_v1/stage_c_independence/frozen_C6_seed0.json` | descriptive comparison |
| T1 tuned seed0 soup valid | `0.12576500436564675` | `results/e2e_dictenv_t1/tuning_decision.json` | descriptive comparison |

Explicitly: **no** CSSD-q1 / CSSD-q2 / FINAL-CLEAN / P1 / DenseTied / T1 / F / R
/ G model is retrained.  The prompt's approximate values (`CSSD-q1 ~= 0.1244`,
`CSSD-q2 ~= 0.1236`) **do not match the local artifacts**; the local artifact
values above are the source of truth and the mismatch is recorded in
`historical_references.json`.

Historical numbers are **unmatched context only**.  Because no matched baseline
is rerun this round, `|delta| < 0.003` may never be described as an architecture
improvement.

---

## 4. Exact RNDB architecture

Let `z_v in R^{D_struct} = [c~ ; alpha_res]`, `K_common = {0}` (the `q=1` common
coordinate), `K_dict = {1,...,32}` (the residual sparse dictionary roles).

### 4.1 Node binding

Parent occurrence (exact):

```
p_{v,k} = z_{v,k} (W_A_S[k,:] odot c_v),   c_v = (q_v W_A_C)/sqrt(D_A)
p_v     = sum_k p_{v,k}                     (parent uses this directly)
```

RNDB occurrence:

```
c_v      = (q_v W_A_C) / sqrt(D_A)
p_{v,k}  = z_{v,k} (W_A_S[k,:] odot c_v)
p_v^common = sum_{k in K_common} p_{v,k}                # untouched linear parent
h_{v,k}  = p_{v,k} + psi_A(p_{v,k})                     # k in K_dict
u_v      = p_v^common + sum_{k in K_dict} h_{v,k}
```

`u_v` then follows the frozen parent path exactly: `index_add_` into the
root-relative `(root, shell)` slot, `node_encoder`, `anchor_encoder`,
`fusion`.  Shell routing is unchanged.

### 4.2 Edge binding

Parent occurrence (exact):

```
g_{uv,k} = [ z_{u,k}+z_{v,k} ; |z_{u,k}-z_{v,k}| ; z_{u,k} z_{v,k} ]
r_{uv,k} = (z_{u,k}+z_{v,k}) W+_k + |z_{u,k}-z_{v,k}| WD_k + (z_{u,k} z_{v,k}) Wx_k
p^E_{uv,k} = r_{uv,k} odot c^E_{uv},      c^E_{uv} = (b_{uv} W_E_C)/sqrt(D_E)
p^E_{uv}   = sum_k p^E_{uv,k}
```

where `W+ = W_E_S[0:D]`, `WD = W_E_S[D:2D]`, `Wx = W_E_S[2D:3D]`, `D = D_struct`.

RNDB occurrence:

```
p^E,common_{uv} = sum_{k in K_common} p^E_{uv,k}        # untouched linear parent
h^E_{uv,k}      = p^E_{uv,k} + psi_E(p^E_{uv,k})        # k in K_dict
u^E_{uv}        = p^E,common_{uv} + sum_{k in K_dict} h^E_{uv,k}
```

`u^E` then follows the frozen parent shellpair aggregation, edge slot encoder
and environment pipeline.  The shellpair taxonomy is unchanged.

### 4.3 Nonlinear operators (the only new modules)

```
psi_A : Linear(96, 32, bias=False) -> SiLU -> Linear(32, 96, bias=False)
psi_E : Linear(48, 16, bias=False) -> SiLU -> Linear(16, 48, bias=False)
```

`psi_A` is shared across all 32 residual dictionary roles, all nodes and all
shells; `psi_E` is shared across all roles, all bonds and all shellpairs.
Forbidden: bias, LayerNorm, BatchNorm, dropout, attention, softmax,
role-specific MLP, per-shell MLP, gate, hypernetwork, extra chemistry input,
dictionary-atom embedding, coarse146, second dictionary, message passing.

### 4.4 Initialisation (frozen, no sweep)

`W1` standard Kaiming-uniform (PyTorch `nn.Linear` default);
`W2 ~ Normal(mean=0, std=0.01)`; no bias.  RNDB therefore starts near the parent
but with `psi != 0` and non-degenerate gradients in both layers.

### 4.5 Strict zeros (hard invariant)

`bias=False` must give, for node and edge, `p_k = 0 => psi(p_k) = 0 => h_k = 0`.
An inactive dictionary role can never produce a learned constant.

---

## 5. Why the common coordinate is excluded

The `q=1` common coordinate `c~` is an independent dense, train-only direction,
not a dictionary atom.  It keeps the parent's original linear binding in both
the node and edge paths.  `psi` is applied only to `K_dict`.

---

## 6. Parameter budget

```
new node psi_A : 96*32 + 32*96 = 6144
new edge psi_E : 48*16 + 16*48 = 1536
new total      : 7680   <= 8000
parent (CSSD-q1)  : 97727
RNDB total        : 105407
relative increase : 105407/97727 - 1 = 0.07858  <= 1.10
```

The exact counts are recomputed from the code and recorded in
`parameter_audit.json`.  If the real parent width differs the formula applies
naturally, but a total above `1.10 x parent_params` is a STOP, and no width is
changed to reach the budget.

---

## 7. Correctness gates (all must PASS before training)

* **G0 parent decomposition identity** — for node and edge, on a synthetic
  fixture **and** a real ZINC batch, the sum over all structural-coordinate
  contributions equals the original parent occurrence, `max_abs_diff <= 1e-6`.
* **G1 psi-off exact parent equivalence** — same freshly instantiated state,
  `psi_A = psi_E = 0`, RNDB forward must equal the parent forward; environment
  and prediction tensors `max_abs_err <= 1e-6`.
* **G2 inactive-role strict zero** — `z_k = 0 => p_k = psi(p_k) = h_k = 0`
  (node and edge).
* **G3 role permutation equivariance** — a random permutation of the residual
  dictionary coordinates, with the matching permutation of `W_A_S` dictionary
  rows and the `W_E_S` dictionary rows in all three blocks, leaves the
  prediction unchanged (common coordinate fixed).
* **G4 chemistry / topology purity** — `psi` reads only the already-formed
  `p_{v,k}` / `p^E_{uv,k}`; no raw `phi`, shell id, role-index embedding, graph
  id, root id, distance or target.
* **G5 C6 / relation / backend frozen** — C6 masks, pair relation, pair bucket,
  anchor, global branch, topology branch and reader are the parent's, reused
  (not copied).
* **G6 official-test blocker** — an attempt to load official test raises /
  is blocked; every payload carries `official_test_loaded = false`.
* Reused test set: focused RNDB tests plus the directly related CSSD and
  clean-mechanism tests, then `uv run research verify`.

---

## 8. Smoke gate (mechanism trainability only)

After all correctness gates PASS, a 5-10 epoch CPU smoke run checks: finite loss
and predictions; `grad(D) != 0`; `grad(psi_A.W1/W2) != 0`; `grad(psi_E.W1/W2)
!= 0`.  Report `mean ||p||`, `mean ||psi(p)||`, ratio, output variance and
effective rank for node and edge.

Failure modes that STOP (no optimizer/weight-decay/normalisation rescue):
NaN/Inf, all-`psi` gradients zero, `D` task gradient zero, or an architecture
exact dead branch.

---

## 9. Training budget

```
seed         = 0
from scratch = yes   (no warm start of any historical checkpoint)
max_epochs   = 320
runs         = exactly 1 RNDB trajectory
```

No seed 1/2/3, no baseline rerun, no width/rank/activation/lr/lambda/horizon
sweep, no multiple RNDB variants, no parallelism of formal training.  Even a
very good result only authorises "seed1 / matched confirmation is justified as
NEXT ROUND" and is not executed this round.

Diagnostics are logged at epochs 1, 20, 40, 80, 160, 240, 320 (train/valid MAE,
reconstruction, dictionary active/effective atoms, max usage, `grad(D)`,
`grad psi_A W1/W2`, `grad psi_E W1/W2`, `||psi_A||`, `||psi_E||`, node/edge
response ratios).  Intermediate valid results are only recorded; they may not
change architecture / LR / width / lambda.

---

## 10. Early-stop rules

Normally run all 320 epochs.  Early STOP only for: NaN/Inf, data corruption,
official-test firewall violation, broken dictionary coding invariant, `psi`
branch exact collapse **with** numerically-zero gradients, or another broken
implementation invariant.  Not for unflattering epoch-40/80 MAE.

---

## 11. Frozen interventions (no retraining)

On the frozen RNDB Top-5 soup state only, with the parent weights, dictionary,
common coordinate, relation, backend and reader all retained:

* `psi_A = psi_E = 0`  -> `M_psi0`, `G_psi = M_psi0 - M_R`; primary mechanism
  gate `G_psi >= 0.003`.
* node-only off -> `M_node_off`; edge-only off -> `M_edge_off`; both off ->
  `M_both_off`; report `G_{psi,node}`, `G_{psi,edge}`.  Frozen inference only,
  no new training arm.
* residual sparse dictionary `alpha -> 0` (keep common coordinate, chemistry,
  relation, backend) -> `M_dict0`, `G_dict = M_dict0 - M_R`; minimum dictionary
  mechanism requirement `G_dict >= 0.010`.
* assignment shuffle with the parent's frozen semantics
  (`p1.shuffled_occ_node_for_molecule`, `p1.shuffled_bond_endpoints_for_molecule`)
  and frozen seeds `(101, 202, 303, 404, 505)`, node and edge separately.

---

## 12. Performance interpretation bands (candidate absolute valid MAE only)

No historical baseline is used as a strict gate.

| case | `M_R` | label |
|---|---|---|
| P0 | `<= 0.120` | `RNDB_SINGLE_SEED_STRONG_SIGNAL` |
| P1 | `0.120 < M_R <= 0.123` | `RNDB_SINGLE_SEED_PROMISING` |
| P2 | `0.123 < M_R <= 0.126` | `RNDB_TASK_NEUTRAL_WITHIN_HISTORICAL_BAND` |
| P3 | `0.126 < M_R <= 0.135` | `RNDB_NO_USEFUL_TASK_GAIN` |
| P4 | `M_R > 0.135` | `RNDB_TASK_REGRESSION` |

`G_hist = M_hist^CSSD-q1 - M_R` is descriptive only and must be labelled
"historical, unmatched comparison".

---

## 13. Final scientific decision table (priority order)

* **A** `M_R <= 0.120`, `G_psi >= 0.003`, `G_dict >= 0.010`, health PASS, psi
  branch alive -> `RNDB_STRONG_SINGLE_SEED_SUPPORTED`.  Next step may only be
  "matched confirmation next round"; not executed here.
* **B** `0.120 < M_R <= 0.123`, `G_psi >= 0.003`, `G_dict >= 0.010`, health
  PASS -> `RNDB_PROMISING_SINGLE_SEED` (no matched baseline, no confirmed gain).
* **C** `0.123 < M_R <= 0.126`, `G_psi >= 0.003`, `G_dict >= 0.010` ->
  `RNDB_MECHANISM_SUPPORTED_TASK_NEUTRAL`.
* **D** `M_R <= 0.123` but `G_psi < 0.003` ->
  `RNDB_GOOD_TRAJECTORY_MECHANISM_NOT_ESTABLISHED`.
* **E** `G_dict < 0.010` (any MAE) -> `RNDB_DICTIONARY_MECHANISM_LOST`.
* **F** `M_R > 0.126` and no strong new mechanism evidence -> `RNDB_NO_GO`.
* **G** psi branch collapse (`psi output ~ 0` or `>95%` params effectively zero,
  with `G_psi < 0.003`) -> `RNDB_BRANCH_COLLAPSE`; do not fix.

No post-hoc rescue is allowed: no psi width / W2 init / bias / weight-decay /
LayerNorm / residual scale change, no node-only or edge-only training, no gate /
hypernetwork / `d_k` conditioning / coarse146, no seed 1, no horizon or LR
change.  Any such idea is recorded as a future hypothesis only.

---

## 14. Official-test firewall

Only `official train` and `official valid` are read (via
`p1run.load_split("train")` / `load_split("valid")`).  The official test split
is never instantiated, encoded, evaluated, peeked at, or used for model
selection.  Every payload records `official_test_loaded = false`; any code path
that would load test raises.

---

## 15. Outputs

`results/e2e_dictenv_rndb_v1/` contains at least: `historical_references.json`,
`parameter_audit.json`, `correctness.json`, `smoke.json`, `run_seed0.json`,
`curve_seed0.csv`, `dictionary_health.json`, `rndb_response_stats.json`,
`psi_disable.json`, `psi_node_disable.json`, `psi_edge_disable.json`,
`dictionary_zero.json`, `assignment_shuffle_node.json`,
`assignment_shuffle_edge.json`, `summary.json`, `REPORT.md`, `DECISION.md`.
Durable notes, claims, decisions and `STATE.yaml` are updated afterwards.
