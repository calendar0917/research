# PEC-v0 — prior-artifact audit (round 0, before any code)

Round name: **PEC-v0** (*Pure Environment Composition*).
Protocol id: `pec_v0`. Study: `zinc-context-gap`.
Written **before** any implementation, cache build, training or GPU use.
This note modifies no historical record.

Local audit base: `git log -1 --oneline` = `883e529` (SDB-v0 durable
analysis/claims/decision). The three SDB commits (`883e529`, `68a6ec9`,
`00bf711`) are present locally and were **not** assumed to be on GitHub main.

---

## 0. What PEC-v0 wants to be

A strictly **no-message-passing, no-recurrence, no mixed-bypass** model in which

1. chemistry (atom/bond category) may not read topology;
2. topology may not read chemistry;
3. the two first meet inside a *binding* of a reusable **rooted structural role**
   with a **chemical primitive** at the same occurrence;
4. that binding forms a per-root **local chemical environment** `E_i` which is
   then **frozen**;
5. the only cross-root computation is **one read-only static pair composition**
   of frozen environments plus a pure-topology relation;
6. a graph reader consumes only pooled unary environments, pooled environment
   pairs and a pure-topology global branch.

The novelty is *not* "a dictionary somewhere in a GNN". It is the specific
object `E_i = H(q_i, sum_v r^V_{iv} ⊗ a(q_v), sum_e r^E_{ie} ⊗ c(b_e), S_i)`
built from **frozen sparse codes of audited pure-topology rooted bases**, with
`E_i` provably immutable after formation.

---

## 1. `prior_artifact_audit` — required questions

### 1.1 How is PEC-v0 fundamentally different from TCCD?

| axis | TCCD-v0..v7 | PEC-v0 |
|---|---|---|
| dictionary input `x_v` | **typed** canonical patch coordinate (topology **and** atom/bond one-hot in the same vector) | chemistry-free: `phi_v ∈ R^65` (node) / `psi_e ∈ R^130` (edge), audited pure topology |
| assignment | learned prototypes `B`, temperature-annealed **softmax**, task-coupled end-to-end | `K=16, s=4` **exact top-s** tied-IHT code over a **K-SVD-fitted, frozen** dictionary |
| environment | one soft code vector per root patch (a prototype mixture) | an explicit occurrence-level **binding tensor** of role ⊗ chemical primitive |
| composition | code-level relation matrices (`Rint = B B^T`, `Rb`, `Rgeo`) between root codes | `F(E_i+E_j, |E_i-E_j|, E_i⊙E_j, rho_ij)` over **frozen** environments |
| status | standalone route **CLOSED** (v7: explicit normalized moment representation 0.4121 vs frozen raw 0.2784, required 0.255) | untested object |

So the TCCD verdict (`decision-tccd-v7-stop-standalone-normalized-moment-20260922`,
`standalone_route_status: CLOSED`, `no_v8_rescue: true`) does **not** transfer
directly: TCCD's failure was a *soft-prototype / normalized-moment
representation* failure, and its dictionary input was chemistry-contaminated.
PEC-v0 removes both of those design choices. This is the one genuine opening.

### 1.2 How is PEC-v0 fundamentally different from SDB-v0?

SDB-v0 (`claim-sdb-v0-no-material-gain-on-strong-backbone-20260923`,
`claim-sdb-v0-sparse-dictionary-replaces-role-axis-20260923`) established:

* (a) the chemistry-free `R^65` role axis **can** be replaced by a reusable
  sparse `K=32, s=8` dictionary coordinate with no loss of the FSAR-R2-AR0
  assignment (Stage 1 PASS, Stage 2 PASS, recovery 1.159, shuffle 0.245);
* (b) on the strong strict-static **S0** backbone the same replacement adds no
  material task gain: `S0+DictBinding 0.143294` vs `S0+DenseBinding 0.144183`
  vs base `0.145674`; gain `0.002381 < 0.003` FAIL, shuffle `0.016131 < 0.02`
  FAIL, and `0.143294 > 0.140794` (historical S0 soup).

Crucially SDB-v0 Stage 4 is an **additive residual readout**
`y = y_S0 + <W_D, C~_D>` on top of a strong mixed backbone; the binding is a
*post-hoc branch*. PEC-v0 makes the binding the **only** chemistry route, has
**no** `y_S0` base, and composes per-root environments rather than reading one
global `C_D`. Therefore (b) is a strong **prior against the magnitude** but not
a test of the object. (a) is a strong **positive prior for Gate 1** and tells us
`K=16, s=4` on a `R^65`/`R^130` basis is the *harder* version of a regime that
already passed at `K=32, s=8` — Gate 1 is expected to pass and is therefore a
correctness/health gate, not the scientific question.

### 1.3 How is PEC-v0 fundamentally different from FSAR?

FSAR-R2-AR0 / -EDGE
(`decision-fsar-r2-ar0-assignment-supported-20260917`,
`decision-fsar-r2-ar0-edge-assignment-supported-20260917`) validated a
**single whole-molecule** pooled assignment statistic
`C = sum_v (phi_v - phibar)(q_v - qbar)^T` (and `C_E` for edges), read by a
*bias-free linear* residual on a strong marginal base. It is a **global
first-order moment**, not a per-root environment, and it has no pair composition
of environments. FSAR's positive result is the reason PEC-v0's *ingredients*
(role ⊗ primitive, pure-topology `phi`/`psi`) are trustworthy; it is not the
same object. PEC-v0 keeps the audited ingredients and changes the object
(per-root environment + static composition).

### 1.4 Why will PEC-v0 not repeat FSAB / compact-v6 branch collapse?

The collapses are on record:

* FSAB (`notes/factorized_structure_attribute_binding.md`,
  `decision-factorized-structure-attribute-binding-stop-20260916`): perf passed
  (seed0 soup 0.12295) but the token encoder went **bit-constant** — S norm std
  `9.3e-10`, effective rank 0 — so the binding channel was unused.
* compact-v6 (`decision-compact-v6-topology-attribute-factorization-nogo-20260910`):
  the factorized attribute branch was **exactly invariant** to within-patch
  attribute/role-association shuffles (max per-molecule `|delta| = 0.0`,
  `e_attribute` norm std `2.3e-10`).
* ASB-Z1 (`notes/adaptive_structure_binding_cell_z1.md`): global constant gate,
  binding branch exactly dead; one seed collapsed to root-only support.

Common cause: a *residual / added* branch inside a strong backbone is not
load-bearing, gets driven to ~0 by weight decay, and is then proven unused by an
exact shuffle-invariance test.

PEC-v0 cannot repeat this in the same way because the environment is **not an
added branch** — there is no strong backbone to bypass it, and chemistry has no
other route to the reader. If the environment collapses, the model cannot
predict at all, which is directly observable. PEC-v0 additionally keeps the
compact-v6/FSAB **exact-shuffle witness** as a Gate-0/Gate-2 requirement
(chemistry-placement shuffle must change `E_i`; assignment shuffle must change
the prediction). This is a genuine structural difference, but it is exactly the
reason the a-priori expectation is still low: the track's pattern is that
binding/placement statistics are redundant with marginals in strong models.

### 1.5 Why not use the B-Null recurrence?

`claim-local-token-null-20260919`: the whole molecule-dependent local 16-D
token family is redundant on the strong backbone (B-Null `e_patch = 0`,
49,343 params, soup `0.123028`; Constant-16 `0.120515`; B-Full `0.118972`).
Recurrence was already addressed by CGA-v0
(`decision-zinc-cga-v0-deprioritise-local-dictionaries-20260923`): the recurrent
contextual refinement is 93–96 % predictable from the local state, adds only
1–4 % of the shift variance, and has no consistent task link. Adding recurrence
would (i) re-open a closed line, (ii) add message passing that the purity
contract forbids, and (iii) import exactly the mixed computation PEC-v0 is
trying to isolate. So recurrence is excluded by both prior evidence and the
research question.

### 1.6 Why is static pair composition of environments not message passing?

A message-passing / relation-refresh model computes a *new* state that is written
back into a local state, which is then read again:

```
h_i' = Update(h_i, Aggregate_j(q_ij))        # FORBIDDEN
```

PEC-v0 computes `E_i` **first and completely**, then:

```
c_ij = F(E_i, E_j, rho_ij)                   # computed once
R_pair = Pool_{i<j}(c_ij)                    # graph-level only
```

There is no path from `c_ij` (or from any `E_j`) back into `E_i`, no second pair
evaluation, no relation refresh, no attention. `c_ij` is never stored as a node
state and is only pooled at graph level. This is the same structural ordering the
strict-static S0 contract already enforces with
`center_update is None` and the `_pool_pairs_to_centres` runtime guard
(`experiments/luyin16/zinc_static_dictionary_pair.py::static_contract_checks`),
and PEC-v0 re-uses that idea as a hard monkeypatch-to-raise Gate-0 test.

### 1.7 Which baselines / checkpoints can be reused without retraining?

Reused (read-only, durable, no retraining):

* **FSAR-R2-AR0 feature cache** `tracks/ksvd/results/fsar_r2_ar0/cache/{train,valid}.pkl.gz`
  (schema `fsar_r2_ar0_features_v1`, 10 000 / 1 000, radius 2, `phi ∈ R^65`).
  PEC-v0 re-derives its own occurrence/edge cache from the same
  `_load_zinc` + `_data_to_graph` + `fsar_v2._explicit_basis_for_patch` path so
  the role basis is bit-comparable.
* **FSAR pure-topology edge role** `build_edge_roles` (`psi ∈ R^130`) from
  `experiments/luyin16/fsar_r2_ar0_edge.py` (AR0-EDGE audited).
* **TCCD correctness-tested tied-IHT** `code/tccd_v0.py::iht_codes`
  (10 steps, deterministic power-iteration step size, unit-normalized atoms,
  exact top-s) and `hard_threshold_rows` / `power_iter_sigma`.
* **SDB-v0 dictionary machinery** `experiments/luyin16/sdb_v0.py`:
  `fit_ksvd`, `normalize_columns`, `random_normalized_dictionary`, `omp_codes`,
  dictionary-health / exact-sparsity / relative-error reports.
* **Strict-static S0 topology-only pair relation**: the pure-topology subset of
  `zinc_patch_path_pooling._pair_relation` (distance one-hot 8, log distance 1,
  overlap 5, boundary 3, log path count 1 = **18-D**). The chemistry-containing
  fields `path_bond_mean` (4) and `adjacent` bond one-hot (4) are **deleted**.
* Performance context only (never re-run, never used for selection):
  strict-static S0 seed0 soup `0.140794`, seed1 soup `0.136423`, base
  `0.145674`; B-Null `0.123028`; B-Full `0.118972`; SDB `S0+Dict 0.143294`;
  SDPK soup `0.139735`.

Explicitly **not** re-run: strict-static S0, B-Null, B-Full, B-Bag, FSAR
M0/MB/MM, FSAR edge baseline, TCCD, DTX, SDPK, SRDA, SDB-v0 Stages 1–4, and any
other historical control whose protocol/identity is already fixed by a durable
artifact.

---

## 2. Dedupe verdict

**No completely equivalent implementation exists.** The closest objects are:

| prior | object | decisive difference from PEC-v0 |
|---|---|---|
| SDB-v0 Stage 4 | global `C~_D` added to S0 as a linear residual | additive on a strong mixed backbone; not per-root; no composition |
| SDPK-v0 | dictionary coordinates inside the S0 static pair kernel | keeps the mixed local token `h0_i`; dictionary is a soft `tau`-gate kernel |
| SRDA-v0 | dictionary assignment defines the relation algebra | keeps the mixed local token / continuous shell descriptor |
| DTX-v0 | dictionary environments × generic topology role cross | aligned cross statistic on top of the mixed S0 backbone |
| TCCD | learned local environments + composition | soft prototypes, chemistry-contaminated dictionary input, CLOSED route |
| FSAB | per-patch S/A/B factorization into one 16-D token | binding collapsed to constant; recurrent pair-centre pipeline retained |
| BCE | explicit support composition of bindings | `Compose` collapsed to a constant; supports *inside* one patch |
| ASBC-Z1 | adaptive binding-conditioned support | gate/branch collapse |
| PSD-v0 | pair-state + function-preserving dictionary | pair-state message-passing encoder, raw input, stopped at Q1 |
| GSCN-v0 | dictionary-core GNN | dictionary after neighbour pooling, rejected |

**Proceed, but with an explicitly pre-committed negative prior and hard STOP
rules.** The affordable falsification order is history dedupe (done) → Gate 0
CPU correctness → Gate 1 label-free CPU → Gate 2 small internal train → at most
one seed-0 formal GPU run. Any clear negative at a gate ends the round with no
rescue (no `K`/`s` sweep, no LISTA, no attention, no recurrence, no dense
bypass, no additional seeds).

---

## 3. What would make this round worth its cost

The round is worth running only because it can *cleanly falsify* the
environment-composition hypothesis with cheap CPU gates:

* Gate 1 falsifies the sparse role dictionary if it cannot preserve even the
  coarse rooted structural role (`shell` / `shellpair`).
* Gate 2 falsifies the composition hypothesis if `TRUE ≈ BAG` or
  `TRUE ≈ SHUFFLE`, or if `SparseDict` is clearly worse than the
  parameter-matched `DenseRole`.
* Gate 2 falsifies the environment hypothesis if chemistry-placement shuffle
  does not materially hurt.

If all of those pass, only then is one seed-0 official-valid GPU run purchased.
