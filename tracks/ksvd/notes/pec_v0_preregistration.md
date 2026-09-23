# Pre-registration — PEC-v0: Pure Environment Composition (ZINC)

# Pre-registration — PEC-v0: Pure Environment Composition (ZINC)

> **Amendment A1 (frozen before the Gate-1 run, 2026-09-23).** §2.1 originally
> wrote the node role as the patch-aggregated FSAR-R2-AR0 coordinate
> `phi_v in R^65` reused once per *node*. That reading makes the Gate-1
> recoverability probe `alpha^V -> shell` unreachable: a per-node code cannot
> predict `shell(i, v)` for an arbitrary root `i`. A1 therefore freezes the
> literal occurrence-level reading of §3 of the round brief:
>
> * node role = the audited FSAR explicit rooted **node** basis row
>   `b^V_{iv} in R^11` of occurrence `(i, v)` in the radius-2 patch of root `i`
>   (`fsar_v2._explicit_basis_for_patch`, chemistry-free);
> * edge role = the audited explicit rooted **edge** basis row
>   `b^E_{ie} in R^15` of occurrence `(i, e)` in the same patch;
> * `alpha^V = SparseDict_V(b^V_{iv})`, `alpha^E = SparseDict_E(b^E_{ie})`,
>   dictionaries fit on occurrence rows from the internal fit split only;
> * `r^V_{iv} = [one_hot(shell(i,v),3); alpha^V_{iv}]`, `r^E_{ie} =
>   [one_hot(shellpair(i,e),5); alpha^E_{ie}]` (unchanged).
>
> Consequences, recorded a priori: (i) `K_V = 16 > 11` is overcomplete, so the
> raw reconstruction gate is weak by construction and the *real* Gate-1 content
> is the coarse-role probe plus the K-SVD-vs-random contrast; (ii) the
> parameter-matched dense control becomes `Linear(11,16) + Linear(15,16)` =
> 416 params, exactly matching the two frozen dictionaries (176 + 240 = 416).
> No other frozen quantity changes; the environment width stays 650-D for
> `dense`/`sparse` and 138-D for `coarse`.

Round name: **PEC-v0** (*Pure Environment Composition, v0*).
Written **before** any Gate-0/1/2 run and before any GPU use.
Protocol id: `pec_v0`. Study: `zinc-context-gap`.
Prior-artifact audit: [`pec_v0_prior_artifact_audit.md`](pec_v0_prior_artifact_audit.md).
This note modifies no historical record.

Regime: canonical PyG ZINC `subset=True` official **train 10 000 / valid 1 000**
(verifiable split fingerprint recorded with every result). **The official ZINC
test is never loaded, instantiated or referenced.** Official **valid** is *not*
used by Gates 0–2; it is reserved for the optional Gate-3 formal seed-0 run.

Everything below is **frozen**. A gate FAIL ends the round at that gate. There
is **no** `K`/`s` sweep, no LISTA, no attention, no entropy/balance loss, no
orthogonality sweep, no multiple dictionary widths, no recurrence rescue, no
dense-bypass rescue, no extra seed bought on a FAIL.

---

## 0. Scientific hypothesis

> A molecule can be represented as a set of **local chemical environments**
> `E_i`, each formed once from **chemical primitives bound to reusable rooted
> structural roles**, and then frozen; a **single read-only static pair
> composition** of frozen environments is sufficient to predict the ZINC
> property.

Two separable claims are tested:

* **Environment claim** — reusable rooted structural roles bound to atom/bond
  primitives form an environment that carries task information, and destroying
  which-primitive-sits-where (chemistry placement shuffle) materially hurts.
* **Composition claim** — `TRUE` static environment composition beats both
  `BAG` (unary only) and `SHUFFLE` (same environments, broken environment↔pair
  correspondence).
* **Dictionary claim** — the sparse dictionary role coordinate is not worse
  than a parameter-matched dense role coordinate (Gate 3 / claim gate only; a
  clear loss downgrades to "structural-role factorization supported but sparse
  dictionary not supported").

---

## 1. Purity contract (hard, audited)

The model may **not** read any of: raw `patch_cont`; typed token; parent token;
B-Full token; B-Bag token; B-Null recurrent state; global typed atom histogram;
global typed bond histogram; `path_bond_mean`; adjacent bond one-hot; any direct
raw atom/bond graph readout; the strict-static mixed pair relation; any
MP / recurrence / attention output.

Forward-access audit rules (Gate 0, enforced by construction + test):

* the dictionary input may not contain an atom or bond category;
* `q_v` / `b_e` may not be read by anything that also reads a shell, a degree,
  a neighbour, a ring flag, `phi` or `psi` before the binding operator;
* `E_i` is computed before any pair information exists;
* no pair quantity may be written back into any `E_i`;
* the reader may not read any chemistry tensor except through `E_i`.

---

## 2. Frozen objects

### 2.1 Roles (audited pure-topology bases, reused bit-comparably)

See **Amendment A1** above.  All role bases are the audited FSAR explicit
rooted occurrence bases (`fsar_v2._explicit_basis_for_patch`, radius 2,
chemistry-free):

* node occurrence role `b^V_{iv} ∈ R^11`;
* edge occurrence role `b^E_{ie} ∈ R^15`.

Chemistry primitives (may not read topology):

* `a(q_v)` = one-hot atom category, 28 ZINC categories;
* `c(b_e)` = one-hot bond category, 4 ZINC categories (observed 1..3 + pad).

### 2.2 Dictionaries (frozen after a single detached K-SVD fit)

| | basis | `K` | `s` | tied-IHT |
|---|---|---|---|---|
| node | `b^V ∈ R^11` | **16** | **4** | 10 steps, deterministic power-iteration step, unit-normalized atoms, exact top-s |
| edge | `b^E ∈ R^15` | **16** | **4** | same |

Initialization = K-SVD (`sdb_v0.fit_ksvd`, exact `s`), atoms unit-normalized;
codes = `tccd_v0.iht_codes` (correctness-tested, tied). No task coupling in
Gates 0–2. No `lambda`, no reconstruction term in the training loss for Gates
0–2. Task-coupled `D` (`TASK_COUPLED_DICT_SUPPORTED` claim) is out of scope for
this round unless Gate 3 authorizes a second, separately pre-registered stage.

`tied-IHT` is used for **encoding only**; `K-SVD` is used for the **fit**.

### 2.3 Environment (formed once per root `i`, then frozen)

Node binding over the radius-2 patch `P_i` of root `i`:

```
r^V_{iv} = [ one_hot(shell(i,v), 3) ; alpha^V_v ]            # 3 + 16 = 19
B^V_i    = sum_{v in P_i} r^V_{iv} ⊗ a(q_v)                  # 19 x 28 = 532
```

Edge binding over the undirected bonds `Q_i` inside `P_i`:

```
r^E_{ie} = [ one_hot(shellpair(i,e), 5) ; alpha^E_e ]        # 5 + 16 = 21
B^E_i    = sum_{e in Q_i} r^E_{ie} ⊗ c(b_e)                  # 21 x  4 =  84
```

with `shell ∈ {0,1,2}` (BFS distance from root, read off the audited node basis)
and `shellpair ∈ {(0,1),(0,2),(1,1),(1,2),(2,2)}` (sorted endpoint shells).

Pure-topology per-root scalars `S_i ∈ R^6` (explicitly topology-only):
`log1p(|P_i|)`, `log1p(|Q_i|)`, molecule degree of root `i`, molecule mean
degree, boundary fraction `|{v in P_i : shell=2}| / |P_i|`, molecule cycle rank
`max(m - n + 1, 0)`.

Environment:

```
E_i = H([ a(q_i) (28) ; vec(B^V_i) (532) ; vec(B^E_i) (84) ; S_i (6) ])  -> R^48
```

`H` is a per-root MLP `650 -> hidden -> 48`, SiLU, **no** adjacency, **no**
cross-root term. Identical `H` in every arm.

**Freeze contract:** once `E_i` exists, no later module may modify it. No
pair→centre, no centre update, no recurrence, no relation refresh, no attention,
no pair-conditioned local update, no context writeback.

### 2.4 Static composition (one read-only pass)

Pure-topology pair relation `rho_ij ∈ R^18` = the strict-static S0 relation with
**`path_bond_mean` (4) and adjacent-bond one-hot (4) deleted**: distance one-hot
(8), `log1p(distance)` (1), overlap (5), boundary overlap (3),
`log1p(path_count)` (1).

```
c_ij = F([ E_i+E_j (48) ; |E_i-E_j| (48) ; E_i⊙E_j (48) ; rho_ij (18) ]) -> R^48
```

`F` is a shared MLP `162 -> 64 -> 48`, SiLU, computed exactly once per unordered
root pair. No `c_ij` state is fed back anywhere.

### 2.5 Reader

```
R_unary  = [ mean_i E_i ; max_i E_i ]                  # 96
R_pair   = [ mean_{i<j} c_ij ; max_{i<j} c_ij ]        # 96
R_topo   = pure-topology global vector (8)             # n, m, cycle rank,
                                                       # degree mean/std/max,
                                                       # log1p(n), log1p(m)
y_hat    = G([R_unary ; R_pair ; R_topo])              # 200 -> 64 -> 1
```

`G` is a shared MLP. No typed global bypass.

---

## 3. Arms and controls (Gate 2)

One composition backbone (`H`, `F`, `G` identical, same seed / data order /
protocol):

| arm | role coordinate |
|---|---|
| **C0** coarse-only | `r^V = one_hot(shell,3)`, `r^E = one_hot(shellpair,5)` (no learned fine role) |
| **CD** DenseRole | `alpha^V = M_V b^V`, `alpha^E = M_E b^E` (learned dense projection, 11→16 / 15→16 = 416 params) |
| **CK** SparseDict | `alpha = tied-IHT_K16_s4(D)`, `D` K-SVD-fitted, frozen (416 params) |

CD and CK are **exactly** parameter-matched (416 role params each). C0 has no
role projection; its environment MLP first hidden width is chosen at build time
so that `|params(C0) - params(CK)| / params(CK) <= 0.02`, and the exact counts
are reported. If that tolerance cannot be met the mismatch is reported and the
comparison is labelled parameter-unmatched.

Controls on the **best non-dictionary arm and CK** (same weights seed):

* **BAG** — `R_pair` removed (unary environment pooling only).
* **SHUFFLE** — environments untouched; within each molecule a random
  permutation `π` maps `E'_i = E_{π(i)}` while `rho_ij` keeps its original
  `(i,j)`; `c_ij = F(E'_i, E'_j, rho_ij)`. Same multiset of environments, same
  multiset of relations, correspondence destroyed. `R_unary` is unaffected, so
  the control isolates composition.
* **CHEM-SHUFFLE** (mechanism witness) — within each molecule, permute atom
  categories and bond categories across occurrences (fixed marginals, fixed
  topology). `E_i` must change; prediction must degrade.

---

## 4. Splits

* `fit` = official-train molecules 0..7999; `monitor` = official-train molecules
  8000..9999. Official valid untouched.
* **Gate 1** fits its K-SVD dictionaries on all 8000 `fit` molecules and reports
  reconstruction on `fit` and `monitor` (no `y` used anywhere).
* **Gate 2** train = first 2000 molecules of `fit`; dev = first 500 molecules of
  `monitor`. Gate-2 dictionaries are re-fit **only** on the 2000 Gate-2 train
  molecules (no leakage from dev or from the rest of `fit`).
* Gate 2 is a **relative** screen only. Absolute MAE is reported as context and
  is explicitly **not** comparable to the strict-static S0 band (10 000 train).

---

## 5. Gate 0 — correctness, CPU only, all must pass

* **Purity** — changing atom/bond categories leaves `phi`, `psi` and every
  `alpha` bit-identical.
* **Chemistry isolation** — changing topology while keeping the category
  multiset leaves the raw chemical primitive tensor unchanged.
* **Assignment sensitivity** — fixed topology, fixed atom multiset, fixed bond
  multiset, shuffle chemistry placement → node/edge binding and every `E_i`
  change.
* **Environment freeze** — changing other environments / pair relations leaves a
  fixed `E_i` **bit-identical**.
* **Static contract** — monkeypatching any pair→centre function to `raise`
  leaves the forward pass successful.
* **Relabel invariance** — node relabel / edge reorder → prediction invariant.
* **Sparse correctness** — every code satisfies `||alpha||_0 <= s` exactly.
* **Gradients** — a real batch backward gives finite / non-zero gradients for
  node dictionary, edge dictionary, chemical embeddings, environment MLP, static
  pair composer and reader.

Any failure: **STOP**.

## 6. Gate 1 — label-free dictionary + role recoverability, CPU

Report per role (node, edge): `E_rec = ||phi - D alpha||^2 / ||phi||^2` for

* K-SVD `D`;
* matched random unit-normalized dictionary;
* optional PCA-16 dense reference (report-only).

Health: active atoms, dead atoms, argmax-used atoms, exact sparsity, effective
usage, code entropy / concentration (report-only).

Coarse-role recoverability (train-fit simple linear/logistic probe, diagnostic
only):

* node: `alpha^V -> shell` (3-class);
* edge: `alpha^E -> shellpair` (5-class).

**Frozen Gate-1 criteria (all must hold):**

1. node `E_rec(dev) <= 0.25` and edge `E_rec(dev) <= 0.25`;
2. `E_rec(K-SVD) <= 0.10 * E_rec(random)` for both roles;
3. used atoms `>= 12 / 16` for both roles;
4. exact `||alpha||_0 = s` for every code;
5. node `shell` macro-F1 `>= majority_class_macro_F1 + 0.15`;
6. edge `shellpair` macro-F1 `>= majority_class_macro_F1 + 0.15`.

Any failure: **STOP** (no `K`/`s` rescue).

## 7. Gate 2 — cheap internal task screen, CPU

Protocol (frozen): Adam `lr 1e-3`, `wd 1e-5`, batch 64, L1 loss, grad clip 5.0,
no scheduler, up to 60 epochs, best-dev checkpoint; for every configuration a
fixed equal-weight Top-5 **epoch checkpoint soup** by dev MAE. One seed (0). Same
data order, same initialization seed for every arm.

**Environment hypothesis**

* `CHEM-SHUFFLE` on the best arm must degrade dev MAE by
  `>= 0.02` (absolute) or `>= 10 %` relative, whichever is smaller.
* If chemistry placement does not materially matter → environment hypothesis
  unsupported.

**Composition hypothesis**

* `TRUE` must beat `BAG` by `>= 0.003` MAE and `TRUE` must beat `SHUFFLE` by
  `>= 0.003` MAE.
* If `TRUE ≈ BAG` or `TRUE ≈ SHUFFLE` → composition unsupported → **STOP**.

**Dictionary hypothesis (screen only)**

* `CK` must be no worse than `CD + 0.002` MAE (i.e. `CK - CD <= 0.002`), and the
  dictionary must be alive (participation > 0; `CHEM-SHUFFLE` and a
  dictionary-neutral intervention must both move the prediction).
* If `CD` is clearly better than `CK` → record
  `ENV_COMPOSITION_SUPPORTED_DENSE_NOT_DICT` and **STOP** the dictionary claim
  (no `K`/`s` rescue).

**Fast STOP / level rules**

* if the candidate is a large band worse than the strict-static historical
  behaviour at matched budget, or train is very low while dev is very bad
  (overfit), or a dictionary branch collapses → STOP;
* if `TRUE ≈ BAG` or `TRUE ≈ SHUFFLE` → STOP, and **do not** add recurrence.

## 8. Gate 3 — official-valid seed-0 formal GPU (only if Gates 0–2 all pass)

Follow `remote-research-runner` exactly: clean committed revision → preflight →
deploy → GPU smoke → formal detached run → pull → local durable analysis.
Buy **seed 0 only**, at most two arms (best non-dictionary pure control and the
`CK` candidate). Official valid only. No official test. No seed 1 unless the
seed-0 gate explicitly authorizes it.

Performance context (not a hard gate, no post-hoc rescue):
strict-static S0 seed0 soup `0.140794`; levels A `<= 0.145`, B `<= 0.140`,
C `0.13x`, D `0.12x`.

## 9. Durable artifacts (to be produced)

* this prior-artifact audit + this preregistration;
* architecture / purity contract (in code + tests);
* implementation + targeted tests;
* parameter accounting;
* label-free dictionary report (Gate 1);
* Gate-2 small-train report (all arms + controls);
* formal seed-0 results (only if reached);
* mechanism interventions (chem-shuffle, assignment shuffle, neutral dictionary);
* decision JSON + Markdown, claim/decision YAML, `STATE.yaml` update.

Every formal result records: local commit, remote commit, dirty flag, GPU, seed,
split fingerprint, reused baselines, baselines explicitly not re-run, wall time,
peak memory, `official_test_loaded = false`, and an exact stop reason.

## 10. Final verdict vocabulary (only these)

`PURE_ENV_COMPOSITION_NOT_VIABLE` ·
`ENVIRONMENT_SUPPORTED_COMPOSITION_NULL` ·
`ENV_COMPOSITION_SUPPORTED_DENSE_NOT_DICT` ·
`PURE_SPARSE_DICT_ENV_COMPOSITION_SUPPORTED` ·
`TASK_COUPLED_DICT_SUPPORTED`

No post-hoc fuzzy conclusion is permitted.
