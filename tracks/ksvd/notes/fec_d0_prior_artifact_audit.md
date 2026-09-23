# FEC-D0 — prior-artifact audit

Round: **FEC-D0** (*Within-Coarse-Role Residual Structural Dictionary Audit*),
protocol id `fec_d0`, study `zinc-context-gap`.

Written **before any code change, dictionary fit or experiment run**.
Lineage HEAD at audit time: `ca2e947013263a2590c7ad4edffff03a46e6b830`
(`record(fec-s1): shared local environment replacement — STRONG (soup 0.130422),
seed1 authorized not executed`). Worktree contains one untracked, git-ignored
artifact directory `tracks/ksvd/results/fec_s1/` (prior round output); no tracked
file is modified.

This audit does **only** two jobs:

1. prove FEC-D0 is not a re-run of a closed line, by locating the *exact*
   prior artifacts it reuses;
2. prove the required rooted structural basis **exists in-tree and is
   root-conditioned, occurrence-level and chemistry-free**, or fail with
   `FEC_D0_ROOTED_BASIS_UNAVAILABLE`.

`official_valid_loaded = false` and `official_test_loaded = false` on every
artifact cited below. No target `y` is read anywhere in this round.

---

## 0. What FEC-D0 is

> After the coarse rooted structural role is *already known* (`shell ∈ {0,1,2}`),
> does the remaining pure-topology variation of a rooted radius-2 node
> occurrence admit a **small, reusable, chemistry-independent, molecule- and
> root-independent sparse structural vocabulary** `D α ≈ φ^⊥`, `K = 16`, `s = 4`?

FEC-D0 is a **label-free dictionary qualification round**: no property training,
no official valid/test, no FEC-S1 retrain, no task coupling. It is the natural
dictionary re-entry *after* FEC-S1 established a strong strict-static,
shared, vocabulary-free environment→composition底座.

---

## 1. Basis audit (the decisive step)

### 1.1 Candidate A — the 146-D `patch_cont` descriptor — REJECTED

| field | value |
|---|---|
| source file | `tracks/ksvd/experiments/luyin16/fec_s0_factorization.py` |
| function | `factorized_shell_descriptor` (L153) |
| dim | 146 |
| semantics | `atom_shell` (0:84 = 3 shells × 28 atom cats), `bond_shell` (84:108 = 6 shell-pairs × 4 bond cats), `root_atom` (108:136 = 28 atom cats), `incident_bonds` (136:140 = 4 bond cats), `scalars` (140:146) |

**Rejected.** Four of its five blocks (140 / 146 coordinates) are explicit
`role × chemistry-primitive` bindings — atom type and bond type one-hots. Only
the 6 scalar coordinates are chemistry-free. §4 of the round brief forbids
`NO atom type / NO bond type / NO aromatic-ring chemistry label`. The 146-D
descriptor is the historical *mixed* local environment, i.e. the very object
FEC-S1 already replaced; it is **not** a pure-topology structural basis.

### 1.2 Candidate B — FSAR explicit rooted **node** basis `b^V_{iv} ∈ R^11` — ACCEPTED

| field | value |
|---|---|
| source file | `tracks/ksvd/experiments/luyin16/fsar_v2.py` |
| function | `_explicit_basis_for_patch(graph, center, radius=2)` (L141) |
| dim | `NODE_BASIS_DIM = 1 + 3 + 1 + 3 + 3 = 11` (L81) |
| conditioning | **root-conditioned occurrence** — one row per node `v` of the radius-2 induced patch of root `center` |
| first audited/used by | FSAR-v2 (`SAE/SABE/SAME` modes), then FSAR-R2-AR0, then PEC-v0/c1/i1 as `b^V` |

Exact coordinate semantics (pure topology only; reads adjacency + rooted BFS
distance + nothing else):

| slice | coordinate | meaning |
|---|---|---|
| `[0]` | `root_indicator` | 1 iff `v == center` |
| `[1:4]` | `shell_one_hot` | BFS distance from root, ∈ {0,1,2} |
| `[4]` | `log1p(degree)` | induced within-patch degree |
| `[5:8]` | `log1p(neighbour_by_shell)` | # induced neighbours at shell 0/1/2 |
| `[8:11]` | `log1p(A^k e_root)` | k = 1,2,3 rooted walk counts |

Purity audit of this basis:

| requirement | status |
|---|---|
| atom type | absent |
| bond type | absent |
| aromatic / ring chemistry label | absent |
| target / | absent |
| learned hidden state / MP state | absent (no `MessagePassing`, no `torch.nn` in function) |
| typed token / parent token | absent |
| ring/cycle pure-topology coordinate | none added beyond the historical 11 |

Rooted walk counts `A^k e_root` are the only "global-ish" coordinate; they are
pure untyped topology and derive from the same rooted radius-2 adjacency used to
define the patch, so they introduce **no chemistry leakage** and **no new
coordinate class** beyond the historically audited basis.

**Verdict: rooted occurrence basis AVAILABLE.** Proceed.

---

## 2. Why FEC-D0 is not a re-run of a closed line

The dictionary object this round is a **root-conditioned occurrence-level
residual structural subrole**, deliberately placed *before* any chemistry
binding. This is a different object from every closed dictionary line:

| line | dictionary object | why FEC-D0 differs |
|---|---|---|
| **TCCD-v0…v7** | fixed-coordinate canonical raw *attributed* patch `x_v ∈ R^714`, K-SVD/OMP + `C^T R C` composition | TCCD codes the complete attributed patch (chemistry included) and failed its Gate-1 continuity check; FEC-D0 codes only the **pure-topology residual subrole within a known coarse rooted role**, no chemistry, no composition, no reader |
| **SDB-v0** | graph/node-level structural dictionary `φ_v ∈ R^65` plus the graph-level assignment tensor `C_D = Σ_v (α_v−ᾱ)(q_v−q̄)^T`, and the strong-backbone residual route | SDB's object is the FSAR-R2-AR0 **patch-aggregated** coordinate; its strong S0 residual route failed. FEC-D0 explicitly forbids reusing `C_D` or the graph-aggregated coordinate; it codes the **occurrence**, not the graph assignment statistic |
| **SDPK-v0 / SRDA-v0** | dictionary promoted into the *pair kernel* / occurrence-level *relation algebra* | pair composition is entirely out of scope this round; FEC-D0 has no pair object at all |
| **PEC-CK (c1/i1), PEC-v0** | frozen sparse dictionary *is* the whole learned role coordinate `[one_hot(shell); α]`, bound to chemistry in an environment | FEC-D0 keeps the coarse exact role as a *given* and only asks whether a residual subrole vocabulary exists; it never forms an environment, never binds chemistry, never trains a reader |
| **DTX-v0** | ring-blind dictionary × generic topology role cross | dictionary placement is a graph-level aligned cross; FEC-D0 is a within-occurrence residual question with a matched-random control |

New decisive content of FEC-D0: the **residualization is the object**. We
*delete* the explicit shell coordinates, standardize per-shell using FIT-only
statistics, and ask whether the remainder admits a reusable sparse vocabulary.
No prior round did this.

---

## 3. Reused infrastructure (read-only, correctness-tested)

| purpose | source | reuse policy |
|---|---|---|
| rooted occurrence basis | `fsar_v2._explicit_basis_for_patch` | call verbatim |
| graph / patch utilities | `zinc_patch_path_pooling` (`_data_to_graph`, `_ego_distances`) | call verbatim |
| K-SVD fit | `tccd_v0.ksvd_fit` (via `sdb_v0.fit_ksvd`), OMP exact top-s, float64, deterministic init | call verbatim, **no re-implementation** |
| sparse coding | `tccd_v0.omp_codes` (exact top-s) | call verbatim |
| random matched dictionary | `tccd_v0.random_normalized_dictionary` (`DICT_SEED`) | call verbatim |
| PCA reference | `sdb_v0.fit_pca_rank` | call verbatim, `rank = min(4, d_res)` |
| deterministic train-internal split | repo convention `SPLIT_SEED = 20260922`, `np.random.RandomState(SPLIT_SEED).permutation(n_train)` → dev = 2000 sorted, fit = remaining 8000 (as in `run_zinc_pair_dict_v0.train_only_split`) | reuse convention |

No new sparse solver, no new dictionary initializer, no sweep of `K`, `s`,
iterations, seed, `λ`, or solver.

---

## 4. Split provenance

The canonical repo train-internal split is the `SPLIT_SEED = 20260922`
permutation convention:

```python
rng = np.random.RandomState(20260922)
perm = rng.permutation(10000)
dev = sorted(perm[:2000])          # HOLDOUT
fit = sorted(set(range(10000)) - set(dev))   # FIT (8000)
```

> Note: PEC-v0's `samples[:8000]` slice is a *contiguous* variant of the same
> 8000/2000 budget, not the permutation split. The permutation convention is
> the one used by `run_zinc_pair_dict_v0` / TCCD and is adopted here, with its
> own fingerprint recorded. This is a *reuse of an existing deterministic
> convention*, not a new split.

Guard: any access to `"val"`, `"test"` or a target tensor raises. FIT statistics
only for residualization / K-SVD / PCA; HOLDOUT is read once for evaluation.

---

## 5. Audit conclusion

1. **`FEC_D0_ROOTED_BASIS_UNAVAILABLE` does NOT fire.** An exact, audited,
   chemistry-free, root-conditioned, occurrence-level pure-topology basis
   exists in-tree: `fsar_v2._explicit_basis_for_patch` → `b^V_{iv} ∈ R^11`.
2. **No prior round is equivalent.** The residual-within-coarse-role
   occurrence dictionary object is new; TCCD / SDB / SDPK / SRDA / PEC / DTX
   differ in object, placement and control (detailed in §2).
3. **All reusable primitives are correctness-tested** in prior rounds and are
   called, not re-implemented.
4. **This round is label-free**: no `y`, no official valid, no official test,
   no FEC-S1 retrain, no task coupling.

Evidence list (read-only): `notes/fec_s1_prior_artifact_audit.md`,
`notes/fec_s0_prior_artifact_audit.md`, `notes/pec_v0_preregistration.md`
(Amendment A1), `notes/sdb_v0_preregistration.md`, `notes/tccd_v0_analysis.md`,
`notes/zinc_static_dictionary_pair_preregistration.md`,
`notes/zinc_static_relational_dictionary_algebra_v0_analysis.md`,
`notes/zinc_no_ring_dictionary_topology_cross_v0_analysis.md`,
`notes/pec_c1_analysis.md`, `notes/pec_i1_analysis.md`,
`tracks/ksvd/experiments/luyin16/fsar_v2.py`,
`tracks/ksvd/experiments/luyin16/sdb_v0.py`,
`tracks/ksvd/code/tccd_v0.py`,
`tracks/ksvd/code/run_zinc_pair_dict_v0.py`,
`tracks/ksvd/experiments/luyin16/fec_s0_factorization.py`.
