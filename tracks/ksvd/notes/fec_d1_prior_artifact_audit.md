# FEC-D1 — prior-artifact audit

Round: **FEC-D1** (*Localized Sparse Structural Dictionary Binding*),
protocol id `fec_d1`, study `zinc-context-gap`.

Written **before any code change, binding-cache build or training run**.

Lineage HEAD at audit time: `9d3b2f69baf202427a046816a9c0fc442b1380f1`
(`record(fec-d0): residual structural dictionary audit — STOP NO_HELDOUT_SUBROLE_SIGNAL`).
Worktree contains one untracked, git-ignored artifact directory
`tracks/ksvd/results/fec_s1/` (prior round output); no tracked file is modified.

This audit does three jobs:

1. prove every **frozen artifact FEC-D1 depends on exists in-tree** and is
   replayable, or STOP;
2. prove FEC-D1 is **not a re-run of any closed dictionary line**;
3. record honestly that the *previous* `FEC-D1` idea (the FEC-D0 residual
   design) is **not** what this round executes, and under which authority the
   new design is run.

`official_test_loaded = false` on every artifact cited below. FEC-D1 must never
load the official test split.

---

## 0. What FEC-D1 is (single scientific variable)

> A node-centric sparse pure-topology dictionary coordinate `alpha_v ∈ R^32`,
> already independently validated as a reusable structural vocabulary (SDB-v0),
> is **localized to each rooted chemical environment**: for every root `i` and
> shell `k` we form the within-shell centred binding of `alpha_v` against the
> primitive atom chemistry `q_v`, and add the resulting local refinement to the
> frozen FEC-S1 local environment `e_i^shared ∈ R^24`.

Path (frozen):

```
phi_v^65 --D_SDB--> alpha_v^32
alpha_v + shell(i,v) + q_v --(within-shell centred binding)--> C_i^D ∈ R^2688
C_i^D --W1(2688->8)--> u_i --W2(8->24)--> Δe_i
e_i^new = e_i^shared + Δe_i              (frozen FEC-S1 downstream)
```

This is **not** a whole-graph prediction residual, **not** a pair-kernel
dictionary, **not** a mixed chemistry dictionary, **not** message passing,
**not** recurrence, **not** a context writeback.

---

## 1. Frozen-artifact existence audit (blocking)

| artifact | required | path | status |
|---|---|---|---|
| FEC-S1 seed-0 best checkpoint (replayable) | yes | `tracks/ksvd/results/fec_s1/states/fec_s1_seed0_selection_state.pt` | **PRESENT** (`best_valid_mae = 0.1367825070246472 @238`; replay 0.13678248443448685, `|Δ| 2.26e-08` on local CPU) |
| SDB frozen K-SVD dictionary `D ∈ R^{65×32}` | yes | `tracks/ksvd/results/sdb_v0/dictionary.pt` (`D_ksvd`) | **PRESENT** (`K=32`, `s=8`, `dict_seed=20260924`, `phi_dim=65`) |
| SDB PCA-32 reference | yes | same `dictionary.pt` (`pca_mean`, `pca_components`) | **PRESENT** (train-fit, frozen) |
| exact `phi_v ∈ R^65` transform | yes | `fsar_r2_ar0.build_phi` + FSAR cache `results/fsar_r2_ar0/cache/{train,valid}.pkl.gz` | **PRESENT**; FSAR phi is bit-identical to a fresh `build_phi` on the raw graph (probe: max abs diff `0.0`), node order = root/patch order |
| FEC-S1 encoded cache (patch order) | yes | `results/zinc_static_dictionary_pair/cache/encoded_{train,valid}.pt` | **PRESENT** (`n_train=10000`, `n_valid=1000`, `official_test_loaded=false`) |
| SDB sparse coding semantics | yes | `tccd_v0.omp_codes` via `sdb_v0.omp_codes` | **PRESENT** (exact top-`s`, float64) |
| FEC-D0 negative verdict recorded | yes | `results/fec_d0/`, `STATE.yaml::fec_d0` | **PRESENT** `FEC_D0_BASIS_HAS_NO_HELDOUT_SUBROLE_SIGNAL` |
| no pre-existing equivalent localized-binding experiment | yes | repo-wide grep for `fec_d1` / `d1_stat` / localized binding | **CONFIRMED ABSENT** (see §4) |

**No missing frozen artifact. The round is not blocked.**

If any of these had been missing, the instruction was to STOP rather than
refit/retrain a "similar" version. Nothing is refit here:

* `D_SDB` is **read** from `dictionary.pt`; K-SVD is never re-run.
* `PCA32` is **read** from the same file; PCA is never refit.
* `phi_v` is **read** from the FSAR cache; `phi65` construction is not changed.
* the FEC-S1 checkpoint is **read**; FEC-S1 is never retrained.
* the sparse solver is **called**; `K`, `s`, normalization and coding
  semantics are untouched.

---

## 2. Why FEC-D1 is not a re-run of any closed line

The new object is the **root-conditioned localization of an already-validated
node-centric dictionary coordinate, bound to primitive chemistry inside
environment formation**. Every prior dictionary round differs in *where the
dictionary lives*:

| line | dictionary object / placement | why FEC-D1 differs |
|---|---|---|
| **TCCD-v0…v7** | dictionary encodes the whole **attributed** local patch `x_v` (chemistry included) | FEC-D1's dictionary is **chemistry-free** and node-centric (`phi_v ∈ R^65`); chemistry is met for the first time at **environment formation**, not inside the dictionary |
| **SDB-v0** | frozen `D_SDB` + chemistry form **one whole-graph centred residual** `C_D` added to a strong backbone (Stage 4: `+0.002381 < 0.003` gate) | FEC-D1 does **not** add a graph prediction residual; `C_i^D` is a **per-root local statistic** injected into the **local environment slot** `e_i`, before any pair/reader computation |
| **SDPK-v0 / SRDA-v0** | dictionary acts inside the **pair kernel / pair relational algebra** | FEC-D1 has no pair object; the statistic is strictly pre-pair and per-root |
| **PEC-v0 / PEC-C1 / PEC-I1 / PEC-CK** | `R^11` occurrence basis → dictionary carries the **fine role coordinate**, replacing the coarse role, bound to chemistry in an environment | FEC-D1 keeps the coarse rooted role `shell(i,v)` as a **given** and adds a *dictionary-specific subrole* channel; it is an **additive refinement of FEC-S1**, not a replacement of the role axis |
| **DTX-v0 / no-ring cross** | dictionary × generic topology role cross at graph level | FEC-D1 is within-environment formation, not a graph-level cross |
| **FEC-D0** | within-coarse-role residual dictionary on `b^V ∈ R^11` (R11 residual basis) | FEC-D0 proved the **R11 residual domain** is low-rank/degree-dominated and gave `FEC_D0_BASIS_HAS_NO_HELDOUT_SUBROLE_SIGNAL`. FEC-D1 **does not use R11**; it uses the already-validated `phi^65 → D32` vocabulary (SDB) |

New decisive content: the **localization + assignment binding is the object**.
`C_i^D` is *within-shell centred*, so it cannot steal any coarse shell ×
chemistry marginal that FEC-S1 already has; the only information it can carry
is the **structural-subrole ↔ chemistry assignment** inside the environment.

---

## 3. Authority for this round (honest record)

The prior `FEC-D1` name was used in two frozen notes for a **different**
proposal — `FEC-D0`-style `r_new = r_coarse + g · r_dict`, degenerating to
FEC-S1, on the **R11 residual basis**:

* `notes/fec_s0_preregistration.md` §195–197 (proposal only, not implemented);
* `notes/fec_s1_preregistration.md` §348 and `notes/fec_s1_prior_artifact_audit.md`;
* `notes/fec_d0_analysis.md` §213 / `STATE.yaml::fec_d0.next_step`:
  *"Do NOT implement FEC-D1. FEC-D1 is authorized only by
  `FEC_D0_RESIDUAL_STRUCTURAL_DICTIONARY_QUALIFIED`, which did not fire."*

That gate did **not** fire, and it governs the **R11 residual design**, which
this round does not execute. FEC-D1 (this round) is a **new pre-registration**
with a **new scientific variable** (localized binding of the SDB-validated
`phi^65 → D32` vocabulary at environment formation), explicitly commissioned by
the round brief. It makes **no claim** of authorization from
`FEC_D0_RESIDUAL_STRUCTURAL_DICTIONARY_QUALIFIED`.

Consequences that are binding:

* the FEC-D0 verdict and its claim/decision stay **frozen and untouched**;
* `R11` is not used, no dictionary is refit, no `K`/`s` sweep, no task coupling,
  no edge dictionary;
* this is a single-variable test with a matched PCA-32 dense control and a
  matched W2 = 0 containment of FEC-S1.

---

## 4. No equivalent experiment exists in-tree

Repo-wide search (excluding generated `results/`):

```
grep -rn "fec_d1\|FEC_D1\|d1_stat\|localized_binding" tracks/ksvd --include=*.py --include=*.md
```

returns only the two *proposal* references above (`fec_s0_preregistration.md`,
`fec_s1_preregistration.md`, `fec_s1_prior_artifact_audit.md`,
`fec_d0_preregistration.md`, `fec_d0_analysis.md`) and no code. There is no
`d1_stat`, no localized within-shell dictionary-binding statistic and no
localized-binding runner/module/test. The new object is unimplemented.

---

## 5. Reused infrastructure (read-only, correctness-tested)

| purpose | source | policy |
|---|---|---|
| frozen sparse dictionary + PCA32 + OMP | `sdb_v0.load_dictionary`, `sdb_v0.omp_codes` (→ `tccd_v0.omp_codes`) | read / call verbatim |
| train-only RMS scaler | `sdb_v0.fit_rms_scaler` / `apply_rms_scaler` | call verbatim |
| `phi_v^65` | `fsar_r2_ar0.build_phi` (node order = graph node order) | call verbatim |
| raw graph / patch order | `zinc_long_range_proxy._data_to_graph`, `zinc_patch_path_pooling._ego_distances` | call verbatim |
| frozen FEC-S1 base + adapter | `fec_s1_shared_local_env.build_fec_s1`, `LocalEnvironmentAdapter` | read checkpoint, wrap adapter |
| FEC-S1 protocol | `zinc_static_dictionary_pair.OPTIMIZED_PROTOCOL` | reuse verbatim |

No new sparse solver, no new dictionary initializer, no `K`/`s`/rank sweep.

---

## 6. Audit conclusion

1. **All frozen artifacts exist and are replayable**; nothing must be
   refit/retrained. The round is not blocked.
2. **No prior round is equivalent**: TCCD (attributed patch), SDB (whole-graph
   residual), SDPK/SRDA (pair kernel), PEC (role replacement), DTX (graph cross)
   and FEC-D0 (R11 residual) all differ in object and placement. Localized
   within-shell dictionary-binding at environment formation is new.
3. **The FEC-D0 STOP is respected**: it governed the R11 residual design, which
   is not run; R11 is not used; the FEC-D0 verdict stays frozen.
4. **`official_test_loaded = false`** and the official test blocker is a hard
   gate of this round.
