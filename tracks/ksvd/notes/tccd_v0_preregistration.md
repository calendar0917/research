# Pre-registration — TCCD-v0: Task-Coupled Compositional Dictionary (ZINC)

Round name: **TCCD-v0** (*Task-Coupled Compositional Dictionary, v0*).
Written **before** any Gate 0/1/2/3/4 run, following the
`remote-research-runner` skill (local code → committed revision → remote
GPU1 compute → pull → local analysis). This note modifies no historical record.

Protocol version: `tccd_v0`. Study: `zinc-context-gap`. Regime: deterministic
A100 **GPU1**, canonical PyG ZINC `subset=True` official **train 10 000 /
valid 1 000**. **Official test is never loaded, instantiated or referenced.**
All data / split / soup semantics are inherited from the canonical ZINC loader
and `tracks/ksvd/protocols/zinc-context-gap.yaml`.

Everything below is **frozen**. No K / radius / sparsity / relation-set /
reader / loss-weight sweep is permitted in this round. A gate FAIL ends the
round at that gate.

---

## 0. Hypothesis

> A molecule can be represented as a set of **reusable local chemical
> environments learned from data**, plus **explicit observed composition
> relations** describing how those environments overlap and are connected in
> the real molecular graph.

Target representation:

    molecule = learned local environments  +  their observed composition.

**Explicitly not tested:** "a dictionary as an ordinary GNN hidden layer"
(GSCN-v0, closed), "a dictionary feature added to a raw GNN" (PSD-v0, closed),
and **any learned node-to-node message passing** is forbidden for the whole
round.

---

## 1. Frozen model: TCCD-v0

### 1.1 Local observation window

For every atom (root) `v` of a molecule take the **maximum radius-2 attributed
rooted patch** `P_v` (BFS ≤ 2). Input may use only:

* raw atom categorical attribute,
* raw bond categorical attribute,
* local adjacency / incidence,
* root identity, shell identity, valid mask.

Forbidden inputs: ring statistics, shell histograms, compact-v4 global
features, engineered pair/path descriptors, 146-D / 23-D summaries, any global
graph statistic. The runner **asserts** that only the above enter `x_v`.

### 1.2 Patch → fixed coordinate vector `x_v ∈ R^F` (frozen construction)

Reuses the repo's exact rooted **colored-incidence canonicalization**
(`tracks/ksvd/code/run_wholegraph_canonical_registration_audit.py::canonical_atom_order`,
pynauty colored incidence), applied to the induced patch.

1. Colored incidence vertices: atom `("node", is_root, distance_from_root,
   atom_type)`; bond vertex `("edge", bond_type)`; incidence edges. This is the
   repository's `tracks/ksvd/experiments/luyin16/typed_patch_tokenizer.py::
   build_colored_incidence` construction (root identity and shell identity are
   part of the colour, which is what makes the canonical labelling
   root/shell-preserving).
2. `pynauty.canon_label` on that colored incidence graph gives the exact
   canonical labelling; the relative rank `rank(v)` of an atom inside the
   canonical atom sequence is a complete permutation-invariant invariant.
3. Slot order: root at slot 0 (shell 0), then shell-1 atoms by increasing
   canonical rank, then shell-2 atoms by increasing canonical rank.
4. Capacity `M = max radius-2 patch atom count over official train`
   (label-free; measured **14** on the canonical ZINC train 10 000). If a
   later-split patch exceeds `M`, it is truncated in the frozen slot order and
   the truncation count is reported. No truncation is expected on train/valid.
5. Block layout of `x_v` (float32), concatenated in this order:

   | block | dim | content |
   |---|---|---|
   | `topology` | `P = M(M-1)/2` | upper-triangle edge indicators of the patch |
   | `bond` | `3P` | bond-category one-hot (train-observed categories; all-zero if no edge) |
   | `atom` | `M·A` | atom-category one-hot (`A` = train-observed atom categories) |
   | `shell` | `3M` | shell-membership one-hot (shell 0/1/2; zero row = padding) |
   | `mask` | `M` | valid-slot indicator |

   With the canonical ZINC train values (`A=21`, `M=14`): `F = 4·91 + 14·21 +
   14·3 + 14 = 714`.

The coordinate construction is **exact-canonical and permutation-invariant by
construction**. This note explicitly **does not assume** that this makes the
Euclidean coordinate space a good dictionary domain — that is Gate 1's job
(§4.4 of the round brief).

### 1.3 Local dictionary (frozen)

* `K = 64`, target sparsity `s = 8`.
* `D ∈ R^{F×K}`, `x_v ≈ D c_v`, `‖c_v‖_0 ≤ s`.
* **Gate 1** uses ordinary detached K-SVD + OMP.
* **End-to-end (Gate 2/3)** uses **tied unrolled iterative hard thresholding**
  sharing the same `D`:

      c^{t+1} = H_s[ c^t + η_t Dᵀ (x − D c^t) ],   t = 0..T−1,
      η_t = 1 / (σ_max(D)² + ε)   (computed detached, 30 power iterations
                                   from a frozen deterministic start vector
                                   `cos(2πk/K)`, no RNG),

  with `H_s` keeping the `s = 8` largest-|·| coefficients. Frozen
  `T = 10`. **No** free LISTA `W_e/W_s`, **no** independent learned encoder
  bypassing `D`. `D` columns are re-normalized to unit norm after every
  optimizer step; if a column norm would fall below `1e-6` it is reset to a
  fresh normalized random column from the frozen RNG stream (and the event is
  logged).

### 1.4 Composition representation — no message passing

Local codes of a graph: `C_G = [c_1;…;c_n] ∈ R^{n×K}`.

* First-order: `m_G = Σ_v c_v ∈ R^K`.
* Relation contractions: `M_G^{(r)} = C_Gᵀ R_G^{(r)} C_G ∈ R^{K×K}`, one-shot,
  no learned message passing, no trainable node-to-node weights in `R`.

Frozen relation set (**5 relations**):

* **R0 — overlap/incidence.** `B_G ∈ {0,1}^{n×N}` patch↔original-atom
  incidence; `R_∩ = B_G B_Gᵀ`, **diagonal removed** (self-overlap is not a
  relation). Pre-registered once.
* **R1 — native bond attachment.** For each bond category `b` observed on
  official train (`3` for ZINC): `A_G^{(b)}[i,j] = 1` iff roots `i,j` are
  joined by a real bond of type `b`.
* **R2 — one fixed global relative-position operator.** Shortest-path distance
  `d_G(i,j)` between roots in the **original** molecule,
  `R_geo[i,j] = exp(−d_G(i,j)/τ)`, **τ = 1 frozen**. Diagonal = 1.

No distance buckets, cycles, or path histograms. No trainable relation
weights. No learned message passing on `c_v`.

### 1.5 Whole-graph code and reader

    h_G = [ m_G, vec_sym(M_G^{(∩)}), vec_sym(M_G^{(b1)}), …, vec_sym(M_G^{(geo)}) ]

where `vec_sym` stacks diagonal + strict upper triangle (`K(K+1)/2 = 2080`
entries each). `dim h_G = 64 + 5·2080 = 10464`.

Frozen reader: **single linear head**

    ŷ = b + uᵀ m_G + Σ_r ⟨W_r, M_G^{(r)}⟩.

No deep reader, MLP, GNN, transformer, raw-graph residual bypass, or
concatenated handcrafted graph statistics.

### 1.6 Task-coupled training (Gate 3 `TASK-D` only)

    L = L_task + λ · L_rec,
    L_rec = (1/N) Σ_v ‖x_v − D c_v‖² / (‖x_v‖² + ε),
    λ = (detached bootstrap L_task) / (detached bootstrap L_rec)

evaluated once on a fixed calibration batch at initialization, then frozen.
No extra L1 (hard sparsity is already exact). No `λ` search on valid.

### 1.7 Arms compared

* **BAG** — `h_G = m_G` only (Gate 2 secondary evidence; parameter count not
  used as a primary claim).
* **REL** — full `h_G` (primary).
* **REL-SHUFFLE** — `REL` with a fixed random row permutation of `C_G` per
  graph (same code multiset, same dimensionality, same parameter count, same
  reader; environment↔topology correspondence destroyed).
* **FROZEN-D** — Gate 1 dictionary, `D` frozen, tied IHT.
* **TASK-D** — `D` initialized from the *same* `FROZEN-D` dictionary and
  updated by `L`.
* **DENSE** — tied **linear dense** control: `z_v = Dᵀ x_v` (width `K`,
  exactly parameter-matched dictionary matrix), no sparsity, identical
  composition relations and reader.

---

## 2. Data split (frozen)

* Canonical ZINC official train 10 000 / valid 1 000. Official test never
  loaded (`official_test_loaded` recorded `false`).
* **Gate 1 / Gate 2 / Gate 3** use an internal split of official train:
  deterministic permutation `seed = 20260922`; **internal-train = 8 000**,
  **internal-dev = 2 000**. `y` is **never** used in Gate 1.
* **Gate 4** (only after all prior gates PASS) uses full official train and
  official valid with the canonical soup/early-stop protocol; official test
  still never loaded.
* Seeds: primary seed **0**. Gate 2 ambiguity rule may add exactly **one**
  paired seed (`1`) — nothing else.
* Determinism: `_seed_everything(seed)`; torch threads fixed; no
  non-deterministic algorithms are requested.

---

## 3. Gate ordering, exact GO / STOP conditions

A gate is evaluated in the frozen order. **Any FAIL stops the round.** No
architecture/module/hyperparameter rescue is permitted after a FAIL.

### Gate -1 — canonical GPU1 baseline (reference only)

*PASS* iff a same-era, same-protocol, same-A100-regime, official-test-unread
canonical strong ZINC baseline exists. Frozen reference candidate:
**B-Full / shared-structural Top-5 soup valid MAE = 0.119818**
(84 495 params; deterministic A100; seed 0; official test never loaded;
`claim-local-token-null-20260919`; also listed in `STATE.yaml`
`development_references`). A same-revision GPU1 re-run of the canonical
strong baseline is required only if Gate 4 is reached.

### Gate 0 — correctness (local targeted tests + GPU1 smoke)

Must all pass, else STOP (implementation failure):

1. **Permutation invariance** — for ≥ 200 randomly relabelled
   (molecule, root) instances: `x_v` equal (max abs diff ≤ 1e-6), OMP/IHT
   code equal (support Jaccard = 1.0), `CᵀRC` equal (≤ 1e-6), prediction equal
   (≤ 1e-6).
2. **Batching invariance** — single-graph vs batched execution, max Δprediction
   ≤ 1e-5.
3. **Relation correctness** — jointly permuting `(C, R, B)` leaves `CᵀRC`
   numerically identical (≤ 1e-6).
4. **Sensitivity sanity** — permuting only the rows of `C` (not `R`) must
   change `CᵀRC` by a clear margin (relative Frobenius change ≥ 0.05 on a
   non-degenerate instance); if it does not, the composition representation is
   degenerate → STOP.
5. **Gradient sanity (TASK-D)** — reconstruction gradient reaches `D`, task
   gradient reaches `D`, `D` update non-zero, no column-norm collapse, and
   codes keep **exact** sparsity (`‖c_v‖_0 = 8` for every patch).

### Gate 1 — local dictionary domain (official train only, no `y`)

Frozen: radius 2, `K=64`, `s=8`, the §1.2 coordinate construction. Detached
K-SVD/OMP only. K-SVD: `D` initialized from `fit` patches by a deterministic
random normalized matrix (`seed 20260922`), then **12 chunk epochs** over the
8 000 internal-train graphs in chunks of 16 384 patches (fixed order
`seed 20260922`), OMP `s=8`; atom columns renormalized after every chunk.

For each numbered test a quantitative PASS condition is frozen:

1. **Permutation stability** — 100 internal-dev molecules × 20 random node
   relabelings. PASS iff `x_v` max abs diff ≤ 1e-6 **and** mean OMP support
   Jaccard = 1.0 **and** ≥ 99 % of patches have Jaccard = 1.0.
2. **Held-out reconstruction** — relative error
   `E = mean_v ‖x_v − D c_v‖² / ‖x_v‖²` on the 2 000 internal-dev molecules,
   with `D` never fitted on them. Compare the learned `D` against the matched
   **random normalized dictionary it was initialized from** (same shape, same
   OMP `s=8`). PASS iff `E_learned / E_random ≤ 0.70` on the held-out patches.
   FAIL if `> 0.90`. (The gain must be on holdout, not only on fit patches —
   both are reported.)
3. **Reuse** — PASS iff all of:
   * dead atoms (support `< 5` patches over internal-dev) ≤ 10 % of `K`;
   * "reused" atoms (support ≥ 20 patches **and** present in ≥ 5 distinct
     molecules) ≥ 60 % of `K`;
   * top-1 atom share of total `Σ_v|c_v|` ≤ 0.20 and top-8 share ≤ 0.80.
4. **Local-coordinate continuity** — the core audit. Patches from a 2 000-graph
   pool are fingerprinted with a typed WL subtree kernel (rounds 0–3) on the
   **raw attributed rooted patch**; *near* pairs = top-scoring non-isomorphic
   patch pairs; *random* pairs = matched on (patch size, root atom category).
   Distance in `x`-space and in OMP `code`-space. AUC that a near pair is
   closer than a random pair. PASS iff **code-space AUC ≥ 0.70**; otherwise the
   gate STOPS (code AUC `< 0.60` is labelled a clear failure; `0.60–0.70` is a
   borderline non-PASS — no seed-1 is allowed at Gate 1). `x`-space AUC and a
   graded tail stratum (rank 801–1600 of the WL ranking) are reported alongside
   as diagnostics.
5. **Atom semantics** (report-only, not a STOP trigger) — for the top-8 used
   atoms, the concentration of exact radius-2 corrected typed canonical keys
   among top-50 activating real patches vs a random-patch baseline.

**Gate 1 STOP:** any of §1/§2/§3/§4 failing → STOP the whole TCCD-v0 round with
the verdict *"fixed-coordinate raw local patch is not a suitable current
K-SVD domain"*. No K / radius / ordering / encoder / reader change may be used
to rescue it in this round.

### Gate 2 — does composition add information? (only if Gate 1 PASS)

Freeze `D` (no task gradient to `D`). Same tied sparse encoder. Internal
split, seed 0. Run `BAG`, `REL`, `REL-SHUFFLE`.

Primary statistic `Δ = MAE_REL-SHUFFLE − MAE_REL` (internal-dev, best
checkpoint):

* PASS iff `Δ ≥ 0.015`;
* FAIL iff `Δ < 0.005`;
* `0.005 ≤ Δ < 0.015` → one paired seed (`1`); PASS iff both seeds same sign
  and mean `Δ ≥ 0.010`, else STOP.

`BAG` is secondary evidence only. If Gate 2 FAILs, STOP with the verdict
*"explicit composition relations add insufficient predictive information"*; no
distance buckets / cycle features / path histograms may be added.

### Gate 3 — task coupling and dictionary uniqueness (only if Gate 2 PASS)

Freeze relation representation, reader, `K`, `s`, split. Compare `FROZEN-D`,
`TASK-D`, `DENSE` (same `D` initialization for FROZEN-D/TASK-D).

* **Test A (task coupling):** PASS iff
  `MAE_FROZEN-D − MAE_TASK-D ≥ 0.005`; else STOP (*no evidence of task
  coupling benefit*).
* **Test B (dictionary uniqueness):** if
  `MAE_TASK-D > MAE_DENSE + 0.005` → STOP (*dictionary inductive bias has no
  predictive advantage*). If `|MAE_TASK-D − MAE_DENSE| ≤ 0.005`, the only
  permitted claim is equal predictive ability plus sparsity/reuse/
  interpretability — **not** superiority.
* **Rate control:** `FROZEN-D` and `TASK-D` MUST both use exact `s = 8`. If
  the implementation makes the rates differ, the comparison is invalid: fix
  rate matching and re-run the whole Gate 3. Compression difference must never
  be reported as a task-aware gain (PSCD-RM-v0 lesson).

### Gate 4 — absolute predictive performance (only if Gate 3 PASS)

Architecture fully frozen. Canonical full ZINC training protocol: full official
train, official valid, canonical soup / checkpoint selection, GPU1. Official
test still forbidden. `Δ_abs = MAE_TCCD − MAE_strong baseline`:

| band | condition |
|---|---|
| VERY STRONG | `Δ_abs ≤ 0` |
| STRONG | `0 < Δ_abs ≤ 0.005` |
| COMPETITIVE | `0.005 < Δ_abs ≤ 0.015` |
| NOT VIABLE | `Δ_abs > 0.015` |

`NOT VIABLE` stops the architecture even if every mechanism gate passed
(conclusion: *mechanism supported, absolute predictive power insufficient*).
No reader / GNN stacking to rescue performance.

### Final claim gate

The TCCD-v0 claim may be made only if all of: Gate 1 PASS; atom reuse; atom
structural coherence; `REL ≫ REL-SHUFFLE`; `TASK-D ≫ FROZEN-D`; `TASK-D` not
materially worse than matched `DENSE`; rate controlled; `Δ_abs ≤ 0.015`; no raw
bypass; no learned message passing; no handcrafted-statistics rescue.

---

## 4. Metrics, diagnostics, artefacts

* Primary metric: MAE (ZINC target), lower better. Internal-dev for Gates 1–3,
  official valid for Gate 4. Best-checkpoint and Top-5-soup reported.
* Gate 1 reports: `F`, `M`, capacity truncation count, fit/holdout relative
  reconstruction error for learned vs random `D`, atom usage / molecule
  coverage / support frequency / coefficient entropy, continuity AUCs,
  atom-semantics concentration.
* Every run records: commit, branch, dirty flag, device, GPU name, seed,
  protocol version, dataset/split fingerprint, config, metrics, wall time,
  peak GPU memory.
* Artefacts: `tracks/ksvd/results/tccd_v0/` (git-ignored) plus a durable note
  `notes/tccd_v0_analysis.md` and a `STATE.yaml` update.

---

## 5. Explicitly forbidden in this round

* Any K / radius / sparsity / τ / relation-set / reader / loss-weight /
  dictionary-size sweep.
* Adding distance buckets, ring/cycle or path statistics, shell histograms,
  compact-v4 features, handcrafted descriptors, or raw-graph bypass.
* Learned node-to-node message passing (GCN/MPNN/GAT/transformer), free
  LISTA encoders, learned encoders around `D`.
* Reading or evaluating official test.
* Re-opening `D` for selection after a gate FAIL; "rescuing" a failed gate.
* Comparing a GPU delta against a historical CPU number.

**Frozen.** After this note is committed, no architecture choice may change.
An implementation bug may be fixed in a new commit and the affected gate
re-run from scratch; a bugfix may not change the architecture.
