# Pre-registration — TCCD-v1: Frozen Composition → End-to-End Dictionary (ZINC)

Round name: **TCCD-v1** (*Task-Coupled Compositional Dictionary, v1*).

This is a **new, explicitly different pre-registered round**, built on new
evidence produced by TCCD-v0. It does **not** amend, reopen, or overrule the
TCCD-v0 pre-registration or verdict; both are retained verbatim.

* TCCD-v0 pre-registration: `notes/tccd_v0_preregistration.md` (unchanged).
* TCCD-v0 analysis / frozen verdict: `notes/tccd_v0_analysis.md`.
* TCCD-v0 claim / decision:
  `records/claims/claim-tccd-v0-fixed-coordinate-patch-not-ksvd-domain-20260921.yaml`,
  `records/decisions/decision-tccd-v0-stop-fixed-coordinate-patch-20260921.yaml`.
* Lineage base commit: **`d863127`**.

Protocol version: `tccd_v1`. Study: `zinc-context-gap`. Regime: deterministic
remote A100 host `res`, **GPU1 only** (`bash scripts/run_remote.sh 1 …`; GPU0 is
forbidden in this round). Canonical PyG ZINC `subset=True` official **train
10 000 / valid 1 000**. **Official test is never loaded, instantiated, or
referenced anywhere in this round.**

---

## 1. What TCCD-v0 actually established (no reinterpretation)

TCCD-v0 is **not** reinterpreted. Its recorded facts are:

* Gate −1 PASS — a canonical strong GPU1 reference already exists
  (**B-Full / shared-structural Top-5 soup valid MAE = 0.119818**, 84 495
  params; `claim-local-token-null-20260919`). Official test unread.
* Gate 0 PASS — all 8 correctness checks on GPU1 (permutation invariance of
  `x_v` / code / `CᵀRC` / prediction; batching invariance; relation
  correctness; decoupling sensitivity; exact sparsity = 8; task and
  reconstruction gradients reach `D`; no column collapse).
* Gate 1 — reconstruction / reuse / permutation / atom semantics all PASS
  comfortably; **local-coordinate continuity FAIL** (code-space AUC **0.4675**,
  `x`-space AUC 0.5019, graded tail 0.5494; PASS threshold ≥ 0.70). The frozen
  v0 verdict was therefore:
  *"a fixed-coordinate canonical raw local patch is not a suitable K-SVD
  domain"*, and v0 **STOPPED at Gate 1** with Gates 2/3/4 NOT RUN.

### 1.1 The corrected reading (TCCD-v1's premise)

The v0 continuity FAIL refutes **only** the proposition:

> the canonical raw-patch space is a structurally smooth Euclidean manifold.

It does **not** refute the proposition:

> canonical raw patches contain a reusable **discrete** local vocabulary.

The v0 data actively support the second proposition:

* held-out reconstruction ratio `E_learned / E_random = 0.0491` (learned 0.0470
  vs matched random `D` 0.9568) — the dictionary genuinely models the domain;
* **64/64 atoms reused**, 0 dead atoms, top-1 mass 0.0515, top-8 0.2879;
* atom semantics (report-only) mean top-50 corrected canonical-key
  concentration **0.610** vs random 0.075 (8.1×) — atoms latch onto repeated
  **exact/local motifs**.

Metric smoothness and a reusable discrete vocabulary are independent
properties. TCCD-v0 tested the first and failed it; it never tested the second's
*usefulness to a task*.

### 1.2 Continuity is demoted to a **diagnostic** in TCCD-v1

In TCCD-v1, x-space / code-space / graded-tail continuity, atom semantic
coherence, reuse, dead atoms and support entropy are **reported diagnostics
only**. They are **never** a PASS/FAIL gate and **never** enter a stop
decision. This is a deliberate, pre-registered change of the role of continuity,
justified above. The v0 threshold (0.70) is not reused as a gate.

---

## 2. Compute discipline — the frozen TCCD-v0 dictionary is REUSED

**The 12-epoch OMP + K-SVD fit is NOT repeated in this round.** This is a hard
rule.

Reused TCCD-v0 artefacts (provenance: TCCD-v0 Gate 1, commit `008b5e3`,
remote A100 GPU1, `results/tccd_v0/`):

| artefact | path | role |
|---|---|---|
| learned dictionary `D₀` | `results/tccd_v0/dictionary_gate1.pkl` | **initialization + frozen control** |
| canonical patch records | `results/tccd_v0/cache/tccd_v0_train_r2_M14_A21_B3_n10000.pkl` | per-graph `X`, `B`, `Rb`, `Rgeo`, `keys`, `y`, `n` |
| canonical construction | `code/tccd_v0.py` | slot order, block layout, relations |

Frozen values (never re-discussed, never swept in this round):

* `K = 64`, `s = 8`, radius `2`, `M = 14`, `A = 21`, `B = 3`, `F = 714`;
* canonical slot order and block layout (§1.2 of v0);
* relation set: `R_∩ = B Bᵀ` (diagonal removed), 3 native bond-type relations,
  `R_geo = exp(−d/τ)`, `τ = 1`;
* single **linear** reader `ŷ = b + uᵀm_G + Σ_r ⟨W_r, M_G^(r)⟩`;
* internal split `seed = 20260922`, **8 000 internal-train / 2 000 internal-dev**.

If `D₀` were missing and unrecoverable, the round would **stop and report**, not
refit. (Verified present, §4.)

The only place TCCD-v0's sparse codes are reproduced is a **single** frozen
`OMP(x_v; D₀, s=8)` encode over official-train/internal-dev, cached once. No
dictionary update happens during this encode.

---

## 3. This round's questions, in order

* **Q1 (Gate A).** Given the learned reusable local vocabulary, does *how the
  environments are composed* carry significantly more predictive information
  than *which environments exist* (a bag)?
* **Q2 (Gate B).** If Q1 holds: does allowing the ZINC property loss to update
  `D` end-to-end beat the frozen K-SVD vocabulary?
* **Q3 (Gate C).** If Q2 holds: is the sparse dictionary inductive bias at least
  as strong as a matched dense latent representation?
* **Q4 (Gate D).** If Q1–Q3 hold: is the absolute MAE competitive?

No other architecture is explored. Any gate FAIL stops the round; later gates
are **NOT RUN**.

---

## 4. Reuse verification (recorded before any formal run)

* `results/tccd_v0/dictionary_gate1.pkl` → `D` shape `(714, 64)` float64,
  `layout {capacity 14, n_atom 21, n_bond 3, num_pairs 91, feature_dim 714}`,
  `seed 20260922`. Present locally **and** on the remote GPU1 host.
* records cache present with `n = 10 000` graphs, **231 664** radius-2 patches
  (internal-train 185 538 / internal-dev 46 126) and per-graph `y`.
* **`ksvd_refit_performed = false`.** Any run whose log shows a K-SVD fit is
  invalid for this round.

---

## 5. Frozen TCCD-v1 objects

### 5.1 Frozen encoding for Gate A

    c_v = OMP(x_v; D₀, s = 8),   C_G ∈ R^{n_G × 64}.

Codes are computed **once**, chunked over patches, and cached. The reader stage
must not recompute sparse coding or relations.

### 5.2 Frozen composition features (identical to v0 §1.4)

    m_G      = Σ_v c_v
    M_G^(r)  = C_Gᵀ R_G^(r) C_G      (r over the frozen 5 relations)
    vec_sym  = diagonal + strict upper triangle (2080 entries)
    h_G      = [ m_G, vec_sym(M_G^(∩)), vec_sym(M_G^(b1)), vec_sym(M_G^(b2)),
                 vec_sym(M_G^(b3)), vec_sym(M_G^(geo)) ]

`dim h_G = 64 + 5·2080 = 10 464`.

### 5.3 Gate A arms

* **BAG** — `h_G = m_G` only. Answers *"which environments exist"*. Secondary
  evidence (different dimensionality than REL).
* **REL** — full frozen `h_G`. Primary.
* **REL-SHUFFLE** — **primary control**. For each graph, keep the **same code
  multiset** and the **same `R_G`**, draw a fixed-seed random permutation of the
  `C_G` rows, and **do not** permute `R_G`:

      h_G^shuffle = compose(Π_G C_G, R_G).

  Environment count and environment type multiset are exactly preserved; only
  the environment ↔ real-topology correspondence is destroyed. `REL` and
  `REL-SHUFFLE` have **identical** feature dimension and parameter count and are
  the matched comparison.

### 5.4 Gate A reader (lightweight, fast, shared)

The pre-registered reader is a **single linear regression** on the frozen
features — the "linear regression" option permitted by the round brief. It is
**identical for all three arms**:

* each feature is standardized per-feature using the **internal-train** mean
  and standard deviation (`std` clamped at `1e-6`); no other preprocessing and
  no per-arm difference exists;
* **ridge regression** with a single, data-independent regularization
  `alpha = n_train` (= `8000` in the formal run), applied unchanged to `BAG`,
  `REL` and `REL-SHUFFLE`. Because the regularizer is identical for `REL` and
  `REL-SHUFFLE` (identical feature dimension and parameter count), they are the
  exactly matched primary comparison;
* there is no optimizer, no epoch schedule and no checkpoint selection, so
  those elements are trivially identical across arms;
* **internal-dev is never used for any selection** (no alpha search, no early
  stopping). The dev-MAE curve over the fixed diagnostic grid
  `alpha ∈ {10, 100, 1000, 10000}` is reported for every arm as a robustness
  diagnostic only, and never enters the decision.

Whole-graph features for all three arms are **pre-computed once and cached**
(`graph_id, bag, rel, rel_shuffle, target, split`); reader fitting never repeats
sparse coding or relation construction.

---

## 6. Gate A — composition decision (highest priority; no end-to-end before it)

Primary statistic:

    Δ_comp = MAE_REL-SHUFFLE − MAE_REL   (internal-dev, ridge `alpha = n_train`).

* **PASS** iff `Δ_comp ≥ 0.015`.
* **FAIL** iff `Δ_comp < 0.005` → **STOP**:
  *"the current CᵀRC composition representation does not provide enough
  assignment-sensitive predictive information."*
  On FAIL: **no** end-to-end run, **no** learned encoder, **no** distance
  buckets, **no** dictionary change, **no** GNN.
* **AMBIGUOUS** iff `0.005 ≤ Δ_comp < 0.015`: exactly **one** paired seed (`1`)
  is allowed. PASS iff the two-seed mean `Δ_comp ≥ 0.010`; else FAIL.

Secondary evidence reported at Gate A: `MAE_BAG − MAE_REL`. BAG is never the
sole GO criterion.

---

## 7. Gate B — task coupling (only if Gate A PASS)

From here **K-SVD is no longer a training algorithm**; its only roles are
(1) initialization and (2) frozen control. **TASK-D is a genuinely
differentiable end-to-end dictionary.**

* Initialization `D ← D₀` for both arms.
* Tied unrolled sparse pursuit sharing the **same** `D` for encoding and
  reconstruction:

      c^{t+1} = H₈[ c^t + η_t Dᵀ(x − D c^t) ],  t = 0…9,
      η_t = 1/(σ_max(D)²+ε)   (detached, deterministic power iteration).

* Exact sparsity `= 8`; **no** free LISTA `W_e/W_s`, **no** extra neural local
  encoder, **no** learned message passing. Task gradient path is
  `L_task → CᵀRC → C → D`.
* Loss `L = L_task + λ·L_rec`, `λ` = detached initial task/reconstruction value
  ratio, calibrated **once**, frozen; **no** dev-MAE loss-weight search.

Arms: **FROZEN-D** (`D = D₀`, not updated) vs **TASK-D**. Same composition
representation, reader, split, initialization, sparsity `= 8`, relation
operators. Seed `0` only.

    Δ_task = MAE_FROZEN-D − MAE_TASK-D.

* **PASS** iff `Δ_task ≥ 0.010` (no extra seed).
* **FAIL** iff `Δ_task ≤ 0` → STOP; **no** seed-1 rescue.
* **AMBIGUOUS** iff `0 < Δ_task < 0.010`: exactly **one** paired seed (`1`);
  PASS iff the two-seed mean `Δ_task ≥ 0.005`; else FAIL.

On FAIL, the future branch
*permutation-invariant learned local encoder → dictionary* may be named, but it
is **not implemented** in this round.

### 7.1 Optimization smoke (§9 of the brief; not a scientific gate)

Before the formal Gate B run, a cheap GPU1 smoke on a small subset verifies:
task gradient reaches `D`; reconstruction gradient reaches `D`; exact sparsity
`= 8`; no column collapse; train loss decreases; no NaN/explosion; batching /
permutation invariance preserved. The smoke may **not** be used to claim
anything about effect size.

### 7.2 Frozen-D protocol matching

Gate A's frozen reader is a **linear probe on pre-computed OMP features**; Gate
B's FROZEN-D is the **tied-IHT torch model trained end-to-end**. These are not
the same training protocol, so Gate A's frozen result is **not** reused as
Gate B's FROZEN-D. FROZEN-D is run fresh at Gate B (cheap relative to TASK-D).
This is recorded, not hidden.

---

## 8. Continuity diagnostics after Gate B (report-only)

After TASK-D, re-measure: x-space continuity, code-space continuity, graded-tail
continuity, atom semantic coherence, atom reuse, dead atoms, support entropy, and
report `AUC_frozen → AUC_task`. These are **diagnostics**; they never decide
PASS/FAIL, in either direction.

---

## 9. Gate C — dictionary uniqueness (only if Gate B PASS)

Matched generic **dense** local representation `z_v = Dᵀ x_v ∈ R^64`
(width-matched to the dictionary, same matrix parameter count), **no GNN**, no
raw-graph bypass, no handcrafted features, same `Zᵀ R Z` composition, same
reader, same protocol. Compare against TASK-D at the Gate-B decision seed(s):

* **PASS** iff `MAE_TASK-D ≤ MAE_DENSE + 0.005`.
* **FAIL / STOP** iff `MAE_TASK-D > MAE_DENSE + 0.005`.

Even on PASS, if `|MAE_TASK-D − MAE_DENSE| ≤ 0.005` the only permitted claim is
*equal predictive ability plus sparsity / reuse / interpretability* — **not**
superiority.

---

## 10. Gate D — absolute performance (only if Gate C PASS)

Reuse the canonical GPU1 baseline (`B-Full`, valid MAE `0.119818`) — it is
**not** re-run if its protocol/artefact is still valid. Run the frozen TCCD-v1
TASK-D at canonical full-data protocol (official train 10 000, official valid,
canonical soup / checkpoint protocol, GPU1). Official test still forbidden.

    Δ_abs = MAE_TCCD − MAE_canonical-GPU1-baseline.

| band | condition |
|---|---|
| VERY STRONG | `Δ_abs ≤ 0` |
| STRONG | `0 < Δ_abs ≤ 0.005` |
| COMPETITIVE | `0.005 < Δ_abs ≤ 0.015` |
| NOT VIABLE | `Δ_abs > 0.015` → STOP |

NOT VIABLE stops the architecture even if every mechanism gate passed. No
GNN / MLP / handcrafted statistics / extra relation heads may be added to
rescue performance.

---

## 11. Decision tree (frozen)

    TCCD-v0 frozen D0 (reused; no refit)
      └── GATE A  BAG / REL / REL-SHUFFLE (frozen OMP codes)
            ├── Δ_comp < 0.005              -> STOP
            └── Δ_comp ≥ 0.005              -> (seed-1 if ambiguous)
                 └── GATE B  FROZEN-D vs TASK-D
                       ├── Δ_task ≤ 0        -> STOP
                       └── Δ_task > 0        -> (seed-1 if ambiguous)
                            └── GATE C  TASK-D vs DENSE
                                  ├── worse by > 0.005 -> STOP
                                  └── else             -> GATE D absolute
                                        └── Δ_abs ≤ 0.015 -> SUCCESS band

---

## 12. Explicitly forbidden in this round

* Refitting / updating K-SVD (no 12-epoch fit, no dictionary fitting).
* Sweeping `K`, sparsity, radius, capacity, canonical ordering.
* Learned local encoder, GNN / message passing, Transformer, raw-graph bypass.
* Handcrafted ring / shell / path / global-statistic features.
* Adding relations based on MAE, reader-capacity sweeps, loss-weight sweeps.
* Reading or evaluating official test.
* More than the pre-registered single extra paired seed, or continuing past a
  FAIL.
* Running Gate B/C/D after a FAIL, or DENSE before Gate B PASS.

## 13. Efficiency requirements

Cache sparse codes; build whole-graph BAG/REL/SHUFFLE features once and reuse;
reuse frozen controls and the existing baseline instead of re-running; do not
repeat identical preprocessing; batch patch coding / relational contractions on
GPU1; pull results with `--light`; run targeted tests only; add seeds only on
ambiguity. If a step approaches the cost of the previous full K-SVD fit, first
check that cached/reusable work is not being redone.

## 14. Record-keeping

Each gate writes a durable note with: commit; GPU1; reused-artefact provenance;
split; seed; exact arm; wall-clock; peak GPU memory where available; MAE;
diagnostic metrics; threshold; PASS/FAIL; whether the next gate is allowed. Each
note explicitly records **whether K-SVD was refit** (expected: **NO — reused
TCCD-v0 `D₀`**). `STATE.yaml`, claims/decisions, and commits are updated at the
end.

**Frozen.** After this note is committed, no architecture choice may change. An
implementation bug may be fixed in a new commit and the affected gate re-run;
a bugfix may not change the architecture.
