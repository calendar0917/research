# PEC-C1 — Pure Environment Composition Confirmatory (pre-registration)

Round: **PEC-C1** · study `zinc-context-gap` · protocol `pec_c1`.
Supersedes nothing. **Does not amend PEC-v0.**

Frozen: 2026-09-23, before any PEC-C1 formal run. Once the formal run starts,
this document is frozen and may only be extended by a dated, explicitly labelled
amendment written **before** the affected run.

---

## 0. Standing of PEC-v0 (unchanged, never reinterpreted)

> **PEC-v0 Gate 1 failed under its own frozen criteria, and PEC-v0 therefore was
> not eligible for Gate 3. That historical verdict is permanent and is not
> modified here.** PEC-C1 is a **new confirmatory experiment**. It does **not**
> redefine Gate 1, does **not** re-fit a threshold, does **not** change the
> architecture, and does **not** retroactively grade PEC-v0. It asks an
> independent question motivated by PEC-v0's already observed mechanic evidence:
> *given the frozen pure architecture, is it absolutely viable in the full-data
> regime, and does the SparseDict coordinate still retain a task advantage over
> a parameter-matched DenseRole control?*

**Required wording in every PEC-C1 artifact (final report §10):**

> PEC-v0 Gate 1 remains a historical frozen FAIL. PEC-C1 does not retroactively
> pass or recalibrate it.

Forbidden in PEC-C1: "Gate 1 should really count as a pass", "0.1115 is close
enough to ignore", "it passes once the threshold is redefined". The correct
statement is: *PEC-v0's conservatively-defined label-free gate was not met;
PEC-C1 is a new confirmatory question raised by later, independent task
evidence, not a modification of that gate.*

### 0.1 PEC-v0 Gate 0 — recorded verbatim

PASS, 8/8 checks.

* purity (codes finite; node/edge basis unchanged under a chemistry swap)
* chemistry isolation (`atom_idx` chemistry-only; node/edge basis topology-only)
* chemistry-placement sensitivity `0.0991` (max abs `E_i` shift)
* environment freeze bit-identical under pair mutation
* static contract (environment module called exactly once; BAG forward finite)
* relabel invariance `|Δ| = 0`
* exact sparsity `l0 = 4` (both roles)
* gradients finite and non-zero on node dict / edge dict / environment MLP / pair composer / reader
* parameters: CK sparse `94,049` = CD dense `94,049`, C0 coarse `94,036`

### 0.2 PEC-v0 Gate 1 — recorded verbatim

Historical verdict: **FROZEN FAIL 7/10**.

* `node_random_ratio = 0.1115 > 0.10`
* `edge_random_ratio = 0.1024 > 0.10`
* `edge_used = 11 < 12`

Background facts (context only; they do **not** change the verdict):

* `alpha^V -> shell` macro-F1 `1.000`
* `alpha^E -> shellpair` macro-F1 `1.000`
* node `E_rec 0.0369` vs random `0.3305`
* edge `E_rec 0.0543` vs random `0.5299`

### 0.3 PEC-v0 Gate 2 — recorded verbatim

Positive CPU screen, verdict `GATE2_PASS_BUY_SEED0`. Context only; it is
**motivation**, not a gate that PEC-C1 re-runs.

* chemistry-placement shuffle degradation `+0.8353`
* TRUE vs BAG gain `+0.0352`
* TRUE vs SHUFFLE gain `+0.0116`
* `CK = 0.461811`
* `CD = 0.467108`
* neutral-dictionary MAE `2.175`

### 0.4 PEC-v0 Gate 3

`WITHHELD`. Not purchased, not run, not implied. Official ZINC **test never
loaded** in PEC-v0 or PEC-C1.

### 0.5 Prior-artifact / equivalence check

`git status --short` clean at `0b72f8d5cdd7498962bdf314fe5386ac201a5c55`;
`0b72f8d` (PEC-v0 durable record) and `883e529` (SDB-v0 durable record) are both
in the local lineage; `origin/main` is `90ad7c9` and is a strict ancestor of
`0b72f8d` (local ahead by 5, behind by 0). No branch, worktree, result
directory, claim or decision for PEC-C1 (or any confirmatory PEC round) exists.
**No equivalent formal run exists.**

---

## 1. The single question

> Under a *completely frozen* PEC-v0 architecture / feature definition /
> dictionary `K,s` / static composition / training semantics, does the
> Pure-Environment → Static-Composition model have enough **absolute capacity**
> in the full `official-train → official-valid` regime, and is the SparseDict
> structural role still **not worse than** — ideally better than — a
> parameter-matched non-dictionary structural-role control?

This is the current information bottleneck. The bottleneck is **not** "why is
the node random ratio 0.1115 instead of 0.099". Gate 1 is not optimized,
re-run, or rescued in this round.

---

## 2. Frozen objects (identical to PEC-v0; no modification authorized)

### 2.1 Environment definition

* patch radius `2`
* node structural basis `b^V ∈ R^11` (audited FSAR explicit rooted node basis)
* edge structural basis `b^E ∈ R^15` (audited FSAR explicit rooted edge basis)
* shell anchors `∈ {0,1,2}`; shellpair anchors `∈ {(0,1),(0,2),(1,1),(1,2),(2,2)}`
* atom primitive `a(q_v)` = one-hot, 28 ZINC categories
* bond primitive `c(b_e)` = one-hot, 4 ZINC categories
* environment MLP `Linear(input → 96) → SiLU → Linear(96 → 48)`, `E_i ∈ R^48`
* occurrence code `r^V = [one_hot(shell,3) ; alpha^V]` (19), `r^E = [one_hot(shellpair,5) ; alpha^E]` (21)
* binding `B^V_i = Σ_v r^V_{iv} ⊗ a(q_v)` (19×28), `B^E_i = Σ_e r^E_{ie} ⊗ c(b_e)` (21×4)
* environment input `[a(q_i) ; vec B^V_i ; vec B^E_i ; S_i]` with `S_i` the 6-D topology-only root scalars

### 2.2 Dictionary — frozen, and genuinely frozen in PEC-C1

| | basis | `K` | `s` | encoding |
|---|---|---|---|---|
| node | `b^V ∈ R^11` | **16** | **4** | tied-IHT, 10 steps, deterministic power-iteration step, unit-normalized atoms, exact top-`s` |
| edge | `b^E ∈ R^15` | **16** | **4** | same |

**Fit (frozen):** one single detached mini-batch **K-SVD** (`sdb_v0.fit_ksvd`,
exact `s`) on the occurrence bases of **all 10,000 official-train molecules**,
`K = 16`, `s = 4`, `epochs = 10`, `DICT_SEED = 20260924`, atoms unit-normalized.
The same two dictionaries are used for both arms; CD's dense map is
initialized from `Dᵀ` exactly as in PEC-v0. No refit, no second fit, no
per-arm fit.

**Frozen status (PEC-C1 requirement):** `d_node` / `d_edge` are created with
`requires_grad = False` and are **excluded from the optimizer parameter list**.
No reconstruction / entropy / balance / orthogonality term is added to the
loss. The recorded drift `max|D − D₀|` must be exactly `0` for CK.

Forbidden: `K` sweep, `s` sweep, `K = 12` rescue for `edge_used = 11`, `K = 32/64`,
different `K` for node/edge, LISTA, **task-coupled dictionary**, entropy /
balance / orthogonality losses, re-fitting the dictionary on official valid,
validation-informed dictionary selection.

### 2.3 Composition (read-only, static)

* `c_ij = F([E_i+E_j ; |E_i−E_j| ; E_i⊙E_j ; ρ_ij])` computed **once**, over 18-D **pure-topology** pair relations
* `ρ_ij` = 8 distance one-hot + log distance + 5 patch overlap + 3 boundary overlap + log path count. **`path_bond_mean` and the adjacent-bond chemistry are deleted** (as in PEC-v0)
* pair MLP `Linear(3·48+18 → 64) → SiLU → Linear(64 → 48)`
* unary pooling: graph mean and graph max of `E_i`
* pair pooling: graph mean and graph max of `c_ij`
* reader `Linear(2·48+2·48+8 → 64) → SiLU → Linear(64 → 1)`, input `[mean E ; max E ; mean c ; max c ; global_topo(8)]`
* topology branch = the 8-D `global_topo` vector only

Absolutely forbidden: pair→centre, message passing, recurrence, relation
refresh, attention, Transformer, GNN update, pair-conditioned environment
update.

### 2.4 Chemistry

No new chemistry features of any kind. No RDKit descriptors, no
aromatic/ring flags beyond the frozen primitive schema, no charge /
hybridization / electronegativity / valence engineering, no chemistry
embeddings that read topology.

### 2.5 Reader / training

Forbidden: width sweep, head sweep, dropout sweep, LR sweep, optimizer change,
scheduler, extra epochs driven by validation behaviour, post-hoc rescue.

---

## 3. Declared deviations and asymmetries (read this section before judging)

### D1 — PEC-v0's dictionary was inadvertently trainable; PEC-C1 freezes it

PEC-v0's **implementation** constructed `d_node` / `d_edge` as
`nn.Parameter` and passed `model.parameters()` to Adam, so the detached K-SVD
dictionary received task gradients and was updated, **contrary to PEC-v0
pre-registration §2.2 ("No task coupling in Gates 0–2")**. Measured on the
local cache: after only 3 epochs on 300 molecules the drift is already
`max|Δd_node| = 0.01496`, `max|Δd_edge| = 0.01497` (atom scale ≈ 1.0).
So PEC-v0's Gate-2 "SparseDict" arm was in fact a **task-adapted** dictionary
initialized from K-SVD.

Consequences, recorded honestly:

* PEC-v0's frozen gate verdicts (Gate 0 PASS, Gate 1 FROZEN FAIL, Gate 2
  PASS/`GATE2_PASS_BUY_SEED0`, Gate 3 WITHHELD) are **unchanged**; Gate 1 was a
  detached K-SVD reconstruction gate and is unaffected; Gate 2's *comparison*
  is still a valid screen between two task-adapted role parameterizations
  (dense linear map vs unit-atom exact-`l0` sparse code).
* The description "frozen dictionary" attached to PEC-v0 Gate 2 in earlier
  wording is **imprecise** and is corrected by this paragraph.
* **PEC-C1 freezes `D`** because this round's brief forbids task-coupled
  dictionaries and reserves task coupling for a separate `PEC-C2`. This is the
  only intentional deviation from PEC-v0's code and it is **conservative**: a
  frozen coordinate can only make the SparseDict arm *worse*, never better.

### D2 — CK / CD trainable-parameter asymmetry

* CK: total `94,049`, **frozen** role map (`d_node`, `d_edge`), trainable `93,633`
* CD: total `94,049`, **trainable** `Linear(11→16)` + `Linear(15→16)`
  (D-initialized), trainable `94,049` — exactly PEC-v0's DenseRole

The **unique functional difference** between the arms is the role coordinate:
CK uses `tied-IHT_{K16,s4}(D)` over a fixed dictionary; CD uses a learned dense
linear map. The trainability difference is a **second, declared** difference,
one-directional against CK.

Interpretation rule pre-registered now:

* If `M_CK ≤ M_CD`, the frozen sparse dictionary wins *despite* the handicap →
  supports the dictionary claim (Cases C/D).
* If `M_CK > M_CD + 0.003` (Case B), the dictionary route closes, **but the
  result is confounded** by D1/D2 and Case B's verdict text must say so
  ("frozen-dictionary failure; a task-coupled dictionary is not excluded —
  that is PEC-C2's question"). Case B is never upgraded to a claim about
  dictionary learning in general.

No third formal arm is run, and no arm is swapped after seeing valid results.

---

## 4. Arms (exactly two, seed 0)

| tag | arm | role coordinate |
|---|---|---|
| `CK_seed0` | SparseDict | `alpha = tied-IHT_{K16,s4}(D_frozen)`, `D` = frozen K-SVD on official train |
| `CD_seed0` | DenseRole | `alpha^V = M_V b^V`, `alpha^E = M_E b^E`, `M` trainable, initialized from `Dᵀ` |

Both arms: same environment interface, same static pair composer, same reader,
same total parameter count (`94,049`), same optimizer, same protocol, same data
order, same initialization seed, same dictionary.

No BAG / SHUFFLE **training** in PEC-C1 (Gate 2 already answered those cheap
falsification questions). §7 mechanism checks are evaluation-only interventions
on the final CK soup.

---

## 5. Data, selection and test policy

* training data: **official ZINC train, full 10,000**
* selection data: **official ZINC valid, 1,000**, evaluation-only
* **official ZINC test: NEVER LOADED** (`official_test_loaded = false` in every artifact)
* no official-valid use for: dictionary fitting, architecture choice, threshold
  tuning, `K`/`s`, reader, or any other fitted quantity
* the dictionary is fit on official train only; nothing is fit on valid

All selection semantics are frozen in §6 before the run.

### 5.1 Historical context only (never re-run, never a gate)

```
strict-static S0   seed0 soup ≈ 0.140794
strict-static S0   seed1 soup ≈ 0.136423
B-Null             ≈ 0.123
B-Full             ≈ 0.119–0.118
```

`B-Null` / `B-Full` are **not** gates for PEC-C1: they belong to a different
computation class. They are printed for orientation only.

---

## 6. Frozen training protocol

| item | value |
|---|---|
| optimizer | Adam |
| learning rate | `1e-3` |
| weight decay | `1e-5` |
| batch size | `64` |
| loss | L1 / mean absolute error |
| gradient clip | `5.0` |
| scheduler | none |
| epochs | **fixed 240, no early stopping, no patience** |
| seed | `0` (torch / numpy / init), arm-identical |
| per-epoch data order | `np.random.RandomState(seed*1000 + epoch).shuffle` (identical for both arms) |
| checkpoint selection | minimum **official-valid MAE** |
| soup | fixed equal-weight **Top-5 epoch-checkpoint prediction soup** by official-valid MAE, evaluated on official valid |
| device | one A100 per arm (GPU0 = CK, GPU1 = CD) |

Nothing in this table may be changed in response to observed validation
behaviour. No early stop. No epoch extension. No LR/scheduler change.

### 6.1 Provenance recorded per run

commit hash (local and remote), dirty flag, GPU id, GPU model, CUDA / torch /
python versions, seed, wall time, peak GPU memory, best official-valid MAE and
epoch, full per-epoch valid curve, Top-5 soup MAE and members, mechanism
intervention results, `official_test_loaded = false`, exact stop reason.

---

## 7. Mechanism confirmation (evaluation-only, on the final CK soup)

PEC-C1 is not a mechanism-fishing round. Exactly three already-frozen
interventions are applied to the CK seed-0 checkpoint, for **validity only** —
never for architecture selection, never to design a new feature:

1. **Chemistry-placement shuffle** — keep topology, the atom multiset and the
   bond multiset; destroy only the structure↔chemistry assignment. Observed
   prediction must change materially (pre-registered: degradation `> 0`); the
   Gate-2 reference magnitude is `+0.8353`.
2. **Neutral dictionary** — replace `D` with a matched random unit-normalized
   dictionary. Prediction must change materially (pre-registered: mean abs
   prediction shift `> 0`); Gate-2 reference `2.175` MAE / shift `> 0`.
3. **Composition relation shuffle** — permute `ρ_ij` rows within each graph,
   keeping `E_i` and the pair endpoints fixed. Prediction must change materially
   (pre-registered: degradation `> 0`), confirming TRUE composition is not
   relation-invariant.

Recorded, not gated on magnitude. **If the model performs well but the
dictionary branch is dead (`D` replacement changes nothing), no dictionary
claim may be made.**

---

## 8. Frozen decision gates (must not be changed after seeing results)

Let `M_CK`, `M_CD` be the fixed Top-5 official-valid soup MAE, and
`Δ_dict = M_CD − M_CK` (positive ⇒ SparseDict better).

### Case A — absolute architecture weak

If `min(M_CK, M_CD) > 0.145` →

```
PURE_ENV_COMPOSITION_ABSOLUTE_WEAK
```

The cheap environment / composition mechanisms may still hold, but the current
pure no-MP architecture lacks absolute capacity. **STOP**; no seed 1; no
`K`/`s`/reader/recurrence rescue.

### Case B — DenseRole dominates SparseDict

If `M_CK > M_CD + 0.003` →

```
ENV_COMPOSITION_SUPPORTED_DENSE_NOT_DICT
```

Pure environment composition may be viable, but the sparse dictionary-specific
claim fails. No `K`/`s` sweep may be used to rescue the dictionary. If the
absolute band is simultaneously strong, the pure-model result may be recorded
separately, but the dictionary route closes. The verdict text must carry the
D1/D2 confound note.

### Case C — SparseDict viable + specific signal

If `M_CK ≤ 0.145` **and** `M_CD − M_CK ≥ 0.003` →

```
PEC_C1_SPARSE_SIGNAL_SEED0
```

→ authorizes a **paired seed-1 confirmatory round**. This is *not* yet a final
dictionary claim; a single seed may never be written up as "stably better".

### Case D — strong pure model regardless of the CK/CD winner

If `min(M_CK, M_CD) ≤ 0.140` →

```
PURE_ENV_COMPOSITION_STRONG_SEED0
```

provided there is no correctness/mechanism failure. A paired seed 1 is
authorized even if `|M_CD − M_CK| < 0.003`, because the pure no-MP architecture
has entered the historical strict-static strong band. If Dense clearly beats
Sparse, seed 1 confirms the **pure architecture**, not the dictionary claim.

### Case E — viable but dictionary tie

If `0.140 < min(M_CK, M_CD) ≤ 0.145` **and** `|M_CD − M_CK| < 0.003` →

```
PURE_ENV_COMPOSITION_VIABLE_DICT_UNRESOLVED
```

**STOP**; seed 1 is **not** auto-purchased. The architecture is viable but the
dictionary-specific evidence is insufficient, and small effects are not to be
chased with more seeds.

Precedence if more than one case matches: **A > B > D > C > E** (A and B are
terminal; D outranks C because entering the strong band is the stronger
statement; C outranks E).

---

## 9. Seed-1 rules

Seed 1 is forbidden by default. It is authorized **only** if:

* **Trigger 1**: `M_CK ≤ 0.145` and `M_CD − M_CK ≥ 0.003` (Case C), or
* **Trigger 2**: `min(M_CK, M_CD) ≤ 0.140` (Case D).

Otherwise `seed1_authorized = false`. Not triggered by "it was close", a noisy
curve, a flattering best checkpoint, CK beating CD by 0.001, or pretty
dictionary usage statistics.

---

## 10. Task-coupled dictionary — still forbidden

Even if CK seed 0 is excellent, **no task-coupled dictionary is run in
PEC-C1**. Only after a paired seed 1, and only if (a) the absolute pure
architecture is stable, (b) the CK-vs-CD direction is stable, and (c) the
dictionary branch is alive, may a **separate `PEC-C2 task-coupling`
pre-registration** be proposed. Nothing may be appended to PEC-C1.

---

## 11. Targeted tests required before the formal run

1. PEC-v0 purity contract still passes
2. environment freeze still holds
3. no pair→centre
4. no recurrence
5. no chemistry-containing relation (`ρ` is topology-only)
6. CK/CD parameter parity (`94,049` each, total)
7. dictionary still exact `l0 ≤ 4`
8. official-test blocker (no code path can load the test split)
9. deterministic forward (same input ⇒ bit-identical prediction)
10. real-batch CUDA backward produces finite gradients

Plus the PEC-C1-specific invariant: **the dictionary is frozen** (excluded from
the optimizer; `requires_grad = False`; one optimizer step leaves `D`
bit-identical).

## 12. Execution

```
targeted tests → commit prereg + implementation → remote preflight →
deploy clean committed revision → A100 smoke (2 epochs, timing) →
formal seed-0 CK on GPU0 and CD on GPU1 → pull → local analysis → records
```

Parallel GPUs only after the smoke/parity check confirms the regime. Long runs
use the skill-mandated durable launch and are **followed to completion**; no run
is left unattended.

> Note on the brief's "no detached run": the mandatory `remote-research-runner`
> skill forbids holding a multi-hour training run in a foreground SSH command
> (an SSH drop kills it and the run is lost). PEC-C1 therefore uses
> `launch_remote.sh` (session-leader, tracked by tag/pid/exit file) **and
> immediately follows it to its `.exit` with `wait_remote.sh` in the same
> session**. No run is orphaned; the `.exit` file is the completion signal.

## 13. Durable artifacts

* this pre-registration
* runner/protocol note (`pec_c1_implementation.md` if the runner needs explanation)
* targeted-test result
* remote preflight / deploy provenance
* `CK_seed0`, `CD_seed0` (curve, best, soup, states hash)
* mechanism confirmation JSON
* `pec_c1_analysis.md`
* `DECISION.md` + decision JSON
* claim YAML(s) + decision YAML
* `STATE.yaml` update

If no code change is needed, a pre-registration/protocol commit is still made
and the formal run uses that committed revision.

## 14. Final report must answer

* **Q1 Absolute viability**: which band does the pure no-MP
  environment-composition model reach after full data — `> 0.145`,
  `0.140–0.145`, or `≤ 0.140`?
* **Q2 Sparse dictionary**: vs the parameter-matched DenseRole — better, tied /
  unresolved, or worse? Exact delta required.
* **Q3 Mechanism integrity**: still chemistry-placement sensitive,
  dictionary-dependent, composition-relation sensitive; still no MP /
  recurrence / mixed bypass?
* **Q4 Next decision**: exactly one of the five frozen verdicts, plus explicit
  `seed1_authorized = true/false`.

## 15. Research discipline

The point of this round is to **stop optimizing Gate 1**. The information
bottleneck is whether a model with no message passing, no recurrence and no
mixed chemistry bypass has absolute capacity in the full ZINC regime, and
whether a sparse reusable structural dictionary still has task value inside
that pure function class. After the two seed-0 arms, **stop immediately and
analyse**, negative or positive. No rescue.
