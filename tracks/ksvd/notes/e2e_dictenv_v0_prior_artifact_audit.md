# E2E-DictEnv-v0 — prior-artifact audit (round 0, before any code)

Round name: **E2E-DictEnv-v0** (*End-to-End Sparse Dictionary-Core Chemical
Environment*).  Protocol id: `e2e_dictenv_v0`.  Study: `zinc-context-gap`.
Written **before** any implementation, cache build, training or GPU use.

Local audit base: `git log -1 --oneline` = `e9d4245`
(`record(fec-d1) LOCAL_BINDING_SUPPORTED_DICTIONARY_NOT_SPECIFIC`), the current
repo phase:

```text
fec-d1-stopped-local-binding-supported-dictionary-not-specific
```

This note modifies no historical record.  It exists to make the round's
scientific boundary explicit: **this round must absorb all prior results**, not
pile a dictionary branch onto FEC-S1.

---

## 0. The round's single hypothesis

> A sparse, shared, task-coupled pure-topology structural dictionary can be the
> **only fine-grained learned structural coordinate** from which local chemical
> environments are constructed (`phi65 -> D -> alpha -> (alpha, q, shell) ->
> E_i -> static composition -> y`), and its value is **specific** relative to a
> parameter-identical dense tied structural coordinate.

The whole point of the round is that the dictionary is **not** competing with a
strong dense local environment for residual credit (that was FEC-D1, and it
failed).  Here the dictionary **replaces** the fine-grained local structural
encoder entirely.

---

## 1. Mandated historical conclusions (must read, must not repeat)

### 1.1 GSCN-v0 — dictionary-as-message-passing-transition: CLOSED

`notes/gscn_v0_analysis.md`:

```
raw baseline soup       ≈ 0.1950
Sparse-GSCN soup        ≈ 0.2123
generic matched control ≈ 0.1982
```

The dictionary was the iterative hidden-state transition on top of raw graph
message passing.  Verdict: gain explained by generic capacity, not by sparse
dictionary learning.

**This round does not redo**: no dictionary-as-transition, no message passing
at all (hard purity contract, §2).

### 1.2 TCCD — prototype vocabulary / attributed whole-patch compression: CLOSED

`notes/tccd_v2_analysis.md`, `notes/tccd_v7_final_normalized_moment_preregistration.md`:

- Task-learned prototype vocabularies can be healthy and assignment-sensitive
  composition is real (Gate A passed), but letting the prototype representation
  carry the full attributed patch/environment gives absolute performance far
  below the strong line (TCCD-v2 absolute gap 0.1675 vs the canonical GPU
  baseline; TCCD-v7 standalone normalized-moment 0.4121 vs frozen raw 0.2784).

**This round does not redo**: the dictionary must NOT simultaneously carry
chemistry + topology + whole-environment compression.  Chemistry enters only as
primitive one-hot factors outside the dictionary; topology enters only as the
audited `phi65`; the environment is formed by binding the sparse role code to
primitive chemistry at root-relative shells.

### 1.3 SDB-v0 — the validated dictionary substrate (positive prior)

`notes/sdb_v0.md`:

- `phi_v in R^65` (FSAR-R2-AR0 pure-topology node coordinate) → `alpha_v in
  R^32`, K=32, s=8, is a **healthy, reusable, pure-topology sparse role
  coordinate**.
- Stage 1 (K-SVD, no y): 32/32 atoms used, exact sparsity 8, top-1/top-8 mass
  0.803/1.0.
- Stage 2 (mechanism vs frozen `phi65` oracle, 3 seeds): mean recovery 1.159
  (dict32 strictly better than the full 65-D axis), assignment-shuffle
  degradation ≈ 0.245 — a **real** assignment signal.
- Stage 3 (task-coupled tied-IHT, seed 0): frozen-D 0.498979 vs task-coupled D
  0.489882, delta −0.009097 — real but marginal, single seed.
- Stage 4 (strict-static S0 backbone residual): +0.002381 gain, below the
  material gate; dictionary not the missing ingredient **on a strong backbone**.

**This round reuses**: `phi65` from the exact audited implementation (FSAR
cache, never rebuilt differently), the frozen K-SVD dictionary
`tracks/ksvd/results/sdb_v0/dictionary.pt` (65×32, sha256 begins
`925d573a5808...`), K=32, s=8, tied-IHT (10 steps), and the SDB "task-coupling
is real but marginal on a strong base" philosophy (λ calibrated once, frozen).

**This round does not redo**: SDB Stage 4's additive residual readout on a
strong base.  There is **no** strong base here.

### 1.4 FEC-D1 — localized binding supported, dictionary NOT specific: the key negative

`notes/fec_d1_analysis.md`:

```
Dict32 localized binding soup = 0.131537
PCA32 localized binding soup  = 0.130231
shuffle                       = 0.144669
```

On the **frozen FEC-S1 strong local environment**, a localized structural
coordinate × chemistry binding helps (+0.0052 over the frozen base) and is
assignment-mediated (+0.0131 shuffle), but the sparse K32/s8 dictionary is
**worse** than the matched dense PCA32 of the same `phi65`.  That round's
verdict: localization + binding is valuable; dictionary-specificity is refuted
*at that placement*.

**Why this round is not FEC-D1 again**: FEC-D1 kept the 24-D FEC-S1 local
environment and asked whether a dictionary branch could refine it.  This round
deletes the FEC-S1 dense local encoder *and its 146-D patch_cont input*
entirely.  The dictionary is the ONLY fine-grained learned structural
coordinate in environment formation.  The matched control is **not** PCA (PCA
was done; it won) but a fully matched *dense tied* coordinate `z = phi @ P`
inside the identical architecture — same parameterization, same objective, in
particular the same λ_rec, same backend, same reader.  This is the strictest
remaining isolation of "sparse tied coding itself" from "dense tied
coordinate".

### 1.5 FEC-D0 — within-coarse-role residual subrole basis: CLOSED (no signal)

`notes/fec_d0_analysis.md`: after the exact shell is removed the audited rooted
basis has no held-out subrole signal beyond degree (G_O = 0.0244 < 0.03), and
the dictionary is "primarily a discretized degree representation" there.

**This round does not redo**: no within-shell residual subrole basis, no
shell-residualized coding.  We code the full audited `phi65` coordinate exactly
as SDB/FSAR did.

### 1.6 FEC-S1 — the strong static line (context, not matched control)

`notes/fec_s1_analysis.md`: FEC-S1 (vocabulary-free shared local environment
146→214→24 + strict-static S0 backend) seed-0 Top-5 soup **0.130422**, best
0.136783.  FEC-S1 is the current strong static anchor.

**This round's posture**: FEC-S1 is a **historical anchor only** (§29), never a
matched architecture comparison; matched causal comparison is SparseDictEnv vs
DenseTiedEnv.  The 146→214→24 adapter is exactly what this round forbids (§23).

### 1.7 PEC-v0/PEC-C1/PEC-I1 — environment-composition line (predecessor round)

`notes/pec_v0_prior_artifact_audit.md`: the no-message-passing,
frozen-environment, static-composition object class was established; PEC-C1/I1
closed the "pure static pooling interface" question
(`decision-pec-i1-stop-static-pooling-not-primary-gap-20260922`).  PEC-v0's own
Gate-1/2 conclusions (sparse role dictionary preserves coarse roles; binding
needed) underpin this round's ingredients.  This round inherits the strict
"environment frozen after formation" contract (G6/G7/G8 hard gates).

### 1.8 Other closed lines (dedupe)

| prior | object | why not repeated here |
|---|---|---|
| SDPK-v0 / SRDA-v0 / DTX-v0 | dictionary inside the pair kernel / relation algebra / aligned graph cross on the mixed S0 backbone | all keep a strong dense local object; dictionary-as-pair-channel; all CLOSED with no absolute gain; this round has no dense local object and no pair-side dictionary |
| GSCN-v0 | dictionary as hidden-state transition | §1.1 |
| TCCD | soft prototypes of attributed patches | §1.2 |
| FSAB / compact-v6 / ASB-Z1 / BCE | binding branches that collapsed to constants inside a strong backbone | this round has no backbone for a branch to hide in; chemistry has only one route to the reader |
| PSCD family | motif dictionaries / port operators | different object class (exact-decode motifs), CLOSED; not relevant to sparse linear codes |

---

## 2. What is genuinely new

1. **Dictionary centrality without a dense competitor.**  Every prior dictionary
   placement (`notes/gscn_v0_analysis.md`, SDB Stage 4, SDPK, SRDA, DTX,
   FEC-D1) placed the dictionary *inside or beside* a strong mixed or dense
   local object.  E2E-DictEnv is the first round in which `alpha = IHT(D, phi)`
   is the **only** fine-grained learned structural coordinate feeding local
   environment formation.  If the local environment must be built, it must be
   built from the dictionary code.
2. **End-to-end task-coupled D with a paired identical-budget dense tied
   control.**  FEC-D1 compared sparse dict vs dense PCA as *frozen input maps*
   on a frozen base.  Here `D` (Sparse) and `P` (Dense) are trainable, the
   objective is identical (`MAE + λ_rec·L_rec` with one frozen λ), and the
   comparison is sparse-IHT-coding vs dense linear coding of the *same*
   matrix initialisation, inside the *same* architecture.
3. **Pure-topology dictionary × primitive chemistry factorized trilinear
   binding with an explicit marginal anchor** (`r_v = [1; alpha_v]`), sum
   aggregation (mass-exposing, per TCCD-v7's size/mass finding), and zero
   normalization of the environment vector (`m_i^V` is not mean-normalized).
4. **No patch_cont 146-D anywhere in the model.**  Only the 6 pure-topology
   scalars of the historical shell descriptor survive, as fixed anchors.  All
   atom/bond marginals are carried by primitive chemistry factors (`q_v`,
   `b_e`) through the binding, never by a dense descriptor.

---

## 3. Reused artifacts (read-only; no retraining)

| artifact | path | use |
|---|---|---|
| FSAR-R2-AR0 feature cache | `tracks/ksvd/results/fsar_r2_ar0/cache/{train,valid}.pkl.gz` | `phi_v`/`atom_idx` per node, aligned with raw ZINC + encoded cache (validated per-molecule) |
| SDB frozen K-SVD dictionary | `tracks/ksvd/results/sdb_v0/dictionary.pt` | init for both arms (`D` sparse, `P` dense), 65×32, sha256 `925d573a58083c3c20f36980dec9bba77dff592b30ea1abf11098ca3a7b7487a`; never refit |
| tied-IHT primitives | `tracks/ksvd/code/tccd_v0.py` (`hard_threshold_rows`, `power_iter_sigma`) | exact top-s thresholding, deterministic step size |
| encoded ZINC cache | `tracks/ksvd/results/zinc_static_dictionary_pair/cache/` | `pair_index`, `pair_relation` (23-D), `pair_bucket`, `global_context` (62-D), `topology_features` (25-D), `patch_cont` scalars, `y`; `official_test_loaded=false` |
| strict-static backend semantics | FEC-S1/S0 (`zinc_patch_path_pooling.py::_MLPBlock`, `zinc_graph_head_function_family.py::GenericReader`) | pair projection 768, relation encoder 1360, distance gate 80, pair encoder 5328, global 62→32→32 = 3136, topology 25→16→8 = 552, reader 302→13→13→1 = 4135 ⇒ **15359** backend params |
| shell incidence primitives | `zinc_patch_path_pooling.py::_shell_pairs_for_radius`, `fec_d1.py::ego_shells` | root-relative shell (0,1,2) and shellpair (6 classes) |
| protocol | `OPTIMIZED_PROTOCOL` (Adam 1e-3, wd 1e-5, batch 128, clip 5, no scheduler, 240 epochs, fixed Top-5 soup) | unchanged |

**Explicitly not re-run**: FEC-S1, FEC-D1, PCA32, S0, SDPK, SRDA, DTX, TCCD,
GSCN, SDB Stages 1–4, and every other historical control.

---

## 4. Parameters of the round (pre-registered; no sweep)

```
K=32, s=8, IHT steps=10, r_atom=64, r_bond=16, env 86->329->48 (SiLU)
Sparse dictionary         65*32        = 2 080
W_R 33x64 + W_C 28x64 + S 3x64         = 4 096
W_B 4x16 + P 6x16                      =   160
environment MLP 86->329->48            = 44 463
local total                            = 50 799
static backend (FEC-S1 exact)          = 15 359
whole model                            = 66 158   (FEC-S1: 66 170; Δ = -12)
```

Must be re-derived and verified by code (§17 of the round spec).  If the
implemented count differs: STOP and locate, never adjust widths to "get close".

---

## 5. Dedupe verdict

**No completely equivalent implementation exists.**  The closest objects:

| prior | object | decisive difference from E2E-DictEnv |
|---|---|---|
| FEC-D1 | dict/PCA localized binding injected into the frozen FEC-S1 24-D environment | keeps the dense FEC-S1 local environment as the base; frozen D/P; residual branch; this round deletes the dense environment and makes D/P the environment's only structural coordinate, end-to-end trainable |
| SDB Stage 3/4 | task-coupled D (weak base) / frozen-D residual on S0 | no local environment at all; a global assignment statistic or linear residual |
| PEC-v0 | environment composition with K=16/s=4 frozen dict on `phi65`+`psi130` | K=32/s=8, task-coupled D, primitive-chemistry factorized trilinear binding with explicit marginal anchor, sum aggregation, no edge-role dictionary, FEC-S1-size static backend |
| TCCD | soft prototypes on attributed patches | exact top-s tied-IHT on pure topology only |
| SDPK/SRDA/DTX | dictionary in pair kernel / relation algebra / graph cross | no pair-side dictionary at all; pairs operate on frozen formed environments only |

**Proceed, with a pre-committed mixed prior**: the ingredient-level evidence
(SDB Stage 1–3, FEC-D1 Gate A/C) is positive; the placement-level evidence
(SDB Stage 4, FEC-D1 Gate B, every absolute anchor) is negative.  The round's
only scientific content is the two-arm contrast `SparseDictEnv vs
DenseTiedEnv` with identical machinery, and the load-bearing interventions
(zero-code, assignment shuffle) on the trained Sparse soup.  Any clear
negative at any gate stops the round with no rescue.

---

## 6. What would make this round worth its cost

1. It is the **first** genuinely end-to-end, fully-matched test of
   `sparse tied dictionary coding` vs `dense tied coordinate` as the sole
   fine-grained structural coordinate of a competitive-feasibility local
   environment model (no FEC-S1 environment, no PCA, no residual, no pair
   dictionary, no message passing).
2. If Sparse ≥ Dense + 0.003 with zero/shuffle gates ≥ 0.010 and health PASS
   and absolute ≤ 0.135, it is the first positive dictionary-core result of the
   whole track and justifies a *new* preregistered generic nonlinear control
   (E2E-DictEnv-C1) — nothing more.
3. If DenseTied is not worse (or Sparse is absolutely weak), it cleanly closes
   the ZINC predictive dictionary-core route, matching the accumulated
   evidence (GSCN, TCCD, SDB-4, SDPK, SRDA, DTX, FEC-D0, FEC-D1), and no
   further placement/K/s/edge-dictionary/LISTA rescue is justified.

Stop rules, decision table and the exact gate thresholds are frozen in
`e2e_dictenv_v0_preregistration.md` (§33–§38 there).
