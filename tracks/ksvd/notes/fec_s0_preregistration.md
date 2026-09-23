# FEC-S0 — pre-registration (Factorized Environment-Composition S0)

Round **FEC-S0** · study `zinc-context-gap` · protocol `fec_s0`.
Written **before** any FEC-S0 code is executed.
Lineage HEAD at freeze time: `d4de88e` (clean worktree). Official test is
**never** loaded. **Zero training / zero HPO / zero dictionary / zero new
performance run.**

Parent rounds: [`pec_v0_analysis.md`](pec_v0_analysis.md),
[`pec_c1_analysis.md`](pec_c1_analysis.md),
[`pec_i1_analysis.md`](pec_i1_analysis.md).
Historical object audited: `strict-static S0` of round
`ZINC-strict-static-dictionary-pair-v0` (commit `033c1b6`).

PEC-I1 ended with the frozen verdict:

```
STATIC_POOLING_NOT_PRIMARY_GAP
seed1_authorized = false
```

and proved the PEC primitives already contain S0's coarse local chemistry
(atom_shell / bond_shell / root_atom / incident_bonds exact to <= 2.75e-8; only
one topology scalar has a scope difference).  This round asks a different,
*function-preserving* question.

---

## 1. The one question

> Is the historical strict-static S0 already, *as a function*, an
> explicit

$$
\text{chemical primitives} + \text{structural roles}
\;\to\; \text{frozen local environments}
\;\to\; \text{read-only static composition}
\;\to\; y
$$

> model whose factorizations were merely folded ahead of time into mixed
> handcrafted input tensors?

This is a **model-factorization / computational-equivalence** question.  It is
**not** a claim about real chemistry, and it is **not** a new architecture.
No weight is fit, changed, re-seeded, re-tuned or re-selected.

## 2. Frozen historical object

| item | value |
|---|---|
| model builder | `zinc_static_dictionary_pair.build_s0(seed=0)` (`StrictStaticPairModel`, `residual_mode="none"`) |
| config | `configs/luyin16/zinc_compact_v4_topology_hinge.yaml` (`center_context` forced `False`) |
| checkpoint | `results/zinc_static_dictionary_pair/states/s0_seed0_selection_state.pt` (sha256 `8d81f129…8fcf03`) |
| encoded inputs | `results/zinc_static_dictionary_pair/cache/encoded_{train,valid}.pt` |
| recorded best valid | `0.14567435123870381` (epoch 164) |
| recorded Top-5 soup | `0.140794` (members `[125,142,159,164,167]`) |
| params / runtime width | `66,228` / `302D` |
| raw data | PyG ZINC `subset=True` official train 10 000 / valid 1 000 |

The soup **member states were never persisted** (only the best selection state
was saved; `notes/zinc_static_dictionary_pair_v1_preregistration.md` records
this).  This round therefore performs the full equivalence audit on the
available best-checkpoint state and reports the soup as recorded provenance
only.  It will **not** retrain to reconstruct soup members.

## 3. Legal contract (recap, frozen)

For each root $i$:

$$
E_i = H(\text{local structural roles},\ \text{local chemical primitives})
$$

where structure and chemistry may meet for the first time through an
**explicit binding**.  Composition may then read an explicit relation:

$$
c_{ij} = F(E_i, E_j, \rho^S_{ij}, \beta^C_{ij})
$$

with $\rho^S$ a topology relation and $\beta^C$ a chemical relation primitive.
Global atom/bond marginals are legal zeroth-order composition statistics.

**Hard prohibition.**  After $E_i$ is formed, no pair / graph context may write
back $E_i' = U(E_i,\dots)$.  No message passing, no pair→centre, no recurrence,
no relation refresh, no attention update, no context writeback.

## 4. Path classification scheme (pre-registered)

Every reachable predictive input path is classified into exactly one of:

* **`FACTORIZED_SHARED`** — the value is a deterministic function of *shared*
  structural-role descriptor(s) and *shared* chemical-primitive descriptor(s),
  composed by an operator **applied identically to every patch** (no learned
  table indexed by the per-patch joint identity).  Tensor-product blocks,
  marginals and fixed pure-topology statistics qualify.  *Gold standard*:
  `atom_shell[k,a] = sum_v 1[shell_i(v)=k]·1[atom(v)=a] / n`.
* **`EXPLICIT_BINDING_LOOKUP`** — the value is a deterministic function of an
  exact **per-patch joint key** (a canonical certificate of the binding of
  structural roles and chemical primitives) followed by a **data-fit
  vocabulary table indexed by that key**.  Provenance is traceable and the key
  is reconstructible from primitives, but the environment operator is a
  per-joint-configuration memory with independent parameters per key; it is
  **not** a shared role × primitive factorization.  A **non-injective**
  (aliased) certificate also collapses distinct bindings onto one environment.
* **`PURE_TOPOLOGY`** — depends only on the untyped graph (may be retained
  verbatim, with source labelled).
* **`OPAQUE`** — cannot be reconstructed from primitives at all.
* **`NOT_REACHABLE`** — not on the S0 prediction graph.

Rationale for separating `FACTORIZED_SHARED` from `EXPLICIT_BINDING_LOOKUP`:
the round's gold standard is a shared role × primitive composition; an
aliased exact-token table assigns independent learned parameters to each joint
configuration and therefore is not a shared environment formation operator.
"Reconstructible" and "factorized" are deliberately not conflated (this is the
round's explicit anti-renaming rule, brief §15 and §21).

## 5. Verdict rule (pre-registered, no fuzzy outcome)

Let `E_local` = reachable **chemistry-bearing local** paths; `E_comp` =
reachable composition / global paths.

* If every path in `E_local ∪ E_comp` is `FACTORIZED_SHARED` / `PURE_TOPOLOGY`
  **and** all numerical equivalence gates (§6) pass →
  **`FEC_S0_FUNCTIONALLY_EQUIVALENT`** (Case A).
* Else if every path in `E_local` is `FACTORIZED_SHARED` / `PURE_TOPOLOGY` but
  some `E_comp` chemistry path is `EXPLICIT_BINDING_LOOKUP` / `OPAQUE` →
  **`FEC_S0_LOCAL_EQUIVALENT_COMPOSITION_BLOCKED`** (Case B; list blockers).
* Else if some `E_local` path is `EXPLICIT_BINDING_LOOKUP` / `OPAQUE` →
  **`FEC_S0_LOCAL_FACTORIZATION_BLOCKED`** (Case C; list blockers). STOP.
* Else if all paths are reconstructible but a numerical gate fails by more
  than tolerance and the excess is **not** attributable to float
  reduction-order →
  **`FEC_S0_NUMERICAL_EQUIVALENCE_FAILED`** (Case D).

The verdict is computed directly from the measured classification and the
frozen tolerances.  No post-hoc widening.

## 6. Numerical tolerance (pre-registered, frozen before results)

All tensors are `float32`.

| layer | exact preferred | fallback tolerance |
|---|---|---|
| raw descriptor blocks (`patch_cont`, `pair_relation`, `global_context`) | bit-identical | `max_abs <= 1e-7` |
| standardized descriptor | bit-identical | `max_abs <= 1e-7` |
| model intermediates (`h`, unary, pair state, `R`) | bit-identical | `max_abs <= 1e-6` |
| prediction | bit-identical | `max_abs <= 1e-6` |
| token / parent token ids | exact integer equality | exact only |

A fallback is accepted only when the excess is explained as a floating
reduction-order difference; otherwise Case D fires.  Because the audit uses one
device (CPU) for both arms, no device confound is present.

## 7. Scope / forbidden

* **No training**, no fine-tune, no refit, no extra epoch, no seed, no
  optimizer, no loss, no reader retraining, no hidden-width change.
* **No dictionary** of any kind (no K-SVD, SparseDict, DenseRole comparison,
  task-coupled dictionary, K/s sweep, residual dictionary, role refinement).
* **No** deletion/zeroing of any S0 path to make a claim look clean.  The
  evaluation-only local-token intervention is used **only** to *measure*
  reachability/load-bearingness, never to define a substitute model.
* Only historical trained checkpoints are loaded.  Official ZINC test is never
  loaded.
* The factorized implementation must reproduce intermediate tensors **and**
  predictions for the same inputs; it may reuse the original S0 modules and
  replace only feature construction / routing.

## 8. Planned artifacts (durable, committed)

```
notes/fec_s0_prior_artifact_audit.md
notes/fec_s0_preregistration.md        (this file)
notes/fec_s0_path_factorization.md
notes/fec_s0_equivalence_analysis.md
experiments/luyin16/fec_s0_factorization.py
tests/test_fec_s0_factorization.py
results/fec_s0/access_audit.json
results/fec_s0/local_descriptor_equivalence.json
results/fec_s0/pair_relation_equivalence.json
results/fec_s0/global_equivalence.json
results/fec_s0/intermediate_equivalence.json
results/fec_s0/prediction_equivalence.json
results/fec_s0/purity_contract.json
results/fec_s0/REPORT.md
results/fec_s0/DECISION.md
results/fec_s0/decision.json
records/claims/claim-fec-s0-*.yaml
records/decisions/decision-fec-s0-*.yaml
STATE.yaml
```

The next-round candidate **FEC-D1** (baseline-preserving sparse structural-role
refinement, $r^{new} = r^{coarse} + g\,r^{dict}$ with $g=0 \Rightarrow$
FEC-D1 $\equiv$ FEC-S0) is recorded as a *proposal only*; it is **not**
implemented or trained this round.
