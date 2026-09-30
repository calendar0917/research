# Prior-artifact audit — `e2e_dictenv_jointbond_v1`

Round: **E2E-DictEnv-JointBond-v1** (`e2e_dictenv_jointbond_v1`), study
`zinc-context-gap`, track `tracks/ksvd`.
Candidate: **JointBond / Key-level Joint Structure–Semantics Fusion**.
Parent: **`CSSD-Sem108`** (`SEM108Model`), code revision
`da2af280d4229aa96fac248eecf38109ee56716b` (current `HEAD` at round start).

This note freezes what is known *before* the round's formal run, so the round
cannot be retro-fitted around its outcome.

---

## 1. Frozen parent object

| item | value |
|---|---|
| parent | `CSSD-Sem108` (`SEM108Model` in `e2e_dictenv_sem108_v1.py`), itself `CSSD-q1 + C6` |
| parent code revision | `da2af280d4229aa96fac248eecf38109ee56716b` (`git rev-parse HEAD` at round start; working tree clean) |
| config | `cm.H1_CONFIG` (`decoder=h1`, `d_e=48`, `K=32`, `s=8`, `lambda=33.95873017865987`, `horizon=320`, `dict_kind=sdb32`) |
| mask | `cm.C6_MASK` (`cssd.CSSD_MASK is cm.C6_MASK`, `cm.c6_equivalence_check()` true) |
| common subspace | train-only `q1`, `rms=[5.082852828320509]` |
| parent interface | `[Sem108 ; size2]` (110-D) `-> fusion(446 -> 114 -> 48)` |
| node path | 3 shells × 48, `(coord @ W_A_S) * (q @ W_A_C)` |
| original edge path | 6 shellpairs × 48, `(g @ W_E_S) * (b @ W_E_C)`, then `index_add_` |
| parent candidate params | **97709** |
| optimizer / soup | Adam `lr=1e-3`, `wd=1e-5`, batch 128, clip 5.0, Top-5 valid soup |
| device | CPU only, 8 threads, no CUDA |
| official test | **never loaded**; every payload `official_test_loaded = false` |

**Parent forward bit-identity was re-verified in this round's implementation:
the frozen `SEM108-seed0_soup_state.pt` re-scores valid MAE exactly
`0.123704927947314` after the refactor that introduces the edge-response hook
(`J11`).  The hook default returns `None`, so no parent number can move.**

## 2. Durable historical references (read-only, never rerun this round)

| round | value | note | artifact |
|---|---|---|---|
| `SEM108` parent seed 0 | soup valid `0.123704927947314`, best `0.13131332652637503` @ 309, members `[238, 255, 257, 282, 309]` | unmatched, historical | `results/e2e_dictenv_sem108_v1/summary.json` |
| `CSSD-q1` seed 0 | soup valid `0.13002798487985273` | unmatched | `results/e2e_dictenv_common_subspace_dictionary_v1/training/final.json` |
| training-protocol-audit control `lr=1e3` | soup `0.12742769679048796` | unmatched | `results/e2e_dictenv_training_protocol_audit_v1/comparison.json` |
| `FINAL-CLEAN` C6 seed 0 | `0.128499...` | unmatched | `results/e2e_dictenv_clean_mechanism_v1/stage_c_independence/frozen_C6_seed0.json` |
| `T1` tuned seed 0 | soup `0.12576500436564675` | unmatched | `results/e2e_dictenv_t1/tuning_decision.json` |
| `RNDB` seed 0 | soup `0.1331174676119699` (`97727 -> 105407` params) | unmatched; rolewise nonlinearity strongly load-bearing, task band unmoved | `results/e2e_dictenv_rndb_v1/summary.json` |

No historical artifact is comparable at matched protocol; every comparison in
this round is explicitly **historical / unmatched / contextual**.  There is no
contemporaneous baseline and the `+4224` branch parameters are un-matched.

## 3. What the parent round already established (frozen prior knowledge)

* **Frozen C6 probe results** (Sem108 soup, read-only): `G_sem_shuffle`
  load-bearing (`>= 0.005`), `G_atom` / `G_bond` block degradations recorded,
  `G_dict0` recorded, `G_corr = max(G_node, G_edge)` driven **entirely** by the
  edge correspondence.
* **Node-binding collapse (recorded, not repaired this round):** in the Sem108
  soup state `W_A_S` / `W_A_C` fell to float32 denormals, so node slots are
  identically zero and the node assignment shuffle is a no-op.  The Sem108
  analysis records this as a learned redundancy collapse (the direct Sem108
  interface already carries atom chemistry), not a coding defect.  JointBond
  keeps that path untouched and only records the same health fact again.
* **RNDB contrast:** RNDB inserted a *shared* nonlinearity between the residual
  role responses and their aggregation (`psi_A` / `psi_E`) and was strongly
  load-bearing at the mechanism level while the task band did not move.  RNDB
  never preserved the endpoint identity of the structural role/semantics pair,
  and its nonlinearity sat inside `sum_k` over dictionary atoms of the *same*
  occurrence.  JointBond is a different object: one response per **real bond**,
  built from the two endpoints' own (structure, atom-type) pairs and the bond
  type, added before the shellpair aggregation.  It is **not** "the existing
  edge product plus a shared nonlinearity", and it must not be reported as a
  re-run of RNDB.
* **luyin19 (mentor direction, 2026-09-30 recording):** the dictionary's purpose
  is a *reusable, transferable* structure–attribute relation, and the immediate
  open question is how to combine structure and semantics — the recording
  explicitly separates "first separate structure and semantics, then study the
  combination".  JointBond is the smallest testable combination operator that
  keeps the endpoint correspondence.

## 4. Code-level facts the implementation must respect

| fact | evidence |
|---|---|
| `data.env_bond_u` / `env_bond_v` are **batch-global** node indices (the P1 pipeline concatenates all molecules before encoding) | `zinc_e2e_dictenv_p1.py` incidence blob + `p1.env_bond_u` direct indexing (no `_OCC_OFFSET_NAMES` entry) |
| `data.dict_atom` is per-node, aligned with `data.dict_phi` rows | `zinc_e2e_dictenv_p1.py` encoded blob construction |
| the edge response is aggregated with `index_add_` on `root * SHELLPAIR_CLASSES + shellpair` | `SEM108Model._environment_from_parts` (now `_edge_env_slots`) |
| the residual code is `coord[:, common_dim:]` (the common coordinate is `coord[:, :1]`) | `CSSDModel.code` |
| the parent structure-assignment shuffle moves `env_bond_u` / `env_bond_v` to another real endpoint pair of the same `(root, shellpair)` group | `p1.shuffled_bond_endpoints_for_molecule` |
| the C6 mask carries no edge-binding zero and no intervention other than zeroed chemistry channels | `cm.C6_MASK` |

## 5. What this round is **not**

Not a message-passing / Transformer / environment-state-update round; not a new
fine structural descriptor; not a DenseTied or parameter-search round; not a
baseline rerun; not a repair of the node-binding collapse; not an official-test
read; not a claim that the sparse dictionary is superior.  The only authorised
work is one correctness suite, one ≤64-step smoke, one seed-0 320-epoch run and
the frozen inference probes.
