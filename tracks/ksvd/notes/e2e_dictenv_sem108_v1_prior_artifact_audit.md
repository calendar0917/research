# Prior-artifact audit — `e2e_dictenv_sem108_v1`

Round: **E2E-DictEnv-Sem108-v1** (`e2e_dictenv_sem108_v1`), study
`zinc-context-gap`, track `tracks/ksvd`.
Candidate: **CSSD-Sem108 / Shell-Resolved Primitive Semantic Interface**.

This note freezes what is known *before* the round's implementation and what
must be treated as durable prior evidence.  It exists so the round cannot be
retro-fitted around the outcome.

---

## 1. Frozen parent object

| item | value |
|---|---|
| parent | `CSSD-q1 + C6` (`CSSDModel`, `tracks/ksvd/experiments/luyin16/e2e_dictenv_common_subspace_dictionary_v1.py`) |
| config | `cm.H1_CONFIG` (`decoder=h1`, `d_e=48`, `K=32`, `s=8`, `lambda=33.95873017865987`, `horizon=320`, `dict_kind=sdb32`) |
| mask | `cm.C6_MASK` (`cssd.CSSD_MASK is cm.C6_MASK`, `cm.c6_equivalence_check()` true) |
| common subspace | train-only `q1`, `rms=[5.082852828320509]` |
| parent local interface | `anchor62 -> anchor_encoder(62->32->32) -> 32D`, `fusion(368->128->48)` |
| parent total params | **97727** (verified against the durable CSSD-q1 artifact) |
| common coordinate | width 33 (`q=1` common + 32 residual sparse atoms), IHT-10 |
| node / edge slots | 3 shells × 48, 6 shellpairs × 32 |
| optimizer | Adam `lr=1e-3`, `weight_decay=1e-5`, batch 128, grad clip 5.0, Adam-eps default |
| soup | Top-5 valid states by valid MAE (member epochs and states only) |
| device | CPU only, no CUDA |
| official test | **never loaded**, every payload `official_test_loaded = false` |

## 2. Durable historical references (read-only, never rerun this round)

| round | value | note | artifact |
|---|---|---|---|
| CSSD-q1 seed 0 | soup valid `0.13002798487985273` | unmatched historical context | `results/e2e_dictenv_common_subspace_dictionary_v1/training/final.json` |
| training-protocol-audit control `lr=1e3` | soup `0.12742769679048796` | unmatched | `results/e2e_dictenv_training_protocol_audit_v1/comparison.json` |
| training-protocol-audit `lr=1e-4` | soup ≈ `0.1279` | unmatched | same |
| FINAL-CLEAN C6 seed 0 | `0.128499...` | unmatched | `results/e2e_dictenv_clean_mechanism_v1/stage_c_independence/frozen_C6_seed0.json` |
| T1 tuned seed 0 | soup `0.12576500436564675` | `A2_COARSE146_SLOT48`, 66132 params; unmatched | `results/e2e_dictenv_t1/tuning_decision.json` |
| RNDB seed 0 | soup `0.1331174676119699` | 97727 → 105407 params; unmatched | `results/e2e_dictenv_rndb_v1/summary.json` |

No historical artifact is comparable at matched protocol; every comparison in
this round is explicitly labelled **historical / unmatched / contextual**.

## 3. The plateau this round attacks

Three independent architecture rounds land in `0.1258 – 0.1331`:

* FINAL-CLEAN C6 `0.1285`;
* T1 (fine coarse146 interface) `0.1258`;
* CSSD-q1 `0.1300`;
* RNDB (nonlinear rolewise dictionary binding, strongly load-bearing) `0.1331`.

Every one of these compresses the primitive chemistry of a patch into the
coarse 62-D anchor **before** the environment decoder, i.e. the decoder never
sees the shell-resolved primitive semantic blocks directly.  The round's single
question is whether that early compression, rather than the dictionary fusion
function, is what bounds the plateau.

## 4. Knowledge needed by the audit and what the code says

Resolved from the *current* implementation (not hard-coded in the round code;
`sem.resolve_sem108_geometry()` asserts agreement and raises otherwise):

* `zpp.SHELL_WIDTH = 146`, `zpp.PATCH_RADIUS = 2`, `zpp.ATOM_CATEGORIES = 28`,
  `zpp.BOND_CATEGORIES = 4`, `zpp.SHELL_PAIRS = ((0,0),(0,1),(0,2),(1,1),(1,2),(2,2))`.
* `fec.SHELL_BLOCKS` layout: `atom_shell (0,84)`, `bond_shell (84,108)`,
  `root_atom (108,136)`, `incident_bonds (136,140)`, `scalars (140,146)`.
* Sem108 = `patch_cont[:, 0:108]` = `atom_shell` (3 shells × 28 atoms) ∥
  `bond_shell` (6 shellpairs × 4 bond categories).
* Anchor layout: `root (0,28) | atom_mass (28,56) | bond_mass (56,60) | size (60,62)`;
  the `size` block contains exactly `[log1p n, log1p m]`.
* Exploratory pre-check on 64 official-train molecules (destandardized with the
  rebuilt train-fit scalers): root raw exact fraction `1.0` (max error `0.0`);
  atom-mass reconstruction `n_nodes × Σ atom_shell` max error `1.9e-6`
  (`fraction_exact(1e-5)=1.0`); bond-mass analogous; `anchor[:,60:62]` vs
  `patch_cont[:,140:142]` standardized max abs diff `0.0`; the two size
  standardizers are bit-identical, so the destandardized difference is `0.0`.
  The full-split audit (A2/A4) is run in-round and recorded in `audit/`.

## 5. Audit A3 — T1 block attribution is **unavailable**

T1's gain (`0.12577`) was obtained with the full coarse146 descriptor present in
its interface.  A clean per-block attribution (atom84 / bond24 / Sem108 /
topology6) would require T1's exact soup/selection tensors.  Those lived only in
the git-ignored `results/e2e_dictenv_t1/states/` directory (`.gitignore` line 31)
and do not exist anywhere on this machine (checked `/home/calendar`, `/tmp`,
`runs/`).  Therefore:

* **A3 = `T1_BLOCK_AUDIT_UNAVAILABLE`**; no T1 retraining, no T1 state
  reconstruction, no surrogate.
* The T1 attribution limitation is recorded in `audit/t1_block_ablation.json`,
  the report and the decision record.  The round proceeds on interface history
  and the non-redundant shell-semantic representation.

## 6. Explicit non-goals / forbidden mechanisms

No RNDB (`psi_A`/`psi_E` absent), no message passing, no new fine structural
descriptor, no `root_atom`/`incident_bonds`/topology scalars as new inputs, no
Sem108-only control, no baseline rerun, no seed sweep, no hyper-parameter
search, no rescue run, no topology/root/incident "rescue" features.
`docs/luyin/luyin19.txt` is not modified.
