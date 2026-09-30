# Pre-registration — `e2e_dictenv_jointbond_v1` (JointBond)

Round: **E2E-DictEnv-JointBond-v1** (`e2e_dictenv_jointbond_v1`), study
`zinc-context-gap`, track `tracks/ksvd`.
Candidate: **JointBond / Key-level Joint Structure–Semantics Fusion**.
Parent: **`CSSD-Sem108`** (`SEM108Model`, `e2e_dictenv_sem108_v1.py`), round
`e2e_dictenv_sem108_v1`, code revision
`da2af280d4229aa96fac248eecf38109ee56716b`.
Prior-artifact audit: [`e2e_dictenv_jointbond_v1_prior_artifact_audit.md`](e2e_dictenv_jointbond_v1_prior_artifact_audit.md).

CPU only · official ZINC test never loaded · write-then-follow, no post-hoc
change to any constant, gate, threshold, probe or band in this document.

---

## 1. Single question

> Before the local-environment aggregation, can binding the structural
> coordinate of *each* bond endpoint to *that* endpoint's atom type, and then
> jointly fusing both endpoints together with the bond type, improve prediction?

**H-jointbond.** The parent forms `(coord @ W_E_S) * (bond_onehot @ W_E_C)` on
the *pair* of endpoints — a shared, role-free product — and RNDB showed a
rolewise nonlinearity on that product is load-bearing yet does not move the task
band.  JointBond instead keeps the endpoint correspondence explicitly
(`h_v = (alpha_v @ A) * (q_v @ C)` per endpoint) and fuses the two endpoints
nonlinearly, then multiplies by the bond-type code — all **before** the
shellpair `index_add_`.

**Falsified if** the branch is inert (`G_branch_off < 1e-6`), or the endpoint
correspondence is not read (both shuffle degradations are `0`), or a correctness
gate fails, or the run violates a frozen constant.

## 2. Exactly one architectural change (frozen)

Everything in the parent `SEM108Model` is retained.  On every **real bond**
occurrence the following *additive residual branch* is summed into the parent
per-bond edge response **before** the shellpair `index_add_`:

```text
alpha_v = coord[v, common_dim:]              # 32-D residual sparse dictionary code (common excluded)
q_v     = one_hot(dict_atom[v])              # 28 atom categories (actual endpoint v)
b_uv    = one_hot(env_bond_type[uv])         # 4 bond categories

h_v     = (alpha_v @ A) * (q_v @ C)          # 16-D
t_uv    = concat(h_u + h_v, abs(h_u - h_v), h_u * h_v)      # 48-D
j_uv    = F(t_uv) * (b_uv @ B)               # 48-D
ue_uv   = parent_edge_response_uv + j_uv     # then the shellpair index_add_
```

* `A: 32 -> 16`, `C: 28 -> 16`, `B: 4 -> 48`, all **bias-free** linear maps;
* `F = Linear(48, 32, bias=False) -> SiLU -> Linear(32, 48, bias=False)`;
* `A`, `C`, `B` and `F`'s first layer: **Kaiming-uniform**
  (`nn.init.kaiming_uniform_(w, a=sqrt(5))`, applied explicitly in the frozen
  order `A, C, B, F[0]`);
* `F`'s last layer: **`Normal(0, 0.01)` initialised once**;
* **no** extra gate, normalisation, depth, dropout, semantic bypass, message
  passing, Transformer, environment-state update, extra chemistry statistic or
  new topology descriptor;
* the branch **never reads the common coordinate**: only
  `coord[:, common_dim:]`.

Structure and semantics are gathered at the **same** endpoint index
(`bond_u` / `bond_v`), so an endpoint's structural role always multiplies that
endpoint's own atom semantics.  The parent's structure-assignment shuffle moves
`bond_u` / `bond_v`, and the branch then reads both quantities at the shuffled
(still real) endpoint — it cannot silently detach structure from semantics.

### 2.1 Implementation discipline

The parent is **extended through one hook**, never copied: `SEM108Model`
gains `_edge_response_delta(coord, data, bond_u, bond_v, mask)` whose default
returns `None` (verified bit-identical parent forward, `J4`), and the edge path
is factored into `_edge_env_parts` / `_edge_env_slots` used by
`_environment_from_parts`.  `JointBondModel` overrides only the hook and adds
the 5 new weight tensors.  The parent keeps every other code path.

## 3. Frozen parent design (unchanged, not re-specified here)

`cm.H1_CONFIG` (`d_e=48`, `K=32`, `s=8`, `lambda=33.95873017865987`,
`horizon=320`, `dict_kind=sdb32`), phi65, CSSD-q1 common subspace
(`q=1`, `rms=[5.082852828320509]`), residual dictionary `Dbar_perp`, IHT-10,
node path (3 shells × 48), original edge path (6 shellpairs × 48→32→48),
`[Sem108 ; size2]` (110-D) interface, environment fusion (446 → 114 → 48),
relation / distance / global / topology backend, moment readout, reader,
reconstruction objective, `cm.C6_MASK` (`cssd.CSSD_MASK is cm.C6_MASK`,
`cm.c6_equivalence_check()`), Adam `lr=1e-3`, `weight_decay=1e-5`, batch 128,
grad clip 5.0, train shuffle offset 91011, eval offset 91012, Top-5 soup.
Nothing is retuned.  The parent's node-binding collapse (`W_A_S` / `W_A_C` →
float32 denormals during the Sem108 run) is **recorded, not repaired**.

## 4. Parameter budget (one shot, never tuned)

```
A 32*16 = 512 ; C 28*16 = 448 ; B 4*48 = 192 ; F 48*32 + 32*48 = 3072
new parameters = 4224
parent (Sem108 candidate) = 97709  ->  candidate total = 101933
relative delta = 4224 / 97709 = 4.3230%   (un-matched by design)
```

The round brief authorises this increment; **no width is changed to match a
budget** and `parameter_matched = false` everywhere.  `J2` asserts the exact
shapes and the exact totals.

## 5. Correctness gates J0–J13 (all must pass before training)

| gate | requirement |
|---|---|
| J0 | frozen parent interface reused: `sem_dim=108`, interface 110, `fusion 446→114→48`, C6 identity + equivalence |
| J1 | Phase-A: parent identity established, parent A2 pass, Sem108 `PROCEED`, prereg present (sha256 recorded) |
| J2 | exact parameter contract: `new=4224`, `total=101933`, frozen shapes |
| J3 | every shared parent state-dict parameter (all except the replaced `fusion.*` / removed `anchor_encoder.*`) bit-identical to a fresh `build_sem108_model`; the only new keys are the 5 `joint_*` weights; nothing removed |
| J4 | hook default is `None`; with the parent weights loaded and `joint_branch_off`, the full prediction and environment are **bit-identical** (`torch.equal`) to `SEM108Model`; with the branch on they differ |
| J5 | routed branch equals the endpoint-pair formulation; swapping **both** endpoints' (structure, semantics) changes nothing (`max|Δ| = 0`); swapping **only** the atom semantics, or **only** the structure codes, changes the branch on a real batch |
| J6 | the same three properties on a distinguishable synthetic endpoint pair (different structures **and** different atom types), plus structure-zero ⇒ `0` |
| J7 | branch-local residual-code zero ⇒ branch output **exactly 0**; branch-on vs branch-off prediction differs; no bias / norm anywhere in the branch; the branch ignores the common coordinate |
| J8 | gradients from the task+reconstruction loss reach `A`, `C`, `B`, `F` and the dictionary `D` (norms `> 0`) |
| J9 | routing / batch offsets: the environment edge slots equal an independent float64 `numpy.add.at` reference on a two-graph synthetic batch (cross-graph bonds `0`), the routed branch equals the manual endpoint formulation, and a shifted endpoint index changes the result |
| J10 | with the parent edge-assignment shuffle, the branch reads structure **and** semantics at the shuffled real endpoint; the shuffled delta differs from the unshuffled delta (never a silent no-op); branch-off predictions equal the parent's under the same shuffle |
| J11 | frozen parent reproduction: the Sem108 seed-0 soup state re-scores its recorded valid MAE within `1e-9` |
| J12 | node-relabelling invariance on real valid molecules (`<= 1e-4`, float32 reduction-order tolerance) |
| J13 | the official-test blocker raises on `official_test_loaded=True`; every payload records `false` |

Any failure ⇒ STOP; no training, no architecture change.

## 6. Smoke (≤ 64 optimizer steps, trainability only)

8 epochs × 1024 official-train molecules / 512 valid = **64 optimizer steps**.
Checks only: finite loss / valid MAE; non-zero gradients at init **and** after
the smoke for `joint_A`, `joint_C`, `joint_B`, `joint_F.0`, `joint_F.2`, `D`,
`fusion.W1`, `fusion.W2`, `W_A_S`, `W_E_S`; branch-on vs branch-off prediction
differs.  The smoke result is **not** used to select any design choice, and the
formal run restarts from the seed-0 initialisation (no continuation).

## 7. Formal run — exactly one trajectory

320 epochs, seed 0, **8 CPU threads**, Adam `lr=1e-3`, `weight_decay=1e-5`,
batch 128, clip 5.0, train shuffle offset 91011, eval offset 91012, identical
Top-5 soup on valid MAE.  One seed; no second run; no early stopping, width,
init, lr, lambda or horizon change after observing the curve.  The formal run
goes through the repository control plane (`research run`,
`--mode scratch`, `test_access = blocked`).

## 8. Frozen inference probes (soup state only, no retraining)

| probe | content |
|---|---|
| **B1** | `joint_branch_off` — remove only `j_uv`; every parent path unchanged |
| **B2** | `joint_alpha_zero` — zero the residual code read by the branch (diagnostic) |
| **B3** | `use_joint_alpha_shuffle` — permute the residual-code rows **within each molecule** for the branch only, seeds `101 / 202 / 303` |
| **B4** | `use_joint_atom_shuffle` — permute the atom one-hot rows **within each molecule** for the branch only, seeds `101 / 202 / 303` |
| **P1** | parent `residual_zero` (diagnostic) |
| **P2** | parent node-assignment shuffle, seeds `101/202/303/404/505` |
| **P3** | parent edge-assignment shuffle, seeds `101/202/303/404/505` |
| **P4** | parent Sem108 row shuffle, seeds `101/202/303/404/505` |

`G_branch_off = M_B1 - M_S`; `G_joint_alpha_shuffle = mean B3 - M_S`;
`G_joint_atom_shuffle = mean B4 - M_S`; `G_dict0`, `G_node`, `G_edge`,
`G_sem_shuffle` as in the parent.  The probe masks are unions of the frozen C6
training mask and the probe; the branch / parent fields are orthogonal.

**Recorded mechanism facts (no gate, reported either way):** whether the
original node binding is still dead (`W_A_S` / `W_A_C` absmax `< 1e-30`),
whether the new branch is dead (all branch parameters `< 1e-30`, or a zero
branch response, or `|G_branch_off| < 1e-6`), and the branch-vs-parent edge
response scales.  The common coordinate is never counted as part of the new
branch's residual-dictionary mechanism.

## 9. Performance intervals and decision rule (frozen before the run)

`M_S` = Top-5 soup valid MAE of the single seed-0 run.

| interval | label |
|---|---|
| `M_S <= 0.120` | `JOINTBOND_FURTHER_CONFIRMATION_BAND` — worth further confirmation |
| `0.120 < M_S <= 0.123` | `JOINTBOND_CANDIDATE_SIGNAL_BAND` |
| `M_S > 0.123` | `JOINTBOND_NO_NEW_TASK_BAND` — no clear breakthrough |

**Buy a matched control** iff the interval is one of the first two **and**
`G_branch_off >= 0.003`.  Otherwise do not buy; record the mechanism result and
stop.  The historical Sem108 value `0.123704927947314` is an unmatched
reference only (there is no contemporaneous baseline in this round, and the
`+4224` parameters are un-matched), so no statement about "the improvement
coming from joint fusion" or "evidence for the sparse dictionary" may be made.
Only validation is reported; the official test is never read.

**Stop discipline.** After the single 320-epoch run and the frozen probes, the
round stops.  No seed 1, no rescue run, no width / init / lr / lambda / horizon
change, no retrained baseline, no second candidate, no DenseTied, no parameter
search, no official-test read.  A deterministic implementation bug may be fixed
and re-verified before the formal run; the architecture, initialisation, budget
and stop conditions may not be adjusted from probe behaviour.

## 10. Result layout

`tracks/ksvd/results/e2e_dictenv_jointbond_v1/`

```
historical_references.json         audit/audit_decision.json
preflight.json  preregistration_snapshot.json  parameter_audit.json
correctness.json                   smoke.json
run_seed0.json  curve_seed0.csv    soup.json
mechanism/{branch_off, joint_alpha_zero, joint_alpha_shuffle, joint_atom_shuffle,
           residual_dict_zero, node_assignment_shuffle, edge_assignment_shuffle,
           sem_root_shuffle, branch_health, binding_health, dictionary_health}.json
summary.json  REPORT.md  DECISION.md
```

Every JSON carries `protocol_version`, `git_commit`, `device = cpu` and
`official_test_loaded = false`.
