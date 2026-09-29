# E2E-DictEnv-RNDB-v1 — prior-artifact audit

Round **E2E-DictEnv-RNDB-v1** · protocol `e2e_dictenv_rndb_v1` · study `zinc-context-gap`.
Method tag: **RNDB = Rolewise Nonlinear Dictionary Binding**.
Starting HEAD: `cfe0716f3c4ac3c72df9521c9e075092b5213ac3`.

This audit is written **before any RNDB implementation or run**.  It fixes which
historical artifacts RNDB reuses, which historical builders are only correctness
references, and answers the eight pre-registered prior-art questions.

---

## 0. The one narrow question

Current CSSD / FINAL-CLEAN binding collapses the structural coordinate before
any nonlinearity:

```
p_v = sum_k p_{v,k}            (linear role aggregation)
p_v -> shell slot -> node_encoder (SiLU MLP, the first nonlinearity)
```

RNDB asks whether a genuinely different **function class** is obtained by moving
a small *shared* nonlinear operator **before** the role aggregation:

```
h_{v,k} = p_{v,k} + psi_A(p_{v,k})       (per residual dictionary role)
u_v     = p_v^common + sum_{k in K_dict} h_{v,k}
```

No new information, no new message passing, no `coarse146` bypass, no new
dictionary, no width/rank/chemistry change.

---

## 1. Reused frozen parent artifacts (identity)

| artifact | path | identity | RNDB use |
|---|---|---|---|
| CSSD core module | `experiments/luyin16/e2e_dictenv_common_subspace_dictionary_v1.py` | blob `208c825978d0ef86faf7cf674023548a1afdb430` at the CSSD formal commit `5f0f284070cde1019e8c2b6bc762b8dd0223022b`; current blob `34f6eebc...` differs **only** by the additive `model_factory` hook (behaviour unchanged for `model_factory=None`) | parent model class `CSSDModel`, frozen trainer `train_cssd`, residual dictionary / IHT-10 coder |
| cleanup-mechanism module | `experiments/luyin16/e2e_dictenv_clean_mechanism_v1.py` | blob `f183e472...` (unchanged since `5f0f284`) | `CleanMechModel`, canonical `C6_MASK`, `H1_CONFIG`, `H1_LAMBDA` |
| clarity-audit module | `experiments/luyin16/e2e_dictenv_h1_clarity_audit.py` | blob `9cb048c0...` (unchanged since `5f0f284`) | `AuditModel` masked forward, `evaluate_mask`, `AuditMask`, assignment-shuffle semantics, `SHUFFLE_SEEDS=(101,202,303,404,505)` |
| P1 data / loader | `experiments/luyin16/e2e_dictenv_p1.py` + `zinc_e2e_dictenv_p1.py` | blob `82e06707...` | `phi65` incidence, `load_split`, `make_env_loader`, shell / shellpair definition, `P1_RELATION_INDICES` |
| P2 config / dictionary | `experiments/luyin16/e2e_dictenv_p2_abs.py` | blob `8bcfba5d...` | `H1_CONFIG` (`decoder="h1"`, `d_e=48`, `K=32`, `s=8`, `lambda_factor=0.25`, `horizon=320`), `load_dictionary`, optimizer constants |
| V0 coder | `experiments/luyin16/e2e_dictenv_v0.py` | blob `df2e87a5...` | `normalized_dictionary`, `tied_iht_codes`, exact top-`s` |
| CSSD q1 common subspace | `results/e2e_dictenv_common_subspace_dictionary_v1/common_subspace.json` | q1, `rms=[5.082852828320509]`, fit on official train only | frozen `U` (65×1) and common scaling |
| CSSD q1 soup | `results/e2e_dictenv_common_subspace_dictionary_v1/training/final.json` | `CSSD-Q1-seed0`, soup `0.13002798487985273` | historical reference only (never loaded into RNDB) |
| dictionary `sdb32` | `results/sdb_v0/dictionary.pt` (loaded by `p2run.load_dictionary`) | 65×32 K-SVD, train-only | RNDB dictionary initialisation |

The parent configuration is resolved from the code, not from this document:
`cm.H1_CONFIG`, `cm.H1_LAMBDA`, `cssd.CSSD_MASK is cm.C6_MASK`, subspace q1.

---

## 2. Answers to the eight pre-registered questions

### Q1. Is RNDB already implemented in the repo?

**No.** A repository-wide search for `RNDB`, `rolewise`, `nonlinear-before`,
`psi_A`, `psi_E` finds no implementation.  The nearest items are:

* `e2e_dictenv_capacity_localization_v2.next_round_shape` **proposes** (in a note,
  never implemented) an iterative composition `E_i^(l+1) = E_i^l + eta_l
  Agg_j psi(E_i^l, E_j^l, r_ij)` — that is a *pair/environment* update with
  message passing, explicitly forbidden here;
* the F / R / G capacity candidates add whole residual *modules* to the parent
  (see Q2), not a reordering of the existing binding operator.

No existing class applies a shared nonlinearity to individual dictionary-role
responses before role aggregation.  RNDB is new.

### Q2. Is RNDB merely a reparameterisation of F (multi-rank bilinear fusion)?

**No.** F (`e2e_dictenv_capacity_localization_v1.FusionCapacityModel`) computes,
at every occurrence,

```
u = u_base + F_NP [ (c W^h_S) odot (q W^h_C) / sqrt(d_h) ]_h ,   h = 1..4
```

i.e. a sum of `H` bilinear terms that are **linear in the coordinate `c`** and
additive on top of the untouched base product; the whole F residual is an affine
function of `c` (for fixed `q`) followed by the *same* later nonlinearity
(`node_encoder`/`fusion`).  F is a *wider linear* binding.

RNDB does not add heads, does not read the coordinate through a new linear map,
and is **not affine in the coordinate**: `sum_k psi_A(p_{v,k})` is a sum of
nonlinear transforms of the individual role products.  For a fixed chemistry
projection `c_v`,

```
parent:  g( sum_k z_k (w_k odot c_v) )
RNDB:    g( sum_{k in K_dict} psi_A( z_k (w_k odot c_v) ) + common )
```

with the same later `g`.  Since `psi_A` is nonlinear and `sum_k psi_A(p_k) !=
psi_A(sum_k p_k)` in general, RNDB's per-role map is not representable by any
finite-head affine F residual.  The two are different function classes.

### Q3. Is RNDB merely a linear compression of FEC-D1's 2688-D joint statistic?

**No.** FEC-D1's statistic is an explicit second-order covariance
`C_ik = sum_{v in shell k} (a_v - a_bar)(q_v - q_bar)^T in R^{32x28}`,
flattened to `2688` dimensions and fed as a handcrafted joint input.  RNDB
computes no outer-product/covariance statistic and adds no joint tensor; it
modifies the operator ordering inside the existing role x chemistry bilinear
binding.  Nothing is compressed and no new descriptor is formed.

### Q4. Is RNDB merely an FSAB / BCE residual branch?

**No.** FSAB (`structural_patch_encoder.FactorizedStructureAttributeBinding`)
and BCE (`binding_composition_encoder`) add a **new structure–attribute binding
stream `B_v`** (a centered role-x-attribute product pooled into a separate
representation) and a **new residual MLP** on top of the existing encoder.
RNDB adds no stream and no representation: it changes where the existing
nonlinearity sits relative to the existing role sum, using only the already
formed products `p_{v,k}`/`p^E_{uv,k}`.

### Q5. Is RNDB merely another T1 `coarse146` bypass?

**No.** T1 (`A2_COARSE146_SLOT48`) exposes the 146-D `patch_cont` coarse
descriptor to the environment decoder, which is a pure information bypass and
made the dictionary non-load-bearing (`G_dict-use` fell to `+0.0179`).  RNDB
never touches `patch_cont`/`atom_shell`/`bond_shell` and never reads raw `phi`
inside `psi`; `psi` sees only the already-formed per-role products.  The
information content is identical to the parent.

### Q6. Why is `nonlinear-before-role-sum` a different function class?

Let `z` be the structural coordinate and `c_v = q_v W_A_C / sqrt(D_A)` the
chemistry projection.  The parent node binding is exactly

```
p_v = sum_k p_{v,k},     p_{v,k} = z_{v,k} (W_A_S[k,:] odot c_v).
```

The environment then applies a first nonlinear map `g` (the node encoder MLP).
The parent therefore computes `g( A(z, c_v) )` where `A` is a **single** map
that has already linearly aggregated all role contributions: the only
`z`-dependence seen by `g` is the `D_A`-vector `sum_k z_k w_k odot c_v`.  Any
role-specific nonlinear interaction of `z` and `c` that is not expressible
through that one pre-aggregated vector is outside the parent class.  F only
enlarges the *linear* pre-aggregation (Q2), it does not lift the aggregation
before `g`.

RNDB instead computes

```
u_v = sum_{k in K_common} p_{v,k} + sum_{k in K_dict} [ p_{v,k} + psi_A(p_{v,k}) ]
    = p_v + sum_{k in K_dict} psi_A(p_{v,k}),
```

so each residual dictionary role passes through a shared nonlinearity **before**
the role sum, and only then does the environment's `g` act.  Because `psi_A` is
nonlinear and shared, the role index is not an extra identity: RNDB is
permutation-equivariant over dictionary roles (G3), and with `psi=0` it is
*exactly* the parent (`sum_k p_{v,k} = p_v`).  The parent is a special case of
RNDB (G1); the reverse does not hold.  That inclusion is the whole point of the
experiment.  The common coordinate is deliberately **excluded** from `psi`,
because it is an independent dense direction, not a dictionary atom (Q8).

### Q7. What is the current CSSD structural-coordinate layout?

From `CSSDModel.code` (width `D_struct = K_ATOMS + common_dim`):

```
z_v = [ c~ ; alpha_res ]        in R^{32 + q}
c~       = (phi_v @ U) / common_rms          # q common coordinates (q = 1 for q1)
alpha_res = IHT_10( Dbar_perp, r_v )          # 32 sparse residual dictionary codes
r_v      = phi_v - (phi_v @ U) U^T
Dbar_perp = colnorm( (I - U U^T) D )
```

For the frozen CSSD-q1 parent: `q = 1`, `K_ATOMS = 32`, `D_struct = 33`.
`W_A_S` has shape `[33, 96]`; `W_E_S` has shape `[3*33, 48]`.

### Q8. Exact common vs residual dictionary coordinate indices

In the CSSD coordinate:

```
K_common = {0}                 # common_dim = q = 1
K_dict   = {1, ..., 32}        # the 32 sparse residual dictionary roles
```

In `W_A_S` (`[q + K, D_A]`): rows `[0:q)` are the common rows (zero-initialised
in `_widen_binding_`), rows `[q:q+K)` are the dictionary rows.

In `W_E_S` (`[3*(q+K), D_E]`, block order `[+ , |.| , odot]`): within block
`b in {0,1,2}`, rows `b*(q+K) + [0:q)` are the common rows and rows
`b*(q+K) + [q:q+K)` are the dictionary rows.

RNDB applies `psi` only to `K_dict` in both bindings; the `K_common` rows keep
the parent's original linear binding.

---

## 3. Historical builders used as correctness references only

* `e2e_dictenv_h1_clarity_audit.AuditModel` / `evaluate_mask` — masked forward
  and evaluation reference (G0/G1/G5).
* `e2e_dictenv_clean_mechanism_v1.CleanMechModel` — the exact parent binding
  being decomposed (G0).
* `e2e_dictenv_capacity_localization_v1.FusionCapacityModel` — F, read for the
  prior-art comparison only; never instantiated by RNDB.
* `fec_d1` / `fec_s0_factorization` / FSAB / BCE — read for the prior-art
  comparison only; never an RNDB input.

---

## 4. No-equivalence verdict

RNDB is not an existing repo mechanism.  It is not F (linear multi-rank
residual), not FEC-D1 (2688-D covariance statistic), not FSAB/BCE (new binding
stream + residual MLP), and not T1 (coarse146 bypass).  It is a strict function
class extension of the frozen CSSD-q1 binding obtained solely by reordering the
existing shared nonlinearity relative to the existing per-role sum.  The round
may proceed to preregistration.
