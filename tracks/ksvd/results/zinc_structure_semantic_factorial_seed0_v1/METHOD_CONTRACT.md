# METHOD_CONTRACT — zinc-structure-semantic-factorial-seed0-v1

Frozen 2×2 factorial on the current canonical `Full`
(`e2e_dictenv_scale_v1`, 408,651 trainable parameters). Source commit
`8126300`. One seed only (`seed=0`), four fresh 240-epoch trajectories:
`S_J / S_M / D_J / D_M`. This document maps every claim to the real source,
shape and loss it depends on.

## 0. The current method, mechanically

### 0.1 phi65, D, U, residual code (item 1)

* `phi65 = data.dict_phi` (`[n_nodes, 65]`, `P2Model.code` input). It is a
  train-only normalized structural descriptor already cached in
  `encoded_train.pt`; the round never rebuilds it.
* `D_raw` is `model.D`, a trainable `[65, 32]` `nn.Parameter` initialised from
  `blob["D_fit"]` (`e2e_dictenv_common_subspace_dictionary_v1.CSSDModel`,
  `P2Model.__init__`).
* `U` is a **fixed buffer** `[65, 1]` from `blob["U_components"]`
  (`zinc_joint_dictionary_decision_v1.fold_subspace` → `CommonSubspace`,
  `kind="q1"`, the uncentred train-mean direction). `common_rms` is the fixed
  train RMS of the raw common coordinate.
* `Dbar = colnorm((I − U Uᵀ) D_raw)` (`CSSDModel.residual_dictionary`,
  `v0.normalized_dictionary`), and `d = phi @ U`,
  `r = phi − d @ Uᵀ`.
* `D_fit`, `U_components`, `U_rms`, the 8000/2000 split and all input
  transforms are fit on the **8000 fit molecules only** (`fold_objects.npz`,
  produced by the frozen `zinc_joint_dictionary_decision_v1` prep). The 2000
  dev rows never enter any dictionary/subspace/scale fit.

### 0.2 Node and edge binding (item 2)

`SEM108Model._environment_from_parts` (the only slot-construction path used by
the canonical Full through `LatentBridgeSEM108`):

* Node: `c = coord[env_occ_node]`, `q = one_hot(dict_atom, 28)`,
  `a_i = c @ W_A_S` (`[n_occ, 96]`), `b_i = q[env_occ_node] @ W_A_C`
  (`[n_occ, 96]`). The per-occurrence product
  `a_i ⊙ b_i / sqrt(96)` is `index_add_`-ed into bucket
  `(env_occ_root, env_occ_shell)` over the `n_nodes × 3` shell grid.
* Edge: `g_i = [z_u+z_v, |z_u−z_v|, z_u⊙z_v]` (endpoints `u`,`v` stay paired),
  `a_i = g_i @ W_E_S` (`[n_bond, 48]`),
  `b_i = one_hot(env_bond_type, 4) @ W_E_C` (`[n_bond, 48]`). The product
  `a_i ⊙ b_i / sqrt(48)` is `index_add_`-ed into bucket
  `(env_bond_root, env_bond_shellpair)` over the `n_nodes × 6` shell-pair grid.
* `coord` is the structural code (width 33, §0.3); the node/edge encoders then
  read the slot tensors unchanged.

`data.batch` / `ptr` offsets are applied to every occurrence/endpoint field by
`p1.env_collate` (`_OCC_OFFSET_BY_COUNT`, `_offset_occurrences`), so buckets
never mix graphs.

### 0.3 Sparse code vs dense control (item 6, §6 of the task)

* S: `alpha_S = tied_IHT(Dbar, r; s=8, steps=10)`
  (`v0.tied_iht_codes`); `z_S = [d / common_rms, alpha_S]` (width 33);
  `r_hat_S = alpha_S @ Dbarᵀ`.
* D: `alpha_D = kappa * (r @ Dbar)` (no IHT, no extra encoder/decoder);
  `z_D = [d / common_rms, alpha_D]` (width 33);
  `r_hat_D = (alpha_D / kappa) @ Dbarᵀ`.
* `kappa` is a frozen non-trainable buffer computed once on the shared initial
  `D`/`U` over all 8000 fit nodes:
  `kappa = sqrt(Σ alpha_S² / Σ (r @ Dbar)²)`, and is shared by `D_J`/`D_M`.
  It matches the **initial residual-code RMS**; it does not match direction,
  rank or gradient geometry.
* `reconstruction_loss` is unchanged:
  `mean_nodes(‖r − r_hat‖² / (‖phi‖² + 1e-12))`, with `λ = H1_LAMBDA =
  33.95873017865987`. All four arms keep the same reconstruction target and
  the same task dictionary `D_L/V_L`.

### 0.4 Sem108 / size2 (item 3)

`semantic_interface = [Sem108(108); size2(2)]` (`SEM108Model.semantic_interface`).
Sem108 is the shell-resolved primitive semantic block:
`atom_shell [3 × 28 = 84]` and `bond_shell [6 × 4 = 24]`, read from
`patch_cont[:, 0:108]`. It therefore carries **shell-conditioned** semantic
statistics — it is not a pure global semantic descriptor and it retains part of
the structural conditioning. `size2 = anchor[:, 60:62]`. Both are untouched by
this round (only the C6 mask is applied, exactly as in canonical Full training).

### 0.5 Fusion → task dictionary → reader (item 4)

Order in `AuditModel.forward` / `LatentBridgeSEM108`:

1. `coord = model.code(phi65)` (structural code).
2. `E = fusion([Sem108; size2; node_slots(3×48); edge_slots(6×32)])`
   (`446 → 342 → 144`, `SiLU`), then the shared task dictionary
   `LatentDictionaryBridge` (16-step unrolled soft-threshold ISTA,
   `D_L [144, 288]`, `V_L [288, 144]`, `rho`, `λ1=0.05`, `λ2=0.01`).
   **The structural dictionary `D`/`U` is separate from the task dictionary
   `D_L/V_L`.**
3. `unary = pool_moments(E, batch)`; `u = pair_projection(E)`; relation
   `15→96→48`; distance gate `1+tanh`; `pair_encoder`; global encoder
   `62→32→32`; topology `25→16→8`; `reader(814→39→39→1)`.
4. C6 mask removes graph atom/bond histograms, unary/pair counts and
   `path_count`; anchor chemistry, Sem108, size2, topology25, pair relations and
   the structural code remain.

### 0.6 Gradients, extra statistics, no message passing (item 5)

* Task gradient `L1(pred, y)` reaches `D` (through the code), `W_A_S/W_A_C`,
  `W_E_S/W_E_C`, `D_L/V_L`, the encoders, fusion, pair path, global/topology
  encoders and the reader.
* Reconstruction gradient `λ · L_rec` reaches `D` (through `Dbar`) and the
  `common` path; it does **not** touch `D_L/V_L` (`CSSDModel.reconstruction_loss`
  uses `residual_dictionary`, not the task bridge).
* Extra statistics kept in all four arms: Sem108 (108), size2 (2),
  atom/bond marginals inside Sem108, shell/shellpair positions (3/6),
  environment composition through `node_slots`/`edge_slots`, pair relations,
  global context, topology25. `M` removes only the within-bucket
  per-occurrence covariance; it does not remove these bypasses.
* No message passing and no Transformer: `_model_has_message_passing` finds no
  such module; the graph is read by fixed moment pooling and pair moments only.

## 1. What is held fixed vs changed (item 6)

| dimension | S_J | S_M | D_J | D_M |
|---|---|---|---|---|
| raw inputs / phi65 | same | same | same | same |
| D_raw / U / common_rms | same | same | same | same |
| trainable parameters | 408,651 | 408,651 | 408,651 | 408,651 |
| initial parameter values | identical | identical | identical | identical |
| coding operator | IHT s=8 | IHT s=8 | dense `kappa·r@Dbar` | dense `kappa·r@Dbar` |
| slot aggregation | per-occurrence product | independent pairing | per-occurrence product | independent pairing |
| Sem108 / size2 / topology / relations | unchanged | unchanged | unchanged | unchanged |
| task dictionary `D_L/V_L` | trained | trained | trained | trained |
| loss | `L1 + λ L_rec` | same | same | same |
| data stream / optimizer / budget | same | same | same | same |

### Supported by this design

* Whether the per-occurrence binding (J) beats the analytic independent-pairing
  expectation (M) at matched marginals, per coding.
* Whether the sparse tied-IHT code beats a same-matrix dense linear code at
  matched initial residual RMS, per binding.
* The coding × binding interaction `I = C_S − C_D = G_J − G_M`.

### Not supported by this design

* It is **not** "all dictionaries vs a dictionary-free MLP": the dense arm keeps
  the trainable shared structural matrix `D` and the task dictionary.
* It does **not** remove all structure–semantics interaction inside the model:
  Sem108, atom/bond marginals, shell/shellpair positions, pair relations and
  topology remain.
* Single seed, fixed 2000-row reused train-inner dev: no multi-seed robustness
  claim, no confirmatory held-out claim.
* `kappa` controls initial residual-code amplitude only, not direction/rank or
  the full gradient geometry of the dense code.

## 2. Inheritance-chain risk (verified, not assumed)

`SEM108Model.__init__` raises if `node_binding != "paired"` and its
`_environment_from_parts` calls its own `_node_binding`/`_edge_env_slots`, and
`CSSDModel.code` overrides `CleanMechModel.code`. Therefore simply setting
`CleanMechSpec(coding="dense_tied", binding="indep")` would **not** take effect
on the canonical Full path. This round implements the operators directly on the
subclass `FactorialFull` and proves, per real `forward(mask=C6)`, that the
operator executed: `operator_checks.json` records `indep_calls > 0`, the dense
code vs the IHT code, the micro-bucket enumeration (mean J over all semantic
permutations == M), per-molecule/batch slot equality and graph-order
invariance.
