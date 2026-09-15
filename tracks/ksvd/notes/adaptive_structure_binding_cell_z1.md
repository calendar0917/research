# Z1 — Local Adaptive Structure-Binding (ASB) Cell on ZINC patches

**Date:** 2026-09-16
**Branch:** `exp/adaptive-structure-binding-z1`
**Commit (formal runs):** `cadef7268563fdd29dd617a3579e1fe48cbe4b28`
**Run tags:** `asb-s0` (GPU0), `asb-s1` (GPU1)
**Remote:** `res` = `hxy@a100-2`, repo `/home/hxy/cy/research`
**Protocol version:** `adaptive_structure_binding_cell_z1_v1`
**Verdict:** **STOP (Case E — branch / structure collapse).** The frozen ASB
cell does **not** learn a binding-conditioned, adaptive support. One seed stays
on the full radius-2 patch with a *global constant* gate whose binding-message
branch is exactly dead; the other seed collapses to a **root-only** support and
its whole encoder output becomes a constant. Performance is a tie with the
connectivity-free B-Bag (+0.00173, sign-flipped across seeds) and clearly worse
than B-Full (+0.00656). The hypothesis is **not cleanly tested** (utilization /
optimization failure), so this is **not** evidence that adaptive structure is
worthless; it is evidence that this implementation does not realize it. No
promotion, no Z2. Official ZINC test **never loaded**.

---

## 1. Question

Can a single **local** cell, run inside each rooted radius-2 patch, **form its
own explicit connected support** from structure–attribute bindings, keep the
real induced bond set, and then run binding-aware computation on that support —
using the *same* binding states that decided selection — while preserving the
frozen downstream (pair system, T=2 pair–centre, head, protocol, Top-5 soup)?

Pre-registered gates (from the brief): only **Z1** (no Z2 outer context, no
motif vocabulary, no hard top-k, no attention/Transformer, no width/depth/sparsity
sweeps, no auxiliary branches); B-Bag must be the *init / degenerate* case, not a
permanent bypass; params ≤ 120,000; per-epoch mechanism diagnostics recorded;
official test locked.

## 2. Pre-registered architecture (exactly one)

`patch_representation="adaptive_structure_binding"`, implemented in
`tracks/ksvd/experiments/luyin16/structural_patch_encoder.py`
(`AdaptiveStructureBindingEncoder`) and wired through
`zinc_patch_path_pooling.py` (`asb_*` kwargs).

Constants (locked): `node_dim=48, edge_dim=24, node_hidden=96, bond_hidden=48,
bind_hidden=32, gate_hidden=24, message_hidden=32, update_hidden=32,
bind_update_hidden=32, fusion_hidden=104, perturb_init=1e-3, gate_bias_init=3.0,
gate_weight_init_std=1e-2`, output `e_struct ∈ R^16`.

Within a patch `(V_i, E_i)` rooted at `r_i`:

- **Primitives.** Node `z_u = node_mlp([atom_emb, root_flag_emb, dist_emb])`;
  bond `base_uv = bond_mlp(bond_emb)`.
- **Binding.** `b_uv = base_uv + bind_delta_mlp([z_u+z_v, |z_u-z_v|, bond_emb])`
  (the delta final layer is `perturb_init=1e-3`, so `b_uv ≈ base_uv` at init).
- **Gate.** `a_uv = sigmoid(gate_mlp(b_uv))` with `gate_bias_init=3.0`
  (`≈ 0.953` open). Hard membership `â_uv = StraightThroughBinary(a_uv)`
  (threshold `> 0.5`).
- **Support.** A node is in the support iff its level-1 activation is on;
  level-2 activation uses `1 - Π(1 - a·â)` over incident/2-hop edges; the node
  activation is the ST-binarised product. `selected = ST(a_soft)`.
- **Induced edges.** Every **real** bond of the patch whose two endpoints are
  both selected (unordered, `src < dst`).
- **Binding-aware computation on the support.** Symmetric message
  `m_v = mean_{u ∈ N(v) ∩ support} message_mlp(b_uv)`; update
  `z2_v = z_v + update_mlp([z_v, m_v])`; binding states updated by
  `bind_update`; support pooled as `bind_mean/bind_std`; fused with the node /
  bond / attribute blocks into `e_struct`.
- **B-Bag degeneration.** With **all gates forced open** and the perturbation
  paths zeroed, the cell reproduces B-Bag exactly (verified below).

**Forbidden and absent:** outer context / Z2, motif vocabulary, learned
dictionary, hard top-k, attention/Transformer, auxiliary losses, width/depth/
sparsity sweeps, `typed_embedding`, official test.

## 3. Training protocol (inherited, unchanged)

Adam `lr=1e-3`, `wd=1e-5`, batch `128`, `max_epochs=240`, `patience=40`,
no scheduler, L1/MAE, single-stage, grad-clip `5.0`, best-official-valid
checkpoint, fixed Top-5 equal-weight parameter soup. Radius-2 extraction, pair
system, T=2 pair–centre, head, and all inherited modules are frozen. Seed 0 and
seed 1 only.

## 4. Parameters

| model | encoder | total |
|---|---|---|
| B-Bag (reference) | 35,168 | 84,511 |
| B-Full (reference) | 35,152 | 84,495 |
| **ASB (Z1 candidate)** | **52,193** | **101,536** |

ASB encoder = 52,193 (Δ **+17,025** vs B-Bag encoder); downstream inherited =
49,343. Budget bound 120,000 → `params_in_range = true`.
`dataset_dependent_params = 0`, `candidate_has_typed_embedding = false`,
`official_test_loaded = false`.

Encoder breakdown: node_mlp 9,360; bond_primitive_mlp 2,376; binding_delta_mlp
4,664; bind_update_mlp 4,664; gate_mlp 625; message_mlp 2,384; update_mlp 4,688;
fusion 21,752; atom/root/distance/bond embeddings 1,680. Growth is entirely the
adaptive cell (binding/gate/message/update + a wider support-aware fusion);
no dataset-dependent parameters were added.

## 5. Tests and sanity (all pass)

- 14 unit tests (`tests/test_adaptive_structure_binding_cell.py`): permutation
  invariance, support connectivity + rootedness, induced-edge correctness,
  full-gate degeneration, no vocab-sized params, gradient viability, batch
  invariance, edge-order invariance. **14 pass** locally and remotely.
- 21/21 sanity checks. Key exact checks:
  - init Δ vs B-Bag = **6.27e-4** (near-function-preserving init),
  - full-gate degeneration Δ vs B-Bag = **8.49e-6** (exact in the limit),
  - batch invariance Δ = **0.0**,
  - gate / binding / message / update / bind_update gradients all nonzero at init,
  - supports connected and rooted; `q16 h64 T2`; recurrent tying preserved.
- Deterministic GPU repro (2 epochs) on GPU0 and GPU1 **bit-identical**:
  selection-state SHA-256
  `fcc397f6ed763e0278fed76686349951eab9ec2b5557b8d312de3020a5909590`,
  best-valid `0.6712156`.

## 6. Results (official valid, Top-5 soup)

| seed | best raw valid | best epoch | soup valid | Top-5 epochs | epochs run | steps | wall |
|---|---|---|---|---|---|---|---|
| 0 | 0.130111 | 225 | 0.126082 | 225/216/222/213/212 | 240 | 18,960 | 6372 s |
| 1 | 0.127911 | 195 | 0.124989 | 195/228/232/187/218 | 235 (early stop) | 18,565 | 3151 s |

**2-seed soup mean = 0.1255354** (raw mean 0.1290110).

| comparison | Δ soup (ASB − ref) | per-seed |
|---|---|---|
| B-Bag (0.1238055) | **+0.0017299** | seed0 −0.0013002, seed1 +0.0047600 (**sign flip**) |
| B-Full (0.1189722) | **+0.0065632** | +0.0062635, +0.0068629 (both worse) |
| A2 (0.1219141) | **+0.0036214** | — |

ASB is a **tie/no-gain** versus the connectivity-free B-Bag (within ±0.002 but
sign-unstable) and clearly **worse** than B-Full. No promotion.

## 7. Mechanism — the decisive result

Per-seed support / gate / state statistics **at the selection checkpoint**
(5864 valid patches; mean patch size 6.114):

| quantity | seed 0 | seed 1 |
|---|---|---|
| selected_node_fraction | **1.0000** | **0.1636** |
| selected_edge_fraction | 1.0000 | **0.0000** |
| distance-1 / distance-2 selection | 1.0 / 1.0 | 0.0 / 0.0 |
| support_size mean/std | 6.114 / 1.515 | **1.0 / 0.0** |
| gate_probability mean | 0.5333 | 0.5000 |
| gate_probability **std** | **0.0** | **0.0** |
| message_norm_mean | **0.0** | **0.0** |
| binding_state_norm_mean | 0.6399 | **0.0** |
| node_state_norm_mean | 0.4775 | **2.6e-13** |
| update_norm_mean | 0.1658 | **1.3e-13** |
| output_token_norm_mean | 0.1128 | 0.0050 |
| e_struct effective rank | 1.0145 | **0.0** |

The weight-level audit of the two selection-state checkpoints is conclusive:

- **Seed 0.** `message_mlp` **all weights/biases = 0.0** (the binding-message
  branch is exactly dead). `gate_mlp` first-layer weight `= 0.0`, last-layer
  weight `= 0.0`, last bias `= 0.1334` → the gate is the **global constant**
  `sigmoid(0.1334) = 0.5333` with *no* dependence on the binding state.
  `bind_delta_mlp` is alive (~0.5) and feeds the pooled binding block, so seed 0
  effectively degenerates to *B-Bag + a node-wise residual update*; `e_struct`
  is near rank-1 (top singular fraction 0.998).
- **Seed 1.** `message_mlp`, `gate_mlp`, `update_mlp`, `bind_delta_mlp`,
  `bind_update`, `node_mlp`, `bond_mlp` are **all exactly 0.0**; only `fusion`
  is tiny-nonzero. The encoder output is a **constant** (rank 0).

Training trajectory (per-epoch curve):

- **Epoch 1 (both seeds):** message branch alive (`message_norm ≈ 0.6`),
  `binding_state_norm ≈ 1.2`, `gate ≈ 0.950`, `gate_std ≈ 1e-5` — i.e. the gate
  is **input-independent already at init**, before any training.
- **By epoch 60:** `message_norm` → `9e-9` (seed 0) / `0.0` (seed 1); the
  binding-message branch dies **early in both seeds**.
- **Gate declines monotonically** in both seeds (0.950 → 0.645 by ep60), i.e.
  the task gradient consistently prefers *pruning* the support.
- **Seed 1:** the gate logit crosses 0 → hard support flips full → **root-only**
  at ~epoch 106–109 (`selected_fraction = 0.1636 = 1/6.114`), after which gate
  and binding gradients are exactly 0 and the whole cell is annihilated
  (`update_norm → 1e-13` by ep160). Root-only is an **absorbing dead state**.
- **Seed 0:** the gate asymptotes to logit ≈ 0.13 (`gate ≈ 0.533`, still > 0.5),
  so the hard support stays full forever; the cell is B-Bag-like but cannot
  improve because the only surviving binding path is the *pooled* block (the
  same information B-Bag already has).

**Interpretation.** The cell never forms a differentiated, binding-conditioned
structure. Two coupled failures produce this:

1. The gate is effectively a **global constant** (tiny gate init +
   hard-threshold step function that gives no continuous, edge-specific forward
   gradient); it is not a function of the binding state.
2. The **near-function-preserving tiny-init perturbation paths** produce
   gradients whose scale is proportional to tiny final-layer weights; Adam with
   coupled weight decay (`wd=1e-5`) then annihilates exactly those parameters
   (a zero/near-zero-gradient parameter loses `≈ lr = 1e-3` per step). The
   message branch dies, and — once the support collapses — the entire encoder
   dies to exactly zero.

The surviving quantity is the pooled binding block, which is the same
connectivity-free object B-Bag already uses. Hence the tie with B-Bag and the
clear loss to B-Full.

## 8. Decision logic (corrected)

The first `decide` run used a rule that required **both** seeds' support
fractions to be non-full before calling support non-trivial; seed 1 was
non-trivial but seed 0 was full, so it mislabelled the result
`D_gates_always_on`. The rule was corrected to classify dead/constant
mechanisms explicitly (`message_norm < 1e-6`, encoder effective rank `< 0.5`,
gate std `< 1e-4`) and to treat per-seed non-triviality independently. The
corrected decision is **`E_branch_or_structure_collapse`** with
`binding_message_branch_dead = true`, `asb_encoder_collapsed = true`,
`gate_input_independent = {0: true, 1: true}`,
`support_nontrivial_per_seed = {0: false, 1: true}`.

## 9. Stop rules honoured

- Official ZINC test **never loaded** (not for any decision).
- Only official train/valid used; no architecture selected on test.
- Exactly one architecture; no width/depth/sparsity/gate-threshold sweep, no
  auxiliary losses, no attention/Transformer, no motif vocabulary, no Z2.
- Seed 0 doubles as the engineering seed; paired seeds 0/1 were pre-registered
  and run once; no extra seeds purchased.
- B-Bag is the init/degenerate case (verified exactly) — not a bypass.

## 10. Provenance & artifacts

Remote results (pulled locally, `.pt` omitted for the report copy):
`tracks/ksvd/results/adaptive_structure_binding_cell/` —
`runs/asb_seed{0,1}.json`, `soup_asb_seed{0,1}.json`,
`curves/asb_seed{0,1}_curve.csv`, `sanity.json`, `diagnostics.json`,
`support_audit.json`, `support_examples_asb_seed{0,1}.json`,
`parameter_accounting.json`, `decision.json`, `repro_det{A,B}_seed0.json`,
`states/asb_seed{0,1}_selection_state.pt`, `soup_states/`.

Code: `structural_patch_encoder.py` (cell), `zinc_patch_path_pooling.py`
(wiring), `zinc_adaptive_structure_binding_cell.py` (runner/stages),
`tests/test_adaptive_structure_binding_cell.py` (14 tests).

## 11. Consequence / next boundary

- **STOP Z1.** Do not promote; do **not** proceed to Z2 (outer context →
  support revision) — the inner support is already degenerate.
- This STOP is a **utilization/collapse failure (Case E)**, not a clean
  refutation of adaptive structure. It does, however, reinforce the existing
  line of evidence (B-Full near rank-1, EOR inner-encoder collapse, ESB
  connectivity-free tie) that patch-local connectivity beyond aggregate
  primitives is not the task-relevant signal on ZINC Cell A.
- **Single minimal, pre-registerable repair candidate (NOT executed here).**
  If the hypothesis is to be tested at all, address the two validated defects
  directly in one new pre-registered cell: (i) give the soft gate a
  *continuous* multiplicative influence on messages / pooled binding states
  while keeping the hard membership for auditing, so the gate receives an
  edge-specific forward gradient and cannot collapse to a global constant;
  (ii) ensure the adaptive parameters are **not** annihilated by Adam weight
  decay (exclude them from `wd`, and/or initialise the residual output so its
  gradient is non-vanishing while the *forward* residual stays zero). A
  structural cost `λ·mean(selected_fraction)` would only prevent the root-only
  collapse; it does not create binding-conditioned structure and is not
  sufficient on its own. This repair is **not** authorized by this record.
