# DESIGN — ECLC-v1 implementation freeze

This file freezes the implementation-level computation graph, inputs,
initialisation order and known limitations. Gates and thresholds are in
`PROTOCOL.md`. Numbers quoted here are from the frozen LCRC-v1 config/cache and
from the ECLC Phase 0 run; they are not GPU results.

---

## 1. Reused frozen assets (read-only)

| asset | path | role |
|---|---|---|
| LCRC code | `tracks/structural-sufficiency/zinc_local_relation_composition_v1/code/lc_{common,data,models,train}.py` | backbone, caches, recipe plumbing |
| LCRC train cache | `.../zinc_local_relation_composition_v1/cache/lc_train.pt` (git-ignored) | official train-10k P3 atom108 / S1 / S2 / global33 / labels |
| LCRC valid cache | `.../cache/lc_valid.pt` (git-ignored) | official valid-1000 |
| LCRC norm | `.../analysis/norm.json` | train-only scalar/global standardisation |
| SSL s0 / s1 | `tracks/structure-contribution/results/bjoc_full10k_pretraining_transfer_v1/pretrain_encoder_s{0,1}.pt` | atom/bond encoder init |
| valid anchor | `zinc_performance_search_v1/analysis/per_molecule_valid.csv` | frozen valid slices |
| fixed worst-10 | `zinc_generalization_audit_v1/analysis/fixed_worst10.json` | frozen worst-10 rows |

The LCRC caches are label-free functions of the official PyG raw graph; their
`labels` field is the official train/valid label. The ECLC matching script does
**not** read the cache and never reads labels; it reconstructs structures from
the PyG train split. The supervised trainer reads the cache labels for train
and valid only.

---

## 2. Base-head input

For each atom `v`, from the cache fields:

```
center_v   = onehot(atom_type_v, 21)                        21
r1_v       = Σ_{e incident to v} onehot(nbr_code_e, 63)     63
r2p_v      = Σ_{2-path v->w->u} onehot(path_cat, 189)      189
r2e_v      = Σ_{2-path} onehot(path_end63, 63)              63
r2c_v      = Σ_{2-path} 1[path_closure]                      1
base_v     = [center, log1p(r1), log1p(r2p), log1p(r2e), log1p(r2c)]  337
```

- `nbr_code = type(neighbour) * 3 + (bond_type - 1)` (63).
- `path_cat = type(middle) * 9 + (bt1-1)*3 + (bt2-1)` (189).
- `path_end63 = type(end) * 3 + (bt2-1)` (63).
- `path_closure` is 1 iff the 2-path end is adjacent to the root.
- The cache arrays are built once from the raw graph by the frozen LCRC
  `build_from_arrays`; the Phase 2 independent reference recomputes them with a
  naive loop and requires exact agreement.

**REAL** input: `[center, log1p(r1)]` (84 dims); the last 253 dims are set to
exactly zero. **NULL** input: all 337 dims. The input dim is the same, the
module shapes are the same, and the parameters are the same. REAL's radius-2
input columns receive zero input and therefore exactly zero gradient.

Why not zero the whole radius-1 block for REAL? The hypothesis is specifically
that the *base contribution* should be forced to depend only on the original
atom type and rooted radius-1 typed adjacency. The center type + typed
radius-1 adjacency is the minimal sufficient statistic for that statement.

---

## 3. Backbone and readout

The backbone is exactly `lc_models.LCRC("A", norm, cfg)`:

```
a_v = AtomMLP(atom_feat_v)                                   108 -> 128 -> 96
b_e = BondMLP([a_u+a_v, |a_u-a_v|, a_u*a_v, bond_feat_e])    295 -> 160 -> 96
S1_v = Σ_{e∋v} b_e,  S1mean_v = S1_v/deg(v)
S1joint_v = Σ_{e=(v,u)} Emb_nbr[type(u), bt]
S2_v = Σ paths [Emb_path(cat189), Emb_end(cat63),
                W_j(U(cat189) ⊙ V(cat63)), Emb_close(closure),
                gate(cat189) * a_u]
h_v = LN(LocalMLP([a_v, S1_v, S1mean_v, S1joint_v, S2_v, scalars_v]))
      LocalMLP: 529 -> 192 -> 64
f_context: 64 -> 64 -> 1
ρ: (33 + 2*64 + 1) -> 64 -> 1
```

No relation head, no junction module, no additional message passing, no
attention. The atom/bond encoders are loaded from the frozen mask-SSL
checkpoint for the seed (s0 for seed 0, s1 for seed ≥1) and are fine-tuned
normally. No supervised DCC-L/LCRC checkpoint is loaded.

New head:

```
f_base: 337 -> 32 (SiLU) -> 1
last Linear weight = 0, bias = 0
```

Parameters:

| module | params |
|---|---:|
| atom_encoder | 26,336 |
| bond_encoder | 62,816 |
| local | 133,645 |
| f_context | 4,225 |
| rho | 10,497 |
| f_base | 10,849 |
| **total** | **248,368** |

Both arms: 248,368. LCRC-A reference: 237,519. The 500k cap is not binding.

---

## 4. Initialisation order

`build_model(arm, seed)`:

1. `torch.manual_seed(seed)` → build the frozen `lc_models.LCRC("A", ...)` and
   load the SSL atom/bond encoders with `lc_models.build_model("A", ...)`.
2. Copy that frozen LCRC-A backbone state into the ECLC wrapper. This makes
   the shared tensors bit-identical to the LCRC-A reference at the same seed,
   not merely distributionally equal.
3. `torch.manual_seed(seed + 90731)` → build `f_base`. The base head is built
   **after** the backbone and consumes its own RNG; it never affects the
   backbone initialisation.
4. For REAL and NULL, the arm string is only a forward-time mask flag. Both
   arms therefore have identical full initial state hashes (including
   `f_base`), and identical initial predictions (`f_base ≡ 0`).

Verified in Phase 2: same full-state SHA for REAL/NULL; per-tensor equality;
step-0 full output equals the frozen LCRC-A output with max abs diff 0.

---

## 5. Loss and pair sampling

Per update:

```
rows      = next 128 rows of the frozen epoch permutation
pair_ids  = torch.randint(0, 350, (8,), generator=Generator(seed+202))
pair_rows = [i0, j0, i1, j1, ..., i7, j7]
pred_mol  = model(collate(train, rows))
pred_pair = model(collate(train, pair_rows)).view(8, 2)
loss      = mean|pred_mol - y[rows]|
          + 0.2 * mean| (pred_pair[:,0] - pred_pair[:,1])
                    - (y[i] - y[j]) |
```

- The molecule batch is unchanged at 128. The pair batch adds 16 molecules and
  is a separate forward; no padding, no cross-batch LayerNorm interaction
  (LayerNorm is per atom).
- The pair stream is identical for REAL and NULL because it depends only on
  `seed`, not on arm.
- The pair loss uses only `y_i, y_j`. It is not a new label, not an external
  teacher, and not independent supervision. The algebra is checked in Phase 2:
  the prediction difference is exactly the difference of the three additive
  terms; a common prediction offset leaves the pair loss unchanged; the pair
  gradient is not proportional to the molecule-L1 gradient. For squared loss
  the pair term is a quadratic form in the same residuals.
- The pair loss is a relational reweighting of the same residuals. Its only
  role in this experiment is to provide a *common* training signal for both
  arms, so a REAL-vs-NULL difference cannot be attributed to different pair
  data or a different pair objective.

---

## 6. Recipe and checkpoint selection

Exactly the frozen LCRC/ZPR recipe: AdamW, LR 1e-3 → 1e-5 with 500-step
warm-up and per-step cosine, wd 1e-5, batch 128, 600 epochs = 79 steps/epoch =
47,400 updates, raw-y L1, clip 5, dropout 0. No EMA, soup, seed averaging or
early stopping. Full valid-1000 every epoch; strict minimum valid MAE single
checkpoint (ties → earlier epoch). `final` and W20/T100 are reported but never
used for selection.

---

## 7. Frozen valid slices

From the pre-existing valid anchor and fixed worst-10 file (no new selection):

| slice | n |
|---|---:|
| full | 1000 |
| ordinary (`y > -3`) | 935 |
| radius-2-unseen ordinary | 225 |
| seen ordinary | 710 |
| minus row172 | 999 |
| minus fixed worst-10 | 990 |
| tail (`y ≤ -3`) | 65 |

The seen/unseen gap is `unseen ordinary MAE − seen ordinary MAE`. All slice
definitions are frozen before GPU and are not changed after seeing results.

---

## 8. Known limitations written before GPU

1. **The pairs are one-pendant-region edits, not tiny functional-group swaps.**
   Median edit region 13 atoms, median body 16 atoms. The body is strictly
   larger, but often only slightly. This is the strictest matching that still
   yields ≥300 pairs on the official 10k subset.
2. **The transferable Phase 0 signal is composition-additive, not edit-class
   identity.** The combined model with a true radius-1 edit-class one-hot is
   worse than 81/100 class-permutation controls. The pilot is licensed by the
   low-order signal (60.8% held-out MAE reduction), not by a categorical edit
   effect.
3. **The pair graph is fragmented.** 285 components, largest 7 molecules; the
   pair loss covers 621/10,000 molecules (6.2%), max degree 5. The pair loss is
   a light reweighting, not a second training set. It cannot by itself teach the
   model all local environments.
4. **Radius-2 content in NULL is a 337-dim count representation, not the full
   P3 atom108.** NULL can distinguish radius-2 contexts through the path/end/
   closure counts. It does not receive the atom108 radius-2 marginals, ring
   membership or global33 through `f_base`; those remain available to
   `f_context` in both arms. The contrast is therefore specifically about the
   *base head's* input restriction, not about total model information.
5. **REAL has 253 dead input columns in `f_base`.** Parameter counts are equal,
   but REAL's effective first-layer capacity is smaller. This biases the
   comparison *toward NULL*; a REAL win is therefore a conservative result, but
   the reverse (NULL win) cannot be interpreted as "radius-1 is useless"
   without acknowledging the capacity asymmetry.
6. **The 600-epoch recipe has a known execution-drift band of ~0.0026** (ZPR-v1
   divergence audit, same trainer, CUDA scatter order). The seed-0 promotion
   thresholds are set accordingly; a 1e-3 tolerance is allowed on the
   "not worse" slices. This is a pre-registered drift allowance, not a
   post-hoc rescue.
7. **Valid is an exposed development split.** All valid numbers are
   `VALID_AS_DEV_DESCRIPTIVE`; checkpoint selection uses valid. The official
   test is never read.
8. **No claim of a new relative-learning paradigm.** Matched-pair / SQRL-style
   relative objectives are prior art. The only claim under test is the
   structural restriction on `f_base` against the parameter-identical NULL.

---

## 9. File map

| file | purpose |
|---|---|
| `code/eclc_common.py` | paths, frozen imports, constants |
| `code/eclc_match.py` | label-free strict pair construction |
| `code/eclc_phase0.py` | support and differential-signal gates |
| `code/eclc_models.py` | ECLC wrapper, `f_base`, base features |
| `code/eclc_train.py` | frozen trainer + pair loss |
| `code/eclc_cpu_checks.py` | Phase 2 gates |
| `code/eclc_analysis.py` | valid tables, slices, mechanism probes |
| `code/eclc_smoke.py` | 2-epoch timing/path smoke |
| `configs/eclc_v1.json` | frozen config |
| `analysis/edit_pairs.json` | frozen pair list |
| `analysis/phase0_gate.json` | Phase 0 gate record |
| `analysis/cpu_checks.json` | Phase 2 gate record |
| `runs/` | execution truth (git-ignored) |
| `REPORT.md`, `DECISION.md` | final five-question report and decision |
