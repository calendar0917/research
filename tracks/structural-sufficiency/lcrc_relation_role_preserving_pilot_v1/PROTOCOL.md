# PROTOCOL — LCRC relation-role-preserving pilot (v1)

Round id: `lcrc-relation-role-preserving-pilot-v1`. Status: architecture
**feasibility pilot**; every number is DEVELOPMENT, and every Test column is
**DEVELOPMENT_TEST** (evaluated every epoch, never used for selection).

## 1. Question

1. If the frozen C1 architecture and the **complete** C1 readout are kept, does
   splitting the 200-d S2 record into two role groups and rescaling them with two
   trainable per-channel gammas keep the C1 prediction (≤ 1.10× Valid RMSE of
   C1(Rf) on Lipophilicity)?
2. Is a real topology-block role boundary (internal vs cross-boundary) a better
   inductive bias for those two gates than an arbitrary but relabelling-invariant
   partition of exactly the same 200-d records (`RR_CONTROL`)?
3. On WC_LOGP, does the REAL model's native block-mask response localise the
   oracle atom contributions better than the CONTROL model's?
4. Is any improvement attributable to prediction, to external structural
   explanation, or to neither?

The previous round (`lcrc_relation_interface_routing_pilot_v1`) replaced the C1
readout with block/interface tokens and failed the retention gate. This round
therefore **does not touch the readout at all**: only the pooling of the S2
columns of the existing local input is split.

## 2. Frozen reading of the LCRC encoding (consumed, never re-defined)

* Atom code `a_v` and bond code `b_e` — the frozen C1 atom/bond encoders
  (`cr_models.C1Trunk` module names/shapes, canonical state, seed 0).
* S1 = the 64-d neighbour code `nbr_code` of `molecular_graph.enumerate_s1`.
* S2 = the typed path record of `molecular_graph.enumerate_s2`, unchanged:
  `path_cat` (400 codes), `path_end63` (80 codes), `path_u`/`path_v`,
  `joint_proj(path_u*path_v)`, `path_closure`, `path_gate`, and the content
  `a[path_end] * path_gate(path_cat)`. The 200-d record
  `[path_emb, end_emb, joint, close, content]` is **atomic** in this round: it is
  never decomposed or re-encoded, only assigned to one of two groups.
* The local operator is the frozen `dissect_model.LocalContent(..., "C1", ...)`
  with one shared dropout, and the local input is the frozen 433-d layout
  `[a_v, Σ b_e, nbr, S2 (200), 9 atom scalars]`.
* The readout is the C1 readout, unchanged:
  `sum_v f(h_v) + rho([g, mean_v h_v, mean_v h^2_v, log1p n_v])`, with `rho` a
  171→64→1 MLP. Nothing about it is replaced, gated or extended.

New parameters: exactly two 200-d vectors (`gamma_internal`, `gamma_cross`),
initialised to 1. Total 229074 vs C1 228674 (+400, +0.17 %).

## 3. Frozen role rules

Both rules assign **every** S2 occurrence to exactly one group; the group of an
occurrence never depends on a label, an oracle value, a row id or an atom index.

* `RR_REAL`: `owner(e1) == owner(e2)` → group 0 (internal), else group 1 (cross),
  where `owner` is the frozen topology-block ownership of the previous round's
  routing cache. The rule is a function of the bond graph only.
* `RR_CONTROL`: `blake2b(CONTROL_SALT | path_cat | path_end63 | path_closure,
  digest_size=8)[0] & 1` → group 0/1, i.e. one class per observed typed-S2
  record. `CONTROL_SALT = "lcrc-relation-role-preserving-pilot-v1/s2-type-hash/control"`.
  The table over the observed types is frozen in
  `cache/s2_control_type_hash_v1.{npz,json}` (with `labels_used=false`,
  `oracle_used=false`, `atom_index_used=false`, `row_id_used=false`) and is
  re-derived from the hash at every load.

Aggregation: `Q = gamma_internal ⊙ Q_internal + gamma_cross ⊙ Q_cross`, where
`Q_internal/Q_cross` are the per-atom sums of the 200-d records of the two
groups. Because the record is atomic and each occurrence has exactly one role,
`Q_internal + Q_cross` is the frozen C1 S2 column block, and with both gammas at
1 the model **is** the frozen C1 model.

The two models are trained from the **same** canonical C1 initial state
(`lcrc_readout_context_pilot_small_v1`, seed 0), with the same recipe, the same
batch order and the same parameter count. No historical weight is loaded.

## 4. Pre-training and post-training gates

`code/rp_checks.py` (11 gates; the `checks` stage runs all of them and writes
`analysis/routing_invariants.json`):

1. **frozen reuse** — the previous round's routing cache files match their frozen
   SHA-256 *and* their arrays are reproduced by a full rebuild from the frozen
   partition;
2. **role partition** — REAL role == `path_cross` == `(owner1 != owner2)`; CONTROL
   role == the frozen hash table; every occurrence is in exactly one group and
   both groups are non-empty on every split;
3. **S2 conservation** — per atom and per split,
   `Q_internal + Q_cross == the frozen C1 S2 aggregation` (typed 104-d and
   content 96-d), within a float32 trace-scaled tolerance;
4. **gamma = 1 equivalence** — on the canonical C1 state, local input columns,
   `h`, `f(h)`, `mean h`, `mean h²`, `rho` and the prediction equal
   `cr_models.ArmA` (the C1 model) exactly;
5. **data/readout identity** — the 433-d layout is unchanged, the non-S2 columns
   are bitwise identical to the C1 collate, the Train-only normalisation buffers
   and the target scaler are recomputed and match, and `f`/`rho` hold the
   canonical weights;
6. **relabelling invariance** — after an atom permutation and a bond
   permutation, the block partition maps bijectively, the role counts are
   unchanged, the per-atom role aggregation and the prediction are unchanged;
7. **parameter identity** — identical total for both models, `C1 part ==
   228674`, `delta == 2 × 200`, identical initial state hash, gammas all ones;
8. **label / oracle blindness** — the role vectors and the batch features are
   unchanged when the targets are permuted; neither role rule reads a label;
9. **trained-state identity** (post-training) — for all four frozen checkpoints:
   the readout closure `pred == Σ f + rho` holds, the saved predictions equal a
   fresh forward pass, and the model still relabels invariantly;
10. **WC_LOGP closure** (post-training) — `Σ_v c_v_oracle == WC_LOGP` for every
    row of every split;
11. **group sizes** — the reported occurrence counts per split and model add up
    to the frozen S2 totals.

## 5. Training

Frozen recipe `Rf` (identical to the C1 baseline):

* seed 0, 360 epochs, batch 128 (last partial batch kept), eval batch 256;
* AdamW, betas `(0.9, 0.999)`, eps `1e-8`, peak LR `1e-3`, weight decay `1e-5`;
* 5 warmup epochs at `peak·(0.1 + 0.9·step/warmup_steps)`, then a per-step cosine
  from peak to `1e-5` over the remaining steps (9720 optimizer steps);
* MSE on the standardised target, global gradient-norm clip 5, FP32,
  no AMP / EMA / SWA / soup / augmentation / resampling.

Every epoch: full `model.eval()` + `no_grad` pass over Train / Valid / Test in
raw target units. Selection: minimum Valid RMSE (strictly smaller replaces, exact
tie keeps the earlier epoch; epoch 0 is never selectable). The Test column of
that same frozen state is reported as DEVELOPMENT_TEST.

Four units, one seed, no early stopping, no re-selection:

| unit | task | model |
|---|---|---|
| 1 | lipo | RR_REAL |
| 2 | wc_logp | RR_REAL |
| 3 | lipo | RR_CONTROL |
| 4 | wc_logp | RR_CONTROL |

## 6. Pre-declared analysis and decision scales

* **retention** — `RR_REAL` Lipophilicity Valid RMSE ≤ 1.10 × C1(Rf) Valid RMSE
  (0.706884841). The CONTROL number is reported next to it but is not part of
  this criterion.
* **prediction increment** — paired molecule-level bootstrap of
  `RMSE(REAL) − RMSE(CONTROL)` (2000 draws, seed 20261117): a signal requires the
  95 % CI to exclude 0 in favour of REAL **and** `P(REAL better) ≥ 0.95`, on Valid
  or Test of Lipophilicity.
* **oracle increment** — the same paired bootstrap on per-molecule oracle
  enrichment (capture of |WC_LOGP| atom mass inside the candidate atom set minus
  the mean capture of matched random connected subgraphs). A signal additionally
  requires REAL's own enrichment to be positive on ≥ 60 % of matched molecules
  with a positive mean.
* **gamma usage** — each frozen model is re-run with both gammas set to 1 (the
  exact C1 aggregation). A model that never moves away from 1 has no role
  mechanism to credit, whatever its RMSE.
* **closure** — every Val/Test prediction is reconstructed from `Σ_v f(h_v)` and
  `rho(·)` within `1e-5 + 1e-6·max(1,|x|)` raw units.

Verdict (exactly one):

* `PREDICTION_PRESERVED` — REAL passes retention;
* `ROLE_INCREMENT_SHOWN` — retention **and** (prediction increment **or** oracle
  increment);
* `ONLY_C1_READOUT_RECOVERY` — both models pass retention and no increment;
* `ROLE_NOT_SUPPORTED` — REAL fails retention and no increment;
* `UNRESOLVED` — anything else (e.g. REAL passes, CONTROL fails, no increment, or
  an incomplete round).

The round stops after the report. No new seed, no hyper-parameter search, no
partition extension and no new architecture are launched from here.

## 7. Comparability

Only runs under this protocol, the same dataset fingerprint and the same split
fingerprint are comparable. The C1(Rf) Lipophilicity baseline is
`DIRECTLY_COMPARABLE` (same split, target, scaler, seed, recipe and canonical
initial state). The C1 WC_LOGP baseline is `RECIPE_DIFFERS` (R3) and is never
used as a threshold. The previous round's R1/R2 numbers are
`NOT_DIRECTLY_COMPARABLE` and stay in an appendix.
