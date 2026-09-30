# Pre-registration — `e2e_dictenv_bond_anchored_triple_joint_tune_v1`

Round: **BondAnchoredTriple-JointTune-v1** (joint adaptation performance screen),
study `zinc-context-gap`, track `ksvd`, runner
`zinc_bond_anchored_triple_joint_tune_v1_mainline`.

This is a **single-candidate 40-epoch performance screen** that starts from the
committed BondAnchoredTriple-v1 (BAT-v1) M1 soup and **jointly fine-tunes the
representation and the composition** while the shared dictionary coordinates
stay fixed.  It does **not** retrain the M0/Sem108 parent, does **not** run any
shuffle / mechanism / ablation / branch-off arm, does **not** use a matched
control or a second seed, and never reads the official test split.  It answers
exactly one question and then stops at a report:

> With the shared dictionary coordinates (`D`, `U`, `common_rms`), the
> semantic interface and the C6 readout rule frozen, can jointly adapting the
> **active edge structure–semantics binding**, the **local environment
> fusion**, the **pair path** and the **triple composition plus Reader** push
> the complete predictor to `valid soup MAE <= 0.120`?

This is a **screen**, not a causal comparison: any difference is the overall
joint-adaptation difference against the BAT-v1 M1 soup start
(`Delta_vs_start = M_joint_soup − M_start`).  It is **not** the independent
effect of the triple operator, and it cannot establish that the three-
environment mechanism or the dictionary coordinates are irreplaceable.

Frozen **before** any structure build, correctness run or training.  Nothing
below is tuned after seeing results.  A changed pre-registration voids the
round.

---

## 0. Two frozen sources and the exact hot start

**Local feature extractor — original Sem108 soup**

* Path: `tracks/ksvd/results/e2e_dictenv_sem108_v1/checkpoints/SEM108-seed0_soup_state.pt`.
* Canonical state hash (`audit.state_sha256` over sorted keys, re-verified at
  load time):
  `7a721984c5579dd76f9bcaff13a2055b8efba71da553eefb3645d94207f5050a`.
* Full-file sha256 `6ec0fdef824d2c93973eaba1847c1173b9d939c75276476caf672d8010a74f89`
  (re-measured at runtime; the value is read from the artifact, never
  completed from a prefix).
* Source run / provenance: `tracks/ksvd/results/e2e_dictenv_sem108_v1/`;
  historical official-valid soup MAE `0.123704927947314` under the C6 mask.
  This is the **background reference** `M_parent`, not the start of this round.

**F, Reader and the fixed normalisation buffers — previous round M1 soup**

* Path: `tracks/ksvd/results/e2e_dictenv_bond_anchored_triple_v1/checkpoints/BAT-v1-seed0_soup_full_state.pt`.
* The state is the arithmetic soup of the BAT-v1 member epochs
  `[50, 57, 43, 49, 71]`; the member list and member valid MAEs are re-read
  from `tracks/ksvd/results/e2e_dictenv_bond_anchored_triple_v1/soup.json` and
  checked.  No single checkpoint (in particular not epoch 30) is substituted.
* Its F and Reader are the **warm start** for this round; its six buffers
  `mu_p, scale_p, mu_old, scale_old, mu_3, scale_3` are the **fixed
  normalisation statistics** and are loaded from this same state (cross-checked
  against `standardizers.json` of the same round).  They are never re-fitted,
  even after the dynamic representation changes.
* Full-file sha256 and canonical state hash are **measured at runtime** and
  recorded; the canonical hash is also cross-checked against the BAT-v1
  run payload if available.  Neither value is fabricated.
* Expected start performance (BAT-v1 soup on official valid, C6):
  `M_start = 0.12300291641423246`.  The online hot start must reproduce this
  value to `|M_start_online − 0.12300291641423246| <= 1e-6` on the full
  official-valid split before training is allowed to start.

**Loading rule.**  Build the Sem108 model with the exact BAT-v1 source chain
(`p2run.load_dictionary(cm.H1_CONFIG.dict_kind)` + `q1` subspace from
`results/e2e_dictenv_common_subspace_dictionary_v1/common_subspace.json` +
`sem.build_sem108_model(D, seed=0, q1)`), load the Sem108 soup with
`strict=True`, drop the old 302-D Reader from the module registry, attach the
new 366-D Reader and the joint `F`, then load
`BAT-v1-seed0_soup_full_state.pt` into exactly those keys and verify
value-for-value.  No re-initialisation of F or the Reader, no re-fitting of the
six buffers, no inherited optimizer state: this round builds a **fresh Adam**.

## 1. Fixed coordinates and frozen modules

Everything is first set `requires_grad_(False)`; only the Section-2 whitelist is
unfrozen.  The following stay bit-identical (checked before/after this round):

* the dictionary parameter `D`, the common basis `U`, `common_rms`;
* the semantic interface (`patch_cont[:, 0:108]` and the `size2` block),
  anchors and all original input preprocessing / standardisation;
* `W_A_S`, `W_A_C`, `node_encoder`;
* `global_encoder`, `topology_encoder`;
* CSSD-q1 coding / residual projection / column normalisation, `K=32`, `s=8`,
  IHT-10 and the encoding form `coord = [c~; alpha_res]`;
* the original C6 mask
  (`global_zero_groups=("atom_histogram","bond_histogram")`,
  `unary_zero_blocks=("count",)`, `pair_zero_blocks=("count",)`,
  `relation_zero_groups=("path_count",)`);
* the dead node binding `W_A_S / W_A_C` (never revived, reset or re-initialised);
* all six fixed normalisation buffers from the BAT-v1 M1 soup.

Freezing the "dictionary coordinates" means: for the same input, the structural
coordinate `[c~; alpha_res]` is always the same.  Task-relevant binding and
composition weights may move; the dictionary basis and the coding rule may not.

## 2. Joint fine-tune whitelist (exact)

1. active edge structure–semantics binding: `W_E_S`, `W_E_C`, `edge_encoder`;
2. local environment fusion: `fusion`;
3. pair encoding: `pair_projection`, `relation_encoder`, `distance_gate`,
   `pair_encoder`;
4. three-environment shared MLP `F`: `Linear(48, 64) → SiLU → Linear(64, 32)`;
5. the BAT-v1 extended Reader: `366 → 13 → 13 → 1` (ReLU, no dropout).

Every actual parameter is reported with name, shape and numel, classified as
frozen or trainable, and the optimizer parameter identity list must equal the
whitelist exactly, with **no duplicate parameters**.  Expected accounting
(measured on the built model, mismatch aborts):

* `W_E_S` 4,752 + `W_E_C` 192; `edge_encoder` 3,920; `fusion` 56,478;
  `pair_projection` 768; `relation_encoder` 1,104; `distance_gate` 80;
  `pair_encoder` 5,328; `F` 5,216; new Reader 4,967;
* **trainable 82,805** (34 tensors);
* backbone without the removed 4,135-parameter old Reader 93,574;
* **total registered parameters 103,757**; the removed old Reader is not
  registered anywhere and cannot enter the optimizer or the clip set.

## 3. Online forward — mandatory dynamic chain and autograd discipline

The training graph is computed online every step:

```text
fixed structural coordinates + original graph input       (no_grad, fixed D/U)
  -> current edge binding / current fusion                (trainable)
  -> current environment E                                (trainable)
  -> current pair_value p_ij                              (trainable)
  -> original C6 pooling -> current z_old (302-D)         (trainable)
  -> current three-environment composition -> z3 (64-D)   (trainable)
  -> fixed standardisation + current Reader               (trainable)
  -> graph-level L1
```

The three-environment operator is unchanged:

```python
# p_bar uses the BAT-v1 mu_p / scale_p
F = Linear(48,64) -> SiLU -> Linear(64,32)
t_ij_k = 0.5 * (
    F(cat([p_bar_ij, p_bar_ik, p_bar_jk]))
    + F(cat([p_bar_ij, p_bar_jk, p_bar_ik]))
)
z3 = cat([mean(t), mean(t*t)])            # 64-D
y_hat = Reader(cat([standardize(z_old), standardize(z3)]))
```

* one triple per unique undirected real key `(i,j)` (one endpoint order,
  `i < j`) and every third node `k != i, j`; object count `m*(n-2)`; no
  sampling, no truncation; zero 64-D summary for graphs without triples;
* the BAT-v1 tuple construction, endpoint symmetry, per-graph pair lookup and
  pair-count-offset batching are reused exactly;
* raw inputs, fixed coordinates, real-key lists and tuple/pair indices may be
  pre-computed **without gradients** because they do not depend on trainable
  weights;
* **no cached `E`, `p_ij` or full `z_old` from BAT-v1 may be used as training
  input** — they are reference-only for initialisation checks;
* the online feature-forward helper must return `pair_value` and `unified`
  (i.e. `z_old`) **with gradients**; cached helpers with `.detach()` /
  `no_grad()` are not usable inside training;
* no message passing, no node/pair state iteration, no attention / transformer,
  no new random layers (this round's F and Reader have no dropout).

**Mode discipline.**  The original feature extractor stays in `eval()` during
training so the original backend dropout (0.05 in `relation_encoder`,
`pair_encoder`, `global_encoder`) is **off**, exactly as during the BAT-v1 cache
extraction.  `eval()` does not disable autograd; unfrozen modules still
back-propagate.  Because `nn.Module.train()` recursively switches children, the
model class overrides `train()` to restore `backbone.eval()` immediately, and a
check confirms that after `model.train()` no `Dropout` module is in training
mode and that the "training-mode" forward is identical to the eval forward.
The six buffers stay fixed; no BatchNorm/LayerNorm/gate/dynamic calibration is
added.

## 4. Data, splits and the fixed training budget

* Official ZINC train 10,000 / valid 1,000; existing repository split, order and
  target handling (`p1run.load_split`, encoded + environment caches).  The
  official **test** split is never instantiated, loaded or evaluated
  (`official_test_loaded=false`; control plane `test_access=blocked`).
* Device: local CPU, 8 threads, float32.  No remote A100 work this round.
* Seed 0.  A new, independent data RNG is fixed and recorded: train order is
  the shuffled index stream from `torch.Generator().manual_seed(0 + 91011)`
  (the repository `TRAIN_SHUFFLE_OFFSET`); valid order is the fixed split
  order.  The RNG stream is recorded per run.
* Exactly **40 epochs**, batch 128.  No pre-training, no warm-up optimizer
  steps, no custom subset, no early stopping, no scheduler, no dropout, no
  architecture variant, no extra arm, no automatic epoch extension.
* Optimizer: **new Adam** over exactly the Section-2 whitelist, uniform
  `lr = 1e-4`, coupled `weight_decay = 1e-5`, fresh moments (old optimizer
  state never loaded).
* Objective: graph-level `L1(y_hat, y)`.  The dictionary and coordinates are
  fixed, so the reconstruction term is constant w.r.t. this round's trainable
  parameters; it is **not** added to the loss and the coding rule is not made
  trainable.
* Global gradient clip 5.0 over **exactly** the optimizer's unfrozen parameter
  list.  Frozen parameters never enter the clip set.
* Per epoch, record train MAE, valid MAE and epoch seconds; the first formal
  epoch estimates the remaining time.  No pilot optimization run.

## 5. Required pre-training checks (abort instead of starting on an anomaly)

1. Source provenance: parent checkpoint full sha + canonical hash; BAT-v1
   checkpoint full sha + canonical hash; member list/values; six buffers equal
   to `standardizers.json`; exact parameter accounting.
2. Online hot start on the full official valid split:
   `|M_start − 0.12300291641423246| <= 1e-6`.  If this fails, locate the
   initialisation / mask / mode / index / precision problem and fix it; the
   round does not start on an unexplained gap.
3. Same small batch of graphs through the BAT-v1 cached path and the online
   path: record max abs errors of `p_ij`, `z_old`, `z3` and the final
   prediction; the prediction error must be `<= 2e-5`.
4. `model.train()` (mandatory mode) forward vs `model.eval()` forward on the
   same batch: bit-identical / max abs diff 0; no dropout active.
5. One normal key-containing training batch, `backward()` **without**
   `optimizer.step()`: loss / output / gradients finite; non-zero task
   gradients on the edge-binding group (`W_E_S`, `W_E_C`, `edge_encoder`),
   `fusion`, `pair_projection`, `relation_encoder`, `distance_gate`,
   `pair_encoder`, `F` and the Reader (record the edge-binding group norms);
   frozen parameters `requires_grad=False` and `.grad is None`; optimizer and
   clip set exactly equal to the whitelist.  Gradients are then cleared and the
   hot-start weights are unchanged.
6. Structure audit: triple count `m*(n-2)`, anchors = unique real bonds, no
   out-of-range / cross-graph row, brute-force enumeration spot check, batch
   pair-offset mapping safe; counts equal the BAT-v1 cache counts
   (train 5,511,568 / valid 546,830 triples).
7. Frozen-subset tensor hash and `D / U / common_rms` hash recorded before and
   after the round; both source checkpoint files byte-identical before/after.

## 6. Soup and the single primary metric

* Save the hot-start (epoch 0) full state and the start valid MAE for
  provenance; it is **not** a soup member.
* Primary metric `M_joint_soup`: arithmetic-mean soup over the **5 best
  epochs among epochs 21–40 by valid MAE** (ties: earlier epoch first),
  averaging **all trainable parameters** (edge binding, fusion, pair path, F and
  Reader).  The soup is not restricted to F+Reader and never pairs a soup
  average with a single-point backbone.
* All frozen parameters and all six buffers are identical across members and
  are assembled unchanged (verified element-wise); no BAT-v1 or epoch-0
  checkpoint enters the soup.
* Also reported: full-run best trained valid MAE (epochs 1–40) and epoch,
  soup member epochs/valid MAEs, last-10-epochs valid mean, final train MAE,
  wall clock, seconds/epoch, peak RSS, parameter accounting, mode/dropout
  checks, frozen-subset and coordinate hashes, and both source hashes.

## 7. Continue gate and stop rule

* Record `M_start` (online hot start), `M_parent` (historical Sem108 replay
  `0.123704927947314`, background only), `M_joint_soup`, and
  `Delta_vs_start = M_joint_soup − M_start`.
* Complete and valid run with `M_joint_soup <= 0.120` →
  **`JOINT_TUNE_SCREEN_PROMISING`** (report only; a future full-budget /
  matched step needs a new pre-registration).
* Complete and valid run with `M_joint_soup > 0.120` →
  **`JOINT_TUNE_SCREEN_NO_STRONG_SIGNAL`** (this combination candidate is
  closed; no rescue).
* Hot-start reproduction failure, correctness failure, numerical failure or an
  incomplete 40-epoch run → `invalid`/`incomplete`; **no performance verdict**
  and no shortened-horizon substitute metric.
* The 0.120 threshold is never relaxed after seeing results, including for a
  `0.000x` miss.  A best-single-checkpoint pass with a failing soup does not
  change the primary-metric verdict.

## 8. Explicitly out of scope

* No M0 retraining, no matched-capacity control, no shuffle, no ablation, no
  branch-off, no mechanism arm.
* No optimizer/denormal diagnostic, no dead-node-binding rescue, no change to
  the three-environment architecture, no width/depth/parameter additions.
* No learning-rate search, no scheduler, no early stop, no seed 1, no extra
  run arm, no automatic continuation, no 320-epoch extension.
* No official test access of any kind.
* The result describes this round's overall joint adaptation; it is not the
  triple operator's isolated contribution and cannot be used to claim that the
  three-environment mechanism or the dictionary coordinates are irreplaceable.

## 9. Reporting

Implementation commit, result commit, run ID, device/threads, full source
checkpoint hashes, resolved paths, frozen/trainable accounting, all checks of
Section 5, the metric table of Section 6, the verdict of Section 7, and an
explicit statement of what was and was not run.
