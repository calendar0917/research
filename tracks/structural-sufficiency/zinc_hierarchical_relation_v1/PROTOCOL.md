# PROTOCOL — zinc_hierarchical_relation_v1

Protocol id: `ssrc-zinc-hierarchical-relation-v1`
Track: `structural-sufficiency`
Status: **FROZEN before the first formal GPU training run.** This file, the
config, the model/data code and the CPU gate outputs are committed together;
the commit hash and file hashes are recorded in `analysis/frozen_manifest.json`
and in every `train.json`. Any change to the model, data contract, recipe,
gate, budget or stop rule invalidates this protocol and requires a new id.

Official ZINC test is **never loaded, predicted or scored** by this protocol.

## 1. Proposition under test

Two separable propositions, evaluated on the official ZINC-12k **valid** split:

- **P-A (hierarchy).** A non-recursive hierarchy of atom, bond, junction and
  rooted induced-patch objects, with one shared coarse-to-fine refinement block,
  train-only scalar support modulation of the fine patch contribution, and a
  vector/second-moment readout, reaches best-valid raw-y MAE **<= 0.073**,
  beats the frozen `P3B_s0` reference after removing the fixed P3B top-10 error
  rows by **>= 0.002**, keeps eval-mode train MAE at the best epoch **<= 0.026**,
  and does not get the gain from one molecule.
- **P-B (one relational interaction).** Adding exactly one shared low-rank
  (r = 16) pair interaction over graph-related junction/patch tokens, with
  real relation ids and no zero-initialised gate, improves the same hierarchy
  under the same initialisation, data order and recipe. B is compared to A as
  a **mechanism contrast**, not as a second chance at the gate; B is eligible
  for promotion only if it also passes the same frozen P-A gate.

Null control C (relation-to-pair permutation within molecule) is implemented
and CPU-tested but is **not** a formal arm of this round. It is run only if B
passes and a new protocol buys the second seed/control.

## 2. Frozen data contract

- `PyG ZINC(root=..., subset=True)`, official train 10,000 / valid 1,000.
  Official test is not instantiated anywhere in the runner.
- Inputs per molecule: raw atom type (28 one-hot), raw bond type (4 one-hot),
  graph structure, the frozen BJOC label-blind atom/bond/junction features,
  rooted induced patches (radius 2), 33 label-blind global topology features,
  graph relations, and a 4-d **train-only** support scalar. No SMILES, logP,
  SA, target cycle penalty, `y`-derived quantity, split-local id, row id or
  molecule id is an input.
- Support key: radius-2 nested-WL bond-centred key counted over the 10k train
  molecules only. Stored as (log-count z-score, quantile over unique train
  keys, unseen flag, clipped count). No per-key embedding or lookup table.
- Standardisation of patch scalars, global features and support log-counts is
  fit on train only and applied unchanged to valid.
- The frozen cache builder is the code in `code/hr_data.py` at the protocol
  commit; `analysis/prep.json` records cache SHA-256 values.
- The SSL initialisation is
  `tracks/structure-contribution/results/bjoc_full10k_pretraining_transfer_v1/pretrain_encoder_s0.pt`
  with SHA-256 `006fabc11efede091420fc2126ef2ef1c46cb3948ab91648a96843c0352bfa97`.
  Only its atom/bond/junction encoders are loaded; they are fine-tuned. No
  supervised P3 reader or P3 checkpoint is loaded.

## 3. Frozen models

Both arms are built by `code/hr_models.py::build_model` with seed 0. They share
the atom/bond/junction encoders, the single `RefineBlock` reused twice at each
level, role embeddings, patch node/edge encoders, support trunk/heads, patch
MLP and reader, and therefore have **identical shared initial values** (the CPU
gate checks the shared-state hash).

- **A — HCE:** atom -> bond -> junction -> rooted induced patch; support gate
  `sigmoid(W_s support)` multiplies the patch node-sum and edge-sum before the
  patch MLP; shared refine twice; readout = per-level sum/mean/second moment +
  count, three local cross moments (`a_u*a_v`, `b_1*b_2`, `b_root*p`), 33
  global features -> 1477 -> 96 -> 96 -> 1.
- **B — SRI:** A plus one interaction over junction+patch tokens. Pair set =
  clipped anchor distance <= 3 or shared ring block (both directions plus
  self). Relation id = `1 + d_bucket*8 + same_block*4 + A^2_bucket`, plus self
  id 0. Shared low-rank q/k/v (r = 16), relation embedding/bias, segment
  softmax per source, scatter-add, `LayerNorm(h + W_o o)`. No zero-init gate,
  no per-pair MLP, no intermediate-atom contraction, no recursion.
- Parameter budget: **<= 500,000**. Frozen counts: A 438,585; B 445,578.

## 4. Frozen recipe and selection

AdamW (`lr=1e-3`, `min_lr=1e-5`, `weight_decay=1e-5`, betas 0.9/0.999,
eps 1e-8), linear warmup 500 steps then per-step cosine, grad clip 5.0,
batch 128, 600 epochs, 79 steps/epoch = 47,400 updates, raw-y L1. Shuffle seed
101; model seed 0. Valid every epoch at batch 256. Selection:
`argmin(valid_mae)` with strict `<`; a tie keeps the earlier epoch. Single
best checkpoint. No grid search, no SWA, no early-stopping patience, no test
feedback.

## 5. Pre-registered decision rule (unchanged from DESIGN)

For each arm independently:

1. best-valid MAE `<= 0.073`;
2. `gain_drop_top10 = mean(AE_P3B over all rows) - mean(AE_candidate over the
   fixed 10 rows with largest P3B_s0 valid AE) >= 0.002` (the fixed rows are
   `[917, 753, 172, 659, 260, 678, 31, 224, 447, 229]`; they are removed from
   both arms before computing the comparison MAE, so this is the
   "improvement survives removal of the worst reference molecules" test);
3. eval-mode train MAE at the selected best epoch `<= 0.026`.

Promotion also requires that the result is not driven by one molecule; the
drop-top-10 rule is the operational version, and the per-molecule table is
reported so a single-row explanation can be checked.

If A and B both fail: close the round, report the failure mode (fit,
generalisation, relation use, or missing information), do not run seed 1, C,
or any hyper-parameter/feature rescue. If exactly one passes: it is eligible
for a second seed and a parameter-matched mechanism control under a **new**
protocol; this protocol does not buy those runs. If both pass: B is eligible
for the same follow-up, and A remains the clean hierarchy reference.

## 6. Budget and stop rules

- Hard budget: **2 A100 GPU-hours per formal run**, **4 A100 GPU-hours total**
  for this round. Smoke, profiling, failed attempts and retries are charged to
  the same budget.
- A formal run is stopped immediately if: loss/valid MAE becomes NaN/Inf;
  valid MAE exceeds 0.50 after epoch 20; the 2-hour cap is reached; or the
  process dies from an infrastructure fault. An infrastructure retry must use
  the same commit, config and seed; it is charged to the budget. A scientific
  failure (bad fit, bad valid) is reported, not retried with new settings.
- Wall time and peak CUDA memory are recorded per run. No run may use a
  different batch size, LR, epoch count or selection rule to fit the budget.

## 7. CPU gates and audits (must pass before launch)

`analysis/cpu_gates.json` (all groups `pass: true`): raw-object fidelity,
edge-direction and atom-renumbering invariance, batch-vs-single, empty /
single-atom / ring / disconnected cases, train/valid support isolation,
shared A/B initialisation, non-zero gradient for every parameter tensor,
A response to support/patch-edge/junction perturbation, B response to
real-vs-null relation assignment, no hidden-id embeddings (role 3, relation 33
only), CPU profiling, deterministic batch order, parameter budget.

`analysis/data_audit.json`: exact raw-object fidelity; 3.86 % of 4,969 patches
from 200 valid molecules have a nested-WL key with more than one induced
signature (C6 vs 2 x C3 synthetic witness: same 2-round WL colours, different
induced signatures); valid support unseen fraction 13.8 %; mean 59.3 tokens
and 903 related pairs per molecule. `analysis/overfit_cpu.json`: 48 molecules,
400 steps, final eval-mode train MAE 0.132 (A) / 0.230 (B) vs constant
baseline 1.005.

## 8. Frozen references

- `P3B_s0` valid predictions:
  `tracks/structural-sufficiency/zinc_recipe_trunk_v2/runs/xBssl_P3N_s0/predictions/valid_best.npz`,
  SHA-256 `8dc5ba2005f9a76bbb995f75dba6760ab471fc0eb57155797bad6d3087aca33a`,
  raw-y valid MAE `0.07622375367576023`, best epoch 377, eval-mode train MAE
  `0.014679793968901504`.
- Audit groups: `tracks/structural-sufficiency/zinc_generalization_audit_v1/analysis/per_molecule.csv`
  (row ids 0..999, `y_raw`, `pred_P3B_s0`, `abs_error_P3B_s0`,
  `r2_unseen_atom_frac`, `n_rings`). The P3B top-10 fixed rows are computed
  from this table once and written into the score JSON.
- All historical results and checkpoints are preserved; this protocol never
  overwrites an older run directory.

## 9. Execution and artifacts

- Launch only from a clean tree at the protocol commit: `rr` deploy/run/pull.
- Arm A on `res-gpu0`, arm B on `res-gpu1`, same commit, separate run dirs
  `runs/HR_A_s0` and `runs/HR_B_s0`.
- Each run writes `curve.csv`, `train.json`, `best.pt`, `final.pt`,
  `predictions/valid_best.npz`; `score` writes `analysis/score_*.json` and
  `analysis/per_molecule_*.csv`; `compare` writes `analysis/comparison.json`.
- Any deviation (including a retry) is appended to `analysis/deviations.md`
  with time, reason, and budget charged. No silent edits.
