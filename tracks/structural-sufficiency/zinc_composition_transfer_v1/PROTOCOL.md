# PROTOCOL — zinc-composition-transfer-v1 (pre-registration)

Written before the first formal GPU run. Round name: `zinc-composition-transfer-v1`.
Local source of truth: `calendar0917/research@01077d5` + this round's commits.
Host: `res` (a100-2, 2 × A100-SXM4-40GB), one process per GPU.

## 0. Question and purchase reason

Final performance target (unchanged): official ZINC subset raw-`y` **test** MAE in
`[0.050, 0.060)`. `valid` 0.075 is a past exploratory gate, not the target and not
a reason to skip a mechanism control.

This round tests one unestablished hypothesis: the explicit-structure model fits
the training set extremely well, but ordinary per-batch empirical risk
minimisation does not constrain whether updates learned on one group of local
structural combinations transfer to another group. Holding model and inference
fixed, does a structure-driven cross-group update constraint improve
generalisation? A second, independent question: does changing the *supervised
label support* of local environments change prediction error?

Directional constraints kept: single model, ≤ 500,000 trainable parameters,
explicit structure statistics, no recursive hidden-state update along the local
adjacency. No new attention/new trunk this round. P3 is only the test platform
for the training mechanism.

## 1. Platform, data, recipe (fixed)

| item | value |
|---|---|
| model | `zinc_recipe_trunk_v2/ad2_models.P3NullModel`, 413,505 parameters (checked this round) |
| init | the **seed-0** train-only masked-SSL encoder `pretrain_encoder_s0.pt` (sha256 `006fabc11e…`), loaded for every arm and **both** supervised seeds; composer/reader init and data order carry the supervised seed |
| data | official PyG ZINC subset `train` 10,000 / `valid` 1,000; raw-`y` L1 |
| batches | 128 molecules, last partial batch kept, 79 updates/epoch, 600 epochs = 47,400 outer updates |
| optimiser | AdamW lr 1e-3 → 1e-5, weight decay 1e-5, betas (0.9, 0.999), eps 1e-8 |
| schedule | `ad_common.cosine_with_warmup_lr(step, total, 500, 1e-3, 1e-5)` applied to update `step` (the v1 trainer's declared index); 500-step linear warm-up then per-step cosine |
| gradient | FP32, outer gradient clip 5, dropout 0 |
| selection | best **valid** MAE, strict `<`, ties → earlier epoch; full 1,000-molecule valid every epoch |
| test | not loaded during development; at most one terminal read (§8) |

Seed-matched SSL was **not** used: the repository precedent for
fixed-encoder multi-seed comparisons
(`zinc_shared_factor_v1`, `zinc_phase2a`) fixes the s0 SSL encoder, and the
user instruction for this round explicitly requires one shared seed-0 SSL
checkpoint. Consequence: seed-to-seed replication here covers supervised
initialisation/data order, **not** SSL-pretraining variance.

Shared 50-epoch ERM prefix per seed: the `E` job writes `fork.pt` at the end of
epoch 50 (model + AdamW state + generator state + global step) and continues to
600 under the *full* 600-epoch schedule. `R`/`S` resume from that exact fork.
Every arm of a seed therefore starts from the same prefix state, and the ERM
prefix is billed once per seed.

## 2. Arms

| arm | objective |
|---|---|
| `E` | ordinary ERM: whole 128-molecule batch, raw-`y` L1 |
| `R` | balanced **random** source/query halves; one-step look-ahead objective |
| `S` | identical to `R` except the source/query partition is the structural greedy split |

Exact objective implemented (`zct_train.meta_grads`):

```
L_A(theta) = mean raw-y L1 over the source half
g_A        = d L_A / d theta
c_A        = min(1, 5 / (||g_A||_2 + 1e-8))
theta'     = theta - alpha_t * c_A * g_A          # one virtual SGD step
J(theta)   = (|A|/|B|) L_A(theta) + (|Q|/|B|) L_Q(theta')
alpha_t    = kappa * lr_t * ramp_t
ramp_t     = 0 for epoch <= 50, linear over epochs 51..100, 1 afterwards
```

Differentiated exactly with `torch.autograd.grad(..., create_graph=True)`; nothing
is detached and no first-order approximation is substituted. The virtual step is
never written back and never touches the AdamW state. Query labels are used only
inside `J`; inference performs no adaptation.

All three arms see the same molecules in the same order each epoch (verified by
per-epoch batch-union hashes), the same number of supervised labels, the same
outer updates and the same learning-rate trajectory. `R`/`S` differ **only** in
the partition. `alpha = 0` reproduces whole-batch ERM (checked numerically).

## 3. Structural partition (frozen before any valid read)

Keys: the old audit's exact nested radius-1 / radius-2 content keys
(`audit_coverage.radius_signatures` recursion at depth 1 and 2). `k1` is built by
the same builder at depth 1 and is asserted to be a function of `k2`; the
committed valid radius-2 unseen fractions are reproduced to 4.8e-10.

Per batch `B`:

* eligible keys: radius-2 keys present in **2..16 different molecules of that
  batch** (batch-local support; never a valid-derived statistic);
* `d(Q|A)` = atoms of `Q` with an eligible `k2` absent from `A` whose `k1` parent
  is present in `A`, divided by `|atoms(Q)|`;
* score = `[d(Q|A) + d(A|Q)] / 2`;
* `R` = balanced random half split from a fixed RNG;
* `S` = `R` plus deterministic greedy pairwise swaps drawn from a fixed RNG
  stream (≤ 256 evaluated proposals per batch), accepted only if the score rises
  and the coarse constraints hold: `r1_unseen(S) <= r1_unseen(R) + 0.01` and
  standardised mean differences of molecule size and ring rank
  `<= max(R's own difference, 0.25)`;
* no label, model prediction or gradient is read when building a partition.

**Intervention strength (train-only, 256 batches, measured before any GPU run):**

| | 64 proposals (pre-fix) | **256 proposals (declared)** | threshold |
|---|---:|---:|---|
| mean symmetric `d`, R | 0.0867 | 0.0867 | — |
| mean symmetric `d`, S | 0.1300 | **0.1648** | — |
| Δ symmetric `d` | +0.0433 | **+0.0781** | ≥ 0.02 ✔ |
| Δ eligible-conditioned ratio | +0.0844 | **+0.1519** | ≥ 0.10 ✔ |
| Δ radius-1 unseen rate | +0.0032 | +0.0052 | ≤ 0.01 ✔ |
| mean standardised size/ring difference | 0.112 / 0.112 | 0.111 / 0.115 | ≤ max(R, 0.25) ✔ |

The 64 → 256 proposal increase is the single pre-authorised construction fix and
was applied exactly once, before any valid result existed. Both numbers are kept
in `analysis/split_strength.json`. Gate verdict: **PASS**.

## 4. Kappa calibration (train-only, no valid grid)

On the seed-0 epoch-50 prefix, with 32 fixed train-only batches and both frozen
partitions (64 episodes per kappa), `ramp = 1` and one virtual step are measured
for `kappa ∈ {0.1, 1, 10, 100}`:

* admissible requires **≥ 75%** of episodes with non-increasing source L1 **and**
  median per-molecule mean `|f(theta') - f(theta)|` in `[0.005, 0.05]` raw-`y`;
* among admissible values pick the one closest to 0.02 (ties → smaller kappa);
* if the coarse grid jumps across the displacement window, at most three
  log-scale bisections inside the bracketing pair are allowed;
* one `kappa` is frozen for `R`, `S` and every later seed; it is not re-tuned.
  If no kappa is admissible, the meta arms are declared "calibration failed" and
  are not run as if the mechanism had been tested.

Process diagnostics at fixed epochs 100 / 150 / 300 / 600 (same 32 episodes):
source/query loss change under the virtual step, parameter step norm, relative
difference and cosine between the `J` gradient and the ERM gradient. These are
process checks, never generalisation evidence.

## 5. Scheduling and budget (hard cap: 8.0 A100-hours **summed over both GPUs**, 8 h wall)

Pre-declared allocation (not measured times):

| stage | cap (GPU·h) |
|---|---:|
| preparation / strength calibration / profiling / fault checks | 0.4 |
| seed 0, three arms (incl. shared 50-epoch prefix) | 2.2 |
| seed 1, three arms | 2.2 |
| support intervention, two arms × two seeds | 1.4 |
| conditional follow-ups and fault reserve | 1.8 |
| **total** | **8.0** |

After profiling, per-run cost = measured steady epoch × 600 + valid cost + save
cost, with ≥ 15% margin, and the schedule is frozen before long runs. Phases may
borrow unused budget but may not eat an unfinished necessary control. If budget is
short, drop in this order: cross-task/third seed → new training arms on the
internal challenge → optional strength repair. Seed 0's random-partition control
is never dropped. A run that cannot finish is reported as truncated, never as a
completed 600-epoch curve.

## 6. Reading rules (fixed now)

Practical scale for allocating later budget (not a noise bound):

* **performance progress**: mean best-valid *and* W20 improve by ≈ 0.003 or more
  versus `E`, in the same direction in both seeds, with no clear reversal in T100;
  or a smaller but stable gain at a better absolute level. All numbers are shown;
* **structural added value**: in addition, `S` must be better than `R` in W20 with
  the same direction in both seeds (mean ≥ ≈ 0.002);
* if `R` and `S` are equally effective, the conclusion is "general training
  constraint", not "structure mechanism";
* one- or two-epoch best-valid wins with reversed W20/T100 are reported as
  instability; a molecule bootstrap is not cross-seed confirmation;
* W20 = min over 20-consecutive-epoch windows of the window's **median** valid MAE
  (not the best 20 epochs); T100 = median valid MAE over the last 100 epochs.

Pre-authorised single strength repair: only if seed 0 shows **both** `R` and `S`
with last-100-epoch eval-train MAE median more than 2× `E` **and** > 0.02 higher in
absolute terms (common severe underfitting), `kappa/10` may be re-run for `R` and
`S` together from the same epoch-50 prefix to the same 600 epochs, once. A failure
of only one meta arm does not trigger tuning; it is reported as such.

## 7. Supervision-support intervention (independent, train-only)

§7.1 of the round instruction is implemented literally: target radius-2 keys with
4..25 molecule support whose radius-1 parent appears in ≥ 50 molecules, `H` grown
by a fixed-RNG structural greedy to 800..1500 molecules with ≥ 40 keys,
`Q_target` 300..500, `D_plus` 400..1000 with ≥ 1 query and ≥ 2 donor molecules per
key, `Q_background` + `Q_target` = 1000, `M` = 1000, `D_minus` structurally
matched, `B` = rest, `F_plus = B + D_plus`, `F_minus = B + D_minus`
(`support_manifest.json`: sizes, key supports, donor/query counts per key,
matching distances, SMDs, mutual-exclusivity assertions).

`E_plus` / `E_minus`, seeds 0/1, same model/init/recipe, 600 epochs, batch 128,
`lr` schedule over each fit's own update count. Checkpoint selection only on the
internal `M` MAE. `Q` labels are read only after every planned checkpoint exists
and is hashed. Primary endpoint = best-`M` checkpoint; the epoch-600 checkpoint is
reported as a selection-sensitivity check and cannot become primary afterwards.

## 8. Terminal test

Only if a single-model configuration shows repeatable valid progress: one
terminal read of the official test split for the frozen candidates and the
matching `E` reference, per-seed single checkpoints, no averaging of predictions.
Test MAE < 0.060 for **both** seeds = "average target reached"; a single seed
below 0.060 = "single hit"; every seed below 0.060 may additionally be called
"reached in all repetitions". If valid shows no repeatable progress, the test is
not opened.

## 9. Precedent check (≤ 30 min, original papers)

* Li et al., *Learning to Generalize: Meta-Learning for Domain Generalization*,
  AAAI 2018 — MLDG: split each minibatch into meta-train/meta-test, take a virtual
  meta-train step and optimise meta-test loss at the updated parameters. Our `R`
  arm is exactly this family applied to a random half split; `S` replaces the
  random split by a structure-derived one.
* Yang et al., *Learning Substructure Invariance for Out-of-Distribution Molecular
  Representations* (MoleOOD), NeurIPS 2022 — substructure-level invariance for
  molecular OOD. Supports the family; does not supply a performance guarantee for
  this construction, nor a structure-partition MLDG precedent on ZINC.

No claim of a new method is made from the mere application of MLDG to P3.

## 10. Amendments

Append-only. Any deviation is dated, justified, and does not rewrite the above.
