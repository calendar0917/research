# PROTOCOL — LCRC-S3-SP v1: root-conditioned selective three-hop path residual

Round id: `lcrc-pcba-s3-selective-v1`
Status: **frozen before the first formal training step.**
Only three kinds of later edit are allowed: (a) measured GPU timings written
into §9, (b) measured audit outcomes written into §8 as PASS/FAIL records,
(c) the amendment log in §12. Nothing in §3–§7, §10 or §11 may change after the
first training step.

## 1. The single question

S3 (`lcrc-pcba-s3-v1`) showed that an explicit one-shot three-hop path residual
beats both W2 and a near-parameter-matched FFN control on official Validation AP
(`0.2714890` vs `0.2652729` vs `0.2640242`, same seed / batch plan / recipe).
Inside each root, S3 still summarises its paths **uniformly**
(`mean`, `sum/√count`, `log1p(count)`), so every chemical path of a root has the
same weight.

Does adding a **root-conditioned selective** summary of those same paths — a
two-head content attention whose query is the root's static atom embedding — and
keeping the original uniform summary as a separate residual, improve official
Validation AP (development dimension) and the official Test AP of the
Validation-selected checkpoint (exposed-test dimension) over S3?

One model is trained. No capacity control, no mechanism ablation, no scaffold
re-split.

## 2. Frozen read-only references

| item | value |
|---|---|
| base model | `lcrc_pcba_backbone_scaling_v1` / `W2`, 994,324 params |
| W2 best-Validation epoch / AP | 24 / `0.26527331610028143` |
| W2 epoch-40 AP | `0.26186691813350904` |
| S3 model | `lcrc_pcba_s3_v1` / `S3`, 1,018,532 params (+24,208) |
| S3 best-Validation epoch / AP | 29 / `0.271489039695166` |
| S3 epoch-40 AP | `0.2677211` |
| S3 seed-0 init state SHA-256 | `a850ac855cc38858cfc0bfa90d2e4d81c27be11ee3af373a37099d9277a90722` |
| W2 seed-0 base state SHA-256 | `6a3df38f00a5a4754c310a82d4ad5f71d7f89e3392f1c914aacc4113a3d6f53b` |
| W2 trunk-init fingerprint | `5f7101a8f378e7c9283caaade8d61eee733c0c9f3cfb65b5584c8a1e495ef63d` |
| global batch plan SHA-256 | `06f437299fa451e449202a990d28f3b8d101e7aec709bcd72a878c8de864b6f6` |
| consumed 40-epoch prefix SHA-256 | `1a24dfca54898dd2b0e7c58d2b2f1956df67b233fdceaa7a376f81a2377a5f58` |
| S3 Train+Valid structural cache | `tracks/structural-sufficiency/lcrc_pcba_s3_v1/runs/s3_cache` (read-only; 39,754,052 paths / 394,136 rows / `labels_present=false` / `test_rows_present=false`) |

The S3 checkpoint of record is `runs/S3/checkpoint_best.pt` (epoch 29,
SHA-256 `ad665168da562120321171f6042a97f746ae4ae7377bd8364fc8d6264af007d0`);
the W2 checkpoint of record is `runs/W2/checkpoint_best.pt` (epoch 24,
SHA-256 `d86fe5e56e898d382acf66eb0533beb915f8ca39144c6c1c74e25e147667d2bd`).
Both are used read-only, for Validation replay and Test inference only.

## 3. Model

```
S3-SP = original W2 + original S3 + Selective Path Residual
```

For a batch with `a = atom_encoder(atom_feat)` (`[n_atoms,192]`) and the
unchanged S3 path embedding `z = s3_branch.encode_paths(batch, a)`
(`[n_paths,48]`), with `root = batch["s3_path_atom"]`:

```
s, counts = s3_branch.root_inputs(z, root, n_atoms)      # unchanged S3, 97 dims
r_original = s3_branch.out(s)                             # unchanged S3

q        = Linear(192,48)(a).reshape(n_atoms, 2, 24)      # static atom encoding
k        = Linear(48,48)(z).reshape(n_paths, 2, 24)
v        = z.reshape(n_paths, 2, 24)                      # no extra projection
score    = (q[root] * k).sum(-1) / sqrt(24)               # [n_paths,2]
weight   = segment_softmax(score, root, n_atoms)          # per root, per head
attn     = scatter_sum(weight.unsqueeze(-1) * v, root)    # [n_atoms,48]
delta    = attn - s[:, :48]                               # attn - uniform mean
r_sel    = Linear(48,128)(delta)                          # zero-initialised
r_sel[counts == 0] = 0

h      = drop(h_raw + r_original + r_sel)                 # one dropout call
logits = post_local(h, batch)                             # unchanged W2 readout
```

`mean_z` is `s[:, :48]`, the S3 branch's own uniform mean, so
`delta ≡ 0` when the attention is uniform and `delta ≡ 0` for a root with a
single path (`attn = z = mean`). The grouped softmax uses a per-root maximum and
a per-root normaliser (`scatter_reduce_(amax)` + `index_add_`); there is no
per-path Python loop, no truncation and no sampling.

Implementation constraints (audited, not asserted):

* `atom_encoder`, `bond_encoder`, `local` and `drop` are each called **once**;
* the S3 encoding (`encode_paths`) is called once and the original S3 residual
  is added unchanged;
* the selective branch reads only `a` (static), `z` (path embedding), `root`,
  `mean_z` and `counts` — never `h`, `h_raw`, `bond_index` or any propagated
  state;
* the post-`local` readout still reads only `h`, `atom_batch`, `n_graphs` and
  `global`;
* no new path enumeration, no S4/S5, no iterative message passing, no new
  molecule-level block.

`SEGMENT_SOFTMAX` is defined as: for every root `r` and head `h`,
`weight[p,h] = exp(score[p,h] - max_{p'∈r} score[p',h]) / Σ_{p'∈r} exp(...)`,
with each root's paths contiguous or not (the implementation is order-free).

## 4. Parameter budget

| tensor | shape | parameters |
|---|---|---:|
| `selective.q` | 48×192 + 48 | 9,264 |
| `selective.k` | 48×48 + 48 | 2,352 |
| `selective.out` | 128×48 + 128 | 6,272 |
| **new total** | | **17,888** |

| model | parameters |
|---|---:|
| W2 | 994,324 |
| S3 | 1,018,532 |
| S3-SP (target) | **1,036,420** |

The new parameters are 1.76 % of the original S3 branch. No FFN is added and no
parameter count is padded to hit a round number. The exact count is verified
with `sum(p.numel())` before training (§8.1).

## 5. Initialisation

From scratch — **no continuation from the S3 epoch-29 checkpoint.**

1. `torch.manual_seed(0)` and build the historical W2 seed-0 arm
   (`bs_model.build_model`), exactly as every previous round did.
2. Build the historical S3 arm (`s3_model.build_model("S3", seed=0)`), whose
   private generator seed is 901 and whose output layer is zeroed.
3. Construct the SP model under `torch.random.fork_rng`, load the W2+S3 state
   bitwise, then initialise only the new tensors:
   * `q`, `k`: kaiming-uniform weight + fan-in-uniform bias from a **private CPU
     generator with seed 1201** (q first, then k);
   * `out`: all-zero weight and bias.

The global torch RNG is not advanced by any SP-specific step: after the build it
sits exactly where the historical S3 build leaves it (one inherited W2
construction), so the dropout stream is the same as S3's.

Consequences that are checked before training:

* original W2 and S3 tensors are bitwise equal to history
  (base `6a3df38f…`, S3 `a850ac85…`);
* epoch-0 logits equal the historical W2 **and** S3 epoch-0 logits;
* the first training step produces a finite, non-zero gradient on
  `selective.out.*`; from step 2–3 on, `selective.q.*` and `selective.k.*` also
  receive finite non-zero gradient;
* the selective residual is not identically zero during or after training.

## 6. The single formal training

| item | value |
|---|---|
| dataset / split | `ogbg-molpcba`, official scaffold split |
| Train / Valid / Test | 350,343 / 43,793 / 43,793 |
| seed | 0 |
| epochs | 40 (no early stopping) |
| batch size | 128 |
| steps per epoch | 2,738 (109,520 total) |
| batch order | the frozen W2 plan `06f43729…b6f6`, prefix `1a24dfca…5f58` |
| optimizer | AdamW |
| peak LR / warmup | 3e-4 / 1 epoch linear from 3e-5 |
| scheduler | per-step cosine to 1e-5 |
| weight decay / betas / eps | 1e-3 / (0.9, 0.999) / 1e-8 |
| dropout / clip / precision | 0.10 / 5.0 / FP32 |
| loss | the original masked BCE over observed entries |
| data pipeline | inherited `bs_train.train_step`; S3 tensors from the frozen Train+Valid cache |

Checkpoint selection: full official Validation AP at **every** epoch 1…40, then

```python
best_epoch = argmax(valid_AP[1:41])   # earliest epoch wins exact ties
```

`checkpoint_best.pt` (selected), `checkpoint_epoch40.pt` (auxiliary endpoint)
and `checkpoint_last.pt` (resumable) are written, with `train_curve.csv`.
No Test score is read during training and no epoch is *not* evaluated.

## 7. Test evaluation (this round explicitly includes the official Test)

The MolPCBA official Test split is **already exposed** — it was read by
`lcrc_pcba_continuation_recipe_v1` on 2026-11-06, after that round's own
prediction freeze. This round's Test role is therefore

```
EXPOSED_TEST_PERFORMANCE_BENCHMARK
```

and the report must state that it is **not** a blind generalisation estimate,
because the split has been read before and the result may influence the next
architecture decision.

Rules, all enforced in `code/sp_test.py`:

1. **Three models, one selection rule**: W2 best-Valid epoch 24, S3 best-Valid
   epoch 29, SP best-Valid epoch of this round. No epoch shopping: exactly one
   checkpoint per model.
2. **Test structure without labels**: a separate label-free Test S3 structure
   cache is built with the *same* enumeration and FeatureSpec as the training
   cache (`runs/test_s3_cache`, `labels_present=false`, `training_scope=false`,
   `test_rows_present=true`). It never enters a DataLoader; the training cache is
   not modified and still contains no Test row.
3. **Freeze then score** (`freeze` → `score`): `freeze` replays Validation AP for
   each checkpoint (must match the recorded value to `1e-4`), writes Test logits,
   and seals every checkpoint/logits/row-id SHA-256 plus the selection manifest.
   Only `score` opens `data/ogb/ogbg_molpcba/raw/graph-label.csv.gz`, through the
   inherited `cc_data.read_official_test_labels` gate, which first re-verifies
   that the raw Train/Valid blocks equal the frozen label artifact.
4. **Scoring**: the official `ogb.graphproppred.Evaluator("ogbg-molpcba")` AP and
   an independent per-task sklearn AP recomputation (they must agree to `1e-9`);
   NaN labels are masked per task; tasks without both classes are excluded and
   the scored-task count is reported as measured (never assumed to be 127).
5. **No refitting**: the Train-only normalisation of the frozen cache is used
   as-is for every arm.
6. Forbidden: using Test labels in training, scanning 40 Test checkpoints,
   retraining a new architecture after seeing Test.

## 8. Pre-training audit (minimum necessary, no extra ablations)

Sixteen checks, of which 1–14 are CPU/GPU math+structure and 15–16 are run-level:

| # | check | criterion |
|---|---|---|
| 1 | total parameters | `sum(p.numel()) == 1,036,420` and new == 17,888 |
| 2 | W2/S3 init parity | base state `6a3df38f…`, S3 state `a850ac85…`, trunk fp `5f7101a8…` |
| 3 | epoch-0 logits | `|Δ| ≤ 2e-5` vs W2 and vs S3 (CUDA; 0 on CPU) |
| 4 | root-local softmax | per root, per head weight sums = 1 (±1e-5); no cross-root mixing |
| 5 | single-path root | `attn − mean == 0` exactly and for a one-path-per-root batch |
| 6 | empty root | finite, `delta == 0`, `r_selective == 0` even with a learned bias |
| 7 | atom renumbering | `|Δlogit| ≤ 2e-5` (float32 scatter order) |
| 8 | batch isolation | 0 paths crossing a molecule boundary |
| 9 | S3 semantics/counts unchanged | re-enumerated fields and per-root counts equal the cache |
| 10 | query source | the selective branch receives the static `a`, and no hidden state |
| 11 | call budget | atom/bond/local/drop/S3-encode/S3-out/selective each exactly 1 |
| 12 | gradient flow | step 1: `out.*`; by step 3: `q.*`, `k.*` finite and > 0 |
| 13 | masked BCE / batch plan | inherited loss object and plan hash `06f43729…b6f6` |
| 14 | checkpoint round-trip | state, optimizer step, RNG hashes and logits identical |
| 15 | no NaN/Inf | asserted every training step on logits, loss and gradient norm |
| 16 | operator count | does not grow with the path count (no per-path loop) |

No structure-distinguishability study, no COUNT/ROOT ablation, no S4/S5, no
second architecture.

## 9. GPU budget

| item | budget |
|---|---|
| host / pool | `res` / `res-gpu1` (A100-SXM4-40GB), do not preempt GPU0 |
| target | ≤ 3 A100 GPU-hours |
| hard cap | ≤ 5 A100 GPU-hours |

Includes the GPU smoke/audit, the single 40-epoch run (with Validation every
epoch), the three-model Test freeze and inference. Historical W2/S3 Test
inference is counted. The CPU Test-cache preprocessing is recorded separately.
If preflight shows the run cannot fit, report — do not shrink data, epochs or
paths.

Measured (filled after the run):

| item | value |
|---|---|
| GPU audit + smoke | |
| SP training, 40 epochs | |
| Test freeze (3 checkpoints, Validation replay + Test logits) | |
| **total** | |

## 10. Decision rules (pre-registered)

Two independently reported dimensions:

* development: `Best Valid AP` of the selected checkpoint vs S3's `0.2714890`;
* exposed Test: `Test AP` of the selected checkpoint vs S3's measured Test AP.

A dimension counts as **improved** only if the gain is ≥ `+0.002` (the same
practical threshold the S3 round used for `NO_CLEAR_GAIN`).

| Valid | Test | label |
|---|---|---|
| improved | improved | `VALID_AND_TEST_GAIN` |
| improved | not improved | `VALID_ONLY_GAIN` |
| not improved | improved | `TEST_ONLY_GAIN` (exploratory, does not validate Valid selection) |
| not improved | not improved | `NO_CLEAR_GAIN` |

Engineering work target (not a significance threshold):

```
Best Valid AP ≥ 0.2745  and  Test AP ≥ TestAP(S3) + 0.003
```

Failure does **not** authorise more heads, longer training, another seed, a
second architecture or another ablation in this round.

## 11. Explicitly out of scope

* COUNT / ROOT / HS3 ablations, S4/S5, extra heads, width changes;
* a parameter-matched FFN control for SP;
* seed 1 or any repeated seed;
* new scaffold re-splits or train-only development folds;
* hyper-parameter tuning of any kind;
* using Test for architecture selection, early stopping or checkpoint choice;
* any change to the frozen W2/S3 code, caches or checkpoints.

## 12. Amendment log

* (pre-run) `A1` — the batch-plan hash in the round brief is transcribed as
  `06f43729fa451e44…`; the value recorded by all three historical runs (W2, S3,
  F) and reproduced by this round's audit is `06f437299fa451e44…`. The repo
  value is used everywhere.
* (pre-run) `A2` — the atom-renumbering tolerance is `1e-5` on CPU and `2e-5` on
  CUDA rather than exactly 0, because renumbering reorders the S3 path list and
  therefore the float32 scatter accumulation order (the S3 round observed the
  same magnitude). Semantics are unchanged; only summation order differs.
* (pre-run) `A3` — the uniform-attention equality `attn − mean == 0` is checked
  with a `1e-6` float32 tolerance for the same reason (a single-path root is
  checked for exact equality).
