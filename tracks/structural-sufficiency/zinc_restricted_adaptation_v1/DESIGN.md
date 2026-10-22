# DESIGN — zinc-restricted-adaptation-v1 (DBT-PA)

**Round:** `zinc-restricted-adaptation-v1`, track `structural-sufficiency`.
**Question (hypothesis, not a diagnosed bottleneck):** if the P3 local stack is
kept at its seed-0 masked-SSL solution and only a small supervised adaptation is
allowed after it, does a single DBT-based model generalise better than the
existing static trunks?

This is one architecture check under a new **60-minute single-GPU** budget. It
does not reopen GRIT/ESA, does not retrain P3B/L/DBT, and does not buy a second
seed, an ensemble, a router or a different output decomposition.

## 1. Candidate

DBT-PA = the DBT model with its local P3 atom/bond/junction stack frozen and
three independent post-encoder rank-8 residual adapters:

```
a = a0 + U_a SiLU(D_a a0)
b = b0 + U_b SiLU(D_b b0)
w = w0 + U_w SiLU(D_w w0)
```

* `D: 96 -> 8` with PyTorch default init; `U: 8 -> 96` weight **and** bias zero.
* The frozen stack runs as one complete local stack and produces `a0,b0,w0`
  from the *original* seed-0 train-only masked-SSL checkpoint
  (`tracks/structure-contribution/results/bjoc_full10k_pretraining_transfer_v1/pretrain_encoder_s0.pt`).
  It is **not** the supervised DBT `best.pt`.
* Internal frozen-stack intermediates are never adapted; adapted `a` is not fed
  back into the bond encoder and adapted `b` is not fed back into the junction
  encoder.
* DBT's directed-bond token set, endpoint relations, four full-graph
  relation-conditioned attention layers, and reader are unchanged. The adapted
  `a/b/w` are the states those modules see.
* The original DBT modules are constructed first under seed 0, so their random
  state matches the original DBT; the adapters are appended afterwards. This
  avoids mixing the supervised DBT interface into the frozen SSL base.
* Frozen base parameters stay in the inference model and count toward the
  500,000 cap; they are not in the optimizer and are forced to eval mode.
  Expected realised total: **457,443** (202,048 frozen + 255,395 trainable).

## 2. Frozen-feature cache

Because the adapters are after the complete frozen stack, `a0,b0,w0` are a
label-free deterministic function of the graph and the SSL weights. The train
and valid caches store only those base outputs (float32), not adapted states and
not labels. The key contains:

* `FEATURE_VERSION = dbt-pa-feat-v1`,
* the SSL checkpoint sha256,
* the sha256 of the loaded encoder state tensors,
* a structural fingerprint of the graph batch (features/indices/pointers;
  labels excluded).

The cache is checked on a few train and valid graphs against a live frozen-stack
forward before training. Test features are built only after the best-valid
checkpoint is frozen. The model can always run the frozen base live; the cache
is compute reuse, not a lookup predictor.

## 3. Recipe and horizon

| field | value |
|---|---|
| data | official PyG ZINC subset, train 10,000 / valid 1,000; raw-`y` MAE |
| seed | 0 |
| batch | 64, `drop_last=False`, 157 updates/epoch |
| optimizer | AdamW on trainable parameters only, peak LR 3e-4, WD 1e-4 |
| schedule | 1,000-update warm-up, then per-update cosine to 3e-6 |
| horizon | `E * 157`, computed from measured steady epoch time, cap 600 |
| loss | raw-`y` L1 |
| dropout | DBT 0.05; frozen stack 0 |
| grad clip | 5.0 |
| valid | every epoch, all 1,000 molecules |
| selection | strict `<` on valid MAE; ties keep the earlier epoch |

The horizon is **not** inherited from the DBT 1,500-epoch schedule. `E` is
chosen only from speed: after a 3–5 epoch timing run, with a 1.15 safety factor,
a 180 s final-scoring reserve, and the GPU time already spent in this round.
If `E < 200`, the formal run is not launched and the blocking measurement is
reported. Valid MAE during timing is not used to choose `E`.

## 4. Narrow checks (declared before the formal run)

1. total inference parameters split into frozen/trainable and <= 500,000;
2. original DBT modules match the seed-0 original initial state;
3. zero-init adapters reproduce the same SSL base + same random DBT downstream
   without adapters (not the supervised DBT output);
4. frozen encoder hash unchanged after several steps, adapters/trunk updated;
   frozen base absent from optimizer and forced eval;
5. `D` has zero gradient on step 1 (expected, because `U=0`) and non-zero by
   step 3; `U` has non-zero gradient on step 1;
6. cached gather is row-aligned with direct base forward; LR endpoints and 157
   updates/epoch are correct.

## 5. Diagnostics and test

The valid fixed four-group table uses the corrected radius-2 **content** keys
relative to train: `signature-covered` vs `signature-unseen` (any atom with an
unseen key) crossed with `ordinary (y > -3)` vs `tail (y <= -3)`. Labels are
used only for this descriptive table; no sampling rule, gate or model input.

The best-valid checkpoint is frozen (with sha256) before test features or test
labels are touched. Test is one descriptive evaluation of that frozen
checkpoint, reusing the aligned P3B/L/DBT predictions. The official test split
is already exposed in this repository; no blind-test claim is made.

## 6. Stop rule

After the formal short run and test report, stop. No automatic seed 1, adapter
rank grid, second adapter design, output decomposition, or long schedule. A
single-seed improvement is only a reason to consider a next pre-registered
round, not evidence of a root cause or of reaching 0.05x.
