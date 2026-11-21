# Protocol — structural word functions + real-binding composition (s0, v1)

Pre-registration written **before** any branch (FULL / BAG_CONT / control) result existed.
Only `WORD_PREFIX`, `GINE_RAW_REF`, `COUNT_RIDGE`, `FP_KRR` and the implementation
checks were known at the time of writing.

## Question

On `ogbg-mollipo` (official 3360 / 420 / 420 split), does a **shared, retained-attribute
nonlinear word function** + **composition under the real variable binding** +
**coordinate elimination** beat (a) the word bag, (b) direct learning of one large
local function, and (c) plain GINE, at a fixed pilot budget?

## Objects (frozen)

* Rooted connected templates with `k = 1..4` atoms, enumerated once:
  **16 templates** with counts `1 / 1 / 3 / 11` for `k = 1, 2, 3, 4`.
* A *word* is a template + the full ordered attribute matrix of its atoms and its
  physical intra-template bonds, in an automorphism frame (root fixed in slot 0).
  Words are **retained attributes only** — no learned atom embedding replaces the
  attribute columns; the learned part starts at `WordMLP(256 → 128 → 64)`.
* A *frame* is one automorphism-orbits representative of the word on the molecule;
  frames per word = `|Aut_root|` ∈ {1, 2, 6}.
* A *sentence* = (word `U`, word `V`) that share ≥ 1 atom (k ≤ 4, cover word size 3–4).
* A *cover* = all frames of `U × V` for one sentence; a *row* = one (frame pair).
* Relation payload is explicit and finite: same-atom matrix `E` (u16), physical
  cross-bond matrix `C` (u16) and 16 cross-bond attribute codes. There is no
  message passing and no recurrence anywhere.

Full corpus (all 4200 molecules, computed once): 1,954,725 words,
2,463,158 frames, 3,316,287 sentences, 9,333,784 covers, 16,863,050 rows
(1.807 rows/cover), 71,004 unique word-frame inputs and 27,188 unique relation
inputs (97.1 % / 99.8 % dedup). All statistics that touch labels (`mu_T`, `nu_k`,
support, target scaler) are computed from **Train only**.

## Data identity (locked)

* raw cache `ogbg_mollipo_sq-pilot-cache-v1.npz`,
  sha256 `fe002287fa9c581278bc5430f5b4a5b26b35d9d4d5207d138f15956d93123c7c`
* `data/swf_base.npz` sha256 `67ddacf274bb628683e4799b58332762315ca2e8258f8c692b9cb7896766ded8`
* `data/swf_sent.npz` sha256 `69585ecbdec0c3f72d4e5b046ee77a1e09cca20662bf4c33d7872c66f2fac1d8`
* no holdout is carved out of Train in this round; the official split is used as is.

## Training recipe (frozen, seed 0 only)

| stage | epochs | peak lr | warmup | early stop |
|---|---|---|---|---|
| `WORD_PREFIX` (bag only) | 160 | 1e-3 | 5 | min 100, patience 40 |
| `FULL` / `BAG_CONT` | 120 | 3e-4 | 2 | min 80, patience 40 |
| `GINE_RAW_REF` | 240 | 1e-3 | 5 | min 160, patience 50 |

* batch 64 (branch/prefix), 128 (GINE), eval batch 256, AdamW wd 1e-4, cosine decay
  to `min_lr`, grad clip 1.0, tf32 on, `torch.use_deterministic_algorithms(True)`.
* every epoch is evaluated in `eval()` mode on **all three** splits; RMSE and MAE are
  reported in the original units (scalers invert exactly).
* `FULL` and `BAG_CONT` start from the **same** deterministic `branch_init.pt`
  (prefix weights loaded, `join` output zero-initialised) and share the same data
  order and horizon; only the sentence path differs.
* Model selection is on Valid only; Test is read once for the recorded table.

## Arms

| arm | meaning |
|---|---|
| `WORD_PREFIX` | bag of shared word functions only (`y_bag`) |
| `FULL` | bag + sentence/relation composition + residual join |
| `BAG_CONT` | bag + join with the sentence context forced to zero (control: is the composition doing anything?) |
| `EARLY_POOL` | attribute permutation pooled **before** binding (control: does the frame-level attribute retention matter?) |
| `FLAT_CONTEXT` | one MLP on the concatenated large-local object (control: is the factorised structure needed?) |
| `GINE_RAW_REF` | same-round GINE, raw features, frozen factory — external performance reference only |
| `COUNT_RIDGE` | Linear-Graphlet-Models-style attributed subgraph counts + ridge |
| `FP_KRR` | Morgan r=2 / 2048 bit / chirality, Tanimoto kernel ridge |

## Decision rule (frozen thresholds)

Let `gap = FULL_valid − GINE_valid`, `gain = BAG_CONT_valid − FULL_valid`,
`margin = FULL_valid − best(COUNT_RIDGE, FP_KRR)_valid`.

* refuse a verdict while `GINE_RAW_REF` or `COUNT_RIDGE` is missing, or while
  `FULL`/`BAG_CONT` are incomplete.
* `STOP_AT_PILOT` if `gap > 0.10` or `margin > 0.03` or `gain < 0`.
* `RUN_EARLY_POOL_AND_FLAT_CONTEXT` if `gap ≤ 0.06` and `gain ≥ 0.01` and `margin ≤ 0.02`.
* `RUN_FLAT_CONTEXT_ONLY` if `gap ≤ 0.10` and `gain ≥ 0.02` and `margin ≤ 0.03`.
* otherwise `END_THIS_ROUND`.

A positive claim needs **all** of: `gap ≤ 0.06`, `gain ≥ 0.01`, `margin ≤ 0.02`,
a paired bootstrap over molecules whose 95 % interval for `gain` excludes 0, and a
consistent sign under the re-labelling / cover-swap invariance checks.

## Implementation checks that gate the runs

`swf_run.py checks` must pass: data identity, binding algebra (combine-then-average
= 0.5 vs average-then-multiply = 0.25 on the same exchangeable pair), cover-swap
symmetry, scatter-vs-loop equality, branch init = prefix, first/second-step gradient
sanity, and exact re-labelling invariance (FP64 ≤ 1e-10, GPU FP32 ≤ 1e-4).

## Budget

4 h wall from T0 `2026-11-21T19:21:32+08:00`, ≤ 8 GPU·h, ≤ 2 GPUs, no launch after
T0+3 h (22:21:32), training stopped by T0+3 h 30 m (22:51:32), ≤ 10 formal units,
≤ 2 engineering reruns. GPU UUIDs and per-run wall times are recorded in
`analysis/resource_ledger.jsonl`.

## Deviations (also in `analysis/deviations.jsonl`)

1. `COUNT_RIDGE` alpha grid was extended from `1e-3 … 1e2` to `1e-3 … 1e5` after the
   frozen grid selected its own upper boundary (`alpha = 100`, valid 1.3697) and the
   validation curve was still decreasing. The frozen-grid result is preserved
   verbatim in `analysis/baselines_frozen_grid.json`; the extended grid is a
   *stronger* baseline (valid 0.9765) and is what the decision uses.
2. The first `GINE_RAW_REF` launch finished training but crashed while writing
   `run.json` (`argparse` handler object in `vars(args)`). Fixed and re-run with the
   identical recipe → engineering rerun #1.
3. Previous-round files on the compute host were moved to
   `/home/hxy/cy/rr-runs/_pre_swf_moveaside_20261121_195208` (not deleted) so the
   new commit could be deployed.
