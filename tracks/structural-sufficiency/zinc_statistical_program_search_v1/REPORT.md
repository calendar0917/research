# REPORT — zinc-statistical-program-search-v1

**Round:** 2026-10-25 23:55 → 2026-10-26 01:15 CST (hard limit 07:55). **Commit:** `e97e08b`
(plus report commits). **Official test: never constructed, indexed or read**
(`official_test_loaded=false` in every `train.json`). **Decision: `NO_USEFUL_GAIN`** —
both seed-0 configurations are closed at official 10k; neither family is claimed
invalid.

---

## 1. What was run

Two independent bodies over the **same legal front-end** (P3 train-only masked-SSL
atom/bond encoders, 89,152 parameters, junction module deleted and not counted),
the **same** official splits (train 10k / valid 1k), the **same** mature DCC-R0
recipe (AdamW, lr 1e-3→1e-5 cosine, 500-step warm-up, wd 1e-5, clip 5, raw-y L1,
batch 128, **600 epochs = 47,400 updates**, full valid every epoch, strict
argmin valid). Both runs completed the full declared horizon — no early stop, no
LR reset, no seed averaging, no test access. Design frozen in `DESIGN.md` before
launch.

| run | body | params | best valid | ep | W20 (start) | T100 | final | train@best (eval) | train@final (eval) | GPU·h |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `QF_s0` | Q-fixed: rooted empirical-measure queries | 248,361 | 0.110569 | 308 | 0.112376 (544) | 0.112406 | 0.112390 | 0.019350 | 0.000431 | 0.427 |
| `FC_s0` | F-content: content-weighted rooted-forest marginals | 205,506 | **0.101948** | 337 | **0.103530** (461) | **0.103601** | 0.103604 | 0.021785 | 0.001330 | 0.393 |
| *DCC-L s0 (historical, same recipe/cache/SSL)* | dual-channel composer | 494,193 | *0.078537* | *489* | *0.079082* | — | *0.079076* | *0.005510* | — | *0.20* |

Total added cost: **1.14 A100·h** (0.31 of it wasted on two invalid runs, §6),
two A100s, wall clock 1 h 20 min. Unused budget: 14.86 GPU·h / 6 h 40 min.

## 2. Did either body beat the reference? No.

* Single model, full 1000-row valid: best body **F-content 0.101948**. That is
  **+0.0234** worse than DCC-L seed 0 (0.078537) and **+0.0259** worse than the
  4-seed DCC-L mean (0.075990); Q-fixed is +0.0320 / +0.0346 worse.
* The loss is not a single-epoch artefact: W20 and T100 are within 0.002 of the
  best value for both bodies (curves are flat from epoch 300 onward), and the
  per-molecule paired comparison against the frozen DCC-L seed-0 predictions is
  clearly positive — QF **+0.0320 [+0.0180, +0.0538]**, FC **+0.0234 [+0.0112,
  +0.0427]** (bootstrap over valid molecules), only 38.7 % / 40.9 % of molecules
  better, median ΔAE +0.0126 / +0.0088.
* Ordinary molecules (`y > −3`, n = 935) are already worse: **+0.0197** (QF),
  **+0.0127** (FC). The very large tail difference (n = 65, `y ≤ −3`: +0.210 /
  +0.178) contributes roughly half of the total gap, i.e. these bodies degrade
  the same hard subgroup where DCC-L is already weakest (DCC-L tail MAE ≈ 0.24).
* **Second seed: not bought, deliberately.** The gap is +0.023…+0.032, i.e.
  12–17× the historical DCC-L seed spread (4-seed sd ≈ 0.0021), and both stable
  measures agree with the best-epoch number. A second supervised seed cannot
  change the conclusion; the round's rule was to buy it only for a competitive
  first seed.
* Reference and bias: the reference is the **historical** DCC-L seed-0 number
  produced by the *same* trainer, cache, split, SSL checkpoint and recipe; it was
  not re-trained this round, so the comparison is not a simultaneous randomised
  contrast. It is not selected on test, and no epoch/checkpoint of ours was
  chosen against any reference (selection is argmin valid only). The DCC-L
  anchor is the *stronger* of the two reference numbers (seed 0 0.0785 vs 4-seed
  mean 0.0760); using either does not change the verdict.

**The bodies are not under-trained — they generalise worse at equal fit.** At
epoch 500 the training fit is essentially identical across the three bodies
(online train MAE 0.0043 / 0.0051 / 0.0044 for Q / F / DCC) while valid MAE is
0.1124 / 0.1030 / 0.0790. Both bodies memorise the 10k set faster than DCC-L and
still lose ≥0.024 on valid.

## 3. Mechanism evidence (frozen weights, forward only — dependency, not retraining)

`analysis/mechanism.json`; measured with hooks on the production forward, so the
numbers are the trained computation.

**Q-fixed** — the rooted measure machinery is real but is *not used as rooted*:

* All six measures are populated: mean per-root counts 6.1 / 5.6 / 12.2 (atoms
  in distance {0–2}/{3–4}/{≥5}) and 8.7 / 5.7 / 11.4 (bonds by min endpoint
  distance), non-zero fraction ≥ 0.978; 1,149 records per root on average.
* Root-pool survival `‖Σ_r h_r‖ / Σ_r‖h_r‖` median **0.432** [p10 0.352, p90
  0.545]: the reader sees a strongly cancelled sum of root vectors.
* **Root-rebinding probe** (rotate which root owns which record inside each
  molecule; molecule-level record multiset and per-root counts kept consistent):
  mean |Δpred| **0.0049**, prediction correlation **0.99999**, valid-MAE change
  **−0.00009**. The trained body is essentially invariant to the root binding, so
  the "root-conditioned" statistics act as a **molecule-level bag read by a sum**;
  the pairing with the root atom carries almost no weight.

**F-content** — the learned weights are genuinely content-dependent and are used,
but the result still generalises worse:

* `w_e` mean 0.841, sd **0.799**, range exactly `[0.1, 2.0]` (both sigmoid ends
  saturated), within-molecule sd median 0.775 — real per-bond modulation, not a
  constant.
* The forest statistics are non-degenerate within molecules: coefficient of
  variation of `r_i` = 0.19 / 0.35 / 0.46 / 0.45 and of `p_e` = 0.80 / 0.64 /
  0.42 / 0.26 for τ = 0.25 / 1 / 4 / 16.
* Frozen-weight dependency: replacing `w_e` by its **per-molecule mean** costs
  +0.0103 MAE; replacing it by **1** costs +0.0304 MAE (mean |Δpred| 0.073). So
  both the forest channel and its within-molecule weight variation shape the
  prediction. This is the `ACTIVE_BUT_NON_BENEFICIAL` pattern already seen in
  SCM-v1: the statistic is read, and is still not useful.

No mechanism claim here comes from a retraining counterfactual; the Q-adaptive /
F-fixed retraining controls were **not** bought (see §4).

## 4. Branches: closed, weakened, never tested

* **Closed by actual measurement (this round):** Q-fixed and F-content at official
  10k with the declared recipe. Both fit and plateau ~0.023–0.032 behind DCC-L;
  the task's rule for a plateaued, clearly-worse, normally-fitting arm applies.
* **Declared but not bought (reason):** **Q-adaptive** (+6,208 params, 2× read
  cost) and **F-fixed** (`w_e ≡ 1`) were conditional on a competitive first seed.
  Neither was competitive, so buying them would be mechanism decoration on an
  eliminated arm. They remain untested, not refuted.
* **Weakened, not refuted:** the *specific* organisation "root-conditioned
  empirical measures + sum pooling" is directly weakened by the rebinding probe —
  the root binding is not exploited, so the Q body was effectively a bag model.
  The *forest-marginal* organisation is weakened as a **performance** route at
  10k, but its statistics are demonstrably active, so "the idea is dead" is not
  supported.
* **Never tested here:** label-scale behaviour of either body (4k/8k), unbounded
  or differently bounded edge-weight parameterisations (the observed saturation at
  both ends of `[0.1, 2.0]` is the single most plausible implementation-level
  cause and was fixed by design, not tuned), ring/three-body/spectral extras,
  and anything on official test.

## 5. Novelty position

Both bodies are engineering compositions of classic objects: a Set-Transformer /
learned-aggregation-style query read of empirical measures (Q) and matrix-forest
theorem marginals of a content-weighted graph (F), each with a rooted or
per-object binding and a non-linear lift before pooling. **No novelty claim is
made.** The one non-standard observation of the round is negative and measurable:
with sum pooling, root-conditioned measure networks collapse to molecule-level
bags (corr 0.99999 under root rebinding), which is the same "information enters
but is cancelled at the pool" family of failure that SCM-v1 diagnosed.

## 6. Cost, failures and boundary

| item | GPU·h | note |
|---|---:|---|
| two invalid seed-0 runs (cancelled ~epoch 130) | 0.31 | implementation error, see below |
| 2-epoch timing on both cards | 0.01 | projections 0.38 / 0.43 GPU·h |
| QF_s0, FC_s0 (600 epochs each) | 0.82 | complete, checkpoints + curves + per-molecule valid predictions kept |
| **total** | **1.14** | of 16 A100·h hard / 14 soft |

**Implementation error (recorded, corrected, re-checked).** The first launch
reused `ad_train.train`, which hands **its own** `train_cache` and `valid_cache`
to a single collate callable; my closure had bound the *train* split, so valid
labels were scored against train features (valid MAE ≈ 2.0 while train fit to
0.07). Both runs were cancelled, their artifacts deleted, nothing from them is
reported. The fix makes one closure dispatch on `cache["split"]`, and a new CPU
check asserts that the same closure is exactly correct for both splits
(`cpu_checks.json`: cache-vs-independent-reference max |Δ| = **0.0** for both
bodies). No other protocol change was made after freezing.

Also disclosed: the frozen architecture-discovery `dist_flat` cache stores
distances as `uint8`, so the int64 unreachable sentinel was truncated to 0. Every
molecule in train/valid is connected (0 disconnected of 11,000, asserted at
prep), so no record is mis-bucketed; the prep recovers unreachable pairs as
`d == 0 ∧ r ≠ u` and keeps distance id 10 reserved, but that id has **no real
support** in this data — the "≥5 / unreachable" group is entirely "≥5" here.

Other checks that passed (`analysis/cpu_checks.json`): vectorised measure read vs
brute-force reference 8.9e-16; empty groups exactly zero, no NaN; Q count
conservation (atom groups sum to n, bond groups to m); F forest identity
`Σr + Σp = N` to 7.1e-15 and enumeration agreement 2.2e-16; atom/bond
permutation, batch recomposition and batch order invariance ≤1.7e-16; train/eval
parity 0.0; finite-difference gradient check ≤4.2e-6 (relative); no BatchNorm;
checkpoint reload exact; optimizer covers 100 % of parameters; 205,506 /
248,361 ≤ 500,000; no label-derived tensor in any batch.

## 7. Answers in one line each

1. **Strongest:** F-content, 205,506 parameters, valid **0.101948** (W20 0.103530,
   T100 0.103601, eval-mode train 0.021785), 0.393 GPU·h.
2. **vs DCC-L:** worse by +0.0234 (seed 0) / +0.0259 (4-seed mean), single seed by
   design, historical anchor not re-trained, no test read, no test-based selection.
3. **Which statistic acts where:** the forest marginals and their learned weight
   variation both move the prediction (+0.030 / +0.010 frozen-weight dependency)
   but do not help; the Q rooted measures are read but collapse to a bag
   (root-rebinding corr 0.99999, pool survival 0.43). All of this is frozen-weight
   evidence, not retraining evidence.
4. **Closed/weakened/untested:** see §4; Q-fixed and F-content closed at 10k;
   Q-adaptive and F-fixed untested; weight-bound parameterisation untested.
5. **Novelty:** none claimed — classic statistics composed in a rooted/object
   binding; the only new fact is the measured bag collapse of rooted measures
   under sum pooling.
6. **Cost/boundary:** 1.14 A100·h, 0 official test reads, 14.86 GPU·h unused; one
   implementation error found, fixed and covered by a new check.
