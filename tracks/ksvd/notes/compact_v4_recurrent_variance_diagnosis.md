# Compact-v4 T=2 recurrent Q16 — training-variance diagnosis

Date: 2026-09-13
Protocol: `compact_v4_recurrent_variance_diagnosis_v1`
Canonical model: compact-v4-smallhead T=2 weight-tied recurrent pair–centre,
h=48, q=16, **82,115 params**.
Official test: **never loaded.**
No knowledge distillation, no teacher/student, no Q24/Q32, no new architecture,
no T=3, no pair-to-pair, no LR/scheduler/dropout/batch sweep, no ensemble-weight
search.

## Question

Is the single model training-variance limited and is the generalisation gap
worth regularising?  This is the prerequisite before any further architecture
work: if seed / checkpoint variance dominates, a single architecture change is
not measurable at the current budget.

## Phase A — canonical reproducible reference

The code path is frozen to the current working tree at git `a3515e7` (plus the
uncommitted, default-inert scheduler hook in `zinc_compact_v4_smallhead_e2e.py`
and an optional, default-`None` per-epoch snapshot hook added here).

* config: Adam, lr `1e-3`, **wd `1e-5`**, batch `128`, max epochs `240`,
  patience `40`, no scheduler, gradient clip `5.0`, L1 loss, best
  official-valid checkpoint, single stage, CPU, `torch.set_num_threads(4)`.
* seed handling: `build_recurrent(seed)` → `build_baseline(seed)` →
  `_seed_everything(seed)`; small head re-initialised under
  `torch.manual_seed(0)`; train shuffle seed `seed + 91011`; eval order fixed.
* full record: `results/compact_v4_recurrent_variance_diagnosis/canonical_reference.json`.

### Canonical seed0 and the reproduction gap

| run | valid MAE | best epoch | source |
|---|---:|---:|---|
| historical 2026-09-12 `recurrent_seed0` | 0.138376 | 199 | frozen, not reproducible |
| **current-code canonical seed0** | **0.140609** | **167** | sequential rerun |
| reproduction gap | **+0.002234** | | |

The current-code seed0 was reproduced **exactly** by three independent
sequential runs (`rA`, `rA2`, the 500/80 long run's first 207 epochs, and the
canonical run here: epoch 60 `train 0.112866 / valid 0.190911`, epoch 80
`train 0.096949 / valid 0.160919`, best `0.140609 @167`).  The historical
seed0 belongs to an unreproducible branch and was **not** chased.

**Critical environment finding.** Running three trainings concurrently (4
threads each) perturbed the seed0 trajectory after ~epoch 40, while the same
run executed alone was bit-identical to `rA`/`rA2`.  All canonical and
regularisation runs in this study were therefore executed **strictly
sequentially, one process at a time**.

### Canonical seeds

| seed | valid MAE | best epoch | train@best | epochs | horizon warning |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.140609 | 167 | 0.074466 | 207 | False |
| 1 | **0.133440** | 234 | 0.075917 | 240 | True |
| 2 | 0.142722 | 94 | 0.087275 | 134 | False |

seed1 exactly equals the historical `0.1334399786`, i.e. the reproduction gap
is **seed0-specific**.  Training reaches train MAE `0.07–0.09` against valid
`0.13–0.14`; the seed spread is `0.0093`.

## Phase B — fixed top-5 checkpoint soup

The repo's locked Top-5 rule is reused verbatim: the five checkpoints with the
lowest selection MAE (ties → earliest epoch), equal-weight arithmetic mean of
the `state_dict`s; no `k` search, no weight search, each soup is still an
82,115-parameter model.  Selection metric = official-valid MAE (this protocol's
early-stop metric).  Per-epoch snapshots were saved with an inert hook.

| seed | best checkpoint | top-5 epochs | top-5 soup | soup improvement |
|---:|---:|---|---:|---:|
| 0 | 0.140609 @167 | 167,168,186,164,144 | **0.137078** | **+0.003532** |
| 1 | 0.133440 @234 | 234,227,202,217,169 | **0.132210** | **+0.001230** |

Mean soup improvement **+0.002381**.  Checkpoint averaging therefore
substantially stabilises the *single* model, and the gain is larger for seed0,
whose raw best checkpoint is the noisier one.

Variance-ceiling diagnostic (not a model proposal):

```
0.5 * (seed0_soup + seed1_soup) valid MAE = 0.127182
```

which is better than the raw 2-seed best-checkpoint ensemble `(0,1) =
0.128593` by `+0.001411`, i.e. soup and seed averaging capture overlapping but
not identical variance.

## Phase C/D — 1 / 2 / 3-model ensemble scaling

### Single models

| seed | valid MAE |
|---:|---:|
| 0 | 0.140609 |
| 1 | 0.133440 |
| 2 | 0.142722 |
| mean ± std | **0.138924 ± 0.004865** |
| range | 0.009282 |

### Equal-weight ensembles (arithmetic mean of predictions, no weight search)

| ensemble | valid MAE | gain over mean of members |
|---|---:|---:|
| 0+1 | 0.128593 | +0.008432 |
| 0+2 | 0.131222 | +0.010444 |
| 1+2 | 0.128127 | +0.009954 |
| 2-seed mean | **0.129314** | **+0.009610** |
| pair range | [0.128127, 0.131222] (spread 0.003095) | |
| **0+1+2** | **0.125962** | **+0.012962** |

* **3-seed over mean single: +0.012962.**
* **3-seed over mean 2-seed: +0.003352 (≥ +0.002).**

### Disagreement / correlation

| quantity | value |
|---|---:|
| mean abs prediction disagreement | 0.09324 |
| pairwise prediction correlation | 0.99682 – 0.99728 |
| pairwise residual correlation | 0.97628 – 0.97945 |

## Phase D — interpretation

* **Case A (large training variance): YES.** The 3-seed ensemble improves on
  the mean 2-seed ensemble by **+0.00335 ≥ +0.002**; single-model std is
  `0.00487`.  The estimator/training variance is large relative to the
  sub-0.002 effects previously chased.
* **Case C (seed-combination dependence): minor.** All three pair ensembles
  fall in `[0.128127, 0.131222]` (spread `0.00310`); the pair benefit is
  therefore not driven by one lucky combination, though pair `(0,2)` is
  visibly weaker and `(1,2)`/`(0,1)` are similar.
* **Checkpoint averaging stabilises single models.**  The fixed Top-5 soup
  improves each seed (`+0.00353`, `+0.00123`), i.e. part of the variance is
  checkpoint-selection noise inside one trajectory, not only seed variance.
* **Regularisation is worth testing** on this basis (Phase E).

## Phase E — minimal single-model regularisation (weight decay only)

Only `weight_decay 1e-5 → 1e-4` was added; architecture, optimizer, LR,
scheduler, batch, stopping and head are identical.  seed0 was run first
(improvement `+0.004814 ≥ +0.002`), so seed1 was purchased.

| seed | canonical wd=1e-5 | wd=1e-4 | improvement | canonical train@best | wd train@best | canonical gap | wd gap |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.140609 @167 | **0.135795 @196** | **+0.004814** | 0.074466 | 0.072494 | 0.066143 | 0.063301 |
| 1 | 0.133440 @234 | **0.137158 @193** | **−0.003718** | 0.075917 | 0.079828 | 0.057523 | 0.057330 |

Paired mean effect `+0.000548`, i.e. inside the ±0.001 no-signal band, with a
**sign flip** across seeds.  The seed0 gain is not a robust regularisation
effect.

## Verdict

* **large training variance, regularization worth pursuing** — supported
  (3-seed scaling `+0.00335`; single std `0.00487`);
* **checkpoint averaging substantially stabilizes single models** — supported
  (Top-5 soup `+0.00353 / +0.00123`);
* **ensemble benefit saturates quickly** — **not** supported (`2→3` still
  `+0.00335`);
* **WD improves single-model generalization** — **not** supported;
* **WD has no meaningful signal** — **supported** (`+0.00481` on seed0,
  `−0.00372` on seed1, sign-inconsistent, paired mean `+0.00055`).

**Do not proceed to a WD sweep and do not test `3e-4`.**  The single-model
variance is real and large, but this first regularisation knob does not control
it; the next step must target variance/training dynamics more directly
(e.g. checkpoint/EMA style stabilisation or an explicit variance-reduction
mechanism), still without distillation, and each candidate must be evaluated
against this 3-seed canonical baseline.

## Provenance / files

* `experiments/luyin16/zinc_compact_v4_recurrent_variance_diagnosis.py`
* `tests/test_compact_v4_recurrent_variance_diagnosis.py` (7 pass)
* canonical-loop change: optional default-`None` `snapshot_dir` in
  `zinc_compact_v4_smallhead_e2e.train_model` (saves a weights-only state per
  epoch; no RNG use, bit-inert when unset) and pass-through in
  `zinc_compact_v4_recurrent_pair_centre.train`.
* `results/compact_v4_recurrent_variance_diagnosis/`:
  `canonical_reference.json`, `runs/`, `snapshots/`, `soup_canonical_seed0/1.json`,
  `soup_states/`, `variance_diagnosis.json`, `soup_ensemble_ceiling.json`,
  `wd_decision.json`, `final_report.json`, `logs/`.
* `official_test_loaded = false` everywhere.
