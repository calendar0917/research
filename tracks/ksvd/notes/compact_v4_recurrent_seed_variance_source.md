# Compact-v4 T=2 recurrent Q16 — seed-variance source decomposition

Date: 2026-09-14
Protocol: `compact_v4_recurrent_seed_variance_source_v1`
Canonical model: compact-v4-smallhead T=2 weight-tied recurrent pair–centre,
h=48, q=16, **82,115 params**.
Official test: **never loaded.**
No distillation, no new architecture, no T=3, no Q24/Q32, no pair-to-pair,
no LR/scheduler/dropout/batch sweep, no EMA/SWA, no Huber/SmoothL1, no
ensemble-weight search, no new soup rule, no WD sweep.
All training `torch.set_num_threads(4)` on CPU; every run sequential unless
explicitly noted.

## Question

Task 1 showed a large single-model training variance (3-seed std `0.00487`)
and that fixed Top-5 checkpoint averaging stabilises it.  This task asks a
follow-up question: **is the seed variance driven by the initialisation or by
the mini-batch/data order?**  The canonical seeds 0/1/2 all use
`init_seed == data_shuffle_seed`, so the two factors are perfectly confounded
in the canonical runs; a 2×2 factorial is the minimal design that separates
them.

Cells (`init_seed`, `shuffle_seed`):

| cell | init | shuffle | role |
|---|:---:|:---:|---|
| A | 0 | 0 | canonical seed0 |
| B | 0 | 1 | crossed |
| C | 1 | 0 | crossed |
| D | 1 | 1 | canonical seed1 |

## Phase A — fixed Top-5 soup is a reproducible stabilizer

Seed 2's soup was the only new Phase-A run; seeds 0/1 were already computed in
Task 1.  The repo's locked Top-5 rule is reused verbatim (five lowest
selection-MAE checkpoints, ties → earliest epoch, equal-weight arithmetic mean
of the `state_dict`s; no `k`/weight search).

| seed | best checkpoint | top-5 epochs | top-5 soup | gain |
|---:|---:|---|---:|---:|
| 0 | 0.140609 @167 | 167,168,186,164,144 | 0.137078 | +0.003532 |
| 1 | 0.133440 @234 | 234,227,202,217,169 | 0.132210 | +0.001230 |
| 2 | 0.142722 @94 | 94,130,120,124,95 | 0.138062 | +0.004661 |

mean `+0.003141`, std `0.001427`, min `+0.001230`, **all three positive** →
`reproducible_stabilizer = true`.  Checkpoint averaging is a robust
single-model stabiliser, not a seed-0 accident.

## Critical finding — the canonical loop is not process-to-process reproducible

The determinism sanity ran the *same* cell (init0, shuffle1) twice as two
separate 61-epoch sequential processes:

| rep | best valid | best epoch |
|---:|---:|---:|
| 1 | 0.171080 | 57 |
| 2 | 0.174371 | 59 |

The two curves are **bit-identical through epoch 26**, then the valid MAE
diverges by `~5e-10` and amplifies chaotically (by epoch 40 train MAE differs
`0.176187` vs `0.160306`; `max_valid_mae_diff = 0.128`).  A forward-only probe
(eval loader and optimiser frozen from the canonical seed0 state) is
bit-identical across processes, so the nondeterminism is in the
**backward/optimiser update**, at an `index_add`-family op.

Consequences:

1. Part of the canonical "seed variance" is **irreducible execution
   nondeterminism of order `~0.003`**, comparable to the effects being
   measured.  A single canonical factorial would be contaminated.
2. Setting `torch.use_deterministic_algorithms(True)` makes the loop
   **bit-reproducible and concurrency-safe** (two concurrent 25-epoch runs of
   the same cell both give `0.21432807221828262`, equal to a sequential run),
   enabling parallel execution.  It changes the numerics, however (epoch-1
   train `1.028089` vs canonical `1.021271`), so absolute values shift.

**Design decision.** The 2×2 factorial was therefore run in the reproducible
deterministic mode, re-running **all four cells** for internal consistency.
The two canonical matched cells A/D were also re-run in the canonical mode to
provide a cross-regime check.  This is an environment/mode change only; no
model, optimiser, loss, schedule or data change was made.

## Phase B — 2×2 factorial

### Deterministic regime (reproducible, primary)

| cell | init | shuffle | valid MAE | best epoch | train@best |
|---|:---:|:---:|---:|---:|---:|
| A | 0 | 0 | 0.146289 | 148 | 0.083762 |
| B | 0 | 1 | 0.142657 | 143 | 0.084671 |
| C | 1 | 0 | 0.142101 | 133 | 0.079950 |
| D | 1 | 1 | **0.139254** | 193 | 0.078178 |

### Canonical regime (A/D reproduced; B/C single noisy samples)

| cell | init | shuffle | valid MAE | best epoch | train@best |
|---|:---:|:---:|---:|---:|---:|
| A | 0 | 0 | 0.140609 | 167 | 0.074466 |
| B | 0 | 1 | 0.142722 | 213 | 0.069374 |
| C | 1 | 0 | 0.138973 | 187 | 0.073541 |
| D | 1 | 1 | **0.133440** | 234 | 0.075917 |

## Phase C — decomposition

### Deterministic regime

| effect | value |
|---|---:|
| init @ shuffle0 (`|A-C|`) | 0.004189 |
| init @ shuffle1 (`|B-D|`) | 0.003403 |
| **mean init effect** | **0.003796** |
| shuffle @ init0 (`|A-B|`) | 0.003632 |
| shuffle @ init1 (`|C-D|`) | 0.002847 |
| **mean shuffle effect** | **0.003239** |
| main init / main shuffle | 0.003796 / 0.003239 |
| interaction `(D-C)-(B-A)` | **+0.000785** (< 0.002 gate) |
| A−D gap | 0.007035 |

### Canonical regime

| effect | value |
|---|---:|
| init @ shuffle0 (`|A-C|`) | 0.001636 |
| init @ shuffle1 (`|B-D|`) | 0.009282 |
| **mean init effect** | **0.005459** |
| shuffle @ init0 (`|A-B|`) | 0.002113 (sign flips) |
| shuffle @ init1 (`|C-D|`) | 0.005533 |
| **mean shuffle effect** | **0.003823** |
| main init / main shuffle | 0.005459 / 0.001710 |
| interaction `(D-C)-(B-A)` | **−0.007646** (≥ 0.002 gate) |
| A−D gap | 0.007169 |

The A−D gap is the one invariant that survives the regime change:
`0.007169` canonical vs `0.007035` deterministic (difference `0.000134`).
The seed0↔seed1 difference is therefore real, but its *internal split* is
regime-dependent.

### Prediction disagreement / residual correlation

Deterministic (mean abs disagreement / residual corr):

| pair | kind | disagree | resid corr |
|---|---|---:|---:|
| A-C | init | 0.094273 | 0.978229 |
| B-D | init | 0.103634 | 0.974329 |
| A-B | shuffle | 0.088405 | 0.980927 |
| C-D | shuffle | 0.089667 | 0.984141 |
| A-D | diagonal | 0.094217 | 0.980823 |
| B-C | diagonal | 0.093510 | 0.972731 |

Canonical:

| pair | kind | disagree | resid corr |
|---|---|---:|---:|
| A-C | init | 0.088028 | 0.981939 |
| B-D | init | 0.091674 | 0.970983 |
| A-B | shuffle | 0.082834 | 0.983994 |
| C-D | shuffle | 0.089576 | 0.980631 |
| A-D | diagonal | 0.089008 | 0.979447 |
| B-C | diagonal | 0.085438 | 0.978228 |

Predictions stay highly correlated (`0.9962–0.9978` deterministic,
`0.9962–0.9976` canonical) and residual correlations are high
(`0.971–0.984`) in both regimes: the cells differ by a small, largely shared
error component, consistent with Task 1.

## Phase D — interpretation

* **Initialization contributes:** mean init effect `0.00380`
  (deterministic, ≥ gate) / `0.00546` (canonical).  Supported.
* **Data-shuffle contributes:** mean shuffle effect `0.00324`
  (deterministic, ≥ gate) / `0.00382` (canonical); the canonical *main*
  shuffle effect is smaller (`0.00171`) because its sign flips with init.
  Supported.
* **No single factor dominates in the reproducible regime.** In deterministic
  mode init and shuffle are comparable (`0.00380` vs `0.00324`, difference
  `0.00056` < gate) with a negligible interaction (`0.00079`).  The A−D gap is
  essentially `init + shuffle`.
* **Strong init×shuffle interaction in the canonical regime.** Interaction
  `0.00765` ≥ gate; `shuffle1` helps only when paired with `init1`
  (D) and hurts at `init0` (B), so the canonical factorial cannot be reduced
  to either factor alone.
* **Irreducible execution nondeterminism is a third source** of order `~0.003`
  and is large enough to flip the apparent interaction.

**Case (interpretation gate): `interaction`** — *strong init×shuffle
interaction in the canonical 4-thread regime; both contribute additively in
the reproducible deterministic regime.*  The deterministic control case is
`both`.

## Verdict

* **init vs shuffle:** **both** materially contribute; in the reproducible
  deterministic regime they are comparable (`~0.0038` vs `~0.0032`) and
  additive, while the canonical 4-thread regime adds a strong
  init×shuffle interaction (`0.0076`).  Attributing the seed variance to a
  single source is **not** supported.
* **A−D (seed0 vs seed1) gap is regime-robust** (`0.0072` in both), but its
  decomposition is not.
* **Top-5 soup is a reproducible single-model stabilizer** — all three seed
  gains positive (mean `+0.00314`, min `+0.00123`).
* **The canonical training loop is not bit-reproducible** (`~0.003` execution
  noise), so no sub-`0.003` single-run effect in this protocol is trustworthy
  without either repeating runs or using deterministic mode.

## Next step (single direction, not executed)

**Trajectory stabilisation rather than further random-seed decomposition.**
The variance is not localised to one controllable factor (init or shuffle),
and the canonical loop itself injects execution noise; the direction with the
largest expected return remains single-model checkpoint/EMA-style trajectory
averaging (already shown to help: soup `+0.00314`), evaluated against the
3-seed canonical baseline.  Do not run a WD sweep, do not test `3e-4`, and do
not treat the canonical 2×2 interaction as a search target.

## Provenance / files

* `experiments/luyin16/zinc_compact_v4_recurrent_seed_variance_source.py`
* `tests/test_compact_v4_recurrent_seed_variance_source.py` (9 pass;
  full suite 787 pass)
* `results/compact_v4_recurrent_seed_variance_source/`:
  `soup_summary.json`, `determinism_sanity.json`,
  `determinism_sanity_deterministic.json`, `cell_{A,B,C,D}.json`,
  `cell_canonical_{B,C}.json`, `decomposition.json`, `decision.json`,
  `final_report.json`, `logs/`.
* canonical states reused/created under
  `results/compact_v4_recurrent_variance_diagnosis/states/`.
* `data_seed` added as a default-`None` pass-through
  (`shead.train_model`, `rec.train`, `vd.run_training`) controlling only the
  two DataLoader generator seeds; `init_seed`/`data_seed` and
  `deterministic_algorithms` are recorded in every run summary.
* `official_test_loaded = false` everywhere.
