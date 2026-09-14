# Compact-v4 T=2 recurrent pair--centre — ZINC 2×2 capacity decomposition (GPU/A100)

Date: 2026-09-14
Protocol: `compact_v4_recurrent_pair_centre_capacity_decomposition_v1`
Commit: `481a7fa` (code) + runs on the deterministic GPU regime
Official ZINC test: **never loaded**.

## Question

The CPU H64→H96 gain (`+0.007587` Top-5 soup) moved three widths together.
This workstream separates them under a single committed GPU implementation:

```text
cell   centre h   patch-encoder hidden   global-encoder hidden   params
A         64              64                      32              85,763  (== frozen H64)
B         96              64                      32              93,059
C         64              96                      48              94,899
D         96              96                      48             103,219  (== frozen H96)
```

`q_dim=16`, `T=2` weight-tied refresh, `center_context_hidden=60`, relation/pair
machinery, readout semantics, small raw head, and the optimizer/protocol are
identical across cells.  No module is added: the only code change is an explicit
``patch_encoder_hidden`` / ``global_encoder_hidden`` override on
``PatchPathModel`` whose defaults preserve the frozen derivation exactly.

## GPU execution regime (pre-registered sanity)

CUDA is a new regime and the model pools with ``index_add_``, which is
atomic/non-deterministic on CUDA.  A short same-seed sanity (cell A, 6 epochs)
showed:

| mode | GPU0 run | GPU0 repeat | GPU1 run | selection-state hash |
|---|---|---:|---:|---|
| default CUDA | 0.3183 | 0.3398 | 0.2851 | all different |
| deterministic CUDA | 0.313497 | 0.313497 | 0.313497 | all identical |

Non-deterministic mode drifts run-to-run by far more than the effects of
interest; **deterministic mode** (`torch.use_deterministic_algorithms(True)`,
`CUBLAS_WORKSPACE_CONFIG=:4096:8`) is bit-identical across both GPUs, repeats,
and concurrent execution.  All formal runs use that frozen regime.

## Results (2 seeds per cell)

| cell | h | pe | ge | params | raw s0 | raw s1 | raw mean | soup s0 | soup s1 | **soup mean** | best ep s0/s1 | epoch time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| **A** | 64 | 64 | 32 | 85,763 | 0.129710 | 0.131415 | 0.130562 | 0.124704 | 0.128032 | **0.126368** | 161/153 | 8.7–8.8 s |
| B | 96 | 64 | 32 | 93,059 | 0.136037 | 0.131039 | 0.133538 | 0.134394 | 0.129339 | 0.131867 | 191/238 | 8.6–8.7 s |
| C | 64 | 96 | 48 | 94,899 | 0.132882 | 0.135817 | 0.134349 | 0.130973 | 0.131563 | 0.131268 | 223/148 | 9.7 s |
| D | 96 | 96 | 48 | 103,219 | 0.135610 | 0.125340 | 0.130475 | 0.132481 | 0.122750 | 0.127615 | 150/240 | 9.7 s |

All 8 runs reached the canonical patience stop (190–240 epochs).  Fixed Top-5
soup: lowest selection-MAE 5 checkpoints, earliest-epoch ties, equal-weight
parameter average (unchanged rule).

## Main effects and interaction

Defined so that a **positive** value means the larger-capacity cell is better
(`MAE(smaller) - MAE(larger)` on the 2-seed soup mean):

| effect | value | reading |
|---|---:|---|
| centre width, narrow encoders (A→B) | **−0.005499** | widening hurts |
| centre width, wide encoders (C→D) | **+0.003652** | widening helps |
| **centre main effect** | **−0.000923** | ~0 |
| encoder capacity, narrow centre (A→C) | **−0.004900** | widening hurts |
| encoder capacity, wide centre (B→D) | **+0.004251** | widening helps |
| **encoder main effect** | **−0.000324** | ~0 |
| **interaction** (D−C−B+A) | **−0.009151** | strong |

Verdict: `NEITHER_AXIS` (both main effects `< 0.001`), best cell **A**.

## Centre-state effective rank (participation ratio, no dead dims)

| cell | `h1` seed0/seed1 | `h2` seed0/seed1 |
|---|---|---|
| A (h64) | 32.4 / 26.5 | 27.0 / 20.4 |
| B (h96) | 45.5 / 39.3 | 35.8 / 26.8 |
| C (h64) | 30.9 / 27.8 | 17.5 / 30.9 |
| D (h96) | 44.1 / 42.6 | 28.7 / 28.9 |

The wider centre state is used (rank grows with `h`), but the used *fraction*
falls; extra rank does not convert into a stable validation gain.

## Verdict

* **No independent capacity axis is supported on the GPU regime.**  Both main
  effects are within `±0.001`; the only large term is a negative interaction.
* **The smallest cell A (85,763 params) has the best 2-seed soup mean
  (0.126368).**  Full H96 (D) is 0.00125 worse and within seed noise.
* **The historical CPU H64→H96 gain does not reproduce here.**  On CPU,
  H64 soup 0.130828 → H96 0.123241 (+0.007587).  On deterministic GPU, the
  same-code A→D moves 0.126368 → 0.127615 (−0.001247).  GPU A improves over the
  CPU H64 checkpoint and GPU D is worse than the CPU H96 checkpoint: this is a
  regime shift, not an architecture claim, and the `<0.002` differences are not
  interpreted as architecture effects.
* Per-seed soup spreads (0.0006–0.0097) exceed every 2x2 effect; 2 seeds are
  not enough to resolve sub-0.002 structure.
* **Parameter-allocation judgement:** the ~86k cell A is the configuration to
  keep.  Widening either the centre alone or the encoders alone hurts; widening
  both only recovers part of that loss.  There is no evidence that spending the
  ~17k extra parameters of H96 buys a real ZINC gain once the execution regime
  is fixed.

Stop rule honoured: no H112/H128, no q-width, no new module, no optimizer sweep,
official test never opened.

## Files

- `experiments/luyin16/zinc_compact_v4_recurrent_pair_centre_capacity_decomposition.py`
- `tests/ksvd/tests/test_compact_v4_recurrent_pair_centre_capacity_decomposition.py` (7 pass)
- `results/compact_v4_recurrent_pair_centre_capacity_decomposition/`:
  `parameter_accounting.json`, `sanity.json`, `runs/`, `soup_cell_*_seed*.json`,
  `diagnostics.json`, `decision.json`, `repro_*`, `states/`, `snapshots/`.
