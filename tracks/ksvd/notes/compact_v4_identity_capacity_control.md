# ZINC cell-A identity / capacity control (A0 learned lookup vs A1 fixed code vs A2 no identity)

Protocol: `compact_v4_identity_capacity_control_v1`
Branch: `exp/identity-capacity-control` (isolated worktree, frozen base commit `3dd0731`)
Official test: **never loaded** (`official_test_loaded = false` everywhere).
Date: 2026-09-15. Hardware: 2x A100-SXM4-40GB (formal runs on GPU1, `torch.use_deterministic_algorithms(True)`).

## 1. Question

The frozen ZINC **cell A** (85,763 params, h=64, q=16, patch_encoder_hidden=64,
global_encoder_hidden=32, T=2 weight-tied recurrent pair--centre) reaches a 2-seed
fixed Top-5 soup valid MAE of **0.126368**. Does that number actually depend on the
**learned per-identity typed lookup** (`typed_embedding`, 36,420 params), or only on
**identity distinguishability / generic shared capacity**? Three parameter-matched
conditions, one variable at a time:

| condition | token channel | identity info |
|---|---|---|
| **A0** | frozen learned hybrid typed lookup (reference, **not** retrained) | per-token learned embedding |
| **A1** | deterministic seed-independent fixed 16-D code + token-shared trainable adapter | fixed distinguishable code |
| **A2** | identical adapter, identity slot forced to exact zeros | none |

## 2. Parameter accounting and fairness

`results/compact_v4_identity_capacity_control/parameter_accounting.json`

| | A0 | A1 | A2 |
|---|---|---|---|
| total | **85,763** | **85,740** | **85,740** |
| backbone | 81,212 | 81,189 | 81,189 |
| head | 4,551 | 4,551 | 4,551 |
| typed lookup | 36,420 | 0 | 0 |
| shared adapter | 0 | 36,397 | 36,397 |
| unified_graph_width | 334 | 334 | 334 |

- `typed_lookup_released_params` = 36,420 reinvested into a **token-shared** adapter of
  36,397 params (Linear(162,201) -> LayerNorm(201) -> ReLU -> Dropout -> Linear(201,16);
  input = [16-D identity slot ; 146-D shell `patch_cont`]).
- A1/A2 are **23 params below** A0 (0.027%), well inside the +/-1% fairness band.
  `a1_a2_total_identical = true`, `a1_a2_adapter_identical = true`.
- The released budget stays in the **token channel**; the graph head is untouched (4,551).
- A1 vs A2 differ in exactly **one** thing: the identity slot is a fixed unit-norm code (A1)
  or exact zeros (A2). Architecture/params/optimizer identical.

## 3. Determinism, provenance, sanity

- `baseline_guard.json`: stored A0 `cell_A_seed0` checkpoint forwarded through the frozen builder
  reproduces the frozen reference. best 0.12970966223819413 vs ref 0.12970963285310427
  (`best_abs_diff = 2.94e-08`); soup 0.12470435226900736 vs ref 0.12470435495121637
  (`soup_abs_diff = 2.68e-09`); `reproduced = true`.
- `repro` smoke on GPU1, 6 epochs, deterministic: `selection_state_sha256 =
  eea84e7ad3481882835170cc95190cce1e7a6eca928b7fcd52a987a97dc0d74d` — **bit-identical** across
  two runs; `peak_gpu_memory_mb = 310` (no OOM risk even on a co-tenant GPU).
- `sanity.json`: **25/25 checks pass**. 43 shared tensors **bit-identical** to A0
  (`A1/A2_shared_max_abs_diff_vs_A0 = 0.0`); A1/A2 adapters bit-identical
  (`A1_A2_adapter_max_abs_diff = 0.0`); parent embedding unchanged; forward/backward finite;
  adapter + parent gradients alive; recurrence flags/refresh and weight tying intact
  (`pair_projection: 4, pair_encoder: 2, center_update: 2`).
- Fixed code (`splitmix64`, seed 20260915): 6,785 rows x 16-D, unit norm, **0 exact duplicate
  rows**, |cos| mean 0.2017 / max 0.8842, OOV row fixed; `no grad`, seed-independent.
- The single-variable control is verified behaviourally: permuting typed tokens changes A1 output
  (`max_abs_diff = 7.8e-04`) but **not** A2 (`max_abs_diff = 0.0`).

## 4. Results

Formal runs: 4 detached runs, one condition/seed each, `--device cuda --deterministic`.
Wall clock 3,436-3,760 s each; peak GPU memory 316-318 MB; epochs 211-240.

### Fixed Top-5 soup valid MAE (primary)

| condition | seed0 | seed1 | **2-seed mean** |
|---|---|---|---|
| A0 (frozen) | 0.124704 | 0.128032 | **0.126368** |
| A1 | 0.121245 | 0.126014 | **0.123629** |
| A2 | 0.121694 | 0.122134 | **0.121914** |

### Best-checkpoint raw valid MAE (secondary)

| condition | seed0 | seed1 | 2-seed mean |
|---|---|---|---|
| A0 (frozen) | 0.129710 | 0.131415 | 0.130562 |
| A1 | 0.127640 | 0.130926 | 0.129283 |
| A2 | 0.125304 | 0.124030 | 0.124667 |

### Deltas vs A0 (soup, negative = better than A0)

| delta | value | vs threshold 0.002 |
|---|---|---|
| A1 - A0 | **-0.002739** | beyond threshold (A1 better) |
| A2 - A0 | **-0.004454** | beyond threshold (A2 better) |
| A2 - A1 | **-0.001715** | within threshold (A1 ~= A2) |

Every one of the 4 per-seed deltas is negative (A1/A2 better than A0 on **both** seeds).
A1/A2 best epochs (199/206 and 210/171) are later than A0's (161/153), so the shared
adapter is not simply under-trained.

### Paired per-molecule bootstrap (2-seed-averaged soup predictions, B=2000, seed 20260917)

| delta | mean | 95% CI | P(first better) |
|---|---|---|---|
| A1 - A0 | -0.003461 | [-0.011143, +0.005813] | 0.790 |
| A2 - A0 | -0.007410 | [-0.014777, +0.000714] | 0.963 |
| A2 - A1 | -0.003949 | [-0.008977, +0.000751] | 0.951 |

The per-molecule CIs include 0 (A1-A0 clearly; A2-A0 and A2-A1 on the boundary), so this is a
**"no evidence the learned lookup is needed"** result, not a proof that the shared adapter is
better. The direction is consistent across seeds and molecules.

## 5. Diagnostics

`results/compact_v4_identity_capacity_control/diagnostics.json`

- Adapter output norm 3.42-3.76 (alive, non-collapsed) for all A1/A2 seeds.
- Centre-state effective ranks (participation) h1 ~26-35, h2 ~19-29 — no rank collapse from
  dropping the lookup.
- Soup prediction disagreement: A1 seed0-vs-seed1 0.0745, A2 0.0805, A1-vs-A2 0.0729-0.0784.
- Common-vs-rare valid MAE gap (per-molecule mean train occurrence quartiles) stays negative
  (common tokens easier) for A1/A2: -0.033 to -0.053, i.e. rarity structure is **not** carried
  by the lookup alone.

## 6. Verdict

**Case III — typed lookup largely dispensable; the capacity-matched shared adapter is not worse
than the learned lookup (A1 and A2 match or beat A0, and A1 ~= A2).**

- Per-identity **learned embedding memory is not required** by cell A: A0 is the worst point
  estimate and is beaten on both seeds by both replacements.
- **Identity distinguishability adds nothing**: A2 (all-zero identity slot) ~= A1 (fixed code),
  delta -0.001715 within the 0.002 threshold; and A2 has the best 2-seed mean.
- Generic shared capacity suffices: 36,397 token-shared params processing the 146-D shell
  descriptor are enough to match/beat the 36,420-param learned table.

Point estimates even favour the shared adapter, but with only 2 seeds and per-molecule CIs that
include 0 this note does **not** claim a win — only that the learned lookup is not needed.

## 7. What this does NOT claim / do-not-reopen

- No claim about the **official ZINC test** (locked, never loaded).
- No claim that the shared adapter is *better* (bootstrap CI includes 0; 2 seeds; per-seed spreads
  0.0004-0.0048 are comparable to the deltas).
- No claim about rare-token donors, KNN embedding sharing, corrected tokenizer, v6 attribute
  branch, SBCI, attention, wider h/q/T, or a new optimizer sweep — none were touched.
- The A0 arm is the frozen 2026-09-14 reference, not a re-trained baseline; the baseline guard
  reproduces the stored checkpoint's forward to 3e-08 under the deterministic regime.

## 8. Reproduction

```bash
# local (isolated worktree)
cd /home/calendar/code/research-identity
PYTHONPATH=$PWD /home/calendar/code/research/.venv/bin/python -m \
  tracks.ksvd.experiments.luyin16.zinc_compact_v4_identity_capacity_control <stage>

# formal runs were launched detached on remote GPU1
RESEARCH_REMOTE_REPO=/home/hxy/cy/research_identity bash \
  .../remote-research-runner/scripts/launch_remote.sh 1 ic-a1s0 \
  python -m tracks.ksvd.experiments.luyin16.zinc_compact_v4_identity_capacity_control \
  train --condition A1 --seed 0 --device cuda --deterministic
```

Artifacts: `results/compact_v4_identity_capacity_control/` (parameter_accounting, sanity,
baseline_guard, diagnostics, decision, report, `runs/`, `soup_identity_*`, `states/`, `curves/`,
`snapshots/`) + `tests/test_compact_v4_identity_capacity_control.py` (5 pass).
