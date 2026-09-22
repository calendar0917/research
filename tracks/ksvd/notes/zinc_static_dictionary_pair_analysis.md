# Analysis — ZINC strict-static dictionary-pair v0

Round: **ZINC-strict-static-dictionary-pair-v0**.
Preregistration: `tracks/ksvd/notes/zinc_static_dictionary_pair_preregistration.md`
(frozen at commit `033c1b6`).
Status: one seed-0 full-training run per arm; round closed.

## 1. What was run

Two A100-SXM4-40GB GPUs, remote commit
`033c1b6be2a8797d30433bd1b4582a86437d47be`, official ZINC train 10000 /
official valid 1000, seed 0, inherited optimized protocol (Adam 1e-3,
wd 1e-5, batch 128, 240 epochs, patience 40, no scheduler, L1, clip 5.0,
best official-valid checkpoint).  **Official test was never loaded** in any
stage (`official_test_loaded = false` on every artifact).

Three full-training runs only: `S0 seed0` (GPU0), then `S-Dense seed0` (GPU0)
and `S-Dict seed0` (GPU1) in parallel.  No seed 1/2/3, no HPO, no rescue.

## 2. Primary metric — best official-valid MAE

| arm | best valid MAE | best epoch | epochs run | Top-5 soup | params | branch params | wall | peak GPU |
|---|---|---|---|---|---|---|---|---|
| **S0** (strict-static) | **0.145674** | 164 | 204 | 0.140794 | 66,228 | 0 | 830 s | 134 MB |
| **S-Dense** | **0.145958** | 170 | 210 | 0.142660 | 71,417 | 5,189 | 885 s | 136 MB |
| **S-Dict** | **0.136050** | 225 | 240 | 0.133829 | 71,429 | 5,201 | 1135 s | 137 MB |

Pre-registered gains (primary, best checkpoint):

```text
M0 - MK  (gain_vs_base_dict)  = +0.009624   >= 0.004
M0 - MD  (gain_vs_base_dense) = -0.000284
MD - MK  (dict_specific_gain) = +0.009908   >= 0.002
```

Pre-registered gains (corroborating fixed Top-5 soup):

```text
M0 - MK = +0.006965   >= 0.004
M0 - MD = -0.001866
MD - MK = +0.008831   >= 0.002
```

Both the primary and the soup metric satisfy the pre-registered Case-A
condition.  Recorded verdict: **A_DICT_SPECIFIC_SIGNAL** (exploratory, seed 0).

The generic dense adapter gave no gain over S0 on either metric
(`-0.000284` / `-0.001866`), so the S-Dict gain is not explained by added
generic capacity.

## 3. Strict-static integrity (core architecture contract)

All hard gates pass for every arm on real valid batches
(`results/zinc_static_dictionary_pair/integrity_gates.json`,
`runs/<arm>_seed0.json -> strict_static_contract`):

| gate | value |
|---|---|
| `center_context == False` | true (all arms) |
| `center_update is None` | true (all arms) |
| forward with `_pool_pairs_to_centres` monkeypatched to raise | succeeds |
| pair encoder calls per forward | 1 |
| relation encoder calls per forward | 1 |
| `h_i` bit-identical under pair-relation mutation | true (max abs diff 0.0) |
| prediction changes under relation mutation (non-vacuous) | true |
| pair-order invariance max abs prediction diff | ≤ 1.9e-6 |

The local state `h_i` is therefore provably independent of the pair state; the
pair is evaluated exactly once and is only pooled at graph level.  This is the
strict-static contract and it holds.

Note: the historical compact-v4 baseline (with `center_context=True`) is **not**
a strict-static model and was not reused as the S0 baseline.  The strict-static
S0 was trained here from scratch.

## 4. Initialization / parameter control

* Dense/dict add 5,189 vs 5,201 parameters (relative mismatch **0.23 %** < 1 %);
  `H = 53` was chosen programmatically.
* All 40 shared tensors (including the 6 reader tensors) are **bit-identical**
  across S0 / S-Dense / S-Dict (`max_abs_diff == 0.0`), so the only difference
  between the two candidate arms is the residual branch.
* Same optimizer / data order / seed / protocol / device regime for all arms.
* Gradient viability at step 0: dictionary / Wq / Wv and both dense MLP layers
  get strictly positive task gradient; `gamma` is finite.  The dictionary
  branch is not a dead branch at initialization.

## 5. Dictionary diagnostics (report-only)

Best checkpoint, official valid, 23,083 patches:

```text
mean assignment entropy         2.0950
effective prototype count       8.126
active prototype count          64 / 64
argmax-used prototype count     12
top-8 assignment mass           0.845
max average assignment mass     0.348
min average assignment mass     0.0014
dictionary coherence mean|cos|  0.453
dictionary coherence max|cos|   1.000
mean ||gamma * delta_i||        0.0692
mean ||h0_i||                   1.886
tau (init 0.200 -> final)       0.210
gamma (init 0.100 -> final)     0.184
```

The dictionary is genuinely used: no dead atoms, no collapse, `gamma` grew
rather than shrank, and the inference ablation is large.

Inference ablation "dictionary residual forced to zero":

```text
S-Dict : mean |prediction shift| 0.3044   max 0.9832
S-Dense: mean |prediction shift| 0.0339   max 0.1563
```

The dictionary residual is ~9x more load-bearing at inference than the matched
dense residual, which is consistent with (but does not by itself prove) a
dictionary-specific mechanism.

## 6. Honest caveats (why this is not a success claim)

1. **Single seed.** This is seed-0 exploratory evidence only.  The training-time
   valid-MAE noise is of the same order as the measured effect.
2. **Noisy valid curve / max-selection.** Per-epoch official-valid MAE is very
   noisy (e.g. S-Dict 0.1360 @225 but 0.1608 @240; S0 0.1457 @164 but 0.1665
   @170).  The primary best-checkpoint metric is a max statistic.  The fixed
   Top-5 soup (0.1338 vs 0.1408 / 0.1427) is the more robust summary and
   corroborates the direction, but the dict soup members (193–232) are all late
   and therefore correlated with the best epoch.
3. **Unequal optimizer steps.** S-Dict never early-stopped (240/240 epochs) so
   its checkpoint was selected from ~18,960 steps, whereas S0 stopped at 204
   (best 164) and S-Dense at 210 (best 170).  Matched-epoch comparison is
   mixed: at epoch 200/210 S-Dict (0.1417 / 0.1455) is clearly better than
   S-Dense (0.1541 / 0.1542), but at epoch 164 S-Dict (0.1623) is worse than
   S0 (0.1457).  The gain is concentrated in the late plateau.
4. **No statistical significance is claimed.** The two-gate decision is
   exploratory by construction.  A paired seed-1 confirmation is required
   before any stronger statement.

Given these caveats, the correct reading is: *the pre-registered Case-A gate
fires on both metrics, and the matched dense control shows no generic-capacity
gain, so the local dictionary route is worth one paired confirmation — but the
effect is single-seed and near the training-noise scale.*

## 7. Answers to the round's core questions

1. **Is S0 truly strict-static?** Yes.  `center_context=False`,
   `center_update=None`, `_pool_pairs_to_centres` never called, `h_i`
   bit-identical under pair mutation, each pair evaluated once, pair-order
   invariant.
2. **S0 seed-0 modern A100 baseline MAE?** best 0.145674 (soup 0.140794),
   66,228 params.
3. **Added parameters?** S-Dense +5,189; S-Dict +5,201 (0.23 % mismatch).
4. **Same training/GPU regime?** Yes: same commit, same A100 GPUs, identical
   inherited protocol/seed/data order, shared tensors bit-identical.
5. **`S-Dict - S-Dense`?** primary `-0.009908` (dict lower), soup `-0.008831`.
6. **Dictionary beyond generic capacity?** In this seed-0 run, yes by the
   pre-registered gate: the dense control does not beat S0, the dictionary does
   on both metrics.
7. **Is the dictionary used, not collapsed/dead?** Yes: 64/64 atoms active,
   effective count 8.13, top-8 mass 0.845, `gamma` 0.1 -> 0.184, ablation mean
   shift 0.304.
8. **Next stage?** Worth one paired seed-1 confirmation *and* the
   occurrence-level dictionary-conditioned pair kernel, under a new
   preregistration.  Not authorized by the current round.
9. **Official test loaded?** No — never, in any stage.
10. **Budget used?** Exactly 3 full-training runs, 2,850.6 s total wall clock
    (47.5 min).

## 8. Artifacts

* `results/zinc_static_dictionary_pair/analysis.json`
* `results/zinc_static_dictionary_pair/parameter_audit.json`
* `results/zinc_static_dictionary_pair/initialization_match.json`
* `results/zinc_static_dictionary_pair/integrity_gates.json`
* `results/zinc_static_dictionary_pair/runs/{s0,dense,dict}_seed0.json`
* `results/zinc_static_dictionary_pair/curves/{s0,dense,dict}_seed0_curve.csv`
* `results/zinc_static_dictionary_pair/RESULTS_SUMMARY.md`
