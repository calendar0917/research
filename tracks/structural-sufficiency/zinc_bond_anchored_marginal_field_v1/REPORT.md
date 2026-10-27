# REPORT — BAMF-v1: bond-anchored marginal field

**Protocol:** `zinc-bond-anchored-marginal-field-v1`  
**Verdict:** **STOP_NO_SIGNAL**  
**Scope:** seed 0 only (the pre-registered seed-0 gate failed); seed 1 was not
launched; Locked1k values were not accessed or scored; official valid and official test
labels were never loaded or scored. An incidental historical-file load is
documented in §6.  
**Formal commits:** frozen `f0d05a55e42e`; formal run `a04eb5591ec4`.  
**Budget:** 2.07 A100·h of the 12 A100·h cap (G_s0 1.564 h, I_s0 0.469 h,
smoke/timing/failures 0.040 h).

---

## 1. Answer in one paragraph

The operator is real: exact whole-molecule marginals differ from independent
per-edge marginals, and a distant atom's potential changes a G edge marginal
five bonds away while leaving I exactly unchanged. But the difference does not
buy prediction. On the frozen Monitor1k, G is **worse** than the matched
independent-marginal control I by **+0.003149** MAE (G 0.127660 vs I 0.124510;
paired bootstrap CI [+0.000011, +0.006245]), and **worse** than the historical
DCC-L by **+0.004509** (L 0.123151; CI [+0.001230, +0.007990]). The
any-unseen/no-penalty subgroup is also worse for G than L by 0.008552. The
seed-0 gate required G−I ≤ −0.004 and G−L ≤ +0.002; both failed. The round
stops without seed 1, without Locked scoring, and without an official-test read.

## 2. Question and boundaries

Sole hypothesis: with DCC-L's local content unchanged, molecule-consistent
latent-state inference converts long-range structural information into better
molecular property prediction than independent per-bond inference.

Boundary audit:

- Fit8k (8,000 rows) only for supervised training; Monitor1k (1,000 rows) only
  for selection and the seed-0 gate. Locked1k values were not accessed or
  scored. Official valid and test labels were never loaded or scored.
- Raw-y L1 only; no target formula, no RDKit Crippen/SA contribution, no
  artificial cycle penalty.
- 497,093 deployed parameters (single model, no ensemble/distillation/pseudo-
  labels), measured by `named_parameters()`.
- No recursive learned atom/bond hidden-state message passing. The G arm is
  exact junction-tree variable elimination, explicitly labeled **algorithmic
  global inference**.
- No grid search over C, S, potential amplitude, width, LR, or steps.

## 3. What was built

DCC-L is copied tensor-for-tensor into a `BAMFModel`. The only architectural
change is that `local_composer.lins[0]` is widened from 296 to 308 inputs; the
first 296 columns are the frozen DCC-L initial columns and the last 12 are zero.
Two shared potential heads are added:

```
log_phi_v = tanh(Linear(96,8)(a_v)).reshape(4,2)
log_psi_e = symmetric 2x2 from tanh(Linear(96,12)(b_e)).reshape(4,3)
```

Parameter audit: 494,193 + 776 + 1,164 + 960 = **497,093**.

G computes exact edge marginals by min-fill junction-tree elimination; I
computes each edge's 2x2 marginal from the same `log_phi`/`log_psi` with a
per-edge softmax. Both use the identical 12-dimensional feature

```
m_e = [ mu_u+mu_v-1, |mu_u-mu_v|, 4(mu_uv - mu_u mu_v) ]   # 4 channels
```

so the comparison is not "more features" or "more parameters"; it is global
consistency vs independent edges with the same potentials.

## 4. Phase-0 CPU evidence

Artifact: `analysis/phase0_cpu.json`; tests:
`uv run pytest -q .../tests` → **14 passed**.

- Induced width was **recomputed** by min-fill on all 10,000 official train
  graphs, not cited: max 3, counts `[0,55,9906,39]` for widths 0..3,
  `smax=4`, max clique size 4, 231,664 cliques.
- float64 dense `2^n` enumeration (n=2..8; trees, rings, dense, disconnected)
  matches logZ, node marginals, and edge marginals to ≤1.8e-15.
- batch = single, empty/single-bond/tree/ring/dense/disconnected all pass;
  atom renumbering, endpoint swap, and bond permutation agree to ≤4.5e-16.
- potentials scaled ±100 remain finite and normalized.
- finite-difference first derivative and second-order autograd through the
  elimination pass; no detach/SFC-v0 `log_q` bug.
- train/eval forward is identical (0.0).
- G and I initial states are bit-identical (`fe4e98c3…` at seed 0); at step 0
  both reproduce the trained DCC-L checkpoint to 4.8e-7.
- after the two zero-init gates (reader's new columns and local composer's new
  columns), the potential heads first receive non-zero gradient on the third
  forward (0-based step 2); recorded in `phase0_cpu.json`.
- six-node chain: changing only `phi[5]` changes G's edge (0,1) marginal by
  **2.88e-5**; I's change is **exactly 0**. The mechanism is active.
- real Fit8k diagnostic (1,024 molecules, 25,718 bonds): mean |G−I| feature
  difference 0.0459, 99.8% of entries >1e-6. A molecule-disjoint ridge probe
  predicts G features from the 296-dim DCC local input with mean test R² 0.791
  and G−I with mean R² 0.525. **Redundancy is substantial**, and this warning
  was carried into the result rather than used to retune C/S/amplitude.

## 5. GPU gate and cost

`rr doctor res` was green (10 ok / 0 fail); jobs ran on `res-gpu0` / `res-gpu1`
with one job per GPU. The 3-epoch full-fit profile projected G at 1.884 A100·h
and I at 0.599 A100·h, both below the 2.5 h per-G-run gate. Actual formal costs:

| run | wall (h) | projected (h) | best Monitor MAE | best epoch/step |
|---|---:|---:|---:|---|
| G_s0 | 1.564 | 1.884 | 0.127660 | 528 / 33,264 |
| I_s0 | 0.469 | 0.599 | 0.124510 | 556 / 35,028 |

Total for all BAMF jobs, including smoke, timing, and three instrumentation
failures: **2.073 A100·h**. Failure cost was 0.0072 h; the failures were
`torch.cuda.reset_peak_memory_stats(device)` and a missing `plan=None` guard in
the timing wrapper, not model/training failures. No formal update was lost.

The G/I microbenchmark measured G forward+backward at ~3.8–5.4× the I cost on
one 128-row batch. The exact elimination is therefore not free, but the formal
G run remained under the per-run gate.

## 6. Seed-0 result (Monitor1k only)

All numbers are MAE on the frozen Monitor1k rows; lower is better. L is the
matched historical DCC-L seed-0 checkpoint, re-evaluated on the same rows
(frozen predictions reproduce to 1.9e-6).

| model | best Monitor | W20 | T100 | final epoch |
|---|---:|---:|---:|---:|
| **G_s0** | **0.127660** | 0.128167 | 0.128187 | 0.128205 |
| **I_s0** | **0.124510** | 0.125224 | 0.125263 | 0.125268 |
| L_s0 | 0.123151 | 0.123819 | 0.123834 | 0.123827 |

Paired seed-0 contrasts:

| contrast | ΔMAE (A−B) | 95% paired bootstrap CI | interpretation |
|---|---:|---|---|
| G_s0 − I_s0 | **+0.003149** | [+0.000011, +0.006245] | G worse; gate ≤−0.004 fails |
| G_s0 − L_s0 | **+0.004509** | [+0.001230, +0.007990] | G worse; gate ≤+0.002 fails |
| I_s0 − L_s0 | +0.001359 | [−0.001778, +0.004571] | I not better; CI spans 0 |
| G_s0 − L_s0, any-unseen/no-penalty (n=301) | **+0.008552** | [+0.000219, +0.016434] | G worse |
| G_s0 − I_s0, any-unseen/no-penalty (n=301) | **+0.004003** | [−0.002615, +0.010733] | G worse, CI spans 0 |
| G_s0 − L_s0, no-penalty (n=964) | +0.003453 | [+0.000386, +0.006460] | G worse |
| G_s0 − L_s0, penalty (n=36) | +0.032797 | [−0.007806, +0.084592] | noisy, also worse |

Robustness: after dropping the 10 rows with the largest |AE_G−AE_L|, the
G−L mean difference is still **+0.003890** (G worse). The failure is not caused
by a handful of rows or by penalty rows.

**Audit note on the L baseline.** The final L Monitor predictions above were
regenerated by evaluating `zinc_dccl_bottleneck_v1/runs/fit8k_s0/best_monitor.pt`
directly; only Fit8k-trained weights are in that checkpoint. The historical
`predictions_best_monitor.pt` file also stores `locked_pred` and
`official_valid_pred` arrays. That file was loaded during at least one earlier analysis
pass to extract only `monitor_pred`; no locked or official-valid array was
inspected, summarized, or used in any gate, metric, or decision. The final
analysis script does not open that file. This is recorded as an incidental
exposure so the audit is unambiguous.

## 7. Pre-registered gate outcome

| seed-0 gate | threshold | observed | result |
|---|---:|---:|---|
| G_s0 − I_s0 | ≤ −0.004 | +0.003149 | **FAIL** |
| G_s0 − L_s0 | ≤ +0.002 | +0.004509 | **FAIL** |
| any-unseen/no-penalty vs L (suggested) | ≥ +0.005 improvement | −0.008552 | **FAIL** |

**STOP_NO_SIGNAL.** Seed 1 was not launched, no Locked1k value was accessed or
scored, and no official test was constructed. The final Monitor-side gate artifact is
`analysis/monitor_gates.json`.

## 8. Required answers

**(A) Does the operator truly produce global conditional information different
from existing DCC local moments?**
Yes at the mechanism level: exact G marginals change when a distant atom's
potential changes, while I cannot; the real Fit8k G−I difference is non-zero on
99.8% of feature entries; G and I are otherwise identical at step 0. But the
information is substantially redundant with DCC's local input (mean test R²
0.791 for G features, 0.525 for G−I). The mechanism is real, not automatically
useful.

**(B) Does it improve unseen-structure prediction at fixed 10k labels, rather
than just adding compute?**
No. G is worse than the matched I control by 0.00315 and worse than DCC-L by
0.00451 on Monitor1k; even the independent-feature control I is 0.00136 worse
than L (CI spans 0). On the any-unseen/no-penalty subgroup G is worse than L by
0.00855. The extra computation did not convert into sample efficiency.

**(C) Is the gain vs the strongest DCC-L large enough to support a next-
generation non-recursive structural computation, rather than another
0.001–0.003 fluctuation?**
No. The point estimate has the wrong sign, is outside the pre-registered
+0.002 floor, and is not a small positive fluctuation. There is no basis for a
Phase-2 recommendation, no extra experiment menu, and no test read.

## 9. What is and is not concluded

- Concluded: this exact operator, at 497,093 parameters and this frozen recipe,
  does not improve over DCC-L or over independent per-edge marginals on the
  frozen Monitor split. The global mechanism is active but not performance-
  relevant here.
- Not concluded: that all global inference is useless, that a different
  optimization schedule could not train the potential heads, or that a
  different factorization could not help. Those would require a new
  pre-registration; none is proposed or started here.
- The Phase-0 redundancy warning (G features largely predictable from DCC local
  input) is consistent with the negative result, but the pre-registered test was
  run as frozen; C/S/amplitude/width/LR/steps were not changed after seeing it.

## 10. Artifacts and audit

| item | path / value |
|---|---|
| protocol | `PROTOCOL.md` |
| design | `DESIGN.md` |
| frozen config | `configs/bamf_v1.json` |
| Phase-0 CPU | `analysis/phase0_cpu.json` (PASS) |
| tests | `tests/test_bamf_inference.py` (14 passed) |
| timing | `analysis/timing_G.json`, `analysis/timing_I.json` |
| Monitor gate | `analysis/monitor_gates.json` |
| frozen seed-0 predictions | `analysis/predictions_seed0_monitor.csv`, sha256 `994f1204…` |
| frozen seed-0 hashes | `analysis/frozen_seed0.json` |
| run records | `analysis/run_records.json` |
| G_s0 best state | sha256 `d3743fb9…` (file `5c826a90…`) |
| I_s0 best state | sha256 `b3e5058a…` (file `3ac88255…`) |
| L_s0 checkpoint | file sha256 `6a145dc6…`; state `21dead3e…` |
| G_s0 | `runs/G_s0/` (`train.json`, curve, best/final checkpoints, predictions) |
| I_s0 | `runs/I_s0/` |
| plan | max width 3, `smax=4`, 231,664 cliques, sha256 `58deff19…` |
| G/I initial state | `fe4e98c31b1f02021a7f680a066fc80140d40289b0a0b30e6748d8795490c340` |
| formal run commit | `a04eb5591ec42e0cde571c4999bb9b4044dff2bf` |

`rr` run ids: `bamf-smoke-g-…54a45010`, `bamf-smoke-i-…1492d0f4`,
`bamf-timing-g2-…043f22b3`, `bamf-timing-i3-…37a31aa0`,
`bamf-G-s0-…1ecb9231`, `bamf-I-s0-…938e4250`; three instrumentation failures
are listed in `analysis/run_records.json`.

Environment: `res` / `a100-2`, driver 550.163.01, torch 2.5.1+cu124, CUDA
12.4, Python 3.12.14, process backend, `res-gpu0`/`res-gpu1`.

**Final status: STOP_NO_SIGNAL. No seed 1; no Locked values accessed; no official test.**
