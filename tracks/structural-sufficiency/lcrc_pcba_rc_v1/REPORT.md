# REPORT — LCRC-RC v1: one-shot role×distance relation composition on a frozen W2 atom state

Round id: `lcrc-pcba-rc-v1`
Protocol: `PROTOCOL.md` (frozen before the first formal step, commit `66ada472` / `7c053741`)
Training commits: RC-REAL `db0cf75b`, RC-BLIND `a947c861` · Freeze commit: `25adc50f`
Test role: **`EXPOSED_TEST_ENGINEERING_COMPARISON_SET`** — the MolPCBA official Test
split has been exposed by an earlier round in this track, so no number below is a blind
generalisation estimate. Selection everywhere is best-official-Validation AP; the Test
column is an engineering comparison only.
Seed: **0 only** (one run per arm, no tuning, no second seed).

All numbers are read back from the committed artifacts: `analysis/test_scores.json`,
`analysis/per_task_valid_test.csv`, `analysis/comparison.json`,
`analysis/task_delta_analysis.json`, `analysis/preflight.json`, `analysis/profile.json`,
`runs/REAL/run.json`, `runs/BLIND/run.json`. Nothing was retrained for this report.

**One-line result.** `RC-REAL` (1,051,022 params) is the best of the three models on both
selections: Test AP **0.255941** (best-Valid) / **0.256166** (epoch 40) vs W2
0.253632 / 0.250985, i.e. **+0.00231 / +0.00518** Test AP. `RC-BLIND` — same capacity, same
backbone, but the role×distance *binding* removed (uniform roles, i.e. only a distance
profile) — lands at **−0.00214** (best-Valid) / **+0.00035** (epoch 40) vs W2. The
`REAL − BLIND` contrast, the only one that isolates the binding from extra capacity, is
**+0.00445** Test / **+0.00352** Valid: positive, in the right direction, and below the
pre-registered +0.005 bar → the round is labelled **`WEAK_RC_SIGNAL`**, not a confirmed win.
The +0.00231 headline is a *small residual of large cancellations* (§3), not a uniform shift.

## 1. Three-way comparison

| model | params | best-Valid epoch | Valid AP (best) | Valid AP@40 | **Test AP (best-Valid)** | Test AP@40 | Train AP@40 | Test BCE | GPU-hour |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LCRC-W2 (historical, re-scored) | 994,324 | 24 | 0.265273 | 0.261867 | 0.253632 | 0.250985 | 0.5819 | 0.048584 | 1.3949 |
| **LCRC-RC-REAL** | 1,051,022 | 33 | **0.267718** | **0.266950** | **0.255941** | **0.256166** | 0.6043 | 0.049581 | 1.9186 |
| LCRC-RC-BLIND | 1,051,022 | 29 | 0.264196 | 0.261850 | 0.251493 | 0.251336 | 0.5880 | 0.049058 | 1.9685 |

| contrast | Δ Test AP (best) | Δ Test AP@40 | Δ Valid AP (best) | Δ Valid AP@40 | improved / degraded tasks (best) | label |
|---|---:|---:|---:|---:|---:|---|
| REAL − W2 | **+0.002309** | +0.005181 | +0.002444 | +0.005083 | 78 / 49 | `WEAK_RC_SIGNAL` |
| BLIND − W2 | −0.002139 | +0.000351 | −0.001078 | −0.000017 | 69 / 58 | `RC_REGRESSION` |
| **REAL − BLIND** | **+0.004448** | +0.004831 | **+0.003521** | +0.005100 | 68 / 59 | `WEAK_RC_SIGNAL` |

Reading of the design: BLIND has exactly the same parameter count and the same distance
machinery as REAL, but its role vectors are uniform, which removes the role×distance
co-occurrence term and leaves only a distance histogram. BLIND ≈ W2 (slightly below on
Test at best-Valid, slightly above at epoch 40, and −0.0011 on Validation) therefore says
the **extra ~57k parameters buy nothing by themselves**; the REAL − BLIND gap is the part
that can be attributed to the binding. That gap is +0.0044 Test / +0.0035 Valid.

Two caveats that keep this from being a stronger claim:

* **Single seed.** The W2 → W3 capacity comparison in the previous round moved Validation AP
  by −0.0029 at +1.1M parameters; the REAL − W2 Test gap (+0.00231) and the REAL − BLIND gap
  (+0.00445) are the same order of magnitude as that single-seed movement. Nothing here is
  statistically established.
* **Selection noise.** Validation and Test agree on the *ordering* (REAL > W2 > BLIND) and
  on the sign of all three contrasts, which is the strongest thing that can be said from one
  seed. Valid − Test gap: REAL +0.01178, W2 +0.01164, BLIND +0.01270 — indistinguishable.

## 2. Per-task structure of the REAL − W2 delta (best-Valid, official Test)

| quantity | value |
|---|---:|
| tasks scored | 127 (task 45 has 0 Test positives) |
| mean Δ (= macro AP delta) | **+0.002309** |
| median Δ | +0.002147 |
| positive / negative / zero tasks | 78 / 49 / 0 |
| Σ positive mass | +1.510646 |
| Σ negative mass | −1.217385 |
| gross movement Σ&#124;Δ&#124; | 2.728031 |
| net / gross | **0.1075** |

**Q(a) — how concentrated are the gains?** The five largest positive deltas are tasks
57 (+0.125007), 31 (+0.107147), 27 (+0.083599), 54 (+0.078976), 118 (+0.076519); they sum to
**+0.471247 = 31.20%** of the total positive mass and **160.7%** of the net. So the gains are
concentrated but not degenerate: the other 73 improving tasks contribute the remaining
1.04 AP of positive mass.

**Q(b) — are the biggest absolute movers driving the headline?** No, and the largest single
task movement is a *loss*. The five largest |Δ| tasks are 120 **−0.230989**, 57 +0.125007,
31 +0.107147, 108 **−0.083777**, 27 +0.083599 (3 positive, 2 negative). Their |Δ| sum is
0.630519 = **23.1%** of gross movement, but their *net* contribution is **+0.000987 = 0.34%
of the net** — they almost cancel. Dropping all five leaves the headline at +0.292274 of
+0.293261 (i.e. 99.7% of it).

**Q(c) — high-support subset (Test positives ≥ 20, n = 108).** Mean Δ **+0.000909**,
median +0.002059, improved **65/108 = 60.2%**, degraded 43/108, ΣΔ +0.098121. W2's mean AP
on that subset is 0.273289, so the relative gain there is **+0.332%.**

Reading: the improvement is broad in *count* (78/127 and 65/108 more tasks improved than
not) but thin in *magnitude* on well-supported tasks (+0.0009 AP), and it is offset by large
single-task losses (120 −0.231, 108 −0.084) that are not statistically distinguishable from
ranking noise on 20–60 positive examples.

## 3. Where "0.33%" comes from, and what is wrong with it

The number is real but it answers a narrower question than the one it was quoted for. Three
distinct issues, all quantified:

**(i) Denominator / stratum.** The same absolute delta produces different percentages
depending on the denominator:

| numerator | denominator | value |
|---|---|---:|
| mean Δ over 108 ge20 tasks = +0.000909 | W2 mean AP on those tasks = 0.273289 | **+0.332%** ← the quoted 0.33% |
| mean Δ over 127 scored tasks = +0.002309 | W2 macro AP = 0.253632 | **+0.910%** |
| mean Δ (absolute) | — | +0.2309 AP points |

The quoted 0.33% is the **high-support-subset relative** gain, not the official macro
relative gain (+0.91%) and not the absolute gain (+0.23 AP points). It reproduces to
0.3324% (Δ = 0.0005 pp from 0.33).

**(ii) Sign cancellation.** The macro delta is a *residual*: +1.51 AP of positive mass minus
−1.22 AP of negative mass. net/gross = 0.107, so ~89% of all task-level movement cancels
out. A single percentage on top of that residual hides that 49 tasks got worse by up to
−0.231 AP. It also means the headline is sensitive to individual tasks:
task 120 alone (−0.231) is worth −0.0018 of the macro mean, i.e. 79% of the entire gain.

**(iii) Mean-of-ratios vs ratio-of-means.** Averaging *per-task* relative changes instead
gives +14.99% (median +1.76%, ge20 mean +4.10%), because 18 low-AP tasks (W2 AP < 0.05, mean
+92%) dominate such an average; the worst offender is task 44 with W2 AP 0.820 where the
change is +665%. Any "x% better" claim in this track must therefore state (a) absolute AP
points, (b) the denominator, and (c) whether it is a ratio of means or a mean of ratios.
This report uses **absolute AP points plus macro-relative percent** only.

## 4. Engineering gates

| gate | result |
|---|---|
| pre-training checks (`analysis/preflight.json`, `--device cuda`) | **10/10 passed** |
| W2 sub-state bitwise identity at init, zero-residual forward == W2 | passed (CPU bitwise, CUDA ≤1e-6) |
| atom renumbering invariance / cross-molecule isolation / bucket partition / disconnected sentinel | passed |
| REAL vs BLIND control (uniform roles ⇒ identical; rebind roles ⇒ REAL changes, BLIND does not) | passed |
| brute-force float64 vs batched implementation | passed (rel ≤ 1e-5) |
| gradient path active for every new module | passed |
| Train-only labels: Test rows NaN in the training artifact, order plan SHA == W2 plan | passed |
| freeze before score (2-step, sealed) | `analysis/test_prediction_manifest.json`, seal `a8df26d1…`, `test_labels_read: false` |
| where each step ran | freeze on `res` GPU1 (`res-gpu1`); `score` on the workstation, because the deployed checkout deliberately carries no `data/ogb/*/raw/graph-label.csv.gz`. The frozen logits were pulled and re-hashed by `score` before any label was opened. |
| Validation replay at freeze vs recorded run.json | max &#124;Δ&#124; **6.7e-7** AP (CUDA reduction order) |
| official OGB AP vs independent sklearn per-task AP | max &#124;Δ&#124; **1.1e-16** over all six scored checkpoints |
| cost vs budget | REAL 1.92 h, BLIND 1.97 h (profile projection 1.27 / 1.33 h, target 2.25 h, hard 3.0 h) |

Step cost (same real batches, `analysis/profile.json`): W2 25.09 ms, REAL 38.26 ms
(+52.5%), BLIND 40.20 ms (+60.3%) — the 57k-parameter branch costs roughly half of the
total step time, which is why each arm took ~1.9 GPU-hours instead of 1.4.

## 5. What this round does and does not show

* **Does:** adding a one-shot, graph-level role×distance co-occurrence term to a frozen W2
  atom state moves official Test macro AP by +0.0023 (REAL) while the same capacity with the
  binding removed moves it by −0.0021 (BLIND). The *ordering* REAL > W2 > BLIND holds on
  Validation and on Test at the best-Validation selection; at epoch 40 BLIND edges W2 by
  +0.00035 (0.251336 vs 0.250985), so "capacity alone is neutral" is the fairer statement at
  that endpoint. The REAL − BLIND contrast (+0.0044 Test, +0.0035 Valid) is what isolates the
  binding.
* **Does not:** establish a >0.005 gain, a statistically significant effect, or a general
  mechanism. One seed, an exposed Test set, and a headline that is a 10%-of-gross residual
  with two large negative task movers cannot support more than "weak, direction-consistent".
* **Implication for the next model.** Extra graph-level capacity (BLIND) is neutral; the
  pairwise *relation* between atom states is where the signal lives. That is exactly what
  the next round (LCRC-PW v1) tests directly, with atom-pair tokens and distance-binned
  attention instead of a graph-level co-occurrence summary.
