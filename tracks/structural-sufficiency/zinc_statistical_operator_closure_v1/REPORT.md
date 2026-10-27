# REPORT — zinc-statistical-operator-closure-v1 (SOC-v1)

Date: 2026-10-28 (UTC+8) · track `structural-sufficiency` · architecture
frozen at `9e7fbae`; analysis code at `2a31c6c` · **official test never loaded** ·
**official valid loaded for description only, never for selection** ·
GPU spend **2.924 A100·h** of the 16 h cap (screen cap 3.0, used 2.923).

## 0. Decision

**STOP_NO_GAIN.** Neither pre-registered SOC arm passes the Fit8k/Monitor
screen gate. The full-10k confirmation and the terminal official-test read were
**not purchased**. The measured seed-mean effect is **−0.000203 MAE** (R arm,
95% CI [−0.002516, +0.002076]) — two orders of magnitude below the ~0.0094
test-MAE improvement needed to reach 0.060, and the per-seed signs disagree.

This is a clean negative result for the specific mechanism: a weight-tied,
explicit-moment closure over the bond/junction incidence graph, initialised at
DCC-L. It does not support a claim that "all iterative structural statistics are
useless"; it does close this operator family at the 10k-label regime.

## 1. Question

Can explicit structural statistics be turned into a *composable, repeated
computation operator* that beats the frozen DCC-L champion (494,193 params;
official-valid 4-seed mean 0.075990; official-test 4-seed mean 0.069400) and
moves a single raw-`y` model toward official test MAE < 0.060?

Two CPU diagnostics were run before the GPU round (train labels only):

* explicit radius-1/radius-2 composition counts added to the frozen 513-d `z`
  improve a small head by only ~0.008–0.013, and a permutation control explains
  most of it; the real correspondence effect is ~0.002–0.003;
* naive alpha-blended iteration on **frozen** encoders degrades Monitor MAE
  monotonically (0.13384 → 0.13400 → 0.13512 → 0.16317 → 0.43316).

The design therefore used a **small, end-to-end, weight-tied** operator that
starts numerically at DCC-L, rather than a frozen-trunk bolt-on or an external
key-count table.

## 2. What was tested

Base: exact frozen DCC-L `DualChannel` (P3 encoders → global composer → local
composer → 513-d reader). New: `MomentClosure`, one shared operator applied
K = 3 times.

For each object (atom, bond, junction), with state width 96:

```
group1, group2 =
  atom     : (incident bonds, empty)
  bond     : (two endpoint atoms, incident junctions)
  junction : (three patch atoms, two incident bonds)

m = mean(group); s = mean(group^2); c = count(group)
x = [h, m1, s1, m2, s2, log1p(c1), log1p(c2), role_embedding(8)]
u = MLP(LayerNorm(x) -> 24 -> 48 -> 96)
h' = h + sigmoid(gate_role) * u
```

All object types update simultaneously (Jacobi), with the **same** projection,
core MLP, role embedding and gate for all roles and all K iterations. No
attention, no pairwise relations, no global token, no per-iteration parameters.
This is honestly **learned iterative state propagation**, not "no message
passing".

| arm | readout | parameters | delta vs DCC-L |
|---|---|---:|---:|
| DCC-L anchor | — | 494,193 | — |
| **R** | final state `(a^3,b^3,w^3)` | 512,888 | +18,695 (+3.78%) |
| **T** | time mean over `t=0..3` | 512,888 | +18,695 (+3.78%) |
| CTRL (not run) | wider reader, no closure | 513,021 | +18,828 (+3.81%) |

Initialisation copies the backbone bit-for-bit from the frozen DCC-L seed
state; only the closure is new. `gate_bias = -5` → `sigmoid = 0.006693`.
Forcing `gate_bias = -30` reproduces DCC-L predictions exactly (max abs diff
**0.0** on 16 real molecules, both readouts). The parameter-matched CTRL was
pre-registered as conditional on a SOC pass; since no arm passed, it was not
run (no post-hoc control fishing).

## 3. Identity and CPU gates

All nine CPU gates pass (`analysis/cpu_gates.json`):

| gate | result |
|---|---|
| parameter count | R/T 512,888; CTRL 513,021 (+0.026%) |
| exact DCC-L init | max abs 0.0 (R/T), 2.4e-7 (CTRL) |
| forward/backward | finite; gradient reaches every closure tensor |
| batch-composition invariance | 3.0e-8 (16 alone vs 16+16) |
| activation | prediction delta 0.110; bond-state drift 0.210 |
| weight tying | K=1 and K=3 have identical parameter count |
| tiny overfit (8 mols, 60 CPU steps) | 0.989 → 0.091 |
| open gate finite | max abs 0.56 at `sigmoid≈1` |
| readouts distinct | mean vs final delta 2.01 |

A real incidence-scatter bug (bond/junction neighbour states were indexed by the
wrong object type) was caught by the batch-invariance gate and fixed **before**
any GPU run. The frozen anchors were reproduced exactly from the released
fit8k checkpoints (0.12315088 / 0.12777157 Monitor).

Data/recipe identity: BJOC `fit8k` (8,000 rows), Monitor 1k selection, frozen
fit-only normalisation, frozen masked-SSL `s0`/`s1` encoders, AdamW lr
1e-3→1e-5 cosine, 500-step warmup, wd 1e-5, batch 128, clip 5, raw-`y` L1,
47,400 updates, 753 epochs. Locked and official test were not loaded.

## 4. Screen results

Frozen DCC-L anchors on the same rows: seed 0 **0.12315088**, seed 1
**0.12777157**, mean **0.12546122**.

### 4.1 R arm (final-state readout)

| seed | SOC Monitor | DCC-L Monitor | Δ (SOC−L) | SOC valid | DCC-L valid | best epoch |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 0.124011 | 0.123151 | **+0.000861** | 0.090063 | 0.089873 | 571 |
| 1 | 0.126506 | 0.127772 | **−0.001266** | 0.097404 | 0.095470 | 580 |
| mean | **0.125259** | **0.125461** | **−0.000203** | 0.093734 | 0.092672 | — |

Paired bootstrap over the 1,000 Monitor molecules (10,000 resamples):

| comparison | Δ MAE | 95% CI | P(Δ<0) | win fraction |
|---|---:|---|---:|---:|
| seed 0 | +0.000861 | [−0.002239, +0.003927] | 0.297 | 0.513 |
| seed 1 | −0.001266 | [−0.004445, +0.001976] | 0.783 | 0.515 |
| seed mean | **−0.000203** | **[−0.002516, +0.002076]** | 0.584 | 0.501 |

Gate: seed mean 0.125259 > 0.1215 → **FAIL**. The point estimate is a 0.16%
improvement; the pre-registered bar was 3.2%.

### 4.2 T arm (trajectory-mean readout)

| seed | SOC Monitor | DCC-L Monitor | Δ (SOC−L) | SOC valid | DCC-L valid | best epoch |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 0.126867 | 0.123151 | **+0.003716** | 0.092100 | 0.089873 | 599 |
| 1 | 0.128599 | 0.127772 | **+0.000827** | 0.090502 | 0.095470 | 583 |
| mean | **0.127733** | **0.125461** | **+0.002271** | 0.091301 | 0.092672 | — |

| comparison | Δ MAE | 95% CI | P(Δ<0) | win fraction |
|---|---:|---|---:|---:|
| seed 0 | +0.003716 | [+0.000120, +0.007809] | 0.020 | 0.503 |
| seed 1 | +0.000827 | [−0.002137, +0.003874] | 0.293 | 0.488 |
| seed mean | **+0.002271** | [−0.000270, +0.005042] | 0.041 | 0.501 |

Gate: **FAIL**, and the T readout is worse than R. Averaging the trajectory
blurs the final state; the final-state readout is the better use of the same
operator.

### 4.3 No targeted subgroup gain

The R seed-mean Δ by structural group (Monitor) flips sign across bins and has
no consistent direction:

| group | n | Δ (SOC−L) |
|---|---:|---:|
| n_atoms Q1 / Q2 / Q3 / Q4 | 290/234/279/197 | +0.00061 / −0.00208 / −0.00016 / +0.00077 |
| n_rings 0 / 1 / 2 / 3+ | 3/84/329/584 | −0.0468 / +0.00116 / +0.00182 / −0.00130 |
| y tail (≤−3) vs ordinary | 74/926 | +0.00371 / −0.00052 |
| n_bonds Q1 / Q2 / Q3 / Q4 | 263/292/229/216 | +0.00208 / −0.00348 / +0.00078 / +0.00041 |

The largest “gain” is a 3-molecule bin; the 74-row heavy tail is worse. There
is no evidence the closure targets the unseen radius-2 dispersion that the
bottleneck audit identified as the main remaining error.

## 5. Mechanism: why the wash

### 5.1 The gates were not “cold-start suppressed”

Final learned gates (sigmoid) are essentially at the 0.006693 initialisation:

| arm/seed | atom | bond | junction |
|---|---:|---:|---:|
| R s0 | 0.005653 | 0.007148 | 0.006111 |
| R s1 | 0.005172 | 0.006816 | 0.004847 |
| T s0 | 0.005767 | 0.007616 | 0.006625 |
| T s1 | 0.004947 | 0.007086 | 0.005317 |

This is not a vanishing-gradient accident: the **train** loss is minimised near
this amplitude. For R s0, forcing the trained closure open gives:

| forced gate | train-2k MAE | Monitor MAE |
|---:|---:|---:|
| 0.0001 | 0.010264 | 0.125707 |
| 0.002 | 0.008820 | 0.125040 |
| **0.006693 (trained)** | **0.008469** | 0.124062 |
| 0.01 | 0.009860 | 0.123769 |
| 0.02 | 0.015423 | **0.123695** |
| 0.05 | 0.028278 | 0.127644 |
| 0.1 | 0.055502 | 0.143736 |

The train-optimal gate is the trained one; opening it further overfits train
noise and only moves Monitor by ~0.0004. The model is heavily overfit
(train-2k 0.0085 vs Monitor 0.124), so extra closure amplitude is not a free
generalisation lever.

### 5.2 The closure’s learned contribution is ~0.001–0.002, and never enough

With the trained closure weights held fixed, disabling the closure (gate −30)
vs the trained gate vs the best forced gate:

| run | closure off | trained gate | best forced gate | best vs off | best vs DCC-L |
|---|---:|---:|---:|---:|---:|
| R s0 | 0.125746 | 0.124062 | 0.123695 (g=0.02) | −0.002051 | +0.000544 |
| R s1 | 0.126693 | 0.126521 | 0.126521 (g=0.0067) | −0.000172 | −0.001251 |
| T s0 | 0.128021 | 0.126895 | 0.125831 (g=0.02) | −0.002190 | +0.002680 |
| T s1 | 0.128612 | 0.128603 | 0.128598 (g=0.002) | −0.000014 | +0.000826 |

The maximum contribution of the learned closure is ~0.0022 (R s0 / T s0). At
the best forced gate, R s1 beats its seed-matched DCC-L anchor by 0.00125, but
R s0 loses by 0.00054 and T s0 by 0.00268; the seed-mean does not beat the
anchor. The operator’s direction is real but too weak, and the SOC-trained backbone
with the closure off is itself worse than DCC-L (R s0 0.125746 vs 0.123151):
the branch perturbs optimisation, then partially compensates. This is the same
“extra branch costs optimisation” pattern as BAMF (+0.003149), COC (+0.007247)
and native-relation (+0.004–0.006), just with a smaller cost and a smaller
benefit.

### 5.3 Split disagreement confirms noise-level effects

T s1 is better than its DCC-L anchor on official valid (0.090502 vs 0.095470)
while being worse on Monitor (0.128599 vs 0.127772). The two dev splits
disagree in sign. No result at this effect size should be promoted to a
mechanism claim.

## 6. Why this does not reach 0.060

* The measured seed-mean effect is **−0.000203** Monitor MAE. The required
  improvement from the DCC-L test level 0.069400 to 0.060 is **−0.009400**.
  Even the most optimistic 95% bound (−0.002516) is only ~27% of the required
  move, on a different split.
* The operator’s maximum forced contribution is ~0.0022, still short of the
  frozen DCC-L anchor on the same rows.
* The remaining error is **unseen radius-2 environment dispersion**, not a
  missing local moment. Local moments of the encoder states cannot create
  coverage for environments that never appear in the 10k labels; they can only
  smooth existing states. The DCB audit already showed the error is
  dispersion/heteroscedasticity (bias²/MSE ≈ 0.001), not a mean shift.
* The 0.060 level is at the known no-MP attention/positional-encoding frontier
  (GRIT ~0.059 with 473k params; CSA-rings ~0.056; ESA+ 0.052 with ZINC-12k and
  RWSE/ring features). Those methods use global attention and rich positional
  encodings, not local moment closure, and the repo’s prior attempts to add
  such branches to DCC-L paid 0.004–0.007 optimisation cost. This round does
  not reopen that line.

## 7. Budget and artifacts

| item | A100·h |
|---|---:|
| smoke (64 updates) | 0.0011 |
| R screen s0 / s1 | 0.6779 / 0.7725 |
| T screen s0 / s1 | 0.6804 / 0.7918 |
| **total** | **2.9237** |

* Screen cap 3.0; hard cap 16.0. No full-10k run, no CTRL, no test read.
* Runs: `soc-smoke-r0-20261028-011415-9314a812`,
  `soc-screen-R-s0-20261028-011613-0fe46447`,
  `soc-screen-R-s1-20261028-011625-7e2d5210`,
  `soc-screen-T-s0-20261028-020617-78f7cbab`,
  `soc-screen-T-s1-20261028-020630-586e2d96`.
* Analysis: `analysis/cpu_gates.json`, `analysis/screen_R.json`,
  `analysis/screen_T.json`, `analysis/force_gate_R_s{0,1}.json`,
  `analysis/force_gate_T_s{0,1}.json`.
* Code: `code/soc_models.py`, `soc_train.py`, `soc_analyze.py`,
  `soc_force_gate.py`, `soc_score.py` (terminal scorer, never invoked),
  `soc_checks.py`, `test_soc.py`.
* Protocol/design: `PROTOCOL.md`, `DESIGN.md`; ledger: `notes/ledger.md`.
* Official test loaded: **False** in every artifact above.

## 8. What would be worth buying next (not started)

1. **Label-count slope 10k→12k** (0.75 A100·h, frozen DCC-L, already recorded
   in `zinc_dccl_bottleneck_v1` §E). This is the cheapest way to test whether
   the 10k label budget, not the architecture, is the binding constraint. It
   requires a new pre-registration that accepts the external-label protocol
   change; it is not a SOC follow-up.
2. **Coverage-aware training objective**, not another operator: reweight or
   group-DRO the L1 loss by radius-2 environment support, so the model spends
   capacity on unseen-environment dispersion. This uses the existing explicit
   statistics as *training weights* rather than as readout features. It is the
   only architecture-adjacent direction the DCB error structure actually
   supports, and it needs a new two-seed Fit8k screen with the same anchor.
3. **Do not** run more moment-closure variants (K, gate bias, rank, T readout,
   larger reader). The forced-gate diagnostic bounds the mechanism at ~0.002
   Monitor and the train-optimal gate is already at the trained value; more
   tuning would be a search on noise.

## 9. What this does not claim

* It does not claim iterative structural statistics are useless in general.
* It does not claim DCC-L is at the information-theoretic ceiling.
* It does not claim the 0.060 target is impossible; it claims this operator
  family, at this label count, does not approach it.
* The CTRL capacity control was not run because the pre-registered condition
  (a SOC pass) was not met; no capacity claim is made.
