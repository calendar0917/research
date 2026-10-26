# REPORT — ZINC DCC Information Audit v1

**Track:** `tracks/structural-sufficiency/zinc_dcc_information_audit_v1/`
**Frozen plan:** `AUDIT_PLAN.md` (sha `ddc6978121bdb91588876a16b6e5f13eee95173cce6b6b9100de0e9ab0dd7f22`)
**Frozen configs:** `configs/audit_v1.json`, `configs/b_phase_freeze.json`, `configs/split_fit8k_probe_audit.json`
**Base commit:** `58e6837b91507aae3feb9ad9370ef75d1959b7ec`
**B-phase commit:** `8b62dff0be595dfebc2062d8bd9ea190bad87e08`
**Decision:** **B — `NONREDUNDANT_BUT_NOT_PREDICTIVE`**
**Bottom line:** **`NO_JUSTIFIED_NEW_MODEL`** for this joint-statistic family.

---

## 0. One-paragraph answer

DCC-L's 513-dimensional reader input `z` retains many *distributional* structural
statistics, but it is exactly blind to some *correspondence-aware* joint
relations: A2 gives a strict activation-level collision and a real-molecule
rebinding control in which `z` and the DCC prediction are invariant while
`T_R1` changes on 100% of molecules. A4 shows that the predefined joint
statistics are not fully reconstructible from `z` by the frozen probes
(`J_latent` AuditEval R² 0.774/0.762; `J_raw` 0.934/0.933), so A does not
automatically imply B and the B phase was allowed to run. On 1,000 never-trained
AuditEval molecules, however, no stable label-prediction increment was found.
The primary Ridge diagnostic gives positive `Gain_J` on both seeds
(0.003688 and 0.003079 MAE), but seed 1's paired 95% interval includes zero,
removing its ten largest-gain rows reverses the sign, the pre-registered Huber
sensitivity changes sign, the dim-matched `ZN` control captures most of the
seed-0 gain, and the Ridge heads are badly miscalibrated (AuditEval MAE
0.26–0.32 versus 0.074 for raw DCC-L). Two ProbeFit molecules account for
96.96% of ProbeFit residual SSE, so the primary head is fitting two extreme
labels (rows 2210 and 3776), not a generalizable relation signal. The audit
therefore supports **H-A** at the operator level, gives **partial support for
H-B** (robust for `J_latent`, marginal/nonlinear for `J_raw`), does **not**
support **H-C**, and does **not** support **H-D**. This is a conclusion under
the current representation, data, split, and diagnostic function; it is not a
proof that all correspondence-aware relation modeling is useless, and it is not
evidence of reaching a 0.05× target.

---

## 1. Scope and access discipline

- Data: official ZINC **train only** (10,000 structures). Official **valid was
  never loaded**; official **test was never constructed, loaded, or read**.
  `audit_common.install_load_guard` refuses `ad_valid.pt`/`ad_test.pt`; B3's
  `GuardedLabels` recorded `audit_labels_accessed: false` for both seeds; B4
  recorded `official_valid_loaded: false`, `official_test_loaded: false`.
- Labels were used only for: (i) Fit8k DCC-L supervision, (ii) ProbeFit audit-head
  fitting, (iii) one AuditEval read per seed after prediction hashes were frozen.
  No label was used for structure statistics, splitting, gates, feature
  selection, or normalization.
- Split is structure-only, frozen before A4/B. Three exact isomorphic duplicate
  pairs were verified with `networkx.is_isomorphic`; all six rows went to Fit8k,
  so no duplicate spans roles. No non-isomorphic WL collisions.
- Frozen split role hashes:
  - Fit8k: `b2666522b589ab118bc52779b74df1100f169d00844ec192cd493b1a84f7ff42`
  - ProbeFit: `67644e905d7dcbc637d19e7e6c7f7cc91ed4ae51399a267ec3a1a2a8a6bb5892`
  - AuditEval: `f2b7f5592cb0cac901554e1a5160c69fe1b58044517077e680ff6597f6410bc4`
- The full-10k historical SSL checkpoints were used only as A1–A4 background
  backbones D0/D1. The B phase used new Fit8k-only SSL encoders, paired
  seed-for-seed with the supervised seeds (s0←SSL s0, s1←SSL s1). This is
  strictly inductive at the Fit8k boundary.
- No official valid was used for checkpoint selection, audit-head training,
  gating, or the main table. No official test number appears in this report.

---

## 2. What was measured

### 2.1 Reader layout (A1)

`z` is exactly the concatenation

```
z = [ z_glob(0:96), a_sum(96:192), a_sq(192:288),
      g(288:321), zl(321:417), zl2(417:513) ]
```

where `zl = sum(c_loc)` and `zl2 = sum(c_loc^2)`. On 1,000 official-train
molecules and both D0/D1:

- reader parity `max |capture - model.forward| = 0.0`;
- independent grouping recomputation of `Wg/Vg/Wl/Vl/n_b` max diff `0.0`;
- reassembly `reader(z) == pred` max diff `0.0`;
- atom relabel / bond permutation / endpoint reversal / batch regrouping
  invariance `≤ 8.94e-08` (float32 noise).

The reader is a pooled-sum architecture. Its input contains no explicit pair,
triple, orientation, or correspondence term. A1 also shows the trained local
channel is low-rank: D0 `c_loc` participation ratio 2.94 (top1 0.572, 16 zero
eigenvalues), D1 3.34; pooled `z` PR 2.28/2.07. This is a trained-weight
property, not an initialization claim.

### 2.2 Algebraic collision (A2)

A2.1 fixes a 6-bond path line graph and two integer bond-state assignments
`X=(u,v,-u,-v,w,-w)` and `Y=(u,-u,v,-v,w,-w)`:

- `sum` max diff `0.0`;
- `sumsq` max diff `0.0`;
- independent float64 reference matches;
- reader output abs diff `0.0`;
- correspondence-dependent `T_R1` changes by `0.7708`, and the
  line-graph adjacent-pair dot sum changes by `370.0`.

This is a strict activation-level non-identifiability certificate for the
pooled `(sum, sumsq)` reader. It is **not** a real-molecule non-identifiability
proof.

A2.2 applies a fixed multiset-preserving rebinding of bond states on 200 real
train molecules (≥6 bonds), holding atom/bond features fixed:

- DCC prediction mean absolute delta `3.05e-08` (D0) / `3.91e-08` (D1), i.e.
  invariant;
- `zl` max diff `3.8e-06` / `7.6e-06` (multiset-preserving float noise);
- exact multiset reader delta `0.0`;
- `T_R1` mean absolute delta `0.0101` / `0.0125`, changed on **100%** of
  molecules.

A2.2 is an information control, not a legal chemical reconstruction and not a
label counterfactual. It demonstrates that a relation statistic can change
while the pooled reader cannot see it.

### 2.3 Predefined joint statistics (A3)

Two frozen families were used:

- **`J_raw` (32 dims):** relation frequencies for four predefined relation
  templates R1–R4 over bond types, plus log1p relation counts. The actual train
  bond vocabulary is `{1,2,3}` only (counts 185,060 / 63,548 / 671 over 249,279
  physical bonds; no aromatic type 4). The 48-dim four-slot template is
  diagnostic only and is never used in models or gates.
- **`J_latent` (4 dims):** `T_R1..T_R4`, computed from within-molecule centered
  local composer states `c_loc`, divided by 96, with empty relation → 0.

A3 validation:

- independent float64 exhaustive reference on 100 rows: `J_raw` max diff `0.0`,
  `J_latent` max diff `4.86e-17`;
- atom relabel / bond permutation / endpoint flip invariance `≤ 2.1e-17`;
- edge cases (empty, single bond, tree path5, C6 ring, fused rings, two
  components) all match expected counts; e.g. C6 R1=6/R2=6/R3=3/R4=15;
- relation pair counts over 10k: R1 mean 34.64, R2 43.93, R3 233.62, R4 46.15;
  R4 empty fraction 0.55%;
- 11 `J_raw` dims are near-constant under the frozen rule (zero fraction ≥0.99
  or std ≤1e-3): indices `[4,5,7,13,15,21,23,26,28,29,31]` (9 exact constants
  plus `R3_freq_3_3` and `R4_empty`). They are excluded symmetrically before
  A4 fitting/scoring and B3 heads; stable `J_raw` dims = 21. `J_latent` has no
  constant dims.
- Correlations: `J_raw` vs `global33` max |r| 0.706; vs D0 `z` 0.084; vs D1
  `z` 0.165. `J_latent` vs `z` max |r| 0.096 (D0) / 0.238 (D1); vs
  `global33` 0.187/0.223. Linear correlation is only a screen; A4 ridge is
  the main test.

### 2.4 Conditional redundancy (A4)

Fit on Fit8k only, 5-fold internal CV, fixed alpha grid
`[1e-4,1e-3,1e-2,1e-1,1,10,100]`; scored on ProbeFit (primary) and AuditEval
(second out-of-sample). Block R² values:

| target | seed | ProbeFit Ridge R² | AuditEval Ridge R² | ProbeFit MLP R² |
|---|---:|---:|---:|---:|
| `J_raw` | D0 | 0.9410 | 0.9338 | **0.9765** |
| `J_raw` | D1 | 0.9323 | 0.9333 | **0.9708** |
| `J_latent` | D0 | −0.1896 | 0.7736 | 0.2386 |
| `J_latent` | D1 | 0.5805 | 0.7618 | 0.7398 |

Gate rule: `STOP_BEFORE_NEW_GPU` iff every primary block at every seed has
`z→J` ProbeFit R² ≥ 0.98 and residual variance share ≤ 0.02. **Gate outcome:
`stop_before_new_gpu = false`.** `J_raw` is close but not sufficient under
Ridge; `J_latent` is clearly not reconstructible by the frozen probes. The
`J_latent_D0` ProbeFit R² is dominated by one ProbeFit outlier (row 3776; its
`T_R2` lies far outside the Fit8k range). AuditEval and the MLP sensitivity
show the same qualitative picture.

**Important nuance:** the `J_raw` MLP sensitivity is 0.9765/0.9708, just below
the 0.98 line. `J_raw` is almost reconstructible nonlinearly; the robust
nonredundant signal is mainly `J_latent`. The gate is an engineering screen, not
a Bayes-sufficiency theorem. It did not stop, but it does not establish a large
independent information increment for `J_raw`.

---

## 3. B phase: training, frozen heads, one-time scoring

### 3.1 DCC-L reference training (B2)

Native `DualChannel` / B-R0 forward, 494,193 parameters, Fit8k only, exactly
47,400 updates (753 epochs), batch 128, AdamW lr 1e-3→1e-5, wd 1e-5, raw-y L1,
warmup 500, clip 5, dropout 0, final-update checkpoint (no early stopping or
selection). Fit8k-only `global33` mean/scale; root/block scalars are not
consumed by the DCC-L/P3N collate.

| seed | checkpoint sha256 | wall | GPU·h | Fit8k train MAE |
|---|---:|---:|---:|---:|
| 0 | `b7b92659b4cdce7c1ced925c5b60dd983b089b426f9e0695cca3bd9f46656dfe` | 1195.1 s | 0.332 | 0.000415 |
| 1 | `ff660cc15c8ca4b554459dae77804f8c3d4a5052f6b42232dcf646350ed2238b` | 1343.2 s | 0.373 | 0.000502 |

Both completed exit 0; 47,400 updates and 753 epochs recorded. The large
Fit8k fit/generalization gap is expected for this fixed overparameterized
recipe; it is part of the audit setting, not a model-selection result.

### 3.2 Frozen audit heads (B3)

For each seed: freeze DCC-L; fit `J_hat` on Fit8k from `z` only (label-free);
fit three ProbeFit Ridge residual heads with the fixed alpha grid
`[1e-4,...,100]`:

- **Z:** full `z`;
- **ZJ:** full `z` + `J_raw` (21 valid dims) + `J_latent` (4 dims);
- **ZN:** full `z` + `J_hat` (25 dims), the dimension-matched no-new-information
  control.

Huber sensitivity: fixed `HuberRegressor(epsilon=1.35, alpha=1e-4)`, one fit,
no tuning. `Huber n_iter` reached 300 in seed 0, so Huber is a sensitivity
check, not a converged main result.

Both seeds: support strata identical (full 316 / low 392 / unseen 292);
`label_guard.audit_labels_accessed = false`; 1,000 forbidden AuditEval rows and
1,000 ProbeFit rows accessed; AuditEval prediction hashes were written before
B4. Main head alpha: seed 0 `100.0`, seed 1 `0.0001` (both on the pre-fixed
grid boundary; not changed after seeing AuditEval).

### 3.3 One-time AuditEval scoring (B4)

B4 read AuditEval labels exactly once per seed, after verifying all prediction
and checkpoint hashes:

- seed 0: `2026-10-26T02:56:04Z`, 1,000 rows, rows sha `f2b7f559…`;
- seed 1: `2026-10-26T02:56:20Z`, 1,000 rows, rows sha `f2b7f559…`.

The first seed-0 attempt failed at the checkpoint hash check **before any label
read** because of a repo-relative path; it was fixed and rerun. No AuditEval
label was read twice.

Main results (MAE on 1,000 AuditEval molecules):

| arm | seed 0 | seed 1 | mean |
|---|---:|---:|---:|
| raw DCC-L | 0.07524 | 0.07423 | 0.07474 |
| Z (z only) | 0.26726 | 0.31983 | 0.29354 |
| ZJ (z + J) | 0.26357 | 0.31675 | 0.29016 |
| ZN (dim-matched null) | 0.26400 | 0.32027 | 0.29214 |
| Huber Z | 0.11427 | 0.12826 | 0.12126 |
| Huber ZJ | 0.11293 | 0.13336 | 0.12315 |
| Huber ZN | 0.11200 | 0.13398 | 0.12299 |

`Gain_J = MAE(Z) − MAE(ZJ)`; `Gain_vs_N = MAE(ZN) − MAE(ZJ)`:

| quantity | seed 0 | seed 1 |
|---|---:|---:|
| `Gain_J` | **0.003688** | **0.003079** |
| `Gain_J` 95% paired bootstrap CI | `[0.001335, 0.005937]` | `[-0.005152, 0.011036]` |
| `Gain_vs_N` | 0.000430 | 0.003521 |
| `Gain_vs_N` 95% CI | `[-0.003502, 0.004416]` | `[-0.004719, 0.011484]` |
| top-10 gain rows removed | **+0.002212** | **−0.001681** |
| `Gain_huber_J` | +0.001332 | −0.005105 |
| `Gain_huber_vs_N` | −0.000931 | +0.000613 |
| Z vs ZJ residual Pearson | 0.996 | 0.960 |
| ZN vs ZJ residual Pearson | 0.987 | 0.959 |

Bootstrap: 10,000 resamples, seed 20261026, unit = AuditEval molecule. This is
fixed-model sample uncertainty, **not** training-seed uncertainty.

### 3.4 Why the Ridge heads are not a clean positive test

The Ridge heads are much worse than raw DCC-L on AuditEval. The reason is
visible in ProbeFit labels only (no extra AuditEval read):

| diagnostic | seed 0 | seed 1 |
|---|---:|---:|
| ProbeFit rows 2210 / 3776 labels | −42.04 / −20.26 | −42.04 / −20.26 |
| top-2 share of ProbeFit residual SSE | **96.96%** | **96.96%** |
| top-2 share of ProbeFit MAE | 33.48% | 33.36% |
| ProbeFit residual std all | 1.010 | 1.028 |
| ProbeFit residual std without top-2 | 0.176 | 0.179 |
| Z head ProbeFit correction min | −24.55 | −29.10 |
| Z head AuditEval correction min | −6.65 | −5.85 |

The main Ridge head is effectively fitting two extreme tail molecules. Their
labels are far outside the AuditEval tail (raw AuditEval residual max 3.09 /
3.63). The head extrapolates a large negative correction to AuditEval, which
explains why Z/ZJ/ZN are worse than the uncalibrated backbone. This is a
finite-sample calibration failure, not evidence that J is predictive. It also
means the positive `Gain_J` is a difference between two miscalibrated heads,
not a clean estimate of what J adds to a well-calibrated predictor.

### 3.5 Strata and H-D

Support strata are computed strictly from Fit8k radius-2 WL environments:
full 316 / low 392 / unseen 292 AuditEval molecules. `Gain_J` by stratum:

| stratum | n | seed 0 Gain_J [95% CI] | seed 1 Gain_J [95% CI] |
|---|---:|---:|---:|
| full support | 316 | +0.0045 `[+0.0021,+0.0073]` | +0.0067 `[−0.0023,+0.0166]` |
| low support | 392 | +0.0037 `[−0.0001,+0.0076]` | −0.0066 `[−0.0224,+0.0082]` |
| unseen | 292 | +0.0027 `[−0.0029,+0.0079]` | +0.0121 `[−0.0053,+0.0286]` |
| ordinary `y>-3` | 932 | +0.0028 `[+0.0008,+0.0048]` | +0.0048 `[−0.0034,+0.0125]` |
| tail `y≤-3` | 68 | +0.0154 `[−0.0043,+0.0342]` | −0.0198 `[−0.0754,+0.0338]` |
| ring-rich `n_rings≥4` | 67 | +0.0076 `[+0.0003,+0.0158]` | +0.0294 `[+0.0057,+0.0525]` |

There is no monotone “harder support → larger gain” pattern. Seed 0 is positive
on full/low/unseen with intervals mostly including 0; seed 1 is negative on low
support and positive on unseen; tail signs disagree. The only both-seed
positive small stratum is ring-rich (n=67, exploratory): it is interesting but
small, uses the same miscalibrated heads, and was not used for the decision.
The tail accounts for only 15.9% of AuditEval absolute error (below the 50%
`BENCHMARK_TAIL_DOMINATED` tag threshold). H-D is not supported.

---

## 4. Seven explicit answers

### Q1. Does the 513-dim DCC-L reader input `z` retain structural relations?

**Partially, but not the correspondence-aware joint relations that A2 targets.**
A1 shows `z` is exactly a collection of pooled sums (`z_glob`, atom sums,
squared atom sums, `global33`, local sums, local squared sums); A4 shows most
`J_raw` count information is linearly recoverable (R² 0.94) and almost fully
recoverable nonlinearly (0.9765/0.9708). So “retains relations” is true for
aggregate counts. It is false for relations that depend on which bond state is
paired with which position/orientation: A2 produces two assignments with
identical `z` and identical reader output, while `T_R1` changes by 0.7708.
The 513-dim reader is not information-free; it is correspondence-blind.

### Q2. Are those relations inaccessible after pooling/readout?

**Yes for the correspondence/orientation part, and the inaccessibility is
exact in the A2.1 construction.** The pooled reader has no term that can
distinguish the two assignments; the collision is at the activation level, not
a failure of a downstream probe. But “inaccessible” must be scoped: A4 shows
that the frozen probes can reconstruct much of `J_raw` and a substantial part
of `J_latent` from `z`; the nonredundant part is a minority of the variance.
The correct statement is “not identifiable from `z` by the tested reader/probe
family,” not “information-theoretically absent from the whole network.”

### Q3. Is there a strict counterexample?

**Yes.** A2.1 is a strict float64 counterexample for the specific pooled
`(sum, sumsq)` aggregation: `sum` diff 0.0, `sumsq` diff 0.0, reader diff 0.0,
while a correspondence-dependent statistic changes. A2.2 extends the phenomenon
to real train molecules under a multiset-preserving rebinding (prediction delta
~3e-08, `T_R1` changed on 100% of molecules). A2.2 is an information control,
not a legal molecule and not a label counterfactual; it should not be
over-interpreted as chemical non-identifiability.

### Q4. Are the new joint statistics non-redundant relative to the full `z`?

**Yes under the pre-fixed probes, with an important qualification.** The A4
gate did not stop: `J_latent` AuditEval R² is 0.774/0.762, and `J_raw` is
0.934/0.933. `J_latent` is robustly nonredundant; `J_raw` is marginal because
the MLP sensitivity reaches 0.9765/0.9708, close to the 0.98 stop line. The
result is a probe-relative statement, not a Bayes-sufficiency theorem. A4 also
does not establish that the nonredundant residual is *useful* for labels; that
is the separate C test in B4.

### Q5. Is there a stable label-prediction increment on never-trained molecules?

**No stable increment was found.** The primary Ridge point estimates are
positive on both seeds (0.003688 and 0.003079), and seed 0's interval is
positive. But seed 1's interval includes zero; removing the ten largest-gain
rows reverses seed 1's sign; the pre-registered Huber sensitivity changes sign
(+0.001332 seed 0, −0.005105 seed 1); the dim-matched `ZN` control captures
most of seed 0's gain (`Gain_vs_N = 0.000430`, CI includes 0); and the Ridge
heads are miscalibrated and dominated by two ProbeFit tail molecules
(96.96% of residual SSE). The honest conclusion is **B:
`NONREDUNDANT_BUT_NOT_PREDICTIVE`** under the current representation/data/
diagnostic function. A positive effect of this size is not established; the
data are also not strong enough to rule out a small effect under a better
calibrated head.

### Q6. Do low-support, unseen local environments, or tail hard cases explain the errors?

**No consistent explanation.** Full/low/unseen and tail strata do not show a
monotone or seed-stable gain pattern; seed 1 is negative on low support and
positive on unseen, while tail signs disagree. The dominant error mechanism in
this audit is not support/tail localization but residual calibration and two
extreme ProbeFit labels. The ring-rich stratum (n=67) is positive on both seeds
(+0.0076, +0.0294; both CIs positive) and is worth a pre-registered replication,
but it is small, exploratory, and produced by the same miscalibrated heads.
H-D is not supported by the pre-registered strata.

### Q7. Should the next round build an original structured-composition model, or stop structural-operator search and pivot to estimation/data mechanisms?

**Do not build a new structured-composition model from this audit. State
`NO_JUSTIFIED_NEW_MODEL`.** The evidence supports the following priority order:

1. **Estimation/calibration first.** The largest observed effect is the audit
   head's failure on two extreme ProbeFit molecules. A pre-registered follow-up
   should use outlier-robust or cross-fitted residual heads, an outlier-aware
   ProbeFit construction, and more ProbeFit data before any architecture claim.
2. **Data/tail mechanisms.** The two extreme labels (y = −42.04 and −20.26)
   dominate ProbeFit MSE and are not representative of AuditEval. Heavy-tail
   calibration, label-noise/uncertainty modeling, and support-aware data
   collection are higher-value next steps than another structural operator.
3. **Targeted replication, not architecture search.** Replicate the ring-rich
   (n=67) signal on a fresh, larger, pre-registered evaluation. Only if a
   calibrated, stable J gain survives should a minimal structured-composition
   model be considered; the current audit does not justify it.
4. **Do not spend more GPU on this 32-dim R1–R4 family** in the current form.
   No new full prediction architecture was trained, no ensemble/distillation/
   residual patching was used, and no width/depth/lr/statistic-family search
   was performed.

---

## 5. Hypothesis verdicts

| hypothesis | verdict | evidence |
|---|---|---|
| **H-A** algebraic unidentifiability | **Supported at the operator level** | A2.1 exact collision (`sum`/`sumsq`/reader diff 0.0) while `T_R1` changes 0.7708; A2.2 real-molecule invariance with `T_R1` changed on 100% of molecules. Not a proof of real-molecule non-identifiability. |
| **H-B** non-redundancy vs full `z` | **Partially supported** | A4 gate passes: `J_latent` Audit R² 0.774/0.762; `J_raw` 0.934/0.933. But `J_raw` MLP R² 0.9765/0.9708, near the 0.98 line; robust nonredundancy is mainly `J_latent`. |
| **H-C** label-prediction increment | **Not supported** | Class B. Primary `Gain_J` positive but seed-1 CI includes 0, top-10 removal reverses sign, Huber sign disagrees, `ZN` control captures seed-0 gain, heads are worse than raw DCC-L and outlier-dominated. |
| **H-D** low-support/tail explanation | **Not supported** | Full/low/unseen and tail gains are mixed across seeds; no monotone pattern. Ring-rich is a small exploratory exception (n=67), not a confirmed explanation. |

A does not imply B; B does not imply C; C would not imply model novelty or
0.05× reachability. This audit respects those separations. It does not claim
“structure is insufficient” or “DCC-L is sufficient” in general.

---

## 6. Strongest supporting evidence

1. **Exact algebraic collision (H-A).** A2.1 has zero difference in `sum`,
   `sumsq`, and reader output, while a correspondence statistic changes
   (T_R1 +0.7708, adjacent-pair sum +370.0). This is a clean activation-level
   certificate, not a training artifact.
2. **Real-molecule information control.** A2.2 leaves DCC predictions invariant
   (~3e-08) while T_R1 changes on every tested molecule; the reader cannot see
   the rebinding.
3. **A4 nonredundancy is real under the frozen probes.** `J_latent` AuditEval
   R² 0.774/0.762 and `J_raw` 0.934/0.933; the gate does not stop. The
   nonredundant residual has nontrivial covariance structure (J_latent residual
   covariance PR 1.26/1.91, top1 0.886/0.672).
4. **A positive primary point estimate.** Seed 0 `Gain_J = 0.003688`, 95% CI
   `[0.001335, 0.005937]`; ZJ beats Z on both seeds. This is why the audit does
   not simply declare “no signal.”
5. **Ring-rich exploratory signal.** n=67, `Gain_J` +0.0076 (seed 0) and
   +0.0294 (seed 1), both CIs positive. This is a lead for a targeted
   replication, not a decision-grade result.

## 7. Strongest opposing evidence

1. **The main heads are miscalibrated.** Z/ZJ/ZN AuditEval MAE 0.26–0.32 versus
   raw DCC-L 0.075. The comparison is between bad calibrators, not between a
   good model and a good model plus J.
2. **Two ProbeFit molecules dominate.** Rows 2210 and 3776 account for 96.96%
   of ProbeFit residual SSE; residual std falls from ~1.01 to ~0.176 without
   them. The Ridge head fits their extreme labels and extrapolates. This is a
   split/finite-sample artifact, visible before any AuditEval label read.
3. **Seed-1 instability.** `Gain_J` CI includes 0; top-10 removal gives
   −0.001681; the top 10 rows contribute 0.004744, more than the full gain.
4. **Robust sensitivity disagrees.** Huber `Gain_J` is +0.001332 (seed 0) and
   −0.005105 (seed 1), mean −0.001887; `Gain_huber_vs_N` is −0.000931/+0.000613.
5. **The dim-matched control captures seed 0.** `Gain_vs_N = 0.000430`, CI
   includes 0; ZN vs ZJ residual Pearson 0.987/0.959. Adding 36 dimensions
   without new information is nearly as good on seed 0.
6. **`J_raw` is nearly reconstructible nonlinearly.** MLP R² 0.9765/0.9708,
   close to the 0.98 stop line; the “real increment” is mainly `J_latent`.
7. **No H-D localization.** Support/tail strata are mixed; the tail error share
   is only 15.9%. The error is not specifically a low-support/tail phenomenon.

---

## 8. Unknown variables and limits

- **Audit-head estimator.** A robust or cross-fitted residual head, or an
  outlier-aware ProbeFit, could in principle reveal a stable small J gain. This
  audit cannot distinguish “J has no predictive value” from “this Ridge head
  cannot estimate it.” That is the main reason the conclusion is B rather than
  a stronger negative claim.
- **Statistic-family completeness.** `J_raw` is 32 dims (3 bond types; R1–R4);
  `J_latent` is 4 dims. No 3-body/angle/torsion/chirality terms, no aromatic
  type 4, no learned relation kernel. A different family could behave
  differently; this audit does not exhaust “joint structural relations.”
- **Backbone training horizon.** The DCC-L reference is a fixed 47,400-update
  final checkpoint with Fit8k train MAE ~4e-4. A different horizon or
  regularization could change the residual structure and the head behavior.
  No early stopping or valid-based selection was used.
- **Split randomness.** ProbeFit contains two extreme tail molecules; a
  different structure-only split could change the head fit. The split was
  frozen before results and is not re-randomized.
- **Bootstrap interpretation.** The paired bootstrap is fixed-model sample
  uncertainty over AuditEval molecules, not training-seed uncertainty. Two
  supervised seeds are reported separately; no seed-average “best” is used.
- **External validity.** Only official ZINC train was used; official valid/test
  were not loaded. No claim is made about the official test or about other
  datasets.
- **Causal/chemical interpretation.** A2.2 is an information control, not a
  chemically valid molecule; A4/B4 are predictive associations, not causal
  effects of “adding a relation.”

---

## 9. What changes

- **No new architecture this round.** The result is `NO_JUSTIFIED_NEW_MODEL`
  for the 32-dim R1–R4 + 4-dim latent family. No structured-composition model
  is proposed; no GPU is requested for one.
- **Stop treating A4 nonredundancy as predictive value.** A4 says the
  representation does not contain all of J; B4 says the residual does not
  produce a stable label gain under this diagnostic. Those are separate facts.
- **Fix the audit before the next architecture question.** The two-molecule
  ProbeFit domination is a concrete, reusable lesson: audit heads trained on
  1,000 molecules with MSE are not robust to ZINC's extreme tail. Future audits
  should pre-register robust/cross-fitted heads and an outlier-aware ProbeFit,
  and should report the top-row SSE share before scoring.
- **Prioritize estimation/data mechanisms.** Heavy-tail calibration, label
  noise/uncertainty, support-aware data collection, and split design are the
  higher-value next steps. A targeted ring-rich replication (n=67 now) is the
  only structural follow-up worth considering, and only with a pre-registered
  larger evaluation.
- **Keep A1–A4 as reusable diagnostics.** The exact counterexample, the
  information-control rebinding, the conditional redundancy gate, and the
  one-time label-read protocol are reusable even though the predictive result
  is negative.

---

## 10. Decision record

`decision.json`:

- class **B** — `NONREDUNDANT_BUT_NOT_PREDICTIVE`;
- tags: `AUDIT_HEAD_CALIBRATION_FAILURE`, `AUDIT_HEAD_OUTLIER_DOMINATED`,
  `HUBER_SENSITIVITY_SIGN_DISAGREES`, `PROBEFIT_OUTLIER_SENSITIVE`,
  `SINGLE_MOLECULE_DRIVEN`, `ZN_CONTROL_NEAR_GAIN`;
- `a4_gate_stop: false`;
- no official valid/test used;
- rationale: `Gain_J` positive but a paired 95% interval includes 0; the gain
  is not stable across seeds/rows.

This is a **class B** conclusion, not class A. Class A would require stable
nonredundancy, ZJ > Z on both seeds, ZJ > ZN, a ≥0.003 MAE gain, and paired
interval/extreme-row checks that survive. The seed-1 interval and extreme-row
stability conditions fail, and the ZN comparison is negligible on seed 0.

---

## 11. Reproduction map

- A1: `code/run_forward_audit.py` → `analysis/forward_audit.json`,
  `analysis/nondegeneracy.json`, `analysis/provenance.json`.
- A2: `code/run_algebraic_audit.py` → `analysis/algebraic_counterexample.json`.
- A3: `code/run_joint_stats.py` → `analysis/joint_stats_checks.json`,
  `analysis/joint_stats_distribution.json`, `analysis/joint_stats_corr.json`,
  `analysis/j_raw.npy`, `analysis/j_raw_null.npy`,
  `analysis/j_latent_D{0,1}.npy`, `analysis/j_latent_null_D{0,1}.npy`,
  `analysis/relation_pair_counts.npy`.
- Split: `code/make_split.py` → `configs/split_fit8k_probe_audit.json`.
- A4: `code/run_redundancy.py` → `analysis/redundancy.json`.
- B2: `code/pretrain_ssl_fit8k.py`, `code/train_dccl_fit8k.py` →
  `runs/DCC_L_fit8k_s{0,1}/final.pt` + `train.json`.
- B3: `code/b3_audit_heads.py` → `analysis/b/s{0,1}/freeze.json`,
  head npz, prediction arrays.
- B4: `code/b4_score.py --seed {0,1} --freeze-dir analysis/b/s{0,1}`
  (run from repo root because freeze paths are repo-relative) →
  `analysis/b/s{0,1}/metrics.json`.
- Decision: `code/make_decision.py --metrics analysis/b/s0/metrics.json
  analysis/b/s1/metrics.json --redundancy analysis/redundancy.json --out
  decision.json`.
- Combined summary: `analysis/b/combined_summary.json`.
- ProbeFit outlier diagnostic (ProbeFit labels only):
  `analysis/b/probe_outlier_diagnostic.json`.

**GPU ledger:** actual **1.601 A100·h** (SSL 0.434+0.447; DCC-L 0.332+0.373;
B3 0.0075+0.0079), under the 4.0 soft cap. No GPU job failed or was cancelled.
B4 was CPU-only. See `RUN_LEDGER.md` for run IDs and hashes.

---

## 12. Deviations and disclosures

- **32-dim `J_raw`, not 48:** the actual bond vocabulary is `{1,2,3}`; the
  4-slot 48-dim template is diagnostic only and never used in models/gates.
- **A4 near-constant rule:** zero fraction ≥0.99 or std ≤1e-3, fixed before B,
  applied symmetrically; 11 dims excluded. Including them does not change the
  gate outcome (0.923/0.916).
- **`J_latent_D0` negative ProbeFit R²:** dominated by row 3776; AuditEval and
  MLP remain far below 0.98. Not a passed gate.
- **Main alpha at grid boundary:** seed 0 `100.0`, seed 1 `0.0001`; the grid
  was frozen before B, and no post-hoc change was made.
- **Huber `n_iter=300`:** seed-0 Huber hit the max iteration count; it is a
  sensitivity check, not a converged main result.
- **B4 first seed-0 attempt:** failed at the checkpoint hash check before any
  label read due to a repo-relative path; fixed and rerun. No erroneous result
  was used and no AuditEval label was read twice.
- **A2.2 rebinding:** information control, not a chemically valid molecule and
  not a label counterfactual.
- **Historical SSL:** full-10k SSL checkpoints were only A1–A4 background
  backbones; B used Fit8k-only SSL. The historical valid numbers quoted in
  `RUN_LEDGER.md` are background, not fresh blind evaluations.
- **No official test:** no test labels, predictions, or caches were used.
- **No new model:** no new full prediction architecture, ensemble, distillation,
  residual patching, or hyperparameter/statistic-family search was performed.
