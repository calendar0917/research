# REPORT — LCRC-HC3-v1

**Round:** `lcrc-hc3-v1`
**Track:** `tracks/structural-sufficiency/zinc_lcrc_hierarchical_composition_v1/`
**Status:** **STOPPED AT STAGE A** under the pre-registered stop rule. Seed 1
and Stage C were **not purchased**. No tuning or rescue run was made.
**Formal training commit:** `cccf380143a45af50ba8af54ad485db2b9a2d859`
(`git_dirty = false` in all three formal `train.json` files).
**Host/regime:** `res`, 2×A100-40GB, driver 550.163.01, torch 2.5.1+cu124.
**Official test:** never opened (`hc_gates` g11 refuses `load_cache("test")`,
`cache_path("test")` and `structure.load_split("test")`).

---

## 0. One-screen answer

The hierarchical arm **H** did not clear the Stage A purchase gates:

* `Δ(H−B)` full Monitor MAE = **−0.001025**, versus the required **≤ −0.004**.
* `Δ(H−F)` full Monitor MAE = **+0.000377** (F slightly better), versus the
  required **≤ −0.003**.
* `Δ(H−B)` ordinary = **−0.001612**, versus the required **≤ −0.002**.
* The `W20` and `T100` smoothing gates also fail for H vs F, and the `W20`
  magnitude gate fails for H vs B.

The H vs B improvement is not broad: it is **larger than the whole gain on one
Monitor molecule** (molecule 374). Excluding that one post-hoc row, H is
**+0.000154 worse than B** and **+0.002163 worse than F** on full Monitor. The
one pre-registered support gate that passes pointwise (`unseen_ordinary`,
−0.004731) becomes **−0.000385** without the same row.

This is a negative result for the proposition that hierarchical three-bond
composition is more effective than a nearly capacity-matched direct encoder. It is not
a claim that F is definitively better; it is a failure to demonstrate H's
advantage at the pre-registered margin.

The zero-GPU Stage IV support-vocabulary erratum was completed and is reported
in §2. Stage C (official train10k / valid, mature DCC-L comparison) was not run,
so **no claim about mature single-model performance is made**.

Cost: **1.91 A100·h** (Stage A only), below the 6 h soft / 8 h hard caps.

---

## 1. Pre-registered question and design

The round asks whether a **shared hierarchical three-bond composition operator**
inside the frozen LCRC-A trunk can simultaneously beat

1. a same-information **flat** encoder of the same directed three-bond paths,
   and
2. the original strong LCRC-A baseline,

under official train10k label limits.

Three arms shared the frozen LCRC trunk, recipe and data order:

| arm | composition |
|---|---|
| **B** | frozen LCRC-A, no new composition |
| **H** | shared `F2: 484→96→48` on three-atom fragments, then shared `F3: 252→96→64` over two fragments sharing the centre bond |
| **F** | same-information direct `Fflat: 684→109→64` on the whole four-atom / three-bond path |

The full definitions are in `PROTOCOL.md`; they were frozen before the first
formal GPU run and were not changed after seeing results. The only post-freeze
addition was `code/hc_analysis.py` (read-only analysis) and the descriptive
one-row sensitivity in §5.5; neither touches training, selection or a gate.

---

## 2. Task 0 — Stage IV support-vocabulary erratum (no GPU)

This was an **additive erratum**, not a retraining and not a rewrite of the
historical Stage IV decision. Stage IV was trained on full official train10k,
but its valid support groups had been built with a Fit8k radius-2 vocabulary.
The correct vocabulary is the full official train10k structure.

| quantity | old (Fit8k vocab) | corrected (train10k vocab) |
|---|---:|---:|
| vocabulary signatures | 11,015 | 12,108 |
| any-unseen | 297 | 251 |
| unseen-ordinary | 267 | 225 |
| seen-ordinary | 668 | 710 |
| tail (`y ≤ −3`) | 65 | 65 (unchanged) |

46 rows move from unseen to seen; none move the other way. Full/ordinary/tail
are unchanged by the vocabulary.

REAL−MEAN unseen-ordinary MAE (negative = REAL better):

| seed | old Δ | corrected Δ | corrected 95% CI |
|---:|---:|---:|---|
| 0 | −0.000703 | **−0.002998** | [−0.016662, +0.010513] |
| 1 | −0.012320 | **−0.014535** | [−0.028392, −0.000721] |

The seed-0 unseen effect still does not reach the −0.004 purchase threshold and
its CI crosses zero; seed 1 strengthens. The historical Stage IV decision was
based on the old grouping and is **not rewritten**. Old predictions and
checkpoints were not modified.

Files: `ERRATUM_stage4_support.md`,
`analysis/erratum_stage4_support.json`.

---

## 3. What was implemented

### 3.1 Shared structural interface

For every undirected physical centre bond `e = {j,k}`, the code enumerates every
directed simple three-bond path `(i,j,k,l)` with four distinct atoms, both
centre orientations. Each directed record stores the four atoms, the three
physical bonds, the centre bond, and the three chord types
`(type(i,k), type(j,k), type(i,l))`.

* `q_path ∈ R^64` per directed record.
* `n_e` = number of undirected paths on centre bond `e`.
* `m_e = Σ_oriented q_path / (2√(1+n_e))`, and `m_e ≡ 0` exactly when
  `n_e = 0`.
* `r_v = Σ_{e incident v} m_e / √(1+degree_v)`.
* `W_comp: Linear(64,64)` with zero weight and zero bias, added immediately
  before the frozen `LocalContent` final LayerNorm. Nothing else in the trunk,
  `f`, `rho` or the two-hop statistics changes.

H uses shared `F2` on `(i,j,k)` and `(k,j,i)`, averages the two 48-d fragment
embeddings, then applies shared `F3` to adjacent fragments sharing the centre
bond. F applies one shared `Fflat` to each direction of the whole path and
averages the two 64-d outputs. F is not an average-then-nonlinearity CMA-MEAN
control.

### 3.2 Parameter table

`analysis/parameter_table.csv`:

| arm | total | frozen base | F2 | F3 | Fflat | W_comp |
|---|---:|---:|---:|---:|---:|---:|
| B | 237,519 | 237,519 | 0 | 0 | 0 | 0 |
| H | 323,391 | 237,519 | 51,216 | 30,496 | 0 | 4,160 |
| F | 323,384 | 237,519 | 0 | 0 | 81,705 | 4,160 |

`H − F = 7` parameters. H and F are therefore nearly capacity-matched (7
parameters, 0.002% of the model) for the scientific comparison: the H-vs-F
contrast isolates composition, not extra width.

### 3.3 Initialisation and activation path

* H/F construct the frozen LCRC under `torch.manual_seed(seed)` with the same
  RNG consumption, load the same seed SSL atom/bond encoders, and replace
  `LocalContent` with an identical subclass that accepts the injection vector
  (same state-dict keys and tensors).
* The base trunk of H/F is tensor-wise equal to B for seeds 0 and 1; the
  historical initial hashes match (`11c698ae…` seed 0, `d9422859…` seed 1).
* H/F share the same `W_comp` initial state; it is exactly zero for both.
* Step-0 H/F outputs are **bitwise equal** to B for both seeds.
* The path encoder receives **exactly zero gradient at step 0** (the Jacobian is
  multiplied by the still-zero `W_comp`), then nonzero gradient from step 1
  onward. `W_comp` itself receives a nonzero gradient at step 0.

The formal runs used the exact frozen recipe: AdamW, peak LR 1e-3, min LR 1e-5,
500-update warm-up then per-update cosine, weight decay 1e-5, batch 128, raw-y
L1, clip 5, exactly 47,400 updates. Fit8k has 63 steps/epoch; Monitor/valid is
evaluated every epoch; strict-minimum checkpoint selection.

---

## 4. CPU correctness gates

All **15 gate groups** pass (`analysis/gates/cpu_gates.json`, `all_pass=true`;
gate commit `1fb3e76`, before formal runs). The important ones for this report:

| gate | evidence |
|---|---|
| g1 parameter table | exact counts above; H−F = 7 |
| g2 initial state | H/F trunk tensor-wise equal to frozen builder, seeds 0/1; `W_comp` zero and identical |
| g3 step 0 | H−B and F−B max abs difference **0.0** (bitwise), both seeds |
| g4 enumeration | 19 graphs vs independent brute-force ordered quadruples: 0 path/chord/centre/`n_e` mismatches |
| g5 CMA semantics | 60 real molecules; z triples and path attributes equal to frozen CMA enumeration |
| g6 old pilot contrast | old block is a fixed nontrainable two-hop projection; new path is trainable 64-d three-bond, four distinct atoms |
| g7 shared inputs | H/F/B see identical atom/bond tensors; all path atoms distinct; chords in 0..3 |
| g8 gradient path | `W_comp` nonzero at step 0; path encoder exactly zero at step 0, nonzero/finite from step 1 |
| g9 invariances | renumbering ≤4.8e-7, bond reversal 0, bond-order permutation 0, path-order 0, batch split/merge ≤1.2e-7 |
| g10 reversal/zero | H forward/reverse `q` difference 0; F 1.5e-8; `m_e ≡ 0` for every `n_e=0` bond |
| g11 data boundary | label mutation leaves predictions bit-identical; no label key in forward batch; test split refused by all loaders |
| g12 recipe/checkpoint | LR endpoints/monotonicity; optimizer covers all parameters; state round-trip exact; resumed 126-step run has the same final hash |
| g13 data/norm | split/label hashes match frozen sources; historical full-train norm reproduced; Fit8k norm uses no Monitor rows |
| g14 coverage | 249,279 physical bonds; 41,629 with `n_e=0`; 207,650 with `n_e>0`; max `n_e=9` |
| g15 Stage-C baseline | historical `lcrcA_s0/s1` valid replay matches saved predictions (≤3.9e-6) and MAEs (≤7e-9) |

The full per-gate detail is in `analysis/gates/cpu_gates.json`; a human-readable
summary is in `TREE_LEDGER.md`.

---

## 5. Stage A — Fit8k / Monitor1k, seed 0

### 5.1 Execution and timing

H and F ran in parallel on the two A100s; B ran immediately after on one A100.
All runs completed exactly 47,400 updates.

| arm | params | best epoch | best Monitor MAE | W20 | T100 | final | GPU·h | wall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B | 237,519 | 486 | 0.121258 | 0.121718 | 0.121746 | 0.121743 | 0.487 | 29.2 min |
| H | 323,391 | 626 | 0.120233 | 0.120483 | 0.120516 | 0.120517 | 0.691 | 41.4 min |
| F | 323,384 | 581 | 0.119856 | 0.120167 | 0.120194 | 0.120211 | 0.730 | 43.8 min |

Two-epoch timing before the stage: H 3.04 s/epoch → 0.637 projected GPU·h;
F 3.55 s/epoch → 0.743 projected GPU·h. B was **not separately timed**; the
frozen LCRC-A is bounded above by H/F and by its historical timings, and B's
actual 0.487 h is below both projections. This is recorded as a protocol
deviation in `TREE_LEDGER.md`; it did not affect any purchase decision.

Support groups on Monitor with the Fit8k vocabulary (same construction as the
frozen SSRC audits):

| group | n |
|---|---:|
| full | 1,000 |
| ordinary (`y > −3`) | 926 |
| tail (`y ≤ −3`) | 74 |
| any-unseen | 317 |
| unseen-ordinary | 271 |
| seen-ordinary | 655 |

### 5.2 Main result and paired deltas

Best-checkpoint Monitor MAE. Negative deltas favor the first arm; CIs are
paired bootstrap 5,000 resamples over rows (reported for interpretation, not as
purchase gates; the protocol gates are point estimates).

| slice | B | H | F | Δ(H−B) [95% CI] | Δ(H−F) [95% CI] |
|---|---:|---:|---:|---|---|
| full | 0.121258 | 0.120233 | 0.119856 | −0.001025 [−0.005601, +0.003401] | +0.000377 [−0.005164, +0.005500] |
| ordinary | 0.065893 | 0.064281 | 0.065112 | −0.001612 [−0.005997, +0.002276] | −0.000831 [−0.006411, +0.003539] |
| tail | 0.814065 | 0.820387 | 0.804900 | +0.006322 [−0.025105, +0.041709] | +0.015487 [−0.016363, +0.055692] |
| any-unseen | 0.186512 | 0.185517 | 0.185556 | −0.000995 [−0.014083, +0.011217] | −0.000039 [−0.016493, +0.014837] |
| unseen-ordinary | 0.121260 | 0.116529 | 0.121633 | −0.004731 [−0.018378, +0.006854] | −0.005104 [−0.022879, +0.008669] |
| seen-ordinary | 0.042985 | 0.042664 | 0.041727 | −0.000321 [−0.002842, +0.002146] | +0.000937 [−0.001052, +0.002948] |
| Monitor \ fixed W10 | 0.070759 | 0.067434 | 0.069719 | −0.003326 [−0.007498, +0.000400] | −0.002285 [−0.007580, +0.001997] |
| Monitor \ position 172 | 0.121354 | 0.120334 | 0.119946 | −0.001020 [−0.005760, +0.003302] | +0.000388 [−0.005457, +0.005352] |

`Monitor \ position 172` is the 173rd sorted Monitor row (molecule id 1656);
it is **not** the Stage C valid row 172, which was not evaluated because Stage C
was not run.

The full H−B and H−F deltas are far below the pre-registered margins, and all
full/ordinary CIs include zero. H is pointwise better than B on ordinary and
unseen-ordinary, but H is slightly worse than F on full and seen-ordinary and
substantially worse than both on the tail (+0.0063 vs B, +0.0155 vs F).

### 5.3 Fixed worst-10 check

The fixed `W10` is the top-10 |error| rows of the frozen DCC-L Fit8k seed-0
best-Monitor checkpoint, fixed upstream before any run of this round.

| arm | W10 MAE |
|---|---:|
| B | 5.120592 |
| H | 5.347371 |
| F | 5.083470 |

For H vs B: net `AE(B) − AE(H)` over all 1,000 rows = **+1.0246**, W10 sum =
**−2.2678**, so the W10 share is **−2.213**. The mathematical gate
`share ≤ 0.50` passes, but only because H is **worse** on W10; the net
improvement comes from outside the fixed extremes. This is not evidence that H's
advantage is broad, and it is not counted as a positive.

`Δ(H−B)` on Monitor \ W10 = −0.003326 passes the point threshold of −0.002,
but the full gain is still far below the main gate and the paired CI includes
zero.

### 5.4 Paired per-molecule comparisons

`analysis/per_molecule_stageA_s0.csv` contains every Monitor row (actual
molecule id, target, support flags, penalty flag, all three predictions, AE and
paired deltas).

| slice | H vs B: H better / ties / H worse | H vs F: H better / ties / H worse |
|---|---|---|
| full | 512 / 0 / 488 | 504 / 0 / 496 |
| ordinary | 472 / 0 / 454 | 464 / 0 / 462 |
| tail | 40 / 0 / 34 | 40 / 0 / 34 |
| any-unseen | 159 / 0 / 158 | 159 / 0 / 158 |
| unseen-ordinary | 140 / 0 / 131 | 136 / 0 / 135 |
| seen-ordinary | 332 / 0 / 323 | 328 / 0 / 327 |

H vs B and H vs F are close to coin flips on row counts. The mean H−B gain is
not a broad shift; it is concentrated in a few rows.

### 5.5 One-row sensitivity (post hoc, not a gate)

Molecule **374** (Monitor position 39, `y = 0.575315`) is an ordinary,
Fit8k-unseen row. Predictions:

| arm | prediction | AE |
|---|---:|---:|
| B | −0.729926 | 1.305241 |
| H | 0.448175 | 0.127140 |
| F | −1.335948 | 1.911263 |

H is much closer on this row; F and B are both far below the target. The paired
deltas on this row are `d_HB = −1.1781` and `d_HF = −1.7841`. With 1,000 rows,
this single row contributes **−0.001178** to the full H−B mean and **−0.001784**
to the full H−F mean — more than the entire observed full H−B gain and opposite
to the observed full H−F sign.

Sensitivity, excluding this one row (descriptive only; the pre-registered gates
already fail, so this cannot rescue or change the decision):

| slice | n | Δ(H−B) excluding row 374 | Δ(H−F) excluding row 374 |
|---|---:|---:|---:|
| full | 999 | **+0.000154** | **+0.002163** |
| ordinary | 925 | −0.000340 | +0.001097 |
| any-unseen | 316 | +0.002730 | +0.005607 |
| unseen-ordinary | 270 | **−0.000385** | +0.001485 |
| seen-ordinary | 655 | −0.000321 | +0.000937 |
| Monitor \ fixed W10 | 989 | −0.002138 | −0.000483 |
| tail | 74 | +0.006322 | +0.015487 |

The only pointwise-passing support gate, `unseen-ordinary −0.004731`, is
**entirely** this row: without it the H−B effect is −0.000385. This is why the
report treats the Stage A support "pass" as non-robust rather than as evidence
for the hierarchy.

This row was **not** in the pre-registered fixed W10 list (that list was fixed
from DCC-L, and row 374 was not among DCC-L's ten largest errors). It is not
used as a gate and was not used for model selection.

### 5.6 Curves and smoothing

| arm | best | best epoch | W20 | W20 epoch | T100 | final |
|---|---:|---:|---:|---:|---:|---:|
| B | 0.121258 | 486 | 0.121718 | 640 | 0.121746 | 0.121743 |
| H | 0.120233 | 626 | 0.120483 | 700 | 0.120516 | 0.120517 |
| F | 0.119856 | 581 | 0.120167 | 679 | 0.120194 | 0.120211 |

Smoothing deltas:

* H−B: W20 −0.001235, T100 −0.001230, final −0.001226, best −0.001025.
* H−F: W20 +0.000316, T100 +0.000322, final +0.000305, best +0.000377.

The signs agree with the best-checkpoint signs (H below B; F below H), but the
magnitudes are far below the pre-registered W20 thresholds. H's curve is
slightly better than B's at every summary, yet not by the purchased margin.

### 5.7 Tail and penalty, never pooled

The Monitor tail is 74 rows (`y ≤ −3`). The artificial ring-penalty label
(`label_excess > 0`) comes from the train cycle audit; the 35-row valid penalty
group is not used here because Stage C was not run. On Monitor, 22 tail rows
also carry the penalty label and 52 do not. These groups are descriptive only.

| group | n | B | H | F | Δ(H−B) |
|---|---:|---:|---:|---:|---:|
| penalty / tail | 22 | 2.389286 | 2.433630 | 2.372507 | +0.044344 |
| penalty / ordinary | 14 | 0.164376 | 0.159841 | 0.140592 | −0.004535 |
| no-penalty / tail | 52 | 0.147625 | 0.137861 | 0.141682 | −0.009764 |
| no-penalty / ordinary | 912 | 0.064381 | 0.062814 | 0.063953 | −0.001567 |

H is slightly better than B on no-penalty tail and ordinary rows, but worse on
the penalty-tail rows; F is better than H on penalty-tail and penalty-ordinary.
The groups are small and are **not** pooled into a causal tail claim. The valid
65-row tail vs 35-row penalty decomposition specified for Stage C is absent
because Stage C was not purchased.

---

## 6. Stage A gate evaluation

All values are point estimates on the pre-registered slices. `PASS`/`FAIL` is
exactly the frozen rule; CIs are not used to relax a gate.

| # | gate | threshold | observed | result |
|---:|---|---:|---:|---|
| 1 | Δ(H−B) full | ≤ −0.004 | −0.001025 | **FAIL** |
| 2 | Δ(H−F) full | ≤ −0.003 | +0.000377 | **FAIL** |
| 3 | Δ(H−B) ordinary | ≤ −0.002 | −0.001612 | **FAIL** |
| 4 | Δ(H−B) unseen-ordinary | ≤ −0.004 | −0.004731 | PASS (pointwise; one-row-sensitive) |
| 5 | Δ(H−B) Monitor \ W10 | ≤ −0.002 | −0.003326 | PASS |
| 5 | W10 share of net AE reduction | ≤ 0.50 | −2.213 | PASS (H worse on W10) |
| 6 | W20(H) − W20(B) | ≤ −0.004 | −0.001235 | **FAIL** |
| 6 | W20(H) − W20(F) | ≤ −0.003 | +0.000316 | **FAIL** |
| 6 | T100(H) − T100(B) | < 0 | −0.001230 | PASS |
| 6 | T100(H) − T100(F) | < 0 | +0.000322 | **FAIL** |
| 6 | final(H) − final(B) | ≤ 0 | −0.001226 | PASS |
| 6 | final(H) − final(F) | ≤ 0 | +0.000305 | **FAIL** |

**Verdict: STOP.** Four of the six numbered gate groups fail, including both
primary full-MAE gates and the ordinary gate. The two support/extreme-row gates
that pass do not compensate: one is one-row-driven and the other passes because
H is worse on the fixed extremes. The protocol's stop rule is unconditional:
no seed 1, no Stage C, no width/LR/path/loss changes.

---

## 7. Decision and interpretation

### 7.1 What is supported

* The implementation is correct at the level the CPU gates can establish:
  enumeration/chords/ownership, reversal and renumbering invariance, shared
  inputs, exact frozen initialisation, bitwise step-0 equality, the prescribed
  gradient path, label/test boundary, recipe, optimizer, checkpoint/resume and
  parameter counts.
* On the internal Fit8k/Monitor split, adding three-bond path information
  changes predictions slightly: H and F are both a little below B at the best
  checkpoint and on the late curve.
* The Stage IV support erratum is a real correction: the unseen set is smaller
  under the full train10k vocabulary, and the seed-1 unseen-ordinary REAL
  advantage strengthens.

### 7.2 What is **not** supported

* **“Hierarchical composition is more effective.”** H is not better than the
  same-information, nearly capacity-matched direct encoder F at the
  pre-registered margin; the point
  estimate is slightly worse (+0.000377 full), and it is worse by +0.002163
  when one row is removed. The H−F comparison is the clean capacity-controlled
  test, and it fails.
* **“H improves the mature single model.”** Stage C was not run. There is no
  train10k/valid result for H or F in this round, no DCC-L comparison, and
  therefore no mature-performance claim. The Stage A split is an internal
  diagnostic, not the mature benchmark.
* **“Three-bond information broadly helps.”** The nominal H−B gain is
  −0.001025, below the −0.004 gate, and is one-row-driven. Row counts are
  essentially tied (512/488), and the full/ordinary CIs include zero. A small
  effect is possible, but this experiment did not establish it.

### 7.3 Why the result is still informative

The design intentionally compares H against a **same-information, nearly
same-capacity** flat encoder. H and F differ by 7 parameters and see the same
paths, atoms, bonds and chords. The failure of H to beat F therefore isolates
the hierarchical sharing/ordering, not extra capacity or extra structural
information. The strong LCRC-A baseline B is also healthy: B's best Monitor MAE
(0.121258) is better than the historical DCC-L Fit8k reference (0.123151), so H
was not competing against a weak baseline.

The one-row sensitivity is a warning about mean-MAE conclusions on 1,000 rows
with heavy-tailed targets: a single large-error molecule can reverse the sign of
a 0.001-level comparison. It is reported for transparency, not used to rewrite
the pre-registered gate.

### 7.4 Separation of claims

| claim | status |
|---|---|
| Hierarchical composition is more effective than same-information flat encoding | **not supported** (Stage A H−F fails; one-row-sensitive) |
| Hierarchical composition improves mature single-model performance | **not tested** (Stage C not purchased) |
| Three-bond path information helps LCRC-A | **weak/not established** (H−B below gate, one-row-driven) |
| The implementation follows the frozen interface and gradient path | **supported by CPU gates** |
| Stage IV unseen grouping should use full train10k vocabulary | **corrected by erratum** |

---

## 8. Budget, provenance and reproducibility

### 8.1 Budget

| item | GPU·h |
|---|---:|
| B seed 0 Fit8k | 0.487 |
| H seed 0 Fit8k | 0.691 |
| F seed 0 Fit8k | 0.730 |
| 2-epoch timing (H/F) | ~0.004 |
| **Stage A total** | **1.911** |
| Stage B (seed 1) | not purchased |
| Stage C (train10k) | not purchased |

Caps: soft 6, hard 8 A100·h, max two GPUs. Stage A used two GPUs in parallel for
H/F and one for B. Spending stopped at 1.91 h; no rescue run.

`analysis/budget.json` records the same numbers and the B-timing deviation.

### 8.2 Provenance

* Protocol/config: `PROTOCOL.md`, `configs/frozen_protocol.json`,
  `configs/freeze_manifest.json`. The setup header date `2026-11-07` in the
  frozen protocol/config is a typo; git commits and run logs are `2026-10-30`.
  The frozen files were left byte-identical. The manifest's `git_dirty=true`
  reflects untracked gate outputs at generation time; the code/cache hashes are
  explicit and the formal runs themselves record `git_dirty=false`.
* Formal training commit: `cccf380143a45af50ba8af54ad485db2b9a2d859`;
  `git_dirty=false` in B/H/F `train.json`.
* Structure caches (hashes match the freeze manifest):
  * train `c1fe632123b71e5966aaa3c4b7ddc54c5518915f2102fac0116fef1afd449dd0`
  * valid `de4f43b8ae8edab977984357f6d4c7664b39034eec25c20604cbbdb17273c7b3`
* Formal checkpoint SHA-256 (best): B `5cfc7874…`, H `67edefe7…`,
  F `757ffd43…`; full hashes are in each `train.json` and the run directory.
* Data boundary: only official train10k and valid1k caches were read; official
  test was never constructed (g11).
* Historical baselines: LCRC-A `lcrcA_s0/s1` checkpoints and saved predictions
  were replayed in g15; DCC-L per-molecule predictions are the frozen
  `zinc_global_operator_search_v1` artifact.

### 8.3 Artifacts

| path | contents |
|---|---|
| `analysis/stageA_s0.json` | full Stage A metrics, deltas, gates, support, provenance |
| `analysis/stageA_s0.md` | human-readable Stage A tables |
| `analysis/per_molecule_stageA_s0.csv` | every Monitor row, predictions, AE, paired deltas, flags |
| `analysis/per_molecule_stageA_s0.npz` | same data in NumPy form (gitignored) |
| `analysis/parameter_table.csv` | B/H/F parameter accounting |
| `analysis/budget.json` | actual and projected cost |
| `analysis/gates/cpu_gates.json` | all 15 CPU gate groups |
| `analysis/erratum_stage4_support.json` | machine-readable erratum |
| `ERRATUM_stage4_support.md` | erratum narrative |
| `TREE_LEDGER.md` | chronological execution/fix/budget record |

---

## 9. What was not run, and why

* **Stage B (seed 1, Fit8k/Monitor):** not purchased. Stage A failed the
  pre-registered gates; running seed 1 would violate the protocol's stop rule
  and could only be a rescue attempt.
* **Stage C (train10k/valid, H/F and B/DCC-L):** not purchased. It requires
  both seeds to support the same proposition. There are therefore no Stage C
  valid tables, no DCC-L paired comparisons, and no valid 65-tail/35-penalty
  decomposition in this report. Their absence is deliberate, not an omitted
  positive result.
* **Official test:** never opened.

The correct conclusion is the pre-registered one: at the tested margin, the
hierarchical three-bond composition did not demonstrate an advantage over the
same-information flat encoder or the strong LCRC-A baseline, so the round stops
and reports the negative result as-is.
