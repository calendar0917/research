# Analysis — `e2e_dictenv_training_protocol_audit_v1`

Workstream Z (ZINC dictionary-environment line), CPU only.  Official ZINC test
never loaded (`official_test_loaded = false` in every payload).  Seed 0 only.

Frozen preregistration:
`notes/e2e_dictenv_training_protocol_audit_v1_preregistration.md`
(sha256 prefix `0fdcef2099cc`), frozen with the module and runner in commit
`ea54826`.  Harness fix (bookkeeping only, no re-training): commit `ced2570`.

Evidence:
`results/e2e_dictenv_training_protocol_audit_v1/comparison.json`,
`analysis_tables.md`, `REPORT.md`, `DECISION.md`,
`shared_prefix/prefix_summary.json`, `fork/fork_integrity.json`,
`control_lr1e3/result.json`, `low_lr1e4/result.json`.

---

## 0. The single question, answered

> How much of the current CSSD-q1 `~0.128–0.130` Top-5 soup plateau comes from
> the fixed training protocol (Adam `lr = 1e-3`, 320 epochs) rather than from
> the architecture?

**At most ~1× the measured soup noise floor.**  In the exact paired fork,
`G_schedule = M_control − M_low_lr = +0.000610` (+1.0× the v2 CPU soup-level
reproducibility floor of `~6e-4`), i.e. **Case A:
`TRAINING_PROTOCOL_NOT_PRIMARY_BOTTLENECK`**.  Both arms finish at
`0.1268–0.1274`, both below the historical `0.130028`, and the shared prefix
alone (epochs 1–280, before any LR change) already soups to `0.129726`.  The
`~0.128–0.130` plateau is not produced by the fixed LR schedule.

The frozen primary metric is the full-run Top-5 soup (the round's operational
definition of "how good a model this protocol delivers").  It is deliberately
robust: it can select the control's transient good epochs.  Section 3 reports
the secondary views, which show a real and large *late-run stability* effect of
the low LR that the soup metric absorbs.

---

## 1. Shared prefix and fork integrity

### 1.1 Prefix health

| item | value |
|---|---|
| epoch-280 valid MAE (fresh) | `0.139555` |
| epoch-280 valid MAE (historical) | `0.141159` |
| tolerance gate (`+0.02`) | PASS → `PREFIX_HEALTHY` |
| epoch-280 train MAE | `0.101473` (bounded, finite) |
| parameters | exactly `97727` |
| dictionary sha256 | `b0c5da98…` (frozen `sdb32`, unchanged) |
| prefix Top-5 soup 1–280 | `0.129726` (members `[239, 258, 265, 277, 280]`) |
| prefix best valid | `0.133978` @ 277 |

All nine prefix-gate checks pass (finite curves, dictionary moved and sha
recorded, both state hashes recorded, exact parameter count, bounded epoch-280
train/valid).  Wall clock `2022.3 s` (7.22 s/epoch, 4 threads), prefix peak RSS `2318 MB`
(continuations `2348 MB`).

### 1.2 Historical trajectory drift (provenance caveat, not a gate failure)

The fresh prefix reproduces the historical run **exactly at epoch 40**
(`Δ = 0.0`) and then drifts chaotically:

| epoch | fresh | historical | Δ |
|---|---|---|---|
| 40 | 0.210843 | 0.210843 | +0.000000 |
| 120 | 0.163887 | 0.171100 | −0.007213 |
| 240 | 0.170078 | 0.154002 | +0.016076 |
| 280 | 0.139555 | 0.141159 | −0.001604 |

over 280 shared epochs: mean |Δ| `0.007038`, median `0.003433`, p90 `0.020866`,
max `0.055604`.  This is consistent with run-to-run nondeterminism of the CPU
training loop (floating-point reduction order / thread scheduling amplified
over epochs, the same phenomenon v2 measured at up to 0.0049 per-epoch between
two runs of one configuration): the two trajectories are different samples of
the same protocol at a level (~7e-3 in valid MAE) **an order of magnitude
larger than the LR effect under test** (~6e-4).  Consequence for attribution: cross-run
comparisons against historical `0.130028` / v2 `0.128723` cannot isolate the
LR schedule; only the paired control-vs-low fork can, and that is the round's
primary metric.

### 1.3 Fork

`FORK_INTEGRITY_OK`, all checks true:

* prediction max |Δ| `0.0` in eval mode **and** `0.0` in train mode with
  dropout active (i.e. the global RNG state is also identical);
* model state dict identical, optimizer `exp_avg` / `exp_avg_sq` / `step`
  identical before the LR edit; after the edit **only** the `lr` field differs;
* the next-epoch batch order hash is identical
  (`ce0df339c4c47620aa64134d8f5f8161cf431e9e757e2d1dfc4db268721fe723`), and all
  `40/40` tail per-epoch `batch_order_hash` values match between arms.

So epochs 281–320 differ between the arms in exactly one variable: the
learning rate (`1e-3` vs `1e-4`).

---

## 2. Primary result (frozen)

| arm | lr 281–320 | best valid | best epoch | **Top-5 soup 1–320** | soup members | members > 280 | tail soup 281–320 | last-10 mean | epoch 320 |
|---|---|---|---|---|---|---|---|---|---|
| CONTROL | `1e-3` | 0.133430 | 311 | **0.127428** | `[258, 277, 297, 311, 313]` | 3 | 0.128055 | 0.146977 | 0.148608 |
| LOW-LR | `1e-4` | 0.127857 | 309 | **0.126818** | `[294, 300, 309, 310, 313]` | 5 | 0.126818 | 0.128832 | 0.128793 |

```text
G_schedule = 0.12742769679048796 - 0.1268179698883905 = +0.0006097269020974572
G / floor  = 1.016
verdict    = TRAINING_PROTOCOL_NOT_PRIMARY_BOTTLENECK   (Case A)
```

Interpretation.  The soup is the best-of-run metric, and the control
continuation still contains transient good epochs (best `0.133430` @ 311,
slightly better than the prefix best `0.133978` @ 277) which the soup selects
(`297, 311, 313`).  Under this metric the two schedules are tied at the floor:
**the attainable model quality of CSSD-q1 does not depend on the late LR.**
This is the correct answer to the frozen question: the `0.128–0.130` plateau
is not a training-protocol artifact.

---

## 3. Secondary views: the low LR is a stability, not a capacity, effect

The frozen secondary metrics separate the arms sharply:

| metric | CONTROL | LOW-LR | difference |
|---|---|---|---|
| epoch-320 valid | 0.148608 | 0.128793 | **+0.019815** |
| last-10 mean (311–320) | 0.146977 | 0.128832 | **+0.018145** |
| tail soup 281–320 | 0.128055 | 0.126818 | +0.001237 (2.1× floor) |
| best valid (full run) | 0.133430 | 0.127857 | +0.005573 |
| generalization gap @320 | 0.049337 | 0.051771 | +0.002435 |

Dynamics (both arms share epoch 280 exactly):

| epoch | CONTROL valid | LOW-LR valid | Δ | CONTROL update | LOW-LR update | CONTROL grad | LOW-LR grad |
|---|---|---|---|---|---|---|---|
| 280 | 0.139555 | 0.139555 | 0.000000 | 0.05040 | 0.05040 | 3.83 | 3.83 |
| 285 | 0.142080 | 0.129705 | +0.012374 | 0.04973 | 0.00495 | 4.14 | 2.63 |
| 290 | 0.147670 | 0.130137 | +0.017533 | 0.04670 | 0.00482 | 4.71 | 2.38 |
| 300 | 0.155437 | 0.128121 | +0.027316 | 0.04556 | 0.00521 | 5.53 | 2.68 |
| 310 | 0.146396 | 0.127895 | +0.018501 | 0.04528 | 0.00566 | 3.75 | 2.39 |
| 320 | 0.148608 | 0.128793 | +0.019815 | 0.04790 | 0.00557 | 4.34 | 2.64 |

Reading:

* The low-LR arm improves **already by epoch 285** (`0.129705`, better than
  the historical epoch-280 value `0.141159`) and stays in `0.1279–0.1301`
  for all 40 tail epochs while the control wanders up to `0.1554`.
* Update norm drops ~9× (`0.0479` → `0.0056`) and pre-clip gradient norm drops
  ~1.6× (`4.34` → `2.64`): the control tail is a high-energy random walk in
  the same good basin; the low-LR tail holds the point.
* Only one soup member is shared between the two arms (epoch **313**).  The
  control keeps prefix members `258/277`; the low-LR arm replaces all five with
  tail epochs — its tail is uniformly near its best, the control's is not.

### 3.1 Train vs valid (frozen `task_vs_valid_delta`)

| arm | train Δ (280→320) | valid Δ (280→320) | reading |
|---|---|---|---|
| CONTROL | −0.002202 | +0.009053 | `overfitting_or_noise` |
| LOW-LR | −0.024452 | −0.010762 | `joint_improvement_late_optimization` |

The control's tail neither fits nor generalizes better — it is late-training
noise.  The low-LR tail is genuine optimization: train and valid fall together
(train `0.1015 → 0.0770`).  Note the smaller *update* norm with a larger
*train improvement*: the low LR follows the loss more efficiently, not just
more slowly.

### 3.2 Reconstruction balance (frozen)

| arm | task-loss Δ | rec-term Δ | rec term @320 |
|---|---|---|---|
| CONTROL | −0.002758 | −1.039e-05 | 4.957e-05 |
| LOW-LR | −0.024893 | −7.801e-08 | 5.988e-05 |

The low-LR gain comes almost entirely from the supervised task loss
(`−0.0249`); the reconstruction term is essentially frozen at its epoch-280
level (`λ·rec ≈ 6.0e-05`).  Neither arm collapses the reconstruction objective
— there is no task/reconstruction trade-off hiding in the result.

### 3.3 Dictionary health (frozen minimal set)

| checkpoint | active | N_eff | top-5 usage | max activation rate | recon (frob) | movement vs fork |
|---|---|---|---|---|---|---|
| epoch 280 prefix | 32 | 20.762 | 3.197 | 0.928 | 0.05257 | — |
| CONTROL @320 | 32 | 20.281 | 3.201 | 0.847 | 0.04954 | 1.0792 |
| LOW-LR @320 | 32 | 20.281 | 3.157 | 0.928 | 0.05442 | 0.2814 |

Both dictionaries stay full (32/32 active, N_eff ≈ 20.3, no collapse, no
atom death).  The low-LR tail barely moves the dictionary (`0.281`), the
control tail churns it (`1.079`, ~3.8× more) while its maximum activation
rate *drops* from `0.928` to `0.847`.  This is consistent with the loss
reading: at `1e-3` the tail is diffusive drift in dictionary/binding space;
at `1e-4` it is a stable descent.

---

## 4. Frozen classification and next step

* Primary: Case A, `G_schedule = +0.000610 < 0.002`.
  Verdict `TRAINING_PROTOCOL_NOT_PRIMARY_BOTTLENECK`.
* Case C/D conditions also fail (`G ≪ 0.005`; `M_L = 0.126818 > 0.125`;
  `> 0.120`).
* `horizon_may_still_be_binding = False`: the low-LR arm's best epoch is
  `309` and its late slope (311→320) is `+7.66e-05` per epoch (vs control
  `+1.51e-03`): no evidence that more than 320 epochs at the fixed protocol
  would keep helping.
* Canonical protocol stays `320@1e-3` for subsequent architecture
  comparisons (the frozen Case-B/C rule that would promote
  `280@1e-3 + 40@1e-4` did not fire).  The low-LR tail is not adopted as
  canonical.
* Next round: the architecture study frozen in the prereg — a
  **Dictionary-Conditioned Semantic Operator**
  (`e_v = Σ_k α_vk f_k(q_v)`), still with **no message passing and no
  environment update**, and not another optimizer/scheduler search.

### 4.1 Historical attribution (Q3 consequence)

The v2 "low-rate warm adaptation" soup (`0.128723`) is *not* attributable to
the low learning rate: the plain from-scratch control tail here soups to
`0.127428` and the shared prefix (before the fork) already soups to `0.129726`
versus the historical `0.130028`.  Given the measured run-to-run drift
(mean |Δ| 0.007 over the same epochs), the historical warm-adaptation gain lay
inside run/basin noise — the same order as, but larger than, the paired LR
effect measured here.  Only the paired fork is decision-grade.

---

## 5. Harness fix — transparency note

The first continuation pass was launched with a runner-side bookkeeping bug:
`_run_continuation` passed `best_seed=tuple(int(v) for v in checkpoint["best"])`,
integer-casting the prefix best **MAE**, so the log showed
`best=0.000000 @ 277` and the stored `best_valid_mae` / `best_epoch` of each
continuation were wrong.

* **What was affected:** only the two derived `best_*` fields in
  `control_lr1e3/result.json` and `low_lr1e4/result.json`.
* **What was not affected:** all training math, both trajectories, every
  per-epoch curve, both soups, the fork-integrity evidence, optimizer states
  and batch order.  No epoch was re-trained.
* **Repair (commit `ced2570`):** the runner now passes
  `(int(epoch), float(mae))`; `_repair_continuation_best` recomputes
  `best_epoch` / `best_valid_mae` from the stored journals, records a
  `harness_fix` note inside each `result.json`, and repairs the resume
  checkpoints.  Re-running `compare --force` printed
  `[repair] TPA-CONTROL-LR1E3: best_valid_mae=0.133430 @ 311` and
  `[repair] TPA-LOW-LR1E4: best_valid_mae=0.127857 @ 309`; `comparison.json`
  was regenerated from the same journals with
  `G_schedule = +0.000610`.  The focused test suite (18/18) includes a
  regression test for the merged-best rule and the repair.

The corrected full-run bests: CONTROL `0.133430` @ **311** (not the prefix's
277 — the control tail did produce one epoch better than the whole prefix),
LOW-LR `0.127857` @ **309**.

---

## 6. Limitations

* **Single seed.**  The paired fork removes seed/architecture variance for the
  LR effect, but the `~6e-4` effect estimate rests on one trajectory pair.
* **Floor-level primary.**  `G/floor = 1.02`: the experiment can exclude a
  *material* LR effect (`≥ 0.003 = 5× floor`) but cannot resolve the sign of a
  sub-floor effect.  This was pre-declared and is not a post-hoc excuse.
* **Metric dependence.**  The frozen soup metric absorbs the low-LR stability
  gain because the control's transient epochs are good enough to enter the
  soup.  A last-epoch or tail-only metric would report a much larger gap
  (`~0.018–0.02`); those are secondary by construction and were not used for
  the verdict.
* **Cross-run attribution blocked by drift.**  Historical comparisons
  (`0.130028`, `0.128723`) are confounded by chaotic trajectory drift of the
  same protocol (`mean |Δ| ≈ 0.007`), which is why Q3 is answered only from
  the paired fork.

## 7. What was deliberately not run

Per the frozen contract: no LR sweep (`3e-4 / 5e-4 / 5e-5`), no scheduler, no
second switch epoch, no horizon extension, no seed 1/2, no architecture /
dictionary / fusion / relation / readout change, no official test access.
Budget spent: `360` epoch-equivalents (280 shared + 40 + 40), wall clock
`2602.0 s` (~43.4 min) at 4 CPU threads, peak RSS `2348 MB`.
