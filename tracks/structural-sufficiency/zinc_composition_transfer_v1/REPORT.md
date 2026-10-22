# zinc-composition-transfer-v1 — does a cross-group update constraint help, and does local-environment label support matter?

Round: `tracks/structural-sufficiency/zinc_composition_transfer_v1/`. Protocol and
every pre-registered number: [`PROTOCOL.md`](PROTOCOL.md) (with amendments A1, A2).
Machine-readable results: `analysis/summary.json`, `analysis/main_table.md`,
`analysis/groups.md`, `analysis/support_summary.json`, `analysis/support.md`,
`analysis/kappa_calibration.json`, `analysis/split_strength.json`,
`analysis/prep_freeze.json`, `analysis/cpu_checks.json`,
`analysis/valid_per_molecule.csv`, `analysis/run_records.json`,
`analysis/budget.json`, `analysis/rr_jobs.json`.

Observation, interpretation and action are separated: sections 3–7 are
observations (each with its measurement definitions), section 8 is interpretation,
sections 9–11 are the pre-registered decision, the costs and the deviations.
Section 0 states the boundary of what was and was not tested.

## 0. Boundary

* **Target, unchanged:** official ZINC subset raw-`y` **test** MAE in
  `[0.050, 0.060)`. That target was **not** reached, and the official test split
  was **not opened** this round (section 9 explains why under the pre-registered
  rule). No test number appears anywhere in this report.
* The round asks two narrow questions on one fixed platform:
  1. does an explicit cross-group update constraint (one-step look-ahead on a
     source/query split, random `R` vs structure-derived `S`) change
     generalisation, and does the *structural* choice of the split add anything
     over a random one?
  2. does the *supervised label support* of rare local environments change error
     on molecules carrying exactly those environments?
* Not reopened: T/N channels, Kekulé/aromatic training, exact key-lookup residual,
  generic masked-SSL auxiliary heads, SAM scans, full GRIT, pair/spectral trunks,
  fusion, new attention blocks, external labels.
* One model, one architecture, one recipe, one frozen SSL encoder, two supervised
  seeds. Nothing here is a claim about "structural sufficiency" as a programme;
  it is a measurement of two interventions on a single training platform.

## 1. Fixed before the first formal run

| item | value |
|---|---|
| model | `P3NullModel` (P3-Null graph), **413,505** parameters |
| init | frozen seed-0 train-only masked-SSL encoder, sha256 `006fabc1…2bfa97`, loaded for both supervised seeds |
| data | official ZINC subset train 10,000 / valid 1,000, raw-`y` L1; test untouched |
| recipe B | AdamW 1e-3→1e-5, 500-step warm-up then per-step cosine, wd 1e-5, betas (0.9,0.999), eps 1e-8, clip 5, FP32, dropout 0, batch 128, 79 updates/epoch, 600 epochs = **47,400** updates |
| arms | `E` ERM on the whole batch; `R` look-ahead on random halves; `S` look-ahead on the structural split |
| shared prefix | 50 epochs of ERM, forked at the end of epoch 50 (model + AdamW + generator + step); `R`/`S` resume from that fork |
| objective | `J = (|A|/|B|)·L_A(θ) + (|Q|/|B|)·L_Q(θ′)`, `θ′ = θ − α_t·min(1, 5/(‖g_A‖+ε))·g_A`, `α_t = κ·lr_t·ramp_t`, ramp 0→1 over epochs 51–100, exact double backward (`create_graph=True`, nothing detached) |
| κ | calibrated on the seed-0 epoch-50 fork (table in §5) → **0.1**, frozen for every arm and both seeds |
| selection | best valid, strict `<`, earlier epoch on tie; full 1000-molecule valid every epoch |

Contract verification from the run records (`analysis/summary.json →
verification`, `analysis/run_records.json`):

* all three arms of a seed have the **same** `init_state_sha256`
  (`21963b43…` for seed 0, `750be561…` for seed 1);
* identical recipe dicts, identical κ across `R`/`S`;
* the per-epoch batch-union hash equals across the three arms for all 600 epochs
  (`8eca27da…` seed 0, `3f6dc33f…` seed 1 at epoch 1) — verified inside the
  trainer against the frozen partition file, and against the `E` curve afterwards;
* the virtual step left parameters, buffers and AdamW state untouched, and `α = 0`
  reproduces whole-batch ERM numerically (loss diff 0.0, gradient max diff 2.4e-7).

## 2. Pre-training checks and intervention strength

`analysis/cpu_checks.json` — all six groups **PASS**:

| check | result |
|---|---|
| `p3b_replay` | stored P3-Null checkpoint replays its recorded valid MAE (0.076224) with max abs diff 9.5e-7 on 32 rows |
| `init_parity` | `E`/`R`/`S` bitwise identical per seed; seed 0 vs 1 differ only outside `encoders.` |
| `alpha_zero` | `J(α=0)` equals ERM: loss diff 0.0, gradient max diff 2.4e-7, batch-split prediction diff 4.2e-7 |
| `toy_meta_gradient` | exact double backward matches finite differences (5.6e-4); the detached first-order variant deviates by 0.90 |
| `real_meta` | finite meta gradients; virtual step leaves parameters/buffers/optimizer state unchanged; both halves are read (label-shift probes) |
| `keys_and_masks` | `k1 = f(k2)`; masks equal `R` plus the recorded swaps; `|R| = n//2` |

Structural split (`analysis/split_strength.json`, measured on 256 official-train
batches before any valid read; the 64 → 256 proposal increase was the single
pre-authorised construction fix, applied once):

| | 64 proposals (pre-fix) | **256 proposals (declared)** | threshold |
|---|---:|---:|---|
| mean symmetric `d`, `R` | 0.0867 | 0.0867 | — |
| mean symmetric `d`, `S` | 0.1300 | **0.1648** | — |
| Δ symmetric `d` | +0.0433 | **+0.0781** | ≥ 0.02 ✔ |
| Δ eligible-conditioned unseen ratio | +0.0844 | **+0.1519** | ≥ 0.10 ✔ |
| Δ radius-1 unseen rate | +0.0032 | +0.0052 | ≤ 0.01 ✔ |
| standardised size/ring difference | 0.112 / 0.112 | 0.111 / 0.115 | ≤ max(`R`, 0.25) ✔ |

The same numbers on the full 47,400-batch partition file: Δ`d_sym` 0.0788, Δ`d_elig`
0.1528, `S > R` on 99.98% of batches. Gate verdict **PASS**; 37.2 of 256 proposals
accepted per batch on average.

Supervision-support package (`analysis/support_manifest.json`), built from
structure only, before any error was computed: `H` = 801, `Q_target` = 350,
`D_plus` = 451, `Q_background` = 650, `Q` = 1000, `M` = 1000, `D_minus` = 451,
`B` = 7098, `F_plus` = `F_minus` = 7549; 90 target keys, ≥ 2 donors and ≥ 1 query
per key, `F_minus` target-key support exactly 0, `Q`/`M` free of target keys,
max |SMD| of the `D_plus`/`D_minus` matching covariates 0.207, mean match distance
3.33.

## 3. Core result — three arms, two seeds, κ = 0.1

Official valid split, raw-`y` L1. `W20` = min over 20-consecutive-epoch windows of
the window's **median** valid MAE; `T100` = median valid MAE over epochs 501–600.
Both are computed on the epoch-level mean absolute error, so they are not affected
by per-arm best-epoch selection.

| run | best valid | @epoch | final (600) | eval-train@best | W20 | T100 |
|---|---:|---:|---:|---:|---:|---:|
| E_s0 | 0.081546 | 340 | 0.083225 | 0.025066 | 0.083187 | 0.083242 |
| R_s0 | 0.080206 | 349 | 0.081533 | 0.017410 | 0.081465 | 0.081522 |
| S_s0 | 0.081490 | 409 | 0.082348 | 0.011646 | 0.082343 | 0.082378 |
| E_s1 | 0.079243 | 404 | 0.081098 | 0.012382 | 0.081160 | 0.081210 |
| R_s1 | 0.077753 | 331 | 0.079516 | 0.019336 | 0.079377 | 0.079490 |
| S_s1 | 0.075678 | 304 | 0.079375 | 0.025352 | 0.079283 | 0.079405 |

Seed means: `E` 0.080394 / W20 0.082173 / T100 0.082226; `R` 0.078979 / 0.080421 /
0.080506; `S` 0.078584 / 0.080813 / 0.080892.

Paired contrasts (positive = the arm is worse):

| contrast | best valid | W20 | T100 | final valid |
|---|---:|---:|---:|---:|
| `R` − `E` s0 | −0.001340 | −0.001722 | −0.001720 | −0.001692 |
| `S` − `E` s0 | −0.000056 | −0.000844 | −0.000864 | −0.000876 |
| `S` − `R` s0 | +0.001284 | +0.000878 | +0.000856 | +0.000815 |
| `R` − `E` s1 | −0.001490 | −0.001783 | −0.001719 | −0.001581 |
| `S` − `E` s1 | −0.003565 | −0.001877 | −0.001805 | −0.001723 |
| `S` − `R` s1 | −0.002075 | −0.000094 | −0.000085 | −0.000141 |

What survives a robustness read (`analysis/valid_per_molecule.csv`, epoch-600
checkpoint, i.e. no best-epoch luck):

| | mean ΔAE | median ΔAE | molecules better | sign-test p | mean ΔAE excl. row 172 |
|---|---:|---:|---:|---:|---:|
| `R`−`E` s0 | −0.00169 | −0.0003 | 50.5% | 0.78 | −0.00107 |
| `S`−`E` s0 | −0.00088 | +0.0000 | 49.8% | 0.92 | −0.00070 |
| `S`−`R` s0 | +0.00082 | +0.0002 | 49.3% | 0.68 | +0.00036 |
| `R`−`E` s1 | −0.00158 | +0.0008 | 48.5% | 0.36 | −0.00022 |
| `S`−`E` s1 | −0.00172 | −0.0006 | 51.7% | 0.30 | −0.00113 |
| `S`−`R` s1 | −0.00014 | −0.0006 | 51.5% | 0.36 | −0.00091 |

Trimmed mean AE (drop each arm's worst 1% of molecules) at epoch 600:
`E` 0.06652 / 0.06421, `R` 0.06619 / 0.06375, `S` 0.06621 / 0.06328 (s0 / s1).
So of the ≈ 0.0017 mean-AE difference, ≈ 0.0003–0.0008 survives dropping the ten
worst molecules, and the median per-molecule error does not move.

Fixed four-group valid errors (710 seen+ordinary / 225 unseen+ordinary /
39 seen+tail / 26 unseen+tail):

| run | all | seen_ordinary | unseen_ordinary | seen_tail | unseen_tail | drop row 172 |
|---|---:|---:|---:|---:|---:|---:|
| E_s0 | 0.08155 | 0.04989 | 0.14082 | 0.19104 | 0.26883 | 0.07922 |
| R_s0 | 0.08021 | 0.04828 | 0.13942 | 0.17631 | 0.29552 | 0.07688 |
| S_s0 | 0.08149 | 0.04802 | 0.14164 | 0.20236 | 0.29371 | 0.07713 |
| E_s1 | 0.07924 | 0.04689 | 0.13228 | 0.22191 | 0.28988 | 0.07391 |
| R_s1 | 0.07775 | 0.04798 | 0.13852 | 0.15065 | 0.25542 | 0.07416 |
| S_s1 | 0.07568 | 0.04930 | 0.13312 | 0.09899 | 0.26400 | 0.07497 |

Two facts about the valid split dominate any 1e-3 comparison on it. Row 172
(`y = −20.34`, seen+tail) has an absolute error of 5.6 (E_s0) / 7.8 (E_s1) at
epoch 600 — one molecule out of 1000 carries 7–10% of the total MAE, and it moves
between arms by more than the arm differences (S_s1 0.7 vs E_s1 5.3 at the best
epoch). And the 225 unseen+ordinary molecules carry 39% of the MAE while nothing
in this round moved them: every arm sits at 0.132–0.142 there, in both seeds.

## 4. What the mechanism actually did

Fixed train-only probe episodes (32 batches × both partitions), measured inside
the training runs at epochs 100/150/300/600 (`probe_epochs.json`):

| run | epoch | median per-molecule |Δf| after one virtual step | parameter step | cos(J-grad, ERM-grad) | relative gradient difference |
|---|---:|---:|---:|---:|---:|
| R_s0 | 100 / 300 / 600 | 0.0032 / 0.0034 / 0.0001 | 3.6e-4 / 2.6e-4 / 5.0e-6 | 0.998 / 0.999 / 1.000 | 5.6% / 2.8% / 3.0% |
| S_s0 | 100 / 300 / 600 | 0.0041 / 0.0027 / 0.0000 | 4.6e-4 / 2.2e-4 / 4.4e-6 | 0.998 / 0.995 / 0.991 | 6.2% / 10.4% / 12.9% |
| R_s1 | 100 / 300 / 600 | 0.0043 / 0.0027 / 0.0001 | 4.5e-4 / 2.4e-4 / 5.0e-6 | 0.998 / 0.996 / 0.998 | 6.0% / 12.2% / 7.7% |
| S_s1 | 100 / 300 / 600 | 0.0057 / 0.0025 / 0.0000 | 4.7e-4 / 2.4e-4 / 3.9e-6 | 1.000 / 0.988 / 0.992 | 2.7% / 12.4% / 14.7% |

Three observations. (i) The intervention is a **small perturbation of the ERM
update direction**: cosine 0.99–1.00, relative gradient change 3–15%. (ii) It
**decays with the learning rate** — `α_t = κ·lr_t·ramp_t` and `lr` falls from
9.6e-4 at epoch 50 to 1e-5 at epoch 600, so by the last 100 epochs the virtual step
moves predictions by ~1e-4, i.e. the constraint is essentially switched off exactly
where `T100` is measured. (iii) The structural split does change the *source*
gradient direction more than the random split (cos 0.93–0.96 vs 0.99), but that
larger perturbation did not translate into a better valid result.

Training-set fit did **not** degrade: eval-mode train MAE at the best epoch is
0.0174 (`R_s0`) and 0.0116 (`S_s0`) against 0.0251 for `E_s0`; at epoch 600 the
online train MAE is 5.8e-4–6.4e-4 for the meta arms versus 6.0e-4–6.6e-4 for `E`,
i.e. all three arms essentially memorise the 10,000 training molecules (raw-`y` L1). The
pre-registered "common severe underfitting" repair was therefore not triggered.

## 5. κ calibration (why the intervention is weak)

Frozen on the seed-0 epoch-50 fork, 32 train-only batches × {R,S} = 64 episodes,
`ramp = 1`, `α` computed at the epoch-50 learning rate 9.868e-4:

| κ | α | source L1 not worse | median per-molecule |Δf| | mean parameter step | admissible |
|---|---:|---:|---:|---:|---|
| **0.1** | 9.87e-5 | 1.000 | **0.00542** | 4.9e-4 | **yes** |
| 1 | 9.87e-4 | 1.000 | 0.0540 | 4.9e-3 | no (above window) |
| 10 | 9.87e-3 | 0.000 | 0.534 | 4.9e-2 | no |
| 100 | 9.87e-2 | 0.000 | 6.88 | 0.49 | no |

The declared rule (≥75% source-stable, median displacement in [0.005, 0.05],
closest to 0.02) selects κ = 0.1 — the only admissible grid point, at the weak edge
of the window. Amendment A1, written **before any meta arm started**, pre-registered
a mid-window escalation κ = 0.4 with an explicit trigger and secondary status.

## 6. Secondary: mid-window κ = 0.4 (pre-registered A1)

Trigger fired (`R`−`E` and `S`−`E` both within ±0.002 on W20 and best valid in
seed 0). Same fork, same 600 epochs, same recipe; only κ changed.

| run | best valid | @epoch | final (600) | eval-train@best | W20 | T100 |
|---|---:|---:|---:|---:|---:|---:|
| R_s0_k04 | 0.078529 | 486 | 0.079416 | 0.006513 | 0.079352 | 0.079411 |
| S_s0_k04 | 0.080360 | 405 | 0.081800 | 0.014584 | 0.081768 | 0.081819 |
| R_s1_k04 | 0.078801 | 413 | 0.079635 | 0.011420 | 0.079591 | 0.079644 |

| contrast | best valid | W20 | T100 | mean ΔAE [95% CI] | median ΔAE | molecules better | sign p |
|---|---:|---:|---:|---|---:|---:|---:|
| `R`@0.4 − `E` (s0) | −0.003017 | −0.003835 | −0.003831 | −0.00302 [−0.00811, +0.00273] | −0.0023 | 53.5% | 0.029 |
| `R`@0.4 − `R`@0.1 (s0) | −0.001677 | −0.002113 | −0.002112 | −0.00168 [−0.00394, +0.00081] | −0.0019 | 53.1% | 0.054 |
| `S`@0.4 − `E` (s0) | −0.001186 | −0.001419 | −0.001422 | −0.00119 [−0.00672, +0.00506] | −0.0022 | 53.4% | 0.034 |
| `S`@0.4 − `S`@0.1 (s0) | −0.001130 | −0.000575 | −0.000559 | −0.00113 [−0.00381, +0.00159] | −0.0001 | 50.5% | 0.78 |
| `R`@0.4 − `E` (s1) | −0.000442 | −0.001569 | −0.001566 | −0.00044 [−0.00311, +0.00234] | +0.0002 | 49.7% | 0.87 |
| `R`@0.4 − `R`@0.1 (s1) | +0.001048 | +0.000214 | +0.000153 | +0.00105 [−0.00313, +0.00746] | −0.0021 | 53.1% | 0.054 |

Reading: the escalation fired in seed 0 and **did not reproduce in seed 1**. At
κ = 0.4 the random-split arm is −0.0038 on W20 in s0 but −0.0016 in s1 — the same
−0.0016…−0.0018 it already had at κ = 0.1. The per-molecule distribution agrees:
s0 shifts (median −0.0023, sign test p = 0.029), s1 does not (median +0.0002,
p = 0.87). The structural arm barely moves between κ = 0.1 and κ = 0.4
(−0.0006 on W20, against −0.0021 for the random split). So the larger seed-0 number is a seed-specific fluctuation, not
a dose-response, and the pre-declared 0.003 practical scale is not met
reproducibly. The seed-1 confirmation is why this is stated as a non-result rather
than as a gain.

## 7. Supervision-support intervention (independent question)

Both fits see 7549 labelled train molecules: `B` (7098) plus either `D_plus` (451,
donors carrying the 90 target radius-2 keys) or `D_minus` (451 structurally matched
molecules with zero target-key support). Selection only on the internal `M` split
(target-free); `Q_target` labels were read only after all eight checkpoints were
written and hashed (`analysis/support_predictions.json`; the eight checkpoint
hashes are also in `analysis/run_records.json`).

| fit | seed | best `M` MAE | @epoch | `Q_target` MAE | `Q_background` MAE |
|---|---|---:|---:|---:|---:|
| plus | 0 | 0.082108 | 513 | 0.082321 | 0.10144 |
| minus | 0 | 0.080424 | 424 | 0.124316 | 0.10041 |
| plus | 1 | 0.079958 | 439 | 0.082108 | 0.10169 |
| minus | 1 | 0.077273 | 420 | 0.115932 | 0.10290 |

| checkpoint | seed | Δ = plus − minus on `Q_target` [95% CI] | `Q_background` Δ | difference-in-differences | key-level Δ [95% CI] | keys better |
|---|---|---:|---:|---:|---|---:|
| best-`M` | 0 | **−0.0420** [−0.0859, −0.0162] | +0.0010 | **−0.0430** | −0.1032 [−0.2572, −0.0198] | 60/90 (p 0.002) |
| epoch 600 | 0 | −0.0413 [−0.0847, −0.0157] | +0.0031 | −0.0444 | −0.1019 [−0.2546, −0.0191] | 61/90 |
| best-`M` | 1 | **−0.0338** [−0.0757, −0.0100] | −0.0012 | **−0.0326** | −0.0920 [−0.2427, −0.0115] | 62/90 (p 0.0004) |
| epoch 600 | 1 | −0.0358 [−0.0780, −0.0119] | −0.0025 | −0.0334 | −0.0953 [−0.2464, −0.0143] | 60/90 |

Per-molecule: 57–59% of the 350 query molecules improve (sign test p 0.0005–0.006),
median ΔAE −0.017 (s0) / −0.011 (s1). Unlike the look-ahead arms, this gain is not a
tail effect: dropping the worst 1% of query molecules leaves 0.0754 (plus) against
0.0982 (minus) in s0 and 0.0742 against 0.0909 in s1. `Q_background` does not move
(plus 0.1014/0.1017 vs minus 0.1004/0.1029, |Δ| ≤ 0.003), so the effect is specific
to the environments whose label support changed. Strata
by target-key training support: Δ = −0.093 (support 4–7), −0.020 (8–15), −0.024
(16–25) in s0; −0.083 / −0.017 / −0.015 in s1 — the rarest environments gain most,
as the mechanism predicts. `Q_target` labels are *not* harder on average than
`Q_background` (y std 1.875 vs 2.114), so this is not a hard-subset artefact.

The plus fit is slightly **worse** on the target-free monitor `M` in both seeds
(+0.0017 / +0.0027), so the `Q_target` gain is not a selection artefact.

## 8. Interpretation

1. **The look-ahead constraint helps a little, and not because of structure.** Both
   meta arms beat the ERM arm in both seeds on the two sustained metrics
   (W20/T100) at κ = 0.1: −0.00172/−0.00178 (`R`), −0.00086/−0.00188 (`S`). The
   direction is consistent; the size is below the pre-declared practical threshold
   (0.003). The κ = 0.4 escalation reached −0.0038 in seed 0 but −0.0016 in seed 1,
   so there is no dose–response and the single larger number is not a result.
2. **The structural partition adds nothing over a random one.** `S`−`R` is +0.0009
   (s0) and −0.0001 (s1) at κ = 0.1, and `S`@0.4 is still worse than `R`@0.4. The
   pre-registered branch applies verbatim: *if R and S are equally effective, the
   conclusion is "general training constraint", not "structure mechanism"*. The
   structural split did perturb the source gradient more (cos 0.93–0.96 vs 0.99) —
   the extra perturbation is measurable in the gradient and absent in the result.
3. **Where the small gain comes from is the tail, not the bulk.** Median
   per-molecule error does not move; the trimmed mean moves by 0.0003–0.0008; the
   gain is carried by a handful of high-|y| molecules, and one of them (row 172)
   can flip the sign of a best-valid comparison by itself. Any claim resting on
   1e-3 differences on this valid split should be treated as fragile.
4. **The unseen+ordinary group — 39% of the MAE — is untouched** by both
   interventions (0.132–0.142 in every arm). This is the same wall the earlier
   rounds hit; an update constraint on the same data does not buy coverage.
5. **The intervention was structurally weak by construction.** Because
   `α_t ∝ lr_t` and the loss is L1, the look-ahead is a 3–15% perturbation of the
   ERM direction that decays to ~0 in the last 100 epochs. The κ escalation tested
   whether the weakness was merely a scale problem: it produced a big shift in
   seed 0 and none in seed 1, so "under-powered" is **not** established either. What
   is established is that this objective, as specified, cannot be relied on to
   produce a reproducible ≥0.003 valid gain. A round that wants to test MLDG-style
   look-ahead on this platform should freeze a **constant** α (not lr-coupled), use
   a smooth inner loss, and pre-register 2–3 seeds before reading any of them.
6. **Label support, by contrast, is a large lever.** On molecules whose rare local
   environments had their labelled examples removed, restoring 451 such molecules
   halves their error (0.116→0.082, 0.124→0.082) while leaving matched background
   molecules unchanged (DiD −0.033/−0.043). This is the strongest and most
   reproducible signal in the round, and it is a statement about *supervision
   placement*, not about architecture: the P3 model can use a local environment it
   has labels for, but it cannot invent one from structure alone.
7. **Nothing here reaches the target.** Every arm sits at 0.076–0.083 valid, i.e.
   25–38% above the 0.060 bar. The interventions move valid by 1e-3 while the gap is
   2e-2 — a factor of twenty — and the target is a *test* number that this round did
   not measure.

## 9. Terminal test decision

Per `PROTOCOL.md` §8, the official test is read only if a single-model
configuration shows repeatable valid progress at the declared practical scale. The
best configuration (`R`@0.4, seed-mean W20 0.0794) improves W20 by 0.0027 on
average over `E`, but that average is built from 0.0038 (s0) and 0.0016 (s1), i.e.
the scale is crossed in one seed only; `R`@0.1, the pre-registered primary, is
−0.00175 in both seeds, below the threshold. The absolute levels (W20 0.0794–0.0796,
T100 0.0794–0.0796) are far from the 0.075 exploration gate and from the 0.060 bar,
and no configuration comes within 0.015 of it. Under the pre-registered rule the test is **not opened**, and no
test prediction, subset selection or step-count tuning is performed. Reported
valid numbers are per-seed single checkpoints; no averaging over seeds.

## 10. Costs, provenance, reproduction

GPU time is measured two ways; both are recorded (`analysis/budget.json`).

| item | GPU·h |
|---|---:|
| profiling (2 epochs per arm) | 0.013 |
| two E runs cancelled after the train-MAE metric bug (§11) | 0.155 |
| E_s0 + E_s1 (600 epochs each, incl. fork write) | 0.687 |
| R_s0, R_s1, S_s0, S_s1 | 3.400 |
| support fits (4 × 600 epochs, 59 updates/epoch) | 0.994 |
| κ calibration probe | 0.005 |
| R_s0_k04, S_s0_k04, R_s1_k04 (A1) | 2.413 |
| **total by job wall** | **7.667 of 8.000** |
| total by trainer-measured runtime | 7.458 |
| CPU: preparation (keys, 2 × 600-epoch partitions, support package) | 0.50 h wall, 0 GPU |
| wall clock, whole round | ≈ 5.1 h |

The 0.155 h of cancelled runs is included; two jobs shared one GPU for ~8 minutes,
so the wall sum is a slight over-estimate of exclusive device time. Headroom at the
end was 0.333 h, below the cost of any further arm (≥ 0.76 h), so no extra arm or
seed was started.

Provenance: host `res` (2 × A100-SXM4-40GB, driver 550.163.01, torch 2.5.1+cu124),
one process per GPU, remote checkout of the commit recorded per run in
`analysis/run_records.json`; the full `rr` job table (start, end, exit code, commit,
pool) is committed as `analysis/rr_jobs.json`. Every run's train/valid label hashes,
SSL-encoder hash, init-state hash and checkpoint hashes are in
`analysis/run_records.json`; the training module `code/zct_train.py` was byte-identical
across every commit used for training (`git diff` between the deployed revisions).

Reproduce (remote, exactly as executed; each line is one GPU job):

```bash
rr deploy res
rr run res zct-prep-1 --pool res-local --cpus 8 --time 1:00:00 -- \
  python tracks/structural-sufficiency/zinc_composition_transfer_v1/code/zct_prep.py --seeds 0,1 --epochs 600
rr run res zct-E-s0 --pool res-gpu0 --gpus 1 --time 0:50:00 -- \
  python tracks/structural-sufficiency/zinc_composition_transfer_v1/code/zct_train.py \
  --arm E --seed 0 --device cuda:0 --fork-out-epoch 50 --eval-train-from 501
rr run res zct-kappa-1 --pool res-gpu1 --gpus 1 --time 0:20:00 -- \
  python tracks/structural-sufficiency/zinc_composition_transfer_v1/code/zct_probe.py \
  --fork tracks/structural-sufficiency/zinc_composition_transfer_v1/runs/E_s0/fork.pt --seed 0 --device cuda:0
rr run res zct-R-s0 --pool res-gpu0 --gpus 1 --time 1:20:00 -- \
  python tracks/structural-sufficiency/zinc_composition_transfer_v1/code/zct_train.py \
  --arm R --seed 0 --device cuda:0 --kappa 0.1 \
  --fork-in tracks/structural-sufficiency/zinc_composition_transfer_v1/runs/E_s0/fork.pt \
  --eval-train-from 501 --diag-epochs 100,150,300,600
# ... S_s0, E_s1, R_s1, S_s1 identically (seed 1 uses runs/E_s1/fork.pt)
rr run res zct-supplus-s0 --pool res-gpu0 --gpus 1 --time 0:30:00 -- \
  python tracks/structural-sufficiency/zinc_composition_transfer_v1/code/zct_support.py train \
  --fit plus --seed 0 --device cuda:0
# ... minus_s0, plus_s1, minus_s1, then the A1 escalation:
rr run res zct-R-s0-k04 --pool res-gpu0 --gpus 1 --time 1:20:00 -- \
  python tracks/structural-sufficiency/zinc_composition_transfer_v1/code/zct_train.py \
  --arm R --seed 0 --device cuda:0 --kappa 0.4 \
  --fork-in tracks/structural-sufficiency/zinc_composition_transfer_v1/runs/E_s0/fork.pt \
  --eval-train-from 501 --diag-epochs 100,150,300,600 \
  --out tracks/structural-sufficiency/zinc_composition_transfer_v1/runs/R_s0_k04
# ... S_s0_k04 (same fork), R_s1_k04 (runs/E_s1/fork.pt), then:
rr run res zct-sup-freeze --pool res-local --cpus 4 --time 0:30:00 -- \
  python tracks/structural-sufficiency/zinc_composition_transfer_v1/code/zct_support.py freeze \
  --runs .../runs/sup_plus_s0,.../runs/sup_minus_s0,.../runs/sup_plus_s1,.../runs/sup_minus_s1
rr run res zct-analyze-sup2 --pool res-local --cpus 2 --time 0:20:00 -- \
  python tracks/structural-sufficiency/zinc_composition_transfer_v1/code/zct_analyze.py support
```

## 11. Amendment A2 — what went wrong and what was fixed

* **Train-MAE diagnostic bug** (found at epoch ~130 of the first `E` runs, before
  any result existed): the running train MAE was computed as `mean(|pred| - y)`
  instead of `mean(|pred - y|)`. Both `E` runs were cancelled (0.155 GPU·h
  recorded) and restarted at the fixed revision; no reported number uses the buggy
  metric.
* **Provenance note:** a deploy landed while three runs were in flight, so their
  recorded `git_commit` is the revision at *write* time. `code/zct_train.py` is
  identical across the relevant commits (`git diff` clean), which is why the runs
  are still comparable; the `rr` job table records the commit each job *started*
  at.
* **The mid-window κ confirmation ran in seed 1 as well** (A1 allowed it once
  ≥ 1.0 GPU·h remained; 1.10 h remained at that point). It is reported as
  secondary, never replacing the primary table.
* **κ = 0.1 is at the floor of the declared displacement window.** This is
  reported rather than repaired: changing the selection rule after seeing the
  calibration would have been tuning.

## 12. Files

* `PROTOCOL.md` — pre-registration, §A1 (κ escalation), §A2 (deviations).
* `configs/zct_v1.json` — frozen configuration and budget.
* `code/zct_keys.py` — radius-1/2 key interning (exact reproduction of the old
  groups, max |Δ| 4.8e-10).
* `code/zct_split.py` — structural partition constructor, strength calibration.
* `code/zct_prep.py` — deterministic preparation of every frozen artifact.
* `code/zct_train.py` — trainer for `E`/`R`/`S` (exact look-ahead, shared prefix
  fork, probes, per-epoch records).
* `code/zct_probe.py` — κ calibration on the epoch-50 fork.
* `code/zct_support.py` — §7 support package, fits, checkpoint freezing.
* `code/zct_checks.py` — the six pre-training checks.
* `code/zct_profile.py` — measured per-epoch cost before freezing the schedule.
* `code/zct_analyze.py`, `code/zct_export.py`, `code/zct_records.py` — analysis,
  per-molecule export, record assembly.
* `tests/test_zct.py` — 7 fast tests (ramp, schedule, partitions, masks, SMD,
  support batching, α = 0 identity).
* `analysis/*` — the records listed at the top of this file, plus
  `support_predictions.json` (the eight frozen support checkpoint hashes and their
  `M`/`Q` MAEs), `group_frozen.json` (the 710/225/39/26 split and its source hash),
  `budget.json` (per-job GPU hours), `profiling.json` (measured per-epoch cost) and
  `rr_jobs.json` (the trimmed `rr` job table).

## 13. Limitations

* Two supervised seeds on **one** shared SSL encoder: seed replication covers data
  order and supervised initialisation, not SSL variance.
* The structural split is one construction (radius-1/2 exact content keys,
  2–16 batch support, greedy swaps). A different construction could behave
  differently; the strength gate says this one moved `d_sym` by 0.078, not that it
  is the strongest possible.
* κ is a single frozen value per arm family; the A1 escalation covers one
  alternative (0.4) and the seed-1 confirmation only for `R`.
* The supervision-support result is a within-key statement: donors and queries
  share radius-2 keys by construction, so it measures whether labelled examples of
  the *same* local environment are usable, not whether the model can extrapolate
  to unseen environments.
* `D_plus`/`D_minus` are matched on size, rings, composition and radius-1 key
  fractions (max |SMD| 0.207); the matching reduces but does not remove structural
  differences.
* The valid split contains one molecule (row 172) that carries 7–10% of the MAE
  and can flip sign-level conclusions; all tables are reported with and without it.
* No test number exists for this round. Nothing here says whether the 0.05x target
  is reachable; it says these two interventions do not reach it.

## 14. 中文小结与回答（A–F）

**一句话结论**：结构驱动的跨组更新约束（look-ahead）只带来约 0.001–0.002 的
两个 seed 同向小改善（seed0 在 κ=0.4 出现过 0.0038，但 seed1 未复现），而且
**随机分组和结构分组效果一样**，所以这是"通用训练约束"效应，不是"结构机制"；但"局部环境标签支撑"是一个大杠杆：把 451 个携带目标环境的
训练分子放回去，目标查询分子的误差从 0.116/0.124 降到 0.082，而匹配背景分子
不动。官方 test 未打开，0.05x 目标未达成。

**A. 是否达到 0.05x（test MAE ∈ [0.050, 0.060)）？**
没有。最好配置（`R`@κ=0.4）valid W20/T100 ≈ 0.0794–0.0796（两个 seed），
全部 arm 都在 0.076–0.083。按预注册规则（valid 无重复性进展时不读 test），**官方 test 完全
未载入**，本轮不产生任何 test 数字；给不出"未达成"之外的 test 结论。

**B. E→R、R→S 各自改变了什么？**
* `E`→`R`（随机半批 look-ahead）：W20 −0.00172（seed0）/ −0.00178（seed1），
  T100 −0.00172 / −0.00172，best-valid −0.00134 / −0.00149；两个 seed 同向；
  换成 κ=0.4 后 s0 变成 −0.0038，但 s1 只有 −0.0016（比 κ=0.1 还差），
  **放大不可复现**，所以按"无剂量-反应、未达 0.003 阈值"处理。逐分子看：中位数不变，
  改善集中在高 |y| 尾部，去掉最差 1% 分子后只剩 0.0003–0.0008。
* `R`→`S`（换成结构分区）：W20 +0.00088（s0）/ −0.00009（s1），符号不一致，
  量级在噪声内；κ=0.4 时 `S` 仍比 `R` 差 0.0024（s0）。也就是说：**结构分区
  没有带来任何额外收益**，反而把源梯度扰动得更大（cos 0.93–0.96 vs 0.99）。
  另外 `S` 在 train 上拟合更好（eval-train@best 0.0116 vs `R` 0.0174），
  但 valid 更差 —— 局部拟合更好≠跨组迁移更好。

**C. 标签支撑（support）是否有帮助？**
有，而且很大、两 seed 一致。`Q_target`（350 个携带 90 个稀有半径-2 key 的分子）：
plus 0.0823 vs minus 0.1243（s0）、0.0821 vs 0.1159（s1）；DiD 相对 `Q_background`
为 −0.0430 / −0.0326；90 个 key 中 60–62 个 key 的误差下降（key 级 bootstrap CI
不含 0）；57–59% 的分子变好；支持度 4–7 的稀有 key 收益最大（−0.093 / −0.083）。
同时 plus 在 target-free 监控集 `M` 上略差（+0.0017/+0.0027），说明不是选择偏差。

**D. 训练约束在"真正缺支撑"的查询上是否起作用？**
两件事要分开：
* 训练集内部（`R`/`S` 的 query 半批）：有微小但两 seed 同向的作用（W20 −0.0017），
  但它是**同分布内**的半批，不是"缺支撑"查询；probe 显示虚拟步只把 ERM 方向改变 3–15%，且在 lr 衰减后
  到 epoch 600 基本归零（位移 ~1e-4），所以约束在最后 100 epoch 几乎是关的。
* 真正"缺支撑"的查询（§7 的 `Q_target`，其 key 的训练支撑被抽走）：**训练约束
  没有直接测这个**。可以说的间接事实是：同一个模型在"有标签支撑"时误差 0.082，
  在"支撑被抽走"时 0.116–0.124 —— 差距说明瓶颈是**监督位置**，而不是更新方向
  的约束形式。若要让约束影响缺支撑查询，需要把 look-ahead 的 query 集直接构造
  成缺支撑分子（本轮的 `S` 只是按结构分区，不是按支撑缺失分区）。

**E. 下一步（只写动作，不写承诺）**
1. 想做 MLDG 类机制的干净测试：α 固定不随 lr 衰减（或按 epoch 归一），内层损失
   换 smooth（MSE/Huber），κ 取窗口中部（0.3–0.5），2–3 seed，用 W20/T100 判定，
   并把 row 172 这类单分子主导的样本单独列出。
2. 结构分区若要再试，应改成"按支撑缺失分区"（`d_missing` 而不是 `d_unseen`），
   让 query 半批真正缺标签支撑 —— 这是 §7 已经证明有效的那一维。
3. 若继续追 0.05x：`unseen+ordinary`（39% 的 MAE）在所有 arm 都没动，需要的是
   覆盖/外推机制（新结构通道或外部数据），而不是又一个更新约束；预算上单轮
   8 GPU·h 内不可能同时做机制 + 覆盖，需要拆成两轮。
4. 数据集口径：本轮 valid 上单分子（row 172）可占 7–10% MAE，后续轮应在协议里
   预先声明"排除/不排除"该分子的报告口径，避免 1e-3 级别结论翻转。

**F. 实际 GPU·h 与墙钟**
按 job 墙钟（含失败/取消）**7.667 GPU·h**（上限 8.0），按 trainer 自测 **7.458
GPU·h**；其中预实验 0.013、被取消的两次 E 0.155、核心 E 0.687、R/S 四臂 3.400、
support 0.994、κ 校准 0.005、κ=0.4 三臂 2.413。CPU 准备 0.50 h（0 GPU·h）。
整轮墙钟约 5.1 h（00:25 → 05:30，含等待、两次重启与一次 smoke 修复）。
