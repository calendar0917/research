# LCRC support-indexed joint transfer v1 — report

Round `lcrc-support-joint-transfer-v1`. Protocol frozen before the first formal
step ([`PROTOCOL.md`](PROTOCOL.md), freeze hash
`fdd2c15f7f47264bbc7173a14a4bf78ff50a3669ced65e9c7ede59420265f9e9`).
All formal runs are at commit `1fe47fd3030a`; analysis-only code was added
afterwards. The prior round's random-public122 Lipo baseline is **not** used as
an official baseline here.

**One-line result.** The support-indexed S2 joint table does **not** transfer as
a useful general improvement. On official Lipo, `C1_JOINT` improves Validation
RMSE by 0.9% but is flat on Test; `C1_ADD` has a 3.0% Test point estimate but is
flat on Validation and its interval crosses zero. On PCBA, neither JOINT arm
improves the primary Validation AP: `FIXED1_JOINT` is clearly worse, and
`W2_JOINT` is negative with an interval that includes zero. The `FIXED1_JOINT`
Test point gain (+0.0027 AP) is not separated from zero and comes with that
Validation loss. The learned table has a large interaction component and the
model relies on it at inference, so the failure is not “interaction unused”.

## Main table

### Lipo (official scaffold split, raw RMSE; lower is better)

| arm | params | best epoch | Valid RMSE | Test RMSE | Valid final | Test final | Valid last-20 | Test last-20 | Δ Valid vs C1 | Δ Test vs C1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C1 (reused A) | 228,674 | 275 | 0.7086243 | 0.7253920 | 0.7157592 | 0.7417875 | 0.7137023 | 0.7401008 | — | — |
| **C1_JOINT** | 215,714 | 275 | **0.7022719** | 0.7257032 | 0.7117707 | 0.7333631 | 0.7106376 | 0.7352235 | −0.006352 (−0.90%) | +0.000311 (+0.04%) |
| **C1_ADD** | 213,794 | 270 | 0.7091835 | **0.7036454** | 0.7239482 | 0.7126554 | 0.7229032 | **0.7128348** | +0.000559 (+0.08%) | −0.021747 (−3.00%) |
| GINE reference | — | — | 0.6091 | 0.6456 | — | — | — | — | — | — |

GINE is a descriptive external reference, not retrained.

### PCBA (official scaffold split, macro AP; higher is better)

| arm | params | best epoch | Valid AP | Test AP | Valid final@25 | Test final@25 | Valid last-20 | Test last-20 | Δ Valid vs original | Δ Test vs original |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| W2 original | 994,324 | 24 | 0.2652758 | 0.2536156 | 0.2630573 | — | 0.2509938 | — | — | — |
| **W2_JOINT** | 973,140 | 25 | 0.2602860 | 0.2524571 | 0.2602838 | 0.2524559 | 0.2487926 | 0.2442970 | −0.004990 | −0.001158 |
| FIXED1 original | 1,128,468 | 25 | 0.2758464 | 0.2570020 | 0.2758455 | — | 0.2647827 | — | — | — |
| **FIXED1_JOINT** | 1,107,284 | 19 | 0.2685940 | 0.2597204 | 0.2679734 | 0.2566969 | 0.2606963 | 0.2522311 | −0.007252 | +0.002718 |
| DYNAMIC reference | 1,128,468 | — | 0.275067 | 0.262226 | — | — | — | — | — | — |

The originals are the recorded checkpoints replayed through the round evaluator
on the same GPU; their recorded and replay values agree (W2 Valid 0.2652733 vs
0.2652758; FIXED1 Valid 0.2758455 vs 0.2758464). `final@25` is the epoch-25
endpoint of the 25-epoch comparison prefix; for the originals it is read from
the historical curve. PCBA last-20 for originals is epochs 6–25 of the
historical 40-epoch run, so it is a same-horizon descriptive statistic; Test
was not evaluated per epoch in the historical W2 run (`—`).

## Answers to the five route questions

### Q1. Does the random122 positive result transfer to official Lipo?

**No, not robustly.** The two parameterisations fail in different places:

* `C1_JOINT − C1`: Valid −0.006352 RMSE (−0.90%, bootstrap 95% CI
  [−0.022641, +0.009425]), Test +0.000311 (+0.04%, CI [−0.022067, +0.022331]).
  The point estimate is a small Valid gain, not a Test gain.
* `C1_ADD − C1`: Valid +0.000559 (+0.08%, CI [−0.035125, +0.038245]), Test
  −0.021747 (−3.00%, CI [−0.057428, +0.013330]). The Test point estimate reaches
  the exploratory 3% threshold, but the interval crosses zero and the primary
  Validation metric does not improve.
* `C1_JOINT − C1_ADD` on Test: +0.022058 in favour of ADD (CI
  [−0.012133, +0.059052]), again not separated from zero.

The 3% ADDITIVE Test point estimate is not driven by the tail or by low-support
molecules: on the 418 Test molecules whose pairs are all Train-seen, ADD
improves RMSE from 0.7221 to 0.6986; the two molecules with unseen pairs get
worse (1.2355 → 1.4077). The gain is broad but modest in MAE (0.5292 → 0.5279)
while the largest errors are slightly worse (max 3.8817 vs 3.6445; top-10
overlap 6/10). It is an exploratory signal, not a confirmed effect.

### Q2. Does JOINT improve PCBA W2?

**No.** `W2_JOINT − W2`: Valid −0.004990 (95% CI [−0.010333, +0.001117]),
Test −0.001158 (CI [−0.008321, +0.006641]). Validation regresses; the Test
point estimate is slightly negative. `W2_JOINT` never beats 0.2653 on Valid in
the 25-epoch prefix and ends at the horizon endpoint (epoch 25).

### Q3. Does FIXED1_JOINT beat FIXED1?

**Not on the primary metric, and the Test gain is not reliable.**
`FIXED1_JOINT − FIXED1`: Valid −0.007252 (95% CI [−0.013611, −0.001194], a
clear loss), Test +0.002718 (CI [−0.001723, +0.008244]). The Test subset with
≥ 20 positives gives +0.006026 AP over 108 tasks (62 improved / 46 degraded),
and the top three absolute task deltas account for only 16.9% of the total
absolute task delta, so the Test signal is broad rather than a few-task
artifact. But it coexists with a systematic Validation loss and cannot be
called a gain under the frozen best-Valid selection rule.

### Q4. Do the two PCBA changes combine into a useful single-model gain?

**No.** The interaction is
`Δ_FIXED1 − Δ_W2`:

* Valid: −0.002263 (95% CI [−0.011934, +0.005531]).
* Test: +0.004101 (95% CI [−0.004259, +0.013312]).

The Test interaction is positive in point estimate (FIXED1 is less negative than
W2), but neither arm improves Validation, and the interval crosses zero. There is
no single-model gain to keep.

### Q5. Does this close the gap to stronger models?

**No.** Test AP remains 0.0203 below 0.28 for `FIXED1_JOINT` (0.25972), 0.0275
below for `W2_JOINT` (0.25246), and 0.0303 below 0.29 for `FIXED1_JOINT`.
On Valid, `FIXED1_JOINT` is 0.0114 below 0.28 and `W2_JOINT` 0.0197 below.
Both stay below the descriptive DYNAMIC reference (0.275067 Valid / 0.262226
Test). Lipo remains far from GINE (best new Test 0.7036 vs 0.6456).

## Verification

All gates are in [`analysis/checks_lipo.json`](analysis/checks_lipo.json),
[`analysis/checks_pcba.json`](analysis/checks_pcba.json),
[`analysis/post_checks_lipo.json`](analysis/post_checks_lipo.json) and
[`analysis/post_checks_pcba.json`](analysis/post_checks_pcba.json). Both post
check files have `passed_all: true`.

| gate | Lipo | PCBA |
|---|---|---|
| official split identity | official CSV == payload == cache order, 3360/420/420 | official CSV == frozen JSON, 350343/43793/43793; raw index hashes match |
| support built from Train only | rebuilt Omega/P equal; 108 pairs, rank 48, 4 components, 441 rectangles | rebuilt Omega/P equal; 214 pairs, rank 67, 4 components, 2,585 rectangles |
| only S2 changed | common state bitwise equal (31 keys); 215,714 / 213,794 params | W2 53 keys, FIXED1 67 keys bitwise equal; 973,140 / 1,107,284 params |
| initial forward on Train-seen types | max abs 1.43e−6 (JOINT vs A) | max abs 3.81e−6 (both) |
| ADDITIVE residual | max abs 3.33e−15 float64 (3.3e−15 after training); float32 model path 1.4e−7 | n/a |
| FIXED1 two-stage | n/a | stage1 1.19e−6, stage2 1.19e−6, m2 2.86e−6; class methods and `layer2_reads_original` unchanged |
| gradients | all 108 table rows and all 48 Theta rows receive gradient; 28/28 tensors finite/non-zero | 105/214 table rows exercised in the 4-batch probe; after one optimiser step all tensors non-zero except heads of tasks with no positives in the probe; every Omega row has ≥1 Train occurrence |
| invariance | atom renumbering ≤2.4e−7, path order 9.5e−7, batch regroup 9.5e−7 | atom renumbering ≤9.54e−6, path order 5.72e−6, batch regroup 1.91e−6 |
| checkpoint replay | A Valid/Test abs diff 1.6e−8 / 8.6e−10; new arms ≤3.1e−7 | historical Valid 5.9e−6 / 4.8e−6; new Valid ≤7.3e−6, Test ≤1.47e−5 |
| plan / batch order | SHA256 `1a3a440f…b1618` exactly | global plan `06f43729…b6f6`; prefix-25 `8c857251…9b40` exactly |

Support coverage (occurrence fraction): Lipo Train/Valid/Test
1.00000/0.99983/0.99950; PCBA 0.99994/0.99998/0.99999. Unseen-pair fallback was
exercised on 13 real Lipo Valid/Test codes: finite values, no new parameters.

## Interpretability

The exact decomposition is `J = J_add + J_int`, `J_add = P J`,
`J_int = (I − P) J`, with closure zero to float64 precision in every model.

| model | table RMS | additive RMS | interaction RMS | interaction fraction | first-layer weight on joint columns |
|---|---:|---:|---:|---:|---:|
| C1_JOINT | 0.571954 | 0.376976 | 0.430140 | **0.752** | 0.274 |
| C1_ADD | 0.378786 | 0.378786 | ~0 | 0.000 | 0.248 |
| W2_JOINT | 0.627081 | 0.348567 | 0.521280 | **0.831** | 0.347 |
| FIXED1_JOINT | 0.619367 | 0.346177 | 0.513592 | **0.829** | 0.342 |

The interaction is large in every JOINT table. On Lipo it is concentrated in
three of the four bipartite components (interaction fractions 0.768, 0.662,
~0, 0.793). The LocalMLP first layer puts 25–35% of its input weight norm on the
joint columns, so the channel is not ignored.

**One frozen ADD_ONLY intervention** on the single best new model (C1_JOINT,
the only new model improving its primary Validation metric) replaces the learned
table by its marginal projection at inference:

| split | before RMSE | after RMSE | Δ |
|---|---:|---:|---:|
| Valid | 0.702272 | 0.832794 | +0.130522 |
| Test | 0.725703 | 0.864643 | +0.138939 |
| Train | 0.272403 | 0.603007 | +0.330604 |

Removing the interaction destroys performance. The table learned a large
pair-specific component that the trained model relies on, even though inserting
the table in place of `joint_proj` did not improve generalization. This is the
main mechanistic finding of the round.

## Budget and execution

* Timing preflight measured real training steps plus full three-split
  evaluation: projected 2.744 GPU·h for all four arms
  ([`analysis/throughput_probe.json`](analysis/throughput_probe.json)).
* Actual formal wall time: C1_JOINT 0.105 h, C1_ADD 0.109 h, FIXED1_JOINT
  1.237 h, W2_JOINT 1.404 h; **2.856 GPU·h** total. With preflight/replay jobs
  the round is ≈2.95 GPU·h, under the 3.0 h target and well under the 4.0 h hard
  cap. Formal wall-clock span 20:29–22:30 (2.02 h), under the 3.5 h cap.
* `FIXED1_JOINT` ran on GPU1 and `W2_JOINT` concurrently on GPU0, both A100s on
  `res`; GPU0 was free when W2 was launched. No arm was shortened. All 360 Lipo
  epochs and all 25 PCBA prefix epochs completed.
* `W2_JOINT` selected epoch 25, the horizon endpoint; `FIXED1_JOINT` selected
  epoch 19. Selection used only Valid, never Test.

## Deviations / notes after freeze

1. **GPU assignment.** The preferred pool was GPU1; `W2_JOINT` used GPU0 while
   GPU1 was occupied by `FIXED1_JOINT`. Same host/regime (A100-40GB,
   driver 550.163.01, torch 2.5.1+cu124); this affects timing only.
2. **PCBA bootstrap wording.** The frozen config specifies 2,000 task-level
   resamples and 200 molecule-level resamples. The protocol text was corrected
   to that split before analysis; the reported macro-AP intervals use 200
   molecule-level resamples, and per-task intervals use 2,000 task resamples.
   Lipo uses 2,000 molecule-level resamples (cheap at n=420).
3. **Post-freeze checks code.** `jt_checks.py` gained two reporting
   refinements after the formal runs: a 5e−5 replay tolerance for newly trained
   AP checkpoints (observed 1.47e−5) and an allowance for zero task-head
   gradients when a probe batch contains no positive label for that task. These
   are check-reporting changes; `jt_lipo_train.py`, `jt_pcba_train.py`,
   `jt_models.py`, `jt_support.py` and the frozen config are unchanged from the
   formal-run commit. `jt_common.py` was restored to its frozen content.
4. **W2 Test reference.** The historical W2 run never evaluated Test
   (`test_evaluated: false`). The W2 “original” Test value is the same-pipeline
   replay on the selected epoch-24 checkpoint: 0.2536156 vs the round brief's
   0.2536165 (abs 9e−7). FIXED1 replay is 0.2570020 vs recorded 0.2570014.
5. **No incomplete arms.** No failures, no restart, no shortened comparison.

## Conclusion and limits

The transfer hypothesis is **not supported**. The only positive point
estimates are C1_JOINT's 0.9% Valid RMSE gain, C1_ADD's 3.0% Test RMSE gain
(with flat Valid and an interval crossing zero), and FIXED1_JOINT's +0.0027 Test
AP (with a clear Valid loss and an interval crossing zero). Against the
exploratory reference sizes, only C1_ADD's Test point estimate reaches the 3%
Lipo bar, and no PCBA arm reaches +0.005 AP (FIXED1_JOINT's +0.0027 is about
half the reference and not separated from zero). None meets a robust “useful
single-model gain” bar. Per the frozen stop rule — both PCBA arms
fail on Validation — typed-S2 table / marginal-decomposition work stops here.

Limits:

* one seed per arm; the intervals are molecule/task resampling uncertainty, not
  training-seed variance;
* PCBA is a 25-epoch prefix; the originals' best epochs happen to lie inside the
  prefix (24 and 25), so best-checkpoint comparisons are fair, but later-epoch
  behaviour is not tested;
* the JOINT arms remove `joint_proj`, so total parameter counts are slightly
  lower than the originals (215,714 vs 228,674; 973,140 vs 994,324; 1,107,284
  vs 1,128,468); this is the defined replacement, not a capacity-matched
  control;
* PCBA Test is an exposed development comparison set; all Test statements are
  descriptive, and no Test value was used for selection;
* the GINE and DYNAMIC references are descriptive external numbers.

No v2 architecture, new structural input, gate, message-passing layer, readout,
auxiliary loss, extra seed, or hyperparameter search was added.

## Reproduce

```bash
# frozen artifacts and checks (pre-training)
uv run --no-sync python tracks/.../code/jt_prepare.py --tasks lipo,pcba
uv run --no-sync python tracks/.../code/jt_checks.py --scope all --device cpu

# formal runs (via rr; one job per arm)
rr run res jt-lipo-c1-joint --pool res-gpu1 -- python .../jt_lipo_train.py --arm C1_JOINT ...
rr run res jt-lipo-c1-add   --pool res-gpu1 -- python .../jt_lipo_train.py --arm C1_ADD ...
rr run res jt-pcba-fixed1-joint --pool res-gpu1 -- python .../jt_pcba_train.py --arm FIXED1_JOINT ...
rr run res jt-pcba-w2-joint     --pool res-gpu0 -- python .../jt_pcba_train.py --arm W2_JOINT ...

# analysis
python .../jt_replay_originals.py --device cuda
python .../jt_analyze.py --lipo --n-boot 2000
python .../jt_analyze.py --pcba --n-boot 2000 --mol-boot 200
python .../jt_tables.py
python .../jt_curves.py
```
