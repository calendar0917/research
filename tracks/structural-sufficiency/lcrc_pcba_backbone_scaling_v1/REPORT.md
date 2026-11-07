# LCRC PCBA Backbone Scaling v1 — report

Round `lcrc-pcba-backbone-scaling-v1`. Protocol frozen in [`PROTOCOL.md`](PROTOCOL.md)
before the first formal step (2 pre-registered amendments, both recorded in §11);
all numbers below are read back from the committed artifacts under `analysis/`
and `runs/`.

**One-line result.** Widening LCRC-A helps: both `W2` (994,324 params) and
`W3` (2,120,844 params) beat the historical A0 endpoint (0.246356) by more than
the pre-registered +0.005 threshold with broad per-task support, but the gain is
**not monotone** — `W2` 0.261867 > `W3` 0.259004 (`W3 − W2 = −0.00286`). Extra
capacity beyond 2× is absorbed by train fit (Train AP 0.5819 → 0.7055), not by
Validation AP.

| model | params | trunk | readout | Val AP@40 (official OGB) | Δ vs A0 | label |
|---|---:|---:|---:|---:|---:|---|
| A0 (historical, not retrained) | 287,644 | 245,184 | 42,460 | 0.246356 | — | — |
| **W2** = LCRC-A-Wide-2x | **994,324** | 848,096 | 146,228 | **0.261867** | **+0.015511** | `PROMISING_SCALE_SIGNAL` |
| **W3** = LCRC-A-Wide-3x | **2,120,844** | 1,809,408 | 311,436 | **0.259004** | **+0.012648** | `PROMISING_SCALE_SIGNAL` |

---

## Q1. Does the 1× replay reproduce A0 exactly?

Yes, to the precision that CUDA allows, on three independent paths.

| check | result |
|---|---|
| historical checkpoint SHA256 | `f110935a…93a437` — matches the frozen record |
| replay of the epoch-40 checkpoint through the new builder (`scale=1`) | AP 0.246358191 vs recorded 0.246356096, **abs Δ 2.1e-6** |
| replay masked BCE | 0.045772612791 vs recorded 0.045772612836, **Δ 4.5e-11** |
| per-task AP max abs Δ vs recorded | 2.47e-4 (one seed-level task; ranking-tie level) |
| `scale=1` vs historical `NMPModel` on identical weights, CPU | **bitwise identical, max Δ 0.0** |
| same, CUDA | max Δ 6.9e-5 (tolerance 1e-4), AP Δ 3.5e-6 |
| `scale=1` state dict keys vs historical A | equal |
| `scale=1` initial state vs historical A init | bitwise equal |
| trunk init fingerprint vs recorded | matches `1b1f1ad3…dcf7c3` |

The residual is `index_add_`/`scatter_reduce_` atomic accumulation on CUDA,
which is not bitwise stable between two module instantiations; the CPU check is
exact and the AP-level agreement is 2e-6. A0 itself was **never retrained** —
only a read-only checkpoint reload and forward replay.

## Q2. Are the parameter counts exactly the targets?

Yes, by `sum(p.numel())` over the constructed model, verified on the GPU host as
part of the pre-training gate and re-verified inside each formal run
(`model_audit.total_parameter_count`, `parameters_exact: true`).

| model | total | target | trunk | target | readout | target |
|---|---:|---:|---:|---:|---:|---:|
| W2 | 994,324 | 994,324 | 848,096 | 848,096 | 146,228 | 146,228 |
| W3 | 2,120,844 | 2,120,844 | 1,809,408 | 1,809,408 | 311,436 | 311,436 |

No padding, no extra parameters, no rounding target.

## Q3. What scales, module by module?

Only widths. Every width is explicit in `bs_common.REFERENCE_WIDTHS` and the
historical defaults (`width=64`, `scorer_hidden=32`, `pool_hidden=88`) never
survive into the scaled models (audited: `width_audit.all_match = true` for all
three scales, and a search for the historical constants returns no leftover).

| module | A0 (1×) | W2 (2×) | W3 (3×) |
|---|---:|---:|---:|
| `atom_hidden` | 128 | 256 | 384 |
| `atom_out` | 96 | 192 | 288 |
| `bond_hidden` | 160 | 320 | 480 |
| `bond_out` | 96 | 192 | 288 |
| `stat_rank` | 32 | 64 | 96 |
| `local_hidden` | 192 | 384 | 576 |
| per-atom `f` width | 64 | 128 | 192 |
| global `rho` hidden | 64 | 128 | 192 |
| readout scorer hidden | 32 | 64 | 96 |
| readout pool hidden | 88 | 176 | 264 |
| frozen: heads / n_out / feats / emb / gates | 4 / 128 / 51,13,9,42 / 2→8,400→1 | same | same |

Unchanged computation: one atom encode, one bond encode, one-shot S1/S2 local
statistics, one per-atom `f`, one global `rho`, one attention-pool readout. The
call-count audit confirms `atom_encoder, bond_encoder, local, f, rho, readout`
each execute **exactly once per forward**, and that everything after `local` is
a function of `(h, atom_batch, n_graphs, global)` only — supplying a batch with
no bond/path fields changes nothing (Δ 1.9e-6 = atomic noise).

## Q4. Validation cadence and measured evaluation cost

The frozen 15 % rule was resolved once, by measurement on an idle A100, and
never revisited (`PROTOCOL.md` §6.1):

| model | step p50 (probe) | projected epoch | full 43,793-row Validation pass | fraction | decision |
|---|---:|---:|---:|---:|---|
| W2 | 28.7 ms | 78.7 s | 5.99 s | 7.61 % | ≤ 15 % |
| W3 | 27.2 ms | 74.4 s | 6.07 s | 8.16 % | ≤ 15 % |

→ **every epoch, epochs 1…40, for both models** (`eval_every = 1`, 40/40
evaluations completed in each run; `eval_every_matches_protocol: true`).

Measured in the formal runs: validation cost 7.21 s/epoch (W2) and 7.36 s/epoch
(W3), i.e. **5.8 % / 5.9 % of the actual 124.9 s / 124.1 s epoch** — the rule
would have selected the same cadence had it been re-measured on the real epoch
time. Primary metric = the epoch-40 official OGB Validation macro AP; the best
diagnostic epoch (W2 e24 0.265273, W3 e25 0.263670) is recorded but **not** used
for any comparison.

## Q5. GPU parallelism and the GPU0 contention question

Both formal runs went to **one GPU at a time, `res-gpu1`, serially**:

| run | window (CST) | pool | `CUDA_VISIBLE_DEVICES` |
|---|---|---|---|
| `bs-w3` | 17:36:28 → 18:59:54 | `res-gpu1` | 1 |
| `bs-w2` | 19:05:28 → 20:29:19 | `res-gpu1` | 1 |

GPU0 was **not** shared, even though it had room in memory. Before deciding I
sampled it 6× at 10 s intervals (`analysis/gpu0_occupancy.json`,
`gpu0_occupancy_samples.csv`):

| GPU | mem used | mem free | utilisation (6 samples) | resident compute process |
|---|---:|---:|---|---|
| 0 | 16,105 MiB | 24,222 MiB | **100 % at every sample** | pid 3841932 `python3 train.py --conv_type nobatch`, up 2 d 19 h, 16,082 MiB |
| 1 | 17 MiB | 40,310 MiB | 0 % at every sample | none |

The pre-registered co-tenancy rule requires *both* free memory with margin
**and** no sustained full-load contention. 24.2 GiB would have fit a 331 MB
model, but the second clause fails on a GPU pinned at 100 % by an external
2-day-old job, so the rule — not a judgement call — put both runs on GPU1. Cost
of the serial choice: total wall clock ≈ 2 h 53 min instead of ≈ 1 h 24 min. It
did not threaten the budget (see Q13). No external process was touched,
signalled, or waited on beyond reading `nvidia-smi`.

## Q6. AP at epochs 10 / 20 / 30 / 40

| model | AP@10 | AP@20 | AP@30 | AP@40 (primary) |
|---|---:|---:|---:|---:|
| W2 | 0.243189 | 0.258635 | 0.262022 | **0.261867** |
| W3 | 0.244091 | 0.262726 | 0.262884 | **0.259004** |
| A0 | n/a | n/a | n/a | **0.246356** |

A0 has no epoch 10/20/30 evaluation: the historical record contains exactly one
Validation evaluation, at epoch 40. That is a property of the frozen artifact,
not something this round could recover without retraining A0 (explicitly
forbidden). Anything comparing A0 to W2/W3 is therefore an **epoch-40 vs
epoch-40** comparison.

At epoch 10 both are still just below A0's endpoint (W2 0.2432, W3 0.2441); they
cross it at **epoch 13 (W2)** and **epoch 11 (W3)** and stay above it for 28 and
30 of the 40 epochs respectively (every epoch from epoch 20 on). The remaining
30 epochs are spent adding ≈ 0.018 / 0.015.

## Q7. ΔAP40 against A0 (the pre-registered metric)

| comparison | ΔAP40 | improved tasks / 127 scored | improved among 107 high-support | mean Δ on 107 | label |
|---|---:|---:|---:|---:|---|
| W2 − A0 | **+0.015511** | 84 | 74 | **+0.017976** | `PROMISING_SCALE_SIGNAL` |
| W3 − A0 | **+0.012648** | 84 | 73 | **+0.011848** | `PROMISING_SCALE_SIGNAL` |
| W3 − W2 | −0.002863 | 56 better / 71 worse | — | — | not monotone |

Both scaled models satisfy all three pre-registered conditions
(ΔAP40 ≥ +0.005, mean Δ on ≥20-positive tasks ≥ +0.003, ≥ 60/107 improved), so
capacity itself is not the binding constraint at 1×. Gains are not concentrated
in a handful of assays: the 10 largest per-task deltas account for 41 % (W2) and
44 % (W3) of the summed positive per-task deltas, and the median per-task delta
is +0.0132 (W2). Task 45 is unscorable (no Validation positives), so all per-task
statements are over 127 tasks.

## Q8. W2 → W3: what did the third width buy?

Nothing on Validation AP: **−0.00286**, which is larger than the epoch-to-epoch
noise floor of the last 10 epochs (±0.0022 / ±0.0031 range) but far too small,
and one seed too few, to call a regression. Per task it is if anything negative: 84 of 127
scored tasks improve going 1×→2×, but going 2×→3× only 56 improve while 71 get
worse (`per_task_deltas.csv`, `delta_w3_minus_w2`).

What it did buy is **fit**: Train AP40 0.5819 (W2) → 0.7055 (W3), i.e. +0.124,
against −0.0029 Validation. The train–validation gap widens from 0.320 to 0.446.
At 3× the frozen from-scratch recipe (40 epochs, lr 3e-4 cosine, wd 1e-3,
dropout 0.10, masked BCE) is fitting the training split faster than it
generalises; a capacity-vs-generalisation statement at 3× would need
regularisation or length changes that this round's protocol forbids.

## Q9. Is the effect broad or concentrated?

Broad across the task set, with a different shape at each width:

| stratum | n | W2 mean Δ | W2 median Δ | W3 mean Δ | W3 median Δ |
|---|---:|---:|---:|---:|---:|
| all scored | 127 | +0.015511 | +0.013186 | +0.012648 | +0.010939 |
| support ≥ 20 positives | 107 | +0.017976 | +0.013689 | +0.011848 | +0.011041 |
| support 5–19 (sparse) | 18 | +0.006431 | −0.003512 | +0.025435 | +0.009358 |
| support < 5 | 2 | −0.034684 | −0.034684 | −0.059654 | −0.059654 |

84 of 127 scored tasks improve under both widths, and the Δ-vs-support rank
correlation is small (+0.168 for W2, +0.119 for W3), so the gain is not simply
"only the big assays moved". The one qualitative difference is the sparse stratum: under W2
its *median* task loses ground (−0.0035, 9/18 improved) while under W3 it gains
(+0.0094, 11/18). The third width shifts capability toward rarely-observed
assays, which the macro average over 127 tasks barely rewards.

## Q10. Train vs Validation scaling

| width | params | Train AP40 (350,343 rows) | Val AP40 (43,793 rows) | gap |
|---|---:|---:|---:|---:|
| A0 | 287,644 | 0.419657 | 0.246356 | 0.173 |
| W2 | 994,324 | 0.581875 | 0.261867 | 0.320 |
| W3 | 2,120,844 | 0.705498 | 0.259004 | 0.446 |

Train AP increases monotonically with width (+0.162, +0.124); Validation AP does
not (+0.0155, −0.0029). The scaling law here is a *memorisation* law — which is
consistent with the round's premise (one-shot, no message passing: capacity goes
into per-atom statistics rather than into new graph-level computation).

## Q11. Last-10-epoch trend

| model | slope / epoch (e31…e40) | e30 → e40 | range | std |
|---|---:|---:|---:|---:|
| W2 | −9.2e-5 | −0.00016 | 0.00222 | 0.00059 |
| W3 | −1.9e-4 | −0.00388 | 0.00310 | 0.00088 |

Neither curve is still rising: both drift down by less than one part in a
thousand over the final decade of epochs, W3 more than W2. The comparison at
epoch 40 is therefore not "still climbing, would have overtaken 2×"; if anything
the W3 deficit widens slightly at the end. Conversely neither model is
degrading in a way that would make a longer run obviously better under this
recipe.

## Q12. Beyond the pre-registered questions: cost

| model | epoch mean | step p50 | steps/s | train s | valid s | peak GPU mem | GPU-hours |
|---|---:|---:|---:|---:|---:|---:|---:|
| W2 | 124.86 s | 42.1 ms | 23.56 | 4,654.5 | 288.4 | 223 MB | 1.3949 |
| W3 | 124.12 s | 41.6 ms | 23.77 | 4,613.1 | 294.3 | 331 MB | 1.3880 |
| A0 (historical) | 119.14 s | — | 23.51 | — | — | — | 0.6629 |

Widening from 288 k → 2.1 M parameters (7.4×) changes throughput by **< 1 %**
(23.51 → 23.56 → 23.77 steps/s) at this batch size: the epoch is data-loader bound (23.6 batches/s of 128 graphs), not
GPU bound. Peak training memory is 331 MB at 3×, which is why co-tenancy on
GPU0 would have been technically possible on the memory axis alone.

## Q13. GPU-hours, budget, failures

| item | GPU-hours |
|---|---:|
| preflight + 3 audit/probe jobs + GPU0 sampling (5 jobs) | 0.0619 |
| `W2` formal 40 epochs | 1.3949 |
| `W3` formal 40 epochs | 1.3880 |
| **formal + overhead** | **2.8448** |
| abandoned attempt (W3 relaunch, see below) | 0.0603 |
| **total charged** | **2.9051** |

Target ≤ 8, hard cap ≤ 12 → **`WITHIN_TARGET`** (36 % of the target, 24 % of the
cap). `analysis/gpu_ledger.json` charges every job actually executed, including
the ones that produced nothing.

Non-productive launches: **2**, both infrastructure, neither touching a formal
epoch.

1. `bs-preflight-20261107-171236` — rejected by the harness before the process
   started (argv passed to `rr run` as a single string instead of `bash -c`):
   0 GPU-seconds.
2. `bs-w3-20261107-173238` — cancelled deliberately 4 min in (≈ 2 epochs, 0.060
   GPU-h) because the command was piped through `tail`, which would have
   swallowed the live epoch log and left `stdout.log` with only the last 60
   lines. Relaunched unchanged; `runs/W3` was rebuilt from scratch, no epoch was
   double-counted, and no partial checkpoint was reused.

Zero OOM, zero non-finite events, zero checkpoint restarts in the two formal
runs. Gradient clipping fired 14 times (W2) and 10 times (W3), all within the
first epochs — the recorded `grad_clip = 5.0` regime, not a protocol deviation.

## Q14. Conclusion

**Capacity is not the bottleneck at 1×, and it stops paying between 2× and 3×.**

- Both scaled models clear the pre-registered "promising" bar by ~3× the
  threshold, with 73–74 of 107 high-support assays improving. Going from 287 k to
  994 k parameters is worth **+0.0155 AP** under an otherwise frozen recipe.
- The third width adds nothing on Validation (−0.0029 vs 2×) while adding +0.124
  Train AP. Under this exact recipe — one-shot encoding, 40 epochs, from
  scratch, no augmentation or regularisation change — the useful capacity
  frontier is **at or below 994 k parameters**.
- Because the widths are the only change, the deficit is not attributable to a
  new module, depth, or inductive bias: it is a capacity-vs-generalisation
  trade-off at fixed optimisation budget.

Recommendation for the next round (not executed here): promote **W2** to the new
reference width, and test the 3× deficit against recipe headroom (longer
schedule or width-scaled weight decay/early-stop-free regularisation) before
concluding anything about LCRC's architectural ceiling. A single seed cannot
distinguish −0.003 from noise at the 0.002–0.003 level.

## Q15. Artifacts, provenance, checkpoints

Round commit history (branch `main`): `fc9ae2a0` protocol freeze → `c1b50376`
cadence resolved → `dc4b053c` tolerance pinned (W3's commit) → `27f5d66c` ledger
accounting (W2's commit) → analysis commit (this file). `bs_common.py`,
`bs_model.py`, `bs_train.py` are **byte-identical** between `dc4b053c` and
`27f5d66c` (`git diff` empty), so both formal runs executed the same training
code; the later commit changed only analysis-side code.

| artifact | path |
|---|---|
| protocol | `PROTOCOL.md` |
| run instructions | `README.md` |
| model/audit/compare code | `code/bs_{common,model,train,audit,compare}.py` |
| tests (25) | `tests/test_bs_{model,training}.py` |
| width, replay and gate audit | `analysis/model_scaling_audit.json` |
| A0 read-only replay | `analysis/a0_replay.json` |
| cadence cost probe | `analysis/evaluation_cost_probe.json` |
| GPU occupancy at launch | `analysis/gpu_preflight.json`, `analysis/gpu0_occupancy.json`, `gpu0_occupancy_samples.csv` |
| per-model epoch-40 records | `analysis/w2_epoch40.json`, `analysis/w3_epoch40.json` |
| comparison + labels | `analysis/scale_comparison.json`, `analysis/decision.json` |
| per-task table | `analysis/per_task_deltas.csv` (128 rows, 127 scorable: 4 AP columns + 3 deltas each) |
| resource ledger | `analysis/gpu_ledger.json` |
| raw job records | `analysis/rr_jobs.json` |
| run directories | `runs/W2/`, `runs/W3/` (git-ignored, pulled) |

Checkpoints (SHA256, from the runs' own `run.json`):

| model | epoch-40 (primary) | best diagnostic (unused) | resumable |
|---|---|---|---|
| W2 | `7fbdbd16f7b2077e77cdb85488bb2506f2e4ee98333d40fbcf7558074e26598a` | `d86fe5e56e898d382acf66eb0533beb915f8ca39144c6c1c74e25e147667d2bd` (e24) | `bfed3ad3…` |
| W3 | `4eb6a42f562a4fa8099d0a40053ce45f3bfd08645ce62999dee961e764cb18cf` | `b402c76c555cf98028961244e42c5df8c089826a623b3b68144b04a5e13ba9e9` (e25) | `ebdd8b5b…` |
| A0 | `f110935a6055f3c087f9d0b40580506a512cdeb09f6af67a56bde787cc93a437` | — | — |

Also recorded per run: batch-plan SHA256 `06f43729…b6f6` (identical to A0's
frozen seed-0 order), RNG hashes at start and end of every epoch, the
per-epoch `train_curve.csv`, and `valid_logits_sha256` / `valid_probs_sha256`
for the epoch-40 Validation pass, so any later claim is checkable against the
exact logits.

## What this round does not establish

- **No Test metric.** Test labels are NaN in the frozen artifact; `test_evaluated`
  is `false` in both runs and the audit asserts no Test access.
- **One seed per model.** Epoch-level AP noise here is 0.0006–0.0031 (std over
  the last 10 epochs), so |ΔAP| below ~0.003 is not interpretable as an effect,
  and the −0.0029 W3−W2 gap sits exactly at that boundary.
- **No paired initialisation claim.** W2 and W3 are different shapes by
  construction, so they are compared as two independent runs, not as a
  width-matched initialisation pair.
- **A capacity null at 3× would not prove a LCRC ceiling.** It only shows that
  this frozen recipe does not convert 3× capacity into Validation AP.
