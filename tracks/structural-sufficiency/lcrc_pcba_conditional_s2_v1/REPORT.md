# LCRC PCBA Conditional S2 v1 — report

Round `lcrc-pcba-conditional-s2-v1`. Protocol frozen in
[`PROTOCOL.md`](PROTOCOL.md) before the first formal step (two append-only
amendments: the initial freeze and the GPU0 co-tenancy decision). All numbers
below are read back from the committed artifacts in `analysis/` and the
git-ignored `runs/`.

**One-line result.** A rank-16 vector gate conditioned on
`(path_cat, path_end63)` **does learn** — at epoch 40 its per-path vector deviation
has mean L2 norm 5.90 (5.96× the mean absolute scalar gate; elementwise RMS
0.492) and is non-zero on 100 % of paths — but it produces **no Validation AP gain**: C2 = 0.260903 vs
W2 = 0.261867 (`D_C2 = −0.000964`), and it also lands below the
near-parameter-matched FFN (`D_CF = −0.000597`). Label:
**`NO_CLEAR_C2_GAIN`**.

| arm | parameters | new params | Val AP@40 | Δ vs W2 | Δ vs F |
|---|---:|---:|---:|---:|---:|
| W2 (historical, not retrained) | 994,324 | — | **0.2618669181** | — | — |
| **C2** = W2 + Conditional S2 | 1,005,076 | 10,752 | **0.2609026157** | **−0.0009643** | −0.0005970 |
| **F** = W2 + near-matched FFN | 1,004,988 | 10,664 | **0.2614996012** | **−0.0003673** | — |

Formal training commit: `c4bb21bb54387cba1fb6d96513b96b6043973f64` (clean
remote checkout). Training was executed at that commit; this report is the
analysis commit on `main`.

---

## Q1. How many parameters did C2 and the FFN actually add?

Exactly the pre-registered counts, by `sum(p.numel())` over the constructed
models and re-verified inside both formal runs:

| arm | module | arithmetic | parameters |
|---|---|---|---:|
| C2 | `E_c: 400×16` | 6,400 | 6,400 |
| C2 | `E_e: 80×16` | 1,280 | 1,280 |
| C2 | `V: 16×192` | 3,072 | 3,072 |
| C2 | **total new** | 6,400 + 1,280 + 3,072 | **10,752** |
| C2 | **total model** | 994,324 + 10,752 | **1,005,076** |
| F | `LayerNorm(128)` | 128 + 128 | 256 |
| F | `Linear(128,40)` | 128×40 + 40 | 5,160 |
| F | `Linear(40,128)` | 40×128 + 128 | 5,248 |
| F | **total new** | 256 + 5,160 + 5,248 | **10,664** |
| F | **total model** | 994,324 + 10,664 | **1,004,988** |

Difference: `10,752 − 10,664 = 88` parameters, **0.82 %** of the C2 increment.
This is a near-matched control, not an exactly matched one.

## Q2. Was the original W2 initialisation and data order strictly reused?

Yes, on five independent checks:

1. **Bitwise subset.** Every one of the 57 historical W2 seed-0 state tensors
   in C2 and F is bitwise equal to the fresh `bs_model.build_model("W2",
   seed=0)` state, and C2's subset equals F's subset. `n_differing = 0`.
2. **Fingerprints.** The W2 trunk init fingerprint is
   `5f7101a8f378e7c9283caaade8d61eee733c0c9f3cfb65b5584c8a1e495ef63d` and the
   FULL-trunk fingerprint is
   `109756bd24b2ab8589012031e3cf0b0389b0d275227b35044e9381a45c00a5db`,
   matching the historical W2 record. The base-state SHA-256 is identical for
   W2/C2/F (`analysis/initialization_audit.json`).
3. **New-module privacy.** `E_c`/`E_e` use private generator
   `seed*1_000_003+29 = 29`; F's `fc1` uses `31`; `V = 0`, `fc2 = 0`. The new
   module construction does not advance the global torch RNG (checked before
   and after; the historical W2 builder's own RNG advance is then reset by
   `seed_everything(0)`, exactly as in the W2 run).
4. **No warm-start.** Neither arm loaded a W2 epoch-40 checkpoint or any
   trained weight; only the re-built untrained seed-0 init state was used.
5. **Data order.** Both runs used the frozen plan
   `06f437299fa451e449202a990d28f3b8d101e7aec709bcd72a878c8de864b6f6`
   and the consumed-40-epoch prefix
   `1a24dfca54898dd2b0e7c58d2b2f1956df67b233fdceaa7a376f81a2377a5f58`,
   with zero Validation or Test rows in the plan. At the end of epoch 1 both
   runs have the same torch CPU RNG hash, confirming the same dropout stream.

`analysis/training_parity.json` records **all checks passed**, including
epoch-0 AP parity: W2 replay 0.02906286, C2 0.02906316, F 0.02906295
(differences ≤ 3e-7, far below the 1e-4 CUDA tolerance).

## Q3. Did C2 really change the S2 content's path-conditioned computation?

Yes. C2's S2 content aggregation is

```python
delta = (E_c(path_cat) * E_e(path_end63)) @ V          # (n_paths, 192)
vector_gate = path_gate(path_cat) + delta              # scalar + rank-16 vector
content_agg = zeros.index_add_(0, path_atom, a[path_end] * vector_gate)
```

Inside one `TULocalContent.forward` call: the path type index is in
`[0, 393] ⊂ [0, 400)`, the endpoint code is in `[0, 78] ⊂ [0, 80)`, and the
gate is exactly `[n_paths, 192]`. The path enumeration, endpoint atom ids,
path counts and every other S1/S2 statistic are unchanged (batch tensors are
bitwise unchanged after a forward). There is exactly one atom encode, one bond
encode, one `local`, one dropout and one readout call per forward; `post_local`
runs with no bond/path field at all. The gate is a one-shot local computation
inside `local`, not an iterative propagation.

## Q4. What are C2's and F's AP at epochs 20 / 25 / 30 / 40?

| epoch | W2 | C2 | F |
|---:|---:|---:|---:|
| 1 | — | 0.095866 | 0.095733 |
| 10 | 0.243189 | 0.242663 | 0.246655 |
| 20 | 0.258635 | 0.255431 | 0.256426 |
| 25 | 0.263057 | 0.260909 | 0.260903 |
| 30 | 0.262022 | 0.260591 | 0.261183 |
| 35 | — | 0.262161 (best) | 0.261648 |
| **40 (primary)** | **0.261867** | **0.260903** | **0.261500** |

C2 is below W2 at every one of e10/e20/e25/e30/e40: −0.000527 at e10, −0.003204 at e20,
−0.002149 at e25, −0.001431 at e30, −0.000964 at e40. It narrows the gap late
but never crosses W2. F is also below W2 at e20/e25, roughly level at e30/e40.

## Q5. Did C2's epoch-40 AP exceed the historical W2 0.2618669?

**No.** `AP_C2 = 0.2609026157`, `D_C2 = −0.0009643024` (−0.368 % relative).
The result is inside the no-clear band `−0.002 < D_C2 < +0.002`; it is not a
`C2_REGRESSION` (threshold ≤ −0.002) either. Best diagnostic C2 epoch was 35
(0.262161), but the protocol forbids substituting it for epoch 40; the best-to-40
drop is −0.001258.

## Q6. Did C2 beat the near-parameter-matched FFN?

**No on the primary metric.** `AP_F = 0.2614996012`,
`D_CF = AP_C2 − AP_F = −0.0005970` (−0.228 % relative to F). F also has the
better Train AP (0.587490 vs 0.584694); its edge is not explained by a smaller
Train−Valid gap (0.32599 vs 0.32379).

The per-task picture is more nuanced: on the 107 high-support assays C2 is
**+0.000925** above F (52/107 improved), while F is better on the 18 sparse
5–19 assays (C2−F mean −0.007646) and on one of the two <5 assays. So the
aggregate `D_CF < 0` is driven by the sparse strata; even on high-support,
however, C2's +0.000925 is far below the +0.003 threshold and only 52/107
assays improve. This is not evidence for the path mechanism.

## Q7. Is the gain broad across high-support assays?

**No.** Against W2, over the 107 assays with ≥20 Validation positives:

```
improved 52 / 107      regressed 55 / 107
mean Δ   −0.001657     median Δ −0.000151
```

Across all 127 scorable assays: 61 improved, 66 regressed, mean −0.000964,
median −0.000170. The positive mass is concentrated: the 5 largest positive
per-task deltas account for **47.6 %** of all positive mass (10 largest:
66.3 %), and the Spearman correlation between ΔAP and Validation support is
−0.005. The 5–19 stratum has a positive mean (+0.005137, 9/18) but a negative
median (−0.003503); the two <5 assays both regress (mean −0.018798). There is
no broad high-support gain to explain.

## Q8. Did Train AP keep rising while Validation AP fell?

Mildly, yes — the added capacity is absorbed by train fit, not by Validation:

| arm | Train AP40 | Train BCE40 | Valid AP40 | Valid BCE40 | Train−Valid gap |
|---|---:|---:|---:|---:|---:|
| W2 | 0.581875 | 0.025215 | 0.261867 | 0.046538 | 0.320008 |
| C2 | 0.584694 | 0.025136 | 0.260903 | 0.046693 | 0.323791 |
| F | 0.587490 | 0.024990 | 0.261500 | 0.046843 | 0.325991 |

C2's Train AP is +0.002819 vs W2 (86/128 tasks improve) while Validation is
−0.000964. F adds +0.005616 Train AP (94/128 tasks) for −0.000367 Validation.
The effect is small compared with the W2→W3 round (+0.124 Train AP for −0.0029
Validation), but the direction is the same: the new modules buy train fit.

Late-curve behavior: C2's best epoch 35 → epoch 40 is −0.001258; e30→e40 is
+0.000312 and e25→e40 is −0.000006. The last-10-epoch slope is
−2.27e−5/epoch (std 5.17e−4), i.e. flat, not a collapse. F: best e34 →
e40 −0.001059; e30→e40 +0.000316; last-10 slope −7.51e−5/epoch (std
4.95e−4). The late decline is present but is not specific to C2.

## Q9. Did the C2 vector gate actually learn a non-zero change?

Yes, unambiguously. On the fixed 64-molecule diagnostic batch (4,966 paths,
1,671 atoms):

| quantity | init | epoch 10 | epoch 25 | epoch 40 |
|---|---:|---:|---:|---:|
| `‖Δg‖` RMS (all 192 dims) | 0 | 0.3014 | 0.4735 | **0.4921** |
| mean per-path `‖Δg‖₂` | 0 | 4.176 | 6.819 | **5.903** |
| `mean‖Δg‖ / mean|g_scalar|` | 0 | 3.375 | — | **5.958** |
| non-zero path fraction | 0 | 1.0 | 1.0 | **1.0** |
| `‖V‖₂` | 0 | 8.579 | — | **11.234** |
| `‖E_c‖₂` | 20.232 | 21.033 | — | **21.286** |
| `‖E_e‖₂` | 8.884 | 10.500 | — | **11.084** |

The gate is not stuck at W2: the per-path vector deviation is several times the
scalar gate's own magnitude, and all three new tensors grow. The gradient audit shows the
expected zero-residual pattern: `V` gets non-zero gradient on step 1, `E_c`/`E_e`
only from step 2 (their step-1 gradient is exactly 0 because `V = 0`); all
become non-zero by step 3. F's `fc2` gets gradient on step 1 and `fc1` from
step 2. The failure is therefore **not** "C2 did not train"; it is that this
particular path-conditioned modulation did not improve held-out AP.

## Q10. What is the real speed cost of the extra path computation?

Measured back-to-back on the same A100 (`res-gpu1`), C2 then F:

| metric | W2 (historical) | C2 | F | C2 vs F |
|---|---:|---:|---:|---:|
| step p50, all epochs | 42.09 ms | 47.13 ms | **40.55 ms** | **+16.2 %** |
| train seconds / epoch | 116.36 s | 130.84 s | **110.90 s** | **+18.0 %** |
| Validation pass / epoch | 7.21 s | 8.26 s | **7.28 s** | **+13.5 %** |
| GPU-hours (process) | 1.3949 | **1.5711** | **1.3364** | **+17.6 %** |
| peak allocated / reserved | 223 / 424 MiB | 284 / 588 MiB | 225 / 424 MiB | — |

C2 is ~16–18 % slower than F despite having **lower** theoretical arithmetic:
per molecule the new C2 path work is ≈0.483 MFLOP (78.5 paths × 6,160 FLOPs)
vs F's ≈0.545 MFLOP (25.97 atoms × 20,992 FLOPs), i.e. ≈0.169 vs
≈0.191 TFLOP per 350,343-molecule epoch. The overhead is the irregular
path-indexed gather + many small `16×192` matmuls, not FLOP count. Relative to
the historical W2 run, C2 is +12.6 % GPU-hours; F is within run-to-run noise of
W2 (−4.2 %, but the two are not a controlled same-session pair).

## Q11. Is Conditional S2 worth developing in the next round?

**No, not as a Validation-AP mechanism under this recipe.** The negative result
is specific and informative:

- the mechanism is real and strongly active (Q9), so this is not a dead gate;
- it still does not beat the scalar-gate W2, and it does not beat a plain
  near-equal-parameter FFN;
- its apparent strengths are local: +0.000925 mean on high-support assays vs F,
  and a positive 5–19 mean vs W2 (+0.005137, 9/18) — but the median is negative
  and the high-support breadth is only 52/107;
- it costs ~16–18 % more per epoch than the FFN for no primary-metric gain.

The honest conclusion is **`NO_CLEAR_C2_GAIN` / `CAPACITY_ONLY_OR_MIXED`-shaped
evidence against the hypothesis**: path-type × endpoint-type vector modulation
is not the missing inductive bias for MolPCBA under the frozen one-shot LCRC
recipe. A future round would need a *different* hypothesis (e.g. a regularised
or low-rank-constrained gate, a different placement, or a multi-seed
confirmation), not a larger rank or a longer schedule. This round authorises no
such follow-up run.

## Q12. GPU hours, commit, report and checkpoint paths

```
training commit      c4bb21bb54387cba1fb6d96513b96b6043973f64  (both runs, clean)
audit                cs2-audit-20261107-213349-f4ec2231   21:33:49 → 21:35:33
C2                   cs2-c2-20261107-213923-7b2da9ba      21:39:26 → 23:13:55
F                    cs2-f-20261107-231627-6028cee5       23:16:31 → 00:36:53
host / pool          res = a100-2, pool res-gpu1, CUDA_VISIBLE_DEVICES=1
driver / torch       550.163.01 / 2.5.1+cu124 (CUDA 12.4)
```

Budget (all `cs2-*` jobs, charged wall clock; `analysis/gpu_ledger.json`; the
in-process `gpu_hours` sum is 2.9075):

| item | GPU-hours |
|---|---:|
| preflight + audit + 6×10 s GPU sampling | 0.0281 |
| C2 formal 40 epochs | 1.5747 |
| F formal 40 epochs | 1.3394 |
| **total charged** | **2.9422** |

Target ≤ 4, hard cap ≤ 6 → **`WITHIN_TARGET`**. No OOM, no non-finite event, no
restart, no failed formal attempt. Gradient clipping fired 14 times in each run
(recorded `grad_clip = 5.0`).

Checkpoints (SHA-256 from each run's own `run.json`; full files under the
git-ignored `runs/`):

| arm | epoch-40 primary | best diagnostic (unused) | resumable last | RNG state |
|---|---|---|---|---|
| C2 | `runs/C2/checkpoint_epoch40.pt`<br>`8f0853766e9f97eef10375ea4bd94ef747886cca24e87663f1920f953ea37c86` | `3d06d0666a153c044917a46cd17e2e76fad7354780bdd4fe3f38a7ecab35ddf2` (e35) | `06388d044a7999979ce89fb1f814ee422d5728c448a8745aa961f440e50f06cf` | `c3180f6552dcca7e0a44284b3e0cfd249d4e339ed07ebb5b7517bd47298e504d` |
| F | `runs/F/checkpoint_epoch40.pt`<br>`a9095c0098479b1919699d43e86541c70803a3290bd36ff2a33347dfc1b94cba` | `921b4001a6da5fe5a5b7ab7d132488cd18216406819bbab1d4956b46145cb263` (e34) | `23de6b39daa09002def3d2b7c8b1a1fee34887d96e8eaa433226d5512db96cd9` | `ccfbaeeb7267224a6ea67f23b13bb99388aa66ff389f9ed6e70c55b2c72fe7d1` |

Each primary checkpoint contains the model state, AdamW state, scheduler step,
RNG state, batch plan, model audit, recipe and environment; the resume audit
reconstructs model/optimizer/RNG/data order exactly
(`analysis/initialization_audit.json → checkpoint_roundtrip`).

---

## Per-task deltas

`analysis/per_task_deltas.csv` has all 128 rows (127 scorable; task 45 has no
Validation positives). Positive = the first-named model is better.

| comparison | n | improved | regressed | mean Δ | median Δ |
|---|---:|---:|---:|---:|---:|
| C2 − W2, all | 127 | 61 | 66 | −0.000964 | −0.000170 |
| C2 − W2, support ≥20 | 107 | **52** | 55 | **−0.001657** | −0.000151 |
| C2 − W2, support 5–19 | 18 | 9 | 9 | +0.005137 | −0.003503 |
| C2 − W2, support <5 | 2 | 0 | 2 | −0.018798 | −0.018798 |
| C2 − F, all | 127 | 62 | 65 | −0.000597 | −0.000161 |
| C2 − F, support ≥20 | 107 | 52 | 55 | +0.000925 | −0.000161 |
| C2 − F, support 5–19 | 18 | 9 | 9 | −0.007646 | −0.001419 |
| F − W2, all | 127 | 63 | 63 (1 tie) | −0.000367 | 0.000000 |
| F − W2, support ≥20 | 107 | 54 | 53 | −0.002582 | +0.000427 |
| F − W2, support 5–19 | 18 | 9 | 9 | +0.012783 | −0.002296 |

Positive-mass concentration (top-5 / top-10 share of all positive per-task
deltas): C2−W2 **47.6 % / 66.3 %**, C2−F 41.0 % / 58.6 %, F−W2 44.1 % / 65.0 %.
The biggest C2−W2 gains are tasks 117 (+0.1710), 28 (+0.1393), 55 (+0.0834),
102 (+0.0686), 85 (+0.0508); the biggest losses are tasks 11 (−0.1030),
69 (−0.0890), 19 (−0.0800), 9 (−0.0605), 120 (−0.0547). The gain is not broad;
it is a small set of large wins cancelled by many small losses.

## Pre-registered decision

`analysis/decision.json`:

```
label = NO_CLEAR_C2_GAIN
D_C2 = −0.0009643      D_F = −0.0003673      D_CF = −0.0005970
```

Checks: `D_C2 ≥ 0.005` false; `D_CF ≥ 0.003` false; high-support mean
≥ 0.003 false; high-support improved ≥ 60/107 false; the support-stratum size
107 is correct. Mixed flags recorded separately: C2 beats W2 false, F matches or
beats C2 true, F actually beats C2 true, high-support mean below 0.003 true,
high-support improved below 60 true, gains concentrated in sparse assays true,
early-peak-then-regress false, positive `D_CF` below threshold false. The label
is a resource decision for one seed, not a significance test.

## Audits and parity

`analysis/initialization_audit.json` — **verdict OK**: base W2 matches the
historical fingerprints; C2/F original subsets bitwise equal; exact parameter
counts; CPU initial logits bitwise equal; CUDA initial logits max |Δ| 9.5e-7
(batch) and 7.6e-5 (full Validation, C2) / 6.1e-5 (F), AP deltas ≤ 3e-7; W2
epoch-0 replay AP 0.02906286 vs recorded 0.02906333 (abs 4.7e-7); gate/FFN
gradients reach all new tensors; checkpoint round-trip exact; Test labels all
NaN and no Test row in the plan.

`analysis/architecture_audit.json` — **verdict OK**: path_cat range [0, 393],
path_end63 range [0, 78], path_end inside the atom range, path count equals the
cache meta (34,369,144); gate shape `[4966, 192]` on the audit batch and
`[n_paths, 192]` generally; one atom/bond/local/dropout/f/rho/readout call per
forward; batch tensors unchanged; post-`local` path needs no structure; atom
renumbering changes logits by ≤4.8e-6; masked BCE equals the observed-only BCE.

`analysis/training_parity.json` — **all checks pass**: same recipe as W2, same
40-epoch plan and prefix, same base-state SHA/fingerprint, same global seed
state, epoch-0 AP parity between arms and vs W2, same end-of-epoch-1 torch RNG
hash, all 40/40 Validation epochs completed.

## What this round does not establish

- **No Test metric.** Test labels are NaN in the frozen artifact and no Test row
  was indexed, predicted or scored (`test_evaluated: false` in both runs).
- **One seed per arm.** The epoch-level AP std over the last 10 epochs is
  ≈5e−4, but one seed cannot separate a −0.001 difference from run-to-run
  variation; the label is a resource decision, not a significance test.
- **No exact computation match.** C2 and F add their module at different
  operators (S2 content vs local representation). The comparison is a
  parameter-capacity comparison; the report does not attribute every difference
  to the path mechanism.
- **No claim that Conditional S2 is universally useless.** It did not help on
  this dataset, recipe and seed, at this rank and placement. The gate was
  demonstrably active, so the negative result is about the inductive bias /
  optimisation trade-off, not about a dead module.
- **No automatic follow-up.** No seed 1, no rank sweep, no length extension, no
  Test access, no additional model.
