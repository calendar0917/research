# REPORT — LCRC PCBA explicit-performance sprint v1

Round `lcrc-pcba-explicit-performance-sprint-v1` · seed 0 · `ogbg-molpcba`
official scaffold split · main metric = epoch-40 official OGB Validation macro AP.
Round code commit: `628c2487` (A1/A2/A3); A0 started on `f6c5e860`.
Protocol: [`PROTOCOL.md`](PROTOCOL.md) (including §10 deviation log).

## 1. Result in one screen

| arm | addition | params | Valid AP40 | Δ vs A0 | frozen label | Train AP40 |
|---|---|---:|---:|---:|---|---:|
| A0 (reference) | frozen LCRC-A, historical run | 287,644 | 0.246356 | — | `REFERENCE` | 0.419657 |
| A1 | task-conditioned attention residual | 328,988 | 0.246219 | **−0.000137** | `REGRESSION` | 0.441263 |
| A2 | physicochemical descriptor residual | 296,732 | 0.243163 | **−0.003193** | `REGRESSION` | 0.424410 |
| A3 | explicit rooted 2-hop branch tokens | 553,948 | 0.248250 | **+0.001894** | `NO_CLEAR_GAIN` | 0.437145 |

* **No candidate reaches the pre-registered screen.** The eligibility threshold
  was Δ ≥ 0.003; the best arm (A3) is +0.0019, below even the 0.002
  weak-signal line.
* **No COMBO was run**: only 0 arms cleared 0.003, while the rule requires at
  least two eligible arms and a maximum Δ ≥ 0.005 (§7 of the protocol).
* **A3's marginal gain is not broad and is smaller than epoch-to-epoch noise.**
  It is concentrated in the 18 low-support assays (5–19 Validation positives:
  mean Δ = +0.0142, contributing 18/127 × 0.0142 ≈ +0.0020 of the +0.0019 macro
  Δ), while the 107 large-support assays average −0.00014. Between epochs 20
  and 40 the A3 curve has std 0.0026 and consecutive-epoch difference std
  0.0014, i.e. the final Δ is within one fluctuation step.
* **A1 and A2 improve Train AP (+0.0216 / +0.0048) without any Validation gain**
  and with a larger Train−Valid gap than the reference (0.1950 / 0.1812 vs
  0.1733). The extra capacity is spent on Train-specific structure.
* 5.86 A100 GPU-hours total (73 % of the 8 h target). **No Test row was ever
  indexed, scored or inferred on.**

Two deviations weaken all causal claims and are carried through every artifact:
the A0 rerun was cancelled by principal instruction at epoch 29/40, so the
reference is the **historical** phase-C A run, not a same-round control; and A3
was evaluated every epoch while A1/A2 kept the pre-registered 10/20/30/40 points
(§3).

## 2. What was frozen before the runs

| object | value |
|---|---|
| base init fingerprint (C1 + original A readout) | `1b1f1ad35ab33f4107bc600166fce838d5e16ff5def2348743c73c0211dcf7c3` |
| 40-epoch batch-order plan | `06f437299fa451e449202a990d28f3b8d101e7aec709bcd72a878c8de864b6f6` |
| consumed-40 prefix | `1a24dfca54898dd2b0e7c58d2b2f1956df67b233fdceaa7a376f81a2377a5f58` |
| feature spec / normalization / split / labels | inherited from `pcba_v1`, Train-only norm refit `6ce50f98…` |
| recipe | batch 128, AdamW, 3e-5→3e-4 warmup (2,738 steps) then cosine to 1e-5, 109,520 updates, dropout 0.10, clip 5.0, FP32, masked BCE |
| evaluation | official OGB `Evaluator("ogbg-molpcba")`, epochs 10/20/30/40, epoch 40 primary |

Every formal run re-verified the two hashes before its first step
(`run.json: batch_plan`), and `run.json: features.hashes` pins the feature cache.

## 3. Deviations from the frozen protocol

1. **A0 rerun cancelled** (principal instruction) at epoch 29/40; 0.994 GPU-h
   charged. The reference is `lcrc_pcba_no_mp_sprint_v1/runs/phaseC_seed0_A`
   (AP40 0.2463560956, params 287,644, commit `b1b917dc`). There is **no
   same-round 40-epoch control**, so a part of every Δ below can be
   implementation/run drift rather than module effect.
   As a partial sanity check, the cancelled A0 and A1 agreed at epoch 20
   (0.238138 vs 0.238163) on the identical data order.
2. **A3 evaluated every epoch** (principal request, 40 official Validation
   evaluations, ≈ +2.7 min); A1/A2 kept 10/20/30/40. Evaluation is inference
   only, consumes no RNG and never selected a checkpoint, so training is
   unchanged; only the reporting density differs.
3. A1/A2 ran on `628c2487`, A0 on `f6c5e860`; the diff between those commits is
   documentation/analysis only — `ep_train.py`, `ep_model.py`, `ep_data.py`,
   `ep_common.py` are untouched.

## 4. Architecture audit

Base (identical in every arm, bitwise-identical at init, verified by preflight
`state_tensors_equal: true`): frozen C1 one-shot S1/S2 local statistics
(245,184 params) + original 4-head graph attention pooling readout
(42,460 params). `logits = C1_f_rho(h) + original_A_pool(h)`.

| arm | module | added reads | new params | total | init |
|---|---|---|---:|---:|---|
| A0 | — | — | 0 | 287,644 | — |
| A1 | `task_conditioned_attention` | `h`, `atom_batch`, `n_graphs` | 41,344 | 328,988 | w,b = 0, private seed 101 |
| A2 | `physchem_descriptor_residual` | `physchem` only | 9,088 | 296,732 | last Linear = 0, private seed 211 |
| A3 | `explicit_rooted_branch_tokens` | `h`, `bt_ptr`, `bt_bucket` | 266,304 | 553,948 | projection = 0, private seed 307 |

* All added modules are **zero-residual at step 0**: epoch-0 Validation AP is
  identical across arms (0.028295) and max logit difference at init ≤ 1.9e-6.
* `reads` is machine-checked: no added module touches `bond_index`,
  `bond_feat`, `nbr_*` or `path_*`; forward FLOPs per molecule 8.69 M (A0),
  9.71 M (A1), 8.71 M (A2), 8.86 M (A3).
* No learnable inter-atom message passing, no hidden-state update along
  `bond_index`, no inter-node QKV self-attention. A1 aggregates frozen static
  `h` per molecule; A3 adjacency is offline preprocessing only.
* The extra modules are **not isolated**: their residual changes the loss and
  therefore the gradient into the shared C1 trunk. The comparison is arm-level,
  not a clean module-isolation experiment.

## 5. Feature construction audit

### A2 — deterministic materialised descriptors

RDKit `2026.09.1`; 10 columns in fixed order TPSA, NumHDonors, NumHAcceptors,
log1p(MolWt), Crippen MolLogP, NumRotatableBonds, TotalFormalCharge,
NumAromaticRings, MaxRingAtomCount, FractionCSP3, plus a missing flag (dim 11).

* Coverage: Train 350,341/350,343 = 0.999994, Valid 43,793/43,793 = 1.0;
  2 molecules fail (`MolFromSmiles → None`, Train rows 5823 and 303198). They are
  kept, filled with the Train mean, flagged `1.0`; both are recorded by row and
  exception.
* Mean/scale fitted on the 350,341 successfully computed **Train** rows only
  (fingerprint `c3f2d59e…`), zero-std → scale 1; Validation reuses Train
  statistics; Test never enters the cache.
* Cache `c720320c…` (21.5 MB), built on CPU in ~60 s with 24 workers, git-ignored.

### A3 — explicit rooted 2-hop branch tokens

Signature per directed bond (undirected cache bonds expanded to both
orientations):

```
( atom_type(v), bond_type(v,u), atom_type(u),
  sorted[(bond_type(u,w), atom_type(w), is_edge(v,w)) for w in N(u), w != v] )
```

stable `blake2b(digest_size=8)` over little-endian uint16 packing → `uint64 % 8192`.

* 394,136 rows attempted (Train+Valid), 22,011,330 tokens, 55.85 tokens per
  molecule, max out-degree 5, 104 genuinely isolated atoms (degree 0), 0 Test
  rows built.
* **The token alphabet is coarse**: only 2,340 distinct signatures map into
  2,044 used buckets, and a single bucket holds 22.0 % of all tokens; the
  minimum implied bucket-collision count is 296. The 8,192-row embedding is
  therefore mostly unused (6,148 empty buckets). This is a property of the
  frozen 16 atom-type / 4 bond-type alphabet, not a bug, but it limits how much
  structural detail the token can carry.
* Cache `861739b8…` (93 MB), built on CPU in ~60 s, git-ignored.

Both caches cover the same 394,136 rows (`rows_attempted_sha256` equal), and the
preflight verified per-batch row alignment, bucket range, pointer layout and
that all Test rows carry zero features.

## 6. Preflight evidence (all gates passed before any formal run)

GPU preflight `ep-preflight-20261106-184544` (`res-gpu1`, 0.0195 GPU-h), 9
checks per arm, all `passed: true`:

* init identity vs frozen C1+A: max logit diff 1.4e-6 (A0/A3) … 1.9e-6 (A1);
* A0 vs the historical A implementation: all state tensors equal, max logit
  diff 9.5e-7;
* epoch-0 Validation AP identical across arms (0.028295) and equal to C1;
* frozen init fingerprint and batch plan equal to the historical values;
* parameter counts equal to the frozen module table;
* A1: per-molecule softmax sum deviation 2.4e-7, permutation invariance 3.0e-7,
  no cross-molecule normalisation (2.4e-7), other graph unchanged (2.4e-7);
* A2: coverage/alignment/normalisation gates, 0 finite Test rows;
* A3: token alignment, bucket range, 0 Test tokens;
* effective non-zero gradients in every added parameter after 3 real updates
  (the only structual zeros are the 4 known `readout.*.fc2.bias`);
* Test isolation: labels all NaN on Test, prediction path raises
  `PermissionError` on Test rows;
* masked BCE and official AP agree with a manual reference (diff 0.0).

Measured step time (p50) and 40-epoch projection at preflight: A0 38.3 ms /
1.15 h, A1 52.4 ms / 1.62 h, A2 41.3 ms / 1.26 h, A3 44.4 ms / 1.37 h.

## 7. Main results

| arm | AP10 | AP20 | AP30 | AP40 | Δ40 | 30→40 | Train AP | Train BCE | Valid BCE | Train−Valid AP | params | GPU-h | peak MB | p50 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A0 historical | — | — | — | 0.246356 | — | — | 0.419657 | 0.030858 | — | 0.173301 | 287,644 | 0.663 | — | — |
| A0 cancelled | 0.221293 | 0.238138 | — | — | — | — | — | — | — | — | 287,644 | 0.994 | — | — |
| A1 | 0.222987 | 0.238163 | 0.245643 | 0.246219 | −0.000137 | +0.00058 | 0.441263 | 0.030400 | 0.04584 | 0.195043 | 328,988 | 1.822 | 414.8 | 61.86 |
| A2 | 0.213801 | 0.235324 | 0.242687 | 0.243163 | −0.003193 | +0.00048 | 0.424410 | 0.030761 | 0.04592 | 0.181247 | 296,732 | 1.440 | 123.2 | 47.21 |
| A3 | 0.219884 | 0.241890 | 0.248873 | 0.248250 | +0.001894 | −0.00062 | 0.437145 | 0.030239 | 0.04606 | 0.188894 | 553,948 | 1.556 | 127.2 | 45.67 |

Frozen labels are mechanical: any Δ < 0 is `REGRESSION`, so A1's −0.000137 is
labelled `REGRESSION` even though it is 0.05 σ of the epoch-level noise. In the
text below, |Δ| < 0.002 is treated as **indistinguishable from zero at one
seed**, which does not overrule the label but qualifies it.

`run.json` for all three completed arms: 40/40 epochs, 109,520 updates,
checkpoint-epoch-40 saved with SHA-256, 20 gradient-clip events each, 0
non-finite events, 4/4 scheduled evaluations.

## 8. Curves and late-epoch behaviour

* A1 rises 0.22299 → 0.23816 → 0.24564 → 0.24622; gain collapses to +0.00058
  over the last 10 epochs, i.e. it has plateaued at the A0 level.
* A2 is below the reference at every measured epoch (0.21380 / 0.23532 /
  0.24269 / 0.24316), with a +0.00048 final-decade change.
* A3's dense curve (every epoch) is non-monotone:

  | epoch | 1 | 5 | 10 | 15 | 20 | 25 | **29** | 30 | 35 | **40** |
  |---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
  | Valid AP | 0.0761 | 0.1863 | 0.2199 | 0.2349 | 0.2419 | 0.2447 | **0.2503** | 0.2489 | 0.2491 | **0.2483** |

  Its maximum is epoch 29 (+0.0040 vs the historical A0, which would have
  looked like a weak positive at that epoch), and it ends −0.00062 lower at
  epoch 40. Over epochs 20–40: mean 0.24691, std 0.00259, min 0.24082, max
  0.25032; consecutive-epoch difference std 0.00141. This is the best empirical
  estimate of the single-seed epoch-level noise available in this round, and it
  is larger than A3's final gain.

## 9. Per-task comparison (127 scorable tasks of 128)

Improved / degraded counts vs the reference per-task AP:

| arm | improved | degraded | tied | mean Δ | median Δ | top-10 share of positive Δ | Spearman(Δ, support) |
|---|---:|---:|---:|---:|---:|---:|---:|
| A1 | 63 | 64 | 0 | −0.00014 | −0.00037 | 49 % | −0.20 |
| A2 | 62 | 65 | 0 | −0.00319 | −0.00023 | 55 % | −0.30 |
| A3 | 61 | 65 | 1 | +0.00189 | −0.00007 | 56 % | −0.01 |

By Validation-positive stratum:

| arm | ge20 (107 tasks) | 5–19 (18 tasks) | <5 (2 tasks) |
|---|---:|---:|---:|
| A1 | +0.00153 (54↑/53↓) | −0.00929 (9↑/9↓) | −0.00713 (0↑/2↓) |
| A2 | −0.00217 (52↑/55↓) | −0.01061 (9↑/9↓) | +0.00870 (1↑/1↓) |
| A3 | −0.00014 (50↑/57↓) | **+0.01422 (11↑/7↓)** | −0.00003 (0↑/1↓) |

* A3's macro Δ is almost exactly the 5–19 stratum contribution
  (18/127 × 0.01422 = +0.00201), and the large-support stratum is slightly
  negative. A gain that lives only in assays with 5–19 positives is the least
  trustworthy kind: AP there is high-variance, and this round has no seed
  replication to separate it from noise.
* Each arm's positive mass is concentrated: the top 10 tasks carry 49–56 % of
  the summed positive per-task Δ, while the median task is flat or negative.
* Largest task gains / regressions (Δ, Validation positives): A1 `+0.100` task
  112 (23), `+0.094` task 117 (9), `−0.191` task 31 (6), `−0.095` task 27 (16);
  A2 `+0.106` task 2 (16), `+0.091` task 120 (13), `−0.147` task 102 (37),
  `−0.142` task 31 (6); A3 `+0.138` task 2 (16), `+0.101` task 120 (13),
  `−0.086` task 102 (37), `−0.067` task 9 (14).
* Tasks 31, 62, 77 and 102 regress in **all three** arms (−0.05 … −0.19). Since
  the arms differ only by their readout residual, this common pattern points at
  shared-trunk trajectory / run-level noise rather than at any one module.

## 10. Reading the three hypotheses

**A1 (task-conditioned readout): no Validation gain, more Train fit.** The
module adds 41k parameters and the largest per-step cost (preflight p50
52.4 ms vs 38.3 ms for A0, +37 %; 414.8 MB peak vs 123–127 MB for A2/A3), yet AP40 is identical to the reference within 0.00014 while
Train AP rises by 0.0216 and the Train−Valid gap widens by 0.022. The
task-conditioned attention is expressive enough to fit Train-specific assay
structure but does not transfer. It is not evidence that per-assay structure is
useless — only that this parameterisation, trained end-to-end for 40 epochs,
buys nothing on Validation.

**A2 (physchem descriptors): slightly negative.** The descriptors are faithful,
deterministic and fully covered, and the module is tiny (+9k), but AP40 is 0.0032
below the reference and the deficit exists at every measured epoch. The most
likely reading is optimisation/regularisation interference (an extra dropout and
an 11→64→128 path on a 287k-parameter model), not "chemistry priors do not
matter": 10 global descriptors cannot express much beyond size/logP, and the
frozen global 42-d descriptor + `rho` already carries similar information.

**A3 (explicit branch tokens): a marginal, concentrated, noise-sized gain.** It
is the only arm above the reference (+0.0019), but (i) below the 0.003 screen
and the 0.002 weak line, (ii) entirely carried by low-support assays, (iii)
smaller than the epoch-level fluctuation and non-monotone (peak at epoch 29,
decline to 40), and (iv) built on a token alphabet where 22 % of all tokens fall
into one bucket. The honest conclusion is **not** "explicit branch structure
helps" but "this tokenisation, at this scale and seed, does not produce a
detectable Validation gain".

## 11. Engineering cost

| item | GPU-h |
|---|---:|
| feature caches (CPU, 24 workers, ~60 s) | 0.000 |
| GPU preflight (incl. one failed attempt, 13 s) | 0.020 |
| 1-epoch end-to-end trainer smoke test | 0.049 |
| A0 rerun, cancelled at epoch 29/40 | 0.994 |
| A1 (40 epochs, 4 evals) | 1.822 |
| A2 (40 epochs, 4 evals) | 1.440 |
| A3 (40 epochs, 40 evals) | 1.556 |
| **total** | **5.881** |

Budget: target 8 h, hard 10 h, max 2 concurrent GPUs. Used 73 % of target. Only
GPU1 was used because GPU0 remained occupied by an external job (16 GB,
100 % utilisation) for the whole round; no preemption or sharing was attempted.
Wall clock: 18:32 (cache build) → 01:07:44 (A3 finished) ≈ 6 h 35 min. The three
formal arms ran sequentially (~1.44–1.82 h each, 128–164 s per epoch); they are
independent and could have been co-scheduled on GPU1 (each uses < 0.5 GB, and
step time is dominated by CPU collate + H2D, not SM occupancy), which would have
saved roughly 1.5 h of wall clock at the price of noisier per-run timing
provenance.

## 12. Decision trace

```
Δ_i = AP40(Ai) − 0.246356 (historical A0)
A1 −0.000137   A2 −0.003193   A3 +0.001894
eligible (Δ ≥ 0.003): []                      -> 0 arms
rule: fewer than 2 eligible        -> no COMBO (SKIPPED_BY_RULE)
```

`analysis/decision.json` records `a0_reference.same_round = false` and the
reason; `analysis/gpu_ledger.json` charges the cancelled A0 and the smoke test;
`analysis/run_registry.json` records each run's commit, pool, GPU visibility,
driver, timings and checkpoint hash.

## 13. What this round does not establish

1. **No Test result.** Test labels are NaN in the frozen artifact and no code
   path indexes Test; Test AP is not measured and no Test claim is made.
2. **Single seed.** With epoch-level AP fluctuations of ≈0.0014–0.0026, only
   effects well above ~0.003 would be interpretable; none were observed.
3. **No same-round A0 control.** A0 was cancelled; Δ values mix module effects
   with run/implementation drift. The A1-vs-cancelled-A0 agreement at epoch 20
   (2.5e-5) is reassuring but not a 40-epoch control.
4. **Arm-level, not module-isolated.** All modules are trained end-to-end
   through the shared C1 trunk, so their gradients also change the trunk.
5. **Not a claim about structural sufficiency in general.** A1/A2/A3 are three
   specific parameterisations with frozen budgets; null results here do not
   prove that per-assay readouts, physicochemical priors or explicit branch
   structure are uninformative in other forms.
6. Attention weights are not chemical explanations; a hashed-token gain would
   not by itself be an "expressivity" gain (parameters and regularisation are
   confounded); and Δ on Validation does not imply a Test gain.

## 14. Artifacts and reproduction

```
analysis/architecture_audit.json     params, reads, init, preflight per arm
analysis/feature_audit.json          A2/A3 construction, coverage, hashes, collisions
analysis/preflight.json              full 4-arm gate report
analysis/run_registry.json           commit/pool/GPU/driver/timing/checkpoint per run
analysis/gpu_ledger.json             budget accounting incl. cancelled A0 and smoke
analysis/epoch40_comparison.csv      main table
analysis/per_task_comparison.csv     128 tasks × {A0,A1,A2,A3} AP and Δ with strata
analysis/task_delta_summary.json     concentration/strata/Spearman analysis
analysis/decision.json               eligibility + COMBO rule trace
analysis/rr_status.json              captured rr provenance
runs/{A1,A2,A3}/run.json             full run records (checkpoint SHA included)
runs/A0/CANCELLED.json               cancelled-control record
runs/prep/build.json                 A2 build audit (A3 superseded, see build_branch.json)
runs/prep/build_branch.json          final A3 build audit
```

```bash
uv run --no-sync pytest -q tracks/structural-sufficiency/lcrc_pcba_explicit_performance_sprint_v1/tests  # 29 passed
rr run res ep-a1 --pool res-gpu1 --result <runs/A1> -- \
  uv run --no-sync python <code>/ep_train.py --arm A1 --seed 0 --device cuda \
  --out-dir <runs/A1> --soft-seconds 6800 --hard-seconds 7500
uv run --no-sync python <code>/ep_analyze.py --runs-dir <runs> \
  --preflight <runs>/preflight/preflight.json --build <runs>/prep/build.json \
  --build-branch <runs>/prep/build_branch.json --historical-a0 <phaseC_seed0_A> \
  --rr-status <analysis>/rr_status.json --out <analysis>
```
