# SSRC Phase 2a — does atom-level local iteration add value on top of a mature static encoder?

ZINC-12k, internal protocol `ssrc-phase2a-zinc-v1`, two seeds, 0.933 A100 GPU-hours.
Baseline: frozen Phase 1e BJOC-P3E, Monitor MAE **0.091071**.

**Round verdict: `CAPACITY_ONLY_INCREMENT`.**
A small extra per-atom stage after `AtomEncoder` buys **≈0.0055 MAE** (6.1 % relative)
over P3E. Rerouting that stage through **real bond messages adds nothing measurable**:
`Gain_MP` is +0.003764 at seed 0 and −0.000055 at seed 1, i.e. the two seeds disagree on
sign and the effect is smaller than the control arm's between-seed spread.
The interaction hypothesis is **not supported**; the capacity hypothesis is.

---

## 1. Question, and what was already excluded before spending GPU

Given a mature static encoder (P3E: atom → bond → junction → group-composer + reader,
413 505 parameters, Fit-only masked-SSL init, MAE 0.091071), does a small amount of
**atom-state propagation on real physical edges** beat a **parameter-matched
non-interactive per-atom computation**?

Both arms are inserted after `AtomEncoder`, before `BondEncoder`; every downstream P3E
component is byte-identical to the frozen baseline.

| arm | construction | adjacency access |
|---|---|---|
| **F** (`node_ffn`) | 2 × pre-norm residual FFN (96→192→96) applied per atom | none, by construction |
| **M** (`local_mp`) | 2 × GINE round on directed physical bonds (forward+reverse), shared 7→96 bond projection, same residual/norm shape, **no** zero-init gate, **no** global attention | real `bond_index`/`bond_feat` |

Dedupe against this track's history was done first and is recorded in
`notes/phase2a_protocol.md` §2. The decisive check: **no prior BJOC variant in this repo
ever updates `a = AtomEncoder(atom_feat)` from neighbour messages.** P3+zero-init gated
bond–bond residuals, SAC second-moment tensors, static pair/distance pooling, junction
ownership / Null-Organized routing, input-statistics and SSL searches all act on `b`,
`w`, `c`, the reader, or graph-level statistics. Nothing here is a renamed rerun.

## 2. Design, and the one thing that makes the comparison meaningful

Parameter match (new modules only): F adds 49 984, M adds 50 752 — **1.54 % apart**,
inside the 5 % band. Total parameters 463 489 vs 464 257 (1.0017×).
Forward GEMM FLOPs on a fixed 128-graph batch: 2.301 G (F) and 2.309 G (M) vs 2.007 G
(P3E) → **1.146× and 1.151×**, i.e. the two arms are within 0.4 % of each other in both
capacity and compute. That is what makes `MAE(F) − MAE(M)` an estimate of the
*interaction* term rather than of capacity.

Shared and frozen: BJOC trunk, global descriptors, Fit-only SSL encoder, training recipe,
batch order, soup rule, epoch budget. Init audit (`init_full_sha256`, trunk parameters
only) is identical across arms **within** a seed and different across seeds
(`d06d5e16…` seed 0, `c79e6b02…` seed 1); the encoder group hash is
`37282569…` at both seeds for all arms, because the SSL encoder is held fixed.

Disclosed deviation for seed 1: only the supervised trajectory varies (composer/reader
scratch init + shuffle seed 102). The two-seed spread reported below is therefore a
**lower bound** on the spread of the original two-seed recipe.

## 3. Pre-GPU gates and tests

`cpu_gates_phase2a.json` — **9/9 gate entries PASS**, covering the 10 pre-registered
checks (G3/G4 share one entry):

| gate | evidence |
|---|---|
| G1 row/endpoint maps | forward+reverse directed edges, endpoint indices in range on ring6/chain7/fused |
| G2 permutation / edge-direction / batch invariance | relabelling ≤ 7.5e-09 for all three kinds |
| G3 F is adjacency-insensitive | `bond_index` roll and `bond_feat` roll → change **0.0** exactly |
| G4 M is adjacency-sensitive | `bond_index` roll → 1.32 / 1.56 max abs change (two seeds) |
| G5 M does not copy `a` | relative branch shift 0.58–0.61 (F 0.71–0.76); prediction changes 0.23 mean abs under bond-feature roll vs 0.12–0.15 for F |
| G6 gradients | all new parameters receive nonzero gradient; no zero-gradient parameter |
| G7 init parity | trunk hash equal across kinds, local block hash equal F vs M |
| G8 no label input | no label/banned token in the batch; `batch_keys` = structural tensors only |
| G9 reader layout | reader input dim 321 = expected; base forward recompute max abs 0.0 |
| G10 execution | forward+backward finite on ring/chain/fused for every kind |

Additionally: **replay** of the frozen P3E soup on the Monitor reproduces the baseline to
`max|d| = 9.537e-07` and `MAE 0.091071030`. Local tests: **20 passed**
(`code/phase2a/tests/test_phase2a.py`; includes 3 artifact-dependent tests).

## 4. Results

Monitor, n = 1000, internal Fit 8000 / Monitor 1000 protocol. Predictions were frozen to
disk **before** any Monitor label was read; no early stopping and no model selection.

| arm | new params | total params | FLOPs vs P3E | MAE s0 | MAE s1 | **MAE mean** | RMSE s0 | RMSE s1 |
|---|---|---|---|---|---|---|---|---|
| P3E (frozen / matched) | 0 | 413 505 | 1.000× | 0.091 071 | 0.090 703 | **0.090 887** | 0.270 310 | 0.273 834 |
| F `NodeFFN-2` | +49 984 | 463 489 | 1.146× | 0.089 160 | 0.085 239 | **0.087 199** | 0.271 445 | 0.266 278 |
| M `LocalMP-2` | +50 752 | 464 257 | 1.151× | 0.085 396 | 0.085 294 | **0.085 345** | 0.275 518 | 0.271 745 |

Per-seed gains (positive = first arm better):

| contrast | seed 0 | seed 1 | mean |
|---|---|---|---|
| `Gain_P3E` = MAE(P3E) − MAE(M) | **+0.005675** | **+0.005409** | **+0.005542** |
| MAE(P3E) − MAE(F) | +0.001911 | +0.005465 | +0.003688 |
| **`Gain_MP` = MAE(F) − MAE(M)** | **+0.003764** | **−0.000055** | **+0.001854** |
| RMSE(M) − RMSE(P3E) | +0.005208 | −0.002090 | +0.001559 |

Between-seed spread of the same arm: **P3E 0.000368, M 0.000102, F 0.003921**.

Molecule-level paired bootstrap (discovery only, not training-seed uncertainty):
seed 0 `M−P3E` = −0.005675 CI [−0.009520, −0.001767]; seed 1 `M−P3E` = −0.005409
CI [−0.010366, −0.000796]; seed 1 `F−P3E` = −0.005465 CI [−0.010689, −0.000391];
seed 0 `M−F` = −0.003764 CI [−0.007803, +0.000264]; seed 1 `M−F` = +0.000055
CI [−0.004420, +0.004413].

**The 0.0055 gain is the stable finding; the 0.0038 M-vs-F gap is not.** The seed-0 gap
came almost entirely from F's seed-0 run being 0.0039 worse than its own seed-1 run, while
M moved by 0.0001. Seed 0 sampled a bad F trajectory — that is exactly the failure mode a
second seed exists to catch.

### Subgroups (two-seed mean MAE)

| group | n | P3E | F | M |
|---|---|---|---|---|
| all | 1000 | 0.0909 | 0.0872 | 0.0853 |
| cycle-penalty **complement** | 957 | 0.0770 | 0.0731 | 0.0712 |
| label-penalised | 43 | 0.4007 | 0.4002 | 0.3998 |
| RDKit large ring | 40 | 0.2318 | 0.2278 | 0.2232 |
| penalty-only, RDKit missed | 3 | 2.6539 | 2.6984 | 2.7555 |
| permutation-sensitive (diagnostic only) | 4 | 1.8821 | 1.8772 | 1.9189 |

The gain is **not** a penalty-group artefact: both arms improve the 957 complement, and the
penalty group is flat for all three models. Per-seed `M_vs_F` on the complement is
−0.004673 (s0, CI excludes 0) and **+0.000830 (s1)** — the subgroup advantage also fails to
replicate. M is consistently the best of the three on the 40 RDKit-large molecules, but
that group's seed noise (F moves 0.221→0.234) is larger than the between-arm gap.

### Efficiency

| | P3E | F | M |
|---|---|---|---|
| forward GEMM FLOPs (128 graphs) | 2.007 G | 2.301 G | 2.309 G |
| step time, seed 1 | 23.99 ms | 29.24 ms (1.22×) | 30.14 ms (1.26×) |
| train wall, seed 0 / seed 1 | — / 550 s | 603 / 677 s | 747 / 690 s |
| inference, 1000 molecules | — | 0.439 s | 0.462 s |

Step-time ratios (1.22× / 1.26×) are larger than the FLOP ratios (1.146× / 1.151×) because
the new modules are small, low-arithmetic-intensity kernels; M pays an extra ~0.9 ms/step
over F for the two propagation rounds.

## 5. The tail, and a pre-registered rule that was specified badly

The frozen seed-0 decision artefact says `INSUFFICIENT_EVIDENCE`. That is the honest
mechanical output and it has **not** been edited. Here is exactly why it happened.

The frozen `STRONG` band required `ΔRMSE ≤ +0.005`; seed 0 delivered **+0.005208**, missing
by 0.000208. Post-hoc (descriptive, no training, no threshold change —
`posthoc_phase2a.json`):

| diagnostic | value |
|---|---|
| MSE difference `M − P3E`, seed 0 | +0.002843, bootstrap CI **[−0.003706, +0.013303]** |
| RMSE difference CI, seed 0 | [−0.012974, +0.015464] |
| worst squared-error contributor | Monitor index 341 / official-train row **2052**, `y = −7.93` |
| its share of the total MSE difference | **1.59×** — one molecule flips the sign |
| is it permutation-unstable? | **yes** (Phase 1e audit: its cycle penalty changes under atom permutation) |
| RMSE on the 996 graph-function rows | P3E 0.147848 vs M 0.142046 (**M better**) |

Seed 1 settled the question: `ΔRMSE = −0.002090` (M better), and again a **single** molecule
accounts for ~2× of the total squared-error difference, this time in M's favour.

At n = 1000 with `max|y| ≈ 8`, RMSE is a one-molecule statistic. An absolute RMSE threshold
is therefore not a usable stability guard at this sample size, and it produced opposite
verdicts on the two seeds. This is a **protocol defect**, logged for future rounds: the
guard should be a paired MSE-difference CI, or be evaluated on the
label-is-a-graph-function complement. The threshold was **not** changed after the fact.

What *was* done: the miss and its diagnosis were written into
`results/phase2a/seed1_purchase_record.md` **before** any seed-1 training, together with the
six-condition checklist, the disclosed judgment that condition 3 ("RMSE not clearly worse")
was satisfied in substance, what was and was not bought, and the round stopping rule. A
reader who rejects that judgment can read this round as a single-seed
`INSUFFICIENT_EVIDENCE` result — and would then have drawn the **wrong** mechanism
conclusion, because seed 0 alone suggests M > F.

## 6. Decision trace

| step | artefact | status |
|---|---|---|
| seed 0, frozen rules | `decision_phase2a.json` | `INSUFFICIENT_EVIDENCE` (no band matched; RMSE proxy missed by 0.000208) |
| tail diagnostic | `posthoc_phase2a.json` | RMSE proxy is a 1-row statistic |
| purchase record (pre-seed-1) | `seed1_purchase_record.md` | disclosed judgment + stopping rule |
| seed 1, frozen rules with **matched** P3E baseline | `decision_seed1_phase2a.json` | `CAPACITY_ONLY_INCREMENT` |
| round verdict | `round_verdict_phase2a.json` | **`CAPACITY_ONLY_INCREMENT`** |

Seed-1 mechanical values behind that status: `Gain_MP = −0.000055 < 0.002`,
`Gain_P3E = +0.005409 ≥ 0.004`, `MAE(P3E) − MAE(F) = +0.005465 ≥ 0.004`,
`complement_gain = +0.005181`, `optimization_ok = true`, `ΔRMSE = −0.002090`.

Why not `STRONG_BACKBONE_MP_INCREMENT`: it requires M to be separated from the
capacity-matched control in **both** seeds. It is separated in neither the sign test
(seed 1 has F marginally ahead) nor the magnitude test (two-seed mean 0.0019 < 0.002 band).
Why not `NO_PRACTICAL_GAIN_OVER_P3E`: the increment over P3E is real, replicated, and
survives on the 957 complement. Why not `NONCOMPETITIVE_COMPUTE_TRADEOFF`: FLOPs are 1.15×,
not ≥1.5×. `buy_seed1 = false` at both seeds; no third seed was bought.

## 7. Isolation and provenance

- Monitor labels read **once**, in the scoring stage, after both predictions were frozen.
  Post-hoc and seed-1 scoring are re-reads of the same frozen arrays; no model, threshold,
  or selection decision changed between reads.
- Reserved labels read **0**. Official valid/test **never loaded**.
- Phase 1e loaded all 10 000 official-train labels, so the internal Reserved split is
  **historically exposed** and is not claimed pristine. This round only ever used Fit (8000)
  and Monitor (1000).
- No early stopping, no soup selection on Monitor, no per-arm tuning. Soup rule, epoch
  budget, batch order and recipe are the frozen Phase 1e ones.
- Same execution regime for all runs: `res` / node `a100-2` / driver 550.163.01 /
  torch 2.5.1+cu124.

Frozen inputs (unchanged): P3E soup `94ba0f5f…`, SSL state `0d045e60…`, encoder group
`37282569…`, P3E Monitor predictions `fc698ccc…`, batch-order hash `d142e307…`,
split `results/phase1b/split_zinc_phase1b.json`, labels
`results/phase1b/zinc_labels_fit_monitor.json`.

New predictions: F s0 `cc7d7124…`, M s0 `35ecc28e…`, P3E s1 `23187431…`,
F s1 `6194001579f732e3…`, M s1 `30bb154065966b22…`.

## 8. GPU ledger (8 runs, 0.933 GPU-h of a 4 h cap = 23.3 %)

| experiment | run_id | commit | GPU-s |
|---|---|---|---|
| `ssrc-p2a-prep4` | `…-82674f7f` | `bc4e46f8` | 7 |
| `ssrc-p2a-smoke` | `…-e5ca109b` | `bc4e46f8` | 19 |
| `ssrc-p2a-F-s0` | `…-6be0ba16` | `bc4e46f8` | 615 |
| `ssrc-p2a-M-s0` | `…-c0459e45` | `bc4e46f8` | 760 |
| `ssrc-p2a-score` | `…-65148438` | `bc4e46f8` | 5 |
| `ssrc-p2a-P3E-s1` | `…-32b0f389` | `adeee6b2` | 562 |
| `ssrc-p2a-F-s1` | `…-cfc28bb9` | `adeee6b2` | 689 |
| `ssrc-p2a-M-s1` | `…-c606f542` | `adeee6b2` | 701 |

Training loops account for 0.908 h of that; the remainder is startup, replay, prediction and
scoring. Full manifest with artifact hashes: `results/phase2a/run_manifest.tsv`.

## 9. Answers to the eight questions

**Q1. Does atom-level local iteration add predictive value over P3E?**
Yes for the *stage*, no for the *edges*. Any small extra per-atom stage helps by ≈0.0055 MAE
(replicated: +0.005675 / +0.005409). Using the graph instead of not using it does not help:
`Gain_MP` = +0.0038 / −0.0001.

**Q2. Is the increment mechanism or capacity?**
Capacity. At matched new parameters (1.5 % apart) and matched FLOPs (0.4 % apart), the
interactive arm is not better than the non-interactive one, and both beat P3E by the same
amount. Round status `CAPACITY_ONLY_INCREMENT`.

**Q3. Is the gain concentrated in the cycle-penalty subgroup or extreme molecules?**
No. Two-seed mean MAE on the 957 complement: P3E 0.0770, F 0.0731, M 0.0712. The 43
penalised molecules are flat (0.4007 / 0.4002 / 0.3998), so the aggregate gain is *not*
coming from the label artefact that dominated Phase 1b–1e.

**Q4. Is the error tail worse?**
Not established either way, and the metric cannot establish it here. Seed 0 `ΔRMSE` was
+0.0052 (one permutation-unstable molecule), seed 1 −0.0021 (one molecule the other way).
M improves median (0.0458 vs 0.0516), p90 (0.1655 vs 0.1823) and p99 (0.6074 vs 0.6904) at
seed 0, and is worse at p95 (0.2599 vs 0.2449). Mixed, dominated by single molecules.

**Q5. Is the cost worth it?**
+15 % forward FLOPs and +22–26 % wall per training step for ≈0.0055 MAE over P3E, with
0.93 GPU-h for the whole round. Cheap; but the increment is capacity, which is available
more simply (wider/one more per-atom block) than by adding a propagation module.

**Q6. Was optimisation healthy, or is this an optimisation artefact?**
Healthy and *matched* optimisation: identical recipe, epoch budget, batch order, soup rule;
soup-fit MAEs 0.0011 / 0.0010 / 0.0013 / 0.0008; final gradients on every new parameter
nonzero (`local.blocks.0` 0.21–0.85, `bond_proj` 0.026–0.054); no NaN, no divergence;
grad-clip events 9.6k–12.1k out of 22 680 steps for every arm including the baseline. The
one optimisation anomaly found is **F's seed-0 trajectory** (0.0039 above its own seed-1
run), which is a control-arm variance problem, not a mechanism result.

**Q7. Is one seed enough? Was seed 1 bought?**
No and yes. Seed 1 was bought under the disclosed conditions in
`seed1_purchase_record.md`, together with a matched seed-1 P3E baseline (same fixed SSL
encoder, same scratch init, same shuffle seed) so `Gain_P3E` stays paired. It changed the
conclusion: `Gain_P3E` replicated to within 0.0003, `Gain_MP` collapsed to zero.

**Q8. Is "P3E + GINE" a novel paper claim?**
No. GINE is Xu et al. (2019); P3E/BJOC is this repo's own encoder; stacking a GINE round
after an atom encoder is textbook engineering, and this round shows it buys nothing over a
parameter-matched FFN. What this round *does* support is a narrow methodological claim:
**at a fixed operating point, a small architectural add-on's gain can be decomposed into
capacity and interaction with a two-arm, matched-parameter, matched-FLOPs design, and the
interaction term here is zero within seed noise.** That is a within-repo negative result,
not a benchmark result: 0.091 → 0.085 is on an internal 8000/1000 split with Fit-only SSL
and is not comparable to official ZINC-12k test numbers (~0.05).

## 10. Next step, and what would be needed to claim otherwise

Judgment: **stop the "more propagation on top of P3E" line.** The mechanism question this
track keeps asking has now been answered negatively twice in different forms (Phase 1b:
iteration ≈ compute, not mechanism; Phase 2a: adjacency access ≈ no adjacency access at
matched capacity). Further widening, gating, depth or seeds of the same idea would be
re-asking a settled question.

What would change the answer — each requires a *new* pre-registration, not a rescue of this
round:

1. **A regime where interaction should matter.** The capacity gain (+0.0055) is large
   relative to the interaction gain (0). If interaction value exists, it should appear where
   the static encoder's receptive field is provably insufficient — i.e. a task whose target
   depends on a graph statistic that P3E's junction/composer path cannot represent. That
   needs a pre-registered *diagnostic* first (does P3E's representation miss it?), not
   another arm.
2. **Scale.** Both arms were +12 % parameters at 0.09 MAE. A capacity-vs-interaction
   decomposition at a much larger budget would test whether the interaction term stays zero;
   this round says nothing about that regime.
3. **Protocol repair.** Any future round must replace the absolute RMSE guard (see §5) and
   must budget ≥2 seeds *per arm* from the start, since the control arm's between-seed
   spread (0.0039) was larger than the effect being tested (0.0038).

Explicitly **not** next: more seeds of F/M, width/depth sweeps, gate initialisation variants,
new SSL objectives, cycle-penalty routing, or ensembles — all are on the frozen no-go list
and none of them addresses the finding that the interaction term is zero.

## 11. Reproduction

```bash
# CPU gates + replay + FLOP audit (no labels beyond Fit)
python tracks/structural-sufficiency/code/phase2a/phase2a_runner.py prep
# one arm, one seed (train -> predict -> freeze); seed 1 requires --authorize-seed1
python tracks/structural-sufficiency/code/phase2a/phase2a_runner.py train   --arm M --seed 0 --device cuda
python tracks/structural-sufficiency/code/phase2a/phase2a_runner.py predict --arm M --seed 0 --device cuda
# scoring (one-shot Monitor read) and replication
python tracks/structural-sufficiency/code/phase2a/phase2a_runner.py score --seeds 0
python tracks/structural-sufficiency/code/phase2a/phase2a_seed1_score.py
python tracks/structural-sufficiency/code/phase2a/phase2a_posthoc.py
pytest tracks/structural-sufficiency/code/phase2a/tests -q
```

All stages refuse to overwrite an existing artefact; predictions are frozen with a
prediction-file hash and an array hash before any label access.
