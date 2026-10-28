# REPORT — ZINC non-local architecture tree v1

**Decision: `STOP` (all three pre-registered hypotheses RED).** No new mechanism, pretraining task
or seed is promoted; no official-test read was purchased. One-sentence conclusion: *on the official
PyG ZINC 10k/1k protocol, a separator-conditioned bridge energy (SCE), an exact graph-equilibrium
field (GEF), and a joint 2-bond-span pretraining task (JSSP) all train normally and fit the train
set, but each lands `+0.0058` to `+0.0071` raw-y MAE **above** frozen LCRC-A seed 0 on the matched
valid-1000 (`0.0824–0.0837` vs `0.076582`), with no broad ordinary/unseen improvement; the new
channels are active and non-redundant, so this is a generalisation failure of the added mechanism,
not an inactive branch.*

Track: `tracks/structural-sufficiency/zinc_nonlocal_architecture_tree_v1`.
Regime: `res`, A100-SXM4-40GB, driver 550.163.01, torch 2.5.1+cu124 (same as the frozen A/R/N
references). Official test was never constructed, loaded or scored this round.

---

## 1. Protocol and provenance

* **Data.** Official PyG ZINC `subset=True`: train 10,000 (raw y), valid 1,000 (the only dev read),
  test 1,000 never loaded. Frozen feature cache and train-only normalisation reused from
  `zinc_local_relation_composition_v1` (`norm fingerprint 8c3e8813…`).
* **Reference.** Frozen `LCRC-A` seed 0 (`runs/lcrcA_s0`, commit `42c01a3c…`): 237,519 parameters,
  best valid **0.076582 @476**, final 0.077366, T100 0.077359, historical test 0.066782 (already
  exposed before this round; not re-read).
* **Recipe.** AdamW, batch 128, 500-step warmup, per-step cosine 1e-3 -> 1e-5, wd 1e-5, grad clip 5,
  raw-y L1, 600 epochs = 47,400 updates, full valid every epoch, strict argmin checkpoint.
* **Runs.** Two concurrent formal jobs maximum. SCE and GEF were launched together at 00:26 on
  `res-gpu0`/`res-gpu1`; JSSP pretraining and downstream ran later on `res-gpu0`. All jobs were
  launched with `rr run` and carry a run id, commit, host, pool, driver, torch and exit code.
* **Commits.** Initial code `185a08e5…`; JSSP/base-downstream path `e94daa5c…`. No uncommitted code
  was run. No parameter, width, rank, layer, λ, mask-ratio or LR search was performed.
* **Selection.** Valid only; the best checkpoint is the strict argmin of full valid MAE. Official
  test was not read before, during or after selection.

### Frozen hypothesis-to-arm mapping

| hypothesis | mechanism | arm | parameters |
|---|---|---|---:|
| H1 SCE | `y = Σ_v f(h_v) + Σ_bridge I_e/√(1+n_bridge) + ρ(G)`, endpoint-exchange-invariant side features | SCE-Real seed 0 | 260,336 (head 22,817) |
| H2 GEF | one exact solve `K=(λI+L_w)^{-1}`, `c_v` into `g(h_v,c_v)−g(h_v,0)`, λ=0.1 fixed label-free | GEF-Real seed 0 | 250,524 (head 13,005) |
| H3 JSSP | masked pretraining where each example hides contiguous 2-bond spans (atoms + physical bonds) | JSSP→A seed 0 | 237,519 |

---

## 2. Phase-0 gates (all passed before GPU)

* Bridge enumeration vs `networkx` on 300 molecules: 0 mismatch; every valid molecule has ≥1 bridge
  (mean 9.77, max 18), including all 251 radius-2-unseen molecules.
* Trunk initialisation is **bit-identical** to historical A_s0: `11c698ae…` per tensor; parameter
  counts 260,336 / 250,524 / 237,519 (<500k).
* SCE is exactly 0 on a bridge-free 6-ring; SCE/GEF are invariant to renumbering and bond direction
  (max drift 2.4e-7), batch split/merge drift 1.2e-7, save/reload bit-identical.
* GEF vs independent float64 brute-force solve: `K` error 8.5e-8, `c` error 7.2e-8, inverse residual
  2e-7; `K` off-diagonals ≥0.33; uniform source gives exactly `c=0`.
* Structural witness: the same local-state multiset under a different correspondence changes both
  SCE and GEF outputs (the operators are not functions of pooled local statistics alone).
* λ=0.1 selected **without labels** by "largest λ with mean reach ≥3 graph steps, p90 ≥4,
  cond≤1e4" (mean reach 3.02, p90 5.79, 49% of mass beyond 3 hops, max cond 62.5).
* At initialisation the SCE/GEF terms are not linearly redundant with the existing pooled features
  (R² 0.738 / 0.466; permutation controls −0.31 / −0.22).

JSSP gates: span enumeration vs brute force 100/100; span closure; hidden-truth swap leaves the
visible masked object unchanged (100 atoms + 100 bonds); coverage; 6-step loss decreases; saved
encoder loads into the downstream A trunk without error. The JSSP encoder init hash equals the
historical SSL init exactly (`91af980a…`).

---

## 3. Main result (matched valid-1000)

All rows are single models, no soup/ensemble/EMA, evaluated at each run's best checkpoint against
the same A_s0 predictions. Δ = run − A_s0; negative is better.

| run | best valid | Δ vs A | Δ drop-0172 | Δ drop-fixed-10 | ordinary Δ | r2-unseen ord Δ | seen ord Δ | tail Δ | wins | best ep | final | T100 | GPU·h |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SCE-Real | 0.082364 | **+0.005782** | +0.002115 | +0.003022 | +0.000459 | +0.002798 | −0.000282 | +0.082353 | 0.512 | 434 | 0.083025 | 0.083070 | 0.5947 |
| GEF-Real | 0.082361 | **+0.005780** | +0.004057 | +0.005084 | +0.004359 | +0.011721 | +0.002025 | +0.026218 | 0.499 | 421 | 0.082809 | 0.082828 | 0.6738 |
| JSSP→A | 0.083658 | **+0.007077** | +0.005760 | +0.005285 | +0.001970 | +0.004565 | +0.001148 | +0.080536 | 0.504 | 452 | 0.084057 | 0.084023 | 0.3562 |

Reference A_s0 group MAEs: all 0.076582, ordinary (n=935) 0.063545, tail (n=65) 0.264113,
r2-unseen ordinary (n=225) 0.130915, seen ordinary (n=710) 0.042195.

Paired row-bootstrap CI95 of the full-MAE difference (10,000 resamples):

* SCE: `[−0.001268, +0.015399]` (crosses 0; point estimate worse, median row −0.000835)
* GEF: `[−0.000114, +0.012386]` (essentially 0 to worse; median +0.000116)
* JSSP: `[+0.000403, +0.014810]` (excludes 0 on the worse side)

### Curve shape (matched epochs)

| epoch | A | SCE | GEF | JSSP |
|---:|---:|---:|---:|---:|
| 150 | 0.111352 | 0.106110 | 0.106910 | 0.107360 |
| 200 | 0.090508 | 0.087420 | 0.092980 | 0.099950 |
| 250 | 0.085453 | 0.094530 | 0.089950 | 0.090170 |
| 300 | 0.080345 | 0.091100 | 0.085580 | 0.088800 |
| 400 | 0.078219 | 0.084780 | 0.082640 | 0.085110 |
| 500 | 0.077244 | 0.083330 | 0.083150 | 0.083970 |
| 600 | 0.077366 | 0.083025 | 0.082810 | 0.084060 |

All three variants start at least as well as A at epoch 150, but A's 250–450 drop is not reproduced.
The variants plateau around 0.0824–0.0840 while A continues to 0.0766. This is not an early-training
failure; it is a late-generalisation gap.

### Group reading

* **SCE** is the only arm with any negative group delta: seen-ordinary `−0.000282` (0.7% relative),
  but ordinary is `+0.000459`, r2-unseen ordinary `+0.002798`, ring-rich `+0.003263`, tail
  `+0.082353`. The full-MAE regression is amplified by the 65 tail molecules (27% of total AE), but
  removing valid:0172 still leaves `+0.002115`, and removing A's fixed worst-10 leaves `+0.003022`.
  The sign does **not** depend on one row.
* **GEF** is worse in every group, including seen-ordinary and unseen-ordinary; this is not a
  tail-only effect.
* **JSSP** is worse in every group; its ordinary/unseen deltas are positive and its tail is as bad
  as SCE's. It is the worst full-MAE arm.

No arm meets the pre-registered GREEN (≥~0.003–0.005 robust improvement) or AMBER (broad
ordinary/unseen improvement, or a sign carried by one tail row) condition. All are RED.

---

## 4. Mechanism diagnostics — the branches were active and non-redundant

Label-free probes on 2,048 train molecules, using the best checkpoints (no labels read):

| arm | term mean | term std | R² of term from existing pooled features | permutation control | reading |
|---|---:|---:|---:|---:|---|
| SCE | +0.0552 | 0.0613 | **0.290** | −0.142 | active, not a re-parameterisation of global pooling |
| GEF | −0.0295 | 0.1074 | **0.442** | −1.560 | active, large variation, not a global-statistics copy |

For SCE the initial R² was 0.738; training moved it *away* from the existing features (0.290), so the
head did not collapse to a redundant bias. All runs reach the same terminal train fit (final MAE
≈0.0010–0.0014; A 0.00131) and their best checkpoints are late (421/434/452), so the failure is not
underfitting. At the best-valid checkpoint the new arms actually have *higher* train MAE
(0.0094 SCE / 0.0094 GEF / 0.0066 JSSP) than A (0.00527), so the selected solutions are not
better-fitting ones — they generalise worse. The failure is therefore **not** "the new channel did
nothing": the channel produced a substantial, structurally new per-molecule signal that did not
generalise to valid.

JSSP's pretraining was also active: 7,560 updates, 96 visits, atom CE 1.12→0.239 and bond CE
0.516→0.064, atom accuracy 0.717→0.917, bond accuracy 0.784→0.976. Its initial encoder hash is
bit-identical to the historical SSL (`91af980a…`), so the downstream difference is attributable to
the pretraining task, not initialisation. The frozen JSSP budget used 14.2% atoms and 8.9% bonds;
the old independent mask used 15.0%/14.9%. The lower bond coverage is a **recorded confound**: a
2-bond span has 3 atoms : 2 bonds, while the atom budget was held at ~15%. The result closes the
frozen JSSP recipe, not a clean causal claim that "jointness alone" is harmful.

### Why the common plateau matters

SCE and GEF are mathematically unrelated operators, yet their best valid MAEs differ by 3e-6
(`0.082364` vs `0.082361`) and JSSP lands only 0.0013 away. Three different interventions — an
extra bridge sum, an extra global solve, and a changed pretraining task — converge to the same
0.082–0.084 plateau, while A reaches 0.0766. The most parsimonious reading is a **shared
capacity/optimisation cost around a sharp late-training optimum**: every intervention adds
flexibility or shifts the optimisation trajectory and loses the specific 250–450 epoch
generalisation path of the local A trunk. The best-valid checkpoints are not over-fit relative to A;
they are different, worse-generalising solutions. This is consistent with the earlier LCRC evidence
that A's advantage is not a missing global statistic.

---

## 5. Budget, wall clock, discipline

| item | GPU·h |
|---|---:|
| SCE-Real-s0 | 0.5947 |
| GEF-Real-s0 | 0.6738 |
| JSSP pretraining s0 | 0.4518 |
| JSSP downstream A s0 | 0.3562 |
| 2-epoch base timing | 0.0020 |
| **total** | **2.0785** |

Soft cap 14, hard cap 16; **11.92 GPU·h unspent**. Wall: first formal launch 00:26, last completion
02:19 (≈1 h 54 min of training); whole round ≈3.5 h including implementation/gates, inside the 7–8 h
target. At most two formal jobs ran concurrently. The reserved ~2 GPU·h was not needed.

Discipline checks:

* Official test: never constructed, loaded, scored or inferred. `official_test_loaded=false` in every
  artifact; no test predictions exist in this track.
* No S2 injection, six-category relation means, graph-level quadratic contraction, rare-environment
  weighting or ring-penalty correction; no ESA+/GRIT reproduction; no external labels.
* No λ/width/rank/layer/LR/mask-ratio sweep; λ and the JSSP span budget were fixed before training.
* No `rr` bypass; every formal run has a run id and regime record. No result was selected on valid
  more than once per arm; the best checkpoint is the strict argmin of the full valid curve.

---

## 6. Answers to the seven questions

**Q1. Did any new computational mechanism beat LCRC-A on official valid?**
No. The best new arms are SCE/GEF at `0.082364`/`0.082361`, `+0.005782`/`+0.005780` worse than
A_s0 `0.076582`; JSSP is `+0.007077` worse. No arm is within the 0.003 purchase band, and none has a
negative point estimate.

**Q2. Is the change broad across molecules/environments, or carried by a tail row?**
Neither broad nor a single-row artifact. SCE's only negative group is seen-ordinary (−0.000282);
its ordinary mean, r2-unseen ordinary, ring-rich and tail are all worse. Dropping valid:0172 still
leaves +0.002115; dropping A's fixed worst-10 leaves +0.003022. GEF and JSSP are worse in every
group. Win rates are 0.512/0.499/0.504 — coin flips, not broad improvement.

**Q3. Were the mechanisms actually used, or did they collapse?**
They were used and non-redundant: SCE term std 0.061 with R² 0.29 from existing pooled features;
GEF term std 0.107 with R² 0.44; both fit train to ≈0.001 and their best epochs are late. JSSP
changed the encoder through 7,560 updates and loaded into the downstream trunk. There is no
collapsed-branch excuse for the negative result.

**Q4. What do the controls/confounds say?**
No mechanism control was purchased, because no arm showed a performance sign; buying SCE-Marginal,
GEF-Fixed or a JSSP bond-budget control would not rescue a +0.0058 deficit under the frozen rules.
Known confounds: (i) SCE/GEF add late capacity, so "information" and "capacity/optimisation cost"
are not separated; the identical plateau across two unrelated operators suggests the cost is shared.
(ii) JSSP changed both span structure and bond coverage (8.9% vs 14.9%); the negative result is for
the frozen recipe, not a pure jointness effect. (iii) The A reference is frozen, not re-trained in
this round, but its initialisation (`11c698ae…`), SSL hash (`006fab…`), recipe, data order and
parameter count were verified; the base path is bit-identical at step 0.

**Q5. Was the budget/test/discipline contract respected?**
Yes: 2.08 GPU·h of 14 soft / 16 hard, ≤2 concurrent formal runs, ~3.5 h wall, 11.92 GPU·h unspent,
test never read, no tuning or new revisions after seeing results, all runs via `rr` with provenance.

**Q6. Should we buy seed 1, a mechanism control, or an official-test read?**
No. A seed-1 replication cannot turn a +0.0058 point estimate into a promotion; the pre-registered
rule requires a robust improvement first. A control would only explain a mechanism that has no
performance sign, and an official-test read is not earned (and would be descriptive, not blind).
The correct action is to keep the negative result and stop spending.

**Q7. What is the next decision?**
`STOP` — no PROMOTE, no REFINE_ONCE, no PIVOT inside this hypothesis family. The three pre-registered
routes are closed. A future round would need a *different* mechanism class with a pre-registered
reason to expect an ordinary/unseen gain, not a re-tune of SCE/GEF/JSSP. See `NEXT_DECISION.md`.

---

## 7. What this implies for the <0.060 test target

A_s0's historical test is `0.066782`; the target is `0.060`. This round's interventions move valid
in the wrong direction by `0.0058–0.0071`, i.e. roughly the size of the gap that would be needed.
The evidence says the missing ingredient is **not** (a) bridge-side composition, (b) a global
equilibrium context, or (c) a harder joint-span SSL objective. Each either adds non-generalising
variation or shifts the pretrained representation away from what the 600-epoch local recipe exploits.
Reaching the target would require a mechanism that improves the ordinary and radius-2-unseen groups
*simultaneously* without paying the late-training capacity cost — none of the tested mechanisms does.
No test claim is made for any new arm; no test was read.

---

## 8. Artifacts

| artifact | path |
|---|---|
| frozen plan / decision log | `PLAN.md`, `DECISION_TREE.md` |
| SCE/GEF/JSSP code | `code/nl_{common,bridge,models,data,train,prep,analysis,probe,jssp,pretrain,jssp_checks}.py` |
| CPU gates | `analysis/cpu_checks.json`, `analysis/jssp_cpu_checks.json` |
| λ / bridge prep | `analysis/nl_prep.json` |
| matched table | `analysis/final_table.{json,md}` (also `stage1.*` for the first two arms) |
| trained-term probes | `analysis/probe_trained_{sce,gef}.json` |
| JSSP pretraining record | `analysis/jssp_pretrain_s0.json` |
| base-path timing record | `analysis/base_timing_s0.json` |
| runs (predictions/checkpoints, gitignored) | `runs/sce-real_s0`, `runs/gef-real_s0`, `runs/jssp_base-real_s0` |
| frozen reference | `zinc_local_relation_composition_v1/runs/lcrcA_s0` |

Reproduction: `uv run python code/nl_train.py --head {sce,gef,base} --mode real --seed 0 --device cuda`
for 600 epochs (SCE/GEF use λ=0.1; base uses `--ssl <JSSP checkpoint>`); analysis via
`uv run python code/nl_analysis.py --runs …`. All formal jobs were launched through `rr` from commit
`e94daa5c…` on `res`, pools `res-gpu0`/`res-gpu1`.
