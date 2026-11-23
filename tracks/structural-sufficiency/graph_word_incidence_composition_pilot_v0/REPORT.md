# GWIC-v0 — graph word–incidence composition pilot — report

**Round:** `graph_word_incidence_composition_pilot_v0`  
**Protocol:** `gwic-v0-graph-word-incidence-composition-pilot` (frozen before training)  
**Status:** completed single-seed engineering screen  
**Seed:** 0 only  
**Task:** `ogbg-mollipo` whole-molecule lipophilicity regression  
**Primary metric:** Dev RMSE in original label units  
**Training commit:** `a7155937bac08c7733f6a2c1d806a395558f57be`  
**Analysis-code commit:** `60d8dd6212fb7ef26aa08641a1a108b38b6fa924`  
**D0 split:** `60ac13c04fb0dab484fc8e7cc28f2a85450794ed3ebeeb0a87d1b41c3532559e` (2856 fit / 504 Dev)  
**Route label:** **`INCONCLUSIVE_SINGLE_SEED`**  
**Substantive reading:** the complete R2 sentence trunk is already competitive; learned word selection adds no clear absolute gain, and real word-to-atom incidence binding shows no incremental value.

## 0. Answer in one paragraph

The complete radius-2 sentence trunk is the strongest part of this round: `SENTENCE_BASE` reaches Dev RMSE **0.5506**, better than the historical D0 coordinates of ACRA (0.5802), path-count ridge (0.6167) and PSC (0.6871), though still behind GINE (0.5266). Adding learned connected-word selection and word content (`WORD_GEOM_NULL`, 619,586 parameters) reaches **0.5432**, a +0.0074 RMSE point improvement over the trunk, but the paired bootstrap interval spans zero (`[-0.0274, +0.0427]`) and this arm also adds about 203k parameters. Adding the real word-to-original-atom incidence geometry (`WORD_REAL`) reaches **0.5500**, i.e. it is **0.0006 RMSE worse than the trunk** (95% bootstrap `[-0.0350, +0.0379]`) and **0.0068 worse than the matched null** (95% `[-0.0280, +0.0156]`). On the same `WORD_REAL` checkpoint, switching the three geometry inputs off changes predictions (mean |Δ| 0.0135, all 504 molecules) and slightly **improves** Dev RMSE (0.5496 vs 0.5500). Thus the incidence path is numerically live but not predictive. Under the frozen rule the route label is **`INCONCLUSIVE_SINGLE_SEED`**, not a promotion; the scientific conclusion is that this single-seed screen gives no evidence that word-to-connection binding is useful here.

## 1. Contract and execution

| item | result |
|---|---|
| three arms, seed 0 | `SENTENCE_BASE`, `WORD_GEOM_NULL`, `WORD_REAL`; all completed |
| D0 fit / Dev | 2856 / 504; disjoint, exactly cover the 3360 official Train rows |
| official Valid/Test labels | never loaded; Test rows never used in any tensor or metric; Valid rows carry NaN labels |
| training commit | `a7155937`; remote `git_dirty=false`, exit code 0 for all runs |
| engineering checks | `analysis/checks.json`: **15/15 PASS**, generated at implementation commit `8615013c` |
| unit tests | 12 passed |
| candidate support audit | **1,124,230** candidates; below the 20M cap, so `support_max=4` for every arm; no per-molecule truncation |
| wall / GPU budget | longest run 1251.5 s (0.348 h); 0.731 GPU-hours total; within 2 h / 4 GPU-h |
| route label | **`INCONCLUSIVE_SINGLE_SEED`** |

Data identity was re-verified before training: raw cache `fe002287…23c7c`, train-label archive prefix `a44d29de…`, D0 split file `675289f6…e845`, recomputed split hash `60ac13c0…559e`. The target scaler is fit-only (`mean=2.185815826311436`, `std=1.2139664740467941`, n=2856). All three runs share the same config, fit/Dev row hashes, recipe and target scaler. `WORD_GEOM_NULL` and `WORD_REAL` share the same parameter ledger (619,586) and the same explicit initial state.

## 2. What was computed

For every atom `v` of every molecule, the sentence is the complete radius-2 rooted ego graph

```
R2(v) = {t : dist(v,t) <= 2}
```

with every real induced bond. A candidate word is a distinct atom support `S` with `v in S`, `2 <= |S| <= 4`, and a connected induced subgraph. Every support is enumerated once; it keeps its real internal induced bonds and the real boundary bonds from `S` to `R2(v) \ S`. No atom-id top-K is used and atom ids are never features.

Each candidate is encoded by a shared MLP from the root atom representation, support atom sum/mean/std, internal bond sum/mean/std, a shared joint representation of every internal bond (`a+b`, `|a-b|`, `a*b`, bond attributes) aggregated by sum/mean, support size, internal bond count, cycle rank, root degree in the word, induced internal distance distribution, and boundary count/type fractions. The output is `w_(v,S) in R^64`.

A shared `ScoreMLP([w_(v,S), h_v])` scores all candidates of a root; a root-wise softmax gives `alpha`, and `z_v = sum_S alpha_(v,S) w_(v,S)`. No hard top-K is used. The sentence trunk `h_v in R^96` is a non-iterative LCRC-style encoder using S1 incident bonds, typed rooted S2 paths, complete R2 atom/bond summaries and fit-standardised scalars. It has no message passing, GINE, Transformer or pretraining.

The binding is

```
c_(v,t) = sum_{S: t in S} alpha_(v,S)
o_uv    = sum_t c_(u,t) c_(v,t)
r_uv    = c_(u,v),   r_vu = c_(v,u)
```

and one `Q` is evaluated once per real undirected bond on

```
[h_u+h_v, |h_u-h_v|, h_u*h_v, z_u+z_v, |z_u-z_v|, z_u*z_v,
 bond_emb, o_uv, r_uv+r_vu, |r_uv-r_vu|] -> q_uv in R^64.
```

`Q` is endpoint-swap invariant. The whole-molecule readout uses `h` sum/mean/std, `z` mean/std, `q` mean/std and 8 deterministic graph scalars through a nonlinear MLP.

## 3. Main results

All metrics are original-label units. `train_rmse_at_best` is the fit-set RMSE at the best-Dev checkpoint.

| arm | params | train RMSE@best | Dev RMSE | Dev MAE | best epoch | epochs run | wall s | GPU h |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SENTENCE_BASE | 416,177 | 0.2795 | **0.5506** | 0.4276 | 81 | 126 | 434.0 | 0.1030 |
| WORD_GEOM_NULL | 619,586 | 0.1751 | **0.5432** | 0.4199 | 96 | 141 | 1136.8 | 0.2981 |
| WORD_REAL | 619,586 | 0.1707 | **0.5500** | 0.4180 | 109 | 154 | 1251.5 | 0.3298 |

The word arms fit the 2856 fit molecules much more tightly (train RMSE 0.171–0.175) than the trunk (0.280), while their Dev RMSE is not better than the trivial spread among the three arms. The train/Dev gap is 0.271 for the trunk and 0.368–0.379 for the word arms: the extra capacity is mostly fitting fit-set idiosyncrasy.

## 4. Primary contrasts

Positive = `WORD_REAL` better.

```
delta_word    = RMSE(SENTENCE_BASE) - RMSE(WORD_REAL)
delta_binding = RMSE(WORD_GEOM_NULL) - RMSE(WORD_REAL)
```

| contrast | point | paired-molecule 95% bootstrap | reading |
|---|---:|---|---|
| `delta_word` | **+0.00058** | `[-0.03498, +0.03786]` | WORD_REAL is tied with the trunk, not better |
| `delta_binding` | **−0.00678** | `[-0.02796, +0.01564]` | null is point-better; no incidence gain |
| `delta_word` MAE | +0.00954 | `[-0.01831, +0.03872]` | no clear MAE gain |
| `delta_binding` MAE | +0.00185 | `[-0.01665, +0.02072]` | no clear MAE gain |

A useful secondary descriptive contrast:

| contrast | point | 95% bootstrap | reading |
|---|---:|---|---|
| `delta_content = RMSE(BASE) − RMSE(GEOM_NULL)` | **+0.00737** | `[-0.02743, +0.04270]` | word content is point-better but not robust; +203k parameters |

So the only positive word-related point estimate is the content-only arm, and it is small and not separated from zero. The real binding arm is not better than either the trunk or the null.

## 5. Mechanism check: the geometry is live but not useful

The three geometry inputs can be switched on one checkpoint without changing parameters. This is the cleanest mechanism test.

| checkpoint | geometry | Dev RMSE | Dev MAE | mean abs prediction change | max abs change | fraction changed >1e-5 |
|---|---|---:|---:|---:|---:|---:|
| WORD_REAL | real | 0.54998 | 0.41805 | — | — | — |
| WORD_REAL | zero | **0.54961** | **0.41761** | 0.01346 | 0.03537 | 1.000 |
| WORD_GEOM_NULL | zero (trained) | 0.54319 | 0.41989 | — | — | — |
| WORD_GEOM_NULL | real | **0.54312** | 0.41993 | 0.00132 | 0.00875 | 0.994 |

On the `WORD_REAL` checkpoint, zeroing the geometry **improves** Dev RMSE by 0.00037 and MAE by 0.00043; on the `WORD_GEOM_NULL` checkpoint, switching real geometry on improves RMSE by only 0.00008. The prediction changes are real (especially for the real-trained checkpoint), so the binding path is not disconnected or silently ignored, but its direction is not a stable predictive benefit. This is consistent with the main paired result: no evidence that word-to-connection incidence carries useful signal at this scale.

## 6. Word-selection diagnostics

Diagnostics are on the 504 Dev molecules with each arm's best checkpoint. `alpha` is a selection weight, not a causal contribution.

| diagnostic | WORD_GEOM_NULL | WORD_REAL |
|---|---:|---:|
| roots with candidates | 13,801 | 13,801 |
| alpha entropy mean / median | 2.314 / 2.302 | 2.174 / 2.301 |
| effective candidates mean / median | 11.82 / 9.998 | 10.55 / 9.986 |
| alpha constant fraction | 0.000 | 0.000 |
| alpha top weight mean | 0.156 | 0.213 |
| alpha per-root std mean | 0.0230 | 0.0466 |
| top candidate size 2 / 3 / 4 | 5,798 / 5,196 / 2,807 | 4,508 / 5,737 / 3,556 |
| all candidate size 2 / 3 / 4 | 30,046 / 63,426 / 78,528 | 30,046 / 63,426 / 78,528 |
| alpha-weighted size 2 / 3 / 4 | 0.217 / 0.391 / 0.393 | 0.244 / 0.391 / 0.365 |
| word effective rank (participation ratio) | 5.04 | 5.05 |
| q effective rank (participation ratio) | 2.59 | 2.93 |
| z effective rank (participation ratio) | 4.47 | 4.34 |
| sampled word-pair cosine mean | 0.472 | 0.500 |

There is **no alpha collapse**: constant-alpha fraction is exactly 0, mean effective candidates is about 10.5, and the top candidate sizes are not dominated by size 2 (only 24.4% of alpha mass in `WORD_REAL`; 36.5% goes to size-4 words). Word representations are not near-constant (participation ratio 5.05, zero near-constant channels, sampled cosine 0.50). The `q` representation has a lower effective rank (2.93) but still non-zero variance and zero near-constant channels; this is a watch item, not a degenerate output.

### High-weight word examples (WORD_REAL, Dev)

Atom feature index 0 is the OGB atomic-number category: index 5 = C, 6 = N, 7 = O, 16 = Cl. `internal_bond_types` lists `[bond_type, stereo, conjugated]`; support indices are molecule-local atom ids. These are examples for human inspection, not chemical discoveries.

| row | root local | support local | support atom types | internal bond types | boundary bonds | alpha |
|---:|---:|---|---|---|---:|---:|
| 3264 | 11 | [10, 11] | [C, O] | [[1, 0, 1]] | 2 | 0.998 |
| 2589 | 6 | [5, 6] | [C, O] | [[1, 0, 1]] | 2 | 0.998 |
| 3426 | 18 | [17, 18] | [C, O] | [[1, 0, 1]] | 2 | 0.998 |
| 53 | 21 | [20, 21, 22] | [C, O, O] | [[1, 0, 1], [0, 0, 1]] | 1 | 0.996 |
| 3091 | 23 | [22, 23] | [C, O] | [[1, 0, 1]] | 1 | 0.963 |

The model often concentrates on small C–O containing supports, but it also selects size-3/4 supports across the data; the alpha-weighted size distribution is 24.4% / 39.1% / 36.5% for sizes 2/3/4. This does not show that the word vocabulary is chemically meaningful; it only shows that the selector is not stuck on the smallest support.

## 7. Historical D0 coordinates

These are different recipes and are **not** a matched benchmark. Positive "C − ref" means `WORD_REAL` is worse.

| model | D0 RMSE | WORD_REAL − reference | reading |
|---|---:|---:|---|
| GINE_REF | 0.5266 | +0.0234 | C is worse than GINE |
| ACRA_REF | 0.5802 | −0.0302 | C is point-better than ACRA |
| PATH_COUNT_RIDGE | 0.6167 | −0.0667 | C is point-better than the ridge |
| PSC_POOL_BAG_PAIR | 0.6871 | −0.1371 | C is point-better than PSC |
| WORD_GEOM_NULL (this round) | 0.5432 | −0.0068 vs C | null is point-better than C |
| SENTENCE_BASE (this round) | 0.5506 | +0.0006 vs C | essentially tied |

The complete R2 trunk alone clears ACRA, the path-count ridge and PSC in point estimate. This is the main positive engineering result of the round. It is a single-seed, best-Dev-selected number and should not be read as a leaderboard claim.

## 8. Answers to the required questions

**1. What does the complete second-order local-subgraph trunk achieve?**  
`SENTENCE_BASE` reaches Dev RMSE **0.5506** / MAE **0.4276** with 416,177 parameters and a train RMSE of 0.2795 at its best epoch. In point estimate it beats the historical ACRA, path-count-ridge and PSC D0 coordinates, but is behind GINE. The trunk is a strong baseline for this round, not a weak control.

**2. Does learned connected-word selection add absolute predictive value?**  
The content-only word arm (`WORD_GEOM_NULL`) is point-better than the trunk by **0.0074 RMSE**, but its 95% paired interval is `[-0.0274, +0.0427]`, it adds 203k parameters, and the real word arm is only **0.0006** better than the trunk. There is no clear absolute predictive gain from word selection in this seed.

**3. Does real word-support/connection binding add extra value?**  
No. `WORD_REAL` is **0.0068 RMSE worse** than the matched content-only null (`WORD_GEOM_NULL`); the interval includes zero. On the same checkpoint, zeroing the geometry slightly improves Dev RMSE. The binding computation is live but does not provide a useful increment.

**4. Did word selection collapse, choose only the simplest structures, or become near-constant?**  
No. Alpha is not constant for any root, mean effective candidates is 10.55, and word/q/z effective ranks are 5.05 / 2.93 / 4.34. Selection mass is distributed across sizes 2/3/4 (24.4% / 39.1% / 36.5%), so it is not only the simplest size-2 words. The lower `q` rank is a watch item but not a collapse.

**5. Is this route worth continuing as a reusable graph-word dictionary?**  
Not on this evidence. The only positive word-content point estimate is small, not robust, and confounded with extra parameters; the matched binding contrast is negative. A larger dictionary / second seed / D1 / official Valid is not justified by this screen. Stop here.

## 9. Verdict and interpretation

Frozen rule applied:

1. `PROMISING_WORD_ROUTE` requires `delta_word >= 0.02`, `delta_binding >= 0.015`, `WORD_REAL Dev <= 0.62`, and non-degeneracy. `delta_word = +0.00058`, `delta_binding = −0.00678`; fails.
2. `WORD_CONTENT_ONLY` requires `delta_word >= 0.02`; fails.
3. `NO_COMPETITIVE_WORD_SIGNAL` requires `delta_word <= 0` or (`WORD_REAL Dev > 0.6167` and `delta_binding < 0.015`). `WORD_REAL` is 0.5500, not worse than the trunk in point estimate; fails literally.
4. Therefore **`INCONCLUSIVE_SINGLE_SEED`**.

The literal label is not a claim that the route is promising. The substantive conclusion is negative for the mechanism: the complete R2 trunk is competitive, word content gives only a small and uncertain point gain, and real incidence binding does not improve on the content-only null. The word selector is non-degenerate, so this is not a “the module did nothing” result; it is a result that the selected words and their positions do not carry enough extra whole-molecule signal here.

## 10. Limitations

* **Single seed (0).** No training-seed variance; the paired bootstrap describes molecule-level uncertainty only.
* **Dev is the selection set.** Best-Dev checkpoint selection makes all reported Dev numbers slightly optimistic; the comparison is controlled across arms but not an unbiased test estimate.
* **Parameter mismatch BASE vs word arms.** The word arms add about 203k parameters; the content-only gain versus BASE is therefore not a pure added-information contrast. The binding contrast (B vs C) is parameter-matched and is the clean mechanism test.
* **Historical coordinates are not matched recipes.** GINE/ACRA/path-count/PSC numbers come from other rounds and only serve as reference points.
* **Dev is small (504).** Bootstrap intervals are wide and the point estimates are not stable treatment effects.
* **No second seed, D1, Valid/Test, K-SVD, extra word sizes or architecture search**, by protocol.
* **Low-rank q.** The q representation has participation ratio 2.93; a different fusion or stronger geometry could behave differently, but this is a hypothesis, not an observed gain.
* **Alpha examples are descriptive.** High alpha is a selection weight, not a causal or chemical-fragment claim.

## 11. Provenance, budget and reproduction

**Budget.** Three runs used two A100-SXM4-40GB GPUs, one process per GPU; `WORD_REAL` and `WORD_GEOM_NULL` ran in parallel, then `SENTENCE_BASE` used a freed GPU. Longest run: 1251.5 s = 0.348 h. Sum of training GPU time: 0.731 GPU-hours. Overall wall from first launch to last completion: about 32 min. Within 2 h / 4 GPU-h.

**Provenance.** Training runs executed commit `a7155937` on host `a100-2`, driver `550.163.01`, torch `2.5.1+cu124`, exit code 0. Run IDs: `gwic-real-s0-20261123-105726-960061e4`, `gwic-null-s0-20261123-105726-2a884248`, `gwic-base-s0-20261123-112218-1a179ec4`. Per-run manifests are in `results/<ARM>_S0/manifest.json`. Re-loading each `state_best.pt` and re-running Dev reproduces the committed predictions within `1e-5` in raw label units. `analysis/checks.json` records 15/15 PASS, generated at `8615013c`.

**Main artefacts.**

```
PROTOCOL.md
REPORT.md
configs/gwic.json
code/gwic_{common,data,structure,tensor,model,train,checks,analyze,synthetic}.py
tests/
analysis/checks.json
results/summary.json
results/main_table.csv
results/paired_bootstrap.json
results/diagnostics.json
results/predictions_{SENTENCE_BASE,WORD_GEOM_NULL,WORD_REAL}_S0.npz
results/{SENTENCE_BASE,WORD_GEOM_NULL,WORD_REAL}_S0/
    manifest.json
    curves.csv
    state_best.pt
    state_final.pt
    predictions_best_fit.{csv,npz}
    predictions_best_dev.{csv,npz}
    train.log
```

**Reproduction commands.**

```bash
# local checks
uv run pytest -q tracks/structural-sufficiency/graph_word_incidence_composition_pilot_v0/tests
uv run python tracks/structural-sufficiency/graph_word_incidence_composition_pilot_v0/code/gwic_checks.py

# remote execution (committed revision only)
rr deploy res
rr run res gwic-real-s0 --pool res-gpu0 \
  --result tracks/structural-sufficiency/graph_word_incidence_composition_pilot_v0/results/WORD_REAL_S0 \
  -- python tracks/structural-sufficiency/graph_word_incidence_composition_pilot_v0/code/gwic_train.py \
     --arm WORD_REAL --seed 0 --device cuda --run-id WORD_REAL_S0 \
     --out tracks/structural-sufficiency/graph_word_incidence_composition_pilot_v0/results/WORD_REAL_S0
rr run res gwic-null-s0 --pool res-gpu1 ...
rr run res gwic-base-s0 --pool res-gpu0 ...

# after pulling the three run directories
uv run python tracks/structural-sufficiency/graph_word_incidence_composition_pilot_v0/code/gwic_analyze.py
```

## 12. Stop condition and non-claims

This round stops here: **no second seed, no D1, no official Valid/Test evaluation, no K-SVD, no additional word sizes, no architecture search**. The result is a single-seed engineering screen.

Non-claims: no causal interpretation of alpha; no claim that high-alpha supports are chemical functional groups; no claim that the R2 trunk is a new paradigm; no claim of a robust word or binding effect; no Test metric. The complete R2 trunk's point estimate over historical ACRA/ridge/PSC is reported as a reference coordinate, not a matched benchmark.

---

*Report generated from frozen outputs; numeric tables are reproducible from `results/summary.json`, `results/paired_bootstrap.json`, `results/diagnostics.json` and the committed run artefacts.*
