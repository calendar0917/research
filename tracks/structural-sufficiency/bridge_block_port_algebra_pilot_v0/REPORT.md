# BBPA-v0 — bridge-block port algebra pilot — report

**Round:** `bridge_block_port_algebra_pilot_v0`  
**Protocol:** `bbpa-v0-bridge-block-port-algebra-pilot` (frozen before formal training)  
**Status:** completed single-seed engineering screen  
**Seed:** 0 only  
**Task:** `ogbg-mollipo` whole-molecule lipophilicity regression  
**Primary metric:** Dev RMSE in original label units  
**Training commit:** `f607d9b534dedd9302ce700f3c27fa9214147f36`  
**D0 split:** `60ac13c04fb0dab484fc8e7cc28f2a85450794ed3ebeeb0a87d1b41c3532559e` (2856 fit / 504 Dev)  
**Route label:** **`CAPACITY_WITHOUT_COMPETITIVENESS`**

## 0. Answer in one paragraph

Recursive tree composition is **functionally real**: with one fixed checkpoint, TREE and LOCAL modes are exactly identical on block trees of diameter ≤ 2 and differ on every deeper molecule, with large state and interface differences. It also has **some predictive capacity**: TREE_PORT reaches Dev RMSE **0.6565**, versus **0.6727** for LOCAL_PORT and **0.7831** for BLOCK_BAG, so the point estimate of the primary contrast is `Δ_comp = 0.0162` in favour of recursion. That increment is **not robust in this single seed** (paired bootstrap 95% CI `[-0.0314, +0.0641]`), and TREE remains above the pre-registered competitive threshold of 0.64. The gain is concentrated in molecules with at least six bridges / block-tree diameter at least six; on small trees TREE is worse than LOCAL. Under the frozen rule this is **`CAPACITY_WITHOUT_COMPETITIVENESS`**, not a promotion signal: recursion can help when there is depth, but this implementation is not competitive with ACRA/GINE or the path-count reference.

## 1. Contract and execution

| item | result |
|---|---|
| three arms, seed 0 | `BLOCK_BAG`, `LOCAL_PORT`, `TREE_PORT`; all completed |
| identical encoders / parameters / recipe | yes; 417,857 parameters, identical initialisation hashes |
| D0 fit / Dev | 2856 / 504; disjoint, exactly cover the 3360 official Train rows |
| official Valid/Test labels | never loaded; Valid rows have zero finite label rows in the audit, Test never touched |
| training commit | `f607d9b5`; remote `git_dirty=false`, exit code 0 for all runs |
| engineering checks | `analysis/checks.json`: **17/17 PASS** (generated at implementation commit `1797c0f2`, frozen in the training commit) |
| unit tests | 16 passed |
| wall / GPU budget | longest run 1277.6 s (0.355 h); 0.589 GPU-hours total; within 2 h / 4 GPU-h |
| route label | **`CAPACITY_WITHOUT_COMPETITIVENESS`** |

Data identity was re-verified before training: raw cache `fe002287…23c7c`, train-label archive `a44d29de…1114c2`, D0 split file `675289f6…e845`, recomputed split hash `60ac13c0…559e`. The target scaler is fit-only (`mean=2.185815826311436`, `std=1.2139664740467941`, n=2856). All three runs share the same config hash, code hash, fit/dev row hashes, recipe, scaler, parameter ledger and shared-initialisation hashes.

## 2. What was computed

Each molecule is decomposed at its real bridge bonds:

1. all bridge bonds are found on the raw typed graph;
2. deleting bridges leaves **structural blocks** (ring systems, single atoms, acyclic fragments);
3. contracting blocks gives a **block tree or forest**;
4. every incidence gets a real **port** at the bridge endpoint atom, with distance-bucketed internal atom/bond summaries (buckets 0,1,2,3,4,5+).

The shared algebra is

```
m_(k→i|j) = M(s_(k→i), bond_ki, port_features_i(k), dist_bucket)
s_(i→j)   = LN(u_(i|j) + U([u_(i|j), A_(i|j)]))
q_e       = Q([s_(i→j)+s_(j→i), |s_(i→j)−s_(j→i)|, s_(i→j)*s_(j→i), bond_e])
```

with one shared `M`, `U`, `Q` over all blocks, directions and depths. `BLOCK_BAG` uses only block-global summaries; `LOCAL_PORT` replaces the recursive neighbour state with the neighbour's local port state `u_(k|i)`; `TREE_PORT` uses the full bidirectional tree DP. The graph readout is a shared MLP over `q` mean/std, block-summary mean/std and public whole-graph summaries; it is not a strict per-block scalar sum.

## 3. Main results

| model | seed | params | train RMSE@best | Dev RMSE | Dev MAE | best epoch | epochs run | wall s | GPU h |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BLOCK_BAG | 0 | 417857 | 0.6863 | 0.7831 | 0.6148 | 54 | 89 | 154.6 | 0.0410 |
| LOCAL_PORT | 0 | 417857 | 0.1982 | 0.6727 | 0.5019 | 63 | 98 | 709.0 | 0.1948 |
| TREE_PORT | 0 | 417857 | 0.0658 | **0.6565** | **0.4933** | 145 | 180 | 1277.6 | 0.3528 |

All metrics are in original label units. TREE ran the full 180-epoch horizon; LOCAL and BLOCK_BAG early-stopped after their best epochs (patience 35, minimum 80), with the same rule for all arms. The best-Dev checkpoints are committed; final states are also kept.

**Primary contrast.** `Δ_comp = RMSE(LOCAL_PORT) − RMSE(TREE_PORT) = +0.0162`, i.e. TREE is better by 0.0162 RMSE in point estimate. Paired molecule bootstrap (5000 resamples): **95% CI `[-0.0314, +0.0641]`**. The MAE contrast is `+0.00865`, 95% CI `[-0.02829, +0.04609]`. The interval includes zero, so this is not a confirmed gain.

**Other contrasts.** The interface/local representation is a clearer improvement over block summaries:

| contrast | point | 95% CI | interpretation |
|---|---:|---|---|
| TREE − BLOCK_BAG | +0.1265 | `[+0.0642, +0.1882]` | TREE better than block bag |
| LOCAL − BLOCK_BAG | +0.1103 | `[+0.0507, +0.1687]` | LOCAL better than block bag |
| LOCAL − TREE | +0.0162 | `[-0.0314, +0.0641]` | TREE point-better, not robust |

So **using ports/interfaces helps robustly**; the additional **recursive depth** is a smaller and uncertain increment in this seed.

**Route rule applied.**

| criterion | value | result |
|---|---|---|
| `Δ_comp ≥ 0.025` | 0.0162 | no |
| `TREE Dev RMSE ≤ 0.64` | 0.6565 | no |
| `TREE MAE ≤ LOCAL MAE + 0.01` | 0.4933 ≤ 0.5119 | yes |
| TREE states non-constant | mean per-dim std 0.891, near-constant fraction 0.000, rank 12.98 | yes |
| `Δ_comp ≤ 0.01` | 0.0162 | no |
| `TREE Dev RMSE > 0.64` | 0.6565 | yes |

Two of the four “promising” conditions fail (`Δ_comp` and the 0.64 RMSE bar); the MAE and non-constant-state conditions hold. The “no gain” condition also fails, while the “capacity but weak” condition holds, so the frozen rule returns **`CAPACITY_WITHOUT_COMPETITIVENESS`**. No threshold was moved after seeing the results.

## 4. Functional verification: the computation is not a renamed local model

This was checked in two ways.

**Discrete structure.** All 3360 D0 molecules passed decomposition invariants: every atom and non-bridge bond belongs to exactly one block; every bridge bond belongs to exactly one interface; expanding blocks and bridges reconstructs each atom and bond once; the two sides of each bridge are complementary; `#bridges = #blocks − #components`; and the DP schedule dependencies are topological. Synthetic cases cover a single block, a single bridge, a four-block chain, a four-block star, a two-component forest, and ortho/meta/para disubstituted benzene with port distances 1/2/3. Predictions are invariant to atom renumbering, bond order, block order, neighbour order and DP root choice, and batch-vs-single differences are below `1e-5`.

**Same-checkpoint mode comparison.** To isolate algebra from independently learned weights, the TREE checkpoint was evaluated twice on the same Dev batches: once in `TREE_PORT` mode and once in `LOCAL_PORT` mode. On the 26 Dev molecules with block-tree diameter ≤ 2, the maximum state difference was **exactly 0.0** and the maximum `q` difference was **0.0**. On all 478 molecules with diameter ≥ 3, the modes differ (fraction 1.0). Across deep molecules the mean per-molecule maximum state L2 difference is **12.08**; `q` differs for **98.3%** of bridges overall. This is the direct functional evidence that TREE carries subtree information beyond the local port state.

**State diagnostics (TREE checkpoint, Dev).**

| diagnostic | value |
|---|---|
| TREE directed states / bridges | 10,940 / 5,470 |
| per-dimension state std, mean / min / max | 0.891 / 0.490 / 1.652 |
| near-constant dims (`std < 1e-4`) | 0.0% |
| TREE state effective rank (participation ratio / entropy rank / top-1 share) | 12.98 / 21.52 / 0.175 |
| LOCAL state effective rank (participation ratio) | 9.40 |
| `q` effective rank (participation ratio) | 14.53 |
| `u_global` effective rank (participation ratio / top-1 share) | 1.72 / 0.726 |

The TREE states are not collapsed: no near-constant channel and a substantially higher effective rank than the local states. The low-rank `u_global` block summary is a watch item, but it is not the only readout input; `q` is higher-rank and the deep-subtree test shows it is used.

One caution: comparing states from the **independently trained** TREE and LOCAL checkpoints gives a difference on every molecule (mean 10.70, max 16.79). That is representation divergence caused by different learned weights, not a pure mode-semantics test, and is not used as coverage evidence.

## 5. Where the gain comes from: a depth-dependent crossover

The aggregate contrast hides a clear heterogeneity. Dev RMSE by bridge count and block-tree diameter (full table in `analysis/tables.md`):

| stratifier | bin | n | BLOCK_BAG | LOCAL_PORT | TREE_PORT |
|---|---|---:|---:|---:|---:|
| bridges | 1 | 3 | 1.0214 | 1.2077 | 1.3447 |
| bridges | 2 | 10 | 0.6295 | 0.4411 | 0.6959 |
| bridges | 3–5 | 64 | 0.5501 | 0.7423 | 0.9129 |
| bridges | 6–10 | 186 | 0.7700 | 0.6684 | 0.6106 |
| bridges | 11+ | 241 | 0.8456 | 0.6551 | 0.5914 |
| diameter | 0–2 | 26 | 0.6660 | 0.6796 | 0.7965 |
| diameter | 3–5 | 124 | 0.6613 | 0.7372 | 0.7771 |
| diameter | 6–8 | 193 | 0.8694 | 0.6567 | 0.6089 |
| diameter | 9–11 | 108 | 0.7282 | 0.6528 | 0.6179 |
| diameter | 12+ | 53 | 0.8700 | 0.6055 | 0.4965 |

Descriptive subgroup bootstraps (positive = TREE better; exploratory, no multiplicity correction):

| subgroup | n | Δ RMSE (LOCAL − TREE) | 95% CI |
|---|---:|---:|---|
| diameter ≤ 2 | 26 | −0.1169 | `[-0.3554, +0.1032]` |
| diameter ≥ 6 | 354 | +0.0519 | `[+0.0012, +0.1014]` |
| bridges ≤ 5 | 77 | −0.1740 | `[-0.3148, −0.0307]` |
| bridges ≥ 6 | 427 | +0.0611 | `[+0.0139, +0.1087]` |
| bridges ≥ 11 | 241 | +0.0637 | `[+0.0028, +0.1285]` |

The overall +0.0162 is a **cancellation**: TREE is better on deep/large molecules and worse on small ones. The crossover is large compared with the average. On this seed, recursion appears to have usable capacity only when the block tree is deep enough; on shallow trees the extra recursive computation adds variance without useful signal; the parameter budget is identical across arms. This is a post-hoc descriptive pattern, not a pre-registered significance claim. It also means the aggregate `Δ_comp` is not a stable “average treatment effect” for this architecture.

Structural coverage is not the explanation for the weak average: 99.82% of D0 molecules contain at least one bridge, 94.55% have block-tree diameter ≥ 3, and 94.8% of Dev molecules are in the deep regime. The screen is structurally well-covered; the problem is predictive value, not lack of bridge blocks.

## 6. Competitiveness against the historical D0 coordinates

| model | D0 RMSE | historical − TREE | reading |
|---|---:|---:|---|
| GINE_REF | 0.5266 | −0.1299 | TREE worse by 0.1299 |
| ACRA_REF | 0.5802 | −0.0763 | TREE worse by 0.0763 |
| PATH_COUNT_RIDGE | 0.6167 | −0.0398 | TREE worse by 0.0398 |
| PSC_POOL_BAG_PAIR | 0.6871 | +0.0306 | TREE better by 0.0306 |
| **TREE_PORT** | **0.6565** | — | this round |
| LOCAL_PORT | 0.6727 | — | this round |

TREE clears the older PSC block/pool coordinate by 0.0306, but it is 0.0398 worse than the path-count ridge, 0.0763 worse than ACRA and 0.1299 worse than GINE. The historical numbers come from different recipes and are not a controlled comparison; they are reference coordinates, not a leaderboard. The pre-registered “competitive” threshold was 0.64, and TREE misses it by 0.0165. The honest summary is: **the recursion buys a little over the local-port arm and over the prior PSC bag, but not enough to be competitive**.

## 7. Error structure and the overfitting signature

The train/Dev gap grows sharply with the interface and recursion:

| model | train RMSE@best | Dev RMSE | gap |
|---|---:|---:|---:|
| BLOCK_BAG | 0.6863 | 0.7831 | 0.0967 |
| LOCAL_PORT | 0.1982 | 0.6727 | 0.4745 |
| TREE_PORT | 0.0658 | 0.6565 | 0.5907 |

TREE has the largest capacity to fit the 2856 fit molecules and also the largest gap. The most likely explanation for the shallow-tree losses is overfitting/noise amplification: the recursive operator has more opportunity to fit idiosyncratic substructures, while small trees provide little depth signal. This is a hypothesis; no regularization search was run, because the protocol forbids it.

Worst TREE errors (full list in `results/worst10_tree.csv`):

| row_id | bridges | diameter | label | TREE pred | TREE abs err | LOCAL abs err |
|---|---:|---:|---:|---:|---:|---:|
| 3144 | 13 | 7 | 4.30 | 1.72 | 2.58 | 2.20 |
| 2241 | 4 | 4 | 4.09 | 1.77 | 2.32 | 1.24 |
| 3739 | 13 | 9 | 3.88 | 1.62 | 2.26 | 1.64 |
| 712 | 5 | 4 | −1.40 | 0.70 | 2.10 | 1.20 |
| 3171 | 1 | 1 | 0.75 | 2.74 | 1.99 | 0.72 |

TREE systematically under-predicts several high-lipophilicity, large molecules, and the single-bridge row 3171 is a clear small-tree failure (TREE error 1.99 versus LOCAL 0.72). This is consistent with the depth-dependent story, not with a uniform improvement.

## 8. Verdict

**`CAPACITY_WITHOUT_COMPETITIVENESS`.**

What is supported:

* the decomposition and the shared M/U/Q algebra are implemented and actually used;
* ports/interfaces give a clear improvement over block summaries (LOCAL and TREE both beat BLOCK_BAG with CIs excluding zero);
* recursion has a positive **point estimate** over LOCAL and is better on deep/large molecules;
* TREE states are non-degenerate and high-rank.

What is not supported:

* a robust overall composition gain (CI includes zero);
* competitiveness with ACRA/GINE/path-count (TREE is worse by 0.04–0.13 RMSE);
* a uniform gain across molecule sizes (TREE is worse on ≤5-bridge molecules);
* any claim from a second seed or official Test.

The result is best read as an **engineering screen that found a real but conditional mechanism**: port-aware blocks are useful, and recursive composition can add value when the block tree is deep, but the current single-seed model overfits small structures and does not reach the competitive range. This is exactly the distinction the protocol asked for: functional verification is positive; predictive gain is conditional and not robust; the scientific claim is exploratory only.

## 9. Limitations

* **Single seed (0).** No seed variance; the paired bootstrap describes molecule-level uncertainty, not training-seed uncertainty.
* **Dev is the selection set.** Best-Dev checkpoint selection makes the reported Dev RMSE slightly optimistic for all arms; the comparison is still controlled, but the absolute numbers are not unbiased test estimates.
* **Small Dev (504).** CIs are wide; the primary CI spans zero.
* **Historical coordinates are not matched recipes.** GINE/ACRA/path-count/PSC numbers come from other rounds and are not a like-for-like benchmark.
* **Subgroup analyses are post hoc.** They are descriptive and uncorrected; they explain the aggregate but do not establish a subgroup effect.
* **No real isomer-controlled test.** Ortho/meta/para port distances are verified synthetically; a real same-composition/different-connection generalization test was not run.
* **No GNN/attention/pretraining/hyperparameter search**, by protocol. The 417k-parameter MLP algebra may simply be capacity-limited or under-regularised for 2856 molecules.
* **`u_global` is low-rank** (participation ratio 1.72); this is a possible bottleneck, but it was not tuned.

## 10. Provenance, budget and reproduction

**Budget.** All three runs used two A100-SXM4-40GB GPUs, one process per GPU. Longest single run: 1277.6 s = 0.355 h. Sum of training GPU time: 0.589 GPU-hours. Both are well inside the 2 h wall / 4 GPU-h budget. The three runs overlapped; BLOCK_BAG used the GPU freed by LOCAL_PORT.

**Provenance.** All runs executed commit `f607d9b5` on host `a100-2` with `git_dirty=false`, exit code 0, torch 2.5.1+cu124. Run IDs and per-file SHA-256 values are in `analysis/run_index.json`; the per-run manifests are in `results/<ARM>_S0/manifest.json`. Re-loading each `state_best.pt` and re-running Dev reproduces the committed `predictions_best_dev.csv` within `4.5e-5` in raw label units (batch/eval rounding), and the curves' best epoch matches each manifest. `analysis/checks.json` records 17/17 pass (generated at implementation commit `1797c0f2`, frozen in the training commit `f607d9b5`).

**Main artefacts.**

```
PROTOCOL.md
REPORT.md
configs/bbpa.json
code/bbpa_*.py
tests/
analysis/checks.json
analysis/tables.md
analysis/paired_bootstrap.json
analysis/state_diagnostics.json
analysis/run_index.json
analysis/budget.json
results/summary.json
results/main_table.csv
results/stratified.csv
results/worst10_tree.csv
results/historical_d0.csv
results/{BLOCK_BAG,LOCAL_PORT,TREE_PORT}_S0/
    manifest.json
    curves.csv
    state_best.pt
    state_final.pt
    predictions_best_fit.csv
    predictions_best_dev.csv
```

**Reproduction commands.**

```bash
# local checks
uv run pytest -q tracks/structural-sufficiency/bridge_block_port_algebra_pilot_v0/tests
uv run python tracks/structural-sufficiency/bridge_block_port_algebra_pilot_v0/code/bbpa_checks.py

# remote execution (committed revision only)
rr deploy res
rr run res bbpa-tree-s0  --pool res-gpu0 --result tracks/.../results/TREE_PORT_S0  -- python tracks/.../code/bbpa_train.py --arm TREE_PORT  --seed 0 --device cuda --run-id TREE_PORT_S0  --out tracks/.../results/TREE_PORT_S0
rr run res bbpa-local-s0 --pool res-gpu1 --result tracks/.../results/LOCAL_PORT_S0 -- python tracks/.../code/bbpa_train.py --arm LOCAL_PORT --seed 0 --device cuda --run-id LOCAL_PORT_S0 --out tracks/.../results/LOCAL_PORT_S0
rr run res bbpa-block-s0 --pool res-gpu1 --result tracks/.../results/BLOCK_BAG_S0  -- python tracks/.../code/bbpa_train.py --arm BLOCK_BAG  --seed 0 --device cuda --run-id BLOCK_BAG_S0  --out tracks/.../results/BLOCK_BAG_S0

# after pulling the three run directories
uv run python tracks/structural-sufficiency/bridge_block_port_algebra_pilot_v0/code/bbpa_analyze.py
```

## 11. Stop condition

This round stops here: **no D1, no second seed, no additional architecture search, no official Valid/Test evaluation, no follow-up training**. The result is a single-seed exploratory screen with a negative competitiveness verdict and a positive functional-verification result.

---

*Report generated from frozen outputs; all numeric tables are reproducible from `results/summary.json` and the committed run artefacts. `analysis/tables.md` contains the full stratified and diagnostic tables.*