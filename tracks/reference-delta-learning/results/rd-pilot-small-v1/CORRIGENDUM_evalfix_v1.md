# Corrigendum: C evaluation used the wrong query domain (`rd-evalfix-v1`)

Date: 2026-11-16  
Track: `reference-delta-learning`  
Protocol: `rd-pilot-small-v1`  
Evaluation revision: `rd-evalfix-v1`  
Error-source commit: `7f51ae14c99ccf5929d3ccf118dd73b41c05b44d`  
Fix commit: `ca475ba838213ffd0ca67f52dae19e71b2c57c40` (branch `rd-pilot-small-v1-evalfix`)  
Remote execution: `rr` on `a100-2`, runs `rd-evalfix-esol2-20261116-193213-505b5a48` and `rd-evalfix-lipo2-20261116-193213-b3e10c64`

## Verdict

The C arm's Val/Test evaluation was invalid before this revision.  The eight
D/C boosters are frozen and byte-identical before/after; **no model was
retrained**.  Only the existing C boosters were re-evaluated, re-selected by
full Val RMSE, and all dependent artifacts were regenerated.

`EXECUTION_PASS = true`.  The corrected scientific verdict is
**`CROSS_TASK_TRADEOFF`**, not `NO_JOINT_INCREMENT_THIS_RECIPE`:

* `ogbg-mollipo` is a `strong_joint_task` (C beats D/N/R by >=3% on both Val
  and Test);
* `ogbg-molesol` beats D/R on Val but is worse than D/R on Test, so it fails
  the "other task not worse by >5%" condition;
* `SUBSTANTIVE_PROGRESS = false`, `COMPETITIVE_WITH_FROZEN_REFERENCES` is true
  for molesol and false for mollipo, and
  `WORTH_STRUCTURAL_COMPARATOR_FOLLOWUP = false` under the unchanged gates.

## Root cause

`rd_pairs.pair_matrix` and `rd_pairs.pair_feature_blocks` read their molecular
features from a hard-coded `x = task.x["train"]`:

```python
# before (7f51ae14)
x = task.x["train"]
x1 = x[query_local]
x2 = x[reference_local]
```

The training call was correct because `query_local` and `reference_local` were
both Train-local.  The evaluator reused the same functions for Val/Test:

```python
# before: rd_arms._evaluate_c / c_state
pair_feature_blocks(task, split, entry, rows)
```

`rows` were local indices into the evaluated split, but the function indexed
`task.x["train"]` with them.  The reference table, reference weights and labels
were built from the true Val/Test split.  Thus the two sides of every C pair
described different molecules, while nothing raised an index error because
`n_valid < n_train` on both datasets (113 < 902; 420 < 3360).  The old C
Val/Test curves, selected states, predictions, comparisons, cases and route
decision were all downstream of this mismatch.

The padded slots in `evaluation_pairs` also used the query's own local index as
a reference sentinel.  They now use Train row `0` with weight `0`, so a padded
slot is explicitly a zero-weight Train reference.

## Fix

The constructors are now keyword-only and require two explicit domains:

```python
pair_matrix(
    *,
    query_x, reference_x,
    query_local, reference_local,
    chunk=8192,
)
pair_feature_blocks(
    *,
    query_x, reference_x,
    query_local, reference_local,
    chunk=8192,
)
```

The pair math is unchanged: `z = concat(q, r, q - r)` in the 6177-dimensional
feature space.  Reverse pairs still use `BlockLayout(reverse=True)`, which
swaps X1/X2 and negates X3 on correctly constructed forward blocks.

Call-site rules now enforced by the code and tests:

| path | `query_x` | `reference_x` | query/reference local indices |
| --- | --- | --- | --- |
| training pairs | `task.x["train"]` | `task.x["train"]` | Train / Train |
| C evaluation, Val | `task.x["valid"]` | `task.x["train"]` | Val / Train |
| C evaluation, Test | `task.x["test"]` | `task.x["train"]` | Test / Train |

Updated call sites: `rd_pairs.pair_matrix`,
`rd_pairs.pair_feature_blocks`, `rd_pairs.evaluation_pairs`,
`rd_arms.build_eval_blocks`, `rd_arms._evaluate_c`, `rd_arms.c_state`,
`rd_arms._sample_pair_matrices`, `rd_arms.run_c`, `rd_runner._predict_c`, and
the native-prediction replay in `rd_reeval`.

`rd_reeval.py` adds an independent implementation for verification,
`forbid_training()` (monkeypatches `xgboost.train`, `Booster.update` and sklearn
`fit`, and counts calls), input-identity/cache fingerprints, a seed-1703 legal
slot sampler with first/last query coverage, independent native-XGBoost prefix
and selected-state replays, and a Train-curve invariance check.

A new evaluation-only stage is available:

```yaml
stage: reevaluate_c
dataset: ogbg-molesol   # or ogbg-mollipo
fit_allowed: false
force_eval_rebuild: true
evaluation_revision: rd-evalfix-v1
```

The stage loads the existing `C_{H1,H2}.ubj`, extracts all 1600 trees, and never
enters `run_c` or `xgboost.train`; if a booster is missing it reports
`BLOCKED_MISSING_MODEL` rather than training a replacement.

## Why the old native-XGBoost cross-check did not catch it

The old cross-check compared the incremental tree replay with a direct native
XGBoost prediction on the *same* pair matrix.  Both sides therefore consumed the
same wrong query features, so their agreement was evidence of replay
consistency, not of query identity.  The old checks also covered Train pair
curves, D predictions and scalar metrics; the C Train path was correct, and the
Val/Test path had no independent row-id/feature-domain assertion.  The new
checks add exactly that independent identity check and an end-to-end oracle.

## New correctness evidence

Thresholds are unchanged; these are gate-relevant checks, not significance
tests.

1. **Sentinel-matrix regression test** (`tests/test_rd_evalfix.py`).  A
   synthetic task whose Train/Val/Test feature blocks are deliberately
   distinct.  The reverse pair rows are now:
   * Val: `[3, 4, 11, 12, -8, -8]`
   * Test: `[3, 4, 21, 22, -18, -18]`

   The old constructor fails the same sentinel test, proving it read Train for
   both splits.

2. **Independent oracle aggregation test** (`y = x[:, 0] + 7`).  End-to-end
   evaluation restores Val `18` and Test `28` (the true split values), not the
   Train row `8`, and not `8 + mu`.  Non-zero `mu/sigma` are used so an
   accidental `+mu` or a missing standardization is visible.

3. **Real-data identity check**.  For each dataset/split, 8 legal slots
   (seed `1703`, first and last query covered) are compared across row ids,
   query blocks, reference blocks, forward pairs, reverse pairs and difference
   blocks.  `max_abs_*_diff = 0.0` and `query_row_id_matches =
   reference_row_id_matches = true` everywhere.  Native XGBoost prefix replays
   are compared at rounds `1, 2, 3, 1600` plus each preset's selected round
   (molesol H1 `61`, H2 `1577`; mollipo H1 `1600`, H2 `1595`), with raw-unit
   and margin errors `0.0` (tolerance `1e-5`).

4. **Full selected-state native recompute**.  The selected C state is replayed
   natively for all Val/Test queries.  `max_abs_prediction_diff = 0.0`, and
   the replayed RMSE/MAE equal the recorded values exactly:
   * molesol: Val `0.8774950405`, Test `0.8905742845`
   * mollipo: Val `0.7129631769`, Test `0.6879425718`

5. **Training-input invariance**.  Train-pair blocks and dense records are
   bitwise identical to the pre-fix construction (`sample_sha256` unchanged,
   `512` sample records).  Train molecule curves are unchanged for every
   recorded round (`max_abs_train_rmse_diff = 0.0`, `9600` rows per preset).

6. **No training in the repair**.  Instrumented counts are
   `train = 0`, `booster_update = 0`, `sklearn_fit = 0` for every stage run.
   The missing-model path is `BLOCKED_MISSING_MODEL`, never a retrain.

7. **Cache isolation**.  The new key includes the revision
   `rd-evalfix-v1`, code SHA-256 (`rd_arms`, `rd_common`, `rd_eval`,
   `rd_pairs`, `rd_reeval`), query split/row-id/feature hashes,
   reference/pair hashes, weight-matrix hashes, scaler, the six configs and the
   C model SHA-256.  Old C curves cannot satisfy the new key.  Keys:
   `efe1b54b59530ea322af45decd4f30856bdbb9bede2429c84559e54d3e37e137`
   (molesol) and `9b8bc922a1b040fd396a8cb4914a31049dfa0195b69ac04bc4cf8c40a2727e72`
   (mollipo).

Tests: `uv run pytest -q tests` -> `37 passed`; `uv run ruff check code tests` ->
clean.

## Frozen model SHA-256 (before = after)

| dataset | preset | SHA-256 |
| --- | --- | --- |
| ogbg-molesol | C_H1 | `9415143d5c8212a7ef18fe5d5611b5e9563a6149d00f032fbca81f53e96b06b8` |
| ogbg-molesol | C_H2 | `32e73107b8f5b268cf5c4f3f21751e7a2411261c9a37fc17a47691621b4122bd` |
| ogbg-molesol | D_H1 | `719c3ab2fde6d7b3f6391aca5b2bdeca397cf3943b49110568e2975aad5b9024` |
| ogbg-molesol | D_H2 | `000df6502fb675f4605b0b2b6defc65fe569639e199dcb1607f37db941b792ec` |
| ogbg-mollipo | C_H1 | `9e75ae321be2c0bba316f5e56c3a72617bfd2a03ba7fcfed0e4c8125e01cf2e5` |
| ogbg-mollipo | C_H2 | `78c128dc67aae760b5490bdcbc6d7031ea9767421ce048fd87771305aae9ccd3` |
| ogbg-mollipo | D_H1 | `7532858029f2179e3ab18a16cd32d795eba71f52196cc6ee1405b32e776737d9` |
| ogbg-mollipo | D_H2 | `2e28a5bf8c9ed835da3841ab43960e38e3e8620a81d59a958bc1531b4be3fd6f` |

`checks.json -> C.model_sha.before/after` records the same values per C preset.
`trained_at` fields in `selection.json` are unchanged from the original run.

## Corrected C selection

C was re-selected only by full Val RMSE, with the original tie-break order
(`H1` before `H2`, earlier round, smaller K, uniform before softmax).  Test was
evaluated but never used for selection.  All 1600 rounds and six configs
(K=4/16/64 x uniform/softmax tau=0.1) were recomputed per prefix.

| dataset | old Val/Test | old C state | corrected Val/Test | corrected C state |
| --- | --- | --- | --- | --- |
| ogbg-molesol | 1.5297 / 1.5824 | H2 r1, K4 softmax | **0.8775 / 0.8906** | H2 r1577, K4 softmax |
| ogbg-mollipo | 0.9991 / 0.9181 | H2 r1, K16 softmax | **0.7130 / 0.6879** | H2 r1595, K16 softmax |

The old curves rose to roughly `2.97 / 3.03` (molesol) and `1.55 / 1.46`
(mollipo) by round 1600; the corrected curves fall:
molesol C H2 Val `1.4998 -> 0.8779`, Test `1.5522 -> 0.8906`;
mollipo C H2 Val `0.9883 -> 0.7130`, Test `0.9099 -> 0.6880`.
C now beats N on all 1600 rounds for both splits on both tasks.

## Corrected comparisons (raw units; Val-selected states)

| dataset | arm | Val RMSE | Test RMSE |
| --- | --- | ---: | ---: |
| ogbg-molesol | D | 0.9270 | 0.8159 |
| ogbg-molesol | N | 1.5293 | 1.5846 |
| ogbg-molesol | R | 0.8814 | 0.8282 |
| ogbg-molesol | C | **0.8775** | 0.8906 |
| ogbg-molesol | G | 0.9353 | 0.9354 |
| ogbg-molesol | L | 0.8686 | 0.8570 |
| ogbg-mollipo | D | 0.7747 | 0.7628 |
| ogbg-mollipo | N | 0.9970 | 0.9178 |
| ogbg-mollipo | R | 0.7448 | 0.7135 |
| ogbg-mollipo | C | **0.7130** | **0.6879** |
| ogbg-mollipo | G | 0.6091 | 0.6456 |
| ogbg-mollipo | L | 0.7052 | 0.7139 |

C-vs-control gains (positive = C better):

| dataset | split | C-D | C-N | C-R |
| --- | --- | ---: | ---: | ---: |
| ogbg-molesol | Val | +5.34% | +42.62% | +0.44% |
| ogbg-molesol | Test | -9.15% | +43.80% | -7.53% |
| ogbg-mollipo | Val | +7.96% | +28.49% | +4.27% |
| ogbg-mollipo | Test | +9.82% | +25.04% | +3.58% |

The paired bootstrap (2000 resamples by query molecule, seed 1703) and
drop-one diagnostics are in `tables/comparisons.csv`; some intervals include
zero, so these are gate comparisons, not significance claims.  R vs D is
unchanged: Val +4.9%/+3.9%, Test -1.5%/+6.5%.  C is within 5% of L on
molesol (Val/Test ratios 1.010/1.039) but not within 5% of G on mollipo
(1.171/1.066).

## Retracted conclusions and artifacts

Explicitly withdrawn:

* "C only selects round 1" / "C collapses to N" on these tasks.
* "C becomes worse with training" / "越训练越不泛化" for C Val/Test.
* `NO_JOINT_INCREMENT_THIS_RECIPE` as the verdict of this round.
* Old C Val/Test curves (`curves_C_H*.csv`), old selected state in
  `selection.json`, old `pred_C.npz` / `pred_C_delta.npz`, old C-D/C-N/C-R
  comparisons/bootstrap/drop-one/coverage C errors, old three Val case ledgers,
  old main table and the old `decision.json` / `REPORT.md` route decision.
* Old report text that treated the C failure as evidence about the pairwise
  target rather than an input-domain bug.

Retained unchanged: the eight D/C boosters, D/N/R selected states and
predictions, G/L point references, reference tables, frozen Train pairs,
sample weights and the six inference configs.  C's Train molecule curve is
unchanged; only its Val/Test input domain and downstream outputs changed.

The old conclusion is replaced by: the corrected C is a genuine task-dependent
tradeoff (strong on mollipo, not robust on molesol Test), and this frozen recipe
still does not pass the joint/structural-follow-up gates.  It is not evidence
that pairwise learning cannot work.

## Cost of the repair

Only evaluation cost, no new training:

| dataset | worker wall | C eval seconds H1/H2 | allocated GPU | GPU compute | allocated GPU-hours |
| --- | ---: | ---: | ---: | ---: | ---: |
| ogbg-molesol | 79.8 s | 16.6 / 27.1 s | 1 A100 (idle for C) | 0 | 0.022 |
| ogbg-mollipo | 297.1 s | 62.6 / 102.1 s | 1 A100 (idle for C) | 0 | 0.083 |

Two-task sum of worker wall = 376.9 s; parallel elapsed was about 5m07s.  Both
workers used 4 CPU threads and `xgboost_device=cpu`, so GPU compute is zero;
counting the two exclusive allocations (even though idle) gives about 0.105
GPU-hours.  The pre-fix training ledger is unchanged; this repair added zero
training trajectories and zero training seconds.

## Reproduction

```bash
# from the repository root
uv run pytest -q tracks/reference-delta-learning/tests
bash tracks/reference-delta-learning/run_remote.sh \
  --purpose "rd-evalfix-v1 reevaluate C ogbg-molesol" \
  --set stage=reevaluate_c --set dataset=ogbg-molesol \
  --set fit_allowed=false --set force_eval_rebuild=true \
  --set evaluation_revision=rd-evalfix-v1 --set xgboost_device=cpu
# same for ogbg-mollipo, then:
bash tracks/reference-delta-learning/run_remote.sh \
  --purpose "rd-evalfix-v1 aggregate + report" \
  --set stage=aggregate --set dataset=all
```

The formal remote runs used `rr` with one process worker per dataset on
`res-gpu0`/`res-gpu1`, at commit `ca475ba8`.  The old caches under the previous
revision are never consulted; the new keys and `evaluation_identity.json`
record the full input identity.

## Residual limitation

This is still a single training seed (seed 0), a frozen 1600-round capacity,
and an exploration gate (3%/5%/50%), not a significance test.  The correction
fixes an input-identity bug; it does not turn the result into a multi-seed or
architecture-search claim.
