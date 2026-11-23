# R2-SENTENCE-REPLICATION-v1 — is the `SENTENCE_BASE` trunk a stable coordinate?

**Round:** `r2_sentence_replication_v1`
**Protocol:** `r2-sentence-replication-v1` (`PROTOCOL.md`, frozen before training)
**Model:** GWIC-v0 `SENTENCE_BASE`, imported unchanged (416,177 parameters)
**Task:** `ogbg-mollipo` whole-molecule lipophilicity regression
**Primary metric:** held-out (Dev) RMSE in original label units at the best-Dev checkpoint
**Runs:** `D0_S1` (frozen D0 split, seed 1) and `D1_S0` (frozen D1 scaffold split, seed 0) — no other run
**Training commit:** `8bb75676` (deployed to `res`, host `a100-2`)
**Engineering checks:** `analysis/checks.json` — **24/24 PASS**, run before the formal jobs
**Verdict:** **`R2_BASELINE_REPLICATED_PRELIMINARY`**

## 0. Answer in one paragraph

The trunk replicates. On the frozen D0 split the second seed reaches Dev RMSE
**0.5464** (MAE 0.4204), i.e. `0.0042` RMSE **better** than the GWIC-v0 seed-0
coordinate `0.5506` — a 0.8 % relative seed difference. On the frozen D1 scaffold
split, seed 0 reaches Dev RMSE **0.5644** (MAE 0.4217), which is `0.0180` worse
than its D0 mean and `0.0110` worse than the GINE D1 historical two-seed mean
(`0.5534`), but `0.073` better than ACRA D1 (`0.6376`). Both numbers sit inside the
pre-registered engineering gates (`D0_S1 ≤ 0.59`, `D1_S0 ≤ 0.61`) with wide
margins, so the label is `R2_BASELINE_REPLICATED_PRELIMINARY`. The aggregate metric
is stable across seed and split; the per-molecule predictions are not (seed
correlation 0.962, mean |Δ| 0.245 in label units). No tuning, no retry and no extra
run were used.

## 1. Contract and execution

| item | result |
|---|---|
| runs executed | exactly `D0_S1`, `D1_S0`; both `exit_code 0`, `smoke=false` |
| model | GWIC-v0 `SENTENCE_BASE`; sources imported byte-for-byte and sha256-pinned |
| architecture/recipe change | none (dims, recipe and arm spec read from the frozen GWIC-v0 config) |
| new parameters / modules / features | none (ledger identical, 416,177; no word/score/Q module; batch key set identical) |
| official Valid/Test labels | never loaded; every Valid label is NaN in the training object |
| training wall | 456.3 s (`D0_S1`) + 375.3 s (`D1_S0`) = 831.6 s = 0.231 h of run wall, executed in parallel |
| GPU time | 0.1089 + 0.0866 = **0.195 GPU-h** (two A100-SXM4-40GB, one process per GPU) |
| budget | ≤ 1 h wall and ≤ 2 GPU-h: met with ~5× / ~10× margin |
| engineering checks | **24/24 PASS** (`analysis/checks.json`, before the formal jobs) |
| checkpoint replay | both `state_best.pt` re-loaded and re-predicted: Dev RMSE reproduced to `<1.1e-6`, stored predictions to `<1.5e-6` (`analysis/replay.json`, `all_ok=true`) |

Environment (recorded per manifest): host `a100-2`, driver `550.163.01`,
`torch 2.5.1+cu124`, CUDA 12.4, Python 3.12.14. Single execution regime for both
runs, so the two numbers are directly comparable to each other.

## 2. What was run

* **`D0_S1`** — frozen D0 split `60ac13c0…559e` (Fit rows sha `7edd3c68…86c0`,
  Dev rows sha `eacce8ac…7193`), seed **1**, target scaler `mean 2.1858158263 /
  std 1.2139664740` (n=2856).
* **`D1_S0`** — frozen D1 scaffold split `c94e2d4b…6bee` (file sha
  `bdcd3f58…1fcc`, Fit rows sha `b9f74030…ebbc`, held-out rows sha
  `342c1de0…80a6`), seed **0**, target scaler `mean 2.1512780115 / std
  1.2159778137` (n=2856).

D1 identity was re-verified rather than trusted (check `B2`): 2856/504 rows, both
sides inside official Train, together covering it exactly, no fit/held-out
overlap, **0 shared Bemis–Murcko scaffolds** (1318 fit / 267 held-out / 1585
total), the stored `scaffold_by_row` map recomputed from SMILES with 0
mismatches, and the whole split regenerated from the frozen algorithm
(`seed=20261122`, target 504) reproducing the same row sets exactly.

Every run recomputed its own target scaler, graph-scalar standardisation,
atom-scalar standardisation and categorical support from **its own Fit rows**;
`category_support` and the atom/graph normalisers differ between runs
(`fit_atom_scalars_sha256` and `graph_norm_sha256` differ, checks `C1`–`C4`). The
realised OGB category support sets happen to coincide between the two 2856-row Fit
sets, which is recorded explicitly in `analysis/replay.json`
(`category_support_sets_identical_across_splits = true`) together with the
different Fit atom/bond counts (76,713 / 77,028 atoms) that show the masks were
really recomputed.

D1 used the **complete** frozen schedule (240 max epochs, 5 warmup, cosine,
min 80 epochs, patience 45). The D0 best epoch played no role in D1 training.

## 3. Main results

All metrics in original label units; `train_rmse` is the Fit RMSE at the best-Dev
checkpoint.

| split | seed | train_rmse | dev_rmse | dev_mae | best_epoch | epochs_run | wall_seconds | gpu_hours |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| D0 | 1 | 0.2267 | **0.5464** | 0.4204 | 90 | 135 | 456.3 | 0.1089 |
| D1 | 0 | 0.3157 | **0.5644** | 0.4217 | 64 | 109 | 375.3 | 0.0866 |

Both runs stopped by the frozen early-stop rule (`best_epoch + 45`), not by the
240-epoch cap: `D0_S1` at epoch 135 (best 90), `D1_S0` at epoch 109 (best 64),
both after `min_epochs = 80`. Final-epoch Dev RMSE was 0.5595 (`D0_S1`) and 0.5830
(`D1_S0`), so the best-Dev selection is doing real work in both runs and the
chosen checkpoints are not lucky last epochs.

`D1_S0` fits its Fit rows less tightly than `D0_S1` (0.3157 vs 0.2267) while being
worse on its held-out side: the D1 scaffold partition is a harder generalisation
target for this trunk, which is what a scaffold split is supposed to do.

## 4. Historical coordinates

Different recipes and different rounds — reference coordinates, **not** a matched
benchmark. Positive "R2 − reference" means the R2 model is worse (higher RMSE).

| model | D0 RMSE | D1 RMSE | R2 − reference (D0) | R2 − reference (D1) |
|---|---:|---:|---:|---:|
| `SENTENCE_BASE` GWIC-v0 s0 | 0.5506 | — | −0.0042 (seed 1) | — |
| GINE (3-seed D0 mean / 2-seed D1 mean) | 0.5266 | 0.5534 | +0.0198 | +0.0110 |
| ACRA | 0.5802 | 0.6376 | −0.0338 | −0.0732 |
| **this round** | **0.5464** | **0.5644** | — | — |

The trunk sits between GINE and ACRA on both splits, exactly where the single-seed
GWIC-v0 coordinate put it: clearly better than ACRA, clearly worse than GINE, and
about 0.011–0.020 RMSE away from GINE rather than in a different regime.

## 5. Answers to the required questions

**1. Does a second seed on D0 support the original 0.5506?**
Yes. `D0_S1` = **0.5464**, `0.0042` better than `0.5506` (0.8 % relative). The two
D0 seeds therefore bracket `0.546–0.551`, and `0.5506` was not a lucky or
unlucky draw at the aggregate level. Best epochs differ (81 vs 90) and epoch
counts differ (126 vs 135), so this is a genuine independent optimisation path that
lands in the same place.

**2. Is D1 seed 0 still close to the GINE historical D1 coordinate?**
Yes, with a small, consistent handicap. `D1_S0` = **0.5644** vs GINE D1 mean
**0.5534** (`+0.0110`, 2.0 % relative). Compared with the same split's ACRA number
(0.6376) the trunk is `0.073` better. So the trunk is a plausible non-GNN stand-in
in the same performance band as GINE, not an equal-or-better replacement.

**3. Is there obvious seed or split sensitivity?**
No abrupt sensitivity, but two clear descriptive facts:

* **Seed (D0, 2 seeds):** aggregate RMSE moves by only `0.0042`, MAE by `0.0072`,
  and per-molecule predictions correlate at `0.962` with mean |Δ| `0.245` and max
  |Δ| `1.58` label units. The *summary statistic* is stable; individual molecule
  predictions are not. Any later comparison must treat ±0.005 RMSE as inside seed
  noise and must not compare per-molecule deltas across seeds.
* **Split (D0 mean 0.5485 vs D1 0.5644):** `+0.0160`, i.e. the D1 scaffold
  partition is harder for this trunk. GINE showed the same direction and a larger
  shift in the parent round (`0.5266 → 0.5534`, `+0.0268`), so the trunk's split
  sensitivity is *smaller* than the GINE reference's, not larger.

There is no sign of a collapse, a run that failed to learn, or an anomalous epoch
count: both runs early-stopped through the frozen rule, with noisy but
normal-looking Dev curves and best epochs well inside the schedule.

**4. Can this model now be fixed as the non-GNN baseline for later experiments?**
Yes, with the qualifications below. The coordinate is reproducible across seed
(±0.005 RMSE) and reasonably stable across a second scaffold split (`+0.016`), it
sits between GINE and ACRA on both splits, and it has no word module, no pair term
and no extra feature that later arms would have to disentangle. Use it as a
reference band `0.546–0.564` on D0/D1 with the same best-Dev rule, the same splits
and the same fit-only statistics. Do **not** treat `0.5464` as "the" number: it is
the better of two D0 seeds.

## 6. Engineering verification (before the formal runs)

`analysis/checks.json`: **24/24 PASS**, wall 95 s, peak RSS 0.65 GB, generated at
R2 code sha `100b510b…e2e2`.

**A — frozen identity.** GWIC-v0 `gwic_common/data/structure/tensor/model/train.py`
and `configs/gwic.json` byte-identical to the pins; aggregate `code_sha` equals
`a58f942e…4a1` (the value inside the GWIC-v0 `SENTENCE_BASE_S0` manifest). Raw
cache, Train-label file, D0 and D1 split hashes all match.

**B — splits.** As in §2; plus label isolation (`B3`): the raw whitelist has no
`labels` and no `test_idx` member, exactly 3360 finite labels on exactly the
official Train rows, **0** finite Valid labels, **0** finite labels on any other row.

**C — fit-only statistics.** Target scaler equals the Fit-label mean/std exactly and
is unchanged when held-out labels are shifted by +100 (the shift does move the Dev
metric, proving the mutation was live). Graph and atom standardisers equal the
Fit-only recomputations and are label-independent. A held-out molecule mutated to
Fit-unsupported categories leaves the support masks unchanged and its embedding
rows exactly zero.

**D — unchanged tensor path.** Parameter ledger identical to GWIC-v0 module by
module and in total (416,177); `state_dict` keys and shapes identical to the frozen
`SENTENCE_BASE_S0` checkpoint (44 entries); seed-0 init hashes identical to the
frozen manifest; the frozen checkpoint replays **RMSE 0.5505590 / MAE
0.4275819** with max |Δ| `1.6e-6` against the committed per-molecule predictions;
and the new entry point, run for 2 epochs on CPU in the v0 data mode, reproduces
the frozen `gwic_train.py` reference **bit-for-bit** — same step count, LR,
training loss, gradient norm, Fit/Dev RMSE/MAE at both epochs, and identical final
state tensors. For a fixed batch the new loader produces identical tensors to the
frozen loader (all 31 keys, values included).

**The one deliberate deviation, measured.** GWIC-v0's protocol text says fit-only
atom scalars but its implementation used `fit ∪ dev` atoms. This round follows the
protocol text (fit-only). Consequence, from check `D6`/`C3`: atom-scalar mean/scale
move by at most `7.9e-3 / 1.3e-3`, the standardised atom-scalar inputs on Dev atoms
shift by at most `4.5e-3` (mean `1.6e-3`), and the resulting standardised prediction
change is at most `6.6e-4` (mean `2.1e-4`) — i.e. `<1e-3` in raw label units,
roughly an order of magnitude below the observed seed difference. Target, graph and
category-support statistics are fit-only in both modes, so this is the only numeric
difference from the v0 path and it is far too small to explain §3.

**E/F/G.** Atom renumbering, bond reordering, batch order/composition and
evaluation batch size (256 vs 37) all leave predictions unchanged (`<1e-5`). One
real optimiser step gives finite, non-zero gradients in all 16 module groups (40
parameter tensors, min norm `6e-5`); 30 AdamW steps on 32 Fit molecules cut the loss
to 0.21× its initial value. No new feature tensor (31 collated keys, exactly the
frozen set), no word-like module (19 embeddings, exactly the frozen 19), 416,177
parameters.

## 7. Budget and provenance

| item | value |
|---|---|
| launched | `rr run res r2-d0-s1 --pool res-gpu0`, `rr run res r2-d1-s0 --pool res-gpu1` |
| run ids | `r2-d0-s1-20261123-115542-83f9a50e`, `r2-d1-s0-20261123-115556-7ecf37ef` |
| commit / branch | `8bb7567645e3` / `main`, remote `git_dirty=false` |
| regime | `a100-2`, A100-SXM4-40GB, driver `550.163.01`, `torch 2.5.1+cu124`, CUDA 12.4 |
| peak VRAM | 112.3 / 112.5 MB |
| GPU hours | 0.195 total (0.109 + 0.087) |
| run wall | 456.3 s and 375.3 s, in parallel: 7 min 41 s from first launch to second completion |

One deployment obstacle is recorded for completeness: the remote checkout carried
**untracked duplicates** of the already-committed GWIC-v0 result files (the
artefacts of the previous round, later committed locally), which made `git merge`
refuse the deploy. All 27 files were verified byte-identical to the committed
versions and then moved aside on the remote into
`.rr-untracked-backup-gwic-results-20261123-115454/` (the existing
`.rr-untracked-backup-*` convention), after which `rr deploy res` succeeded. No
tracked file, no frozen artefact and no result was modified or deleted by that step.

## 8. Limitations

* **One seed per cell.** D0 has two seeds in total (0 from GWIC-v0, 1 here) and D1
  only one. There is no seed standard deviation for D1, and the D0 seed spread is a
  two-point estimate.
* **Dev is the selection set.** Best-Dev checkpoint selection makes every reported
  Dev number slightly optimistic; the rule is identical across runs, so the
  comparison is controlled but not unbiased.
* **D1 is not a blind test.** It re-partitions official Train rows with a second
  frozen scaffold split; no Valid/Test label was involved.
* **Historical coordinates are not matched recipes.** GINE/ACRA use different
  optimisers, budgets, parameter counts and epochs. The deltas in §4 are context,
  not treatment effects.
* **±0.005 is seed noise for the aggregate metric**, and per-molecule agreement
  across seeds is only 0.962 correlation. Future arms must be compared on the same
  split with the same selection rule, and paired molecule-level claims need the
  bootstrap, not point deltas.
* **One deliberate, measured deviation** from the v0 implementation (fit-only atom
  scalars, §6). It is `<1e-3` in raw label units but it is not zero.
* **No official Valid/Test metric**, no second dataset, no ensemble, no dictionary,
  no architecture screening — by protocol.

## 9. Stop condition and non-claims

The round stops here. Nothing beyond `D0_S1` and `D1_S0` was run, no threshold was
retuned after seeing results, and no failed gate needed repair.

Non-claims: no statistical-significance statement (the verdict is an engineering
criterion); no matched comparison with GINE/ACRA; no claim that the R2 trunk is
better than GINE; no official Valid/Test number; no claim about *why* D1 is harder
beyond "it is a scaffold partition of the same rows"; no per-molecule interpretation
of the seed spread.

## 10. Artefacts and reproduction

```
PROTOCOL.md   REPORT.md
configs/r2.json                       protocol/pins (model config comes from the frozen GWIC-v0 config)
code/r2_{common,data,train,checks,analyze,replay}.py
tests/test_r2.py                      9 tests, all passing
analysis/checks.json                  24/24 PASS (pre-training)
analysis/replay.json                  checkpoint re-prediction + split statistics + D0 seed spread
results/main_table.csv                the table in §3
results/summary.json                  verdict, gates, comparisons, budget
results/D0_S1/{manifest.json,curves.csv,state_best.pt,state_final.pt,
               predictions_best_{fit,dev}.{csv,npz},train.log}
results/D1_S0/{... same ...}
```

```bash
# local gate (must print 24/24 PASS before any formal run)
uv run pytest -q tracks/structural-sufficiency/r2_sentence_replication_v1/tests
uv run python tracks/structural-sufficiency/r2_sentence_replication_v1/code/r2_checks.py

# formal runs (committed revision only)
rr deploy res
rr run res r2-d0-s1 --pool res-gpu0 \
  --result tracks/structural-sufficiency/r2_sentence_replication_v1/results/D0_S1 \
  -- python tracks/structural-sufficiency/r2_sentence_replication_v1/code/r2_train.py \
     --split D0 --seed 1 --device cuda --run-id D0_S1 \
     --out tracks/structural-sufficiency/r2_sentence_replication_v1/results/D0_S1
rr run res r2-d1-s0 --pool res-gpu1 \
  --result tracks/structural-sufficiency/r2_sentence_replication_v1/results/D1_S0 \
  -- python tracks/structural-sufficiency/r2_sentence_replication_v1/code/r2_train.py \
     --split D1 --seed 0 --device cuda --run-id D1_S0 \
     --out tracks/structural-sufficiency/r2_sentence_replication_v1/results/D1_S0

# after rr pull
uv run python tracks/structural-sufficiency/r2_sentence_replication_v1/code/r2_replay.py
uv run python tracks/structural-sufficiency/r2_sentence_replication_v1/code/r2_analyze.py
```

---

*All numbers above are taken from the committed manifests, curves, predictions and
the two analysis JSON files; nothing is quoted from memory or from a previous
round's prose.*