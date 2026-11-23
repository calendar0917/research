# WOBR-v0 — Report

**Protocol:** `PROTOCOL.md` and `configs/wobr.json`  
**Formal source commit:** `dcf4562888ab093858bf8825826ede7530c37580`  
**Run:** `results/seed0/run.json`  
**Seed:** 0  
**Test policy:** `DEVELOPMENT_TEST` (evaluated every epoch; never used for
gradients, standardization, checkpoint selection, or early stopping)  
**Remote regime:** `res`, pool `res-gpu0`, node `a100-2`, 1× A100-SXM4-40GB,
driver 550.163.01, Python 3.12.14, torch 2.5.1+cu124 / CUDA 12.4

## 1. Bottom line

The implementation is engineering-correct, but this pilot is **not
predictively competitive**.

```text
Valid RMSE = 0.7815   ->  NO_COMPETITIVE_SIGNAL
Word evidence verdict  ->  STOP_THIS_IMPLEMENTATION
```

The word mechanism is not simply inactive: multi-atom words carry **85.3%** of
Train alpha mass and **67.9%** of Train absolute contribution. The stop verdict
is driven by the frozen predictive gate (official Valid RMSE > 0.70), not by an
absence of selected multi-atom words.

No WOBR-v1, second seed, hyperparameter search, control arm, or model redesign
was attempted.

## 2. Data and label normalization

| item | value |
|---|---|
| raw cache SHA-256 | `fe002287fa9c581278bc5430f5b4a5b26b35d9d4d5207d138f15956d93123c7c` |
| official split | Train 3360 / Valid 420 / Test 420 |
| split ids cover | `0..4199`, disjoint |
| target scaler | official Train only: mean `2.1621607145`, population std `1.2115398401` |
| atom-scalar scaler | official Train atoms only (a feature scaler, not a label statistic) |
| Test label | `DEVELOPMENT_TEST` |
| full OGB fields | 9 atom fields, 3 bond fields, full legal category widths |
| unseen Train-support rows | zeroed after initialization |

Valid contains one atom position with a Train-unsupported category; Test
contains zero. Because training batches contain only Train molecules, those
unsupported embedding rows receive no supervision.

All metrics below are in **raw label units** unless explicitly marked
“standardized”. Raw units are the reporting default; standardized RMSE is
included to make the threshold interpretation explicit.

## 3. Engineering verification

Full pre-training checks ran on all 4200 rows and all 1,530,239 candidates.
All **11/11 checks passed**.

| check | result |
|---|---|
| real, connected, induced candidates; singleton per root; exact internal/boundary | PASS |
| atom renumbering, edge reversal, bond order, candidate order | PASS; float64 max error `4.44e-16` (`< 1e-8`) |
| sparsemax vs two independent references; nonnegative; normalized; exact zeros; padding zero | PASS; max sum error `8.88e-16`; max reference error `4.44e-16`; packed reference error `1.11e-16`; padding max `0` |
| same-count/different-connection synthetic pair differs in bilinear terms | PASS; max `|ΔQ| = 0.00814`, max `|Δyhat| = 0.02022` |
| exact contribution identity and per-bond word-pair expansion | PASS; float64 synthetic `1.11e-16`, float32 real batch `2.86e-6` (`< 1e-4`), expansion `6.80e-8` |
| batch vs single inference | PASS; max `7.15e-7` |
| finite, nonzero gradients for word encoder, score, `a`, `U`, bond interaction | PASS |
| reproducible seed-0 initialization and frozen explicit values | PASS |
| data SHA, official split, Train-only scaler, metric restoration, full embedding widths, zeroed unsupported rows | PASS |
| isolated atoms and disconnected components preserved | PASS; 3 isolated atoms, 1 multi-component molecule |
| candidate totals and real-batch resources | PASS |

Candidate coverage:

| split | molecules | atoms | words | words by size 1/2/3/4 | max per root |
|---|---:|---:|---:|---|---:|
| Train | 3360 | 90,514 | 1,214,744 | 90,514 / 197,122 / 415,002 / 512,106 | 54 |
| Valid | 420 | 11,467 | 156,420 | 11,467 / 25,170 / 53,187 / 66,596 | 48 |
| Test | 420 | 11,587 | 159,075 | 11,587 / 25,506 / 53,982 / 68,000 | 54 |
| all | 4200 | 113,568 | 1,530,239 | 113,568 / 247,798 / 522,171 / 646,702 | 54 |

Every root has at least the singleton. There are no missing-root cases. The
complete structure build took 105.6 s and used 344.9 MB of NumPy arrays
(process peak RSS 783 MB after build).

One real 64-molecule batch had 1,713 atoms, 1,869 bonds, and 23,080 words;
collate 0.017 s, forward 0.315 s, backward 0.247 s, peak VRAM 290.8 MB in the
isolated resource probe (formal training peak 833.7 MB including optimizer and
activations).

### Necessary scalar fix relative to the parent GWIC helper

The parent `_word_scalars` populated the induced pairwise-distance block
`[4:7]` using the root-distance of the right-hand atom of each pair. That
depended on support slot order and failed atom-relabelling invariance. WOBR
computes the true shortest-path distance inside the induced support for
`[4:7]`. The singleton block is all zeros either way, so singleton scalars are
unchanged. All other scalar positions, including the boundary-type fractions,
follow the GWIC layout. This is the only implementation correction to the
copied GWIC structure logic; it is why check 2 passes.

## 4. Preflight and budget

The two-epoch full-data preflight produced:

| item | value |
|---|---:|
| epochs | 2 |
| mean epoch | 9.165 s |
| peak VRAM | 831.5 MB |
| projected formal process (240 epochs + overhead) | 0.641 GPU-h |
| projected total including preflight | 0.676 GPU-h |
| formal budget | 2.0 GPU-h |
| total budget including preflight | 2.5 GPU-h |
| budget gate | PASS |

The formal run early-stopped at epoch 100, so actual cost was lower than the
240-epoch projection:

| item | value |
|---|---:|
| structure prep | 105.4 s |
| training loop + per-epoch full evaluation | 984.8 s |
| formal process wall | 1093.5 s = 0.304 GPU-h |
| formal training only | 0.274 GPU-h |
| preflight process | 0.035 GPU-h |
| formal + preflight | 0.339 GPU-h |

No candidate cuts, batch changes, model changes, or config changes were made
after the preflight. No equivalent-vectorization change was needed.

## 5. Formal training result

Single model, seed 0, AdamW peak lr `3e-4`, cosine schedule with 5-epoch
warmup, weight decay `1e-4`, batch 64, gradient clip 5, float32, no AMP.
Checkpoint selection was minimum official Valid RMSE. The model has
**423,923** parameters.

| split | n | RMSE (raw) | MAE (raw) | RMSE (standardized) |
|---|---:|---:|---:|---:|
| Train | 3360 | 0.6045 | 0.4703 | 0.4990 |
| Valid | 420 | **0.7815** | 0.5889 | 0.6450 |
| Test (`DEVELOPMENT_TEST`) | 420 | 0.7633 | 0.5907 | 0.6301 |

- Best Valid epoch: **55**
- Early stop: **epoch 100** (`epoch 55 + patience 45`; minimum 80 satisfied)
- Maximum allowed epochs: 240
- Train–Valid RMSE gap at best: **0.1769**
- Mean-predictor context: Valid RMSE `1.2181`, Test RMSE `1.1173`
- Test is lower than Valid by `0.0181`; it was not used to choose the epoch.

The frozen verdict rule gives:

```text
Valid RMSE 0.7815 > 0.70  ->  NO_COMPETITIVE_SIGNAL
```

The historical GINE range `0.61–0.64` is **descriptive only**. It comes from a
different recipe and is not a matched baseline; no “worse than GINE” formal
comparison is claimed here. The mean-predictor numbers are context, not a
control arm.

## 6. Word-selection and contribution diagnostics

One forward pass of the best-Valid checkpoint over all 4200 rows; no
retraining, no checkpoint switching, no Valid/Test-label ranking of words.

### 6.1 Selection

| statistic (Train / Valid / Test) | Train | Valid | Test |
|---|---:|---:|---:|
| roots | 90,514 | 11,467 | 11,587 |
| nonzero candidates per root, mean | 9.735 | 9.908 | 9.950 |
| nonzero candidates per root, median | 9 | 9 | 9 |
| top-1 alpha, mean | 0.2806 | 0.2756 | 0.2752 |
| alpha mass, size 1 | 0.1465 | 0.1479 | 0.1461 |
| alpha mass, size 2 | 0.1065 | 0.1046 | 0.1038 |
| alpha mass, size 3 | 0.3014 | 0.2955 | 0.2969 |
| alpha mass, size 4 | 0.4456 | 0.4520 | 0.4532 |
| `z` participation ratio / 64 | 3.27 | 3.29 | 3.37 |
| `z` entropy rank / 64 | 5.13 | 5.21 | 5.26 |
| `z` top-1 eigenvalue share | 0.484 | 0.488 | 0.475 |
| self-term RMS | 0.813 | 0.915 | 0.909 |
| bond-term RMS | 0.771 | 0.744 | 0.781 |
| self cancellation | 0.241 | 0.255 | 0.271 |
| bond cancellation | 0.534 | 0.516 | 0.535 |
| total cancellation | 0.262 | 0.267 | 0.295 |
| bond-term nonzero fraction (`>1e-8`) | 1.000 | 1.000 | 1.000 |

The 64-d `z_v` is effectively only about **3.3 dimensions** by participation
ratio, despite all 64 output dimensions being non-constant. Sparsemax is
genuinely sparse: the check's exact-zero fraction on real candidate entries was
0.809. Multi-atom words dominate alpha mass (`0.8535` on Train), so the model
does not collapse to singleton ownership.

### 6.2 Contributions

Contribution decomposition, standardized units:

```text
c_(v,S) = alpha_(v,S) * ( dot(a, w_(v,S))
          + 0.5 * sum_{u adjacent v} Q_(v,u)(w_(v,S), z_u) )
```

The exact identity passed over all 4200 molecules: max absolute error
`1.29e-6` in float32 (tolerance `1e-4`). Reloading `predictions_best.npz`
matched the analysis forward pass to `3.03e-6`. The per-bond expansion check
also passed at `6.80e-8`.

| contribution statistic (all words) | Train | Valid | Test |
|---|---:|---:|---:|
| instances | 1,214,744 | 156,420 | 159,075 |
| mean (standardized) | 0.002326 | 0.002484 | 0.002745 |
| mean absolute (standardized) | 0.010470 | 0.010141 | 0.009860 |
| sum | 2825.62 | 388.48 | 436.63 |
| sum absolute | 12718.16 | 1586.27 | 1568.54 |
| positive fraction | 0.628 | 0.635 | 0.635 |
| cancellation `1 - |sum|/sum_abs` | 0.778 | 0.755 | 0.722 |

Raw-unit contributions are `standardized * 1.2115398401`. The per-word arrays
in `results/seed0/contributions_best.npz` contain both
`contribution_std` and `contribution_raw`; `word_types.csv` contains raw and
standardized means/stds.

Singleton vs multi-atom, all splits:

| group | count | mean | mean absolute | abs share | net share |
|---|---:|---:|---:|---:|---:|
| singleton | 113,568 | -0.02582 | 0.04490 | 0.3213 | -0.8031 |
| multi-atom | 1,416,671 | 0.004647 | 0.007605 | 0.6787 | 1.8031 |

This is a decomposition, not a causal claim: singleton words have larger
per-word absolute contributions on average but fewer instances; multi-atom
words collectively carry more absolute contribution. The strong net-share
values reflect cancellation and should not be interpreted as effect sizes.

## 7. Exact word reuse

Word classes are exact rooted-word canonical certificates: root fixed in slot
0, exhaustive permutation of at most three non-root atoms, all 9 atom and 3
bond categorical fields, exact integer bytes (not hashed IDs).

| item | value |
|---|---:|
| exact rooted word classes | 37,908 |
| classes by size 1/2/3/4 | 76 / 1,113 / 8,740 / 27,979 |
| qualifying classes (≥20 Train molecules and ≥3 Train scaffolds) | 4,896 |
| qualifying by size 1/2/3/4 | 56 / 361 / 1,490 / 2,989 |
| Train word instances | 1,214,744 |
| qualifying Train instances | 1,102,245 |
| Train molecules covered by qualifying classes | 3360 / 3360 |

The top 20 classes ranked by **Train mean absolute contribution** contain 12
singletons, 7 size-2 classes, and 1 size-3 class; no size-4 class enters the
top 20. Examples from `analysis/word_reuse.json`:

| size | Train molecules | Train scaffolds | Train mean `|c|` | Train mean `c` (raw) |
|---:|---:|---:|---:|---:|
| 2 | 37 | 26 | 0.2909 | -0.3524 |
| 3 | 404 | 165 | 0.2511 | -0.3042 |
| 2 | 404 | 165 | 0.2361 | -0.2861 |
| 1 | 694 | 320 | 0.4735 | -0.5736 |

The exact top-20 ranking and certificates are in
`analysis/word_reuse.json`; `analysis/word_types.csv` contains the top 2000
qualifying classes with per-split raw/standardized statistics. Reuse is real:
multi-atom classes appear across
hundreds of Train molecules and many scaffolds. Individual size-4 classes are
numerous but their per-class mean contribution is lower; size-4 alpha mass is
nevertheless large because there are many such candidates.

## 8. Verdict and interpretation

Frozen predictive thresholds:

```text
<= 0.65          PREDICTIVE_CANDIDATE
(0.65, 0.70]     WEAK_PREDICTIVE_SIGNAL
> 0.70           NO_COMPETITIVE_SIGNAL
```

Observed Valid RMSE `0.7815` gives **`NO_COMPETITIVE_SIGNAL`**. The frozen word
gate then gives **`STOP_THIS_IMPLEMENTATION`**.

Why this is not labeled `PREDICTIVE_BUT_WORD_EVIDENCE_WEAK`: that label applies
only when prediction is at least weakly competitive but multi-atom evidence is
weak. Here prediction fails the first gate, even though multi-atom alpha mass
is `0.8535` and multi-atom absolute-contribution share is `0.6790`.

What this round establishes:

- the word-owned bilinear construction can be implemented exactly, with a
  verified contribution identity, invariant candidates, sparse root selection,
  and active multi-atom words;
- this single frozen implementation, recipe, and seed does **not** produce a
  competitive Valid RMSE on this split;
- the failure is not explained by words being unused. Selection and
  contribution diagnostics show substantial multi-atom use, but `z` has very
  low effective dimension and the model overfits Train relative to Valid.

What it does **not** establish:

- that word ownership or bilinear responses are intrinsically insufficient;
- that any particular word causes lipophilicity;
- that the model is worse than GINE (different recipe, no matched baseline);
- that another seed, capacity, optimizer, or data pipeline would fail or
  succeed. None were run. No redesign is attempted here.

## 9. Engineering and provenance notes

- Raw cache and split identities were checked before training; full check 9
  passed.
- The full checks ran against commit `dcf45628`; formal training and analysis
  used the same commit. Analysis was a separate post-training process on the
  saved best checkpoint.
- The remote checkout had untracked FGCR output files from an earlier run.
  They were moved to timestamped `.rr-untracked-backup-*` directories before
  deployment; no tracked file was modified and no result was deleted.
- The formal run wrote `metrics.csv`, `curves.csv`, `best.pt`, `last.pt`,
  `predictions_best.npz`, `run.json`, and `train.log`. The analysis wrote
  `contributions_best.npz`, `contributions_best.csv`, and the analysis JSON/CSV
  artifacts.
- The best checkpoint is selected by Valid RMSE only. Test metrics are
  `DEVELOPMENT_TEST` and are reported for completeness, not as a terminal
  test result.
- Training used no AMP and no auxiliary loss. `h_v` was used only as the
  candidate-selection query; it has no direct prediction path.
- The word encoder and score MLP use dropout 0.1 in training; evaluation is
  deterministic.

## 10. Artifacts

```
PROTOCOL.md
configs/wobr.json
code/
tests/
analysis/
  checks.json
  candidate_coverage.json
  preflight_gpu.json
  word_selection.json
  word_contributions.json
  word_reuse.json
  word_types.csv
  high_contribution_words.csv
  analysis_summary.json
  verdict.json
  curves.json
results/preflight_seed0/
  run.json, curves.csv, metrics.csv, predictions_best.npz, train.log
results/seed0/
  run.json, metrics.csv, curves.csv, train.log
  best.pt, last.pt, predictions_best.npz
  contributions_best.npz, contributions_best.csv
```

Key reproduction entry points are listed in `PROTOCOL.md` §12. All reported
numbers are from the committed source revision and the run manifests above.