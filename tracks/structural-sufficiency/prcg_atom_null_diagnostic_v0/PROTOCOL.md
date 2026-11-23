# PRCG-ATOM-NULL — graph-word necessity diagnostic (protocol)

**Protocol id** `prcg-atom-null-diagnostic-v0`
**Status** frozen before the formal ATOM_ONLY training run.
**Parent round** `tracks/structural-sufficiency/predictive_residual_composition_grammar_pilot_v0`
(frozen; all parent files are read-only).
**Scope** one diagnostic question, one split, one seed, one new model arm. This is not a
new architecture search, not PRCG-v1, and not a benchmark submission.

## 1. Question

Does the PRCG-v0 predictive word lexicon add incremental predictive value over a
**no-merge atom representation**? Equivalently: is there evidence that forming structural
words by merging is useful, or is the observed Dev number explainable by the same model
without any merges?

The question is deliberately narrow. ATOM_ONLY is the natural null arm: it uses exactly the
same model, data, split, width, initialisation and recipe, but the merge grammar is empty.

## 2. Read-only parent inputs

All of the following are read, never modified:

* `predictive_residual_composition_grammar_pilot_v0/REPORT.md`, `PROTOCOL.md`,
  `configs/prcg.json`, `code/`, `tests/`, `grammar/`, `results/`, `analysis/`;
* the frozen D0 split and label-safe raw cache named by the parent `PROTOCOL.md`.

The parent config sha256 is
`d4afc405ec3723c27ccae956b689352c06d9aa358278077775b25e3e873c88ce`.
The parent predictive/frequency grammars, checkpoints and results are reused as committed.

## 3. Part A — Ridge scoring audit (CPU only)

The parent discovery screen (`RoundScreener`) scores a candidate merge by the two-direction,
scaffold-disjoint, cross-fitted held-out MSE gain. It residualises the candidate column
against the current design and uses the denominator

`resid_tr^T resid_tr + lambda`.

The strict augmented Ridge, with the candidate column appended to the design and the same
`lambda = 5` on its coefficient, has the Frisch–Waugh denominator

`c_tr^T resid_tr + lambda`.

The difference is `lambda * ||S||_P^2`, where `S = (X^T X + lambda P)^{-1} X^T c`.
For each of the 32 predictive discovery rounds we will:

1. replay the original candidate set, both scaffold folds, matching and selected rule;
2. score every candidate with the original `RoundScreener`;
3. score every candidate by directly solving the augmented Ridge normal equations
   `[X, c]` on each training fold with the same `lambda` (candidate penalty 1, intercept 0);
4. verify the direct solve agrees with the Schur/Frisch–Waugh formula;
5. compare the two scores, their fold signs, the selected-rule gains, and the candidate
   ordering (Spearman, Kendall, top-1, top-5 overlap, constraint-aware selection).

The audit reports actual impacts only. It does not alter the parent grammar, report or
results. If the strict score materially changes the selected-rule reading, the parent
predictive-discovery interpretation must be contracted explicitly in the new report.

## 4. Part B — the only new arm: ATOM_ONLY

### 4.1 Construction

* Grammar is empty (`Grammar()` with zero rules). Every atom remains a single-atom word.
* Every real molecular bond is an inter-word bond; there are no crossing bonds and no
  multi-atom words.
* The PRCG-v0 `Compose` module is still instantiated (same parameter count), but no merge
  object exists, so it is never executed.
* Atom encoding, `Q`, `f`, global readout, raw fields and the eight graph scalars are the
  unchanged PRCG-v0 modules. No atom-only-specific network, embedding table, readout or
  auxiliary objective is added.
* Data, D0 split, model width, initialisation seed, training recipe and early-stopping
  rule are exactly the frozen PRCG-v0 ones (`dims`, `recipe`, seed 0). The effective config
  is the parent config plus one arm entry, and the parent config hash is asserted before
  training.

### 4.2 Required checks before the formal run is accepted

* atom coverage: every atom occurs in exactly one final word; all words are size 1;
* bond coverage: every real bond is exactly one inter-word bond; no intra-word bonds and
  no crossing bonds;
* renumbering invariance (atom renumbering and bond reordering);
* batch composition and batch order invariance;
* label isolation: finite labels only on official Train, no official Valid label, Fit/Dev
  disjoint and covering official Train, Fit-only target and graph standardisers;
* construction claims: `Compose` is instantiated but has zero forward calls and no
  gradient path; the word state equals the atom state; parameter count and initialisation
  hashes equal PRED_GRAMMAR.

### 4.3 Run

One seed (0), one formal training run, CPU, 4 threads, deterministic algorithms. No smoke
run is used as a result; any smoke check is only an engineering gate.

## 5. Part C — comparison

Reuse the committed parent results:

| arm | Dev RMSE |
| --- | --- |
| `PRED_GRAMMAR` | 0.7115 |
| `FREQ_GRAMMAR` | 0.7486 |
| `PRED_PORT_MEAN` | 0.7108 |

New: ATOM_ONLY Fit RMSE, Dev RMSE, Dev MAE, best epoch and wall time. The primary contrast is

`delta_merge = RMSE(ATOM_ONLY) - RMSE(PRED_GRAMMAR)`.

Positive `delta_merge` means merging (PRED_GRAMMAR) is better. Uncertainty is a descriptive
paired bootstrap over the same 504 Dev molecules, 10 000 resamples, seed 20261202, common
resample indices for all four arms, 95 % percentile interval. Also report the analogous
contrasts against FREQ_GRAMMAR and PRED_PORT_MEAN, and per-molecule win rates.

Atom coverage is reported exactly:

* covered-atom fraction = 1 for every arm (every atom is in exactly one word);
* singleton-atom fraction: PRED ≈ 58.6 %, FREQ ≈ 9.9 %, ATOM_ONLY = 100 % on Fit.

The token counts are not matched (ATOM_ONLY has more words per molecule than PRED_GRAMMAR);
this is a stated confound, not a hidden one.

## 6. Part D — decision rule (fixed before the run)

Let `delta = delta_merge` and `[lo, hi]` be its paired-bootstrap 95 % interval.

| label | condition | action |
| --- | --- | --- |
| `ATOM_ONLY_MATCHES_OR_BEATS_PRED` | `delta <= 0` | No evidence that merging has incremental predictive value. Stop the PRCG composition-model route. |
| `PRED_CLEAR_ADVANTAGE_OVER_ATOM_ONLY` | `delta >= 0.01` **and** `lo > 0` | Record an initial signal that structural merging may be useful. Do not claim a word-contribution law. |
| `PRED_NUMERICALLY_BETTER_BUT_NOT_CLEAR` | otherwise | No clear evidence of incremental merge value; stop this round. |

No automatic PRCG-v1, second seed, D1, Valid/Test or other new network is authorised by any
outcome. The only permitted next step is human review of this report.

## 7. Stopping rules

Stop after the Ridge audit, the ATOM_ONLY seed-0 run, the empty-grammar checks, the
paired-bootstrap comparison and the short report. Do not train a second arm, seed or
dataset, and do not evaluate Valid/Test.

## 8. Deliverables

`PROTOCOL.md`, `configs/atom_only.json`, `code/` (`pand_*`), `tests/test_pand.py`,
`results/ATOM_ONLY_seed0.{json,pt}` plus its manifest, `analysis/ridge_audit.json`,
`analysis/ridge_candidates.csv`, `analysis/atom_checks.json`, `analysis/summary.json`,
`analysis/bootstrap.json`, `analysis/arms.csv`, `analysis/contrasts.csv`, and `REPORT.md`.

## 9. Falsification / interpretation limits

The diagnostic is falsified as a *positive* claim for merging if ATOM_ONLY is not worse than
PRED_GRAMMAR. It is an initial positive signal only under the clear-advantage rule above.
A single seed and 504 Dev molecules cannot establish a general word-contribution law. The
granularity confound (ATOM_ONLY has more tokens) remains and must be stated in the report.