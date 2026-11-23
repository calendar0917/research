# PRCG-ATOM-NULL — graph-word necessity diagnostic

**Protocol** `prcg-atom-null-diagnostic-v0` (`PROTOCOL.md`, frozen before the formal
ATOM_ONLY run)
**Parent** `predictive_residual_composition_grammar_pilot_v0` (read-only)
**Data** D0 only: 2856 Fit / 504 Dev molecules, scaffold-disjoint; official Valid/Test never
opened.
**New model arm** `ATOM_ONLY`, seed 0, one formal training run, CPU, 4 threads,
`use_deterministic_algorithms(True)`.
**Compute** ATOM_ONLY: 1.38 min wall, 0 GPU-hours, peak RSS ≈ 0.71 GB.
**Code** `code/pand_*.py`, tests `tests/test_pand.py`, artefacts in `results/`, `analysis/`.

## Verdict

**`PRED_CLEAR_ADVANTAGE_OVER_ATOM_ONLY`**

ATOM_ONLY is much worse than PRED_GRAMMAR: Dev RMSE **0.9620** vs **0.7115**,
`delta_merge = 0.2504`, paired-bootstrap 95 % CI **[0.1859, 0.3162]**, 10 000/10 000
resamples positive. This is an **initial signal that structural merging may be useful** in
this composition model. It is **not** a word-contribution law and it does not establish that
predictive residual gain is the right merge criterion.

The Ridge audit found **no substantive change**: all 32 originally selected rules remain
bidirectionally positive under the strict augmented Ridge, with no sign change, no
constraint-aware selection change, and Spearman ≥ 0.99938. The parent predictive-discovery
screen therefore does **not** need contraction; its downstream interpretation is still the
separate question tested by ATOM_ONLY.

| arm | Dev RMSE | 95 % CI | Dev MAE | Fit RMSE | best epoch | wall | words/mol | singleton atoms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `FREQ_GRAMMAR` | 0.7486 | 0.6997–0.7976 | 0.5907 | 0.5330 | 89 | 1.62 min | 10.23 | 9.90 % |
| `PRED_GRAMMAR` | **0.7115** | 0.6601–0.7635 | **0.5460** | 0.4753 | 111 | 1.72 min | 20.32 | 58.58 % |
| `PRED_PORT_MEAN` | **0.7108** | 0.6625–0.7602 | 0.5557 | 0.5268 | 84 | 1.45 min | 20.32 | 58.58 % |
| `ATOM_ONLY` | 0.9620 | 0.8963–1.0265 | 0.7418 | 0.9110 | 110 | 1.38 min | 26.86 | 100 % |

`ATOM_ONLY` Fit MAE is 0.7293. Its `z` participation ratio is 1.671 (variance 0.00552), above
the non-degeneracy floor used by the parent pilot. The atom-only arm is not a collapsed
network: it simply underfits much more (Dev/Fit 1.056) than the composition arms
(PRED_GRAMMAR 1.497). The composition arm has more capacity to fit and still transfers
better on Dev.

## 1. Ridge scoring audit (Part A)

Replay of the original 32 predictive rounds is exact: selected patterns, parent
certificates, applications and final 32-rule grammar all match the committed parent trace;
maximum selected-gain difference vs the stored trace is `1.1e-16`. The replay is Fit-only
and CPU-only; no neural network was trained for this part.

### 1.1 The denominator difference is real but small for selected rules

The frozen `RoundScreener` uses

`resid_tr^T resid_tr + lambda`.

The strict augmented Ridge (candidate appended to `[X, c]`, same `lambda = 5`, candidate
penalty 1, unpenalised intercept) uses the Frisch–Waugh denominator

`c_tr^T resid_tr + lambda`.

The two differ by `lambda * ||S||_P^2 ≥ 0`, where
`S = (X^T X + lambda P)^{-1} X^T c`. Directly solving the augmented normal equations agrees
with the Schur formula to `3.7e-15`; the gap identity agrees to `3.0e-11`. The audit therefore
compares the frozen score against an independently solved reference, not against another
reformulation of the same code.

For the 32 **selected** rules:

* denominator ratio `strict / old`: median **1.0040**, max **1.0238**;
* absolute selected-gain change: mean **1.8e-5**, max **1.33e-4**;
* 32/32 remain **bidirectionally positive** under the strict Ridge;
* **0** selected-rule fold sign changes;
* **0** constraint-aware selection changes.

Across all original candidates the denominator ratio reaches 1.36 and the largest mean-gain
change is 1.0e-3 (on a candidate that was not selected), but the ordering is essentially
unchanged. The pre-registered 1e-6 numerical-difference flag fires in 28 rounds; that is a
sensitivity flag, not a substantive change. Only round 3 exceeds 1e-4 (1.33e-4), and there
the selected rule remains rank 1, top-1 is unchanged, top-5 overlap is 5/5, and Spearman is
0.99968.

### 1.2 Candidate ordering

Across rounds: minimum Spearman **0.99938**, minimum Kendall **0.99263**, zero top-1
changes, and top-5 overlap 4 or 5. The selected candidate is rank 1 under the strict score
in every round. The parent screen's selected rules and the parent report's statement that
they pass a two-direction held-out-MSE screen are therefore robust to this algebra check.
This says nothing about whether the selected lexicon is *better than atoms*; that is the
ATOM_ONLY comparison below.

## 2. ATOM_ONLY construction and checks (Part B)

The arm uses an empty `Grammar()`:

* every atom is a single-atom word (90 514 words over 3360 D0 molecules);
* every one of the 98 561 real bonds is an inter-word bond;
* there are **0** crossing bonds and **0** intra-word bonds;
* the `Compose` module is instantiated with 24 240 parameters + 96 LayerNorm parameters,
  but a forward hook records **0 calls**, its parameters have **no gradient**, and the word
  state is exactly the atom state (`max |z − atom_z| = 0`).

The model is not atom-only-specific: parameter ledger **105 025** and initialisation hashes
are identical to `PRED_GRAMMAR` (all four arms share the same initialisation). The training
recipe is the frozen parent recipe (AdamW, peak 3e-4, wd 1e-4, batch 64, 5-epoch warmup,
cosine, max 240 epochs, min 80, patience 45 on Dev RMSE).

Empty-grammar checks: **19/19 pass** (`analysis/atom_checks.json`), covering atom coverage,
bond coverage, renumbering invariance, bond-order invariance, batch composition/order
invariance and label isolation. Post-training invariance errors are ≤ `6.0e-7` for
renumbering, `4.8e-7` for bond order and `1.2e-7` for batch composition.

## 3. Comparison (Part C)

Primary contrast:

```
delta_merge = RMSE(ATOM_ONLY) - RMSE(PRED_GRAMMAR)
            = 0.2504   (95% CI 0.1859 .. 0.3162; 100% of resamples > 0)
```

The MAE contrast is `0.1957` (95 % CI 0.1433 .. 0.2495). ATOM_ONLY is better on **39.7 %**
of the 504 Dev molecules and worse on **60.3 %**; median per-molecule absolute-error
difference is `0.1243` (positive = ATOM_ONLY worse). The residual correlation with
PRED_GRAMMAR is only 0.588, so this is not a small re-ranking of the same errors.

Against the other committed arms:

| contrast | point | 95 % CI | P(>0) |
| --- | ---: | ---: | ---: |
| `ATOM_ONLY − PRED_GRAMMAR` | +0.2504 | 0.1859 .. 0.3162 | 1.000 |
| `ATOM_ONLY − FREQ_GRAMMAR` | +0.2133 | 0.1517 .. 0.2744 | 1.000 |
| `ATOM_ONLY − PRED_PORT_MEAN` | +0.2511 | 0.1869 .. 0.3148 | 1.000 |

The parent `delta_discovery = FREQ − PRED` is unchanged at +0.0371 (95 % CI −0.0098 ..
+0.0844, 94.1 % of resamples > 0); this diagnostic does not turn that comparison into a
replicated result.

### Atom coverage

All arms cover 100 % of atoms exactly once. The fraction of atoms that remain single-atom
words is:

| grammar | Fit | Dev |
| --- | ---: | ---: |
| PRED (and PRED_PORT_MEAN) | **58.58 %** (44 937 / 76 713) | 57.26 % |
| FREQ | **9.90 %** (7 598 / 76 713) | 9.79 % |
| ATOM_ONLY | **100 %** (76 713 / 76 713) | 100 % |

ATOM_ONLY has **26.86 words/molecule** on Fit, versus 20.32 for PRED and 10.23 for FREQ.
The comparison is therefore **not token-matched**: the atom-only arm has more tokens, not
fewer. Its much worse Dev RMSE is not explained by having too few readout tokens.

## 4. Decision (Part D)

The pre-registered clear-advantage rule is met (`delta_merge ≥ 0.01`, CI lower > 0):

> **PRED_GRAMMAR has a clear advantage over ATOM_ONLY in this D0-seed-0 comparison.**
> Record this as an initial signal that structural merging may be useful in this model.
> Do not claim that a word-contribution law has been established.

The Ridge audit does not require contraction of the parent predictive-rule discovery:
the frozen screen and the strict augmented Ridge select the same rules, with both fold
gains positive and essentially unchanged candidate ordering.

No automatic follow-up is authorised by this result: **no PRCG-v1, no second seed, no D1,
no Valid/Test, and no other new network**. The workspace stops here.

## 5. Confounds and limits

1. **Granularity / token count is not matched.** ATOM_ONLY has 26.86 words/molecule versus
   PRED's 20.32. The result says the composition model with PRED's merged words transfers
   better than the same model with only atoms; it does not isolate "compositional structure"
   from token count, readout aggregation or the specific 32-rule lexicon.
2. **The atom-only arm is a null representation, not a neutral capacity control.** It cannot
   represent a multi-atom substructure as one state; Q only sees inter-word bonds. The
   result is about this model family, not about all atom-level models.
3. **Single seed, single split, 504 Dev molecules.** The bootstrap interval is descriptive;
   it is not a replication or a population-level guarantee.
4. **The predictive-vs-frequency lexicon comparison remains uncertain.** The new result
   does not resolve the parent granularity confound or the `delta_discovery` interval that
   includes zero.
5. **No Valid/Test.** By protocol. Absolute numbers are D0-Dev only and the parent model is
   still far from the R2-SENTENCE / GINE references (0.5506 / 0.5266).
6. The Ridge audit is an algebra check of the discovery screen, not a re-validation of the
   downstream model benefit. The ATOM_ONLY comparison supplies that downstream check.

## 6. Not claimed

Not PRCG-v1, not GWIC-v1, not BBPA-v1, not an R2-SENTENCE result, not a D1/Valid/Test
result, not a second seed or second dataset, and not a general statement that graph words
are necessary for molecular regression. The only claim is the D0-seed-0 comparison in this
report.

## 7. Reproduction

```bash
R=tracks/structural-sufficiency/prcg_atom_null_diagnostic_v0
uv run python $R/code/pand_ridge_audit.py       # 32-round replay + strict Ridge audit, ~30 s
uv run python $R/code/pand_atom_train.py --device cpu --threads 4   # one formal ATOM_ONLY run, ~83 s
uv run python $R/code/pand_atom_checks.py      # 19/19 empty-grammar / invariance / leakage checks, ~10 s
uv run python $R/code/pand_analyze.py          # paired bootstrap, coverage, decision
uv run python $R/tests/test_pand.py            # cheap algebra/construction tests
```

Artefacts: `results/ATOM_ONLY_seed0.{json,pt}` + manifest,
`analysis/{ridge_audit.json,ridge_candidates.csv,atom_checks.json,summary.json,bootstrap.json,arms.csv,contrasts.csv}`.
The parent directory was read only; the parent test suite still passes with 0 failures.