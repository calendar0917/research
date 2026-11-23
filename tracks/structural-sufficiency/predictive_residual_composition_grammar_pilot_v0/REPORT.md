# PRCG-v0 — Predictive Residual Composition Grammar (pilot report)

**Protocol** `prcg-v0-predictive-residual-composition-grammar-pilot` (`PROTOCOL.md`, frozen
before training)
**Data** D0 only: 2856 Fit / 504 Dev molecules, scaffold-disjoint; official Valid/Test never
opened.
**Seed** 0, three arms, no search. **Device** CPU, 4 threads, `use_deterministic_algorithms(True)`
(reruns are bitwise identical).
**Compute** 4.8 min of training for the three arms (5.2 min wall including featurisation),
0 GPU-hours, peak RSS ≈ 0.8 GB.
**Code** `code/`, tests `tests/`, artefacts `grammar/`, `results/`, `analysis/`.

## Verdict

**`LEXICON_SIGNAL_WITHOUT_PORT_GAIN`**

| quantity | value |
| --- | --- |
| Dev RMSE `FREQ_GRAMMAR` | **0.7486** (95 % bootstrap CI 0.6997 – 0.7976) |
| Dev RMSE `PRED_GRAMMAR` | **0.7115** (0.6601 – 0.7635) |
| Dev RMSE `PRED_PORT_MEAN` | **0.7108** (0.6625 – 0.7602) |
| `Δ_discovery` = FREQ − PRED | **+0.0371** (−0.0098 – +0.0844), 94.1 % of resamples > 0 |
| `Δ_port` = PORT_MEAN − PRED | **−0.0007** (−0.0220 – +0.0205), 47.8 % of resamples > 0 |
| Dev MAE | FREQ 0.5907 / PRED **0.5460** / PORT_MEAN 0.5557 |
| Fit RMSE | FREQ 0.5330 / PRED 0.4753 / PORT_MEAN 0.5268 |

The predictive lexicon is the better lexicon inside this model, by 0.037 RMSE (5 % relative,
~0.030 MAE), but the interval includes zero, so this is *suggestive, not established* with one
seed on 504 molecules. The attachment-port ablation is flat: replacing real port roles with
the word's mean port role changes Dev RMSE by less than 0.001. The absolute level (0.711) is
far from the R2-SENTENCE line (0.5506 / 0.5464) and from GINE (0.5266), so this pilot is **not
competitive** with either.

## 1. Stage 1 — what the two lexicons actually contain

Both grammars were discovered on D0 Fit only, with identical candidate sets, identical
matching and the **same final rule count (32 each)**; they differ only in the acceptance
score (two-direction scaffold-disjoint cross-fitted Ridge residual gain vs Fit frequency).

| | predictive | frequency |
| --- | --- | --- |
| rules | 32 | 32 |
| rule sizes (2/3/4/5) | 15 / 6 / 5 / 6 | 8 / 10 / 5 / 9 |
| words per molecule (Fit) | 20.32 | 10.23 |
| word-size histogram (Fit) | 44 937 / 9 357 / 2 552 / 554 / 638 | 7 598 / 7 247 / 6 281 / 4 667 / 3 422 |
| word types | 51 | 51 |
| median type support | 147 molecules | 298 molecules |
| cross-scaffold type reuse | 96.1 % | 96.1 % |
| multi-port word fraction | 16.0 % | 63.7 % |

The two lexicons are **chemically different families**, not noise-level variants of each other:

* predictive (top accepted rules, mean held-out MSE gain per fold direction):
  1. `C–N` single conjugated, `k2[5160.7208][0-1:2]`, gain **+0.091** / +0.088, 368 molecules;
  2. `C–N` single, `k2[5160.7208][0-1:1]`, +0.042, 1455 molecules;
  3. `C=O` (double conjugated), `k2[5160.7208][0-1:6]`, +0.037, 1603 molecules;
  4. `C(ar)–N` single conjugated, `k2[5161.6184][0-1:2]`, +0.030, 1526 molecules;
  5. thiazole `k3[5161.6184.15400]` (aromatic C, aromatic N, S), +0.019, 105 molecules;
  6. `k4[5160.5161.6184.7208]` (C, C(ar), N, O), +0.017, 541 molecules;
  7. nitrile `C≡N`, `k2[5160.6184][0-1:10]`, +0.007;
  8. `C(ar)–O`, +0.007.
  These are exactly the polar / H-bonding fragments that dominate lipophilicity.
* frequency: the first rules are hydrocarbon and fluorocarbon bulk — aromatic `C(ar)–C(ar)`,
  four-carbon aromatic chains, quaternary `C(ar)–C(ar)` blocks, then `C–F`, i.e. the most
  *frequent* motifs in the cache. Their residual gain is 0 by construction (not scored).

Every accepted rule must pass the two-direction, scaffold-disjoint, cross-fitted ridge screen;
the full trace (candidates per round, both fold gains, applications, support) is in
`grammar/discovery_trace.json`, and replays deterministically (check C3).

## 2. Stage 2 — the composition model

105 025 parameters, identical for the three arms (init hashes asserted equal before training):

* atom words from frozen core-field embeddings + MLP; all other atom fields are context only;
* one **shared symmetric** `Compose(z_left, z_right, crossing-bond core, port roles)`, applied
  recursively by word size, LayerNorm, no per-type table, no bypass;
* **symmetric** inter-word term `Q` (only `z_A+z_B`, `|z_A−z_B|`, `z_A⊙z_B`, likewise for port
  roles and the local context) — the earlier asymmetry was found and fixed during the
  invariance audit, and is now covered by a permanent check;
* readout = `Σ f(word) + Σ q(edge) + small MLP` on per-molecule aggregates + 8 graph scalars;
  no iterative message passing.

## 3. Stage 3 — pre-training gates: 22/22 pass

`analysis/checks.json` (regenerated with the final code). The ones that carry weight:

| gate | result |
| --- | --- |
| C1 real connected induced words, exact coverage | 0 violations over the 3360 official-Train molecules (98 561 bonds), both grammars |
| C3 discovery-round replay | exact, round by round |
| D1 same type ⇒ same internal state | max within-type `z` dispersion **1.7e-6** (float32 noise) |
| E1 atom renumbering | 120/120 the two canonical orders differ only by a graph **automorphism**; partition reproduces exactly in rank space, word-type multiset identical, prediction max diff **4.8e-7** |
| E2 / E3 bond order, batch order | 2.4e-7 / 6.0e-8 |
| F1/F2/F3 REAL vs MEAN | only `x_l, x_r, e_i, e_j` change; single-port words identical (85.3 % of the check sample); checkpoint toggles the mode |
| G1/G2 | `Compose`/`Q` gradients finite non-zero; 30 steps reduce loss 1.93 → 0.99 |
| H1/H2 | target & graph standardisers are Fit-only; Dev-label shift leaves the rules identical |
| I1 non-degeneracy at init | per-size `Compose` dispersion 0.34/0.53/2.64/2.64, `z` participation ratio 1.72, variance 0.17 — the K-SVD collapse signature is absent |

Two genuine code defects were found by this gate set and fixed **before** training (they are
part of the pilot's result, not incidental): the canonical labelling was not
automorphism-equivariant (it broke renumbering invariance), and the inter-word edge list
indexed the final partition by *position* while all per-word tensors are indexed by *history*
index, silently corrupting every `q` term for multi-word molecules. A third defect (an
asymmetric `Q` input) was fixed for the same reason.

## 4. Stage 4 — training

Frozen R2 recipe: AdamW, peak 3e-4, wd 1e-4 (no decay on biases/norms), clip 5, batch 64
molecules, 5-epoch warmup, cosine to 1e-5 over 240 epochs, min 80, patience 45 on Dev RMSE,
best-Dev checkpoint, Fit-only standardisation, whole-molecule MSE only.

| arm | best epoch | epochs run | wall |
| --- | --- | --- | --- |
| FREQ_GRAMMAR | 89 | 135 | 1.6 min |
| PRED_GRAMMAR | 111 | 157 | 1.7 min |
| PRED_PORT_MEAN | 84 | 130 | 1.5 min |

## 5. Stage 5 — results

Paired bootstrap over the 504 Dev molecules, 10 000 resamples, common resample indices
(`analysis/bootstrap.json`).

```
arm                Dev RMSE  95% CI            Dev MAE   Fit RMSE   words/mol  z-rank
FREQ_GRAMMAR        0.7486   0.6997-0.7976     0.5907    0.5330      10.23     5.32
PRED_GRAMMAR        0.7115   0.6601-0.7635     0.5460    0.4753      20.32     6.61
PRED_PORT_MEAN      0.7108   0.6625-0.7602     0.5557    0.5268      20.32     6.54

contrast                                   point     95% CI            P(>0)
delta_discovery  = RMSE(FREQ) - RMSE(PRED) +0.0371  -0.0098..+0.0844    0.941
delta_port       = RMSE(MEAN) - RMSE(PRED) -0.0007  -0.0220..+0.0205    0.478
```

Per-molecule paired diagnostics: PRED beats FREQ on **55.8 %** of Dev molecules (median
|error| improvement 0.058); PORT_MEAN beats PRED on 54.4 % (median 0.022) yet loses on the
squared-error mean — it wins a few large-error molecules and loses on the median. Train–Dev
gap is large in every arm (Dev/Fit 1.35–1.50), i.e. the model is over-parameterised for 2856
molecules and all three arms are in an overfitting regime.

**Port channel diagnosis.** Real ports make the model fit Fit much better (Fit RMSE 0.475 vs
0.527) while Dev is unchanged, and the ablation can only act on the 16 % of predictive-grammar
tokens that have more than one attachment port (single-port words have mean role = real role
by construction). The port channel is therefore *live but used for Fit-specific capacity, not
for transferable signal*.

## 6. Confounds and limits (read before believing `Δ_discovery`)

1. **Granularity confound — the main alternative explanation.** Equal rule *count* does not
   give equal tokenisation: the predictive grammar yields 20.3 words/molecule (mostly size-2)
   while the frequency grammar yields 10.2 (28 % of them size 4–5). The predictive arm
   therefore has ~2× as many readout tokens, and the gain could come from finer decomposition
   rather than from "predictive residual structure" per se. These three arms cannot separate
   the two. What the arms *do* establish is that screening merges by residual gain produces a
   better-working lexicon than picking the most frequent merges at the same rule count.
2. **Single seed, single split, 504 Dev molecules.** The interval on `Δ_discovery` includes 0;
   the win rate is 56 %. This is not a replicated effect.
3. **A much weaker architecture than the references.** 0.711 vs 0.5506 (R2-SENTENCE seed 0),
   0.5464 (seed 1), 0.5266 (GINE) — the model is ~0.16 RMSE behind. The pilot tests grammar
   *within* a fixed small composer; it says nothing about whether this lexicon would help a
   strong model.
4. **No Valid/Test, no second dataset** — by protocol. Absolute numbers are D0-Dev only.
5. The frequency lexicon is a strong control but not a *neutral* one: frequency and residual
   gain correlate (C–C and C–F are both frequent and predictive). The measured gap is the part
   where they disagree.

## 7. The four questions

**1. Does cross-molecule predictive residual gain form reusable words?** Yes: 32 rules
survived a two-direction, scaffold-disjoint, cross-fitted held-out-MSE screen on Fit only, and
they are chemically coherent (C–N, C=O, C(ar)–N, thiazole, nitrile, C(ar)–O). Type reuse is
96 % across scaffolds, median type support 147 molecules. Two of the accepted merges are
large (per-fold gain +0.09 and +0.04) and monotone in both directions.

**2. Does that lexicon beat a pure-frequency lexicon of the same rule count?** On Dev RMSE,
yes by 0.0371 (5 % relative; MAE 0.045), with PRED better on 56 % of molecules — but the
paired 95 % interval spans zero (−0.010 to +0.084), and the predictive grammar also produces
twice as many word tokens, so the comparison is confounded by granularity. Verdict:
**directionally yes, not established.**

**3. Are real attachment ports separable as a source of gain?** No. `Δ_port` is −0.0007
(−0.022 to +0.021). Real ports improve Fit RMSE by 0.051 while leaving Dev unchanged, and only
16 % of tokens can respond to the ablation at all. The port identity is learnable and used,
but it does not buy transferable accuracy here.

**4. Is the model competitive without an iterative GNN or the R2 sentence mechanism?** No.
0.7115 vs 0.5506/0.5464 (R2-SENTENCE) and 0.5266 (GINE): 0.16–0.19 RMSE worse, with a Dev/Fit
ratio of 1.5. A non-iterative, shared-`Compose`, bag-of-words-style readout of this size is
not competitive with either, with or without the predictive lexicon.

## 8. Not claimed

Not a GWIC-v1, BBPA-v1 or R2-SENTENCE result; no claim about D1, Valid/Test, a second seed, a
second dataset, or about GNNs in general. No PRCG-v1 is proposed here. The only claim is the
D0-seed-0 comparison above, with the granularity confound stated.

## 9. Reproduction

```bash
R=tracks/structural-sufficiency/predictive_residual_composition_grammar_pilot_v0
uv run python $R/code/prcg_discover.py     # 32 predictive + 32 frequency rules, ~50 s
uv run python $R/code/prcg_checks.py       # 22 pre-training gates, ~40 s
uv run python $R/tests/test_prcg.py        # 30 exactness/invariance/leakage tests
uv run python $R/code/prcg_train.py --device cpu --threads 4   # three arms, ~5 min
uv run python $R/code/prcg_analyze.py      # bootstrap, lexicon, route label
```

Artefacts: `grammar/{predictive,frequency}_grammar.json`, `grammar/discovery_trace.json`,
`grammar/stats.json`, `results/{FREQ_GRAMMAR,PRED_GRAMMAR,PRED_PORT_MEAN}_seed0.{json,pt}`,
`analysis/{checks,summary,bootstrap}.json`, `analysis/{arms,contrasts}.csv`.