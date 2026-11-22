# REPORT — acra_relation_generalization_pilot_v1

Status: **complete; no full-Train revalidation was triggered.** One seed (0),
one shared 160-epoch horizon, five arms, no hyperparameter search.  Test labels
and Test predictions were never read.  New models were evaluated on the frozen
Dev split only; the official Valid split appears solely in Phase A's frozen-weight
diagnostic of the parent ACRA_REL checkpoint.

Round directory:
`tracks/structural-sufficiency/acra_relation_generalization_pilot_v1/`

| item | value |
|---|---|
| protocol commit (frozen plan) | `d45ff7887dd9` (horizon commit `0654ff83c6ca`) |
| formal-run commit | `0654ff83c6ca` |
| structure cache sha256 | `1f6d7a3f870513a83b50f6aa6edbe0143f37c2a883308627f47145e2078fdc3b` |
| frozen split sha256 | `60ac13c04fb0dab484fc8e7cc28f2a85450794ed3ebeeb0a87d1b41c3532559e` |
| raw cache sha256 | `fe002287fa9c581278bc5430f5b4a5b26b35d9d4d5207d138f15956d93123c7c` |
| remote environment | `a100-2`, A100-SXM4-40GB, driver 550.163.01, torch 2.5.1+cu124, CUDA 12.4, Python 3.12.14 |
| Train-sub / Dev | 2,856 / 504 molecules; 1,354 / 231 scaffolds |
| dev label scaler | mean 2.1858, std 1.2140 (Train-sub only) |
| formal compute | **0.4847 A100 GPU-hours**, 5 runs, at most 2 GPUs in parallel |
| peak VRAM | 739–769 MB per run |
| wall time | 336–361 s per run |
| engineering gates | all 13 passed (`analysis/checks.json`) |

## 1. What was implemented

The parent data objects are unchanged: every real bond produces two directed
events; `S` is the radius-1 ball around the source in `G − bond`, `C` the
radius-2 ball around the target; the shared rooted-frame encoder, atom term,
connection response and whole-molecule objective are the parent ones.  The
frozen split is scaffold-grouped and label-blind; the target scaler, category
support masks and all gate support counts are fitted on Train-sub only.

Five arms (seed 0, same recipe):

| arm | relation term | parameters |
|---|---|---|
| `ACRA_REL_REF` | original full term | 157,823 |
| `ACRA_MOMENT_ONLY` | `gate * moment / max(1,m)` only | 157,815 |
| `ACRA_HIST_ONLY` | `gate * histogram / max(1,m)` only | 145,407 |
| `ACRA_TOPO_COUPLED` | `gamma[cat] * dot + bias[cat]` | 157,831 (+8 `theta`) |
| `ACRA_SUPPORT_GATE` | original full term; `rho(n)` on the local interaction | 157,823 |

No GNN, message passing, Graph Transformer or extra attention was introduced.

## 2. Phase A: exact algebra and the old checkpoint

The identity below was checked against the literal parent pair loop:

```
moment    = 0.5 * (Q.K - sum_e q_e.k_e) / sqrt(rank)
histogram = sum_category count[category] * bias[category]
E_rel     = gate * (moment + histogram) / max(1, m)
```

* Real molecules: 3,780 Train+Valid molecules, 1,706,206 bond pairs,
  max |direct − decomposition| = **1.43e-06**;
  max |moment + histogram − E_rel| = **0.0**.
* Random connected synthetic graphs (12 graphs, 6–10 atoms, real attribute
  rows): max direct error **1.79e-07**;
  moment + histogram − E_rel = **0.0**.
* Per-molecule relation-category counts sum exactly to `m(m−1)/2`; zero
  violations.

The old checkpoint (`ACRA_REL_S0`, best epoch 101, gate 0.2274) has
mean |moment| 13.46 and mean |histogram| 16.85 before the gate/normaliser, so
the moment is 44 % of the absolute component mass.  The two large components
cancel strongly: mean E_rel is only −0.0118 (std units).  Frozen-weight removal
on the official Valid split:

| variant | Valid RMSE (label) | Valid MAE (label) | mean abs prediction change |
|---|---|---|---|
| base | **0.7166** | **0.5029** | — |
| without moment | 0.7220 | 0.5091 | 0.0995 |
| without histogram | 0.7470 | 0.5362 | 0.1357 |
| without both | 0.7592 | 0.5407 | 0.1790 |

Removing the histogram costs 0.0305 RMSE, removing the moment 0.0054; the
removal is a frozen-weight diagnostic, **not** a retrained causal estimate.

Scale and description relations (Train+Valid, per molecule):

| term | Pearson vs n_bonds | vs n_atoms | vs cycle_rank | vs unseen fraction |
|---|---|---|---|---|
| abs moment | 0.243 | 0.262 | 0.049 | 0.028 |
| abs histogram | 0.457 | 0.472 | 0.228 | 0.070 |
| abs E_rel | 0.025 | 0.038 | −0.062 | 0.079 |
| signed E_rel | **0.663** | **0.673** | 0.401 | −0.065 |

Signed E_rel shifts from −0.200 in the smallest bond-count quartile to +0.165
in the largest, while |E_rel| per bond falls from 0.0133 to 0.0049.  The old
relation term therefore carries a size-dependent offset component; this is a
scale effect to keep separate from chemical relation information.

Fixed-histogram permutation check (edge representations, pair identities and
the 8-category histogram fixed; only the category-to-pair assignment permuted):
20 real molecules, max |ΔE_rel| = **3.28e-07**, i.e. the original ACRA_REL is
exactly invariant.  This is an expressivity property, not performance evidence.

## 3. Main Dev result

All numbers are label units on the 504 Dev molecules at the best-by-Dev-RMSE
checkpoint.  Train-sub RMSE is the error at that same epoch; the gap is
Dev − Train-sub.

| model | params | best epoch | Train-sub RMSE | Dev RMSE | Dev MAE | gap | gate |
|---|---|---|---|---|---|---|---|
| `ACRA_REL_REF` | 157,823 | 77 | 0.407 | **0.5555** | **0.4291** | 0.148 | 0.221 |
| `ACRA_MOMENT_ONLY` | 157,815 | 70 | 0.453 | 0.6005 | 0.4680 | 0.148 | 0.124 |
| `ACRA_HIST_ONLY` | 145,407 | 72 | 0.432 | 0.5850 | 0.4451 | 0.153 | 0.270 |
| `ACRA_TOPO_COUPLED` | 157,831 | 92 | **0.360** | 0.5947 | 0.4578 | **0.235** | 0.210 |
| `ACRA_SUPPORT_GATE` | 157,823 | 91 | 0.386 | 0.5664 | 0.4428 | 0.181 | 0.238 |

Dev molecule-level paired bootstrap (positive = candidate better than
`ACRA_REL_REF`):

| comparison | ΔRMSE [95 % CI] | ΔMAE |
|---|---|---|
| MOMENT_ONLY vs REF | **−0.0450 [−0.0779, −0.0148]** | −0.0389 |
| HIST_ONLY vs REF | −0.0296 [−0.0596, +0.0001] | −0.0159 |
| TOPO_COUPLED vs REF | **−0.0392 [−0.0676, −0.0100]** | −0.0287 |
| SUPPORT_GATE vs REF | −0.0110 [−0.0380, +0.0148] | −0.0137 |

Only one training seed was run.  The intervals quantify evaluation-molecule
uncertainty, not training-seed uncertainty.  A CI containing zero is reported
as unresolved, not as equivalence.

## 4. H1 — what carries the original relation gain?

**Closer to the topological histogram than to the global second moment, but
not purely a histogram.**  Three independent signals agree:

1. The old checkpoint loses 0.0305 Valid RMSE when the histogram is removed,
   versus 0.0054 for the moment.
2. Replacing the full relation term by moment-only is resolved worse
   (−0.0450 RMSE); histogram-only is 12.4k parameters smaller and only
   −0.0296 worse, with the Dev CI touching zero.
3. The full REF remains the best model, so the joint term has a small extra
   value over histogram alone.  That residual (+0.0296) is unresolved and
   cannot be separated here from the extra 12.4k relation parameters and the
   moment–histogram interaction.

The honest reading is: the histogram component is the main transferable part
of this relation parameterisation; the global second moment alone is not a good
substitute; the full term's advantage over a pure histogram is too small and
too confounded with capacity to call a mechanism.

## 5. H2 — does true topology-conditioned edge-pair interaction help?

**Not on whole-molecule Dev accuracy.**  `ACRA_TOPO_COUPLED` is resolved worse
than REF on Dev RMSE (−0.0392, CI [−0.0676, −0.0100]) and has the largest
generalisation gap (0.235 vs 0.148).  It fits Train-sub much better (0.360 vs
0.407) but transfers less.

The mechanism is genuinely active.  Trained `gamma` departs from 1 by up to
0.210; on 200 real Dev molecules with at least two relation categories, all 200
change `E_rel` under a category-to-pair permutation that holds the histogram
fixed (max ΔE_rel = 0.0759).  The original REF stays invariant (max 5.96e-07).
With `gamma = 1` the two relation operators are equivalent to 8.05e-07.
A forced non-constant gamma witness reaches 0.1126.  So the negative result is
not a dead branch; it is an overfitting/transfer result.

Novelty stratification explains the split:

| unseen-event fraction | n | REF RMSE | TOPO RMSE | TOPO − REF [95 % CI] |
|---|---|---|---|---|
| 0 | 144 | 0.4917 | **0.4461** | **+0.0455 [+0.0002, +0.0909]** |
| (0, 0.2] | 213 | **0.5265** | 0.6316 | **−0.1051 [−0.1509, −0.0613]** |
| (0.2, 0.5] | 127 | **0.6464** | 0.6696 | −0.0232 [−0.0832, +0.0349] |
| (0.5, 1] | 20 | 0.6551 | **0.6188** | +0.0363 [−0.0962, +0.2054] |

The topology-conditioned term helps only in the no-novelty bucket (and that
result is marginal); it is resolved worse as soon as a small fraction of local
combinations is unseen.  With one seed and only 20 molecules in the highest
bucket, this is a pattern, not a stable ranking.

## 6. H3 — does the Support Gate reduce unseen-combination error?

**No whole-molecule improvement; a positive matched-pair signal.**

* Overall Dev: 0.5664 RMSE / 0.4428 MAE vs REF 0.5555 / 0.4291; both point
  estimates worse, the RMSE CI includes zero.
* No novelty bucket shows a resolved change.  Point estimates: RMSE
  −0.0039 / −0.0114 / −0.0104 / −0.0494 and MAE +0.0048 / −0.0200 / −0.0175 /
  −0.0555 for the `0`, `(0,0.2]`, `(0.2,0.5]`, `(0.5,1]` buckets; every CI
  includes zero and the highest bucket has only 20 molecules.
* Low-novelty molecules do **not** degrade in any resolved way; the bucket-0
  MAE is 0.3834 vs 0.3882 (a 0.0048 point improvement, not resolved).
* High-novelty molecules do **not** show a reliable gain.

On the 474 real Train-sub×Dev matched pairs (both endpoints in official Train;
cluster bootstrap by the Dev molecule), however, the Support Gate is the best
arm on relative changes:

| arm | ΔMAE | ΔRMSE | direction accuracy | Pearson(pred Δ, true Δ) |
|---|---|---|---|---|
| REF | 0.4329 | 0.5422 | 0.800 | 0.8507 |
| MOMENT_ONLY | 0.4283 | 0.5440 | 0.844 | 0.8564 |
| HIST_ONLY | 0.4272 | 0.5474 | 0.812 | 0.8542 |
| TOPO_COUPLED | 0.4151 | 0.5347 | 0.806 | 0.8557 |
| **SUPPORT_GATE** | **0.3870** | **0.4993** | **0.846** | **0.8744** |

Paired cluster bootstrap of the ΔMAE difference on the same pairs:
`SUPPORT_GATE − REF = +0.0459 [+0.0047, +0.0890]` (positive = candidate better).
The other arms' differences are unresolved (+0.0046 to +0.0178).  The Support
Gate also moves the mean predicted Δ closer to the truth (−0.069 vs −0.047;
true −0.141).  This is the only resolved positive signal in the round, but it
is a relative/matched-pair signal, not a whole-molecule accuracy gain.

Event-level support diagnostic (Dev events, `n` = distinct Train-sub molecules):

| support `n` | events | REF mean abs interaction | SUPPORT_GATE mean abs interaction | mean rho | REF mean abs r | SUPPORT_GATE mean abs r |
|---|---|---|---|---|---|---|
| 0 | 4,039 | 0.0794 | **0.0535** | 0.50 | 0.0923 | 0.0921 |
| 1 | 1,685 | 0.0798 | 0.0591 | 0.58 | 0.0921 | 0.0938 |
| 2–4 | 3,282 | 0.0849 | 0.0773 | 0.67 | 0.0962 | 0.1062 |
| 5–9 | 3,038 | 0.0892 | 0.0874 | 0.79 | 0.0995 | 0.1115 |
| 10–49 | 8,660 | 0.0959 | 0.0954 | 0.90 | 0.1109 | 0.1234 |
| 50+ | 9,342 | 0.0600 | 0.0741 | 0.98 | 0.0788 | 0.0941 |

The gate does shrink the interaction at low support, as designed.  It does not
shrink the total response `r`: the scalar part becomes larger on the same events
(e.g. mean |scalar_s+scalar_c| 0.0849 vs 0.0387 at support 0).  The model
reallocates rather than removes response mass.  That is a plausible explanation
for why the whole-molecule error is not reduced while relative changes improve.

Dev novelty (unseen-event fraction) has median 0.104, mean 0.149, maximum
0.818.  It is mildly confounded with size: Pearson(novelty, n_bonds) = −0.340,
vs n_atoms −0.349, vs cycle_rank −0.167.  The support buckets are therefore
not size-matched; the small high-novelty bucket is also the smallest-molecule
bucket.

## 7. Outlier sensitivity

Full Dev RMSE is the primary number; no molecule was deleted to claim an
improvement.  Top-5 squared-error exclusion does not change any ranking:

| arm | full RMSE | RMSE without own top5 | RMSE without top5 union |
|---|---|---|---|
| REF | 0.5555 | 0.5260 | 0.5053 |
| SUPPORT_GATE | 0.5664 | 0.5350 | 0.5162 |
| HIST_ONLY | 0.5850 | 0.5520 | 0.5385 |
| TOPO_COUPLED | 0.5947 | 0.5597 | 0.5389 |
| MOMENT_ONLY | 0.6005 | 0.5709 | 0.5556 |

The five largest squared errors account for 10.5–12.3 % of each arm's total.
Leave-one-molecule-out RMSE ranges are 0.00034–0.00049, i.e. no single molecule
drives the ranking.  The largest single error is molecule row 2099 for REF
(squared error 4.24) and row 2804 for TOPO (6.52); labels and predictions are in
`analysis/outlier_sensitivity.json`.  Error quantiles and the full worst-case
record are in the same file.

## 8. Structural mechanism capability

`analysis/structural_cases.json` records 200 real Dev molecules with at least
two relation categories.  Holding every edge representation and the joint
category histogram fixed, a category-to-pair permutation leaves REF `E_rel`
unchanged (max 5.96e-07) while the trained TOPO checkpoint changes all 200 cases
(max 0.0759).  The forced non-constant gamma control reaches 0.1126.  The
`gamma=1` equivalence check over all 504 Dev molecules has max error 8.05e-07.

This establishes that the TOPO operator can see relation-category/pair
coupling that the original histogram+moment form cannot.  It says nothing about
chemical validity or predictive usefulness; the trained TOPO model was worse on
Dev, especially on novel local combinations.

## 9. Answers to the five required questions

1. **Is the original REL gain closer to a global second moment or a topological
   histogram?**  Closer to the histogram.  Frozen removal of the histogram
   hurts 5.7× more than removal of the moment; moment-only training is resolved
   worse than the full term, while histogram-only is close to it.  The full
   term still wins by 0.0296 Dev RMSE, but that residual is unresolved and
   confounded with 12.4k extra relation parameters.  The moment alone is not
   the mechanism.
2. **Does genuinely topology-conditioned edge-pair interaction produce a new
   gain?**  Not on whole-molecule Dev accuracy.  It is mechanistically active
   and can distinguish category-to-pair assignments that REF cannot, but it
   overfits Train-sub (gap 0.235) and is resolved worse on Dev.  Its only
   positive stratum is the no-novelty bucket.
3. **Does the Support Gate reduce errors on unseen structural combinations?**
   Not on whole-molecule Dev RMSE/MAE, and not in any novelty bucket with a
   resolved CI.  It does improve real cross-molecule matched-pair ΔMAE by
   0.0459 (CI [+0.0047, +0.0890]) and direction accuracy from 0.800 to 0.846.
   It also reallocates response mass from the interaction to the scalar terms
   rather than shrinking the total response at low support.
4. **Which improvements can be explained by capacity, graph size or outlier
   molecules?**  Capacity can explain part of the full-vs-histogram residual;
   it cannot explain the failure of moment-only (same capacity as REF − 8
   parameters) or TOPO (+8 parameters, worse).  Signed E_rel has a strong
   size relation (Pearson 0.663 with bond count) and a size-dependent offset,
   so size/scale is a real confound in the old relation term.  The arm ranking
   is unchanged after removing each model's top-5 squared errors or the union
   of all top-5 sets; outliers do not create the ranking.
5. **Is it worth developing a genuinely relation-conditioned non-GNN
   architecture?**  Not from this round's evidence.  The relation form is
   expressive enough to couple categories to pairs, but the trained TOPO model
   does not transfer; the histogram component carries most of the useful
   signal; and the only positive new signal is the Support Gate's matched-pair
   improvement.  A next architecture would need to target relative changes and
   control the Train-sub overfitting first.  No next round is started here.

## 10. Conditional full-Train revalidation

The pre-registered rule required a new candidate to improve Dev RMSE or Dev MAE
by at least 0.02 versus REF and show no obvious matched-pair degradation.  The
best new candidate by Dev RMSE is `ACRA_SUPPORT_GATE` at 0.5664 vs REF 0.5555:
ΔRMSE = **−0.0110**, ΔMAE = **−0.0137**.  It does not qualify.  Therefore:

* no candidate was retrained on the full official Train;
* no new official Valid result was read;
* the round stops without a full-Train revalidation.

## 11. What is not established

* **Single seed.**  No training-seed variance.  All CIs are evaluation-molecule
  bootstraps; differences of 0.01–0.03 RMSE are not stable model rankings.
* **Capacity is not matched.**  HIST_ONLY has 12.4k fewer parameters; the full
  REF residual over it cannot be assigned to the pair term alone.
* **Official Valid is not a fresh test for the new arms.**  It was never read
  for them; the old checkpoint's Phase A diagnostics use the parent Valid
  labels and are clearly labelled as such.
* **Matched pairs are real replacements, not counterfactuals.**  The parent
  directory guarantees the same exact context, but the pair difference is still
  an observed replacement.  Only whole-model Δŷ is evaluated.
* **Novelty is confounded with size** and the highest bucket has only 20 Dev
  molecules.
* **TOPO capability is not chemical evidence.**  The permutation witness is an
  algebraic input intervention.

## 12. Deviations and execution facts

1. The frozen split uses acyclic molecules' own SMILES as their scaffold key
   (RDKit Murcko returns an empty scaffold for them) to avoid one giant acyclic
   group.  This is recorded in the split JSON and protocol.
2. `analysis/relation_permutation_invariance.json` and
   `analysis/structural_cases.json` are additional capability artifacts; they
   do not change any model or training rule.
3. The old parent checkpoint's Phase A diagnostics use the official Valid
   labels because the parent round already reported that number; no new arm was
   evaluated on Valid.
4. Formal runs: `ACRA_REL_REF_S0` and `ACRA_MOMENT_ONLY_S0` in parallel on
   `res-gpu0`/`res-gpu1`; then `ACRA_HIST_ONLY_S0` and
   `ACRA_TOPO_COUPLED_S0`; then `ACRA_SUPPORT_GATE_S0`.  All exited 0.  Wall
   times 336–361 s, total 0.4847 GPU-hours (24 % of the 2-hour budget).
5. The remote host had untracked parent result directories; they were moved to
   `.rr-untracked-backup-acra-v2/` before deployment (they are not part of this
   round's evidence).  All formal runs used commit `0654ff83c6ca`.

## 13. Deliverables

`PROTOCOL.md`, this `REPORT.md`, `code/`, `tests/`,
`data/split_train_sub_dev.json`, `data/acra_cache_lipo_v2.npz`,
`analysis/relation_factorization.json`, `analysis/relation_term_provenance.json`,
`analysis/support_generalization.json`, `analysis/structural_cases.json`,
`analysis/pair_delta.json`, `analysis/outlier_sensitivity.json`,
`analysis/checks.json`, `analysis/horizon.json`,
`analysis/training_curves.png`, `results/main_table.csv`,
`results/summary.json`, and per-run checkpoints/curves/predictions/manifests/
`rr_meta.json` under `results/ACRA_*_S0/`.

The round is complete.  No next-round experiment was started.