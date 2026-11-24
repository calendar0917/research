# SES-v0 — 非 MP 结构证据选择快速探索 — protocol

Round: `structural_evidence_selection_pilot_v0`  
Status: frozen before any formal training run.  
Task: `ogbg-mollipo` whole-molecule lipophilicity regression, official split.  
Seed: `0` only. Test: `DEVELOPMENT_TEST`; it never selects a checkpoint.

## 1. Question

With the strong non-MP R2 sentence encoder kept fixed, can the model select a
small set of **real induced structural evidence** and stay close to the full
graph? This round is a mechanism screen, not a paper experiment. It does not
test a structural dictionary, additive decomposition, Hamiltonian, GSRO or graph
matrix solving. No teacher, distillation, pretraining, pseudo-label, message
passing or Transformer is used. The predictor may not read any hidden
representation of unselected structure.

## 2. Data and label access

* Frozen raw cache `tracks/structural-query-learning/data/ogbg_mollipo_sq-pilot-cache-v1.npz`,
  sha256 `fe002287…23c7c`. Official Train/Valid/Test = 3360/420/420.
* The brief authorises full Train/Valid/Test evaluation every epoch. All 4200
  labels are read from the same frozen cache; the local label vault is compared
  (`max |Δ| = 2.29e-7`). Valid/Test labels never enter a gradient, a normaliser,
  a category support, a candidate or a selector.
* Target z-score, graph-scalar standardisation, atom-scalar standardisation and
  categorical support are fitted on the **full official Train** (3360) and
  shared by DENSE / RANDOM / SELECT. Checkpoint selection is Valid only.

## 3. Candidate evidence

For each original graph `G=(V,E)`:

```
k = max(1, min(n-1, floor(0.65*n)))
```

(n=1 keeps the unique node; no such row exists in this dataset). BFS from every
atom in a canonical atom order, taking the first `k` popped atoms. The canonical
order is a pynauty `canon_label` of the coloured atom/bond incidence graph; raw
atom ids are never model inputs. Supports are de-duplicated. If more than four
unique supports exist, the first is the BFS support from canonical atom rank 0;
the other three are chosen greedily by maximum minimum-Jaccard distance, with a
canonical tie-break. If only 1–3 unique supports exist they are kept. A support
never crosses connected components; a component smaller than `k` is reported as
below-budget coverage and the full graph is never substituted. Result: 25
molecules have 3 candidates, 4175 have 4; one common-valid molecule has a
one-atom component and therefore three below-k candidates.

Each candidate is an induced subgraph with only its selected real atoms, its
real induced bonds and their raw OGB categorical features. Degree, minimum ring,
cycle membership, S1, typed S2, radius-2 summaries, atom/graph scalars and
normalisation are rebuilt inside that candidate. Original atom features (which
encode some local chemical environment) are retained and are explicitly not
claimed to be independent molecular properties.

## 4. Model and arms

Predictor: frozen `GwicModel(use_words=False)`, 416,177 parameters, imported
byte-for-byte from `graph_word_incidence_composition_pilot_v0`; no word/score/
incidence module is instantiated. The only adaptation is candidate-subgraph
input assembly; `readout_in` (post R2 pooling, pre readout; 440 dims) is exposed
for the selector.

* `DENSE`: complete original graph, MSE on standardized labels.
* `RANDOM`: every valid candidate independently encoded and predicted; loss is
  the molecule-equal mean over candidates of squared error. Evaluation reports
  both the exact uniform single-candidate expectation and one fixed label-blind
  candidate index. Ensemble averaging is never reported as the model score.
* `SELECT`: same candidates and predictor, plus
  `score_j = MLP(440,64,1)(detach(z_j))` with SiLU. Per-molecule softmax over
  valid candidates; `p_train = 0.9 p + 0.1/m`; loss
  `mean_m sum_j p_train_j (y_j-y)^2`. Evaluation is `argmax score` and only
  that one candidate prediction. No Gumbel/RL/STE/soft-mask and no extra loss.

All three predictors share a bitwise-identical seed-0 initial state; SELECT's
selector is the only extra parameter block (28,289 parameters).

## 5. Recipe, horizon and resources

AdamW, peak 1e-3, min 1e-5, weight decay 1e-3, batch 64 molecules, 5-epoch
linear warmup then per-step cosine, dropout 0.1, standardized-label MSE,
gradient clip 5. Common horizon is chosen from 120/90/60 **only by measured
throughput** on real candidate encoding plus full three-split evaluation; the
preflight models are discarded and formal runs reinitialise. Target formal wall
8–10 min, hard 15 min, total ≤ 0.5 GPU·h. GPU0 prioritises SELECT, GPU1 RANDOM,
DENSE starts on the first free GPU. No post-hoc change of horizon, candidate
count, node fraction, LR or width.

## 6. Evaluation and diagnostics

Every epoch evaluates Train, Valid and Test in `eval()`; Valid selects the best
checkpoint. SELECT uses hard argmax, RANDOM uses expected RMSE, DENSE uses
full-graph RMSE. Paired molecule-level bootstrap (1000) is reported for
SELECT−DENSE and SELECT−RANDOM on Valid/Test.

Pre-training checks (`analysis/checks.json`) cover: official identity; candidate
connectivity/induced-edge legality/budget; S1/S2/R2 internal-only fields;
single-vs-batch prediction; outside-structure invariance; other-candidate
independence; DENSE and candidate relabelling invariance including candidate
generation; SELECT argmax equivalence; finite non-zero predictor and selector
gradients; bitwise-shared predictor initial state; parameter ledgers; frozen
DENSE equivalence; absence of forbidden predictor inputs.

Post-training diagnostics use only the best checkpoints and existing
candidates (no new search, gradient explainer or molecule edit): coverage and
overlap; frozen selection intervention (argmax / uniform expectation / minimum
score); independent RANDOM predictor on SELECT's chosen candidate; post-hoc
candidate oracle; six Valid case displays. The oracle uses labels and is never a
score, gradient or checkpoint rule.

## 7. Limits

About 65% of nodes are retained, so this is not an extreme-sparsity or minimal
sufficient subgraph claim. Exact prediction recomputation shows dependence on
the chosen evidence; it does not prove the selection action is free of
unselected information, nor that the chosen structure has chemical causal
effect. Single seed, exploratory screen; no automatic follow-up seed, ratio or
selector search.
